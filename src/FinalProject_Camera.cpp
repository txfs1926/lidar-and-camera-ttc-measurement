
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "ros/ros.h"
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>

#include "ttc_measurement/dataStructures.h"
#include "ttc_measurement/matching2D.hpp"
#include "ttc_measurement/objectDetection2D.hpp"
#include "ttc_measurement/lidarData.hpp"
#include "ttc_measurement/camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, char *argv[])
{
    /* INIT ROS */
    ros::init(argc, argv, "listener");
    ros::NodeHandle n;

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    cv::VideoCapture cap(2); // open the camera
    int imgStepWidth = 1; 

    string detectorName = "AKAZE";
    string descriptorName = "SIFT"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

    // object detection
    string yoloBasePath = "/home/txfs1926/ttc_ws/src/ttc_measurement/dat/yolo/";//dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar

    // calibration data for camera and lidar
    cv::Mat cameraExtrinsicMat;
    cv::Mat cameraMat;
    cv::Mat distCoeff;
    cv::Size imageSize;

    cv::FileStorage fs("/home/txfs1926/ttc_ws/src/ttc_measurement/dat/calibration.yaml", cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        ROS_ERROR("Invalid calibration filename!");
        return -1;
    }

    fs["CameraExtrinsicMat"] >> cameraExtrinsicMat;
    fs["CameraMat"] >> cameraMat;
    fs["DistCoeff"] >> distCoeff;
    fs["ImageSize"] >> imageSize;

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera (10FPS -> delta_t = 0.1)
                                                  // TODO: replace this with real FPS
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results
    double tDetector = 0, tDescriptor = 0;

    /* MAIN LOOP OVER ALL IMAGES */

    for (;;)
    {
        double t = (double)cv::getTickCount();
        /* LOAD IMAGE INTO BUFFER */

        // get a new frame from camera
        cv::Mat img;
        cap >> img;

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);
        if (dataBuffer.size()+1 > dataBufferSize)
        {
            dataBuffer.erase(dataBuffer.begin());
            ROS_INFO("REPLACE IMAGE IN BUFFER done");
        }

        ROS_INFO("#1 : LOAD IMAGE INTO BUFFER done");


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;        
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);
        if ((dataBuffer.end() - 1)->boundingBoxes.empty())
        {
            ROS_WARN("#2 : NO AVALIABLE BOUNDING BOXES ARE DETECTED!");
            dataBuffer.clear();
            continue;
        }
        

        ROS_INFO("#2 : DETECT & CLASSIFY OBJECTS done");


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from ros topic
        std::vector<LidarPoint> lidarPoints;
        lidarPoints.reserve(40000);
        bool loaded = loadLidarFromMessage(lidarPoints);
        if (!loaded)
        {
            ROS_ERROR("#3 : Could not get any lidar points!");
            return -1;
        }

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        ROS_INFO("#3 : CROP LIDAR POINTS done");


        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, cameraExtrinsicMat, cameraMat, distCoeff, imageSize);

        // Visualize 3D objects
        bVis = true;
        if(bVis)
        {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(8.0, 40.0), cv::Size(1200, 800), true);
        }
        bVis = false;

        ROS_INFO("#4 : CLUSTER LIDAR POINT CLOUD done");
        
        
        // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
        // continue; // skips directly to the next image without processing what comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        

        if (detectorName.compare("SHITOMASI") == 0){
            detKeypointsShiTomasi(keypoints, imgGray, tDetector, bVis);
        }
        else if(detectorName.compare("HARRIS") == 0){
            detKeypointsHarris(keypoints, imgGray, tDetector, bVis);
        }
        else{
            detKeypointsModern(keypoints,img,detectorName,tDetector,bVis);
        }


        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorName.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            ROS_INFO(" NOTE: Keypoints have been limited!");
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        ROS_INFO("#5 : DETECT KEYPOINTS done");


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        
        if(descriptorName.compare("AKAZE") == 0 && detectorName.compare(descriptorName) != 0)
        {
            ROS_ERROR("Akaze descriptors can be used only with Akaze/Kaze Keypoints");
            return -1;
        }
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorName, tDescriptor);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        ROS_INFO("#6 : EXTRACT DESCRIPTORS done");


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorName, descriptorType, matcherType, selectorType);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            ROS_INFO("#7 : MATCH KEYPOINT DESCRIPTORS done");

            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            ROS_INFO("#8 : TRACK 3D OBJECT BOUNDING BOXES done");


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }
                // compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar; 
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    bVis = true;
                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, cameraExtrinsicMat, cameraMat, distCoeff, imageSize, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));
                        ROS_INFO("TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        ROS_INFO("Press key to continue to next frame");
                        cv::waitKey(0); // if (cv::waitKey(10) >= 0) break;
                    }
                    bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches            

        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        t = t * 1000 / 1.0;
        double fps = 1/t;
        ROS_INFO("FPS = %f", fps);

    } // eof loop over all images

    return 0;
}
