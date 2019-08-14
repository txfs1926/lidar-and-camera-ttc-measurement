
#ifndef lidarData_hpp
#define lidarData_hpp

#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>

#include "ros/ros.h"
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include "ttc_measurement/dataStructures.h"

void cropLidarPoints(std::vector<LidarPoint> &lidarPoints, float minX, float maxX, float maxY, float minZ, float maxZ, float minR);
void loadLidarFromFile(std::vector<LidarPoint> &lidarPoints, std::string filename);

void showLidarTopview(std::vector<LidarPoint> &lidarPoints, cv::Size worldSize, cv::Size imageSize, bool bWait = true);
void showLidarImgOverlay(cv::Mat &img, std::vector<LidarPoint> &lidarPoints, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT, cv::Mat *extVisImg = nullptr);

static inline bool convertPointCloud2ToLidarPoint(const sensor_msgs::PointCloud2 &input, std::vector<LidarPoint> &output)
{
  output.clear();
  // Get the x/y/z field offsets
  int x_idx = getPointCloud2FieldIndex(input, "x");
  int y_idx = getPointCloud2FieldIndex(input, "y");
  int z_idx = getPointCloud2FieldIndex(input, "z");
  int r_idx = getPointCloud2FieldIndex(input, "intensity");
  if (x_idx == -1 || y_idx == -1 || z_idx == -1)
  {
    ROS_ERROR("x/y/z coordinates not found! Cannot convert to std::vector<LidarPoint>!");
    return (false);
  }
  else if (r_idx == -1)
  {
    ROS_ERROR("intensity not found! Cannot convert to std::vector<LidarPoint>!");
    return (false);
  }
  int x_offset = input.fields[x_idx].offset;
  int y_offset = input.fields[y_idx].offset;
  int z_offset = input.fields[z_idx].offset;
  int r_offset = input.fields[r_idx].offset;

  uint8_t x_datatype = input.fields[x_idx].datatype;
  uint8_t y_datatype = input.fields[y_idx].datatype;
  uint8_t z_datatype = input.fields[z_idx].datatype;
  uint8_t r_datatype = input.fields[r_idx].datatype;

  // Copy the data points
  for (size_t cp = 0; cp < input.width; ++cp)
  {
    // Copy x/y/z
    LidarPoint lpt;
    lpt.x = sensor_msgs::readPointCloud2BufferValue<float>(&input.data[cp * input.point_step + x_offset], x_datatype);
    lpt.y = sensor_msgs::readPointCloud2BufferValue<float>(&input.data[cp * input.point_step + y_offset], y_datatype);
    lpt.z = sensor_msgs::readPointCloud2BufferValue<float>(&input.data[cp * input.point_step + z_offset], z_datatype);
    lpt.r = sensor_msgs::readPointCloud2BufferValue<float>(&input.data[cp * input.point_step + r_offset], r_datatype);
    output.push_back(lpt);
  }
  return (true);
}

bool loadLidarFromMessage(std::vector<LidarPoint> &lidarPoints);
#endif /* lidarData_hpp */
