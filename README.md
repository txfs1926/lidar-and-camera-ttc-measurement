# lidar-and-camera-ttc-measurement

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* ROS
  * All OSes: [click here for installation instructions](http://wiki.ros.org/ROS/Installation) (ROS Kinetic is recommended)
  
## Device
* Camera (Logitech C920 PRO HD Webcam@640×480px, 60fps)
* LiDAR (Leishen-lidar C16 from Shenzhen Leishen Intelligence System Co, Ltd., [Driver](https://github.com/tongsky723/lslidar_C16))
  
## Usage
Make sure that USB camera and LiDAR are connected and LiDAR driver is working properly. (For `lslidar_c16` please refer [the insturction](https://blog.csdn.net/learning_tortosie/article/details/84679149) to better prepare for your LiDAR. In addition, edit `cv::VideoCapture cap(2);` to setup your camera.
### Note
1) The weight file under folder dat/yolo/ is a dummy file. Open `dat/yolo/yolo.weight` via your code editor and follow the procedure to download the true pre-trained weight to replace the existed dummy file. 

2) If you would like to try out this program with your sensors, please replace with your calibration parameters in the file `dat/calibration.yaml` first.

Then run:

  ```
  cd /path/to/your/workspace
  mkdir src
  cd src
  git clone https://github.com/txfs1926/lidar-and-camera-ttc-measurement.git
  cd ..
  catkin_make
  source devel/setup.bash
  rosrun ttc_measurement ttc_measurement
  ```

## TODOs
Please refer TODOs in source code files.
