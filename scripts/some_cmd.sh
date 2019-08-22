# NOTE: DO NOT RUN DIRECTLY

## convert .bag to .png and .pcd
mkdir pcd
rosrun pcl_ros bag_to_pcd <input_file.bag> /lslidar_point_cloud ./pcd
mkdir img
python bag2img.py <input_file.bag> ./img /usb_cam/image_raw

## lslidar_c16 driver
mkdir -p lslida_C16/src
cd lslida_C16/src
git clone https://github.com/tongsky723/lslidar_C16.git
source ~/lslida_C16/devel/setup.bash
roslaunch lslidar_c16_decoder lslidar_c16.launch --screen

