#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
 
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
 
// Mutex: //
boost::mutex cloud_mutex;
 
struct callback_args{
	// structure used to pass arguments to the callback function
	PointCloudT::Ptr clicked_points_3d;
	pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};
 
void pp_callback(const pcl::visualization::PointPickingEvent& event, void* args)
{
	struct callback_args* data = (struct callback_args *)args;
	if (event.getPointIndex() == -1)
		return;
	PointT current_point;
	event.getPoint(current_point.x, current_point.y, current_point.z);
	data->clicked_points_3d->points.push_back(current_point);
	// Draw clicked points in red:
	pcl::visualization::PointCloudColorHandlerCustom<PointT> red(data->clicked_points_3d, 255, 0, 0);
	data->viewerPtr->removePointCloud("clicked_points");
	data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
	data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
	std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;
}

int main (int argc, char *argv[])
{
    //pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));
    // Load .pcd file from argv[1]
    int ret = pcl::io::loadPCDFile (argv[1], *cloud);
    if (ret < 0) {
        PCL_ERROR("Couldn't read file %s\n", argv[1]);
        return -1;
    }
	std::cout << cloud->points.size() << std::endl;
 
	//viewer->addPointCloud(cloud, "bunny");
 
	cloud_mutex.lock();    // for not overwriting the point cloud
 
	// Display pointcloud:
	viewer->addPointCloud(cloud, argv[1]);
	viewer->setCameraPosition(0, 0, -2, 0, -1, 0, 0);
 
	// Add point picking callback to viewer:
	struct callback_args cb_args;
	PointCloudT::Ptr clicked_points_3d(new PointCloudT);
	cb_args.clicked_points_3d = clicked_points_3d;
	cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(viewer);
	viewer->registerPointPickingCallback(pp_callback, (void*)&cb_args);
	std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;
 
	// Spin until 'Q' is pressed:
	viewer->spin();
	std::cout << "done." << std::endl;
 
	cloud_mutex.unlock();
 
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
    return 0;
}
