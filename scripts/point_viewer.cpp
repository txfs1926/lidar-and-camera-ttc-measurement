

#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

// typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZI PointT;

void viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    // set background to black (R = 0, G = 0, B = 0)
    viewer.setBackgroundColor (0, 0, 0);
}

void viewerPsycho (pcl::visualization::PCLVisualizer& viewer)
{
    // you can add something here, ex:  add text in viewer
}

int main (int argc, char *argv[])
{
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    // Load .pcd file from argv[1]
    int ret = pcl::io::loadPCDFile (argv[1], *cloud);
    if (ret < 0) {
        PCL_ERROR("Couldn't read file %s\n", argv[1]);
        return -1;
    }

    pcl::visualization::CloudViewer viewer("Cloud Viewer");

    // blocks until the cloud is actually rendered
    viewer.showCloud(cloud);

    // use the following functions to get access to the underlying more advanced/powerful
    // PCLVisualizer

    // This will only get called once
    viewer.runOnVisualizationThreadOnce (viewerOneOff);

    // This will get called once per visualization iteration
    viewer.runOnVisualizationThread (viewerPsycho);
    while (!viewer.wasStopped ()) {
        // you can also do cool processing here
        // FIXME: Note that this is running in a separate thread from viewerPsycho
        // and you should guard against race conditions yourself...
    }

    return 0;
}

