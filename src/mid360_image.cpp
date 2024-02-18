#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/io/png_io.h>


#include <livox_ros_driver2/CustomMsg.h>

#include <iostream>
#include <queue>
#include <mutex>
#include <thread>
#include <omp.h>

using namespace std;
using PointType = pcl::PointXYZI;

//config params
static int LiDAR_SKIP = 0;
ros::Publisher pub_depth_image;

std::mutex mtx_lidar;

deque<livox_ros_driver2::CustomMsgConstPtr> vec_;
deque<pcl::PointCloud<PointType>> cloudQueue;
deque<double> timeQueue;

pcl::PointCloud<PointType>::Ptr depthCloud (new pcl::PointCloud<PointType>());

static cv::Mat project_image;
static pcl::PointCloud<PointType>::Ptr copy_depthCloud (new pcl::PointCloud<PointType>());

void LivoxMsgToPcl(const livox_ros_driver2::CustomMsgConstPtr &cloud_, pcl::PointCloud<PointType>::Ptr &cloud_out)
{
    cloud_out->clear();
    auto msg_ = cloud_;

    for (int i = 0; i < msg_->points.size(); i++)
    {
        PointType pt;
        pt.x = msg_->points[i].x;
        pt.y = msg_->points[i].y;
        pt.z = msg_->points[i].z;
        pt.intensity = msg_->points[i].reflectivity;

        cloud_out->push_back(pt);
    }
    
    ROS_INFO("Completed LIVOX Msg convert to pcl XYZI points.\n");
    
    return;
}

float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

void getColor(float p, float np, float&r, float&g, float&b) 
{
        float inc = 6.0 / np;
        float x = p * inc;
        r = 0.0f; g = 0.0f; b = 0.0f;
        if ((0 <= x && x <= 1) || (5 <= x && x <= 6)) r = 1.0f;
        else if (4 <= x && x <= 5) r = x - 4;
        else if (1 <= x && x <= 2) r = 1.0f - (x - 1);

        if (1 <= x && x <= 3) g = 1.0f;
        else if (0 <= x && x <= 1) g = x - 0;
        else if (3 <= x && x <= 4) g = 1.0f - (x - 3);

        if (3 <= x && x <= 5) b = 1.0f;
        else if (2 <= x && x <= 3) b = x - 2;
        else if (5 <= x && x <= 6) b = 1.0f - (x - 5);
        r *= 255.0;
        g *= 255.0;
        b *= 255.0;
}


void ImageCallback(const sensor_msgs::ImageConstPtr &image_msg)
{
    
    ros::Time cur_image_time = image_msg->header.stamp;

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);

    cv::Mat cv_image, circle_image;
    cv_image = cv_ptr->image;

    double scale_factor = 0.5;
    cv::Size new_size(static_cast<int>(cv_image.cols * scale_factor), static_cast<int>(cv_image.rows * scale_factor));
    cv::Mat scaled_image;
    cv::resize(cv_image, scaled_image, new_size);

    circle_image = scaled_image.clone();

    mtx_lidar.lock();
    copy_depthCloud = depthCloud;
    mtx_lidar.unlock();

    if(copy_depthCloud->empty())
        return;


    #pragma omp parallel for num_threads(6)
    for (auto & pt : copy_depthCloud->points)
    {

        float dist,r,g,b;
        int row = round((atan2(pt.z,sqrt(pow(pt.x, 2) + pow(pt.y, 2))) * (180.0f / M_PI)) / 0.3);
        int col = round((atan2(pt.x, pt.y) * (180.0f / M_PI)) / 0.3);
        row = 256 - row;

        if (row < 0 || row >= 512 || col < 0 || col >= 640)
            continue;

        dist = pointDistance(pt);
        getColor(dist,50,r,g,b);
        // circle_image.at<cv::Vec3b>(row, col) = cv::Vec3b(r, g, b);  //根据距离赋值三通道
        cv::circle(circle_image, cv::Point2f(col,row), 0, cv::Scalar(r, g, b), 5);//(x,y)
        // ROS_INFO("创建成功\n");  
    }

    // cv::addWeighted(scaled_image, 1.0,circle_image, 0.5, 0, scaled_image);

    cv_bridge::CvImage bridge;
    bridge.image = circle_image;
    bridge.encoding = "bgr8";
    sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
    imageShowPointer->header.stamp = cur_image_time;
    pub_depth_image.publish(imageShowPointer);
    // cv::imwrite("/home/hong/slam/hhh_ws/src/nonlinear_opt_in_slam/data/saveImageRGB.png", scaled_image);
    // cv::imwrite("/home/hong/slam/hhh_ws/src/nonlinear_opt_in_slam/data/saveRangeImageRGB.png",circle_image);

    // ROS_INFO("*****************************************\n");
    // ROS_WARN("*****************************************\n");

    return;
}

void LivoxCallback(const livox_ros_driver2::CustomMsgConstPtr& cloud_msg)
{
    static int lidar_count = -1;
    //间隔处理
    if (++lidar_count % (LiDAR_SKIP + 1) != 0)
        return;
    
    //订阅 lidar -> camera 的静态tf树（外参）
    static tf::TransformListener listener;
    static tf::StampedTransform transform;
    try
    {
        /* code */
        listener.waitForTransform("world", "livox_frame", ros::Time::now(), ros::Duration(0.01));
        listener.lookupTransform("world", "livox_frame", ros::Time::now(), transform);
    }catch(tf::TransformException ex)
    {
        std::cerr << "lidar no tf to world." << '\n';
        return;
    }

    //提取TF变换参数
    double xCur,yCur,zCur,rollCur,pitchCur,yawCur;
    xCur = transform.getOrigin().x();
    yCur = transform.getOrigin().y();
    zCur = transform.getOrigin().z();
    tf::Matrix3x3 m(transform.getRotation());
    m.getRPY(rollCur, pitchCur, yawCur);
    Eigen::Affine3f transNow = pcl::getTransformation(xCur,yCur,zCur,rollCur,pitchCur,yawCur);

    auto msg = cloud_msg;

    // 提取点云并降采样
    pcl::PointCloud<PointType>::Ptr laser_cloud(new pcl::PointCloud<PointType>());
    LivoxMsgToPcl(msg,laser_cloud);
    pcl::PointCloud<PointType>::Ptr laser_cloud_DS (new pcl::PointCloud<PointType>());
    static pcl::VoxelGrid<PointType> downSizeFilter;
    // downSizeFilter.setLeafSize(0.2,0.2,0.2);
    // downSizeFilter.setInputCloud(laser_cloud);
    // downSizeFilter.filter(*laser_cloud_DS);
    // *laser_cloud = *laser_cloud_DS;

    //保留 +x
    pcl::PointCloud<PointType>::Ptr laser_cloud_filter (new pcl::PointCloud<PointType>());
    for (auto& pt:laser_cloud->points)
    {
        if (pt.x >= 0)
        {
            laser_cloud_filter->push_back(pt);
        }
    }
    *laser_cloud = *laser_cloud_filter;

    // 转换lidar -> cam（lidar和相机的标定外参）
    pcl::PointCloud<PointType>::Ptr laser_cloud_offset (new pcl::PointCloud<PointType>());
    Eigen::Affine3f transOffset = pcl::getTransformation(0.,0.,0.,0.,0.,0.);// 填写正确的外参
    pcl::transformPointCloud(*laser_cloud,*laser_cloud_offset,transOffset);
    *laser_cloud = *laser_cloud_offset;

    // 根据tf树 转换新点到全局系下
    pcl::PointCloud<PointType>::Ptr laser_cloud_global (new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud,*laser_cloud_global,transNow);
    *depthCloud = *laser_cloud_global;
    // 保存转换后的新点云
    double timeScanCur = msg->header.stamp.toSec();
    // timeQueue.push_back(timeScanCur);
    // cloudQueue.push_back(*laser_cloud_global);

    // 累计5s内的点云
    // while (!timeQueue.empty())
    // {
    //     /* code */
    //     if (timeScanCur - timeQueue.front() > 5.0)
    //     {
    //         cloudQueue.pop_front();
    //         timeQueue.pop_front();
    //     }else{
    //         break;
    //     }
    // }

    std::lock_guard<std::mutex> lock(mtx_lidar);

    // depthCloud->clear();
    // for (int i = 0; i < (int)cloudQueue.size(); ++i)
        // *depthCloud += cloudQueue[i];
   

    pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(depthCloud);
    downSizeFilter.filter(*depthCloudDS);
    *depthCloud = *depthCloudDS;

    return;
}

int main(int argc,char **argv)
{
    setlocale(LC_ALL, "");
    ros::init(argc,argv,"ls_slam");
    ros::NodeHandle nh;
    ROS_INFO("\033[1;32m**** Mid360 to RangeImage Node Start. ****\033[0m\n");

    ros::Subscriber lidar_;
    ros::Subscriber image_;

    pub_depth_image =   nh.advertise<sensor_msgs::Image>("/fusion_image",   5);
    

    lidar_ = nh.subscribe("/livox/lidar",5,LivoxCallback,ros::TransportHints().tcpNoDelay());
    image_ = nh.subscribe("/camera/image",5,ImageCallback,ros::TransportHints().tcpNoDelay());

    
    ros::spin();
    return 0;

}

/*
    //transform to rangeImage
    depthCloud->width = depthCloud->size();
    depthCloud->height = 1;
    We now want to create a range image from the above point cloud, with a 1deg angular resolution
    float angularResolution = (float) (  1.0f * (M_PI/180.0f));  //   1.0 degree in radians
    float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians x
    float maxAngleHeight    = (float) (360.0f * (M_PI/180.0f));  // 180.0 degree in radians y
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    float noiseLevel= 0.00f;
    float minRange = 0.10f;
    int borderSize = 1;
  
    pcl::RangeImage rangeImage;
    rangeImage.createFromPointCloud(*depthCloud, angularResolution, maxAngleWidth, maxAngleHeight,
                                  sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
    std::cout << "******** \n"<< rangeImage << "\n";
   

    float* ranges = rangeImage.getRangesArray();
	unsigned char* rgb_image = pcl::visualization::FloatImageUtils::getVisualImage(ranges, rangeImage.width, rangeImage.height);
	pcl::io::saveRgbPNGFile("/home/hong/slam/hhh_ws/src/nonlinear_opt_in_slam/data/saveRangeImageRGB.png", rgb_image, rangeImage.width, rangeImage.height);
*/
    