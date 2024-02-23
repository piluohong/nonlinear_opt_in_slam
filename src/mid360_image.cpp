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

ros::Subscriber lidar_;
ros::Subscriber image_;

ros::Publisher pub_depth_image;
ros::Publisher pubCloud;


//config params
static int LiDAR_SKIP = 0;

std::mutex mtx_lidar;

deque<livox_ros_driver2::CustomMsgConstPtr> vec_;
deque<pcl::PointCloud<PointType>> cloudQueue;
deque<double> timeQueue;
double laser_time;


pcl::PointCloud<PointType>::Ptr depthCloud (new pcl::PointCloud<PointType>());
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colorCloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
static pcl::PointCloud<PointType>::Ptr copy_depthCloud (new pcl::PointCloud<PointType>());

static cv::Mat project_image;
static Eigen::Matrix4f extrinsicMat_RT; // 外参旋转矩阵3*3和平移向量3*1
static cv::Mat intrisicMat(3, 4, cv::DataType<double>::type);// 内参3*4的投影矩阵，最后一列是三个零
static cv::Mat distCoeffs(3, 3, cv::DataType<double>::type);

void CalibrationData()
{
    //hku1.bag -> config
    extrinsicMat_RT(0, 0) = 0.00162756;
	extrinsicMat_RT(0, 1) = -0.999991;
	extrinsicMat_RT(0, 2) = 0.00390957;
	extrinsicMat_RT(0, 3) = 0.0409257;
	extrinsicMat_RT(1, 0) = -0.0126748;
	extrinsicMat_RT(1, 1) = -0.00392989;
	extrinsicMat_RT(1, 2) = -0.999912;
	extrinsicMat_RT(1, 3) = 0.0318424;
	extrinsicMat_RT(2, 0) = 0.999918;
	extrinsicMat_RT(2, 1) = 0.00157786;
	extrinsicMat_RT(2, 2) = -0.012681;
	extrinsicMat_RT(2, 3) = -0.0927219;
	extrinsicMat_RT(3, 0) = 0.0;
	extrinsicMat_RT(3, 1) = 0.0;
	extrinsicMat_RT(3, 2) = 0.0;
	extrinsicMat_RT(3, 3) = 1.0;

    intrisicMat.at<double>(0, 0) =  863.590518437255;
	intrisicMat.at<double>(0, 1) = 0.000000e+00;
	intrisicMat.at<double>(0, 2) =  621.666074631063;
	intrisicMat.at<double>(0, 3) = 0.000000e+00;
	intrisicMat.at<double>(1, 0) = 0.000000e+00;
	intrisicMat.at<double>(1, 1) =  863.100180533059;
	intrisicMat.at<double>(1, 2)  = 533.971978652819;
	intrisicMat.at<double>(1, 3) = 0.000000e+00;
	intrisicMat.at<double>(2, 0) = 0.000000e+00;
	intrisicMat.at<double>(2, 1) = 0.000000e+00;
	intrisicMat.at<double>(2, 2) = 1.000000e+00;
	intrisicMat.at<double>(2, 3) = 0.000000e+00;

	// distCoeffs.at<double>(0) = -0.008326874784366894;
	// distCoeffs.at<double>(1) = -0.06967846599874981;
	// distCoeffs.at<double>(2) = 0.006185220615585947;
	// distCoeffs.at<double>(3) = -0.01133018681519818;
	// distCoeffs.at<double>(4) = 0.5462976722456516;

}

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
    double cur_laser_time;
    mtx_lidar.lock();
    copy_depthCloud = depthCloud;
    cur_laser_time = laser_time;
    mtx_lidar.unlock();
    
    ros::Time cur_image_time = image_msg->header.stamp;
    double cur_time = cur_image_time.toSec();

    if (abs(cur_time - cur_laser_time) > 0.1)
        return;

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat cv_image;
    cv_image = cv_ptr->image;
    cv::Mat circle_image(1024, 1280, CV_8UC3, cv::Scalar(255,255,255));

    circle_image = cv_image.clone();
    //原始图像缩放0.5倍
    // double scale_factor = 0.5;
    // cv::Size new_size(static_cast<int>(cv_image.cols * scale_factor), static_cast<int>(cv_image.rows * scale_factor));
    // cv::Mat scaled_image;
    // cv::resize(cv_image, scaled_image, new_size);
    
    if(copy_depthCloud->empty())
        return;

    // float row_res = 77.2 / 1024;
    // float col_res = 70.4 / 1280;

    cv::Mat X(4, 1, cv::DataType<double>::type);
	cv::Mat Y(3, 1, cv::DataType<double>::type);

    #pragma omp parallel for num_threads(6)
    for (auto & pt : copy_depthCloud->points)
    {
        // int row = round((atan2(pt.z,sqrt(pow(pt.x, 2) + pow(pt.y, 2))) * (180.0f / M_PI)) / row_res);
        // int col = round((atan2(pt.x, pt.y) * (180.0f / M_PI)) / col_res);
         
         X.at<double>(0,0) = pt.x;
         X.at<double>(1,0) = pt.y;
         X.at<double>(2,0) = pt.z;
         X.at<double>(3,0) = 1;

        Y = intrisicMat  * X; //相机坐标投影到像素坐标
        cv::Point2f u_v;
        u_v.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
		u_v.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        if(u_v.x < 0 || u_v.y < 0 || u_v.x > 1280 || u_v.y > 1024 )
            continue;

        float dist,r,g,b;
        dist = pointDistance(pt);
        getColor(dist,50,r,g,b);
        cv::circle(circle_image, cv::Point2f(u_v.x,u_v.y), 0, cv::Scalar(r, g, b),5);
        
        pcl::PointXYZRGBNormal p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;
        p.r = cv_image.at<cv::Vec3b>(u_v.y,u_v.x)[0];
        p.g = cv_image.at<cv::Vec3b>(u_v.y,u_v.x)[1];
        p.b = cv_image.at<cv::Vec3b>(u_v.y,u_v.x)[2];

        colorCloud->points.push_back(p);
        // ROS_INFO("创建成功\n");  
    }

    // cv::addWeighted(cv_image, 1.0,circle_image, 0.5, 0, cv_image);
    cv_bridge::CvImage bridge;
    bridge.image = circle_image;
    bridge.encoding = "bgr8";
    sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
    imageShowPointer->header.stamp = ros::Time::now();
    pub_depth_image.publish(imageShowPointer);

    colorCloud->width = colorCloud->points.size();
    colorCloud->height = 1;
    ROS_WARN(">>> %d\n",colorCloud->width);
    sensor_msgs::PointCloud2 color_msg;
    pcl::toROSMsg(*colorCloud, color_msg);			   
    color_msg.header.frame_id = "camera_link";		   
    color_msg.header.stamp = ros::Time::now();
    pubCloud.publish(color_msg);

    colorCloud->clear();

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
    
    //保留 +x 区域
    pcl::PointCloud<PointType>::Ptr laser_cloud_filter (new pcl::PointCloud<PointType>());
    for (auto& pt:laser_cloud->points)
    {
        if (pt.x >= 0)
        {
            laser_cloud_filter->push_back(pt);
        }
    }
    *laser_cloud = *laser_cloud_filter;

    //转换lidar -> cam（lidar和相机的标定外参）
    Eigen::Matrix3f linearPart = extrinsicMat_RT.topLeftCorner<3, 3>();
    Eigen::Vector3f translation = extrinsicMat_RT.topRightCorner<3, 1>();
    pcl::PointCloud<PointType>::Ptr laser_cloud_offset (new pcl::PointCloud<PointType>());
    Eigen::Affine3f transOffset;
    transOffset.linear() = linearPart;
    transOffset.translation() = translation;
    pcl::transformPointCloud(*laser_cloud,*laser_cloud_offset,transOffset);
    *laser_cloud = *laser_cloud_offset;

    // 根据tf树 转换新点到全局系下
    pcl::PointCloud<PointType>::Ptr laser_cloud_global (new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud,*laser_cloud_global,transNow);
    *depthCloud = *laser_cloud_global;

    // 保存转换后的新点云
    
    double timeScanCur = msg->header.stamp.toSec();
    timeQueue.push_back(timeScanCur);
    cloudQueue.push_back(*laser_cloud_global);

    //累计1s内的点云
    while (!timeQueue.empty())
    {
        /* code */
        if (timeScanCur - timeQueue.front() > 1)
        {
            cloudQueue.pop_front();
            timeQueue.pop_front();
        }else{
            break;
        }
    }
    depthCloud->clear();
    std::lock_guard<std::mutex> lock(mtx_lidar);
    laser_time = timeScanCur;
    for (int i = 0; i < (int)cloudQueue.size(); ++i)
        *depthCloud += cloudQueue[i];
   
    // pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
    // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    // downSizeFilter.setInputCloud(depthCloud);
    // downSizeFilter.filter(*depthCloudDS);
    // *depthCloud = *depthCloudDS;

    return;
}

int main(int argc,char **argv)
{
    setlocale(LC_ALL, "");
    ros::init(argc,argv,"ls_slam");
    ros::NodeHandle nh;
    ROS_INFO("\033[1;32m**** Mid360 to RangeImage Node Start. ****\033[0m\n");

    CalibrationData();

    pub_depth_image =   nh.advertise<sensor_msgs::Image>("/fusion_image",1,true);
    pubCloud = nh.advertise<sensor_msgs::PointCloud2>("/livox/color_lidar", 1,true);	
    

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
    