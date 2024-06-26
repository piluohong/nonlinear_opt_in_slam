#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#define PCL_NO_PRECOMPILE
#include <pcl/filters/crop_box.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/io/png_io.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <livox_ros_driver2/CustomMsg.h>

#include <iostream>
#include <queue>
#include <mutex>
#include <thread>
#include <omp.h>

struct LivoxPoint
{
    PCL_ADD_POINT4D;
    float intensity;           // intensity
    std::uint32_t offset_time; // LIVOX: time from beginning of scan in nanoseconds
    int line;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(LivoxPoint,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint32_t, offset_time, offset_time)(int, line, line))

struct PointXYZIRGBNormal : public pcl::PointXYZRGBNormal
{
    PCL_ADD_INTENSITY;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRGBNormal,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, normal_x, normal_x)(float, normal_y, normal_y)(float, normal_z, normal_z)(std::uint8_t, r, r)(std::uint8_t, g, g)(std::uint8_t, b, b)(float, curvature, curvature))

using PointType = pcl::PointXYZRGBNormal;
using namespace std;
ros::Subscriber lidar_;
ros::Subscriber image_;

ros::Publisher pub_depth_image;
ros::Publisher pubCloud;
ros::Publisher pub_;

static int lidar_count = 1;
static int lidar_nums = 0;
std::vector<int> lidar_pts{10000};

// config params
static int LiDAR_SKIP = 0;

std::mutex mtx_lidar;

deque<livox_ros_driver2::CustomMsgConstPtr> vec_;
deque<pcl::PointCloud<PointType>> cloudQueue;
deque<double> timeQueue;
double laser_time;

static pcl::PointCloud<PointType>::Ptr depthCloud(new pcl::PointCloud<PointType>());

std::vector<std::pair<sensor_msgs::Image::ConstPtr, double>> image_buff;

sensor_msgs::Image::ConstPtr image_ros;
double image_time = 0.0;

static Eigen::Matrix4f extrinsicMat_RT; // 外参旋转矩阵3*3和平移向量3*1
static Eigen::Affine3f transOffset, C_to_L;
static cv::Mat intrisicMat(3, 4, cv::DataType<double>::type); // 内参3*4的投影矩阵，最后一列是三个零
static cv::Mat intrisic(3, 3, cv::DataType<double>::type);    // 内参3*3矩阵
static cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);  // 畸变向量

static Eigen::Matrix4f mat;

void CalibrationData()
{
    // hku1.bag -> config lidar->cam
    //  extrinsicMat_RT(0, 0) = 0.00162756;
    //  extrinsicMat_RT(0, 1) = -0.999991;
    //  extrinsicMat_RT(0, 2) = 0.00390957;
    //  extrinsicMat_RT(0, 3) = 0.0409257;
    //  extrinsicMat_RT(1, 0) = -0.0126748;
    //  extrinsicMat_RT(1, 1) = -0.00392989;
    //  extrinsicMat_RT(1, 2) = -0.999912;
    //  extrinsicMat_RT(1, 3) = 0.0318424;
    //  extrinsicMat_RT(2, 0) = 0.999918;
    //  extrinsicMat_RT(2, 1) = 0.00157786;
    //  extrinsicMat_RT(2, 2) = -0.012681;
    //  extrinsicMat_RT(2, 3) = -0.0927219;
    //  extrinsicMat_RT(3, 0) = 0.0;
    //  extrinsicMat_RT(3, 1) = 0.0;
    //  extrinsicMat_RT(3, 2) = 0.0;
    //  extrinsicMat_RT(3, 3) = 1.0;

    // intrisicMat.at<double>(0, 0) =  863.590518437255;
    // intrisicMat.at<double>(0, 1) = 0.000000e+00;
    // intrisicMat.at<double>(0, 2) =  621.666074631063;
    // intrisicMat.at<double>(0, 3) = 0.000000e+00;
    // intrisicMat.at<double>(1, 0) = 0.000000e+00;
    // intrisicMat.at<double>(1, 1) =  863.100180533059;
    // intrisicMat.at<double>(1, 2)  = 533.971978652819;
    // intrisicMat.at<double>(1, 3) = 0.000000e+00;
    // intrisicMat.at<double>(2, 0) = 0.000000e+00;
    // intrisicMat.at<double>(2, 1) = 0.000000e+00;
    // intrisicMat.at<double>(2, 2) = 1.000000e+00;
    // intrisicMat.at<double>(2, 3) = 0.000000e+00;

    // hku_main_building -> config cam->lidar
    //   extrinsicMat_RT(0, 0) = -0.00113207;
    //   extrinsicMat_RT(0, 1) = -0.0158688;
    //   extrinsicMat_RT(0, 2) = 0.999873;
    //   extrinsicMat_RT(0, 3) = 0.050166;
    //   extrinsicMat_RT(1, 0) = -0.9999999;
    //   extrinsicMat_RT(1, 1) = -0.000486594;
    //   extrinsicMat_RT(1, 2) = -0.00113994;
    //   extrinsicMat_RT(1, 3) = 0.0474116;
    //   extrinsicMat_RT(2, 0) = 0.000504622;
    //   extrinsicMat_RT(2, 1) = -0.999874;
    //   extrinsicMat_RT(2, 2) = -0.0158682;
    //   extrinsicMat_RT(2, 3) = -0.0312415;
    //   extrinsicMat_RT(3, 0) = 0.0;
    //   extrinsicMat_RT(3, 1) = 0.0;
    //   extrinsicMat_RT(3, 2) = 0.0;
    //   extrinsicMat_RT(3, 3) = 1.0;

    //   intrisicMat.at<double>(0, 0) = intrisic.at<double>(0, 0) = 863.4241;
    //   intrisicMat.at<double>(0, 1) = intrisic.at<double>(0, 1) = 0.000000e+00;
    //   intrisicMat.at<double>(0, 2) = intrisic.at<double>(0, 2) = 640.6808;
    //   intrisicMat.at<double>(0, 3) = 0.000000e+00;
    //   intrisicMat.at<double>(1, 0) = intrisic.at<double>(1, 0) = 0.000000e+00;
    //   intrisicMat.at<double>(1, 1) = intrisic.at<double>(1, 1) = 863.4171;
    //   intrisicMat.at<double>(1, 2) = intrisic.at<double>(1, 2) = 518.3392;
    //   intrisicMat.at<double>(1, 3) = 0.000000e+00;
    //   intrisicMat.at<double>(2, 0) = intrisic.at<double>(2, 0) = 0.000000e+00;
    //   intrisicMat.at<double>(2, 1) = intrisic.at<double>(2, 1) = 0.000000e+00;
    //   intrisicMat.at<double>(2, 2) = intrisic.at<double>(2, 2) = 1.000000e+00;
    //   intrisicMat.at<double>(2, 3) = 0.000000e+00;

    // mid360_455 camera->lidar
    extrinsicMat_RT(0, 0) = -0.00113207;
    extrinsicMat_RT(0, 1) = -0.0158688;
    extrinsicMat_RT(0, 2) = 0.999873;
    extrinsicMat_RT(0, 3) = 0.00; // 0.050166;
    extrinsicMat_RT(1, 0) = -0.9999999;
    extrinsicMat_RT(1, 1) = -0.000486594;
    extrinsicMat_RT(1, 2) = -0.00113994;
    extrinsicMat_RT(1, 3) = 0.00; // 0.0474116;
    extrinsicMat_RT(2, 0) = 0.000504622;
    extrinsicMat_RT(2, 1) = -0.999874;
    extrinsicMat_RT(2, 2) = -0.0158682;
    extrinsicMat_RT(2, 3) = 0.05; //-0.0312415;
    extrinsicMat_RT(3, 0) = 0.0;
    extrinsicMat_RT(3, 1) = 0.0;
    extrinsicMat_RT(3, 2) = 0.0;
    extrinsicMat_RT(3, 3) = 1.0;

    intrisicMat.at<double>(0, 0) = intrisic.at<double>(0, 0) = 653.97880377;
    intrisicMat.at<double>(0, 1) = intrisic.at<double>(0, 1) = 0.000000e+00;
    intrisicMat.at<double>(0, 2) = intrisic.at<double>(0, 2) = 650.88028755;
    intrisicMat.at<double>(0, 3) = 0.000000e+00;
    intrisicMat.at<double>(1, 0) = intrisic.at<double>(1, 0) = 0.000000e+00;
    intrisicMat.at<double>(1, 1) = intrisic.at<double>(1, 1) = 617.04141382;
    intrisicMat.at<double>(1, 2) = intrisic.at<double>(1, 2) = 364.93377359;
    intrisicMat.at<double>(1, 3) = 0.000000e+00;
    intrisicMat.at<double>(2, 0) = intrisic.at<double>(2, 0) = 0.000000e+00;
    intrisicMat.at<double>(2, 1) = intrisic.at<double>(2, 1) = 0.000000e+00;
    intrisicMat.at<double>(2, 2) = intrisic.at<double>(2, 2) = 1.000000e+00;
    intrisicMat.at<double>(2, 3) = 0.000000e+00;

    // // std::cout << intrisic << std::endl;

    // //k1,k2,p1,p2,k3
    // distCoeffs.at<double>(0) = 0.043652678729709814;//-0.1080;
    // distCoeffs.at<double>(1) =  2.997254797883524;//0.1050;
    // distCoeffs.at<double>(2) = -13.835973324062932;//-1.2872e-04;
    // distCoeffs.at<double>(3) =  22.033042492264435;//5.7923e-05;
    // distCoeffs.at<double>(4) = 0.000000;//-0.0222;

    // 绕y轴旋转180°
    //  mat << -1,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,1;

    // Eigen::Matrix3f linearPart = mat.topLeftCorner<3, 3>();
    // Eigen::Vector3f translation = mat.topRightCorner<3, 1>();
    // transOffset.linear() = linearPart;
    // transOffset.translation() = translation;

    // 转换lidar -> cam（lidar和相机外参）
    Eigen::Matrix3f linearPart = extrinsicMat_RT.topLeftCorner<3, 3>();
    Eigen::Vector3f translation = extrinsicMat_RT.topRightCorner<3, 1>();

    transOffset.linear() = linearPart;
    transOffset.translation() = translation;

    C_to_L.linear() = linearPart;
    C_to_L.translation() = translation;

    transOffset = transOffset.inverse();
}

void LivoxMsgToPcl(const livox_ros_driver2::CustomMsgConstPtr &cloud_, pcl::PointCloud<PointType>::Ptr &cloud_out)
{
    auto msg_ = cloud_;
#pragma omp parallel for num_threads(12)
    for (int i = 0; i < msg_->points.size(); i++)
    {

        PointType pt;
        pt.x = msg_->points[i].x;
        pt.y = msg_->points[i].y;
        pt.z = msg_->points[i].z;
        // pt.intensity = msg_->points[i].reflectivity;
        // pt.line = msg_->points[i].line;
        // pt.offset_time = msg_->points[i].offset_time;
        cloud_out->push_back(pt);
    }

    ROS_INFO("Completed LIVOX Msg convert to Livox(pcl) points.\n");

    return;
}

float pointDistance(Eigen::Vector3f p)
{
    return sqrt(p(0) * p(0) + p(1) * p(1) + p(2) * p(2));
}

void getColor(float p, float np, float &r, float &g, float &b)
{
    float inc = 6.0 / np;
    float x = p * inc;
    r = 0.0f;
    g = 0.0f;
    b = 0.0f;
    if ((0 <= x && x <= 1) || (5 <= x && x <= 6))
        r = 1.0f;
    else if (4 <= x && x <= 5)
        r = x - 4;
    else if (1 <= x && x <= 2)
        r = 1.0f - (x - 1);

    if (1 <= x && x <= 3)
        g = 1.0f;
    else if (0 <= x && x <= 1)
        g = x - 0;
    else if (3 <= x && x <= 4)
        g = 1.0f - (x - 3);

    if (3 <= x && x <= 5)
        b = 1.0f;
    else if (2 <= x && x <= 3)
        b = x - 2;
    else if (5 <= x && x <= 6)
        b = 1.0f - (x - 5);
    r *= 255.0;
    g *= 255.0;
    b *= 255.0;
}

void ros2cv(const sensor_msgs::Image::ConstPtr &image_ros, cv::Mat &image, cv::Mat &intrisic, cv::Mat &distCoeffs, bool corr)
{
    // ROS_Image -> OpenCV_Image
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_ros, sensor_msgs::image_encodings::RGB8);
    image = cv_ptr->image;

    if (corr)
    {
        /* code */
        // 图像去畸变
        cv::Mat map1, map2;
        cv::Size imageSize = image.size();
        cv::initUndistortRectifyMap(intrisic, distCoeffs, cv::Mat(), cv::getOptimalNewCameraMatrix(intrisic, distCoeffs, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);
        cv::remap(image, image, map1, map2, cv::INTER_LINEAR); // correct the distortion
    }

    return;
}

void ImageCallback(const sensor_msgs::ImageConstPtr &image_msg)
{
    // 相机数据打包
    std::lock_guard<std::mutex> lock(mtx_lidar);
    ros::Time image_time = image_msg->header.stamp;
    double cur_image_time = image_time.toSec();
    image_buff.push_back(std::make_pair(image_msg, cur_image_time));
    std::cout << "image_buff size: " << image_buff.size() << std::endl;
    // ROS_WARN("*******\n");

    // cv::addWeighted(cv_image, 1.0,circle_image, 0.5, 0, cv_image);

    // cv::imwrite("/home/hong/slam/hhh_ws/src/nonlinear_opt_in_slam/data/saveRangeImageRGB.png",circle_image);
    // ROS_INFO("*****************************************\n");

    return;
}

template <typename T>
inline bool HasInf(const T &p)
{
    return (std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z));
}

template <typename T>
inline bool HasNan(const T &p)
{
    return (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z));
}

template <typename T>
inline bool IsNear(const T &p1, const T &p2)
{
    return ((abs(p1.x - p2.x) < 1e-7) || (abs(p1.y - p2.y) < 1e-7) ||
            (abs(p1.z - p2.z) < 1e-7));
}

// void VelodyneCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
// {
//     // pcl::PointCloud<pcl::PointXYZ> msg;
//     // pcl::fromROSMsg(*cloud_msg, msg);
//     // int available_nums = 0;
//     // for (size_t i = 0; i < msg.size(); i++)
//     // {
//     //      available_nums += 1;
//     // }

//     // lidar_nums = available_nums;
//     // lidar_pts.push_back(lidar_nums);
//     // std::cout << ">>> lidar_count: "<< lidar_count << ", Current Frame points: " << lidar_nums << std::endl;
//     // lidar_count++;

//     static int lidar_count = -1;
//     //间隔处理
//     if (++lidar_count % (LiDAR_SKIP + 1) != 0)
//         return;

//     //订阅 lidar -> camera 的静态tf树（外参）
//     static tf::TransformListener listener;
//     static tf::StampedTransform transform;
//     try
//     {
//         /* code */
//         listener.waitForTransform("world", "livox_frame", ros::Time::now(), ros::Duration(0.001));
//         listener.lookupTransform("world", "livox_frame", ros::Time::now(), transform);
//     }
//     catch(tf::TransformException ex)
//     {
//         std::cerr << "lidar no tf to world." << '\n';
//         return;
//     }

//     //提取TF变换参数
//     double xCur,yCur,zCur,rollCur,pitchCur,yawCur;
//     xCur = transform.getOrigin().x();
//     yCur = transform.getOrigin().y();
//     zCur = transform.getOrigin().z();
//     tf::Matrix3x3 m(transform.getRotation());
//     m.getRPY(rollCur, pitchCur, yawCur);
//     Eigen::Affine3f transNow = pcl::getTransformation(xCur,yCur,zCur,rollCur,pitchCur,yawCur);

//     auto msg = cloud_msg;

//     // 提取点云并降采样
//     pcl::PointCloud<PointType>::Ptr laser_cloud(new pcl::PointCloud<PointType>());
//     pcl::fromROSMsg(*msg,*laser_cloud);
//     // pcl::PointCloud<PointType>::Ptr laser_cloud_DS (new pcl::PointCloud<PointType>());
//     static pcl::VoxelGrid<PointType> downSizeFilter;
//     // downSizeFilter.setLeafSize(0.2,0.2,0.2);
//     // downSizeFilter.setInputCloud(laser_cloud);
//     // downSizeFilter.filter(*laser_cloud_DS);
//     // *laser_cloud = *laser_cloud_DS;

//     //保留 +x 区域
//     pcl::PointCloud<PointType>::Ptr laser_cloud_filter (new pcl::PointCloud<PointType>());
//     for (auto& pt:laser_cloud->points)
//     {
//         if (pt.x >= 0)
//         {
//             laser_cloud_filter->push_back(pt);
//         }
//     }
//     *laser_cloud = *laser_cloud_filter;

//     //转换lidar -> cam（lidar和相机的标定外参）
//     pcl::PointCloud<PointType>::Ptr laser_cloud_offset (new pcl::PointCloud<PointType>());
//     pcl::transformPointCloud(*laser_cloud,*laser_cloud_offset,transOffset);
//     *laser_cloud = *laser_cloud_offset;

//     // 根据tf树 转换新点到全局系下
//     pcl::PointCloud<PointType>::Ptr laser_cloud_global (new pcl::PointCloud<PointType>());
//     pcl::transformPointCloud(*laser_cloud,*laser_cloud_global,transNow);

//      // 保存新的点云
//     double timeScanCur = msg->header.stamp.toSec();
//      std::lock_guard<std::mutex> lock(mtx_lidar);
//     *depthCloud = *laser_cloud_global;
//     // timeQueue.push_back(timeScanCur);
//     // cloudQueue.push_back(*laser_cloud_global);

//     // //累计1s内的点云
//     // while (!timeQueue.empty())
//     // {
//     //     /* code */
//     //     if (timeScanCur - timeQueue.front() > 1)
//     //     {
//     //         cloudQueue.pop_front();
//     //         timeQueue.pop_front();
//     //     }else{
//     //         break;
//     //     }
//     // }
//     // depthCloud->clear();

//     laser_time = timeScanCur;
//     // for (int i = 0; i < (int)cloudQueue.size(); ++i)
//     //     *depthCloud += cloudQueue[i];

//     // pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
//     // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
//     // downSizeFilter.setInputCloud(depthCloud);
//     // downSizeFilter.filter(*depthCloudDS);
//     // *depthCloud = *depthCloudDS;
//     sensor_msgs::PointCloud2 raw_msg;
//     raw_msg = *msg;
//     raw_msg.header.stamp = ros::Time::now();
//     raw_msg.header.frame_id = "laser_link";
//     pub_.publish(raw_msg);

//     return;
// }
// void laserCloudAndImageHandler_(const livox_ros_driver2::CustomMsg::ConstPtr& cloud_msg)
// {

//     //筛选有效点
//     int avaliabe_nums = 0;
//     // for(size_t i = 0;i < cloud_msg->point_num;i++)
//     // {
//     //     // if(livoxpt.line < 3 && ((livoxpt.tag & 0x30) == 0x10 ||
//     //     // (livoxpt.tag & 0x30) == 0x00) &&
//     //     //     !HasInf(livoxpt) && !HasNan(livoxpt) &&
//     //     //     !IsNear(livoxpt,cloud_msg->points[i - 1])){
//     //     //     avaliabe_nums += 1;
//     //     // }
//     // }
//     auto msg = cloud_msg;
//     pcl::PointCloud<LivoxPoint>::Ptr cloud_(new pcl::PointCloud<LivoxPoint>());

//     LivoxMsgToPcl(cloud_msg,cloud_);
//     //盒子滤波
//     pcl::CropBox<LivoxPoint> crop;
//     crop.setNegative(true);
//     crop.setMin(Eigen::Vector4f(-0.6, -0.6, -0.6, 1.0));
//     crop.setMax(Eigen::Vector4f(0.6, 0.6, 0.6, 1.0));
//     crop.setInputCloud(cloud_);
//     crop.filter(*cloud_);
//     pcl::transformPointCloud(*cloud_,*cloud_,transOffset);

//     for(auto& pt : cloud_->points)
//     {
//         pt.z = pt.z +3;
//     }
//      sensor_msgs::PointCloud2 raw_msg;
//      pcl::toROSMsg(*cloud_,raw_msg);
//     raw_msg.header.stamp =  msg->header.stamp;
//     raw_msg.header.frame_id ="livox_frame";

//     pub_.publish(raw_msg);

//     // lidar_nums = avaliabe_nums;
//     // lidar_pts.push_back(lidar_nums);
//     // std::cout << ">>> lidar_count: "<< lidar_count << ", Current Frame points: " << lidar_nums << std::endl;
//     // lidar_count++;
// }
void laserCloudAndImageHandler(const sensor_msgs::ImageConstPtr &image_msg, const livox_ros_driver2::CustomMsg::ConstPtr &cloud_msg)
{
    // mtx_lidar.lock();
    // image_ros = image_buff.back().first;
    // image_time = image_buff.back().second;
    // mtx_lidar.unlock();
    // auto msg = cloud_msg;
    image_ros = image_msg;
    cv::Mat show_image, cv_image, circle_image;
    ros2cv(image_ros, show_image, intrisic, distCoeffs, false);
    // ROS_WARN("Debug: %d\n",(int)show_image.empty());

    if (show_image.empty())
        return;
    cv_image = show_image.clone();
    std::cout << "cv_image size: " << cv_image.size() << std::endl;

    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);
    Eigen::Vector3f livox_pt;

    // ROS_WARN(">>>>1<<<<, %d\n",cloud_msg->point_num);
    pcl::PointCloud<PointType>::Ptr colorCloud(new pcl::PointCloud<PointType>());
    for (auto it = cloud_msg->points.begin(); it != cloud_msg->points.end(); ++it)
    {
        auto livoxpt = *it;
        Eigen::Vector3f pt(livoxpt.x, livoxpt.y, livoxpt.z);
        Eigen::Vector3f livox_pt = transOffset * pt;
        X.at<double>(0, 0) = livox_pt(0);
        X.at<double>(1, 0) = livox_pt(1);
        X.at<double>(2, 0) = livox_pt(2);
        X.at<double>(3, 0) = 1;
        Y = intrisicMat * X; // 相机坐标投影到像素坐标
        cv::Point2f u_v;
        u_v.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        u_v.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        float dist, r, g, b;
        dist = pointDistance(pt);
        getColor(dist, 40, r, g, b);
        
     if (u_v.x < 0 || u_v.y < 0 || u_v.x > cv_image.cols || u_v.y > cv_image.rows || livoxpt.x < 0)
                continue;
        pcl::PointXYZRGBNormal p;
        p.x = livoxpt.x;
        p.y = livoxpt.y;
        p.z = livoxpt.z;
        p.r = cv_image.at<cv::Vec3b>(u_v.y, u_v.x)[0];
        p.g = cv_image.at<cv::Vec3b>(u_v.y, u_v.x)[1];
        p.b = cv_image.at<cv::Vec3b>(u_v.y, u_v.x)[2];

        colorCloud->points.push_back(p);
        cv::circle(show_image, cv::Point2f(u_v.x, u_v.y), 1, cv::Scalar(r, g, b), 5);
    }

    cv_bridge::CvImage bridge;
    bridge.image = show_image;
    bridge.encoding = "rgb8";
    sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
    imageShowPointer->header.stamp = cloud_msg->header.stamp;
    pub_depth_image.publish(imageShowPointer);

    colorCloud->width = colorCloud->points.size();
    colorCloud->height = 1;
    ROS_INFO(">>> %d", colorCloud->width);

    sensor_msgs::PointCloud2 color_msg;
    pcl::toROSMsg(*colorCloud, color_msg);
    color_msg.header.frame_id = "livox_frame";
    color_msg.header.stamp = cloud_msg->header.stamp;
    pubCloud.publish(color_msg);
    // colorCloud->clear();
    // depthCloud->clear();
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "ls_slam");
    ros::NodeHandle nh;
    ROS_INFO("\033[1;32m**** Mid360 to RangeImage Node Start. ****\033[0m\n");

    static int cnt = 0;
    if (cnt == 0)
    {
        CalibrationData();
        cnt++;
    }

    // 利用ROS的时间同步机制
    message_filters::Subscriber<sensor_msgs::Image> subImage;
    message_filters::Subscriber<livox_ros_driver2::CustomMsg> subLaserCloud;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, livox_ros_driver2::CustomMsg> syncPolicy; // 时间戳近似机制
    typedef message_filters::Synchronizer<syncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;
    subImage.subscribe(nh, "/camera/image", 1);
    subLaserCloud.subscribe(nh, "/livox/lidar", 1);
    sync_.reset(new Sync(syncPolicy(10), subImage, subLaserCloud));
    sync_->registerCallback(boost::bind(laserCloudAndImageHandler, _1, _2));

    // lidar_ = nh.subscribe("/livox/lidar",10000,laserCloudAndImageHandler);
    // lidar_ = nh.subscribe("/velodyne_points",5,VelodyneCallback,ros::TransportHints().tcpNoDelay());
    // image_ = nh.subscribe("/camera/image",10000,ImageCallback);

    pub_depth_image = nh.advertise<sensor_msgs::Image>("/fusion_image", 1, true);
    pubCloud = nh.advertise<sensor_msgs::PointCloud2>("/color_lidar", 1, true);

    ros::spin();
    // int sums, aver;
    // for (size_t i = 0; i < lidar_pts.size(); i++)
    // {
    //     sums += lidar_pts[i];
    // }
    // printf(">>> Average Frame points: %d\n",sums/lidar_count);

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
