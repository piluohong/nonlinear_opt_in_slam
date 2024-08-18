#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>


#include "net.h"
#include "mat.h"
#include "opencv2/opencv.hpp"
#include "chrono"
#include "tracking.h"

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

#define IMG_H 512
#define IMG_W 640

struct LivoxPoint {
    PCL_ADD_POINT4D;
    float intensity; // intensity
    std::uint32_t offset_time; // LIVOX: time from beginning of scan in nanoseconds
    int line;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  } EIGEN_ALIGN16;


POINT_CLOUD_REGISTER_POINT_STRUCT(LivoxPoint,
                                 (float, x, x)
                                 (float, y, y)
                                 (float, z, z)
                                 (float, intensity, intensity)
                                 (std::uint32_t, offset_time, offset_time)
                                 (int, line,line))

// struct PointXYZIRGBNormal : public pcl::PointXYZRGBNormal
// {
//     PCL_ADD_INTENSITY;

//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// } EIGEN_ALIGN16;
// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRGBNormal,
//                                     (float,x,x)
//                                     (float,y,y)
//                                     (float,z,z)
//                                     (float,intensity,intensity)
//                                     (float,normal_x,normal_x)
//                                     (float,normal_y,normal_y)
//                                     (float,normal_z,normal_z)
//                                     (std::uint8_t,r,r)
//                                     (std::uint8_t,g,g)
//                                     (std::uint8_t,b,b)
//                                     (float,curvature,curvature))




using PointType = pcl::PointXYZRGBNormal;
using namespace std;
ros::Subscriber lidar_;
ros::Subscriber image_;

ros::Publisher pub_depth_image;
ros::Publisher pub_fea_image;
ros::Publisher pubcolorCloud;
ros::Publisher pubfullCloud;
ros::Publisher pubfeaCloud;
ros::Publisher pubfeasize;

static int lidar_count = 1;
static int lidar_nums = 0;
std::vector<int> lidar_pts{10000};

//config params
// static int LiDAR_SKIP = 0;

std::mutex mtx_lidar;

deque<livox_ros_driver2::CustomMsgConstPtr> vec_;
deque<livox_ros_driver2::CustomMsgConstPtr> cloudQueue;
deque<ros::Time> timeQueue;
double laser_time;

pcl::PointCloud<PointType>::Ptr colorCloud (new pcl::PointCloud<PointType>());
pcl::PointCloud<pcl::PointXYZI>::Ptr depthCloud (new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr feaCloud (new pcl::PointCloud<pcl::PointXYZI>());


std::deque<std::pair<sensor_msgs::CompressedImageConstPtr, ros::Time>> image_buff;

sensor_msgs::CompressedImageConstPtr image_ros;
double image_time = 0.0;

static Eigen::Matrix4f extrinsicMat_RT; // 外参旋转矩阵3*3和平移向量3*1
static Eigen::Affine3f transOffset, C_to_L;
static cv::Mat intrisicMat(3, 4, cv::DataType<double>::type);// 内参3*4的投影矩阵，最后一列是三个零
static cv::Mat intrisic(3, 3, cv::DataType<double>::type);			   // 内参3*3矩阵
static cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);// 畸变向量

static Eigen::Matrix4f mat;

class ncnn_image
{
    public:
    
    ncnn_image() { 
        
         net.opt.num_threads=4;
         net.load_param("/home/hong/slam/lvio/src/ncnn_images/models/model.param");
         net.load_model("/home/hong/slam/lvio/src/ncnn_images/models/model.bin");
         
    }

    ncnn::Net net;
    corner_tracking tracker;
     

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    const float mean_vals_inv[3] = {0, 0, 0};
    const float norm_vals_inv[3] = {255.f, 255.f, 255.f};

   

    std::vector<cv::Point2f> ncnn_solve(cv::Mat& mat_in,pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud);
};

 std::vector<cv::Point2f> ncnn_image::ncnn_solve(cv::Mat& mat_in,pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud)
 {
    cv::Mat score(IMG_H, IMG_W, CV_8UC1);
    cv::Mat desc(IMG_H, IMG_W, CV_8UC3);
    ncnn::Mat in;
    ncnn::Mat out1, out2;

    
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        // ex.set_num_threads(4);

        // cv::resize(mat_in, mat_in, cv::Size(IMG_W, IMG_H));

        //////////////////////////  opencv image to ncnn mat  //////////////////////////
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        in = ncnn::Mat::from_pixels(mat_in.data, ncnn::Mat::PIXEL_BGR, mat_in.cols, mat_in.rows);
        in.substract_mean_normalize(mean_vals, norm_vals);

        //////////////////////////  ncnn forward  //////////////////////////

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        ex.input("input", in);
        ex.extract("score", out1);
        ex.extract("descriptor", out2);

        //////////////////////////  ncnn mat to opencv image  //////////////////////////

        std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
        out1.substract_mean_normalize(mean_vals_inv, norm_vals_inv);
        out2.substract_mean_normalize(mean_vals_inv, norm_vals_inv);

      //memcpy((uchar*)score.data, out1.data, sizeof(float) * out1.w * out1.h);
        out1.to_pixels(score.data, ncnn::Mat::PIXEL_GRAY);
        out2.to_pixels(desc.data, ncnn::Mat::PIXEL_BGR);

        std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
        //////////////////////////  show times  //////////////////////////
        std::chrono::duration<double> time_used_1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        std::chrono::duration<double> time_used_2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3-t2);
        std::chrono::duration<double> time_used_3 = std::chrono::duration_cast<std::chrono::duration<double>>(t4-t3);

        std::cout<<"time_used 1 : "<<time_used_1.count()*1000<<"ms"<<std::endl;
        std::cout<<"time_used 2 : "<<time_used_2.count()*1000<<"ms"<<std::endl;
        std::cout<<"time_used 3 : "<<time_used_3.count()*1000<<"ms"<<std::endl;

        // cv::Mat new_score  = score.clone();
        // cv::Mat new_desc  = desc.clone();
        auto fea_pts_vec =  tracker.extractFeature(score);
        std::cout << "fea_size : "<<fea_pts_vec.size() << std::endl;
        
        for (auto& p : fea_pts_vec)
        {
            // pcl::PointXYZI pt;
            
            cv::circle(mat_in, p, 2, cv::Scalar(0, 255, 0), 2);


            
        }
        cv_bridge::CvImage fea_bridge;
        fea_bridge.image = mat_in;
        fea_bridge.encoding = "rgb8";
        sensor_msgs::Image::Ptr feaimageShowPointer = fea_bridge.toImageMsg();
        feaimageShowPointer->header.stamp = ros::Time::now();
        pub_fea_image.publish(feaimageShowPointer);
       
        return fea_pts_vec;
 }

ncnn_image ncnn_;

void CalibrationData()
{
    //hku1.bag -> config lidar->cam
    // extrinsicMat_RT(0, 0) = 0.00162756;
	// extrinsicMat_RT(0, 1) = -0.999991;
	// extrinsicMat_RT(0, 2) = 0.00390957;
	// extrinsicMat_RT(0, 3) = 0.0409257;
	// extrinsicMat_RT(1, 0) = -0.0126748;
	// extrinsicMat_RT(1, 1) = -0.00392989;
	// extrinsicMat_RT(1, 2) = -0.999912;
	// extrinsicMat_RT(1, 3) = 0.0318424;
	// extrinsicMat_RT(2, 0) = 0.999918;
	// extrinsicMat_RT(2, 1) = 0.00157786;
	// extrinsicMat_RT(2, 2) = -0.012681;
	// extrinsicMat_RT(2, 3) = -0.0927219;
	// extrinsicMat_RT(3, 0) = 0.0;
	// extrinsicMat_RT(3, 1) = 0.0;
	// extrinsicMat_RT(3, 2) = 0.0;
	// extrinsicMat_RT(3, 3) = 1.0;

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

    // hku_main_building (r3live) -> config cam->lidar
  extrinsicMat_RT(0, 0) = -0.00113207;
  extrinsicMat_RT(0, 1) = -0.0158688;
  extrinsicMat_RT(0, 2) = 0.999873;
  extrinsicMat_RT(0, 3) = 0.050166;
  extrinsicMat_RT(1, 0) = -0.9999999;
  extrinsicMat_RT(1, 1) = -0.000486594;
  extrinsicMat_RT(1, 2) = -0.00113994;
  extrinsicMat_RT(1, 3) = 0.0474116;
  extrinsicMat_RT(2, 0) = 0.000504622;
  extrinsicMat_RT(2, 1) = -0.999874;
  extrinsicMat_RT(2, 2) = -0.0158682;
  extrinsicMat_RT(2, 3) = -0.0312415;
  extrinsicMat_RT(3, 0) = 0.0;
  extrinsicMat_RT(3, 1) = 0.0;
  extrinsicMat_RT(3, 2) = 0.0;
  extrinsicMat_RT(3, 3) = 1.0;

  intrisicMat.at<double>(0, 0) = intrisic.at<double>(0, 0) = 431.71205;//863.4241;
  intrisicMat.at<double>(0, 1) = intrisic.at<double>(0, 1) = 0.000000e+00;
  intrisicMat.at<double>(0, 2) = intrisic.at<double>(0, 2) = 320.3404;//640.6808;
  intrisicMat.at<double>(0, 3) = 0.000000e+00;
  intrisicMat.at<double>(1, 0) = intrisic.at<double>(1, 0) = 0.000000e+00;
  intrisicMat.at<double>(1, 1) = intrisic.at<double>(1, 1) = 431.70855;//863.4171;
  intrisicMat.at<double>(1, 2) = intrisic.at<double>(1, 2) = 259.1696;//518.3392;
  intrisicMat.at<double>(1, 3) = 0.000000e+00;
  intrisicMat.at<double>(2, 0) = intrisic.at<double>(2, 0) = 0.000000e+00;
  intrisicMat.at<double>(2, 1) = intrisic.at<double>(2, 1) = 0.000000e+00;
  intrisicMat.at<double>(2, 2) = intrisic.at<double>(2, 2) = 1.000000e+00;
  intrisicMat.at<double>(2, 3) = 0.000000e+00;

    // midd360_MER139 camera->lidar
    // extrinsicMat_RT(0, 0) = -0.00113207;
	// extrinsicMat_RT(0, 1) = -0.0158688;
	// extrinsicMat_RT(0, 2) = 0.999873;
	// extrinsicMat_RT(0, 3) = 0.00;//0.050166;
	// extrinsicMat_RT(1, 0) = -0.9999999;
	// extrinsicMat_RT(1, 1) = -0.000486594;
	// extrinsicMat_RT(1, 2) = -0.00113994;
	// extrinsicMat_RT(1, 3) = 0.00;//0.0474116;
	// extrinsicMat_RT(2, 0) = 0.000504622;
	// extrinsicMat_RT(2, 1) = -0.999874;
	// extrinsicMat_RT(2, 2) =  -0.0158682;
	// extrinsicMat_RT(2, 3) = -0.15;//-0.0312415;
	// extrinsicMat_RT(3, 0) = 0.0;
	// extrinsicMat_RT(3, 1) = 0.0;
	// extrinsicMat_RT(3, 2) = 0.0;
	// extrinsicMat_RT(3, 3) = 1.0;

    // intrisicMat.at<double>(0, 0) = intrisic.at<double>(0,0)=  3722.25879;
	// intrisicMat.at<double>(0, 1) = intrisic.at<double>(0,1)= 0.000000e+00;
	// intrisicMat.at<double>(0, 2) = intrisic.at<double>(0,2)=  675.1853;
	// intrisicMat.at<double>(0, 3) = 0.000000e+00;
	// intrisicMat.at<double>(1, 0) = intrisic.at<double>(1,0)= 0.000000e+00;
	// intrisicMat.at<double>(1, 1) = intrisic.at<double>(1,1)=  3646.6228;
	// intrisicMat.at<double>(1, 2) = intrisic.at<double>(1,2)= 30.56952;
	// intrisicMat.at<double>(1, 3) = 0.000000e+00;
	// intrisicMat.at<double>(2, 0) = intrisic.at<double>(2,0)= 0.000000e+00;
	// intrisicMat.at<double>(2, 1) = intrisic.at<double>(2,1)= 0.000000e+00;
	// intrisicMat.at<double>(2, 2) = intrisic.at<double>(2,2)= 1.000000e+00;
	// intrisicMat.at<double>(2, 3) = 0.000000e+00;
    
    // // std::cout << intrisic << std::endl;

    

    //绕y轴旋转180°
    // mat << -1,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,1;

    // Eigen::Matrix3f linearPart = mat.topLeftCorner<3, 3>();
    // Eigen::Vector3f translation = mat.topRightCorner<3, 1>();
    // transOffset.linear() = linearPart;
    // transOffset.translation() = translation;

     //mid360_455 camera->lidar 640x480
    // extrinsicMat_RT(0, 0) = -0.00113207;
	// extrinsicMat_RT(0, 1) = -0.0158688;
	// extrinsicMat_RT(0, 2) = 0.999873;
	// extrinsicMat_RT(0, 3) = -0.055;//0.050166;
	// extrinsicMat_RT(1, 0) = -0.9999999;
	// extrinsicMat_RT(1, 1) = -0.000486594;
	// extrinsicMat_RT(1, 2) = -0.00113994;
	// extrinsicMat_RT(1, 3) = -0.0242;//0.0474116;
	// extrinsicMat_RT(2, 0) = 0.000504622;
	// extrinsicMat_RT(2, 1) = -0.999874;
	// extrinsicMat_RT(2, 2) =  -0.0158682;
	// extrinsicMat_RT(2, 3) = -0.0322;//-0.0312415;
	// extrinsicMat_RT(3, 0) = 0.0;
	// extrinsicMat_RT(3, 1) = 0.0;
	// extrinsicMat_RT(3, 2) = 0.0;
	// extrinsicMat_RT(3, 3) = 1.0;

    // intrisicMat.at<double>(0, 0) = intrisic.at<double>(0,0)=  381.73895263671875;
	// intrisicMat.at<double>(0, 1) = intrisic.at<double>(0,1)= 0.000000e+00;
	// intrisicMat.at<double>(0, 2) = intrisic.at<double>(0,2)=  314.430419921875;
	// intrisicMat.at<double>(0, 3) = 0.000000e+00;
	// intrisicMat.at<double>(1, 0) = intrisic.at<double>(1,0)= 0.000000e+00;
	// intrisicMat.at<double>(1, 1) = intrisic.at<double>(1,1)=  381.2456970214844;
	// intrisicMat.at<double>(1, 2) = intrisic.at<double>(1,2)= 241.4613494873047;
	// intrisicMat.at<double>(1, 3) = 0.000000e+00;
	// intrisicMat.at<double>(2, 0) = intrisic.at<double>(2,0)= 0.000000e+00;
	// intrisicMat.at<double>(2, 1) = intrisic.at<double>(2,1)= 0.000000e+00;
	// intrisicMat.at<double>(2, 2) = intrisic.at<double>(2,2)= 1.000000e+00;
	// intrisicMat.at<double>(2, 3) = 0.000000e+00;

    //k1,k2,p1,p2,k3
    distCoeffs.at<double>(0) = -0.056032437831163406;//-0.1080;
    distCoeffs.at<double>(1) =  0.0686575248837471;//0.1050;
    distCoeffs.at<double>(2) = -0.0011853401083499193;//-1.2872e-04;
    distCoeffs.at<double>(3) =  0.00010860788461286575;//5.7923e-05;
    distCoeffs.at<double>(4) =  -0.021965153515338898;//-0.0222;

    //转换lidar -> cam（lidar和相机外参）
    Eigen::Matrix3f linearPart = extrinsicMat_RT.topLeftCorner<3, 3>();
    Eigen::Vector3f translation = extrinsicMat_RT.topRightCorner<3, 1>();

    transOffset.linear() = linearPart;
    transOffset.translation() = translation;

    C_to_L.linear() = linearPart;
    C_to_L.translation() = translation;
    
    transOffset = transOffset.inverse();

}

void LivoxMsgToPcl(livox_ros_driver2::CustomMsgConstPtr &cloud_, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_out)
{
    // auto msg_ = cloud_;
 #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < cloud_->point_num; i++)
    {
  
        pcl::PointXYZI pt;
        pt.x = cloud_->points[i].x;
        pt.y = cloud_->points[i].y;
        pt.z = cloud_->points[i].z;
        pt.intensity = cloud_->points[i].reflectivity;
        // pt.line = msg_->points[i].line;
        // pt.offset_time = msg_->points[i].offset_time;
        cloud_out->push_back(pt);
    }
    
    // ROS_INFO("Completed LIVOX Msg convert to Livox(pcl) points.\n");
    
    return;
}

float pointDistance(Eigen::Vector3f p)
{
    return sqrt(p(0)*p(0) + p(1)*p(1) + p(2)*p(2));
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

void ros2cv(sensor_msgs::CompressedImageConstPtr &image_ros,cv::Mat &image,cv::Mat &intrisic,cv::Mat &distCoeffs,bool corr)
{
    // ROS_Image -> Openshow_image
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_ros, sensor_msgs::image_encodings::RGB8);
    image = cv_ptr->image;

    if (image.cols == 640 || image.cols == 1280)
    {
      cv::resize(image, image, cv::Size(640, 512), 0, 0, cv::INTER_LINEAR);
    }

    if (corr)
    {
      /* code */
      // 图像去畸变
      cv::Mat map1, map2;
      cv::Size imageSize = image.size();
      cv::initUndistortRectifyMap(intrisic, distCoeffs, cv::Mat(), cv::getOptimalNewCameraMatrix(intrisic, distCoeffs, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);
      cv::remap(image, image, map1, map2, cv::INTER_LINEAR); // correct the distortion

    }
    
    // return;
}

void ImageCallback(const sensor_msgs::CompressedImageConstPtr &image_msg)
{
     // 相机数据打包
    std::lock_guard<std::mutex> lock(mtx_lidar);
    ros::Time image_time = image_msg->header.stamp;
    image_buff.push_back(std::make_pair(image_msg, image_time));
    
    
        // ROS_WARN("*******\n");

    // cv::addWeighted(show_image, 1.0,circle_image, 0.5, 0, show_image);
    

    // cv::imwrite("/home/hong/slam/hhh_ws/src/nonlinear_opt_in_slam/data/saveRangeImageRGB.png",circle_image);
    // ROS_INFO("*****************************************\n");
}

template <typename T>
inline bool HasInf(const T& p) {
  return (std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z));
}

template <typename T>
inline bool HasNan(const T& p) {
  return (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z));
}

template <typename T>
inline bool IsNear(const T& p1, const T& p2) {
  return ((abs(p1.x - p2.x) < 1e-7) || (abs(p1.y - p2.y) < 1e-7) ||
          (abs(p1.z - p2.z) < 1e-7));
}



void LivoxCallback(const livox_ros_driver2::CustomMsgConstPtr& cloud_msg)
{
    ros::Time timeScanCur = cloud_msg->header.stamp;
    cloudQueue.push_back(cloud_msg);
    timeQueue.push_back(timeScanCur); 
 
    if((timeScanCur - timeQueue.front()).toSec() > 0.2)
    {  
        timeQueue.pop_front();
        cloudQueue.pop_front();
        
    }else{
        return;
    }

    cv::Mat show_image, cv_image,fea_image;
    mtx_lidar.lock();
    image_ros = image_buff.back().first;
    image_buff.pop_front();
    mtx_lidar.unlock();

    // show_image = cv::Mat::zeros(720, 1280, CV_8UC3);
    ros2cv(image_ros,show_image,intrisic,distCoeffs,false);
    // std::cout << show_image.size() << std::endl;
    if(show_image.empty()){
        return;
    }

    fea_image = show_image.clone();
    cv_image = show_image.clone();
    std::vector<cv::Point2f> fea_pts;
    fea_pts = ncnn_.ncnn_solve(fea_image,feaCloud);
    //pub feaCloud

    // std::cout << "show_image size: " << show_image.size() << std::endl;

    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);
    Eigen::Vector3f cam_pt;
  
    // ROS_WARN(">>>>1<<<<, %d\n",cloud_msg->point_num);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(6)
    for (size_t n = 0; n < cloudQueue.size(); ++n) {
        for (size_t i = 0; i < cloudQueue[n]->point_num; ++i) {
            if ((i % 3 == 0 && cloudQueue[n]->points[i].line < 6))
            {

            Eigen::Vector3f pt(cloudQueue[n]->points[i].x,cloudQueue[n]->points[i].y,cloudQueue[n]->points[i].z);
            cam_pt = transOffset * pt;
            
            cv::Mat X(4, 1, CV_64F);
            X.at<double>(0, 0) = cam_pt(0);
            X.at<double>(1, 0) = cam_pt(1);
            X.at<double>(2, 0) = cam_pt(2);
            X.at<double>(3, 0) = 1;

            cv::Mat Y = intrisicMat * X;
            cv::Point2f u_v;
            u_v.x = std::floor(Y.at<double>(0, 0) / Y.at<double>(2, 0));
            u_v.y = std::floor(Y.at<double>(1, 0) / Y.at<double>(2, 0));

            if (pt(0) < 0 || u_v.x < 0 || u_v.y < 0 || u_v.x > cv_image.cols || u_v.y > cv_image.rows)
                continue;
            
            auto it = std::find(fea_pts.begin(), fea_pts.end(),u_v);
            
            if(it != fea_pts.end()){
                #pragma omp critical
                {
                    pcl::PointXYZI fea_p;
                    fea_p.x = pt(0);
                    fea_p.y = pt(1);
                    fea_p.z = pt(2);
                    fea_p.intensity = 255;
                    feaCloud->points.push_back(fea_p);
                }
            }
        
            float dist = pointDistance(pt);
            float r, g, b;
            getColor(dist, 100, r, g, b);

            // 临界区保护，防止对 colorCloud 进行并发操作
            #pragma omp critical
            {
                pcl::PointXYZI depth_pt;
                depth_pt.x = pt(0);
                depth_pt.y = pt(1);
                depth_pt.z = pt(2);
                depth_pt.intensity = cloudQueue[n]->points[i].reflectivity;
                depthCloud->points.push_back(depth_pt);

                pcl::PointXYZRGBNormal p;
                p.x = pt(0);
                p.y = pt(1);
                p.z = pt(2);
                p.r = show_image.at<cv::Vec3b>(u_v.y, u_v.x)[0];
                p.g = show_image.at<cv::Vec3b>(u_v.y, u_v.x)[1];
                p.b = show_image.at<cv::Vec3b>(u_v.y, u_v.x)[2];
                
                colorCloud->points.push_back(p);
                cv::circle(cv_image, u_v, 2, cv::Scalar(r, g, b), 3);
            }

        }
     }
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_used_1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    std::cout<<"time_used : "<<time_used_1.count()*1000<<"ms"<<std::endl;
     
    cv_bridge::CvImage bridge;
    cv_bridge::CvImage fea_bridge;
    bridge.image = cv_image;
    fea_bridge.image = fea_image;
    bridge.encoding = "rgb8";
    fea_bridge.encoding = "rgb8";
    sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
    sensor_msgs::Image::Ptr feaimageShowPointer = fea_bridge.toImageMsg();
    imageShowPointer->header.stamp = cloud_msg->header.stamp;
    feaimageShowPointer->header.stamp = cloud_msg->header.stamp;
    pub_depth_image.publish(imageShowPointer);
    pub_fea_image.publish(feaimageShowPointer);

    colorCloud->width = colorCloud->points.size();
    colorCloud->height = 1;
    feaCloud->width = feaCloud->points.size();
    feaCloud->height = 1;
    printf(">>> Color cloud size : %d\n",colorCloud->width);
    printf(">>> Fea cloud size : %d\n",feaCloud->width);

    std_msgs::Float32 fea_size_msg;
    fea_size_msg.data = feaCloud->width;
    pubfeasize.publish(fea_size_msg);
    
    sensor_msgs::PointCloud2 color_msg, raw_msg, fea_msg;
    pcl::toROSMsg(*depthCloud, raw_msg);	
    pcl::toROSMsg(*colorCloud, color_msg);
    pcl::toROSMsg(*feaCloud, fea_msg);
    raw_msg.header.frame_id = "livox_frame";				   
    color_msg.header.frame_id = "livox_frame";
    fea_msg.header.frame_id = "livox_frame";
    raw_msg.header.stamp = cloud_msg->header.stamp;		   
    color_msg.header.stamp = cloud_msg->header.stamp;
    fea_msg.header.stamp = cloud_msg->header.stamp;
    pubfullCloud.publish(raw_msg);
    pubcolorCloud.publish(color_msg);
    pubfeaCloud.publish(fea_msg);

    colorCloud->clear();
    depthCloud->clear();
    feaCloud->clear();

}

int main(int argc,char **argv)
{
    setlocale(LC_ALL, "");
    ros::init(argc,argv,"ls_slam");
    ros::NodeHandle nh;
    ROS_INFO("\033[1;32m**** Mid360 to RangeImage Node Start. ****\033[0m\n");
    static int cnt = 0;
    if(cnt == 0)
    {
        CalibrationData();
        cnt++;
    }

    // lidar_ = nh.subscribe("/velodyne_points",5,VelodyneCallback,ros::TransportHints().tcpNoDelay());
    image_ = nh.subscribe<sensor_msgs::CompressedImage>("/camera/image_color/compressed",1000,ImageCallback);
    lidar_ = nh.subscribe("/livox/lidar",1000,LivoxCallback);

    pub_depth_image  =  nh.advertise<sensor_msgs::Image>("/fusion_image",1,true);
    pub_fea_image   =  nh.advertise<sensor_msgs::Image>("/fea_image",1,true);
    pubcolorCloud  =  nh.advertise<sensor_msgs::PointCloud2>("/color_lidar", 1,true);
    pubfullCloud  =  nh.advertise<sensor_msgs::PointCloud2>("/raw_lidar", 1,true);
    pubfeaCloud  =  nh.advertise<sensor_msgs::PointCloud2>("/fea_lidar", 1,true);
    pubfeasize  =  nh.advertise<std_msgs::Float32>("/fea_size", 1,true);		

    
    ros::spin();
    std::cout << "image_buff size: " << image_buff.size() << std::endl;
    image_buff.clear();
    return 0;

}


// int sums, aver;
    // for (size_t i = 0; i < lidar_pts.size(); i++)
    // {
    //     sums += lidar_pts[i];
    // }
    // printf(">>> Average Frame points: %d\n",sums/lidar_count);

// void VelodyneCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg){
    // pcl::PointCloud<pcl::PointXYZ> msg;
    // pcl::fromROSMsg(*cloud_msg, msg);
    // int available_nums = 0;
    // for (size_t i = 0; i < msg.size(); i++)
    // {
    //      available_nums += 1;
    // }

    // lidar_nums = available_nums;
    // lidar_pts.push_back(lidar_nums);
    // std::cout << ">>> lidar_count: "<< lidar_count << ", Current Frame points: " << lidar_nums << std::endl;
    // lidar_count++;

//     static int lidar_count = -1;
//     //间隔处理
//     if (++lidar_count % (LiDAR_SKIP + 1) != 0)
//         return;
    
//     //订阅 lidar -> camera 的静态tf树（外参）
//     static tf::TransformListener listener;
//     static tf::StampedTransform transform;
//     try
//     {
        
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
//     for (int i = 0; i < (int)cloudQueue.size(); ++i)
//         *depthCloud += cloudQueue[i];
   
//     pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
//     downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
//     downSizeFilter.setInputCloud(depthCloud);
//     downSizeFilter.filter(*depthCloudDS);
//     *depthCloud = *depthCloudDS;
//     sensor_msgs::PointCloud2 raw_msg;
//     raw_msg = *msg;
//     raw_msg.header.stamp = ros::Time::now();
//     raw_msg.header.frame_id = "laser_link";
//     pub_.publish(raw_msg);

//     return;
// }

// void LivoxCallback_(const livox_ros_driver2::CustomMsg::ConstPtr& cloud_msg){
    
    //筛选有效点
    // int avaliabe_nums = 0;
    // for(size_t i = 0;i < cloud_msg->point_num;i++)
    // {
    //     // if(cloud_msg->points[i].line < 3 && ((cloud_msg->points[i].tag & 0x30) == 0x10 ||
    //     // (cloud_msg->points[i].tag & 0x30) == 0x00) &&
    //     //     !HasInf(cloud_msg->points[i]) && !HasNan(cloud_msg->points[i]) &&
    //     //     !IsNear(cloud_msg->points[i],cloud_msg->points[i - 1])){
    //     //     avaliabe_nums += 1;
    //     // }
    // }
    // auto msg = cloud_msg;
    // pcl::PointCloud<LivoxPoint>::Ptr cloud_(new pcl::PointCloud<LivoxPoint>());
   
    // LivoxMsgToPcl(cloud_msg,cloud_);
    // //盒子滤波
    // pcl::CropBox<LivoxPoint> crop;
    // crop.setNegative(true);
    // crop.setMin(Eigen::Vector4f(-0.6, -0.6, -0.6, 1.0));
    // crop.setMax(Eigen::Vector4f(0.6, 0.6, 0.6, 1.0));
    // crop.setInputCloud(cloud_);
    // crop.filter(*cloud_);
    // pcl::transformPointCloud(*cloud_,*cloud_,transOffset);

    
    // for(auto& pt : cloud_->points)
    // {
    //     pt.z = pt.z +3;
    // }
    //  sensor_msgs::PointCloud2 raw_msg;
    //  pcl::toROSMsg(*cloud_,raw_msg);
    // raw_msg.header.stamp =  msg->header.stamp;
    // raw_msg.header.frame_id ="livox_frame";
    
    
    // pub_.publish(raw_msg);

    // lidar_nums = avaliabe_nums;
    // lidar_pts.push_back(lidar_nums);
    // std::cout << ">>> lidar_count: "<< lidar_count << ", Current Frame points: " << lidar_nums << std::endl;
    // lidar_count++;
// }
