/*
 * @Author: piluohong 1912694135@qq.com
 * @Date: 2024-01-29 10:27:52
 * @LastEditors: piluohong 1912694135@qq.com
 * @LastEditTime: 2024-01-29 17:49:49
 * @FilePath: /slam/hhh_ws/src/ls_slam/src/main_ceres.cpp
 * @Description:Ceres 用于slam
 */

#include <iostream>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include "readfile.h"

#include <ceres/ceres.h>
#include <Eigen/Dense>

using namespace Eigen;
// using namespace ceres;

void Publish_Graph_for_Visulization(ros::Publisher *pub,
                                    std::vector<Eigen::Vector3d> &Vertexs, //  存顶点向量
                                    std::vector<Edge> &Edges,              //  存边的向量
                                    int color = 0)
{
    //  创建MarkerArray消息，用于发布
    visualization_msgs::MarkerArray m_array;
    //  point --red
    visualization_msgs::Marker m; //  创建一个maker, 用来画顶点
    {
        //  赋值顶点的maker的基本消息，初始化位置为0
        m.header.frame_id = "map";                   //  参考系
        m.header.stamp = ros::Time::now();           //  时间戳
        m.id = 0;                                    //  标识
        m.ns = "ls_slam_node";                       //  命名空间，      用于分组
        m.type = visualization_msgs::Marker::SPHERE; //  形状==球形
        m.pose.position.x = 0.0;
        m.pose.position.y = 0.0;
        m.pose.position.z = 0.0;
        m.scale.x = 0.2;
        m.scale.y = 0.2;
        m.scale.z = 0.2;
        m.action = visualization_msgs::Marker::ADD;

        if (0 == color)
        {
            m.color.r = 1.0;
            m.color.g = 0.0;
            m.color.b = 0.0;
        }
        else
        {
            m.color.r = 0.0;
            m.color.g = 1.0;
            m.color.b = 0.0;
        }

        m.color.a = 1.0;
        m.lifetime = ros::Duration(0);
    }
    //  linear --blue
    visualization_msgs::Marker edge;
    {
        edge.header.frame_id = "map";
        edge.header.stamp = ros::Time::now();
        edge.action = visualization_msgs::Marker::ADD;
        edge.ns = "karto";
        edge.id = 0;
        edge.type = visualization_msgs::Marker::LINE_STRIP;
        edge.scale.x = 0.1;
        edge.scale.y = 0.1;
        edge.scale.z = 0.1;

        if (color == 0)
        {
            edge.color.r = 0.0;
            edge.color.g = 0.0;
            edge.color.b = 1.0;
        }
        else
        {
            edge.color.r = 0.0;
            edge.color.g = 1.0;
            edge.color.b = 1.0;
        }
        edge.color.a = 1.0;
    }
    uint id = 0;
    //  加入节点
    for (size_t i = 0; i < Vertexs.size(); i++)
    {
        /* code */
        m.id = id;
        m.pose.position.x = Vertexs[i](0);
        m.pose.position.y = Vertexs[i](1);
        m_array.markers.push_back(visualization_msgs::Marker(m));
        id++;
    }
    //  加入边
    for (size_t i = 0; i < Edges.size(); i++)
    {
        /* code */
        edge.points.clear(); //  清空消息中线条点列表

        geometry_msgs::Point pi;
        pi.x = Vertexs[Edges[i].xi](0);
        pi.y = Vertexs[Edges[i].xi](1);
        edge.points.push_back(pi);

        geometry_msgs::Point pj;
        pj.x = Vertexs[Edges[i].xj](0);
        pj.y = Vertexs[Edges[i].xj](1);
        edge.points.push_back(pj);

        edge.id = id;
        m_array.markers.push_back(visualization_msgs::Marker(edge));
        id++;
    }

    pub->publish(m_array);
}

struct point3f
{
    double x;
    double y;
    double yaw;
};

struct RelativePoseResidual {
    RelativePoseResidual(double dx, double dy,double dtheta) : dx_(dx), dy_(dy),dtheta_(dtheta) {}

    template <typename T>
    bool operator()(const T* const point1, const T* const point2, T* residual) const {

        // T rotated_x = cos(dtheta_) * (point1[0] - point2[0]) - sin(dtheta_) * (point1[1] - point2[1]) + point2[0] + T(dx_);
        // T rotated_y = sin(dtheta_) * (point1[0] - point2[0]) + cos(dtheta_) * (point1[1] - point2[1]) + point2[1] + T(dy_);

        // Calculate residuals
        residual[0] = point1[0] - point2[0] - T(dx_);
        residual[1] = point1[1] - point2[1] - T(dy_);
        residual[2] = point1[2] - point2[2] - T(dtheta_);
        
        return true;
    }

private:
    const double dx_;
    const double dy_;
    const double dtheta_;
};

int main(int argc,char **argv)
{

    ros::init(argc,argv,"ls_slam");
    ros::NodeHandle nh;

    ros::Publisher before_Graph_pub,after_Graph_pub;

    before_Graph_pub = nh.advertise<visualization_msgs::MarkerArray>("beforePoseGraph", 1, true);
    after_Graph_pub = nh.advertise<visualization_msgs::MarkerArray>("afterPoseGraph", 1, true);
    
    //用Eigen的顶点和边类进行写入和读取数据
    std::vector<Eigen::Vector3d> Vertexs;
    std::vector<Edge> Edges;

    std::string vertexpath = "/home/hhh/project_hhh/temp/slam/hhh_ws/src/ls_slam/data/intel-v.dat";
    std::string Edgepath = "/home/hhh/project_hhh/temp/slam/hhh_ws/src/ls_slam/data/intel-e.dat";

    ReadVertexInformation(vertexpath,Vertexs);
    ReadEdgesInformation(Edgepath,Edges);

    // Publish_Graph_for_Visulization(&before_Graph_pub,Vertexs,Edges,0);

    ceres::Problem problem;
    //提取节点和边
    auto vex_0 = Vertexs[5];
    auto vex_1 = Vertexs[6];
    auto vex_2 = Vertexs[7];
    auto edge_7_6 = Edges[7];
    auto edge_7_5 = Edges[8];
   
    std::cout << "node5 [ "<< vex_0(0) << " " << vex_0(1) << " " << vex_0(2) << " ]\n";
    std::cout << "node6 [ "<< vex_1(0) << " " << vex_1(1) << " " << vex_1(2) << " ]\n";
    std::cout << "node7 [ "<< vex_2(0) << " " << vex_2(1) << " " << vex_2(2) << " ]\n"; 

    std::cout << "edge76 [ "<< edge_7_6.measurement(0) << " " << edge_7_6.measurement(1) << " " << edge_7_6.measurement(2) << " ]\n";
    std::cout << "edge75 [ "<< edge_7_5.measurement(0) << " " << edge_7_5.measurement(1) << " " << edge_7_6.measurement(2) << " ]\n";



    double pose_last_0[3] = {vex_0(0),vex_0(1),vex_0(2)}; //node5
    double pose_last_1[3] = {vex_1(0),vex_1(1),vex_1(2)}; //node6
    
    double pose_cur[3] = {vex_2(0),vex_2(1),vex_2(2)};
    
    problem.AddParameterBlock(pose_cur,3);
    problem.AddParameterBlock(pose_last_0,3); //node5
    problem.AddParameterBlock(pose_last_1,3); //node6


    //添加残差块：当前点与上一点的相对位姿变换
    RelativePoseResidual* residual_75 = new RelativePoseResidual(edge_7_5.measurement(0), edge_7_5.measurement(1),edge_7_5.measurement(2));
    RelativePoseResidual* residual_76 = new RelativePoseResidual(edge_7_6.measurement(0), edge_7_6.measurement(1),edge_7_6.measurement(2));
    
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<RelativePoseResidual, 3, 3, 3>
                            (residual_75), nullptr, pose_cur, pose_last_0);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<RelativePoseResidual, 3, 3, 3>
                            (residual_76), nullptr, pose_cur, pose_last_1);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;         // 设置最大迭代次数
    options.function_tolerance = 1e-6;        // 设置函数值容差
    options.gradient_tolerance = 1e-6                                                                   ;        // 设置梯度容差
    options.parameter_tolerance = 1e-8;       // 设置参数容差
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出优化后的结果
    std::cout << "Optimized current pose: [" << pose_cur[0] << ", " << pose_cur[1] << " " << pose_cur[2] << "]" << std::endl;
    std::cout << "Optimized node5 pose: [" << pose_last_0[0] << ", " << pose_last_0[1] << " " << pose_last_0[2] << "]" << std::endl;
    std::cout << "Optimized node6 pose: [" << pose_last_1[0] << ", " << pose_last_1[1] << " " << pose_last_0[2] << "]" << std::endl;
    
    ros::spin();

    return 0;
}
