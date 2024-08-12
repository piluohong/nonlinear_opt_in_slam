
#include <iostream>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>

#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
// #include<g2o/solvers/csparse/linear_solver_csparse.h>

#include <g2o/types/slam2d/types_slam2d.h>

#include "readfile.h"
using namespace g2o;

// 每个误差项优化变量维度为3，误差值维度为3
typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>> Block;

//  发布rviz可视化的Marker话题
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
/*
    @brief g2o通用优化流程;
    **通用使用流程**

    1）创建一个线性求解器LinearSolver

    2）创建BlockSolver。用上面定义的线性求解器初始化。----> 构建 H b

    3）创建总求解器solver。从GN, LM, DogLeg 中选一个，再用上述块求解器BlockSolver初始化。

    4）创建SparseOptimizer 稀疏优化器。

    5）定义图的顶点和边。

    6）设置优化参数，开始执行优化。
*/

int main(int argc, char *argv[])
{
    /* code */
    ros::init(argc, argv, "ls_slam");
    ros::NodeHandle nh;
    //  1.话题发布者
    ros::Publisher before_Graph_pub;
    ros::Publisher after_Graph_pub;
    before_Graph_pub = nh.advertise<visualization_msgs::MarkerArray>("beforePoseGraph", 1, true);
    after_Graph_pub = nh.advertise<visualization_msgs::MarkerArray>("afterPoseGraph", 1, true);
    //  2.读取数据，将顶点、边的信息存到Vertexs、Edges中
    std::vector<Eigen::Vector3d> Vertexs;
    std::vector<Edge> Edges;

    std::string vertexpath = "/home/hhh/project_hhh/temp/slam/nonlinear_opt/src/ls_slam/data/killian-v.dat";
    std::string Edgepath = "/home/hhh/project_hhh/temp/slam/nonlinear_opt/src/ls_slam/data/killian-e.dat";
    ReadVertexInformation(vertexpath, Vertexs); //  读取数据，将顶点信息存到Vertexs中
    ReadEdgesInformation(Edgepath, Edges);      //  读取数据，将边的信息存到Edges中

    Publish_Graph_for_Visulization(&before_Graph_pub,Vertexs,Edges,0);

    //  3.使用g2o进行图优化求解
    //  3.1 创建线性求解器
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>();
    //  3.2 创建块求解器
    Block *solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));
    // Block *solver_ptr = new Block(linearSolver);
    //  3.3 创建总求解器
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));
    // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    //  3.4 创建稀疏优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver); // 
    optimizer.setVerbose(true);

    //  3.5 添加顶点
    for (size_t i = 0; i < Vertexs.size(); i++)
    {
        /* code */
        g2o::VertexSE2 *v = new g2o::VertexSE2();
        v->setEstimate(Vertexs[i]);
        v->setId(i);
        if (0 == i)
        {
            /* code */
            v->setFixed(true); //  第一个顶点设为固定点
        }
        optimizer.addVertex(v);
    }
    
    //  3.6 添加边
    
    for (size_t i = 0; i < Edges.size(); i++)
    {
        /* code */
        g2o::EdgeSE2 *edge(new g2o::EdgeSE2());
        Edge tmp_Edge = Edges[i];
        
        edge->setId(i);
        edge->setVertex(0, optimizer.vertices()[tmp_Edge.xi]); //  边的第1个顶点
        edge->setVertex(1, optimizer.vertices()[tmp_Edge.xj]); //  边的第2个顶点
        edge->setMeasurement(tmp_Edge.measurement);            //  测量值
        edge->setInformation(tmp_Edge.infoMatrix);             //  信息矩阵
        optimizer.addEdge(edge);
    }

    //  3.7 设置优化参数，并求解
    optimizer.initializeOptimization(); //  初始化
    //  用于优化过程中的控制
    g2o::SparseOptimizerTerminateAction *terminateAction = new g2o::SparseOptimizerTerminateAction;
    terminateAction->setGainThreshold(1e-6);           //  设置收敛阈值为1e-4，即当优化器中的误差变化小于该值时，优化过程将停止
    optimizer.addPostIterationAction(terminateAction); //  在每次迭代后检查是否已达到收敛条件
    std::cout<<"*******************"<<std::endl;  //  @@@@@@@@@@@@@@@@@@@@@@@@
    optimizer.optimize(100);
    printf(">>>> finish solve !\n");
    optimizer.save("/home/hhh/project_hhh/temp/slam/nonlinear_opt/src/ls_slam/data/optimized_poses.g2o");

    //  取出求解结果
    for (size_t i = 0; i < Vertexs.size(); i++)
    {
        /* code */
        g2o::VertexSE2 *v = static_cast<g2o::VertexSE2 *>(optimizer.vertices()[i]);
        Vertexs[i] = v->estimate().toVector();
    }
    Publish_Graph_for_Visulization(&after_Graph_pub,Vertexs,Edges,1);

    ros::spin();
    return 0;

}