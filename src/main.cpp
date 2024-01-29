#include <gaussian_newton.h>
#include <readfile.h>
#include <thread>
#include <chrono>


void PublishGraphForVisulization(ros::Publisher* pub,
                                 std::vector<Eigen::Vector3d>& Vertexs, //  存顶点向量
                                 std::vector<Edge>& Edges,              //  存边的向量
                                 int color = 0){
    visualization_msgs::MarkerArray marray; //声明要发布的 MarkerArray消息

    //point--red
    visualization_msgs::Marker m;   //声明一个maker  用来画出顶点
    //  赋值顶点的maker的基本消息，初始化位置为 0
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.id = 0;
    m.ns = "ls-slam";
    m.type = visualization_msgs::Marker::SPHERE;
    m.pose.position.x = 0.0;
    m.pose.position.y = 0.0;
    m.pose.position.z = 0.0;
    m.scale.x = 0.1;
    m.scale.y = 0.1;
    m.scale.z = 0.1;

    if(color == 0)
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

    //linear--blue
    visualization_msgs::Marker edge;
    edge.header.frame_id = "map";
    edge.header.stamp = ros::Time::now();
    edge.action = visualization_msgs::Marker::ADD;
    edge.ns = "karto";
    edge.id = 0;
    edge.type = visualization_msgs::Marker::LINE_STRIP;
    edge.scale.x = 0.1;
    edge.scale.y = 0.1;
    edge.scale.z = 0.1;

    if(color == 0)
    {
        edge.color.r = 0.0;
        edge.color.g = 0.0;
        edge.color.b = 1.0;
    }
    else
    {
        edge.color.r = 1.0;
        edge.color.g = 0.0;
        edge.color.b = 1.0;
    }
    edge.color.a = 1.0;

    m.action = visualization_msgs::Marker::ADD;
    uint id = 0;

    //加入节点
    for (uint i=0; i<Vertexs.size(); i++)   //  遍历每个顶点，将位置赋值
    {
        m.id = id;
        m.pose.position.x = Vertexs[i](0);
        m.pose.position.y = Vertexs[i](1);
        marray.markers.push_back(visualization_msgs::Marker(m));
        id++;
    }

    //加入边
    for(int i = 0; i < Edges.size();i++)    //  遍历每个边，将位置赋值
    {
        Edge tmpEdge = Edges[i];
        edge.points.clear();

        geometry_msgs::Point p;
        p.x = Vertexs[tmpEdge.xi](0);
        p.y = Vertexs[tmpEdge.xi](1);
        edge.points.push_back(p);

        p.x = Vertexs[tmpEdge.xj](0);
        p.y = Vertexs[tmpEdge.xj](1);
        edge.points.push_back(p);
        edge.id = id;

        marray.markers.push_back(visualization_msgs::Marker(edge));
        id++;
    }

    pub->publish(marray);
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "ls_Slam");

    ros::NodeHandle nodeHandle;

    // beforeGraph
    ros::Publisher beforeGraphPub,afterGraphPub;
    beforeGraphPub = nodeHandle.advertise<visualization_msgs::MarkerArray>("beforePoseGraph",1,true);
    afterGraphPub  = nodeHandle.advertise<visualization_msgs::MarkerArray>("afterPoseGraph",1,true);


    std::string VertexPath = "/home/hhh/project_hhh/temp/slam/nonlinear_opt/src/ls_slam/data/intel-v.dat";
    std::string EdgePath = "/home/hhh/project_hhh/temp/slam/nonlinear_opt/src/ls_slam/data/intel-e.dat";

    std::vector<Eigen::Vector3d> Vertexs;
    std::vector<Edge> Edges;

    ReadVertexInformation(VertexPath,Vertexs);  //  读取数据，将顶点信息存到Vertexs中
    ReadEdgesInformation(EdgePath,Edges);       //  读取数据，将边的信息存到Edges中
    
    
    //  发布优化前的顶点-边的信息
    PublishGraphForVisulization(&beforeGraphPub,
                                Vertexs,
                                Edges);
    
    double initError = ComputeError(Vertexs,Edges);
    std::cout <<"initError:"<<initError<<std::endl; //  打印误差的最初值
    
    int maxIteration = 64; //  最大迭代次数
    double epsilon = 1e-3;  //  精度要求阈值

    auto t1 = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(16);
    #pragma omp parallel for num_threads(6) 
    for(int i = 0; i < maxIteration; i++)
    {
        std::cout <<"Iterations:"<<i<<std::endl;    //  输出迭代次数
        Eigen::VectorXd dx = LinearizeAndSolve(Vertexs,Edges);  //  一次的迭代求解：线性化与求解
        std::cout<<"*****"<<std::endl;

        //进行更新
        //TODO--Start
        //  进行所有的位姿(x,y,yaw)更新 将上面求解的dx叠加到x上
        for (size_t j = 0; j < Vertexs.size(); j++)
        {
            /* code */
            //  更新x
            Vertexs[j](0) += dx(j*3);
            Vertexs[j](1) += dx(j*3+1);
            Vertexs[j](2) += dx(j*3+2);

             //限制角度 
            if (Vertexs[j](2) > M_PI)
                Vertexs[j](2) -= 2 * M_PI;
            else if (Vertexs[j](2) < -M_PI)
                Vertexs[j](2) += 2 * M_PI;
            
        }
        
        //TODO--End

        double maxError = -1;
        for(int k = 0; k < 3 * Vertexs.size();k++)
        {
            if(maxError < std::fabs(dx(k)))
            {
                maxError = std::fabs(dx(k));
            }
        }

        if(maxError < epsilon)
            break;
    }

    auto t2 = std::chrono::high_resolution_clock::now();;
    auto t_cnt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf(">>> Cost time: %ld ms\n",t_cnt);

    double finalError  = ComputeError(Vertexs,Edges);
    std::cout <<"FinalError:"<<finalError<<std::endl;

    PublishGraphForVisulization(&afterGraphPub,
                                Vertexs,
                                Edges,1);
    ros::spin();

    // printf(">>> %d\n",Vertexs.size());
    return 0;
}




