#include "gaussian_newton.h"
#include <eigen3/Eigen/Jacobi>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Householder>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/LU>


#include <iostream>


//位姿-->转换矩阵
Eigen::Matrix3d PoseToTrans(Eigen::Vector3d x)
{
    Eigen::Matrix3d trans;
    trans << cos(x(2)),-sin(x(2)),x(0),
             sin(x(2)), cos(x(2)),x(1),
                     0,         0,    1;

    return trans;
}


//转换矩阵－－＞位姿
Eigen::Vector3d TransToPose(Eigen::Matrix3d trans)
{
    Eigen::Vector3d pose;
    pose(0) = trans(0,2);
    pose(1) = trans(1,2);
    pose(2) = atan2(trans(1,0),trans(0,0));

    return pose;
}

//计算整个pose-graph的误差
double ComputeError(std::vector<Eigen::Vector3d>& Vertexs,
                    std::vector<Edge>& Edges)
{
    double sumError = 0;
    for(int i = 0; i < Edges.size();i++)
    {
        Edge tmpEdge = Edges[i];
        Eigen::Vector3d xi = Vertexs[tmpEdge.xi];
        Eigen::Vector3d xj = Vertexs[tmpEdge.xj];
        Eigen::Vector3d z = tmpEdge.measurement;    //  观测值
        Eigen::Matrix3d infoMatrix = tmpEdge.infoMatrix;    //  信息矩阵

        Eigen::Matrix3d Xi = PoseToTrans(xi);
        Eigen::Matrix3d Xj = PoseToTrans(xj);
        Eigen::Matrix3d Z  = PoseToTrans(z);

        Eigen::Matrix3d Ei = Z.inverse() *  Xi.inverse() * Xj;

        Eigen::Vector3d ei = TransToPose(Ei);


        sumError += ei.transpose() * infoMatrix * ei;
    }
    return sumError;
}


/**
 * @brief CalcJacobianAndError
 *         计算jacobian矩阵和error
 * @param xi    fromIdx
 * @param xj    toIdx
 * @param z     观测值:xj相对于xi的坐标
 * @param ei    计算的误差
 * @param Ai    相对于xi的Jacobian矩阵
 * @param Bi    相对于xj的Jacobian矩阵
 */
void CalcJacobianAndError(Eigen::Vector3d xi,Eigen::Vector3d xj,Eigen::Vector3d z,
                          Eigen::Vector3d& ei,Eigen::Matrix3d& Ai,Eigen::Matrix3d& Bi)
{
    //TODO--Start
    Eigen::Matrix2d R_ij_ ;
    R_ij_ << cos(z(2)), sin(z(2)),
            -sin(z(2)), cos(z(2));  //  赋值
    Eigen::Matrix2d R_i_;
    R_i_ << cos(xi(2)), sin(xi(2)),
            -sin(xi(2)), cos(xi(2));  //  赋值

    Eigen::Matrix2d d_R_i__to_sita;
    d_R_i__to_sita<< -sin(xi(2)), cos(xi(2)),
                     -cos(xi(2)), -sin(xi(2));

    //  计算平移和旋转部分误差
    ei.block(0,0,2,1) = R_ij_ * (R_i_*(xj.block(0,0,2,1) - xi.block(0,0,2,1)) - z.block(0,0,2,1));  //  平移部分
    ei(2) = xj(2) - xi(2) - z(2); //  旋转角度

     //将角度 限制在 -pi ~ pi
    if (ei(2) > M_PI)
        ei(2) -= 2 * M_PI;
    else if (ei(2) < -M_PI)
        ei(2) += 2 * M_PI;   
    
    //  对Xi的雅克比
    Ai.block(0,0,2,2) = -R_ij_ * R_i_;
    Ai.block(0,2,2,1) = R_ij_ * d_R_i__to_sita * (xj.block(0,0,2,1) - xi.block(0,0,2,1));
    Ai.block(2,0,1,3) << 0,0,-1;

    Bi.setIdentity();

    //  对Xj的雅克比
    Bi.block(0,0,2,2) = R_ij_ * R_i_;
    // Bi.block(0,2,3,1)<< 0,0,1;
    // Bi.block(2,0,1,2)<< 0,0;

    //TODO--end
}

/**
 * @brief LinearizeAndSolve
 *        高斯牛顿方法的一次迭代．
 * @param Vertexs   图中的所有节点
 * @param Edges     图中的所有边
 * @return          位姿的增量
 */
Eigen::VectorXd  LinearizeAndSolve(std::vector<Eigen::Vector3d>& Vertexs,
                                   std::vector<Edge>& Edges)
{
    //申请内存
    Eigen::MatrixXd H(Vertexs.size() * 3,Vertexs.size() * 3);//H矩阵的维度(点个数*单点纬度)*(点个数*单点纬度)
    Eigen::VectorXd b(Vertexs.size() * 3);  //  b矩阵的维度(点个数*单点纬度)

    H.setZero();    //  置零
    b.setZero();    //  置零

    //固定第一帧
    Eigen::Matrix3d I;
    I.setIdentity();
    // std::cout << I << std::endl;
    H.block(0,0,3,3) += I;

    //构造H矩阵　＆ b向量
    omp_set_num_threads(16);
	#pragma omp parallel for
    for(int i = 0; i < Edges.size();i++)
    {
        //提取信息,提取边的两个顶点序号
        Edge tmpEdge = Edges[i];
        Eigen::Vector3d xi = Vertexs[tmpEdge.xi];
        Eigen::Vector3d xj = Vertexs[tmpEdge.xj];
        Eigen::Vector3d z = tmpEdge.measurement;
        Eigen::Matrix3d infoMatrix = tmpEdge.infoMatrix;

        //计算误差和对应的Jacobian
        Eigen::Vector3d ei;
        Eigen::Matrix3d Ai;
        Eigen::Matrix3d Bi;
        
        CalcJacobianAndError(xi,xj,z,ei,Ai,Bi);
        
        
        //TODO--Start
        b.block(3*tmpEdge.xi,0,3,1) += Ai.transpose() * infoMatrix.transpose() * ei;
        b.block(3*tmpEdge.xj,0,3,1) += Bi.transpose() * infoMatrix.transpose() * ei;

        H.block(3*tmpEdge.xi,3*tmpEdge.xi,3,3) += Ai.transpose() * infoMatrix * Ai;
        H.block(3*tmpEdge.xi,3*tmpEdge.xj,3,3) += Ai.transpose() * infoMatrix * Bi;
        H.block(3*tmpEdge.xj,3*tmpEdge.xi,3,3) += Bi.transpose() * infoMatrix * Ai;
        H.block(3*tmpEdge.xj,3*tmpEdge.xj,3,3) += Bi.transpose() * infoMatrix * Bi;
        //TODO--End

        // std::cout<<"****************"<<i<<std::endl;

    }
    // std::cout << b.block(300,0,3,1) << std::endl;
    //求解
    Eigen::VectorXd dx;

    //TODO--Start
    // dx = H.colPivHouseholderQr().solve(-b);//有了H和b即可以求解dx
    // dx = H.inverse() * (-b);
    dx = H.ldlt().solve(-b);
    std::cout<<"xxxxxxxxxxxxxxx finish solve."<<std::endl;
    
    //TODO-End

    return dx;
}











