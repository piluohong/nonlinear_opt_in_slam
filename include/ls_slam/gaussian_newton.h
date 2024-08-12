// #ifndef GAUSSIAN_NEWTON_H
// #define GAUSSIAN_NEWTON_H
#pragma once
#include <vector>
#include <eigen3/Eigen/Core>
#include <omp.h>
// 边类
typedef struct edge
{
  int xi,xj;
  Eigen::Vector3d measurement;  //  位置观测值
  Eigen::Matrix3d infoMatrix;   //  信息矩阵
}Edge;


Eigen::VectorXd  LinearizeAndSolve(std::vector<Eigen::Vector3d>& Vertexs,
                                   std::vector<Edge>& Edges);

double ComputeError(std::vector<Eigen::Vector3d>& Vertexs,
                    std::vector<Edge>& Edges);








// #endif
