#ifndef WEIGHT_INIT_H
#define WEIGHT_INIT_H

#include <array>
#include <random>
#include <cmath>
#include <Eigen/Dense>

static std::default_random_engine generator;

Eigen::MatrixXf InitWeight(int num_input,int num_output);  
Eigen::VectorXf InitBias(int num_output,float epsilon=0.01);
Eigen::MatrixXf Kaiming(const std::array<int,2>& mat_shape,int num_input);
Eigen::MatrixXf Xavier(const std::array<int,2>& mat_shape,int num_input,int num_output);
  
float Uniform(float lower_bound,float upper_bound);
float Gaussian(float mean,float stddev);

#endif// WEIGHT_INIT_H
