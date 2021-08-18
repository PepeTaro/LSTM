#ifndef RNN_H
#define RNN_H

#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <cassert>
#include <Eigen/Dense>

#include "utils.hpp"

class Rnn{
protected:
  float learning_rate_;  
  int hidden_size_;
  int input_size_;
  int output_size_;
  int time_step_;
      
  std::vector<Eigen::VectorXf> y_t_;
  std::vector<Eigen::VectorXf> h_t_;
  std::vector<Eigen::VectorXf> a_t_;

  Eigen::MatrixXf w_hx_;
  Eigen::MatrixXf w_hh_;
  Eigen::MatrixXf w_qh_;
  Eigen::VectorXf b_h_;
  Eigen::VectorXf b_q_;

public:
  Rnn(int input_size,int output_size,int hidden_size,float learning_rate=1e-1);
  std::vector<Eigen::VectorXf> Run(const Eigen::VectorXf& inputs,int len);  
  std::vector<Eigen::VectorXf> Net(const std::vector<Eigen::VectorXf>& inputs);
  std::vector<Eigen::VectorXf> Forward(const std::vector<Eigen::VectorXf>& inputs);  
  float GetLoss(const std::vector<Eigen::VectorXf>& inputs,
		const std::vector<Eigen::VectorXf>& labels);
  void Backward(const std::vector<Eigen::VectorXf>& inputs,
		const std::vector<Eigen::VectorXf>& outputs,
		const std::vector<Eigen::VectorXf>& labels);
};

#endif// RNN_H
