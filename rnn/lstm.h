#ifndef LSTM_H
#define LSTM_H

#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <cassert>
#include <Eigen/Dense>

#include "utils.hpp"
#include "weight_init.h"

class LSTM{
protected:
  float learning_rate_;  
  int hidden_size_;
  int input_size_;
  int output_size_;
  int time_step_;
  int minibatch_size_;
  int batch_count_;
  
  std::vector<Eigen::VectorXf> alpha_t_;
  std::vector<Eigen::VectorXf> beta_t_;
  std::vector<Eigen::VectorXf> gamma_t_;
  std::vector<Eigen::VectorXf> delta_t_;
  std::vector<Eigen::VectorXf> z_t_;

  // the gradients for minibatch
  Eigen::MatrixXf dw_ix_;
  Eigen::MatrixXf dw_ih_;
  Eigen::VectorXf db_i_;
  
  Eigen::MatrixXf dw_fx_;
  Eigen::MatrixXf dw_fh_;
  Eigen::VectorXf db_f_;
  
  Eigen::MatrixXf dw_ox_;
  Eigen::MatrixXf dw_oh_;
  Eigen::VectorXf db_o_;

  Eigen::MatrixXf dw_cx_;
  Eigen::MatrixXf dw_ch_;
  Eigen::VectorXf db_c_;

  Eigen::MatrixXf dw_qh_;
  Eigen::VectorXf db_q_;

  // For the input gate
  Eigen::MatrixXf w_ix_;
  Eigen::MatrixXf w_ih_;
  Eigen::VectorXf b_i_;
  std::vector<Eigen::VectorXf> i_t_;
  
  // For the forget gate
  Eigen::MatrixXf w_fx_;
  Eigen::MatrixXf w_fh_;
  Eigen::VectorXf b_f_;
  std::vector<Eigen::VectorXf> f_t_;
  
  // For the output gate
  Eigen::MatrixXf w_ox_;
  Eigen::MatrixXf w_oh_;
  Eigen::VectorXf b_o_;
  std::vector<Eigen::VectorXf> o_t_;
  
  // For the candiate gate
  Eigen::MatrixXf w_cx_;
  Eigen::MatrixXf w_ch_;
  Eigen::VectorXf b_c_;
  std::vector<Eigen::VectorXf> c_tilde_t_;

  // For the output
  Eigen::MatrixXf w_qh_;
  Eigen::VectorXf b_q_;

  // For the memory cell
  std::vector<Eigen::VectorXf> c_t_;

  // For the hidden state
  std::vector<Eigen::VectorXf> h_t_;
  
public:
  LSTM(int input_size,int output_size,int hidden_size,int minibatch_size=32,float learning_rate=1e-1);
  
  std::vector<Eigen::VectorXf> Generate(const std::vector<Eigen::VectorXf>& prefix,
					int text_len) const;
  std::vector<Eigen::VectorXf> Net(const std::vector<Eigen::VectorXf>& inputs);
  std::vector<Eigen::VectorXf> Forward(const std::vector<Eigen::VectorXf>& inputs);  
  float GetLoss(const std::vector<Eigen::VectorXf>& inputs,
		const std::vector<Eigen::VectorXf>& labels);
  void Backward(const std::vector<Eigen::VectorXf>& inputs,
		const std::vector<Eigen::VectorXf>& outputs,
		const std::vector<Eigen::VectorXf>& labels);

  void ClearGradients();
  void UpdateAllParams();
  template <typename T>
  void UpdateParam(T& param,T& g);  
  template <typename T>
  void GradClipping(T& g,float theta);  
};

#endif// LSTM_H
