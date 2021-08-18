#include "rnn.h"

Rnn::Rnn(int input_size,int output_size,int hidden_size,float learning_rate):
  input_size_(input_size),output_size_(output_size),
  hidden_size_(hidden_size),learning_rate_(learning_rate)
{
  w_hx_ = Eigen::MatrixXf::Random(hidden_size_,input_size_);
  w_hh_ = Eigen::MatrixXf::Random(hidden_size_,hidden_size_);
  w_qh_ = Eigen::MatrixXf::Random(output_size_,hidden_size_);
  b_h_  = Eigen::VectorXf::Zero(hidden_size_);
  b_q_  = Eigen::VectorXf::Zero(output_size_);
}

std::vector<Eigen::VectorXf> Rnn::Run(const Eigen::VectorXf& inputs,int len){
  std::vector<Eigen::VectorXf> outputs;
  Eigen::VectorXf h(hidden_size_);// Initial state of the hidden state.
  h.setZero();

  outputs.push_back(inputs);
  Eigen::VectorXf x = inputs;
  for(int i=0;i<len;++i){
    Eigen::VectorXf a = w_hx_*x + w_hh_*h + b_h_;    
    h = a.unaryExpr(&Tanh);        
    Eigen::VectorXf y = w_qh_*h + b_q_;      
    Eigen::VectorXf o = y.unaryExpr(&Tanh);
    outputs.push_back(o);
      
    x = Softmax(o);
  }

  return outputs;
}
  
std::vector<Eigen::VectorXf> Rnn::Net(const std::vector<Eigen::VectorXf>& inputs){
  std::vector<Eigen::VectorXf> outputs;
  Eigen::VectorXf h(hidden_size_);// Initial state of the hidden state.
  h.setZero();

  for(auto x : inputs){
    Eigen::VectorXf a = w_hx_*x + w_hh_*h + b_h_;    
    h = a.unaryExpr(&Tanh);        
    Eigen::VectorXf y = w_qh_*h + b_q_;      
    Eigen::VectorXf o = y.unaryExpr(&Tanh);
    outputs.push_back(o);
  }

  return outputs;
}

std::vector<Eigen::VectorXf> Rnn::Forward(const std::vector<Eigen::VectorXf>& inputs){
  std::vector<Eigen::VectorXf> outputs;
  time_step_ = inputs.size();
    
  y_t_.clear();
  h_t_.clear();
  a_t_.clear();

  Eigen::VectorXf h(hidden_size_);// Initial state of the hidden state.
  h.setZero();    
  h_t_.push_back(h);
      
  for(auto x : inputs){
    Eigen::VectorXf a = w_hx_*x + w_hh_*h + b_h_;
    a_t_.push_back(a);
    
    h = a.unaryExpr(&Tanh);    
    h_t_.push_back(h);
    
    Eigen::VectorXf y = w_qh_*h + b_q_;
    y_t_.push_back(y);
      
    Eigen::VectorXf o = y.unaryExpr(&Tanh);
    outputs.push_back(o);
  }

  return outputs;
}
  
float Rnn::GetLoss(const std::vector<Eigen::VectorXf>& inputs,
	      const std::vector<Eigen::VectorXf>& labels)
{
  float loss = 0;
  std::vector<Eigen::VectorXf> outputs = Net(inputs);
  int size = outputs.size();
  for(int t=0;t<size;++t){            
    loss += CategoricalCrossEntropy(outputs[t],labels[t]);
  }

  return loss;
}

void Rnn::Backward(const std::vector<Eigen::VectorXf>& inputs,
	      const std::vector<Eigen::VectorXf>& outputs,
	      const std::vector<Eigen::VectorXf>& labels)
{
  std::vector<Eigen::VectorXf> do_t;
  Eigen::VectorXf error;
  Eigen::ArrayXf f_prime;
  Eigen::ArrayXf g_prime;
  Eigen::MatrixXf da_t;
  Eigen::MatrixXf dy_t;

  // Should I move these variables to the member variables?
  Eigen::MatrixXf dw_hx(hidden_size_,input_size_);
  Eigen::MatrixXf dw_hh(hidden_size_,hidden_size_);
  Eigen::MatrixXf dw_qh(output_size_,hidden_size_);
  Eigen::VectorXf db_h(hidden_size_);
  Eigen::VectorXf db_q(output_size_);
  Eigen::VectorXf dh_t(hidden_size_);
    
  dw_hx.setZero();
  dw_hh.setZero();
  dw_qh.setZero();
  db_h.setZero();
  db_q.setZero();
  dh_t.setZero();
    
  for(int t=0;t<time_step_;++t){
    error = GradientOfCategoricalCrossEntropy(outputs[t],labels[t]);
    do_t.push_back(error);
  }

  // BPTT
  for(int t=time_step_-1;t>=0;--t){
    f_prime = a_t_[t].unaryExpr(&DerivativeOfTanh).array();      
    da_t = (dh_t.array()*f_prime).matrix();

    g_prime = y_t_[t].unaryExpr(&DerivativeOfTanh).array();
    dy_t = (do_t[t].array()*g_prime).matrix();
      
    db_q  += dy_t;      
    dw_qh += dy_t*h_t_[t+1].transpose();
    dh_t  += w_qh_.transpose()*dy_t;

    dw_hx += da_t*inputs[t].transpose();
    db_h  += da_t;
    dw_hh += da_t*h_t_[t].transpose();

    dh_t  = w_hh_.transpose()*da_t;
  }
    
  // Update
  w_qh_ -= learning_rate_*dw_qh;
  w_hx_ -= learning_rate_*dw_hx;
  w_hh_ -= learning_rate_*dw_hh;  
  b_q_  -= learning_rate_*db_q;
  b_h_  -= learning_rate_*db_h;
}
