#include "lstm.h"

LSTM::LSTM(int input_size,int output_size,int hidden_size,int minibatch_size,float learning_rate):
  input_size_(input_size),output_size_(output_size),hidden_size_(hidden_size),
  minibatch_size_(minibatch_size),batch_count_(1),learning_rate_(learning_rate)
{
  
  w_ix_ = InitWeight(input_size_,hidden_size_);
  w_ih_ = InitWeight(hidden_size_,hidden_size_);
  b_i_  = Eigen::VectorXf::Zero(hidden_size_);

  w_fx_ = InitWeight(input_size_,hidden_size_);
  w_fh_ = InitWeight(hidden_size_,hidden_size_);
  b_f_  = Eigen::VectorXf::Zero(hidden_size_);

  w_ox_ = InitWeight(input_size_,hidden_size_);
  w_oh_ = InitWeight(hidden_size_,hidden_size_);
  b_o_  = Eigen::VectorXf::Zero(hidden_size_);

  w_cx_ = InitWeight(input_size_,hidden_size_);
  w_ch_ = InitWeight(hidden_size_,hidden_size_);
  b_c_  = Eigen::VectorXf::Zero(hidden_size_);

  w_qh_ = InitWeight(hidden_size_,output_size_);
  b_q_  = Eigen::VectorXf::Zero(output_size_);

  
  dw_ix_ = Eigen::MatrixXf::Zero(hidden_size_,input_size_);
  dw_ih_ = Eigen::MatrixXf::Zero(hidden_size_,hidden_size_);
  db_i_  = Eigen::VectorXf::Zero(hidden_size_);

  dw_fx_ = Eigen::MatrixXf::Zero(hidden_size_,input_size_);
  dw_fh_ = Eigen::MatrixXf::Zero(hidden_size_,hidden_size_);
  db_f_  = Eigen::VectorXf::Zero(hidden_size_);

  dw_ox_ = Eigen::MatrixXf::Zero(hidden_size_,input_size_);
  dw_oh_ = Eigen::MatrixXf::Zero(hidden_size_,hidden_size_);
  db_o_  = Eigen::VectorXf::Zero(hidden_size_);

  dw_cx_ = Eigen::MatrixXf::Zero(hidden_size_,input_size_);
  dw_ch_ = Eigen::MatrixXf::Zero(hidden_size_,hidden_size_);
  db_c_  = Eigen::VectorXf::Zero(hidden_size_);

  dw_qh_ = Eigen::MatrixXf::Zero(output_size_,hidden_size_);
  db_q_  = Eigen::VectorXf::Zero(output_size_);    
}

std::vector<Eigen::VectorXf> LSTM::Generate(const std::vector<Eigen::VectorXf>& prefix,
				       int text_len) const
{
  std::vector<Eigen::VectorXf> outputs;
  Eigen::VectorXf h(hidden_size_);// Initial state of the hidden state.
  Eigen::VectorXf memory_cell(hidden_size_); // Memory cell
  h.setZero();
  memory_cell.setZero();
    
  // Temporary variables
  Eigen::VectorXf alpha;
  Eigen::VectorXf input_gate;
  Eigen::VectorXf beta;
  Eigen::VectorXf forget_gate;
  Eigen::VectorXf gamma;
  Eigen::VectorXf output_gate;
  Eigen::VectorXf delta;
  Eigen::VectorXf candidate_gate;    
  Eigen::VectorXf z;
  Eigen::VectorXf q;
  Eigen::VectorXf x;
  
  int prefix_size = prefix.size();
  // Warm-up period
  for(int i=0;i<prefix_size;++i){
    outputs.push_back(prefix[i]);
    x = prefix[i];

    alpha = w_ix_*x + w_ih_*h + b_i_;    
    input_gate = alpha.unaryExpr(&Sigmoid);
    
    beta = w_fx_*x + w_fh_*h + b_f_;    
    forget_gate = beta.unaryExpr(&Sigmoid);
    
    gamma = w_ox_*x + w_oh_*h + b_o_;    
    output_gate = gamma.unaryExpr(&Sigmoid);

    delta = w_cx_*x + w_ch_*h + b_c_;    
    candidate_gate = delta.unaryExpr(&Tanh);

    // Update the memory cell
    memory_cell =
      forget_gate.array()*memory_cell.array() + input_gate.array()*candidate_gate.array();

    // Update the hidden state
    h = output_gate.array()*memory_cell.unaryExpr(&Tanh).array();    
  }
  
  for(int i=0;i<text_len;++i){
    alpha = w_ix_*x + w_ih_*h + b_i_;    
    input_gate = alpha.unaryExpr(&Sigmoid);
    
    beta = w_fx_*x + w_fh_*h + b_f_;    
    forget_gate = beta.unaryExpr(&Sigmoid);
    
    gamma = w_ox_*x + w_oh_*h + b_o_;    
    output_gate = gamma.unaryExpr(&Sigmoid);

    delta = w_cx_*x + w_ch_*h + b_c_;    
    candidate_gate = delta.unaryExpr(&Tanh);

    // Update the memory cell
    memory_cell =
      forget_gate.array()*memory_cell.array() + input_gate.array()*candidate_gate.array();

    // Update the hidden state
    h = output_gate.array()*memory_cell.unaryExpr(&Tanh).array();

    z = w_qh_*h + b_q_;
    q = z.unaryExpr(&Linear);
    x = Softmax(q);
    
    outputs.push_back(x);
  }

  return outputs;
}

std::vector<Eigen::VectorXf> LSTM::Net(const std::vector<Eigen::VectorXf>& inputs){
  std::vector<Eigen::VectorXf> outputs;
  Eigen::VectorXf h(hidden_size_);// Initial state of the hidden state.
  Eigen::VectorXf memory_cell(hidden_size_); // Memory cell
  h.setZero();
  memory_cell.setZero();

  // Temporary variables
  Eigen::VectorXf alpha;
  Eigen::VectorXf input_gate;
  Eigen::VectorXf beta;
  Eigen::VectorXf forget_gate;
  Eigen::VectorXf gamma;
  Eigen::VectorXf output_gate;
  Eigen::VectorXf delta;
  Eigen::VectorXf candidate_gate;    
  Eigen::VectorXf z;
  Eigen::VectorXf q;

  for(auto x : inputs){
    alpha = w_ix_*x + w_ih_*h + b_i_;    
    input_gate = alpha.unaryExpr(&Sigmoid);
    
    beta = w_fx_*x + w_fh_*h + b_f_;    
    forget_gate = beta.unaryExpr(&Sigmoid);
    
    gamma = w_ox_*x + w_oh_*h + b_o_;    
    output_gate = gamma.unaryExpr(&Sigmoid);

    delta = w_cx_*x + w_ch_*h + b_c_;    
    candidate_gate = delta.unaryExpr(&Tanh);

    // Update the memory cell
    memory_cell =
      forget_gate.array()*memory_cell.array() + input_gate.array()*candidate_gate.array();

    // Update the hidden state
    h = output_gate.array()*memory_cell.unaryExpr(&Tanh).array();

    z = w_qh_*h + b_q_;
    q = z.unaryExpr(&Linear);
    
    outputs.push_back(q);
  }

  return outputs;
}

std::vector<Eigen::VectorXf> LSTM::Forward(const std::vector<Eigen::VectorXf>& inputs){
  std::vector<Eigen::VectorXf> outputs;
  Eigen::VectorXf h(hidden_size_);// Initial state of the hidden state.
  Eigen::VectorXf memory_cell(hidden_size_); // Memory cell
  h.setZero();
  memory_cell.setZero();

  time_step_ = inputs.size();
  
  // All the containers are reset before feedforwarding.
  alpha_t_.clear();
  beta_t_.clear();
  gamma_t_.clear();
  delta_t_.clear();
  i_t_.clear();
  f_t_.clear();
  o_t_.clear();
  c_tilde_t_.clear();
  c_t_.clear();
  h_t_.clear();
  
  h_t_.push_back(h);
  c_t_.push_back(memory_cell);

  // Temporary variables
  Eigen::VectorXf alpha;
  Eigen::VectorXf input_gate;
  Eigen::VectorXf beta;
  Eigen::VectorXf forget_gate;
  Eigen::VectorXf gamma;
  Eigen::VectorXf output_gate;
  Eigen::VectorXf delta;
  Eigen::VectorXf candidate_gate;    
  Eigen::VectorXf z;
  Eigen::VectorXf q;
    
  for(auto x : inputs){
    alpha = w_ix_*x + w_ih_*h + b_i_;    
    input_gate = alpha.unaryExpr(&Sigmoid);
    
    beta = w_fx_*x + w_fh_*h + b_f_;    
    forget_gate = beta.unaryExpr(&Sigmoid);
    
    gamma = w_ox_*x + w_oh_*h + b_o_;    
    output_gate = gamma.unaryExpr(&Sigmoid);

    delta = w_cx_*x + w_ch_*h + b_c_;    
    candidate_gate = delta.unaryExpr(&Tanh);
    
    // Update the memory cell
    memory_cell =
      forget_gate.array()*memory_cell.array() + input_gate.array()*candidate_gate.array();

    // Update the hidden state
    h = output_gate.array()*memory_cell.unaryExpr(&Tanh).array();

    // Compute the outputs
    z = w_qh_*h + b_q_;
    q = z.unaryExpr(&Linear);

    alpha_t_.push_back(alpha);
    beta_t_.push_back(beta);
    gamma_t_.push_back(gamma);
    delta_t_.push_back(delta);
    z_t_.push_back(z);
    
    i_t_.push_back(input_gate);
    f_t_.push_back(forget_gate);
    o_t_.push_back(output_gate);
    c_tilde_t_.push_back(candidate_gate);

    c_t_.push_back(memory_cell);
    h_t_.push_back(h);
      
    outputs.push_back(q);    
  }

  return outputs;
}
  
float LSTM::GetLoss(const std::vector<Eigen::VectorXf>& inputs,
	      const std::vector<Eigen::VectorXf>& labels)
{
  float loss = 0;
  std::vector<Eigen::VectorXf> outputs = Net(inputs);
  int size = outputs.size();
  
  for(int t=0;t<size;++t){            
    loss += CategoricalCrossEntropy(outputs[t],labels[t]);
  }

  return loss/size;
}

void LSTM::Backward(const std::vector<Eigen::VectorXf>& inputs,
		    const std::vector<Eigen::VectorXf>& outputs,
		    const std::vector<Eigen::VectorXf>& labels)
{
  // Errors
  std::vector<Eigen::VectorXf> dq_t;  
  // Temporary variables
  Eigen::VectorXf dq;
  Eigen::VectorXf dz_t;
  Eigen::VectorXf dh_t;
  Eigen::VectorXf dinput_t;  
  Eigen::VectorXf dalpha_t;
  Eigen::VectorXf dforget_t;
  Eigen::VectorXf dbeta_t;
  Eigen::VectorXf doutput_t;
  Eigen::VectorXf dgamma_t;
  Eigen::VectorXf dcandidate_t;
  Eigen::VectorXf ddelta_t;

  for(int t=0;t<time_step_;++t){
    dq = (1.0/time_step_)*GradientOfCategoricalCrossEntropy(outputs[t],labels[t]);
    dq_t.push_back(dq);
  }

  // BPTT
  #pragma omp parallel
  #pragma omp for  
  for(int t=time_step_-1;t>=0;--t){
    // For the output layer
    dz_t = dq_t[t].array()*z_t_[t].unaryExpr(&DerivativeOfLinear).array();
    dw_qh_ += dz_t*h_t_[t].transpose();
    db_q_  += dz_t;
    
    dh_t = w_qh_.transpose()*dz_t;
    
    // For the input gate
    dinput_t = dh_t.array()*o_t_[t].array()*c_t_[t+1].unaryExpr(&DerivativeOfTanh).array()*c_tilde_t_[t].array();
    dalpha_t = dinput_t.array()*alpha_t_[t].unaryExpr(&DerivativeOfSigmoid).array();
    dw_ix_ += dalpha_t*inputs[t].transpose();
    dw_ih_ += dalpha_t*h_t_[t].transpose();
    db_i_  += dalpha_t;
    
    // For the forget gate
    dforget_t = dh_t.array()*o_t_[t].array()*c_t_[t+1].unaryExpr(&DerivativeOfTanh).array()*c_t_[t].array();
    dbeta_t = dforget_t.array()*beta_t_[t].unaryExpr(&DerivativeOfSigmoid).array();

    dw_fx_ += dbeta_t*inputs[t].transpose();
    dw_fh_ += dbeta_t*h_t_[t].transpose();
    db_f_  += dbeta_t;

    // For the output gate
    doutput_t = dh_t.array()*c_t_[t+1].unaryExpr(&Tanh).array();
    dgamma_t = doutput_t.array()*gamma_t_[t].unaryExpr(&DerivativeOfSigmoid).array();

    dw_ox_ += dgamma_t*inputs[t].transpose();
    dw_oh_ += dgamma_t*h_t_[t].transpose();
    db_o_  += dgamma_t;
    
    // For the candidate gate
    dcandidate_t = dh_t.array()*o_t_[t].array()*c_t_[t+1].unaryExpr(&DerivativeOfTanh).array()*i_t_[t].array();
    ddelta_t = dcandidate_t.array()*delta_t_[t].unaryExpr(&DerivativeOfSigmoid).array();

    dw_cx_ += ddelta_t*inputs[t].transpose();
    dw_ch_ += ddelta_t*h_t_[t].transpose();
    db_c_  += ddelta_t;
    
  }  

  UpdateAllParams();
}

void LSTM::ClearGradients(){  
  dw_ix_.setZero();
  dw_ih_.setZero();
  db_i_.setZero();

  dw_fx_.setZero();
  dw_fh_.setZero();
  db_f_.setZero();

  dw_ox_.setZero();
  dw_oh_.setZero();
  db_o_.setZero();

  dw_cx_.setZero();
  dw_ch_.setZero();
  db_c_.setZero();

  dw_qh_.setZero();
  db_q_.setZero();
}

void LSTM::UpdateAllParams(){  
  if(batch_count_ == minibatch_size_){
    // Update the input gate
    UpdateParam(w_ix_,dw_ix_);
    UpdateParam(w_ih_,dw_ih_);
    UpdateParam(b_i_,db_i_);

    // Update the forget gate
    UpdateParam(w_fx_,dw_fx_);
    UpdateParam(w_fh_,dw_fh_);
    UpdateParam(b_f_,db_f_);
    
    // Update the output gate
    UpdateParam(w_ox_,dw_ox_);
    UpdateParam(w_oh_,dw_oh_);
    UpdateParam(b_o_,db_o_);

    // Update the candidate gate
    UpdateParam(w_cx_,dw_cx_);
    UpdateParam(w_ch_,dw_ch_);
    UpdateParam(b_c_,db_c_);

    // Update the output layer
    UpdateParam(w_qh_,dw_qh_);
    UpdateParam(b_q_,db_q_);

    ClearGradients();
    batch_count_ = 1;
  }else{
    batch_count_++;
  }
}

template <typename T>
void LSTM::UpdateParam(T& param,T& g){
  GradClipping(g,1.0);
  param -= learning_rate_*g;
}

template <typename T>
void LSTM::GradClipping(T& g,float theta){
  float norm = sqrtf(g.array().pow(2).sum());
  if(norm > theta){
    g *= theta/norm;
  }
}


