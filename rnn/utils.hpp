#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <cassert>
#include <Eigen/Dense>

const int voc_size = 95;

inline float LogSumExp(const Eigen::VectorXf& logits){
  Eigen::VectorXf logits_minus_max;
  Eigen::VectorXf exp_logits_minus_max;
  Eigen::VectorXf max_vector;
  float sum_of_exps;
  float return_value;
  float max;
  
  max_vector = Eigen::VectorXf(logits.size());
  max = logits.maxCoeff();
  max_vector.setConstant(max);

  logits_minus_max = logits - max_vector;
  exp_logits_minus_max = logits_minus_max.array().exp();
  
  sum_of_exps  = exp_logits_minus_max.sum();
  return_value = max + log(sum_of_exps);

  return return_value;
}

inline Eigen::VectorXf Softmax(const Eigen::VectorXf& logits){
  Eigen::VectorXf softmax;
  Eigen::VectorXf lse_vector;
  float lse;

  lse = LogSumExp(logits);
  lse_vector = Eigen::VectorXf(logits.size());  
  lse_vector.setConstant(lse);

  softmax = (logits - lse_vector).array().exp();
  return softmax;
}

inline float CategoricalCrossEntropy(const Eigen::VectorXf& logits,const Eigen::VectorXf& labels){
  static float epsilon = 1e-12;
  Eigen::VectorXf softmax;
  Eigen::VectorXf log_of_softmax;
  Eigen::VectorXf labels_times_log_of_softmax;
  Eigen::VectorXf epsilon_vec = Eigen::VectorXf::Constant(logits.size(),epsilon);
  float loss;
    
  softmax = Softmax(logits);
  log_of_softmax = (epsilon_vec + softmax).array().log(); // logを使用するため,softmaxの確率が0だと困るので非常に小さい数epsilonを加えている
  labels_times_log_of_softmax = (labels.array())*(log_of_softmax.array());// 成分ごとに掛け合わせていることに注意。
  loss = -1.0 * labels_times_log_of_softmax.sum();
  
  return loss;
}

inline Eigen::VectorXf GradientOfCategoricalCrossEntropy(const Eigen::VectorXf& logits,const Eigen::VectorXf& labels){
  Eigen::VectorXf softmax;
  
  softmax = Softmax(logits);  
  return (softmax - labels);
}

inline float Sigmoid(float x){
  return 1.0/(1.0 + exp(-x));
}

inline float DerivativeOfSigmoid(float x){
  return Sigmoid(x)*(1.0 - Sigmoid(x));
}

inline float Tanh(float x){
  return 2.0/(1.0 + exp(-2*x)) - 1.0;
}

inline float DerivativeOfTanh(float x){
  return 1.0 - powf(Tanh(x),2);
}

inline float Linear(float x){
  return x;
}

inline float DerivativeOfLinear(float x){
  return 1;
}

inline Eigen::VectorXf OneHotEncode(int index){  
  assert(index >= 32 and index <= 126);
  int encode_idx = index - 32;
  
  Eigen::VectorXf output = Eigen::VectorXf::Zero(voc_size);
  output[encode_idx] = 1;
  return output;
}

inline std::string Int2Str(int idx){
  assert(idx >= 0 and idx <= voc_size);

  char c = (idx+32);
  return std::string(1,c);  
}

inline void EncodeSentence(const std::string& sentence,
			   std::vector<Eigen::VectorXf>& inputs,
			   std::vector<Eigen::VectorXf>& labels)
{
  inputs.clear();
  labels.clear();  
  int size = sentence.length();
  
  for(int i=0;i<size-1;++i){        
    inputs.push_back(OneHotEncode((int)sentence[i]));
    labels.push_back(OneHotEncode((int)sentence[i+1]));    
  }    
}

inline void Str2Data(const std::string& sentence,
		     std::vector<Eigen::VectorXf>& inputs)
{
  inputs.clear();
  int size = sentence.length();
  
  for(int i=0;i<size;++i){
    inputs.push_back(OneHotEncode((int)sentence[i]));
  }    
}

inline void DecodeSentence(const std::vector<Eigen::VectorXf>& inputs){
  for(auto c : inputs){
    int max_idx;
    c.maxCoeff(&max_idx);    
    std::cout << Int2Str(max_idx);
  }
  std::cout << std::endl;
}

#endif// UTILS_HPP
