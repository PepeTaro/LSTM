#include "weight_init.h"
  
Eigen::MatrixXf InitWeight(int num_input,int num_output){
  Eigen::MatrixXf weight;
  weight = Xavier({num_output,num_input},num_input,num_output);
  return weight;    
}
  
Eigen::VectorXf InitBias(int num_output,float epsilon){
  Eigen::VectorXf bias;
  bias = Eigen::VectorXf::Constant(num_output,epsilon);

  return bias;
}
  
Eigen::MatrixXf Kaiming(const std::array<int,2>& mat_shape,int num_input){
  Eigen::MatrixXf output(mat_shape[0],mat_shape[1]);
  float mean = 0;
  float stddev = sqrt(2.0/num_input);

  for(int i=0;i<mat_shape[0];++i){
    for(int j=0;j<mat_shape[1];++j){	
      output(i,j) = Gaussian(mean,stddev); 
    }
  }
    
  return output;
}
  
Eigen::MatrixXf Xavier(const std::array<int,2>& mat_shape,int num_input,int num_output){
  Eigen::MatrixXf output(mat_shape[0],mat_shape[1]);
  float bound = sqrt(6.0/(num_input+num_output));
	
  for(int i=0;i<mat_shape[0];++i){
    for(int j=0;j<mat_shape[1];++j){
      output(i,j) = Uniform(-bound,+bound);
    }
  }
    
  return output;
}

float Uniform(float lower_bound,float upper_bound){
  std::uniform_real_distribution<float> distribution(lower_bound,upper_bound);
  return distribution(generator);
}
  
float Gaussian(float mean,float stddev){
  std::normal_distribution<float> distribution(mean,stddev);
  return distribution(generator);
}
