#include "rnn.h"

int main(){
  const int vec_size = 97;

  std::vector<Eigen::VectorXf> inputs;
  std::vector<Eigen::VectorXf> labels;
  
  EncodeSentence("Fuck you ! bitch?",inputs,labels);
  //EncodeSentence("fuck",inputs,labels);
  //EncodeSentence("!!!",inputs,labels);

  /*
  std::cout << "Inputs:" << std::endl;
  for(auto s : inputs){
    int max_idx;
    s.maxCoeff(&max_idx);    
    std::cout << Int2Str(max_idx) << std::endl;
  }
  std::cout << "Lables:" << std::endl;
  for(auto s : labels){
    int max_idx;
    s.maxCoeff(&max_idx);    
    std::cout << Int2Str(max_idx) << std::endl;
  }
  */
  
  Rnn rnn(vec_size,vec_size,10,1.0);
  
  for(int e=0;e<5000;++e){
    std::vector<Eigen::VectorXf> outputs = rnn.Forward(inputs);    
    rnn.Backward(inputs,outputs,labels);
    
    for(int t=0;t<outputs.size();++t){
      int max_idx;
      Eigen::VectorXf prob = Softmax(outputs[t]);
      prob.maxCoeff(&max_idx);
      std::cout << "[Epoch:" << e << "]Max index:" << Int2Str(max_idx) << std::endl;
    }
    std::cout << std::endl;
    
    float loss = rnn.GetLoss(inputs,labels)/outputs.size();
    std::cout << "Loss:" << loss << std::endl;
    
  }

  std::vector<Eigen::VectorXf> outputs = rnn.Run(inputs[0],10);
  std::cout << "Run:" << std::endl;
  for(auto s : outputs){
    int max_idx;
    Softmax(s).maxCoeff(&max_idx);    
    std::cout << Int2Str(max_idx);
  }
  std::cout << std::endl;
  
}
