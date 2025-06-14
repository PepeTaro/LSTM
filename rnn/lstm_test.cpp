#include <string>
#include <iostream>
#include <fstream>
#include "lstm.h"

const int time_step = 20;

void InsertData(std::vector<std::vector<Eigen::VectorXf>>& inputs,
		std::vector<std::vector<Eigen::VectorXf>>& labels,
		const std::string& text)
{
  
  std::vector<Eigen::VectorXf> input;
  std::vector<Eigen::VectorXf> label;

  int step = text.size() - time_step;
  if(step < 0) step = 0;
  
  for(int i=0;i<step;++i){
    EncodeSentence(text.substr(i).c_str(),input,label);

    inputs.push_back(input);
    labels.push_back(label);

    //DecodeSentence(input);
    //DecodeSentence(label);    
  }  
}

void GenerateDataset(std::vector<std::vector<Eigen::VectorXf>>& inputs,
		     std::vector<std::vector<Eigen::VectorXf>>& labels,
		     const std::string& filename)  
{
  std::string line;
  std::ifstream file(filename);
  if(not file.is_open()){
    std::cerr << "[!] Error There is no such file:" << filename << std::endl;
    exit(-1);
  }

  while(std::getline(file,line)){
    if(line.empty()) continue;
    InsertData(inputs,labels,line.c_str());
  }
  file.close();  
}

void GenerateText(const LSTM& lstm,const std::string& prefix,int text_len){
  std::vector<Eigen::VectorXf> input;
  std::vector<Eigen::VectorXf> outputs;
  
  Str2Data(prefix,input);
  outputs = lstm.Generate(input,text_len);
  std::cout << "Generated text:" << std::endl;
  DecodeSentence(outputs);  
}

int main(){
  
  std::vector<std::vector<Eigen::VectorXf>> inputs;
  std::vector<std::vector<Eigen::VectorXf>> labels;
  std::vector<Eigen::VectorXf> outputs;
  
  GenerateDataset(inputs,labels,"../text.txt");
  /*
  InsertData(inputs,labels,"Mathematics is the most important displine in science.");
  InsertData(inputs,labels,"For mathematics is used all over the places in science.");
  InsertData(inputs,labels,"I also like mathematics because it gives me a power.");
  InsertData(inputs,labels,"But some people think mathematics is hard.");
  InsertData(inputs,labels,"But it's not true!");
  */
  //InsertData(inputs,labels,"Mathematics is the most important displine in science.");
  
  LSTM lstm(voc_size,voc_size,128,32,1.0);  
  int data_size = inputs.size();

  //GenerateText(lstm,"Mathematics is ",50);
  GenerateText(lstm,"You would not",100);
  
  for(int e=0;e<500;++e){
    float loss = 0;    
    for(int n=0;n<data_size;++n){
      outputs = lstm.Forward(inputs[n]);
      lstm.Backward(inputs[n],outputs,labels[n]);
      loss += lstm.GetLoss(inputs[n],labels[n])/outputs.size();
    }
    lstm.UpdateAllParams();
    std::cout << "[Epoch " << e << "] Loss:" << loss << std::endl;
    if(e%100 == 0){
      //GenerateText(lstm,"Mathematics is ",50);
      GenerateText(lstm,"You would not",100);
    }
  }
  
  //GenerateText(lstm,"Mathematics is ",50);
  GenerateText(lstm,"You would not",100);
}
