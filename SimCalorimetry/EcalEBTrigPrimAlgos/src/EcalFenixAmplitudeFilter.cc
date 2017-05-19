#include <SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h>
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

EcalFenixAmplitudeFilter::EcalFenixAmplitudeFilter()
  :inputsAlreadyIn_(0), shift_(6) {
  }

EcalFenixAmplitudeFilter::~EcalFenixAmplitudeFilter(){}

int EcalFenixAmplitudeFilter::setInput(int input, int fgvb)
{
  if(input>0X3FFFF)
    {
      std::cout<<"ERROR IN INPUT OF AMPLITUDE FILTER"<<std::endl;
      return -1;
    }
  if(inputsAlreadyIn_<5)
    {
      //std::cout << " EcalFenixAmplitudeFilter::setInput inputsAlreadyIn_<5 input " << input << std::endl;
      buffer_[inputsAlreadyIn_]=input;
      fgvbBuffer_[inputsAlreadyIn_]=fgvb;
      inputsAlreadyIn_++;
    }
  else
    {
      for(int i=0; i<4; i++)
      {
         buffer_[i]=buffer_[i+1];
	 //std::cout << " EcalFenixAmplitudeFilter::setInput inputsAlreadyIn buffer " << buffer_[i] << std::endl; 
         fgvbBuffer_[i]=fgvbBuffer_[i+1];
      }
      buffer_[4]=input;
      fgvbBuffer_[4]=fgvb;
    }
  return 1;
}

void EcalFenixAmplitudeFilter::process(std::vector<int> &addout,std::vector<int> &output, std::vector<int> &fgvbIn, std::vector<int> &fgvbOut)
{
  // test

  inputsAlreadyIn_=0;
  for (unsigned int i =0;i<5;i++){
     buffer_[i]=0;//FIXME: 5
     fgvbBuffer_[i]=0;
  }
  
  // test end

  //std::cout << "  EcalFenixAmplitudeFilter::process(std::vector<int> &addout size  " << addout.size() << std::endl;  
  for (unsigned int i =0;i<addout.size();i++){
    
    setInput(addout[i],fgvbIn[i]);
    for (unsigned int i =0;i<5;i++){
      // std::cout << " buffer_ " << buffer_[i];
    }
    //std::cout << "  " << std::endl;
    process();
    output[i]=processedOutput_;
    fgvbOut[i]=processedFgvbOutput_;
  }
  // shift the result by 1!
  for (unsigned int i=0 ; i<(output.size());i++){
    if (i!=output.size()-1){
       output[i]=output[i+1];
       fgvbOut[i] = fgvbOut[i+1];
    }
    else{
      output[i]=0;
      fgvbOut[i] = 0;
    }
  }  
  return;
}

void EcalFenixAmplitudeFilter::process()
{
  //UB FIXME: 5
  processedOutput_ = 0;
  processedFgvbOutput_ = 0;
  if(inputsAlreadyIn_<5) return;
  int output=0;
  int fgvbInt = 0;
  for(int i=0; i<5; i++)
  {

    output+=(weights_[i]*buffer_[i])>>shift_;
    //std::cout << " AmplitudeFilter buffer " << buffer_[i] << " weight " << weights_[i] << " output " << output << std::endl;
    if((fgvbBuffer_[i] == 1 && i == 3) || fgvbInt == 1)
    {
      fgvbInt = 1;
    }
  }
  if(output<0) output=0;
  if(output>0X3FFFF)  output=0X3FFFF;
  processedOutput_ = output;
  processedFgvbOutput_ = fgvbInt;
}

void EcalFenixAmplitudeFilter::setParameters(uint32_t raw,const EcalTPGWeightIdMap * ecaltpgWeightMap,const EcalTPGWeightGroup * ecaltpgWeightGroup)
{
  uint32_t params_[5];
  const EcalTPGGroups::EcalTPGGroupsMap & groupmap = ecaltpgWeightGroup -> getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr it = groupmap.find(raw);
  if (it!=groupmap.end()) {
    uint32_t weightid =(*it).second;
    const EcalTPGWeightIdMap::EcalTPGWeightMap & weightmap = ecaltpgWeightMap -> getMap();
    EcalTPGWeightIdMap::EcalTPGWeightMapItr itw = weightmap.find(weightid);
    (*itw).second.getValues(params_[0],params_[1],params_[2],params_[3],params_[4]);

    // we have to transform negative coded in 7 bits into negative coded in 32 bits
    // maybe this should go into the getValue method??
    //std::cout << "peak flag settings" << std::endl;
    for (int i=0;i<5;++i){
      weights_[i] = (params_[i] & 0x40) ?    (int)( params_[i] | 0xffffffc0) : (int)(params_[i]);

      // Construct the peakFlag for sFGVB processing
      //peakFlag_[i] = ((params_[i] & 0x80) > 0x0) ? 1 : 0;
      //std::cout << " " << params_[i] << std::endl;
      //std::cout << " " << peakFlag_[i] << std::endl;
    }
    //std::cout << std::endl;
  }
  else edm::LogWarning("EcalTPG")<<" could not find EcalTPGGroupsMap entry for "<<raw;
}




