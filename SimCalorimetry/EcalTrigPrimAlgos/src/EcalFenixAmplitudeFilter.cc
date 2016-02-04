#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h>
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

EcalFenixAmplitudeFilter::EcalFenixAmplitudeFilter()
  :inputsAlreadyIn_(0), shift_(6) {
  }

EcalFenixAmplitudeFilter::~EcalFenixAmplitudeFilter(){
}

int EcalFenixAmplitudeFilter::setInput(int input)
{
  if(input>0X3FFFF)
    {
      std::cout<<"ERROR IN INPUT OF AMPLITUDE FILTER"<<std::endl;
      return -1;
    }
  if(inputsAlreadyIn_<5)
    {
      buffer_[inputsAlreadyIn_]=input;
      inputsAlreadyIn_++;
    }
  else
    {
      for(int i=0; i<4; i++) buffer_[i]=buffer_[i+1];
      buffer_[4]=input;
    }
  return 1;
}

void EcalFenixAmplitudeFilter::process(std::vector<int> &addout,std::vector<int> &output)
{
  // test
  inputsAlreadyIn_=0;
  for (unsigned int i =0;i<5;i++) buffer_[i]=0;//FIXME: 5
  
  // test end
  
  for (unsigned int i =0;i<addout.size();i++){
    
    setInput(addout[i]);
    output[i]=process();
  }
  // shift the result by 1!
  for (unsigned int i=0 ; i<(output.size());i++){
    if (i!=output.size()-1) output[i]=output[i+1];
    else output[i]=0;
  }  
  return;
}

int EcalFenixAmplitudeFilter::process()
{
  //UB FIXME: 5
  if(inputsAlreadyIn_<5) return 0;
  int output=0;
  for(int i=0; i<5; i++)
    {
      output+=(weights_[i]*buffer_[i])>>shift_;
    }
  if(output<0) output=0;
  if(output>0X3FFFF)  output=0X3FFFF;
  return output;
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
    for (int i=0;i<5;++i){
      weights_[i] = (params_[i] & 0x40) ?    (int)( params_[i] | 0xffffffc0) : (int)(params_[i]);
    }
  }
  else edm::LogWarning("EcalTPG")<<" could not find EcalTPGGroupsMap entry for "<<raw;
}




