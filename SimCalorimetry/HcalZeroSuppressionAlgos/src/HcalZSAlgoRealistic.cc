#include "SimCalorimetry/HcalZeroSuppressionAlgos/interface/HcalZSAlgoRealistic.h"
#include <iostream>

HcalZSAlgoRealistic::HcalZSAlgoRealistic(bool mp, int levelHB, int levelHE, int levelHO, int levelHF) : 
  HcalZeroSuppressionAlgo(mp),
  thresholdHB_(levelHB),
  thresholdHE_(levelHE),
  thresholdHO_(levelHO),
  thresholdHF_(levelHF)
{
  usingDBvalues = false;
}

HcalZSAlgoRealistic::HcalZSAlgoRealistic(bool mp) : 
  HcalZeroSuppressionAlgo(mp)
{
  thresholdHB_ = -1;
  thresholdHE_ = -1;
  thresholdHO_ = -1;
  thresholdHF_ = -1;
  usingDBvalues = true;
}


  
//template <class DIGI> 
//For HBHE Data Frame 
bool HcalZSAlgoRealistic::keepMe(const HBHEDataFrame& inp, int threshold, uint32_t hbhezsmask) const{
  
  bool keepIt=false;
  //int mask = 999;
  if ((usingDBvalues) && (threshold < 0) && (m_dbService != 0)){
    threshold = (m_dbService->getHcalZSThreshold(inp.id()))->getValue();
  }
  
  //determine the sum of 2 timeslices
  for (int i=0; i< inp.size()-1 && !keepIt; i++) {
    int sum=0;
    
    for (int j=i; j<(i+2); j++){
      sum+=inp[j].adc();
      //pedsum+=pedave;
    }
    if ((hbhezsmask&(1<<i)) !=0) continue; 
    else if (sum>=threshold) keepIt=true;
  }
  return keepIt;
}

//For HO Data Frame 
bool HcalZSAlgoRealistic::keepMe(const HODataFrame& inp, int threshold, uint32_t hozsmask) const{
  
  bool keepIt=false;
  //  int mask = 999;
  if ((usingDBvalues) && (threshold < 0) && (m_dbService != 0)){
    threshold = (m_dbService->getHcalZSThreshold(inp.id()))->getValue();
  }
  
  //determine the sum of 2 timeslices
  for (int i=0; i< inp.size()-1 && !keepIt; i++) {
    int sum=0;
    
    for (int j=i; j<(i+2); j++){
      sum+=inp[j].adc();
      //pedsum+=pedave;
    }
    if ((hozsmask&(1<<i)) !=0) continue; 
    else if (sum>=threshold) keepIt=true;
  }
  return keepIt;
}

//For HF Data Frame 
  bool HcalZSAlgoRealistic::keepMe(const HFDataFrame& inp, int threshold, uint32_t hfzsmask) const{
  
  bool keepIt=false;
  //  int mask = 999;
  if ((usingDBvalues) && (threshold < 0) && (m_dbService != 0)){
    threshold = (m_dbService->getHcalZSThreshold(inp.id()))->getValue();
  }
  
  //determine the sum of 2 timeslices
  for (int i=0; i< inp.size()-1 && !keepIt; i++) {
    int sum=0;
    
    for (int j=i; j<(i+2); j++){
      sum+=inp[j].adc();
      //pedsum+=pedave;
    }
    if ((hfzsmask&(1<<i)) !=0) continue; 
    else if (sum>=threshold) keepIt=true;
  }
  return keepIt;
}


bool HcalZSAlgoRealistic::shouldKeep(const HBHEDataFrame& digi) const{
  // uint32_t hbhezsmask = digi.zsCrossingMask();
  if (digi.id().subdet()==HcalBarrel) 
    return keepMe(digi,thresholdHB_,digi.zsCrossingMask());
  else return keepMe(digi,thresholdHE_,digi.zsCrossingMask());
}  

bool HcalZSAlgoRealistic::shouldKeep(const HODataFrame& digi) const{
  return keepMe(digi,thresholdHO_,digi.zsCrossingMask());
}

bool HcalZSAlgoRealistic::shouldKeep(const HFDataFrame& digi) const{
  return keepMe(digi,thresholdHF_,digi.zsCrossingMask());
}
