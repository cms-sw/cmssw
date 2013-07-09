#include "HcalZSAlgoRealistic.h"
#include <iostream>

HcalZSAlgoRealistic::HcalZSAlgoRealistic(bool mp, int levelHB, int levelHE, int levelHO, int levelHF, std::pair<int,int> HBsearchTS, std::pair<int,int> HEsearchTS, std::pair<int,int> HOsearchTS, std::pair<int,int> HFsearchTS) : 
  HcalZeroSuppressionAlgo(mp),
  thresholdHB_(levelHB),
  thresholdHE_(levelHE),
  thresholdHO_(levelHO),
  thresholdHF_(levelHF),
  HBsearchTS_(HBsearchTS),
  HEsearchTS_(HEsearchTS),
  HOsearchTS_(HOsearchTS),
  HFsearchTS_(HFsearchTS)
{
  usingDBvalues = false;
}

HcalZSAlgoRealistic::HcalZSAlgoRealistic(bool mp, std::pair<int,int> HBsearchTS, std::pair<int,int> HEsearchTS, std::pair<int,int> HOsearchTS, std::pair<int,int> HFsearchTS) : 
  HcalZeroSuppressionAlgo(mp),
  HBsearchTS_(HBsearchTS),
  HEsearchTS_(HEsearchTS),
  HOsearchTS_(HOsearchTS),
  HFsearchTS_(HFsearchTS)
{
  thresholdHB_ = -1;
  thresholdHE_ = -1;
  thresholdHO_ = -1;
  thresholdHF_ = -1;
  usingDBvalues = true;

}


  
//template <class DIGI> 
//For HBHE Data Frame 
bool HcalZSAlgoRealistic::keepMe(const HBHEDataFrame& inp, int start, int finish, int threshold, uint32_t hbhezsmask) const{
  
  bool keepIt=false;
  //int mask = 999;
  if ((usingDBvalues) && (threshold < 0) && (m_dbService != 0)){
    threshold = (m_dbService->getHcalZSThreshold(inp.id()))->getValue();
  }

  //determine the sum of 2 timeslices
  for (int i = start; i < finish && !keepIt; i++) {
    int sum=0;
    
    for (int j = i; j < (i+2); j++){
      sum+=inp[j].adc();
      //pedsum+=pedave;
    }
    if ((hbhezsmask&(1<<i)) !=0) continue; 
    else if (sum>=threshold) keepIt=true;
  }
  return keepIt;
}

//For HO Data Frame 
bool HcalZSAlgoRealistic::keepMe(const HODataFrame& inp, int start, int finish, int threshold, uint32_t hozsmask) const{
  
  bool keepIt=false;
  //  int mask = 999;
  if ((usingDBvalues) && (threshold < 0) && (m_dbService != 0)){
    threshold = (m_dbService->getHcalZSThreshold(inp.id()))->getValue();
  }

  //determine the sum of 2 timeslices
  for (int i = start; i < finish && !keepIt; i++) {
    int sum=0;
    
    for (int j = i; j < (i+2); j++){
      sum+=inp[j].adc();
      //pedsum+=pedave;
    }
    if ((hozsmask&(1<<i)) !=0) continue; 
    else if (sum>=threshold) keepIt=true;
  }
  return keepIt;
}

//For HF Data Frame 
bool HcalZSAlgoRealistic::keepMe(const HFDataFrame& inp, int start, int finish, int threshold, uint32_t hfzsmask) const{
  
  bool keepIt=false;
  //  int mask = 999;
  if ((usingDBvalues) && (threshold < 0) && (m_dbService != 0)){
    threshold = (m_dbService->getHcalZSThreshold(inp.id()))->getValue();
  }
  
  //determine the sum of 2 timeslices
  for (int i = start; i < finish && !keepIt; i++) {
    int sum=0;
    
    for (int j = i; j < (i+2); j++){
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
  if (digi.id().subdet()==HcalBarrel) {
    
    int start  = std::max(0,HBsearchTS_.first);
    int finish = std::min(digi.size()-1,HBsearchTS_.second);

    /*
    std::cout << " HBsearchTS_ : " <<  HBsearchTS_.first 
	      << ", " << HBsearchTS_.second << std::endl;
    std::cout << " HB start, finish = " << start << ", "
	      << finish << std::endl;
    */

    return keepMe(digi,start,finish,thresholdHB_,digi.zsCrossingMask());


  }
  else {

    int start  = std::max(0,HEsearchTS_.first);
    int finish = std::min(digi.size()-1,HEsearchTS_.second);
    return keepMe(digi,start,finish,thresholdHE_,digi.zsCrossingMask());

  }
}  

bool HcalZSAlgoRealistic::shouldKeep(const HODataFrame& digi) const{

  int start  = std::max(0,HOsearchTS_.first);
  int finish = std::min(digi.size()-1,HOsearchTS_.second);
  return keepMe(digi,start,finish,thresholdHO_,digi.zsCrossingMask());
}

bool HcalZSAlgoRealistic::shouldKeep(const HFDataFrame& digi) const{

  int start  = std::max(0,HFsearchTS_.first);
  int finish = std::min(digi.size()-1,HFsearchTS_.second);
  return keepMe(digi,start,finish,thresholdHF_,digi.zsCrossingMask());
}
