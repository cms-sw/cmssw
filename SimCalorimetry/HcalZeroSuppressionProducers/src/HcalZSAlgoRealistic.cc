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
  
template <class Digi>
bool HcalZSAlgoRealistic::keepMe(const Digi& inp, int start, int finish, int threshold, uint32_t zsmask) const{
  if ((usingDBvalues) && (threshold < 0) && (m_dbService != 0)){
    threshold = (m_dbService->getHcalZSThreshold(inp.id()))->getValue();
  }

  //determine the sum of 2 timeslices
  for (int i = start; i < finish; i++) {
    if ((zsmask&(1<<i)) !=0) continue;
    if ((inp[i].adc()+inp[i+1].adc())>=threshold) return true;
  }
  return false;
}

//zs mask not used for QIE10,11

template<>
bool HcalZSAlgoRealistic::keepMe<QIE10DataFrame>(const QIE10DataFrame& inp, int start, int finish, int threshold, uint32_t zsmask) const{
  if ((usingDBvalues) && (threshold < 0) && (m_dbService != 0)){
    threshold = (m_dbService->getHcalZSThreshold(inp.id()))->getValue();
  }
  
  //determine the sum of 2 timeslices
  for (int i = start; i < finish; i++) {
    if ((inp[i].adc()+inp[i+1].adc())>=threshold) return true;
  }
  return false;
}

template<>
bool HcalZSAlgoRealistic::keepMe<QIE11DataFrame>(const QIE11DataFrame& inp, int start, int finish, int threshold, uint32_t zsmask) const{
  if ((usingDBvalues) && (threshold < 0) && (m_dbService != 0)){
    threshold = (m_dbService->getHcalZSThreshold(inp.id()))->getValue();
  }
  
  //determine the sum of 2 timeslices
  for (int i = start; i < finish; i++) {
    if ((inp[i].adc()+inp[i+1].adc())>=threshold) return true;
  }
  return false;
}

bool HcalZSAlgoRealistic::shouldKeep(const HBHEDataFrame& digi) const{
  if (digi.id().subdet()==HcalBarrel) {
    int start  = std::max(0,HBsearchTS_.first);
    int finish = std::min(digi.size()-1,HBsearchTS_.second);
    return keepMe(digi,start,finish,thresholdHB_,digi.zsCrossingMask());
  } else {
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

bool HcalZSAlgoRealistic::shouldKeep(const QIE10DataFrame& digi) const{
  int start  = std::max(0,HFsearchTS_.first);
  int finish = std::min((int)digi.samples()-1,HFsearchTS_.second);
  return keepMe(digi,start,finish,thresholdHF_,0);
}

bool HcalZSAlgoRealistic::shouldKeep(const QIE11DataFrame& digi) const{
  HcalDetId hid(digi.id());
  if (hid.subdet()==HcalBarrel) {
    int start  = std::max(0,HBsearchTS_.first);
    int finish = std::min(digi.samples()-1,HBsearchTS_.second);
    return keepMe(digi,start,finish,thresholdHB_,0);
  } else {
    int start  = std::max(0,HEsearchTS_.first);
    int finish = std::min(digi.samples()-1,HEsearchTS_.second);
    return keepMe(digi,start,finish,thresholdHE_,0);
  }
}
