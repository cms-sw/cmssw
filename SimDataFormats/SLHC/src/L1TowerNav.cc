#include "SimDataFormats/SLHC/interface/L1TowerNav.h"

#include <cmath>

int L1TowerNav::getOffsetIEta(int iEta,int offset)
{
  int newIEta=iEta+offset;
  
  //first check for a +ve to -ve transistion (there is no iEta==0)
  if( (newIEta)/std::abs(newIEta) != iEta/std::abs(iEta) ) {
    if(offset>0) newIEta++; //-ve to +ve 
    else newIEta--; //+ve to -ve
  }
  return newIEta;
  //now bounds check
  //if(std::abs(newIEta)<=kIEtaAbsHFMax) return newIEta;
  // else return kNullIEta;
  
}

int L1TowerNav::getOffsetIPhi(int iEta,int iPhi,int offset)
{
  if(std::abs(iEta)<=kIEtaAbsHEMax) return getOffsetIPhiHBHE(iPhi,offset);
  else return getOffsetIPhiHF(iPhi,offset);
  
}

int L1TowerNav::getOffsetIPhiHBHE(int iPhi,int offset)
{
  if(iPhi<=0 || iPhi>kIPhiMax) return kNullIPhi;

  offset=offset%kIPhiMax;
  iPhi+=offset;

  //technically as we bounds check earilier and ensure the offset is 0 to 71, we can only be a max of 71 out
  //however leave as is incase things change, for example we may wish to allow out of bounds phi
  while(iPhi<=0) iPhi+=kIPhiMax;
  while(iPhi>kIPhiMax) iPhi-=kIPhiMax;


  return iPhi;
}

int L1TowerNav::getOffsetIPhiHF(int iPhi,int offset)
{
  if(iPhi<=0 || iPhi>kIPhiMax) return kNullIPhi;
  if((iPhi%kHFIPhiScale)!=1) return kNullIPhi; //invalid HF iPhi


  offset=(offset*kHFIPhiScale)%kIPhiMax;
  iPhi+=offset;
 
  //technically as we bounds check earilier and ensure the offset is 0 to 71, we can only be a max of 71 out
  //however leave as is incase things change, for example we may wish to allow out of bounds phi
  while(iPhi<=0) iPhi+=kIPhiMax;
  while(iPhi>kIPhiMax) iPhi-=kIPhiMax;

  return iPhi;
}
