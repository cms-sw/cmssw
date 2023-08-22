#include <SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2SpikeTagger.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalEBPhase2SpikeTagger::EcalEBPhase2SpikeTagger(bool debug)
  : debug_(debug) 
{
}

EcalEBPhase2SpikeTagger::~EcalEBPhase2SpikeTagger(){

}

void EcalEBPhase2SpikeTagger::setParameters(EBDetId detId, 
					 const EcalLiteDTUPedestalsMap*  ecaltpPed, 
                                         const EcalEBPhase2TPGLinearizationConstMap* ecaltpLin, 
					 const EcalTPGCrystalStatus *ecaltpBadX)

{

  if (debug_) std::cout << " EcalEBPhase2SpikeTagger::setParameters " << std::endl;

  EcalLiteDTUPedestalsMap::const_iterator itped = ecaltpPed->getMap().find(detId);
  if (itped != ecaltpPed->end())
    peds_ = &(*itped);
  else
    std::cout << " could not find EcalLiteDTUPedestal entry for " << detId << std::endl;


  const EcalEBPhase2TPGLinearizationConstMap& linMap = ecaltpLin->getMap();
  EcalEBPhase2TPGLinearizationConstMapIterator it = linMap.find(detId.rawId());
  if (it != linMap.end()) {
    linConsts_ = &(*it);
  } else
    std::cout << " could not find EcalEBPhase2TPGLinearizationConstMap entry for " << detId.rawId() << std::endl;


}



bool EcalEBPhase2SpikeTagger::process(const std::vector<int>  &linInput) 
{

  bool isASpike;
  isASpike=false;
  
  if ( debug_) {
    std::cout<< "EcalEBPhase2SpikeTagger::process  linearized digis " << std::endl;
    for (unsigned int i =0; i<linInput.size();i++){
      std::cout <<" "<<std::dec<< linInput[i];
    }
    std::cout<<std::endl;
  }
    
  // dummy for now. It needs the algorythm to be implememted/plugged in here
                                                                                       
  return isASpike;    

}

