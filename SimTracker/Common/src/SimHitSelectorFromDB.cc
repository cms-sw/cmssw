#include "SimTracker/Common/interface/SimHitSelectorFromDB.h"

SimHitSelectorFromDB::SimHitSelectorFromDB():theNewSimHitList(0){}

std::vector<PSimHit> SimHitSelectorFromDB::getSimHit(std::auto_ptr<MixCollection<PSimHit> >& simhit, 
						     std::map<uint32_t, std::vector<int> >& detId){
  theNewSimHitList.clear();
  for(MixCollection<PSimHit>::iterator it = simhit->begin(); it!= simhit->end();it++){
    if(detId.size()!=0){
      uint32_t tkid = (*it).detUnitId();
      if (detId.find(tkid) != detId.end()){
	theNewSimHitList.push_back((*it));
      }
    }else{
      theNewSimHitList.push_back((*it));
    }
  }
  return theNewSimHitList;
}
