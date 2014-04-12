#include "SimTracker/Common/interface/SimHitSelectorFromDB.h"

SimHitSelectorFromDB::SimHitSelectorFromDB():theNewSimHitList(0){}

//std::vector<PSimHit> SimHitSelectorFromDB::getSimHit(std::auto_ptr<MixCollection<PSimHit> >& simhit, 
std::vector<std::pair<const PSimHit*,int> > SimHitSelectorFromDB::getSimHit(std::auto_ptr<MixCollection<PSimHit> >& simhit, 
						     std::map<uint32_t, std::vector<int> >& detId){
  theNewSimHitList.clear();
  int counter =0;
  for(MixCollection<PSimHit>::iterator it = simhit->begin(); it!= simhit->end();it++){
    counter++;
    if(detId.size()!=0){
      uint32_t tkid = (*it).detUnitId();
      if (detId.find(tkid) != detId.end()){
	//	theNewSimHitList.push_back((*it));
	//	std::cout << "Hit in the MAP " << counter << std::endl;
	theNewSimHitList.push_back(std::make_pair(&(*it), counter));
      }
    }else{
      //      theNewSimHitList.push_back((*it));
      //      std::cout << "Hit NOT in the MAP " << counter << std::endl;
      theNewSimHitList.push_back(std::make_pair(&(*it),counter));
    }
  }
  return theNewSimHitList;
}
