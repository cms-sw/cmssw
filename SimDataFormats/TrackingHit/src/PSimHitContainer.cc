#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>

using namespace edm;

void PSimHitContainer::insertHits(PSimHitSingleContainer& p){
  _data=p;
}

void PSimHitContainer::insertHit(const PSimHit& p){
  _data.push_back(p);
}

void PSimHitContainer::clear(){
  _data.clear();
}

unsigned int PSimHitContainer::size(){
  return _data.size();
}
