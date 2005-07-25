
#include <SimDataFormats/CaloHit/interface/PCaloHitContainer.h>

using namespace edm;
void PCaloHitContainer::insertHits(PCaloHitSingleContainer& p){
  _data=p;
}
void PCaloHitContainer::insertHit(PCaloHit& p){
  _data.push_back(p);
}
