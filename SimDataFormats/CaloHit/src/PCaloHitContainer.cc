
#include <SimDataFormats/SimCaloHit/interface/PCaloHitContainer.h>

using namespace edm;
void PCaloHitContainer::insertHits(std::string name, PCaloHitSingleContainer& p){
  _data[name]=p;
}
void PCaloHitContainer::insertHit(std::string name, PCaloHit& p){
  _data[name].push_back(p);
}
