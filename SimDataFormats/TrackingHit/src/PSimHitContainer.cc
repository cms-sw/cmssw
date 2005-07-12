#include <SimDataFormats/SimTkHit/interface/PSimHitContainer.h>

using namespace edm;

void PSimHitContainer::insertHits(std::string name, PSimHitSingleContainer& p){
  _data[name]=p;
}

void PSimHitContainer::insertHit(std::string name, const PSimHit& p){
  _data[name].push_back(p);
}

void PSimHitContainer::clear(std::string name){
  _data[name].clear();
}

unsigned int PSimHitContainer::size(std::string name){
  return _data[name].size();
}
