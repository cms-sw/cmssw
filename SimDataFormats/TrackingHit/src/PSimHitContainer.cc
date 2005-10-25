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

unsigned int PSimHitContainer::size() const {
  return _data.size();
}

std::vector<PSimHit>::const_iterator PSimHitContainer::begin () const
{
  return _data.begin () ;
}

std::vector<PSimHit>::const_iterator PSimHitContainer::end () const
{
  return _data.end () ;
}

std::vector<PSimHit>::iterator PSimHitContainer::begin () 
{
  return _data.begin () ;
}

std::vector<PSimHit>::iterator PSimHitContainer::end () 
{
  return _data.end () ;
}
