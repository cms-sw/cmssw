#ifndef MCTruth_PSimHitMap_h
#define MCTruth_PSimHitMap_h

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <map>

class PSimHitMap
{
public:
  PSimHitMap(const edm::InputTag &);
  
  void fill(const edm::Event & e);

  const edm::PSimHitContainer & hits(int detId) const;

  std::vector<int> detsWithHits() const;

private:
  std::map<int, edm::PSimHitContainer> theMap;
  edm::PSimHitContainer theEmptyContainer;
  edm::InputTag simHitsTag;
};

#endif

