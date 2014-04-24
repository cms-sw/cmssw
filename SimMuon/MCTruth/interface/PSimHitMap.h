#ifndef MCTruth_PSimHitMap_h
#define MCTruth_PSimHitMap_h

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include <map>

class PSimHitMap
{
 public:
  PSimHitMap(const edm::InputTag & iT, edm::ConsumesCollector && iC)
    :  theMap(), theEmptyContainer()
  {
    sh_token = iC.consumes<CrossingFrame<PSimHit> >(iT);
  }

  void fill(const edm::Event & e);

  const edm::PSimHitContainer & hits(int detId) const;

  std::vector<int> detsWithHits() const;

 private:
  std::map<int, edm::PSimHitContainer> theMap;
  edm::PSimHitContainer theEmptyContainer;
  //  edm::InputTag simHitsTag;
  edm::EDGetTokenT<CrossingFrame<PSimHit> > sh_token;
};

#endif
