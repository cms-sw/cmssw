#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromRegionHits_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"


#include <memory>

class TrackingRegion;
 
namespace edm { class Event; class EventSetup; }

class SeedGeneratorFromRegionHits {
public:

  template <typename GEN>
  SeedGeneratorFromRegionHits(GEN aGenerator): SeedGeneratorFromRegionHits(std::move(aGenerator), nullptr, nullptr) {}

  template <typename GEN, typename COMP>
  SeedGeneratorFromRegionHits(GEN aGenerator, COMP aComparitor): SeedGeneratorFromRegionHits(std::move(aGenerator), std::move(aComparitor), nullptr) {}

  template <typename GEN, typename COMP, typename CREA>
  SeedGeneratorFromRegionHits(GEN aGenerator, COMP aComparitor, CREA aSeedCreator):
    theHitsGenerator{std::move(aGenerator)}, theComparitor{std::move(aComparitor)}, theSeedCreator{std::move(aSeedCreator)}
  {}



  // make job
  void run(TrajectorySeedCollection & seedCollection, const TrackingRegion & region, 
	   const edm::Event& ev, const edm::EventSetup& es);
 
private:
  std::unique_ptr<OrderedHitsGenerator> theHitsGenerator;
  std::unique_ptr<SeedComparitor> theComparitor;
  std::unique_ptr<SeedCreator> theSeedCreator;
};
#endif 
