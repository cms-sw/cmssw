#ifndef SeedGeneratorFromHitPairsConsecutiveHits_H
#define SeedGeneratorFromHitPairsConsecutiveHits_H

#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromHitPairs.h"
//#include "Utilities/Notification/interface/TimingReport.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/EventSetup.h"
/** \class SeedGeneratorFromHitPairsConsecutiveHits
 * Specialises the SeedGeneratorFromHitPairs for the case of consecutive hits.
 */

class SeedGeneratorFromHitPairsConsecutiveHits 
        : public SeedGeneratorFromHitPairs { 
public:


  using SeedGeneratorFromHitPairs::seeds;

  vector<TrajectorySeed> seeds(const edm::EventSetup& c,
      const SeedHitPairs & hitPairs, const TrackingRegion& region);

private:
  /// from base class
  virtual HitPairGenerator *  pairGenerator() const = 0;

  //  TimingReport::Item * theTimer;

};

#endif
