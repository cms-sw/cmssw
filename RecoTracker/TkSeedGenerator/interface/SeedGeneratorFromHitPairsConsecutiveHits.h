#ifndef SeedGeneratorFromHitPairsConsecutiveHits_H
#define SeedGeneratorFromHitPairsConsecutiveHits_H

#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromHitPairs.h"
//#include "Utilities/Notification/interface/TimingReport.h"
#include "DataFormats/TrackingSeed/interface/TrackingSeed.h"
#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"
/** \class SeedGeneratorFromHitPairsConsecutiveHits
 * Specialises the SeedGeneratorFromHitPairs for the case of consecutive hits.
 */

class SeedGeneratorFromHitPairsConsecutiveHits 
        : public SeedGeneratorFromHitPairs { 
public:
  SeedGeneratorFromHitPairsConsecutiveHits();

  using SeedGeneratorFromHitPairs::seeds;

  virtual TrackingSeedCollection seeds(
      const SeedHitPairs & hitPairs, const TrackingRegion& region);

private:
  /// from base class
  virtual HitPairGenerator *  pairGenerator() const = 0;

  //  TimingReport::Item * theTimer;

};

#endif
