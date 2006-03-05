#ifndef SeedGeneratorFromHitPairs_h
#define SeedGeneratorFromHitPairs_h


#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromTrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
//#include "FWCore/Framework/interface/EventSetup.h"
typedef OrderedHitPairs SeedHitPairs;

/** \class SeedGeneratorFromHitPairs
 *
 * Genreates seed from a TrackingRegion using a vector of HitPairs 
 * from provided SeedHitPairGenerator.  The actual seed construction 
 * from hit pair is passed to concrete implementation.
 */

class SeedGeneratorFromHitPairs :  public SeedGeneratorFromTrackingRegion {
public:

  using SeedGeneratorFromTrackingRegion::seeds;

  /// from base class 
  virtual  vector<TrajectorySeed> seeds(const edm::EventSetup& c, const TrackingRegion& region) {
    return seeds(c,pairGenerator()->hitPairs(region), region);
  } 

  /// concrete seed generator should construct hits from hit pairs
virtual    vector<TrajectorySeed> seeds(const edm::EventSetup& c,
      const SeedHitPairs & hitPairs, const TrackingRegion& region) = 0;

  virtual const TrackingRegion * trackingRegion() const = 0;

private:

  virtual HitPairGenerator * pairGenerator() const = 0;

};

#endif
