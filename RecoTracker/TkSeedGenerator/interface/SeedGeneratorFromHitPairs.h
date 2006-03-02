#ifndef SeedGeneratorFromHitPairs_h
#define SeedGeneratorFromHitPairs_h


#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "DataFormats/TrackingSeed/interface/TrackingSeed.h"
#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromTrackingRegion.h"

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
  virtual  TrackingSeedCollection seeds( const TrackingRegion& region) {
     return seeds(pairGenerator()->hitPairs(region), region);
  } 

  /// concrete seed generator should construct hits from hit pairs
  virtual TrackingSeedCollection seeds(
      const SeedHitPairs & hitPairs, const TrackingRegion& region) = 0;

  virtual const TrackingRegion * trackingRegion() const = 0;

private:

  virtual HitPairGenerator * pairGenerator() const = 0;
  
};

#endif
