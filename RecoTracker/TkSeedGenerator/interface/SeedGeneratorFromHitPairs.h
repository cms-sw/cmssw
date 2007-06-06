#ifndef SeedGeneratorFromHitPairs_h
#define SeedGeneratorFromHitPairs_h

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
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

  SeedGeneratorFromHitPairs(const edm::ParameterSet& conf):  SeedGeneratorFromTrackingRegion(conf){}

  /// from base class 
  virtual  void seeds(TrajectorySeedCollection &output,const edm::EventSetup& c, const TrackingRegion& region) {
    OrderedHitPairs pairs;
    pairGenerator()->hitPairs(region,pairs,c); 
    return seeds(output,c,pairs, region);
  } 

  /// concrete seed generator should construct hits from hit pairs
    virtual   void seeds(TrajectorySeedCollection &output,
			 const edm::EventSetup& c,
			 const SeedHitPairs & hitPairs, 
			 const TrackingRegion& region){};

  virtual const TrackingRegion * trackingRegion() const = 0;

private:

  virtual HitPairGenerator * pairGenerator() const = 0;

};

#endif
