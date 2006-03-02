#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromHitPairsConsecutiveHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
//#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

//using namespace PixelRecoUtilities;

template <class T> T sqr( T t) {return t*t;}

SeedGeneratorFromHitPairsConsecutiveHits::
    SeedGeneratorFromHitPairsConsecutiveHits() 
{ 
  //  theTimer = initTiming("TkSeedGenerator seed construction",1);
}

TrackingSeedCollection
SeedGeneratorFromHitPairsConsecutiveHits::seeds(
    const SeedHitPairs & hitPairs, const TrackingRegion& region)
{
  //  TimeMe tm( *theTimer, false);

  // SeedGenerator::SeedContainer result;
  TrackingSeedCollection  result;
  GlobalError vtxerr( sqr(region.originRBound()),
                      0, sqr(region.originRBound()),
                      0, 0, sqr(region.originZBound()));
  SeedHitPairs::const_iterator ip;
  for (ip = hitPairs.begin(); ip != hitPairs.end(); ip++) {
    //  try {
      //     BasicTrajectorySeed* seedp =
      //  new SeedFromConsecutiveHits( ip->outer(), ip->inner(),
      //                      region.origin(), vtxerr);
    TrackingSeed productSeed;
    productSeed.addHit(&(ip->outer()));
    productSeed.addHit(&(ip->inner()));
    
    result.push_back( productSeed);
  }
  //   catch( DetLogicError& err) {
  //      cout << "Warning: " << err.what() << endl;
  //    }
  //}
  return result;
}
