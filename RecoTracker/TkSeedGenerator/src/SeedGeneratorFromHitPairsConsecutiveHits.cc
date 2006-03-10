#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromHitPairsConsecutiveHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
//#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

//using namespace PixelRecoUtilities;

template <class T> T sqr( T t) {return t*t;}

// SeedGeneratorFromHitPairsConsecutiveHits::
// SeedGeneratorFromHitPairsConsecutiveHits(): SeedGeneratorFromHitPairs()
// { 
//   //  theTimer = initTiming("TkSeedGenerator seed construction",1);
// }


void
SeedGeneratorFromHitPairsConsecutiveHits::seeds(TrajectorySeedCollection &output,
						const edm::EventSetup& iSetup,
						const SeedHitPairs & hitPairs, 
						const TrackingRegion& region)
{

  //  TimeMe tm( *theTimer, false);
  
  // SeedGenerator::SeedContainer result;
  //  vector<TrajectorySeed>   result;
  GlobalError vtxerr( sqr(region.originRBound()),
		      0, sqr(region.originRBound()),
		      0, 0, sqr(region.originZBound()));
  SeedHitPairs::const_iterator ip;

  for (ip = hitPairs.begin(); ip != hitPairs.end(); ip++) {
    SeedFromConsecutiveHits *seedfromhits=
      new SeedFromConsecutiveHits( ip->outer(), ip->inner(),
				   region.origin(), vtxerr,iSetup);

    output.push_back(*(seedfromhits->TrajSeed()) );
  }

}
