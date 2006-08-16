#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
//#include "ClassReuse/GeomVector/interface/Pi.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"
#include <algorithm>

using namespace std;

RecHitsSortedInPhi::RecHitsSortedInPhi( const vector<const TrackingRecHit*>& hits,
					const TrackerGeometry* theGeometry) 
{
  //initTiming();
  // TimeMe tm1( *theFillTimer, false);
  for (vector<const TrackingRecHit*>::const_iterator i=hits.begin(); i!=hits.end(); i++) {
    theHits.push_back( Hit(*i,theGeometry));
  }
  //TimeMe tm2( *theSortTimer, false);
  sort( theHits.begin(), theHits.end(), HitLessPhi());
}

void RecHitsSortedInPhi::hits( float phiMin, float phiMax, 
			       vector<const TrackingRecHit*>& result) const
{
  if ( phiMin < phiMax) {
    if ( phiMin < -Geom::pi()) {
      copyResult( unsafeRange( phiMin + Geom::twoPi(), Geom::pi()), result);
      copyResult( unsafeRange( -Geom::pi(), phiMax), result);
    }
    else if (phiMax > Geom::pi()) {
      copyResult( unsafeRange( phiMin, Geom::pi()), result);
      copyResult( unsafeRange( -Geom::pi(), phiMax-Geom::twoPi()), result);
    }
    else {
      copyResult( unsafeRange( phiMin, phiMax), result);
    }
  }
  else {
    copyResult( unsafeRange( phiMin, Geom::pi()), result);
    copyResult( unsafeRange( -Geom::pi(), phiMax), result);
  }
}

vector<const TrackingRecHit*> RecHitsSortedInPhi::hits( float phiMin, float phiMax) const
{
  vector<const TrackingRecHit*> result;
  hits( phiMin, phiMax, result);
  return result;
}

RecHitsSortedInPhi::Range 
RecHitsSortedInPhi::unsafeRange( float phiMin, float phiMax) const
{
  //TimeMe tm1( *theSearchTimer, false);
  return Range( lower_bound( theHits.begin(), theHits.end(), Hit(phiMin), HitLessPhi()),
		upper_bound( theHits.begin(), theHits.end(), Hit(phiMax), HitLessPhi()));
}

/*
void RecHitsSortedInPhi::initTiming() 
{
  if (timingDone) return;

  TimingReport& tr(*TimingReport::current());

  theFillTimer   =   &tr["RecHitsSortedInPhi construct"];
  theSortTimer   =   &tr["RecHitsSortedInPhi sort"];
  theCopyResultTimer =   &tr["RecHitsSortedInPhi result copy"];
  theSearchTimer =   &tr["RecHitsSortedInPhi binary search"];

  static bool detailedTiming =
    SimpleConfigurable<bool>(false,"RecHitsSortedInPhi:detailedTiming").value();

  if (!detailedTiming) {
    theFillTimer->switchOn(false);
    theSortTimer->switchOn(false);
    theCopyResultTimer->switchOn(false);
    theSearchTimer->switchOn(false);
  }
  timingDone = true;
}
*/


//TimingReport::Item* RecHitsSortedInPhi::theFillTimer = 0;
//TimingReport::Item* RecHitsSortedInPhi::theSortTimer = 0;
//TimingReport::Item* RecHitsSortedInPhi::theCopyResultTimer = 0;
//TimingReport::Item* RecHitsSortedInPhi::theSearchTimer = 0;
//bool RecHitsSortedInPhi::timingDone = false;
