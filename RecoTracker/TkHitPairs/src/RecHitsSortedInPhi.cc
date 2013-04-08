#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include <algorithm>

using namespace std;

RecHitsSortedInPhi::RecHitsSortedInPhi( const std::vector<Hit>& hits)
{
  for (std::vector<Hit>::const_iterator i=hits.begin(); i!=hits.end(); i++) {
    theHits.push_back(HitWithPhi(*i));
  }
  sort( theHits.begin(), theHits.end(), HitLessPhi());
}

void RecHitsSortedInPhi::hits( float phiMin, float phiMax, vector<Hit>& result) const
{
  if ( phiMin < phiMax) {
    if ( phiMin < -Geom::fpi()) {
      copyResult( unsafeRange( phiMin + Geom::ftwoPi(), Geom::fpi()), result);
      copyResult( unsafeRange( -Geom::fpi(), phiMax), result);
    }
    else if (phiMax > Geom::pi()) {
      copyResult( unsafeRange( phiMin, Geom::fpi()), result);
      copyResult( unsafeRange( -Geom::fpi(), phiMax-Geom::ftwoPi()), result);
    }
    else {
      copyResult( unsafeRange( phiMin, phiMax), result);
    }
  }
  else {
    copyResult( unsafeRange( phiMin, Geom::fpi()), result);
    copyResult( unsafeRange( -Geom::fpi(), phiMax), result);
  }
}

vector<RecHitsSortedInPhi::Hit> RecHitsSortedInPhi::hits( float phiMin, float phiMax) const
{
  vector<Hit> result;
  hits( phiMin, phiMax, result);
  return result;
}

RecHitsSortedInPhi::Range 
RecHitsSortedInPhi::unsafeRange( float phiMin, float phiMax) const
{
  return Range( 
      lower_bound( theHits.begin(), theHits.end(), HitWithPhi(phiMin), HitLessPhi()),
	upper_bound( theHits.begin(), theHits.end(), HitWithPhi(phiMax), HitLessPhi()));
}
