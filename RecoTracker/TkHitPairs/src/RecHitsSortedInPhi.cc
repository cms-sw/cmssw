#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

#include <algorithm>
#include<cassert>



RecHitsSortedInPhi::RecHitsSortedInPhi(const std::vector<Hit>& hits, GlobalPoint const & origin, DetLayer const * il) :
  layer(il),
  isBarrel(il->isBarrel()),
  x(hits.size()),y(hits.size()),z(hits.size()),drphi(hits.size()),
  u(hits.size()),v(hits.size()),du(hits.size()),dv(hits.size()),
  lphi(hits.size())
{

  // standard region have origin as 0,0,z (not true!!!!0
  // cosmic region never used here
  // assert(origin.x()==0 && origin.y()==0);

  for (std::vector<Hit>::const_iterator i=hits.begin(); i!=hits.end(); i++) {
    theHits.push_back(HitWithPhi(*i));
  }
  std::sort( theHits.begin(), theHits.end(), HitLessPhi());

  for (unsigned int i=0; i!=theHits.size(); ++i) {
    auto const & h = *theHits[i].hit();
    auto const & gs = static_cast<BaseTrackerRecHit const &>(h).globalState();
    auto loc = gs.position-origin.basicVector();
    float lr = loc.perp();
    // float lr = gs.position.perp();
    float lz = gs.position.z();
    float dr = gs.errorR;
    float dz = gs.errorZ;
    // r[i] = gs.position.perp();
    // phi[i] = gs.position.barePhi();
    x[i] = gs.position.x();
    y[i] = gs.position.y();
    z[i] = lz;
    drphi[i] = gs.errorRPhi;
    u[i] = isBarrel ? lr : lz;
    v[i] = isBarrel ? lz : lr;
    du[i] = isBarrel ? dr : dz;
    dv[i] = isBarrel ? dz : dr;
    lphi[i] = loc.barePhi();
  }
  
}


RecHitsSortedInPhi::DoubleRange RecHitsSortedInPhi::doubleRange(float phiMin, float phiMax) const {
  Range r1,r2;
  if ( phiMin < phiMax) {
    if ( phiMin < -Geom::fpi()) {
      r1 = unsafeRange( phiMin + Geom::ftwoPi(), Geom::fpi());
      r2 = unsafeRange( -Geom::fpi(), phiMax);
    }
    else if (phiMax > Geom::pi()) {
     r1 = unsafeRange( phiMin, Geom::fpi());
     r2 = unsafeRange( -Geom::fpi(), phiMax-Geom::ftwoPi());
    }
    else {
      r1 = unsafeRange( phiMin, phiMax);
      r2 = Range(theHits.begin(),theHits.begin());
    }
  }
  else {
    r1 =unsafeRange( phiMin, Geom::fpi());
    r2 =unsafeRange( -Geom::fpi(), phiMax);
  }

  return (DoubleRange){{int(r1.first-theHits.begin()),int(r1.second-theHits.begin())
	,int(r2.first-theHits.begin()),int(r2.second-theHits.begin())}};
}


void RecHitsSortedInPhi::hits( float phiMin, float phiMax, std::vector<Hit>& result) const
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

std::vector<RecHitsSortedInPhi::Hit> RecHitsSortedInPhi::hits( float phiMin, float phiMax) const
{
  std::vector<Hit> result;
  hits( phiMin, phiMax, result);
  return result;
}

RecHitsSortedInPhi::Range 
RecHitsSortedInPhi::unsafeRange( float phiMin, float phiMax) const
{
  auto low = std::lower_bound( theHits.begin(), theHits.end(), HitWithPhi(phiMin), HitLessPhi());
  return Range( low,
	       std::upper_bound(low, theHits.end(), HitWithPhi(phiMax), HitLessPhi()));
}
