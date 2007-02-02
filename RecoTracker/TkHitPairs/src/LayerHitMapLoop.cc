#include "RecoTracker/TkHitPairs/interface/LayerHitMap.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

using namespace std;

LayerHitMapLoop::LayerHitMapLoop(const LayerHitMap & map)
  : theMap(map), safeBinRz(true), setStatus(0)
{
  hitItr = theMap.theHits.begin();
  hitEnd = theMap.theHits.end();
  theNextPhi = skip;
}

LayerHitMapLoop::LayerHitMapLoop(
    const LayerHitMap & map, const RangeF & phiRange, const RangeF & rzRange)
  : theMap(map), 
    theRangeRz(rzRange), theRangeRzSafe(rzRange),
    theRangePhi(phiRange), setStatus(0)
{

  hitItr = hitEnd = theMap.theHits.end();
  if (theMap.empty()) { theNextPhi = skip; return; } 

  int irz1 = theMap.idxRz(rzRange.min());
  int irz2 = theMap.idxRz(rzRange.max()); 
  theBinsRz     = RangeI( max( irz1,  0), min( irz2,  theMap.theNbinsRZ-1));
  theBinsRzSafe = RangeI( irz1+1, irz2-1); 

  if (theRangePhi.first >= -M_PI && theRangePhi.second < M_PI) {
    theNextPhi = inRange;
  } else {
    theNextPhi = firstEdge;
    while (theRangePhi.first <  -M_PI) theRangePhi.first  += 2*M_PI;
    while (theRangePhi.second >= M_PI) theRangePhi.second -= 2*M_PI;
  } 

  theBinsPhi  = RangeI (
      int ( (theRangePhi.min() + M_PI)/theMap.theCellDeltaPhi ),
      int ( (theRangePhi.max() + M_PI)/theMap.theCellDeltaPhi ));

  theBinRz = theBinsRz.min()-1; 

}

void LayerHitMapLoop::setSafeRzRange(const RangeF & rzSafe, bool * status)
{ 

  theRangeRzSafe = rzSafe;
  setStatus = status;

  theBinsRzSafe = 
      RangeI( theMap.idxRz(rzSafe.min())+1, theMap.idxRz(rzSafe.max())-1);

} 


const TkHitPairsCachedHit * LayerHitMapLoop::getHit()
{



  while (hitItr < hitEnd) {

    const TkHitPairsCachedHit * hit = &*hitItr++;
    if (safeBinRz)      return hit;
   
    else if (theRangeRzSafe.inside(hit->rOrZ()))
	return hit;
    else if (setStatus && theRangeRz.inside(hit->rOrZ())) {
      *setStatus = false;
      return hit;
    }
  }

  if (nextRange()) return getHit();
  else return 0;
}

bool LayerHitMapLoop::nextRange()
{

  switch (theNextPhi) {
  case inRange:
    if (++theBinRz > theBinsRz.max()) return false; 
    safeBinRz = theBinsRzSafe.inside(theBinRz);
    hitItr = theMap.cell(theBinRz,theBinsPhi.min()).
                 range_from(theRangePhi.min()).min();
    hitEnd = theMap.cell(theBinRz,theBinsPhi.max()).
                 range_upto(theRangePhi.max()).max();
    return true;
  case firstEdge:
    if (++theBinRz > theBinsRz.max()) return false; 
    safeBinRz = theBinsRzSafe.inside(theBinRz);
    hitItr = theMap.cell(theBinRz,theBinsPhi.min()).
                 range_from(theRangePhi.min()).min();
    hitEnd = theMap.cell(theBinRz,theMap.theNbinsPhi-1).
                 range().max();
    theNextPhi = secondEdge;
    return true;
  case secondEdge:
    hitItr = theMap.cell(theBinRz,0).
                 range().min();
    hitEnd = theMap.cell(theBinRz,theBinsPhi.max()).
                 range_upto(theRangePhi.max()).max();
    theNextPhi = firstEdge;
 
    return true;
  default: 
    return false;
  }
}
