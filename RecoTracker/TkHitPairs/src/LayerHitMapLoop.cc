#include "RecoTracker/TkHitPairs/interface/LayerHitMap.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"


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
  std::cout<<"l3"<<std::endl;
  hitItr = hitEnd = theMap.theHits.end();
  if (theMap.empty()) { theNextPhi = skip; return; } 
  std::cout<<"l4"<<std::endl;
  int irz1 = theMap.idxRz(rzRange.min());
  int irz2 = theMap.idxRz(rzRange.max()); 
  theBinsRz     = RangeI( max( irz1,  0), min( irz2,  theMap.theNbinsRZ-1));
  theBinsRzSafe = RangeI( irz1+1, irz2-1); 
  std::cout<<"l5"<<std::endl;
  if (theRangePhi.first >= -M_PI && theRangePhi.second < M_PI) {
    theNextPhi = inRange;
  } else {
    theNextPhi = firstEdge;
    while (theRangePhi.first <  -M_PI) theRangePhi.first  += 2*M_PI;
    while (theRangePhi.second >= M_PI) theRangePhi.second -= 2*M_PI;
  } 
  std::cout<<"l6"<<std::endl;
  theBinsPhi  = RangeI (
      int ( (theRangePhi.min() + M_PI)/theMap.theCellDeltaPhi ),
      int ( (theRangePhi.max() + M_PI)/theMap.theCellDeltaPhi ));
  std::cout<<"l7"<<std::endl;
  theBinRz = theBinsRz.min()-1; 
}

void LayerHitMapLoop::setSafeRzRange(const RangeF & rzSafe, bool * status)
{ 
	cout<<"l8"<<endl;
  theRangeRzSafe = rzSafe;
  setStatus = status;
	cout<<"l9"<<endl;
  theBinsRzSafe = 
      RangeI( theMap.idxRz(rzSafe.min())+1, theMap.idxRz(rzSafe.max())-1);
	cout<<"l10"<<endl;
} 


const TkHitPairsCachedHit * LayerHitMapLoop::getHit()
{
	cout<<"l11"<<endl;
  while (hitItr < hitEnd) {
	cout<<"l12"<<endl;
    const TkHitPairsCachedHit * hit = &*hitItr++;
    if (safeBinRz) {
      cout<<"s1"<<endl;
      return hit;
    }
    else if (theRangeRzSafe.inside(hit->rOrZ())) {
	cout<<"s2"<<endl;
	return hit;}
    else if (setStatus && theRangeRz.inside(hit->rOrZ())) {
      *setStatus = false;
 	cout<<"s3"<<endl;
      return hit;
    }
  }
  cout<<"l13 "<<nextRange()<<endl;
  if (nextRange()) return getHit();
  else return 0;
}

bool LayerHitMapLoop::nextRange()
{
  //  cout<<"l14"<<endl;
  switch (theNextPhi) {
  case inRange:
    //  cout<<"l15"<<endl;
    if (++theBinRz > theBinsRz.max()) return false; 
    safeBinRz = theBinsRzSafe.inside(theBinRz);
    hitItr = theMap.cell(theBinRz,theBinsPhi.min()).
                 range_from(theRangePhi.min()).min();
    hitEnd = theMap.cell(theBinRz,theBinsPhi.max()).
                 range_upto(theRangePhi.max()).max();
    return true;
  case firstEdge:
    // cout<<"l16"<<endl;
    if (++theBinRz > theBinsRz.max()) return false; 
    safeBinRz = theBinsRzSafe.inside(theBinRz);
    hitItr = theMap.cell(theBinRz,theBinsPhi.min()).
                 range_from(theRangePhi.min()).min();
    hitEnd = theMap.cell(theBinRz,theMap.theNbinsPhi-1).
                 range().max();
    theNextPhi = secondEdge;
    return true;
  case secondEdge:
    // cout<<"l17"<<endl;
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
