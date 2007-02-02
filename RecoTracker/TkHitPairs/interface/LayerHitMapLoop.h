#ifndef LayerHitMapLoop_H
#define LayerHitMapLoop_H

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkHitPairs/interface/TkHitPairsCacheCell.h"

class LayerHitMap;


class LayerHitMapLoop {
public:
  typedef std::vector<TkHitPairsCachedHit>::const_iterator HitIter;
  typedef PixelRecoRange<HitIter> HitIterRange;
  typedef PixelRecoRange<float> RangeF;
  typedef PixelRecoRange<int> RangeI;
  enum PhiOption { inRange, firstEdge, secondEdge, skip };

  LayerHitMapLoop( const LayerHitMap & map);
  LayerHitMapLoop( const LayerHitMap & map,  
                   const RangeF & phiRange, const RangeF & rzRange);

  const TkHitPairsCachedHit * getHit();
  void setSafeRzRange(const RangeF & rzSafeRange, bool * status);

private:
  inline bool nextRange();

private:
  const LayerHitMap & theMap;

  RangeI  theBinsRz,  theBinsRzSafe,  theBinsPhi;
  RangeF theRangeRz,  theRangeRzSafe, theRangePhi;
  int      theBinRz; 
  PhiOption theNextPhi; 

  bool safeBinRz, *setStatus; 
  HitIter hitItr, hitEnd;
};

#endif
