#ifndef LayerHitMap_H
#define LayerHitMap_H

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "RecoTracker/TkHitPairs/interface/TkHitPairsCacheCell.h"
#include "RecoTracker/TkHitPairs/interface/TkHitPairsCellManager.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
class LayerHitMapLoop;

class LayerHitMap {

public:
  typedef PixelRecoRange<float> Range;
  typedef vector<TkHitPairsCachedHit>::const_iterator HitIter;
  LayerHitMap() : theCells(0) { }
  //  LayerHitMap(const vector<SiPixelRecHit>& hits);
  // LayerHitMap(SiPixelRecHitCollection ): theCells(0);
  LayerHitMap(const LayerWithHits& );
  LayerHitMap(const LayerHitMap & lhm); 
  ~LayerHitMap() { delete theCells; }

  LayerHitMapLoop loop() const;
  LayerHitMapLoop loop(const Range & phiRange, const Range & rzRange) const;

  bool empty() const { return theHits.empty(); }

private:
  void initCells() const;
  int idxRz(float rz) const { return int((rz-theLayerRZmin)/theCellDeltaRZ); }
  TkHitPairsCacheCell & cell(int idx_rz, int idx_phi) const
  { return (*theCells)(idx_rz,idx_phi); } 
 
  friend class LayerHitMapLoop;

private:
  //  edm::ParameterSet conf_;
   mutable TkHitPairsCellManager * theCells;
   //TkHitPairsCellManager * theCells;
  mutable vector<TkHitPairsCachedHit> theHits;
  float theLayerRZmin, theCellDeltaRZ;
  float theLayerPhimin, theCellDeltaPhi; 
  int theNbinsRZ, theNbinsPhi;
};
#endif 
