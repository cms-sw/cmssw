#ifndef LayerHitMap_H
#define LayerHitMap_H


#include "RecoTracker/TkHitPairs/interface/TkHitPairsCacheCell.h"
#include "RecoTracker/TkHitPairs/interface/TkHitPairsCellManager.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
class LayerHitMapLoop;

class LayerHitMap {

public:
  typedef PixelRecoRange<float> Range;
  typedef std::vector<TkHitPairsCachedHit>::const_iterator HitIter;
  LayerHitMap() : theCells(0) { }
  LayerHitMap(const LayerWithHits*  ,  const edm::EventSetup& iSetup);
  LayerHitMap(const LayerHitMap & lhm,  const edm::EventSetup& iSetup); 
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
  mutable std::vector<TkHitPairsCachedHit> theHits;
  float theLayerRZmin, theCellDeltaRZ;
  float theLayerPhimin, theCellDeltaPhi; 
  int theNbinsRZ, theNbinsPhi;
};
#endif 
