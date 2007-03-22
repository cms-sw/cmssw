#ifndef LayerHitMap_H
#define LayerHitMap_H


#include "RecoTracker/TkHitPairs/interface/TkHitPairsCacheCell.h"
#include "RecoTracker/TkHitPairs/interface/TkHitPairsCellManager.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"
#include <vector>

class LayerHitMapLoop;

class LayerHitMap {

public:
  typedef PixelRecoRange<float> Range;
  typedef ctfseeding::SeedingHit TkHitPairsCachedHit;
  typedef std::vector<TkHitPairsCachedHit>::const_iterator HitIter;

  LayerHitMap() : theCells(0) { }
  LayerHitMap( const DetLayer* layer, const std::vector<ctfseeding::SeedingHit> & hits);
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
  mutable TkHitPairsCellManager * theCells;
  mutable std::vector<ctfseeding::SeedingHit> theHits;
  float theLayerRZmin, theCellDeltaRZ;
  float theLayerPhimin, theCellDeltaPhi; 
  int theNbinsRZ, theNbinsPhi;
};
#endif 
