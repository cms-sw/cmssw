#ifndef MultiHitGeneratorFromChi2_H
#define MultiHitGeneratorFromChi2_H

/** A MultiHitGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include "CombinedMultiHitGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include <utility>
#include <vector>


class MultiHitGeneratorFromChi2 : public MultiHitGeneratorFromPairAndLayers {

typedef CombinedMultiHitGenerator::LayerCacheType       LayerCacheType;

public:
  MultiHitGeneratorFromChi2( const edm::ParameterSet& cfg);

  virtual ~MultiHitGeneratorFromChi2() { delete thePairGenerator; }

  virtual void init( const HitPairGenerator & pairs,
      const std::vector<ctfseeding::SeedingLayer> & layers, LayerCacheType* layerCache);

  virtual void hitSets( const TrackingRegion& region, OrderedMultiHits & trs, 
      const edm::Event & ev, const edm::EventSetup& es);

  const HitPairGenerator & pairGenerator() const { return *thePairGenerator; }
  const std::vector<ctfseeding::SeedingLayer> & thirdLayers() const { return theLayers; }

private:

  bool checkPhiInRange(float phi, float phi1, float phi2) const;
  std::pair<float,float> mergePhiRanges(
      const std::pair<float,float> &r1, const std::pair<float,float> &r2) const;


private:
  HitPairGenerator * thePairGenerator;
  std::vector<ctfseeding::SeedingLayer> theLayers;
  LayerCacheType * theLayerCache;
  const ClusterShapeHitFilter* filter;

  bool useFixedPreFiltering;
  float extraHitRZtolerance;
  float extraHitRPhitolerance;
  float extraPhiKDBox;
  float dphi;
  const MagneticField* bfield;
  float nomField;
  double nSigmaRZ, nSigmaPhi, fnSigmaRZ;
  bool chi2VsPtCut;
  double maxChi2;
  std::vector<double> pt_interv;
  std::vector<double> chi2_cuts;
  bool refitHits;
  bool debug;
  std::string filterName_;
  std::vector<int> detIdsToDebug;
  bool useSimpleMF;
  std::string mfName_;  
};
#endif


