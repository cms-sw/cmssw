#ifndef MultiHitGeneratorFromChi2_H
#define MultiHitGeneratorFromChi2_H

/** A MultiHitGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "RecoTracker/TkSeedGenerator/interface/MultiHitGenerator.h"
#include "CombinedMultiHitGenerator.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"


#include <utility>
#include <vector>

class HitPairGeneratorFromLayerPair;

class dso_hidden MultiHitGeneratorFromChi2 final : public MultiHitGeneratorFromPairAndLayers {

typedef CombinedMultiHitGenerator::LayerCacheType       LayerCacheType;

public:
  MultiHitGeneratorFromChi2(const edm::ParameterSet& cfg);

  virtual ~MultiHitGeneratorFromChi2();

  void initES(const edm::EventSetup& es) override; 

  virtual void hitSets( const TrackingRegion& region, OrderedMultiHits & trs, 
                        const edm::Event & ev, const edm::EventSetup& es,
                        SeedingLayerSetsHits::SeedingLayerSet pairLayers,
                        std::vector<SeedingLayerSetsHits::SeedingLayer> thirdLayers);

private:
  using HitOwnPtr = mayown_ptr<BaseTrackerRecHit>;

  bool checkPhiInRange(float phi, float phi1, float phi2) const;
  std::pair<float,float> mergePhiRanges(
      const std::pair<float,float> &r1, const std::pair<float,float> &r2) const;

  void refit2Hits(HitOwnPtr & hit0,
		  HitOwnPtr & hit1,
		  TrajectoryStateOnSurface& tsos0,
		  TrajectoryStateOnSurface& tsos1,
		  const TrackingRegion& region, float nomField, bool isDebug);
  /*
  void refit3Hits(HitOwnPtr & hit0,
		  HitOwnPtr & hit1,
		  HitOwnPtr & hit2,
		  TrajectoryStateOnSurface& tsos0,
		  TrajectoryStateOnSurface& tsos1,
		  TrajectoryStateOnSurface& tsos2,
		  float nomField, bool isDebug);
  */
private:
  const ClusterShapeHitFilter* filter;
  TkTransientTrackingRecHitBuilder const * builder;
  TkClonerImpl cloner;

  bool useFixedPreFiltering;
  float extraHitRZtolerance;
  float extraHitRPhitolerance;
  float extraZKDBox;
  float extraRKDBox;
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
  std::string filterName_;
  std::string builderName_;

  bool useSimpleMF_;
  std::string mfName_;

  std::vector<int> detIdsToDebug;



};
#endif


