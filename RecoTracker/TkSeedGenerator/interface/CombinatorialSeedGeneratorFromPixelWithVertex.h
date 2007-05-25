#ifndef CombinatorialSeedGeneratorFromPixelWithVertex_H
#define CombinatorialSeedGeneratorFromPixelWithVertex_H
/** \class CombinatorialSeedGeneratorFromPixelWithVertex
 *  A concrete (not yet regional) seed generator providing seeds constructed 
 *  from combinations of hits in pairs of pixel layers 
 */
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"    
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromLayerPairs.h"
#include "RecoTracker/TkHitPairs/interface/PixelSeedLayerPairs.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class PixelSeedLayerPairs;

class CombinatorialSeedGeneratorFromPixelWithVertex : public SeedGeneratorFromLayerPairs {
 public:
  
  CombinatorialSeedGeneratorFromPixelWithVertex(const edm::ParameterSet& conf);
  ~CombinatorialSeedGeneratorFromPixelWithVertex(){delete pixelLayers;} 
  
  void init(const SiPixelRecHitCollection &coll, const reco::VertexCollection &vtxcoll, const edm::EventSetup& c);
  void  run(TrajectorySeedCollection &,  const edm::EventSetup& c);
 private:
  //  edm::ParameterSet conf_;
  GlobalTrackingRegion region;
  const reco::VertexCollection *pixelVertices_;
  PixelSeedLayerPairs* pixelLayers;
  uint32_t numberOfVertices_;
  //float  vertexRadius_, vertexDeltaZ_, vertexZSigmas_, ptMin_;
  float  vertexRadius_, vertexDeltaZ_, fallbackDeltaZ_, ptMin_;
  bool   mergeOverlaps_;
};
#endif


