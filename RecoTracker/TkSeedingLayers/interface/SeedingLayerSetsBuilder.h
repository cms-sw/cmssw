#ifndef TkSeedingLayers_SeedingLayerSetsBuilder_H
#define TkSeedingLayers_SeedingLayerSetsBuilder_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>
namespace edm { class EventSetup; class ConsumesCollector;}

class SeedingLayerSetsBuilder {

public:

  SeedingLayerSetsBuilder();
  SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC);
  SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector&& iC);
  ~SeedingLayerSetsBuilder();

  ctfseeding::SeedingLayerSets layers() const;

private:
  std::vector<std::vector<std::string> > layerNamesInSets(
    const std::vector<std::string> & namesPSet) ;
  edm::ParameterSet layerConfig(const std::string & nameLayer,const edm::ParameterSet& cfg) const;
  std::map<std::string,int> nameToId;

private:
  struct LayerSpec { 
    LayerSpec();
    ~LayerSpec();
    std::string name; 
    std::string pixelHitProducer;
    bool usePixelHitProducer;
    std::string hitBuilder;
    bool useErrorsFromParam; double hitErrorRPhi; double hitErrorRZ; 
    bool useProjection;

    GeomDetEnumerators::SubDetector subdet;
    ctfseeding::SeedingLayer::Side side;
    int idLayer;
    std::shared_ptr<ctfseeding::HitExtractor> extractor;

    std::string print() const;
  }; 
  std::vector<std::vector<LayerSpec> > theLayersInSets;
};
#endif
