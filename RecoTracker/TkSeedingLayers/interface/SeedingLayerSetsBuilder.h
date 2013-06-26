#ifndef TkSeedingLayers_SeedingLayerSetsBuilder_H
#define TkSeedingLayers_SeedingLayerSetsBuilder_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>
namespace edm { class EventSetup; }

class SeedingLayerSetsBuilder {

public:

  SeedingLayerSetsBuilder(){}
  SeedingLayerSetsBuilder(const edm::ParameterSet & cfg);

  ctfseeding::SeedingLayerSets layers(const edm::EventSetup& es) const; 

private:
  std::vector<std::vector<std::string> > layerNamesInSets(
    const std::vector<std::string> & namesPSet) ;
  edm::ParameterSet layerConfig(const std::string & nameLayer,const edm::ParameterSet& cfg) const;

private:
  struct LayerSpec { 
    std::string name; 
    std::string pixelHitProducer; edm::InputTag matchedRecHits,rphiRecHits,stereoRecHits;  
    bool usePixelHitProducer, useMatchedRecHits, useRPhiRecHits, useStereoRecHits;
    std::string hitBuilder;
    bool useErrorsFromParam; double hitErrorRPhi; double hitErrorRZ; 
    bool useRingSelector; int minRing; int maxRing;
    bool useSimpleRphiHitsCleaner;
    bool skipClusters; edm::InputTag clustersToSkip;
    bool useProjection;
    std::string print() const;
  }; 
  std::vector<std::vector<LayerSpec> > theLayersInSets;
};
#endif
