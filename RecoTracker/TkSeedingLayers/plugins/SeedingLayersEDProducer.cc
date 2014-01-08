#include "RecoTracker/SeedingLayerSetsHits/interface/SeedingLayerSetsHits.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"

class SeedingLayersEDProducer: public edm::EDProducer {
public:
  SeedingLayersEDProducer(const edm::ParameterSet& iConfig);
  ~SeedingLayersEDProducer();

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  SeedingLayerSetsBuilder builder_;
};

SeedingLayersEDProducer::SeedingLayersEDProducer(const edm::ParameterSet& iConfig):
  builder_(iConfig)
{
  produces<SeedingLayerSetsHits>();
}
SeedingLayersEDProducer::~SeedingLayersEDProducer() {}

void SeedingLayersEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ctfseeding::SeedingLayerSets layerSets = builder_.layers(iSetup);

  // Ensure all SeedingLayers objects have same number of SeedingLayer's
  unsigned int nlayers = layerSets[0].size();
  for(size_t i=0; i<layerSets.size(); ++i) {
    if(nlayers != layerSets[i].size())
      throw cms::Exception("Configuration") << "Assuming all SeedingLayers to have same number of layers, Layers " << i << " has " << layerSets[i].size() << " while 0th has " << nlayers;
  }

  // Get hits
  std::auto_ptr<SeedingLayerSetsHits> prod(new SeedingLayerSetsHits(nlayers));
  for(const ctfseeding::SeedingLayers& layers: layerSets) {
    for(const ctfseeding::SeedingLayer& layer: layers) {
      std::pair<unsigned int, bool> index = prod->insertLayer(layer.name(), layer.detLayer());
      if(index.second) {
        // layer was really inserted, we have to pass also the hits
        prod->insertLayerHits(index.first, layer.hits(iEvent, iSetup));
      }
    }
  }
  //prod->print();

  iEvent.put(prod);
}

DEFINE_FWK_MODULE(SeedingLayersEDProducer);
