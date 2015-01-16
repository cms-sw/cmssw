#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class dso_hidden SeedingLayersEDProducer: public edm::stream::EDProducer<> {
public:
  SeedingLayersEDProducer(const edm::ParameterSet& iConfig);
  ~SeedingLayersEDProducer();

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
private:
  SeedingLayerSetsBuilder builder_;
};

SeedingLayersEDProducer::SeedingLayersEDProducer(const edm::ParameterSet& iConfig):
  builder_(iConfig, consumesCollector())
{
  produces<SeedingLayerSetsHits>();
}
SeedingLayersEDProducer::~SeedingLayersEDProducer() {}

void SeedingLayersEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if(builder_.check(iSetup)) {
    builder_.updateEventSetup(iSetup);
  }

  // Get hits
  std::auto_ptr<SeedingLayerSetsHits> prod(new SeedingLayerSetsHits(builder_.numberOfLayersInSet(),
                                                                    &builder_.layerSetIndices(),
                                                                    &builder_.layerNames(),
                                                                    builder_.layerDets()));
  std::vector<unsigned int> idx; ctfseeding::SeedingLayer::Hits hits; 
  builder_.hits(iEvent, iSetup,idx,hits);
  hits.shrink_to_fit();
  prod->swapHits(idx,hits);
  //prod->print();

  iEvent.put(prod);
}

void
SeedingLayersEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    std::vector<std::string> temp1;
    temp1.reserve(0);
    desc.add<std::vector<std::string> >("layerList",temp1);
  }
  {
    edm::ParameterSetDescription psd0;
    desc.add<edm::ParameterSetDescription>("MTOB",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    desc.add<edm::ParameterSetDescription>("TEC",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    desc.add<edm::ParameterSetDescription>("MTID",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    desc.add<edm::ParameterSetDescription>("FPix",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    desc.add<edm::ParameterSetDescription>("MTEC",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    desc.add<edm::ParameterSetDescription>("MTIB",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    desc.add<edm::ParameterSetDescription>("TID",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    desc.add<edm::ParameterSetDescription>("TOB",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    desc.add<edm::ParameterSetDescription>("BPix",psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    desc.add<edm::ParameterSetDescription>("TIB",psd0);
  }
  descriptions.add("seedingLayersEDProducer",desc);
}


DEFINE_FWK_MODULE(SeedingLayersEDProducer);
