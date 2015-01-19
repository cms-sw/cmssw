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
    desc.add<std::vector<std::string> >("layerList",temp1)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.setAllowAnything();
    /*
    psd0.add<edm::InputTag>("skipClusters","");
    psd0.add<std::string>("TTRHBuilder","ESPTTRHBuilderPixelOnly");
    psd0.add<std::string>("HitProducer","SiPixelRecHits");
    psd0.add<bool>("useErrorsFromParam",true);
    psd0.add<double>("hitErrorRZ",0.006);
    psd0.add<double>("hitErrorRPhi",0.0027);
    */
    desc.add<edm::ParameterSetDescription>("BPix",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.setAllowAnything();
    /*
    psd0.add<edm::InputTag>("skipClusters","");
    psd0.add<std::string>("TTRHBuilder","ESPTTRHBuilderPixelOnly");
    psd0.add<std::string>("HitProducer","SiPixelRecHits");
    psd0.add<bool>("useErrorsFromParam",true);
    psd0.add<double>("hitErrorRZ",0.0036);
    psd0.add<double>("hitErrorRPhi",0.0051);
    */
    desc.add<edm::ParameterSetDescription>("FPix",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.setAllowAnything();
    /*
    psd0.add<edm::InputTag>("rphiRecHits",edm::InputTag("siStripMatchedRecHits","rphiRecHit"));
    psd0.add<edm::InputTag>("matchedRecHits",edm::InputTag("siStripMatchedRecHits","matchedRecHit"));
    psd0.add<edm::InputTag>("skipClusters","");
    psd0.add<std::string>("TTRHBuilder","WithTrackAngle");
    psd0.add<bool>("useSimpleRphiHitsCleaner",true);
    psd0.add<double>("minGoodCharge",-2069.0);
    */
    desc.add<edm::ParameterSetDescription>("TIB",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.setAllowAnything();
    /*
    psd0.add<edm::InputTag>("rphiRecHits",edm::InputTag("siStripMatchedRecHits","rphiRecHit"));
    psd0.add<edm::InputTag>("matchedRecHits",edm::InputTag("siStripMatchedRecHits","matchedRecHit"));
    psd0.add<edm::InputTag>("skipClusters","");
    psd0.add<std::string>("TTRHBuilder","WithTrackAngle");
    psd0.add<bool>("useSimpleRphiHitsCleaner",true);
    psd0.add<double>("minGoodCharge",-2069.0);
    */
    desc.add<edm::ParameterSetDescription>("MTIB",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.setAllowAnything();
    /*
    psd0.add<edm::InputTag>("rphiRecHits",edm::InputTag("siStripMatchedRecHits","rphiRecHit"));
    psd0.add<edm::InputTag>("matchedRecHits",edm::InputTag("siStripMatchedRecHits","matchedRecHit"));
    psd0.add<edm::InputTag>("skipClusters","");
    psd0.add<std::string>("TTRHBuilder","WithTrackAngle");
    psd0.add<bool>("useSimpleRphiHitsCleaner",true);
    psd0.add<bool>("useRingSlector",false);
    psd0.add<int>("minRing",5);
    psd0.add<int>("maxRing",7);
    psd0.add<double>("minGoodCharge",-2069.0);
    */
    desc.add<edm::ParameterSetDescription>("TID",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.setAllowAnything();
    /*
    psd0.add<edm::InputTag>("rphiRecHits",edm::InputTag("siStripMatchedRecHits","rphiRecHit"));
    psd0.add<edm::InputTag>("matchedRecHits",edm::InputTag("siStripMatchedRecHits","matchedRecHit"));
    psd0.add<edm::InputTag>("skipClusters","");
    psd0.add<std::string>("TTRHBuilder","WithTrackAngle");
    psd0.add<bool>("useSimpleRphiHitsCleaner",true);
    psd0.add<bool>("useRingSlector",false);
    psd0.add<int>("minRing",5);
    psd0.add<int>("maxRing",7);
    psd0.add<double>("minGoodCharge",-2069.0);
    */
    desc.add<edm::ParameterSetDescription>("MTID",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.setAllowAnything();
    /*
    psd0.add<edm::InputTag>("rphiRecHits",edm::InputTag("siStripMatchedRecHits","rphiRecHit"));
    psd0.add<edm::InputTag>("matchedRecHits",edm::InputTag("siStripMatchedRecHits","matchedRecHit"));
    psd0.add<edm::InputTag>("skipClusters","");
    psd0.add<std::string>("TTRHBuilder","WithTrackAngle");
    psd0.add<bool>("useSimpleRphiHitsCleaner",true);
    psd0.add<double>("minGoodCharge",-2069.0);
    */
    desc.add<edm::ParameterSetDescription>("TOB",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.setAllowAnything();
    /*
    psd0.add<edm::InputTag>("rphiRecHits",edm::InputTag("siStripMatchedRecHits","rphiRecHit"));
    psd0.add<edm::InputTag>("matchedRecHits",edm::InputTag("siStripMatchedRecHits","matchedRecHit"));
    psd0.add<edm::InputTag>("skipClusters","");
    psd0.add<std::string>("TTRHBuilder","WithTrackAngle");
    psd0.add<bool>("useSimpleRphiHitsCleaner",true);
    psd0.add<double>("minGoodCharge",-2069.0);
    */
    desc.add<edm::ParameterSetDescription>("MTOB",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.setAllowAnything();
    /*
    psd0.add<edm::InputTag>("rphiRecHits",edm::InputTag("siStripMatchedRecHits","rphiRecHit"));
    psd0.add<edm::InputTag>("matchedRecHits",edm::InputTag("siStripMatchedRecHits","matchedRecHit"));
    psd0.add<edm::InputTag>("skipClusters","");
    psd0.add<std::string>("TTRHBuilder","WithTrackAngle");
    psd0.add<bool>("useSimpleRphiHitsCleaner",true);
    psd0.add<bool>("useRingSlector",false);
    psd0.add<int>("minRing",5);
    psd0.add<int>("maxRing",7);
    psd0.add<double>("minGoodCharge",-2069.0);
    */
    desc.add<edm::ParameterSetDescription>("TEC",psd0)->setComment("");
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.setAllowAnything();
    /*
    psd0.add<edm::InputTag>("rphiRecHits",edm::InputTag("siStripMatchedRecHits","rphiRecHit"));
    psd0.add<edm::InputTag>("matchedRecHits",edm::InputTag("siStripMatchedRecHits","matchedRecHit"));
    psd0.add<edm::InputTag>("skipClusters","");
    psd0.add<std::string>("TTRHBuilder","WithTrackAngle");
    psd0.add<bool>("useSimpleRphiHitsCleaner",true);
    psd0.add<bool>("useRingSlector",false);
    psd0.add<int>("minRing",5);
    psd0.add<int>("maxRing",7);
    psd0.add<double>("minGoodCharge",-2069.0);
    */
    desc.add<edm::ParameterSetDescription>("MTEC",psd0)->setComment("");
  }

  descriptions.add("seedingLayersEDProducer",desc);
  descriptions.setComment("");
}


DEFINE_FWK_MODULE(SeedingLayersEDProducer);
