#include <memory>
#include <string>

#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

class BarrelLCToSCAssociatorEDProducer : public edm::global::EDProducer<> {
  public:
    explicit BarrelLCToSCAssociatorEDProducer(const edm::ParameterSet&);
    ~BarrelLCToSCAssociatorEDProducer() override;

  private:
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

    edm::EDGetTokenT<SimClusterCollection> SCCollectionToken_;
    edm::EDGetTokenT<reco::CaloClusterCollection> LCCollectionToken_;
    edm::EDGetTokenT<hgcal::LayerClusterToSimClusterAssociator> associatorToken_;
};

BarrelLCToSCAssociatorEDProducer::BarrelLCToSCAssociatorEDProducer(const edm::ParameterSet& pset) {
  produces<hgcal::SimToRecoCollectionWithSimClusters>();
  produces<hgcal::RecoToSimCollectionWithSimClusters>();

  SCCollectionToken_ = consumes<SimClusterCollection>(pset.getParameter<edm::InputTag>("label_scl"));
  LCCollectionToken_ = consumes<reco::CaloClusterCollection>(pset.getParameter<edm::InputTag>("label_lcl"));
  associatorToken_ =
      consumes<hgcal::LayerClusterToSimClusterAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

BarrelLCToSCAssociatorEDProducer::~BarrelLCToSCAssociatorEDProducer() {}

void BarrelLCToSCAssociatorEDProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<hgcal::LayerClusterToSimClusterAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  Handle<SimClusterCollection> SCCollection;
  iEvent.getByToken(SCCollectionToken_, SCCollection);

  Handle<reco::CaloClusterCollection> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  hgcal::RecoToSimCollectionWithSimClusters recSimColl = theAssociator->associateRecoToSim(LCCollection, SCCollection);
  hgcal::SimToRecoCollectionWithSimClusters simRecColl = theAssociator->associateSimToReco(LCCollection, SCCollection);

  auto rts = std::make_unique<hgcal::RecoToSimCollectionWithSimClusters>(recSimColl);
  auto str = std::make_unique<hgcal::SimToRecoCollectionWithSimClusters>(simRecColl);

  iEvent.put(std::move(rts));
  iEvent.put(std::move(str));
}

DEFINE_FWK_MODULE(BarrelLCToSCAssociatorEDProducer);

