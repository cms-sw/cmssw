// -*- C++ -*-
//
// Package:    SimTracker/TrackAssociatorProducers
// Class:      TrackAssociatorByHitsProducer
//
/**\class TrackAssociatorByHitsProducer TrackAssociatorByHitsProducer.cc SimTracker/TrackAssociatorProducers/plugins/TrackAssociatorByHitsProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 05 Jan 2015 20:38:27 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackAssociatorByHitsImpl.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

//
// class declaration
//

using SimHitTPAssociationList = TrackAssociatorByHitsImpl::SimHitTPAssociationList;

class TrackAssociatorByHitsProducer : public edm::global::EDProducer<> {
public:
  explicit TrackAssociatorByHitsProducer(const edm::ParameterSet&);
  ~TrackAssociatorByHitsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  edm::EDGetTokenT<SimHitTPAssociationList> simHitTpMapToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::EDPutTokenT<reco::TrackToTrackingParticleAssociator> putToken_;

  TrackAssociatorByHitsImpl::SimToRecoDenomType SimToRecoDenominator;
  const double quality_SimToReco;
  const double purity_SimToReco;
  const double cut_RecoToSim;
  const bool UsePixels;
  const bool UseGrouped;
  const bool UseSplitting;
  const bool ThreeHitTracksAreSpecial;
  const bool AbsoluteNumberOfHits;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackAssociatorByHitsProducer::TrackAssociatorByHitsProducer(const edm::ParameterSet& iConfig)
    : trackerHitAssociatorConfig_(iConfig, consumesCollector()),
      simHitTpMapToken_(consumes<SimHitTPAssociationList>(iConfig.getParameter<edm::InputTag>("simHitTpMapTag"))),
      tTopoToken_(esConsumes()),
      putToken_(produces<reco::TrackToTrackingParticleAssociator>()),
      SimToRecoDenominator(TrackAssociatorByHitsImpl::denomnone),
      quality_SimToReco(iConfig.getParameter<double>("Quality_SimToReco")),
      purity_SimToReco(iConfig.getParameter<double>("Purity_SimToReco")),
      cut_RecoToSim(iConfig.getParameter<double>("Cut_RecoToSim")),
      UsePixels(iConfig.getParameter<bool>("UsePixels")),
      UseGrouped(iConfig.getParameter<bool>("UseGrouped")),
      UseSplitting(iConfig.getParameter<bool>("UseSplitting")),
      ThreeHitTracksAreSpecial(iConfig.getParameter<bool>("ThreeHitTracksAreSpecial")),
      AbsoluteNumberOfHits(iConfig.getParameter<bool>("AbsoluteNumberOfHits")) {
  std::string tmp = iConfig.getParameter<std::string>("SimToRecoDenominator");
  if (tmp == "sim") {
    SimToRecoDenominator = TrackAssociatorByHitsImpl::denomsim;
  } else if (tmp == "reco") {
    SimToRecoDenominator = TrackAssociatorByHitsImpl::denomreco;
  }

  if (SimToRecoDenominator == TrackAssociatorByHitsImpl::denomnone) {
    throw cms::Exception("TrackAssociatorByHitsImpl") << "SimToRecoDenominator not specified as sim or reco";
  }
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TrackAssociatorByHitsProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  auto thAssoc = std::make_unique<TrackerHitAssociator>(iEvent, trackerHitAssociatorConfig_);

  iEvent.emplace(putToken_,
                 std::make_unique<TrackAssociatorByHitsImpl>(
                     iEvent.productGetter(),
                     std::move(thAssoc),
                     &iSetup.getData(tTopoToken_),
                     //warning: make sure the TP collection used in the map is the same used in the associator!
                     &iEvent.get(simHitTpMapToken_),
                     SimToRecoDenominator,
                     quality_SimToReco,
                     purity_SimToReco,
                     cut_RecoToSim,
                     UsePixels,
                     UseGrouped,
                     UseSplitting,
                     ThreeHitTracksAreSpecial,
                     AbsoluteNumberOfHits));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TrackAssociatorByHitsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackAssociatorByHitsProducer);
