// -*- C++ -*-
//
// Package:    SimTracker/TrackAssociatorProducers
// Class:      TrackAssociatorByChi2Producer
//
/**\class TrackAssociatorByChi2Producer TrackAssociatorByChi2Producer.cc SimTracker/TrackAssociatorProducers/plugins/TrackAssociatorByChi2Producer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 06 Jan 2015 16:13:00 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociatorBaseImpl.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackAssociatorByChi2Impl.h"
#include "TrackGenAssociatorByChi2Impl.h"

//
// class declaration
//

class TrackAssociatorByChi2Producer : public edm::global::EDProducer<> {
public:
  explicit TrackAssociatorByChi2Producer(const edm::ParameterSet&);
  ~TrackAssociatorByChi2Producer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  edm::EDPutTokenT<reco::TrackToTrackingParticleAssociator> tpPutToken_;
  edm::EDPutTokenT<reco::TrackToGenParticleAssociator> genPutToken_;
  const double chi2cut_;
  const bool onlyDiagonal_;
};

//
// constructors and destructor
//
TrackAssociatorByChi2Producer::TrackAssociatorByChi2Producer(const edm::ParameterSet& iConfig)
    : bsToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      magFieldToken_(esConsumes()),
      tpPutToken_(produces<reco::TrackToTrackingParticleAssociator>()),
      genPutToken_(produces<reco::TrackToGenParticleAssociator>()),
      chi2cut_(iConfig.getParameter<double>("chi2cut")),
      onlyDiagonal_(iConfig.getParameter<bool>("onlyDiagonal")) {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TrackAssociatorByChi2Producer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  auto const& magField = iSetup.getData(magFieldToken_);
  auto const& beamSpot = iEvent.get(bsToken_);

  iEvent.emplace(
      tpPutToken_,
      std::make_unique<TrackAssociatorByChi2Impl>(iEvent.productGetter(), magField, beamSpot, chi2cut_, onlyDiagonal_));
  iEvent.emplace(genPutToken_,
                 std::make_unique<TrackGenAssociatorByChi2Impl>(magField, beamSpot, chi2cut_, onlyDiagonal_));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TrackAssociatorByChi2Producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackAssociatorByChi2Producer);
