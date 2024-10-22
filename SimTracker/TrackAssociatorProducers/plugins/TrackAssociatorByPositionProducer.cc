// -*- C++ -*-
//
// Package:    SimTracker/TrackAssociatorProducers
// Class:      TrackAssociatorByPositionProducer
//
/**\class TrackAssociatorByPositionProducer TrackAssociatorByPositionProducer.cc SimTracker/TrackAssociatorProducers/plugins/TrackAssociatorByPositionProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 05 Jan 2015 22:05:57 GMT
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

#include "TrackAssociatorByPositionImpl.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

//
// class declaration
//
using SimHitTPAssociationList = TrackAssociatorByPositionImpl::SimHitTPAssociationList;

class TrackAssociatorByPositionProducer : public edm::global::EDProducer<> {
public:
  explicit TrackAssociatorByPositionProducer(const edm::ParameterSet&);
  ~TrackAssociatorByPositionProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<SimHitTPAssociationList> theSimHitTpMapToken;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> thePropagatorToken;
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> theGeometryToken;
  edm::EDPutTokenT<reco::TrackToTrackingParticleAssociator> thePutToken;
  double theQminCut;
  double theQCut;
  double thePositionMinimumDistance;
  TrackAssociatorByPositionImpl::Method theMethod;
  bool theMinIfNoMatch;
  bool theConsiderAllSimHits;
};

//
// static data member definitions
//
static TrackAssociatorByPositionImpl::Method parseMethodName(const std::string& meth) {
  if (meth == "chi2") {
    return TrackAssociatorByPositionImpl::Method::chi2;
  } else if (meth == "dist") {
    return TrackAssociatorByPositionImpl::Method::dist;
  } else if (meth == "momdr") {
    return TrackAssociatorByPositionImpl::Method::momdr;
  } else if (meth == "posdr") {
    return TrackAssociatorByPositionImpl::Method::posdr;
  } else {
    throw cms::Exception("BadParameterName") << meth << " TrackAssociatorByPostionImpl::Method name not recognized.";
  }
}

//
// constructors and destructor
//
TrackAssociatorByPositionProducer::TrackAssociatorByPositionProducer(const edm::ParameterSet& iConfig)
    : theSimHitTpMapToken{consumes<SimHitTPAssociationList>(iConfig.getParameter<edm::InputTag>("simHitTpMapTag"))},
      thePropagatorToken{esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("propagator")))},
      theGeometryToken{esConsumes()},
      thePutToken{produces<reco::TrackToTrackingParticleAssociator>()},
      theQminCut{iConfig.getParameter<double>("QminCut")},
      theQCut{iConfig.getParameter<double>("QCut")},
      thePositionMinimumDistance{iConfig.getParameter<double>("positionMinimumDistance")},
      theMethod{parseMethodName(iConfig.getParameter<std::string>("method"))},
      theMinIfNoMatch{iConfig.getParameter<bool>("MinIfNoMatch")},
      theConsiderAllSimHits{iConfig.getParameter<bool>("ConsiderAllSimHits")} {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TrackAssociatorByPositionProducer::produce(edm::StreamID,
                                                edm::Event& iEvent,
                                                const edm::EventSetup& iSetup) const {
  iEvent.emplace(thePutToken,
                 std::make_unique<TrackAssociatorByPositionImpl>(iEvent.productGetter(),
                                                                 &iSetup.getData(theGeometryToken),
                                                                 &iSetup.getData(thePropagatorToken),
                                                                 &iEvent.get(theSimHitTpMapToken),
                                                                 theQminCut,
                                                                 theQCut,
                                                                 thePositionMinimumDistance,
                                                                 theMethod,
                                                                 theMinIfNoMatch,
                                                                 theConsiderAllSimHits));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TrackAssociatorByPositionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackAssociatorByPositionProducer);
