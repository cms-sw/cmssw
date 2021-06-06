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
  ~TrackAssociatorByPositionProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<SimHitTPAssociationList> theSimHitTpMapToken;
  std::string thePname;
  double theQminCut;
  double theQCut;
  double thePositionMinimumDistance;
  TrackAssociatorByPositionImpl::Method theMethod;
  bool theMinIfNoMatch;
  bool theConsiderAllSimHits;
};

//
// constants, enums and typedefs
//

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
      thePname{iConfig.getParameter<std::string>("propagator")},
      theQminCut{iConfig.getParameter<double>("QminCut")},
      theQCut{iConfig.getParameter<double>("QCut")},
      thePositionMinimumDistance{iConfig.getParameter<double>("positionMinimumDistance")},
      theMethod{parseMethodName(iConfig.getParameter<std::string>("method"))},
      theMinIfNoMatch{iConfig.getParameter<bool>("MinIfNoMatch")},
      theConsiderAllSimHits{iConfig.getParameter<bool>("ConsiderAllSimHits")} {
  //register your products
  produces<reco::TrackToTrackingParticleAssociator>();
}

TrackAssociatorByPositionProducer::~TrackAssociatorByPositionProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TrackAssociatorByPositionProducer::produce(edm::StreamID,
                                                edm::Event& iEvent,
                                                const edm::EventSetup& iSetup) const {
  using namespace edm;

  Handle<SimHitTPAssociationList> assocList;
  iEvent.getByToken(theSimHitTpMapToken, assocList);

  edm::ESHandle<Propagator> theP;
  iSetup.get<TrackingComponentsRecord>().get(thePname, theP);

  edm::ESHandle<GlobalTrackingGeometry> theG;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theG);

  std::unique_ptr<reco::TrackToTrackingParticleAssociatorBaseImpl> impl{
      new TrackAssociatorByPositionImpl(iEvent.productGetter(),
                                        theG.product(),
                                        theP.product(),
                                        assocList.product(),
                                        theQminCut,
                                        theQCut,
                                        thePositionMinimumDistance,
                                        theMethod,
                                        theMinIfNoMatch,
                                        theConsiderAllSimHits)};

  std::unique_ptr<reco::TrackToTrackingParticleAssociator> toPut{
      new reco::TrackToTrackingParticleAssociator(std::move(impl))};

  iEvent.put(std::move(toPut));
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
