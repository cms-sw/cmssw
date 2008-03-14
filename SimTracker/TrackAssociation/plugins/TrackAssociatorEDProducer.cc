//
// Original Author:  Stefano Magni
//         Created:  Fri Mar  9 10:52:11 CET 2007
// $Id$
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class decleration
//

class TrackAssociatorEDProducer : public edm::EDProducer {
public:
  explicit TrackAssociatorEDProducer(const edm::ParameterSet&);
  ~TrackAssociatorEDProducer();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  edm::ESHandle<TrackAssociatorBase> theAssociator;
  edm::InputTag label_tr;
  edm::InputTag label_tp;
  std::string associator;
};

TrackAssociatorEDProducer::TrackAssociatorEDProducer(const edm::ParameterSet& pset):
  label_tr(pset.getParameter< edm::InputTag >("label_tr")),
  label_tp(pset.getParameter< edm::InputTag >("label_tp")),
  associator(pset.getParameter< std::string >("associator"))
{
  produces<reco::SimToRecoCollection>();
  produces<reco::RecoToSimCollection>();
}


TrackAssociatorEDProducer::~TrackAssociatorEDProducer() {
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackAssociatorEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
   using namespace edm;

   Handle<TrackingParticleCollection>  TPCollection ;
   iEvent.getByLabel(label_tp, TPCollection);
     
   Handle<reco::TrackCollection> trackCollection;
   iEvent.getByLabel (label_tr, trackCollection );

   //associate tracks
   LogTrace("TrackValidator") << "Calling associateRecoToSim method" << "\n";
   reco::RecoToSimCollection recSimColl=theAssociator->associateRecoToSim(trackCollection,
									  TPCollection,
									  &iEvent);
   LogTrace("TrackValidator") << "Calling associateSimToReco method" << "\n";
   reco::SimToRecoCollection simRecColl=theAssociator->associateSimToReco(trackCollection,
									  TPCollection, 
									  &iEvent);

   std::auto_ptr<reco::RecoToSimCollection> rts(new reco::RecoToSimCollection(recSimColl));
   std::auto_ptr<reco::SimToRecoCollection> str(new reco::SimToRecoCollection(simRecColl));

   iEvent.put(rts);
   iEvent.put(str);
}

// ------------ method called once each job just before starting event loop  ------------
void 
TrackAssociatorEDProducer::beginJob(const edm::EventSetup& setup) {
  // Get associator from eventsetup

  setup.get<TrackAssociatorRecord>().get(associator,theAssociator);
   
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrackAssociatorEDProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackAssociatorEDProducer);
