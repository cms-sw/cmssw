//
// Original Author:  Stefano Magni
//         Created:  Fri Mar  9 10:52:11 CET 2007
// $Id: TrackAssociatorEDProducer.cc,v 1.4 2009/02/03 11:08:05 vlimant Exp $
//
//


// system include files
#include <memory>
#include <string>

// user include files
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
  bool  theIgnoremissingtrackcollection;
};

TrackAssociatorEDProducer::TrackAssociatorEDProducer(const edm::ParameterSet& pset):
  label_tr(pset.getParameter< edm::InputTag >("label_tr")),
  label_tp(pset.getParameter< edm::InputTag >("label_tp")),
  associator(pset.getParameter< std::string >("associator")),
  theIgnoremissingtrackcollection(pset.getParameter<bool>("ignoremissingtrackcollection"))
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
     
   Handle<edm::View<reco::Track> > trackCollection;
   bool trackAvailable = iEvent.getByLabel (label_tr, trackCollection );

   std::auto_ptr<reco::RecoToSimCollection> rts;
   std::auto_ptr<reco::SimToRecoCollection> str;

   if (theIgnoremissingtrackcollection && !trackAvailable){
     //the track collection is not in the event and we're being told to ignore this.
     //do not output anything to the event, other wise this would be considered as inefficiency.
   }else{
     //associate tracks
     LogTrace("TrackValidator") << "Calling associateRecoToSim method" << "\n";
     reco::RecoToSimCollection recSimColl=theAssociator->associateRecoToSim(trackCollection,
									    TPCollection,
									    &iEvent);
     LogTrace("TrackValidator") << "Calling associateSimToReco method" << "\n";
     reco::SimToRecoCollection simRecColl=theAssociator->associateSimToReco(trackCollection,
									    TPCollection, 
									    &iEvent);
     
     rts.reset(new reco::RecoToSimCollection(recSimColl));
     str.reset(new reco::SimToRecoCollection(simRecColl));

     iEvent.put(rts);
     iEvent.put(str);
   }
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
