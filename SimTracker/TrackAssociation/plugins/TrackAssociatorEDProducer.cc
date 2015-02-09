//
// Original Author:  Stefano Magni
//         Created:  Fri Mar  9 10:52:11 CET 2007
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/EDGetToken.h"



//
// class decleration
//

class TrackAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit TrackAssociatorEDProducer(const edm::ParameterSet&);
  ~TrackAssociatorEDProducer();
  
private:
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  
  bool  theIgnoremissingtrackcollection;

  edm::EDGetTokenT<TrackingParticleCollection> TPCollectionToken_;
  edm::EDGetTokenT<edm::View<reco::Track> > trackCollectionToken_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> associatorToken_;

};

TrackAssociatorEDProducer::TrackAssociatorEDProducer(const edm::ParameterSet& pset):
  theIgnoremissingtrackcollection(pset.getUntrackedParameter<bool>("ignoremissingtrackcollection",false))
{
  produces<reco::SimToRecoCollection>();
  produces<reco::RecoToSimCollection>();

  TPCollectionToken_    = consumes<TrackingParticleCollection>(pset.getParameter< edm::InputTag >("label_tp"));
  trackCollectionToken_ = consumes<edm::View<reco::Track> >(pset.getParameter< edm::InputTag >("label_tr")); 
  associatorToken_      = consumes<reco::TrackToTrackingParticleAssociator>(pset.getParameter<edm::InputTag>("associator") );

}


TrackAssociatorEDProducer::~TrackAssociatorEDProducer() {
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackAssociatorEDProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
   using namespace edm;

   edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator;
   iEvent.getByToken(associatorToken_,theAssociator);

   Handle<TrackingParticleCollection>  TPCollection ;
   iEvent.getByToken(TPCollectionToken_,TPCollection);

   Handle<edm::View<reco::Track> > trackCollection;
   bool trackAvailable = iEvent.getByToken(trackCollectionToken_, trackCollection );

   std::auto_ptr<reco::RecoToSimCollection> rts;
   std::auto_ptr<reco::SimToRecoCollection> str;

   if (theIgnoremissingtrackcollection && !trackAvailable){
     //the track collection is not in the event and we're being told to ignore this.
     //do not output anything to the event, other wise this would be considered as inefficiency.
   } else {
     //associate tracks
     LogTrace("TrackValidator") << "Calling associateRecoToSim method" << "\n";
     reco::RecoToSimCollection recSimColl=theAssociator->associateRecoToSim(trackCollection,
									    TPCollection);

     LogTrace("TrackValidator") << "Calling associateSimToReco method" << "\n";
     reco::SimToRecoCollection simRecColl=theAssociator->associateSimToReco(trackCollection,
									    TPCollection);
     
     rts.reset(new reco::RecoToSimCollection(recSimColl));
     str.reset(new reco::SimToRecoCollection(simRecColl));

     iEvent.put(rts);
     iEvent.put(str);
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackAssociatorEDProducer);
