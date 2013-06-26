// -*- C++ -*-
//
// Package:    TrackMCQuality
// Class:      TrackMCQuality
// 
/**\class TrackMCQuality TrackMCQuality.cc SimTracker/TrackMCQuality/src/TrackMCQuality.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Fri Mar 27 15:19:03 CET 2009
// $Id: TrackMCQuality.cc,v 1.3 2013/01/09 03:49:01 dlange Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"

//
// class decleration
//

class TrackMCQuality : public edm::EDProducer {
   public:
      explicit TrackMCQuality(const edm::ParameterSet&);
      ~TrackMCQuality();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  edm::ESHandle<TrackAssociatorBase> theAssociator;
  edm::InputTag label_tr;
  edm::InputTag label_tp;
  std::string associator;
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
TrackMCQuality::TrackMCQuality(const edm::ParameterSet& pset):
  label_tr(pset.getParameter< edm::InputTag >("label_tr")),
  label_tp(pset.getParameter< edm::InputTag >("label_tp")),
  associator(pset.getParameter< std::string >("associator"))
{
  
  produces<reco::TrackCollection>();
}


TrackMCQuality::~TrackMCQuality()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackMCQuality::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  iSetup.get<TrackAssociatorRecord>().get(associator,theAssociator);


   using namespace edm;
   Handle<TrackingParticleCollection>  TPCollection ;
   iEvent.getByLabel(label_tp, TPCollection);
     
   Handle<edm::View<reco::Track> > trackCollection;
   iEvent.getByLabel (label_tr, trackCollection );

   reco::RecoToSimCollection recSimColl=theAssociator->associateRecoToSim(trackCollection,
									  TPCollection,
									  &iEvent,&iSetup);
   
   //then loop the track collection
   std::auto_ptr<reco::TrackCollection> outTracks(new reco::TrackCollection(trackCollection->size()));
   
   for (unsigned int iT=0;iT!=trackCollection->size();++iT){
     edm::RefToBase<reco::Track> track(trackCollection, iT);
     bool matched=false;
     //find it in the map
     if (recSimColl.find(track)!=recSimColl.end()){
       // you can get the data if you want
       std::vector<std::pair<TrackingParticleRef, double> > tp= recSimColl[track];
       matched=true;
     }
     else{
       matched=false;
     }     

     //copy the track into the new container
     (*outTracks)[iT] = reco::Track(*track);
     if (matched){
       (*outTracks)[iT].setQuality(reco::TrackBase::qualitySize); //is not assigned to any quality. use it as a fake/matched flag
     }
   }
   
   iEvent.put(outTracks);
}

// ------------ method called once each job just before starting event loop  ------------
void 
TrackMCQuality::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrackMCQuality::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackMCQuality);
