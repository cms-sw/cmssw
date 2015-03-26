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
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"

//
// class decleration
//

class TrackMCQuality : public edm::global::EDProducer<> {
   public:
      explicit TrackMCQuality(const edm::ParameterSet&);
      ~TrackMCQuality();

   private:
      virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
      
      // ----------member data ---------------------------

  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> label_tr;
  edm::EDGetTokenT<TrackingParticleCollection> label_tp;
  edm::EDGetTokenT<edm::View<reco::Track> > label_associator;
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
  label_tr(consumes<reco::TrackToTrackingParticleAssociator>(pset.getParameter< edm::InputTag >("label_tr"))),
  label_tp(consumes<TrackingParticleCollection>(pset.getParameter< edm::InputTag >("label_tp"))),
  label_associator(consumes<edm::View<reco::Track> >(pset.getParameter< edm::InputTag >("associator")))
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
TrackMCQuality::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{

   using namespace edm;
   Handle<reco::TrackToTrackingParticleAssociator> associator;
   iEvent.getByToken(label_associator,associator);

   Handle<TrackingParticleCollection>  TPCollection ;
   iEvent.getByToken(label_tp, TPCollection);
     
   Handle<edm::View<reco::Track> > trackCollection;
   iEvent.getByToken(label_tr, trackCollection );

   reco::RecoToSimCollection recSimColl=associator->associateRecoToSim(trackCollection,
                                                                       TPCollection);
   
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

//define this as a plug-in
DEFINE_FWK_MODULE(TrackMCQuality);
