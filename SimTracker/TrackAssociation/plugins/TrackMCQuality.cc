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

class TrackMCQuality final : public edm::global::EDProducer<> {
   public:
      explicit TrackMCQuality(const edm::ParameterSet&);
      ~TrackMCQuality();

   private:
      virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
      
      // ----------member data ---------------------------

  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> label_assoc;
  edm::EDGetTokenT<TrackingParticleCollection> label_tp;
  edm::EDGetTokenT<edm::View<reco::Track> > label_tr;

  using Product=std::vector<float>;  
 
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
  label_assoc(consumes<reco::TrackToTrackingParticleAssociator>(pset.getParameter< edm::InputTag >("associator"))),
  label_tp(consumes<TrackingParticleCollection>(pset.getParameter< edm::InputTag >("trackingParticles"))),
  label_tr(consumes<edm::View<reco::Track> >(pset.getParameter< edm::InputTag >("tracks")))
{
  produces<Product>();
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
   iEvent.getByToken(label_assoc,associator);

   Handle<TrackingParticleCollection>  TPCollection ;
   iEvent.getByToken(label_tp, TPCollection);
     
   Handle<edm::View<reco::Track> > trackCollection;
   iEvent.getByToken(label_tr, trackCollection );

   reco::RecoToSimCollection recSimColl=associator->associateRecoToSim(trackCollection,
                                                                       TPCollection);
   
   //then loop the track collection
   std::unique_ptr<Product> product(new Product(trackCollection->size(),0));

   
   for (unsigned int iT=0;iT!=trackCollection->size();++iT){
     auto & prod = (*product)[iT];

     edm::RefToBase<reco::Track> track(trackCollection, iT);

     //find it in the map
     if (recSimColl.find(track)==recSimColl.end()) continue;
       
     auto const & tp = recSimColl[track];

     if (tp.empty()) continue;  // can it be?
     // nSimHits = tp[0].first->numberOfTrackerHits();
     prod = tp[0].second;
     // if (tp[0].first->charge() != track->charge()) isChargeMatched = false;
     if ( (tp[0].first->eventId().event() != 0) || (tp[0].first->eventId().bunchCrossing() != 0) ) prod=-prod;

       
   }
   
   iEvent.put(std::move(product));
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackMCQuality);
