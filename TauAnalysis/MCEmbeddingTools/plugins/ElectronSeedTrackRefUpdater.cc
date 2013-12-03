// -*- C++ -*-
//
// Package:    ElectronSeedTrackRefUpdater
// Class:      ElectronSeedTrackRefUpdater
//
/**\class ElectronSeedTrackRefUpdater ElectronSeedTrackRefUpdater.cc TauAnalysis/ElectronSeedTrackRefUpdater/src/ElectronSeedTrackRefUpdater.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Fri Apr  9 12:15:56 CEST 2010
// $Id: ElectronSeedTrackRefUpdater.cc,v 1.1 2012/03/01 17:03:28 fruboes Exp $
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
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"

#include <DataFormats/Math/interface/deltaR.h>
//
// class decleration
//
using namespace reco;


class ElectronSeedTrackRefUpdater : public edm::EDProducer {
   public:
      explicit ElectronSeedTrackRefUpdater(const edm::ParameterSet&);
      ~ElectronSeedTrackRefUpdater();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::InputTag _inSeeds;
      edm::InputTag _inPreId;
      edm::InputTag _targetTracks;

      std::string preidgsf_;
      std::string preidname_;
      // ----------member data ---------------------------
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
ElectronSeedTrackRefUpdater::ElectronSeedTrackRefUpdater(const edm::ParameterSet& iConfig) :
  _inSeeds(iConfig.getParameter< edm::InputTag > ("inSeeds")),
  _inPreId(iConfig.getParameter< edm::InputTag > ("inPreId")),
  _targetTracks(iConfig.getParameter< edm::InputTag > ("targetTracks"))
{
   preidgsf_ = iConfig.getParameter<std::string>("PreGsfLabel");
   preidname_= iConfig.getParameter<std::string>("PreIdLabel");

   produces<ElectronSeedCollection>(preidgsf_);
   produces<PreIdCollection>(preidname_);
   //produces<edm::ValueMap<reco::PreIdRef> >(preidname_);

}


ElectronSeedTrackRefUpdater::~ElectronSeedTrackRefUpdater()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ElectronSeedTrackRefUpdater::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace reco;

   auto_ptr<ElectronSeedCollection> output_preid(new ElectronSeedCollection);
   auto_ptr<PreIdCollection> output_preidinfo(new PreIdCollection);

   edm::Handle<reco::TrackCollection> hTargetTracks;
   iEvent.getByLabel( _targetTracks, hTargetTracks);

   edm::Handle<ElectronSeedCollection> hElectronSeeds;
   iEvent.getByLabel( _inSeeds, hElectronSeeds);

   edm::Handle<PreIdCollection> hPreId;
   iEvent.getByLabel( _inPreId, hPreId);

   for (PreIdCollection::const_iterator it = hPreId->begin(); it!=hPreId->end(); ++ it){
      PreId newPreId(*it);
      TrackRef currentTrRef = it->trackRef();
      if (  currentTrRef.isNull() || !currentTrRef.isAvailable() ) {
         std::cout << "XXXX whoops! Org track ref empty!!" << std::endl;  
         // keep things as they are;
         output_preidinfo->push_back(newPreId);
         continue;
      }
      size_t newIndex = -1;
      bool found = false;
      for (size_t i=0; i<hTargetTracks->size();++i){
        
           if ( deltaR( currentTrRef->momentum(), hTargetTracks->at(i).momentum()) < 0.001){
              newIndex = i;
              found = true;
              break;
           }
      } 
      if (found) {
         TrackRef trackRef(hTargetTracks, newIndex);
         newPreId.setTrack(trackRef);
         output_preidinfo->push_back(newPreId); // temp hack for mumu filtering
      } else {
         std::cout << "XXXX whoops! Cannot set track ref!!" << std::endl;
      }
   }
   


   for (ElectronSeedCollection::const_iterator it = hElectronSeeds->begin(); it!=hElectronSeeds->end(); ++ it){
      ElectronSeed newSeed(*it);
      TrackRef currentTrRef = it->ctfTrack ();
      if (  currentTrRef.isNull() || !currentTrRef.isAvailable() ) {
         std::cout << "XXXX whoops! Org track ref empty!!" << std::endl;
         // keep things as they are;
         output_preid->push_back(newSeed);
         continue;
      }
      size_t newIndex = -1;
      bool found = false;
      for (size_t i=0; i<hTargetTracks->size();++i){

           if ( deltaR( currentTrRef->momentum(), hTargetTracks->at(i).momentum()) < 0.001){
              newIndex = i;
              found = true;
              break;
           }
      }  
      if (found) {
         TrackRef trackRef(hTargetTracks, newIndex);
         newSeed.setCtfTrack(trackRef);
         output_preid->push_back(newSeed); // temporary hack for Zmumu filtering
      } else {
         std::cout << "XXXX whoops! Cannot set track ref!!" << std::endl;
      }
   }




   iEvent.put(output_preid,preidgsf_);
   iEvent.put(output_preidinfo,preidname_);

   //iEvent.put(newCol);

}

// ------------ method called once each job just before starting event loop  ------------
void
ElectronSeedTrackRefUpdater::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
ElectronSeedTrackRefUpdater::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronSeedTrackRefUpdater);
