// -*- C++ -*-
//
// Package:    ElectronSeedTrackRefUpdaterAndMerger
// Class:      ElectronSeedTrackRefUpdaterAndMerger
//
/**\class ElectronSeedTrackRefUpdaterAndMerger ElectronSeedTrackRefUpdaterAndMerger.cc TauAnalysis/ElectronSeedTrackRefUpdaterAndMerger/src/ElectronSeedTrackRefUpdaterAndMerger.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Fri Apr  9 12:15:56 CEST 2010
// $Id: ElectronSeedTrackRefUpdaterAndMerger.cc,v 1.1 2012/03/01 17:03:28 fruboes Exp $
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


class ElectronSeedTrackRefUpdaterAndMerger : public edm::EDProducer {
   public:
      explicit ElectronSeedTrackRefUpdaterAndMerger(const edm::ParameterSet&);
      ~ElectronSeedTrackRefUpdaterAndMerger();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::InputTag _inSeeds1;
      edm::InputTag _inPreId1;
      edm::InputTag _inSeeds2;
      edm::InputTag _inPreId2;

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
ElectronSeedTrackRefUpdaterAndMerger::ElectronSeedTrackRefUpdaterAndMerger(const edm::ParameterSet& iConfig) :
  _inSeeds1(iConfig.getParameter< edm::InputTag > ("inSeeds1")),
  _inPreId1(iConfig.getParameter< edm::InputTag > ("inPreId1")),
  _inSeeds2(iConfig.getParameter< edm::InputTag > ("inSeeds2")),
  _inPreId2(iConfig.getParameter< edm::InputTag > ("inPreId2")),
  _targetTracks(iConfig.getParameter< edm::InputTag > ("targetTracks"))
{
   preidgsf_ = iConfig.getParameter<std::string>("PreGsfLabel");
   preidname_= iConfig.getParameter<std::string>("PreIdLabel");

   produces<ElectronSeedCollection>(preidgsf_);
   produces<PreIdCollection>(preidname_);
   //produces<edm::ValueMap<reco::PreIdRef> >(preidname_);

}


ElectronSeedTrackRefUpdaterAndMerger::~ElectronSeedTrackRefUpdaterAndMerger()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ElectronSeedTrackRefUpdaterAndMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace reco;

   auto_ptr<ElectronSeedCollection> output_preid(new ElectronSeedCollection);
   auto_ptr<PreIdCollection> output_preidinfo(new PreIdCollection);

   edm::Handle<reco::TrackCollection> hTargetTracks;
   iEvent.getByLabel( _targetTracks, hTargetTracks);

   edm::Handle< ElectronSeedCollection  > hSeeds1;
   iEvent.getByLabel( _inSeeds1, hSeeds1 );
   edm::Handle< ElectronSeedCollection  > hSeeds2;
   iEvent.getByLabel( _inSeeds2, hSeeds2 );

   std::vector< edm::Handle< ElectronSeedCollection > > colsSeed;
   colsSeed.push_back(hSeeds1);
   colsSeed.push_back(hSeeds2);


   edm::Handle<PreIdCollection> hPreId1;
   iEvent.getByLabel( _inPreId1, hPreId1);
   edm::Handle<PreIdCollection> hPreId2;
   iEvent.getByLabel( _inPreId2, hPreId2);

   std::vector< edm::Handle< PreIdCollection > > colsPreId;
   colsPreId.push_back(hPreId1);
   colsPreId.push_back(hPreId2);


   for (  std::vector< edm::Handle< PreIdCollection > >::iterator itCols=colsPreId.begin();
          itCols!=colsPreId.end();
          ++itCols    )
   {
     for (PreIdCollection::const_iterator it = (*itCols)->begin(); it!=(*itCols)->end(); ++ it){
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
        } else {
           std::cout << "XXXX whoops! Cannot set track ref!!" << std::endl;
        }
        output_preidinfo->push_back(newPreId);
     }
   }   


   for (  std::vector< edm::Handle< ElectronSeedCollection > >::iterator itCols=colsSeed.begin();
          itCols!=colsSeed.end();
          ++itCols    )
   {
     for (ElectronSeedCollection::const_iterator it = (*itCols)->begin(); it!=(*itCols)->end(); ++ it){
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
        } else {
           std::cout << "XXXX whoops! Cannot set track ref!!" << std::endl;
        }
        output_preid->push_back(newSeed);
     }
   }


   iEvent.put(output_preid,preidgsf_);
   iEvent.put(output_preidinfo,preidname_);

   //iEvent.put(newCol);

}

// ------------ method called once each job just before starting event loop  ------------
void
ElectronSeedTrackRefUpdaterAndMerger::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
ElectronSeedTrackRefUpdaterAndMerger::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronSeedTrackRefUpdaterAndMerger);
