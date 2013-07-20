// -*- C++ -*-
//
// Package:    ReduceHcalRecHitCollectionProducer
// Class:      ReduceHcalRecHitCollectionProducer
// 
/*\class ReduceHcalRecHitCollectionProducer ReduceHcalRecHitCollectionProducer.cc 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jie Chen
//         Created:  Mon Apr 12 16:41:46 CDT 2010
// $Id: ReduceHcalRecHitCollectionProducer.cc,v 1.2 2013/02/27 22:47:59 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include <iostream>

//
// class declaration
//

class ReduceHcalRecHitCollectionProducer : public edm::EDProducer {
   public:
      explicit ReduceHcalRecHitCollectionProducer(const edm::ParameterSet&);
      ~ReduceHcalRecHitCollectionProducer();
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
   private:
      edm::InputTag recHitsLabel_;
      std::string reducedHitsCollection_;
      edm::InputTag inputCollection_;
      TrackDetectorAssociator trackAssociator_;
      TrackAssociatorParameters parameters_;
      double  ptcut_;
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
ReduceHcalRecHitCollectionProducer::ReduceHcalRecHitCollectionProducer(const edm::ParameterSet& iConfig)
{
  recHitsLabel_ = iConfig.getParameter< edm::InputTag > ("recHitsLabel");

  reducedHitsCollection_ = iConfig.getParameter<std::string>("reducedHitsCollection");
  
   //register your products
  produces< HBHERecHitCollection > (reducedHitsCollection_) ;

    inputCollection_ = iConfig.getParameter< edm::InputTag >("inputCollection");    ptcut_= iConfig.getParameter< double >("TrackPt");

    produces< DetIdCollection >() ;
   // TrackAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );
   trackAssociator_.useDefaultPropagator();
 
}


ReduceHcalRecHitCollectionProducer::~ReduceHcalRecHitCollectionProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ReduceHcalRecHitCollectionProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   using namespace std;

   using reco::TrackCollection;

   Handle<HBHERecHitCollection> recHitsHandle;
   iEvent.getByLabel(recHitsLabel_,recHitsHandle);
   if( !recHitsHandle.isValid() ) 
     {
       edm::LogError("ReduceHcalRecHitCollectionProducer") << "RecHit collection not found";
       return;
     }
   
   //Create empty output collections
   std::auto_ptr< HBHERecHitCollection > miniRecHitCollection (new HBHERecHitCollection) ;
    
//loop through tracks. 
   Handle<TrackCollection> tkTracks;
   iEvent.getByLabel(inputCollection_,tkTracks);
   std::auto_ptr< DetIdCollection > interestingDetIdCollection( new DetIdCollection() ) ;
   for(TrackCollection::const_iterator itTrack = tkTracks->begin();
       itTrack != tkTracks->end();                      
       ++itTrack) {
        if(itTrack->pt()>ptcut_){
  
           TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *itTrack, parameters_, TrackDetectorAssociator::InsideOut);
  
          if(info.crossedHcalIds.size()>0){
             //loop through hits in the cone
             for(std::vector<const HBHERecHit*>::const_iterator hit = info.hcalRecHits.begin(); 
                 hit != info.hcalRecHits.end(); ++hit)
             {
                DetId hitid=(*hit)->id();
                HBHERecHitCollection::const_iterator iRecHit = recHitsHandle->find(hitid);
                if ( (iRecHit != recHitsHandle->end()) && (miniRecHitCollection->find(hitid) == miniRecHitCollection->end()) )
                   miniRecHitCollection->push_back(*iRecHit);
             }
             

          }
        }
   }

   iEvent.put( miniRecHitCollection,reducedHitsCollection_ );



}
//define this as a plug-in
DEFINE_FWK_MODULE(ReduceHcalRecHitCollectionProducer);
