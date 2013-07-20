// -*- C++ -*-
//
// Package:    HighPtTrackEcalDetIdProducer
// Class:      HighPtTrackEcalDetIdProducer
// 
/*\class HighPtTrackEcalDetIdProducer HighPtTrackEcalDetIdProducer.cc 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jie Chen
//         Created:  Mon Apr 12 16:41:46 CDT 2010
// $Id: HighPtTrackEcalDetIdProducer.cc,v 1.2 2013/02/27 22:47:59 wmtan Exp $
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
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

//
// class declaration
//

class HighPtTrackEcalDetIdProducer : public edm::EDProducer {
   public:
      explicit HighPtTrackEcalDetIdProducer(const edm::ParameterSet&);
      ~HighPtTrackEcalDetIdProducer();
      void beginRun(const edm::Run&, const edm::EventSetup&) override;
      void produce(edm::Event&, const edm::EventSetup&) override;
   private:

      edm::InputTag inputCollection_;
      const CaloTopology* caloTopology_;
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
HighPtTrackEcalDetIdProducer::HighPtTrackEcalDetIdProducer(const edm::ParameterSet& iConfig)
{
   inputCollection_ = iConfig.getParameter< edm::InputTag >("inputCollection");    ptcut_= iConfig.getParameter< double >("TrackPt");

    produces< DetIdCollection >() ;
   // TrackAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );
   trackAssociator_.useDefaultPropagator();
  
}


HighPtTrackEcalDetIdProducer::~HighPtTrackEcalDetIdProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

void
HighPtTrackEcalDetIdProducer::beginRun(const edm::Run & run, const edm::EventSetup & iSetup)  
{
   edm::ESHandle<CaloTopology> theCaloTopology;
   iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
   caloTopology_ = &(*theCaloTopology); 
}

// ------------ method called to produce the data  ------------
void
HighPtTrackEcalDetIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using reco::TrackCollection;
//   if(!iSetup) continue;
   Handle<TrackCollection> tkTracks;
   iEvent.getByLabel(inputCollection_,tkTracks);
   std::auto_ptr< DetIdCollection > interestingDetIdCollection( new DetIdCollection() ) ;
   for(TrackCollection::const_iterator itTrack = tkTracks->begin();
       itTrack != tkTracks->end();                      
       ++itTrack) {
        if(itTrack->pt()>ptcut_){
           TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *itTrack, parameters_, TrackDetectorAssociator::InsideOut);
           if(info.crossedEcalIds.size()==0) break;

           if(info.crossedEcalIds.size()>0){
              DetId centerId = info.crossedEcalIds.front();

              const CaloSubdetectorTopology* topology = caloTopology_->getSubdetectorTopology(DetId::Ecal,centerId.subdetId());
              const std::vector<DetId>& ids = topology->getWindow(centerId, 5, 5); 
              for ( std::vector<DetId>::const_iterator id = ids.begin(); id != ids.end(); ++id )
                 if(std::find(interestingDetIdCollection->begin(), interestingDetIdCollection->end(), *id) 
                    == interestingDetIdCollection->end()) 
                    interestingDetIdCollection->push_back(*id);            
           }
        }

   }
   iEvent.put(interestingDetIdCollection);

}
//define this as a plug-in
DEFINE_FWK_MODULE(HighPtTrackEcalDetIdProducer);
