// -*- C++ -*-
//
// Package:    GsfTrackMixer
// Class:      GsfTrackMixer
//
/**\class GsfTrackMixer GsfTrackMixer.cc TauAnalysis/GsfTrackMixer/src/GsfTrackMixer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Fri Apr  9 12:15:56 CEST 2010
// $Id: GsfTrackMixer.cc,v 1.2 2013/03/29 15:55:19 veelken Exp $
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
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include <DataFormats/Math/interface/deltaR.h>
//
// class decleration
//
using namespace reco;


class GsfTrackMixer : public edm::EDProducer {
   public:
      explicit GsfTrackMixer(const edm::ParameterSet&);
      ~GsfTrackMixer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::InputTag _col1;
      edm::InputTag _col2;


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
GsfTrackMixer::GsfTrackMixer(const edm::ParameterSet& iConfig) :
  _col1(iConfig.getParameter< edm::InputTag > ("collection1")),
  _col2(iConfig.getParameter< edm::InputTag > ("collection2"))
{

   // ?setBranchAlias?
   produces<reco::GsfTrackCollection>();
   produces<reco::TrackExtraCollection>();
   produces<reco::GsfTrackExtraCollection>();
   //produces<reco::GsfTrackCollection>().setBranchAlias( alias_ + "GsfTracks" );
   //produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
   //produces<reco::GsfTrackExtraCollection>().setBranchAlias( alias_ + "GsfTrackExtras" );
   //produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
   //produces<std::vector<Trajectory> >() ;
   //produces<TrajGsfTrackAssociationCollection>();
 

}


GsfTrackMixer::~GsfTrackMixer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GsfTrackMixer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace reco;

   // see RecoTracker/TrackProducer/plugins/GsfTrackProducer.cc
   // and RecoTracker/TrackProducer/src/GsfTrackProducerBase.cc
   //std::auto_ptr<TrackingRecHitCollection> outputRHColl (new TrackingRecHitCollection);
   std::auto_ptr<reco::GsfTrackCollection> outputTColl(new reco::GsfTrackCollection);
   std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
   std::auto_ptr<reco::GsfTrackExtraCollection> outputGsfTEColl(new reco::GsfTrackExtraCollection);
   //std::auto_ptr<std::vector<Trajectory> >    outputTrajectoryColl(new std::vector<Trajectory>);
 
   reco::TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
   reco::GsfTrackExtraRefProd rGsfTrackExtras = iEvent.getRefBeforePut<reco::GsfTrackExtraCollection>();
   reco::GsfTrackRefProd rTracks = iEvent.getRefBeforePut<reco::GsfTrackCollection>();


   edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
   edm::Ref<reco::GsfTrackExtraCollection>::key_type idxGsf = 0;	

   edm::Handle< reco::GsfTrackCollection  > hcol1, hcol2;
   iEvent.getByLabel( _col1, hcol1);
   iEvent.getByLabel( _col2, hcol2);

   std::vector< edm::Handle< reco::GsfTrackCollection  > > cols;
   cols.push_back(hcol1); 
   cols.push_back(hcol2); 

   for (  std::vector< edm::Handle< reco::GsfTrackCollection  > >::iterator itCols=cols.begin();
          itCols!=cols.end();
          ++itCols    )
   {
     for (GsfTrackCollection::const_iterator it = (*itCols)->begin(); it!=(*itCols)->end(); ++ it){
         GsfTrack gsfTrack = *it;
         TrackExtra te = *(it->extra());
         GsfTrackExtra ge = *(it->gsfExtra());

         reco::GsfTrackExtraRef terefGsf = reco::GsfTrackExtraRef ( rGsfTrackExtras, idxGsf ++ );
         reco::TrackExtraRef teref= reco::TrackExtraRef ( rTrackExtras, idx ++ );

         gsfTrack.setExtra( teref );
         gsfTrack.setGsfExtra( terefGsf );

         outputTColl->push_back(gsfTrack);
         outputTEColl->push_back(te);
         outputGsfTEColl->push_back(ge);   
     }
   }   



   iEvent.put(outputTColl);
   iEvent.put(outputTEColl);
   iEvent.put(outputGsfTEColl);

}

// ------------ method called once each job just before starting event loop  ------------
void
GsfTrackMixer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
GsfTrackMixer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(GsfTrackMixer);
