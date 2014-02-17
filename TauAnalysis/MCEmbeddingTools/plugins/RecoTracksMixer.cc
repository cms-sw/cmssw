// -*- C++ -*-
//
// Package:    RecoTracksMixer
// Class:      RecoTracksMixer
//
/**\class RecoTracksMixer RecoTracksMixer.cc TauAnalysis/RecoTracksMixer/src/RecoTracksMixer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Fri Apr  9 12:15:56 CEST 2010
// $Id: RecoTracksMixer.cc,v 1.3 2010/11/08 16:03:27 friis Exp $
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

//
// class decleration
//

class RecoTracksMixer : public edm::EDProducer {
   public:
      explicit RecoTracksMixer(const edm::ParameterSet&);
      ~RecoTracksMixer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::InputTag _tracks1;
      edm::InputTag _tracks2;

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
RecoTracksMixer::RecoTracksMixer(const edm::ParameterSet& iConfig) :
  _tracks1(iConfig.getParameter< edm::InputTag > ("trackCol1")),
  _tracks2(iConfig.getParameter< edm::InputTag > ("trackCol2"))
{

   produces<reco::TrackCollection>();
}


RecoTracksMixer::~RecoTracksMixer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
RecoTracksMixer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   std::vector< edm::Handle<reco::TrackCollection> > cols;
   edm::Handle<reco::TrackCollection> tks1;
   iEvent.getByLabel( _tracks1, tks1);

   edm::Handle<reco::TrackCollection> tks2;
   iEvent.getByLabel( _tracks2, tks2);

   cols.push_back(tks1);
   cols.push_back(tks2);

   std::auto_ptr< reco::TrackCollection  > newCol(new reco::TrackCollection );

   //std::cout << "##########################################\n";
   //int i  = 0;
   std::vector< edm::Handle<reco::TrackCollection> >::iterator it = cols.begin();
   for(;it != cols.end(); ++it)
   {
     //std::cout << " col " << i++ << std::endl;
     for ( reco::TrackCollection::const_iterator itT = (*it)->begin() ; itT != (*it)->end(); ++itT)
     {
       /*
       std::cout << " " << itT->vx()
           << " " << itT->vy()
           << " " << itT->vz()
           << " " << itT->pt()
           << std::endl;*/

       newCol->push_back(*itT);
     }

   }

   iEvent.put(newCol);

}

// ------------ method called once each job just before starting event loop  ------------
void
RecoTracksMixer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
RecoTracksMixer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(RecoTracksMixer);
