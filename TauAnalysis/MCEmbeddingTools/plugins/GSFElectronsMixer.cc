// -*- C++ -*-
//
// Package:    GSFElectronsMixer
// Class:      GSFElectronsMixer
//
/**\class GSFElectronsMixer GSFElectronsMixer.cc TauAnalysis/GSFElectronsMixer/src/GSFElectronsMixer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Fri Apr  9 12:15:56 CEST 2010
// $Id: GSFElectronsMixer.cc,v 1.1 2011/10/13 08:29:03 fruboes Exp $
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

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

//
// class decleration
//

class GSFElectronsMixer : public edm::EDProducer {
   public:
      explicit GSFElectronsMixer(const edm::ParameterSet&);
      ~GSFElectronsMixer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::InputTag _electrons1;
      edm::InputTag _electrons2;

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
GSFElectronsMixer::GSFElectronsMixer(const edm::ParameterSet& iConfig) :
  _electrons1(iConfig.getParameter< edm::InputTag > ("col1")),
  _electrons2(iConfig.getParameter< edm::InputTag > ("col2"))
{

   produces<reco::GsfElectronCollection>();
}


GSFElectronsMixer::~GSFElectronsMixer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GSFElectronsMixer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   std::vector< edm::Handle<reco::GsfElectronCollection> > cols;
   edm::Handle<reco::GsfElectronCollection> tks1;
   iEvent.getByLabel( _electrons1, tks1);

   edm::Handle<reco::GsfElectronCollection> tks2;
   iEvent.getByLabel( _electrons2, tks2);

   cols.push_back(tks1);
   cols.push_back(tks2);

   std::auto_ptr<reco::GsfElectronCollection> finalCollection( new reco::GsfElectronCollection ) ;

   //std::cout << "##########################################\n";
   //int i  = 0;
   std::vector< edm::Handle< reco::GsfElectronCollection > >::iterator it = cols.begin();
   for(;it != cols.end(); ++it)
   {
     //std::cout << " col " << i++ << std::endl;
     for ( reco::GsfElectronCollection::const_iterator itT = (*it)->begin() ; itT != (*it)->end(); ++itT)
     {
       /*
       std::cout << " " << itT->vx()
           << " " << itT->vy()
           << " " << itT->vz()
           << " " << itT->pt()
           << std::endl;*/

       finalCollection->push_back(*itT);
     }

   }

   iEvent.put(finalCollection);

}

// ------------ method called once each job just before starting event loop  ------------
void
GSFElectronsMixer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
GSFElectronsMixer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(GSFElectronsMixer);
