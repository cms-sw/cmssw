// -*- C++ -*-
//
// Package:    HSCPAnalyzer
// Class:      HSCPAnalyzer
// 
/**\class HSCPAnalyzer HSCPAnalyzer.cc SUSYBSMAnalysis/HSCPAnalyzer/src/HSCPAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Rizzi Andrea
//         Created:  Mon Sep 24 09:30:06 CEST 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"


#include <iostream>
#include <TH1F.h>
#include <TH2F.h>
//
// class decleration
//

using namespace reco;

class HSCPAnalyzer : public edm::EDAnalyzer {
   public:
      explicit HSCPAnalyzer(const edm::ParameterSet&);
      ~HSCPAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
 
      TH1F * h_pt;
      TH1F * h_mass;
      TH1F * h_massProton;
      TH2F * h_dedx;
      TH2F * h_dedxCtrl;
      TH1F * h_dedxMIP;
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
HSCPAnalyzer::HSCPAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


HSCPAnalyzer::~HSCPAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HSCPAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<reco::MuonCollection> pIn;
   iEvent.getByLabel("muons",pIn);
   const reco::MuonCollection & muons = * pIn.product();
 
   reco::MuonCollection::const_iterator muonIt = muons.begin();
   for(; muonIt != muons.end() ; ++muonIt)
    {
      TrackRef tkMuon = muonIt->track();
      TrackRef staMuon = muonIt->standAloneMuon();
      TrackRef combMuon = muonIt->combinedMuon();
      if(tkMuon.isNonnull())
           h_pt->Fill(tkMuon->pt()); 
    }


   Handle<TrackDeDxEstimateCollection> dedxH;
   iEvent.getByLabel("dedxTruncated40",dedxH);


   const TrackDeDxEstimateCollection & dedx = *dedxH.product();
   for(size_t i=0; i<dedx.size() ; i++)
    {
      if(dedx[i].first->normalizedChi2() < 5 && dedx[i].first->numberOfValidHits()>=8)
       {
         float dedxVal= dedx[i].second;
         float p= dedx[i].first->p();

         h_dedx->Fill(p, dedxVal);   
         h_dedxCtrl->Fill(p, dedxVal);   

         if(p > 5 && p < 30 )    h_dedxMIP->Fill( dedxVal);   

         float k=919/2.78*0.0012;
         float mass=p*sqrt(k*dedxVal-1);

         h_mass->Fill(mass); 
         
         if(p < 2. && mass > 0.200 )
          {
           h_massProton->Fill(mass);
          }
        }
    }

}


// ------------ method called once each job just before starting event loop  ------------
void 
HSCPAnalyzer::beginJob(const edm::EventSetup&)
{
  edm::Service<TFileService> fs;
  TFileDirectory subDir = fs->mkdir( "mySubDirectory" );
  h_pt =  subDir.make<TH1F>( "mu_pt"  , "p_{t}", 100,  0., 1000. );
  h_dedx =  subDir.make<TH2F>( "dedx_p"  , "\\frac{dE}{dX} vs p", 100,  0., 1500., 100,0,8 );
  h_dedxCtrl =  subDir.make<TH2F>( "dedx_lowp"  , "\\frac{dE}{dX} vs p", 100,  0., 3., 100,0,8 );
  h_dedxMIP =  subDir.make<TH1F>( "dedxMIP"  , "\\frac{dE}{dX}  ",100,0,8 );

  h_mass =  subDir.make<TH1F>( "mass"  , "Mass (dedx)", 100,  0., 1500.);
  h_massProton =  subDir.make<TH1F>( "massProton"  , "Proton Mass (dedx)", 100,  0., 2.);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCPAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPAnalyzer);
