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
// $Id: HSCPAnalyzer.cc,v 1.2 2007/10/10 10:14:20 arizzi Exp $
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
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include <iostream>
#include <vector>
#include <TH1F.h>
#include <TH2F.h>
//
// class decleration
//

using namespace reco;
using namespace std;
using namespace edm;

class HSCPAnalyzer : public edm::EDAnalyzer {
   public:
      explicit HSCPAnalyzer(const edm::ParameterSet&);
      ~HSCPAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::InputTag m_dedxSrc;
      bool m_haveSimTracks;
      bool m_useWeights;

      TH1F * h_pt;
      TH1F *  h_massSel;
      TH1F *  h_massSelFit;
      TH1F * h_mass;
      TH1F * h_massFit;
      TH1F * h_massProton;
      TH1F * h_massProtonFit;
      TH2F * h_dedx;
      TH2F * h_dedxCtrl;
      TH2F * h_dedxFit;
      TH2F * h_dedxFitCtrl;
      TH1F * h_dedxMIP;
      TH1F * h_dedxFitMIP;
      TH1F * h_simmu_pt; 
      TH1F * h_simmu_eta; 
      TH1F * h_simhscp_pt; 
      TH1F * h_simhscp_eta; 
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
    m_dedxSrc=iConfig.getParameter<InputTag>("dedxSrc");
    m_haveSimTracks=iConfig.getParameter<bool>("haveSimTracks");
    m_useWeights=iConfig.getParameter<bool>("useWeights");

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
   double w=1.;
   if(m_useWeights)
   {
       Handle<double>  weightH;
       Handle<double>  scaleH;
       iEvent.getByLabel("genEventWeight",weightH);
       iEvent.getByLabel("genEventScale",scaleH);

       w= * weightH.product()      ;
       double s= * scaleH.product()      ;
       cout << "W: " << w << "   S: "<< s << endl;
   }

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
           h_pt->Fill(tkMuon->pt(),w); 
    }


   Handle<TrackDeDxEstimateCollection> dedxH;
//   iEvent.getByLabel("dedxTruncated40",dedxH);
   iEvent.getByLabel(m_dedxSrc,dedxH);
   Handle<vector<float> >  dedxFitH;
   iEvent.getByLabel("dedxFit",dedxFitH);

   const TrackDeDxEstimateCollection & dedx = *dedxH.product();
   const vector<float> & dedxFit = *dedxFitH.product();

   for(size_t i=0; i<dedx.size() ; i++)
    {
      if(dedx[i].first->normalizedChi2() < 5 && dedx[i].first->numberOfValidHits()>=8)
       {
         float dedxVal= dedx[i].second;
         float dedxFitVal= dedxFit[i];
         float p= dedx[i].first->p();

         h_dedx->Fill(p, dedxVal,w);   
         h_dedxCtrl->Fill(p, dedxVal,w);   

         h_dedxFit->Fill(p, dedxFitVal,w);   
         h_dedxFitCtrl->Fill(p, dedxFitVal,w);   

         if(p > 5 && p < 30 )  
          {
             h_dedxMIP->Fill( dedxVal,w);   
             h_dedxFitMIP->Fill( dedxFitVal,w);   
             if(dedxVal >3.22)
              {
              std::cout << dedx[i].first->normalizedChi2() << " " << dedx[i].first->numberOfValidHits() << " " << p <<std::endl;
              }
          }
         float k=919/2.82*0.0012;
         float k2=919/2.58*0.0012;
         float mass=p*sqrt(k*dedxVal-1);
         float mass2=p*sqrt(k2*dedxFitVal-1);

         h_mass->Fill(mass,w); 
         h_massFit->Fill(mass2,w); 
        if(p > 30 && dedxVal > 3.45  ) {  h_massSel->Fill(mass,w); }
        if(p > 30 && dedxFitVal > 3.3 ) { h_massSelFit->Fill(mass2,w);     }        

         if(p < 1.2 && mass2 > 0.200 )
          {
           h_massProton->Fill(mass,w);
          }
         if(p < 1.2 && mass > 0.200 )
          {
           h_massProtonFit->Fill(mass2,w);
          }
        }
    }

if(m_haveSimTracks)
 {
  Handle<edm::SimTrackContainer> simTracksHandle;
  iEvent.getByLabel("g4SimHits",simTracksHandle);
  const SimTrackContainer simTracks = *(simTracksHandle.product());

  SimTrackContainer::const_iterator simTrack;
  for (simTrack = simTracks.begin(); simTrack != simTracks.end(); ++simTrack){

      if (abs((*simTrack).type()) > 1000000) {
        h_simhscp_pt->Fill((*simTrack).momentum().perp(),w);
        h_simhscp_eta->Fill(((*simTrack).momentum().eta()),w);
      }

      if (abs((*simTrack).type()) == 13) {
        h_simmu_pt->Fill((*simTrack).momentum().perp(),w);
        h_simmu_eta->Fill(((*simTrack).momentum().eta()),w);
    }
  }
}


}


// ------------ method called once each job just before starting event loop  ------------
void 
HSCPAnalyzer::beginJob(const edm::EventSetup&)
{
  edm::Service<TFileService> fs;
  TFileDirectory subDir = fs->mkdir( "Reco" );
  h_pt =  subDir.make<TH1F>( "mu_pt"  , "p_{t}", 100,  0., 1500. );
  h_dedx =  subDir.make<TH2F>( "dedx_p"  , "\\frac{dE}{dX} vs p", 100,  0., 1500., 100,0,8 );
  h_dedxCtrl =  subDir.make<TH2F>( "dedx_lowp"  , "\\frac{dE}{dX} vs p", 100,  0., 3., 100,0,8 );
  h_dedxMIP =  subDir.make<TH1F>( "dedxMIP"  , "\\frac{dE}{dX}  ",100,0,8 );
  h_dedxFit =  subDir.make<TH2F>( "dedx_p_FIT"  , "\\frac{dE}{dX} vs p", 100,  0., 1500., 100,0,8 );
  h_dedxFitCtrl =  subDir.make<TH2F>( "dedx_lowp_FIT"  , "\\frac{dE}{dX} vs p", 100,  0., 3., 100,0,8 );
  h_dedxFitMIP =  subDir.make<TH1F>( "dedxMIP_FIT"  , "\\frac{dE}{dX}  ",100,0,8 );

  h_massSel =  subDir.make<TH1F>( "massSel"  , "Mass (dedx), with selection", 100,  0., 1500.);
  h_massSelFit =  subDir.make<TH1F>( "massSel_FIT"  , "Mass (dedx), with selection", 100,  0., 1500.);
  h_mass =  subDir.make<TH1F>( "mass"  , "Mass (dedx)", 100,  0., 1500.);
  h_massFit =  subDir.make<TH1F>( "mass_FIT"  , "Mass (dedx)", 100,  0., 1500.);
  h_massProton =  subDir.make<TH1F>( "massProton"  , "Proton Mass (dedx)", 100,  0., 2.);
  h_massProtonFit =  subDir.make<TH1F>( "massProton_FIT"  , "Proton Mass (dedx)", 100,  0., 2.);

  TFileDirectory subDir2 = fs->mkdir( "Sim" );
  h_simmu_pt =  subDir2.make<TH1F>( "mu_sim_pt"  , "p_{t} mu", 100,  0., 1500. );
  h_simmu_eta  =  subDir2.make<TH1F>( "mu_sim_eta"  , "\\eta mu", 50,  -4., 4. );
  h_simhscp_pt = subDir2.make<TH1F>( "mu_hscp_pt"  , "p_{t} hscp", 100,  0., 1500. );
  h_simhscp_eta =  subDir2.make<TH1F>( "mu_hscp_eta"  , "\\eta hscp" , 50,  -4., 4. );

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCPAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPAnalyzer);
