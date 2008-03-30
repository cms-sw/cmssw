// -*- C++ -*-
//
// Package:    GenSimValidation
// Class:      GenSimValidation
// 
/**\class GenSimValidation GenSimValidation.cc SUSYBSMAnalysis/HSCP/src/GenSimValidation.cc

 Description: MC analysis of HSCP

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jie Chen
//         Created:  Fri Nov 9 09:30:06 CDT 2007
// Updated by Loic Quertenmont
//                   Sun Mar 30 11:14:00 CDT 2008


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"


#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"

#include <iostream>
#include <TH1F.h>
#include <TH2F.h>

//
// class decleration
//

using namespace reco;
using namespace std;
using namespace edm;

class HSCPMCAnalyzer : public edm::EDAnalyzer {
   public:
      explicit HSCPMCAnalyzer(const edm::ParameterSet&);
      ~HSCPMCAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      std::string GenJetAlgorithm;      

 //GEN
      TH1F * h_genhscp_pt; 
      TH1F * h_genhscp_eta; 
      TH1F * h_genhscp_beta; 
      TH1F * h_genhscp_met;
      TH1F * h_genhscp_met_nohscp;
      TH1F * h_genhscp_scaloret;
      TH1F * h_genhscp_scaloret_nohscp;

      TH1F * h_1sthardgenjet_pt;
      TH1F * h_1sthardgenjet_eta;
      TH1F * h_2ndhardgenjet_pt;
      TH1F * h_2ndhardgenjet_eta;
      TH1F * h_numgenjet;
      TH2F * h_numgenjet1sthardjetpt;
      TH2F * h_etmiss1sthardjetpt;
      TH2F * h_1sthardjetpt2ndhardjetpt;

      TH2F * h_HscpMetCorrelation;

 // SIM
      TH1F * h_simmu_pt; 
      TH1F * h_simmu_eta; 
      TH1F * h_simhscp_pt; 
      TH1F * h_simhscp_eta; 
      TH1F * h_simhscp_beta; 




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
HSCPMCAnalyzer::HSCPMCAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
    GenJetAlgorithm=iConfig.getParameter<string>( "GenJetAlgorithm" );

}


HSCPMCAnalyzer::~HSCPMCAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HSCPMCAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;


//------------ Event Info ----------------

  //get Jets info in the events.
  Handle<GenJetCollection> genJets;
  iEvent.getByLabel( GenJetAlgorithm, genJets );
  //Loop over the two leading GenJets and fill some histograms
  int jetInd = 0;
  int hardJetInd=0;
  double hardestjet_pt=0;
  double secondhardjet_pt=0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end(); ++ gen ) {

    if(jetInd==0){
      hardestjet_pt=gen->pt();
      h_1sthardgenjet_pt->Fill( gen->pt() );   
      h_1sthardgenjet_eta->Fill( gen->eta() );
    }
    if(jetInd==1){
      secondhardjet_pt=gen->pt();
      h_2ndhardgenjet_pt->Fill( gen->pt() );   
      h_2ndhardgenjet_eta->Fill( gen->eta() );
    }
    if(gen->pt()>30 && abs(gen->eta())<3) hardJetInd++;
    jetInd++;
  }
  h_numgenjet->Fill(jetInd);
  h_numgenjet1sthardjetpt->Fill(hardestjet_pt,hardJetInd);
  h_1sthardjetpt2ndhardjetpt->Fill(hardestjet_pt,secondhardjet_pt);

  /*
  Handle<GenMETCollection> genmet;
  iEvent.getByLabel("genMet",genmet);
  const GenMETCollection *genmetcoll=genmet.product();
  const GenMET gMET = genmetcoll->front();
  genMET->invisibleEnergy();

  */

//------------ Track Info ----------------
  //get gen track info
  Handle<CandidateCollection> genParticles;
  iEvent.getByLabel( "genParticleCandidates", genParticles );
  double hscppx=0;
  double hscppy=0;
  double missingpx=0;
  double missingpy=0;
  double missingpx_nohscp=0;
  double missingpy_nohscp=0;
  double scalorEt=0;
  double scalorEt_nohscp=0;

  for( size_t i = 0; i < genParticles->size(); ++ i ) {
    const Candidate & p = (*genParticles)[ i ];
    if (p.status()==1 ){ //for final state particles
      if(abs(p.pdgId())>1000000){
        h_genhscp_pt->Fill(p.pt());
        h_genhscp_eta->Fill(p.eta());
        h_genhscp_beta->Fill(p.p()/p.energy());

        hscppx+=p.px();
        hscppy+=p.py();
      }
      if(abs(p.pdgId())!=12 && abs(p.pdgId())!=14 && abs(p.pdgId())!=16){
	missingpx-=p.px();
	missingpy-=p.py();
	scalorEt+=p.et();
	if(abs(p.pdgId())<1000000){
	  missingpx_nohscp-=p.px();
	  missingpy_nohscp-=p.py();
	  scalorEt_nohscp+=p.et();
	}
      }

    }

  }
  h_genhscp_met->Fill(sqrt(missingpx*missingpx+missingpy*missingpy));;
  h_genhscp_met_nohscp->Fill(sqrt(missingpx_nohscp*missingpx_nohscp+missingpy_nohscp*missingpy_nohscp));;
  h_genhscp_scaloret->Fill(scalorEt);
  h_genhscp_scaloret_nohscp->Fill(scalorEt_nohscp);

  h_etmiss1sthardjetpt->Fill(sqrt(missingpx_nohscp*missingpx_nohscp+missingpy_nohscp*missingpy_nohscp),hardestjet_pt);


  double Hscp_phi = atan2(hscppx,hscppy);
  double MET_phi  = atan2(missingpx_nohscp, missingpy_nohscp);

  if(Hscp_phi<0)Hscp_phi+=6.28318;
  if(MET_phi<0)  MET_phi+=6.28318;

  double DeltaPhi = Hscp_phi - MET_phi;
  double DeltaPt  = sqrt(pow(hscppx,2)+pow(hscppy,2)) - sqrt(pow(missingpx_nohscp,2)+pow(missingpy_nohscp,2));

   h_HscpMetCorrelation->Fill(DeltaPhi,DeltaPt);





  //get sim track infos
  Handle<edm::SimTrackContainer> simTracksHandle;
  iEvent.getByLabel("g4SimHits",simTracksHandle);
  const SimTrackContainer simTracks = *(simTracksHandle.product());

  SimTrackContainer::const_iterator simTrack;
  missingpx=0;
  missingpy=0;


  for (simTrack = simTracks.begin(); simTrack != simTracks.end(); ++simTrack){

      if (abs((*simTrack).type()) > 1000000) {
  //      h_simhscp_pt->Fill((*simTrack).momentum().perp());
        h_simhscp_pt->Fill((*simTrack).momentum().pt());
        h_simhscp_eta->Fill((*simTrack).momentum().eta());
//        h_simhscp_beta->Fill((*simTrack).momentum().vect().mag()/(*simTrack).momentum().e());
        h_simhscp_beta->Fill((*simTrack).momentum().P()/(*simTrack).momentum().e());


      }

      if (abs((*simTrack).type()) == 13) {
        h_simmu_pt->Fill((*simTrack).momentum().pt());
        h_simmu_eta->Fill(((*simTrack).momentum().eta()));
    }
  }


}


// ------------ method called once each job just before starting event loop  ------------
void 
HSCPMCAnalyzer::beginJob(const edm::EventSetup&)
{
  edm::Service<TFileService> fs;

  TFileDirectory subDir1 = fs->mkdir( "Gen" );

  //gen plots
  h_genhscp_pt = subDir1.make<TH1F>( "hscp_pt"  , "p_{t} hscp", 100,  0., 1500. );
  h_genhscp_eta =  subDir1.make<TH1F>( "hscp_eta"  , "\\eta hscp" , 50,  -4., 4. );
  h_genhscp_beta =  subDir1.make<TH1F>( "hscp_beta"  , "\\beta hscp" , 100, 0., 1. );
  h_genhscp_met =  subDir1.make<TH1F>( "hscp_met"  , "missing E_{T} hscp" , 100, 0., 1500. );
  h_genhscp_met_nohscp =  subDir1.make<TH1F>( "hscp_met_nohscp"  , "missing E_{T} w/o hscp" , 100, 0., 1500. );
  h_genhscp_scaloret =  subDir1.make<TH1F>( "hscp_scaloret"  , "scalor E_{T} sum" , 100, 0., 1500. );
  h_genhscp_scaloret_nohscp =  subDir1.make<TH1F>( "hscp_scaloret_nohscp"  , "scalor E_{T} sum w/o hscp" , 100, 0., 1500. );

  //genjets plots
  h_1sthardgenjet_pt = subDir1.make<TH1F>( "1sthardgenjet_pt"  , "p_{t} 1st hard genjet", 100,  0., 1500. );
  h_1sthardgenjet_eta = subDir1.make<TH1F>( "1sthardgenjet_eta"  , "\\eta 1st hard genjet", 50, -4., 4. );
  h_2ndhardgenjet_pt = subDir1.make<TH1F>( "2ndhardgenjet_pt"  , "p_{t} 2nd hard genjet", 100,  0., 1500. );
  h_2ndhardgenjet_eta = subDir1.make<TH1F>( "2ndhardgenjet_eta"  , "\\eta 2nd hard genjet", 50,  -4., 4. );
  h_numgenjet = subDir1.make<TH1F>( "numgenjets"  , "# genjet", 100,  0., 100. );
  h_numgenjet1sthardjetpt = subDir1.make<TH2F>( "numgenjet1sthardjetpt","#hardjets vs. hardest jet p_{t}",100,  0., 1500., 100,  0., 50.);
  h_etmiss1sthardjetpt = subDir1.make<TH2F>( "etmiss1sthardjetpt","E_{t} Miss vs. hardest jet p_{t}",100,  0., 1500., 100,  0., 1500.);
  h_1sthardjetpt2ndhardjetpt = subDir1.make<TH2F>( "1sthardjetpt2ndhardjetp"," hardest jet p_{t} vs. 2nd hardest jet p_{t}",100,  0., 1500., 100,  0., 1500.);
  h_HscpMetCorrelation =  subDir1.make<TH2F>( "HscpMetCorrelation"," HSCP - MET : DeltaPhi Vs DeltaPt", 500,  -3.14, 3.14, 800,  -100., 100.);

  TFileDirectory subDir2 = fs->mkdir( "Sim" );
  //simtrack plots
  h_simmu_pt =  subDir2.make<TH1F>( "mu_sim_pt"  , "p_{t} mu", 100,  0., 1500. );
  h_simmu_eta  =  subDir2.make<TH1F>( "mu_sim_eta"  , "\\eta mu", 50,  -4., 4. );
  h_simhscp_pt = subDir2.make<TH1F>( "hscp_pt"  , "p_{t} hscp", 100,  0., 1500. );
  h_simhscp_eta =  subDir2.make<TH1F>( "hscp_eta"  , "\\eta hscp" , 50,  -4., 4. );
  h_simhscp_beta =  subDir2.make<TH1F>( "hscp_beta"  , "\\beta hscp" , 100, 0., 1. );

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCPMCAnalyzer::endJob() {

} 

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPMCAnalyzer);
