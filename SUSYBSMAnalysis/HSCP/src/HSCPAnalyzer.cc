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
// $Id: HSCPAnalyzer.cc,v 1.27 2009/02/04 10:50:57 delaer Exp $
//
//


// system include files
#include <memory>
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
//#include "SimDataFormats/GeneratorProducts/interface/GenInfoProduct.h"

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "Math/GenVector/VectorUtil.h"

#include <TH1F.h>
#include <TH2F.h>
#include <TF1.h>

#include <iostream>
#include <vector>

//
// class decleration
//

using namespace susybsm;
using namespace reco;
using namespace std;
using namespace edm;

class HSCPStandardPlots {
 public:
  HSCPStandardPlots(TFileDirectory  subDir);
  void fill(const HSCParticle & hscp, double w);

 private:
  TH2F * h_massVsMass;
  TH2F * h_betaVsBeta;
  TH2F * h_massVsBeta;
  TH2F * h_massVsPtError;
  TH2F * h_massVsMassError;
  TH2F * h_massVsPt;
  TH2F * h_massVsBeta_tk;
  TH2F * h_massVsPtError_tk;
  TH2F * h_massVsMassError_tk;

 public:
  TH1F * h_massTk;
  TH1F * h_massDt;
  TH1F * h_massAvg;
  TH1F * h_tofBeta;
  TH1F * h_tofInvBetaErr;
  TH1F * h_dedxHits;
  TH1F * h_dedxBeta;
  TH2F * h_dedxHitsBeta;
  TH1F * h_deltaBeta;
  TH1F * h_deltaBetaInv;
  TH1F * h_tofInvBeta;
  TH1F * h_tofBetaPull;

};

class CutMonitor {
 public:
  CutMonitor(std::string name,edm::Service<TFileService> fs):
    m_name(name), m_plots(fs->mkdir(name)), m_newevent(true),m_evCounter(0),m_candCounter(0),m_tot(0) {}
 
  void newEvent(float w) {
    m_newevent=true; 
    m_tot+=w; 
  }
  
  void passed(const HSCParticle & hscp, double w) {
    m_plots.fill(hscp,w);  
    if(m_newevent) m_evCounter+=w;
    m_candCounter+=w;
    m_newevent=false;
  }

  void print() {
    cout << fixed << setprecision(1) << setw(6) <<  m_candCounter << "(" << fixed << setprecision(1) << setw(6) << m_evCounter << ")";
  }
  
  void printEff() {
    cout << fixed << setprecision(3) << setw(5) <<  m_candCounter/m_tot << "(" << fixed << setprecision(3) << setw(5) << m_evCounter/m_tot << ")";
  }
 
  void printName() {
    cout << m_name;
  }

 private:
  std::string m_name;
  HSCPStandardPlots m_plots;
  bool m_newevent;
  double m_evCounter;
  double m_candCounter;
  double m_tot;
}; 

HSCPStandardPlots::HSCPStandardPlots(TFileDirectory  subDir) {
  h_massVsMass =          subDir.make<TH2F>("tof_mass_vs_dedx_mass","Mass tof vs Mass dedx", 100,0,1200,100,0,1200);
  h_massVsBeta =          subDir.make<TH2F>("avgMass_vs_avgBeta","Mass(avg) vs Beta(avg)", 100,0,1200,50,0,1);
  h_massVsPtError =       subDir.make<TH2F>("avgMass_vs_ptError","Mass(avg) vs log(ptError) (beta < 0.85)", 100,0,1200,25,0,10);
  h_massVsPt =            subDir.make<TH2F>("mass_vs_pt","Mass vs pt", 100,0,1200,100,0,1500);
  h_massVsMassError =     subDir.make<TH2F>("avgMass_vs_MassError","Mass(avg) vs log(masstError)", 100,0,1200,100,0,2);
  h_massVsBeta_tk =       subDir.make<TH2F>("tkMass_vs_tkBeta","Mass(tk) vs Beta(tk)", 100,0,1200,50,0,1);
  h_massVsPtError_tk =    subDir.make<TH2F>("tkMass_vs_ptError","Mass(tk) vs log(ptError) (beta < 0.85)", 100,0,1200,25,0,10);
  h_massVsMassError_tk =  subDir.make<TH2F>("tkMass_vs_MassError","Mass(tk) vs mass error", 100,0,1200,100,0,2);
  h_betaVsBeta =          subDir.make<TH2F>("tof_beta_vs_beta","INVBeta tof vs INVbeta dedx (Pt>30)", 100,0,3,100,0,3);

  h_massTk =              subDir.make<TH1F>("mass_tk","Mass tk", 100,0,1200);
  h_massDt =              subDir.make<TH1F>("mass_dt","Mass dt", 100,0,1200);
  h_massAvg =             subDir.make<TH1F>("mass_avg","Mass avg", 100,0,1200);

  h_dedxHits =            subDir.make<TH1F>("dedx_hits","# hits dedx", 25,-0.5,24.5);
  h_dedxBeta =            subDir.make<TH1F>("dedx_beta","Beta tk", 100,0,1);
  h_dedxHitsBeta =        subDir.make<TH2F>("dedx_hits_vs_beta","dedx #hits vs beta tk", 25,-0.5,24.5,100,0,1);

  h_deltaBeta =           subDir.make<TH1F>("delta_beta","Delta Beta", 200,-1,1);
  h_deltaBetaInv =        subDir.make<TH1F>("delta_betaInv","Delta BetaInv", 200,-3,3);

  h_tofBeta =             subDir.make<TH1F>( "tof_beta"  , " tof beta  ",100,0,1);
  h_tofInvBeta =          subDir.make<TH1F>( "tof_Invbeta"  , " tof beta  ",500,0,5);
  h_tofInvBetaErr =       subDir.make<TH1F>( "tof_inv_beta_err"  , " tof beta err  ",100,0,1);
  h_tofBetaPull =         subDir.make<TH1F>( "tof_beta_pull"  , " tof beta pull  ",100,-10,10);
}

void HSCPStandardPlots::fill(const HSCParticle & hscp, double w) {
  double avgMass = (hscp.massDt()+hscp.massTk())/2.;
  double avgMassError;
  double ptMassError=(hscp.Tk().track()->ptError()/hscp.Tk().track()->pt());
  ptMassError*=ptMassError;
  double ptMassError2=(hscp.Dt().first->track()->ptError()/hscp.Dt().first->track()->pt());
  ptMassError2*=ptMassError2;
  double ib2 = hscp.Dt().second.invBeta*hscp.Dt().second.invBeta;
  double dtMassError=hscp.Dt().second.invBetaErr*(ib2/sqrt(ib2-1)) ;
  dtMassError*= dtMassError;
  double dedxError = 0.2*sqrt(10./hscp.Tk().nDedxHits())*0.4/hscp.Tk().invBeta2();
  double tkMassError = dedxError/(2.*hscp.Tk().invBeta2()-1);
  tkMassError*=tkMassError;
  avgMassError=sqrt(ptMassError/4+ptMassError2/4.+dtMassError/4.+tkMassError/4.);
  h_massTk->Fill(hscp.massTk(),w);
  h_massDt->Fill(hscp.massDt(),w);
  h_massAvg->Fill(avgMass,w);
  h_dedxHits->Fill(hscp.Tk().nDedxHits(),w);
  h_dedxBeta->Fill(1./sqrt(hscp.Tk().invBeta2()),w); 
  h_dedxHitsBeta->Fill(hscp.Tk().nDedxHits(),1./sqrt(hscp.Tk().invBeta2()),w);
  h_deltaBeta->Fill(1./hscp.Dt().second.invBeta-1./sqrt(hscp.Tk().invBeta2()),w);
  h_deltaBetaInv->Fill(hscp.Dt().second.invBeta-sqrt(hscp.Tk().invBeta2()),w);
  h_massVsMass->Fill(hscp.massDt(),hscp.massTk(),w);
  h_massVsBeta->Fill(avgMass,2./(sqrt(hscp.Tk().invBeta2())+hscp.Dt().second.invBeta),w);
  h_massVsPtError->Fill(avgMass,log10(hscp.Tk().track()->ptError()),w);
  h_massVsPt->Fill(avgMass,hscp.Tk().track()->pt(),w);
  h_massVsMassError->Fill((hscp.massDt()+hscp.massTk())/2.,avgMassError,w);
  h_massVsMassError_tk->Fill(hscp.massTk(),tkMassError,w);
  h_massVsBeta_tk->Fill(hscp.massTk(),1./sqrt(hscp.Tk().invBeta2()),w);
  h_massVsPtError_tk->Fill(hscp.massTk(),log10(hscp.Tk().track()->ptError()),w);
  h_betaVsBeta->Fill(hscp.Dt().second.invBeta,sqrt(hscp.Tk().invBeta2()),w);
  h_tofBeta->Fill(1./hscp.Dt().second.invBeta,w);
  h_tofInvBetaErr->Fill(hscp.Dt().second.invBetaErr,w);
  h_tofInvBeta->Fill(hscp.Dt().second.invBeta,w);
  if(hscp.Dt().second.invBeta !=0 && hscp.Dt().second.invBetaErr !=0 && hscp.Dt().second.invBetaErr < 1000.)
    h_tofBetaPull->Fill((hscp.Dt().second.invBeta-1)/hscp.Dt().second.invBetaErr,w);
}

class HSCPAnalyzer : public edm::EDAnalyzer {
   public:
      explicit HSCPAnalyzer(const edm::ParameterSet&);
      ~HSCPAnalyzer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      double cutMin(TH1F * h, double ci);

      // ----------member data ---------------------------
      edm::InputTag m_dedxSrc;
      bool m_haveSimTracks;
      bool m_useWeights;
       
      // RECO DEDX
      TH1F * h_pt;
      TH1F * h_dedxMassSel;
      TH1F * h_dedxMass;
      TH1F * h_dedxMassMu;
      TH1F * h_dedxMassProton;
      TH2F * h_dedx;
      TH2F * h_dedxCtrl;
      TH1F * h_dedxMIP;
      TH2F * h_dedxMassMuVsPtError;
      TH1F * h_dedxMIPbeta;
      TH1F * h_dedxMIPbetaCut;
	
      // RECO TOF
      TH2F * h_tofBetap;
      TH2F * h_tofBetaPullp;
      TH2F * h_tofMassp;
      TH1F * h_tofMass;
      TH1F * h_tofMass2;
      TH1F * h_tofBeta;
      TH1F * h_tofBetaErr;
      TH1F * h_tofBetaPull;
      TH1F * h_tofBetaPullCut;
      TH2F * h_tofBetapCut;
      TH1F * h_tofMassCut;
      TH1F * h_tofBetaCut;
      TH2F * h_tofBetaPullpCut;
      TH1F * h_tofPtSta;
      TH1F * h_tofPtComb;

      //ANALYSIS
      TH1F * h_pSpectrumAfterSelection[6]; 
      TH1F * h_massAfterSelection[6];
      TH2F * h_massVsMass;
      TH2F * h_betaVsBeta;
      TH2F * h_massVsMassSel;
      TH2F * h_massVsBeta;
      TH2F * h_massVsPtError;
      TH2F * h_massVsMassError;
      TH1F * h_tkmu_pt;
      TH1F * h_stamu_pt;
      TH1F * h_combmu_pt;

      //ANALYSIS TK
      TH2F * h_massVsBeta_tk;
      TH2F * h_massVsPtError_tk;
      TH2F * h_massVsMassError_tk;
      	
      //Counters
      double selected;
      double selectedTOF;
      double selectedDedx;
      double tot;
      double selectedAfterCut[20];

      //Standard plots
      CutMonitor * cuts[40];

      // SIM
      TH1F * h_simmu_pt; 
      TH1F * h_simmu_eta; 
      TH1F * h_simhscp_pt; 
      TH1F * h_simhscp_eta; 
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
    tot =0;
    selected = 0;
    selectedTOF = 0;
    selectedDedx = 0;
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
void HSCPAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
/* 
  Handle< double > genFilterEff;
  iEvent.getByLabel( "genEventRunInfo", "FilterEfficiency", genFilterEff);
  double filter_eff = *genFilterEff;
   
  Handle< double > genCrossSect;
  iEvent.getByLabel( "genEventRunInfo", "PreCalculatedCrossSection", genCrossSect); 
  double external_cross_section = *genCrossSect;
   
  Handle< double > agenCrossSect;
  iEvent.getByLabel( "genEventRunInfo", "PreCalculatedCrossSection", agenCrossSect); 
  double auto_cross_section = *agenCrossSect;
*/
    
  //double auto_cross_section = gi->cross_section(); // is automatically calculated at the end of each RUN --  units in nb
  //double external_cross_section = gi->external_cross_section(); // is the precalculated one written in the cfg file -- units is pb
  //double filter_eff = gi->filter_efficiency();
  //cout << "auto Xsec: " << auto_cross_section << "nb    precalc Xsev "<< external_cross_section << "pb" << "    filter eff: "<<filter_eff <<endl;

  //Event flags
  int dedxSelectionLevel = -1;
  //bool highptmu =false; 

  // Compute the weight
  double w=1.;
  if(m_useWeights) {
    Handle<double> xsHandle;
    iEvent.getByLabel ("genEventRunInfo","PreCalculatedCrossSection", xsHandle);
    Handle<double> effHandle;
    iEvent.getByLabel ("genEventRunInfo","FilterEfficiency", effHandle);
    Handle<double> weightHandle;
    iEvent.getByLabel ("weight", weightHandle);
    Handle<int> procIdHandle;
    iEvent.getByLabel("genEventProcID",procIdHandle);
    w = * weightHandle;
    if(w>10) cout << "HIGH Weight: " << w << " Proc id = " << *procIdHandle << " eff " << *effHandle << "   Xsec "<< *xsHandle << endl;
/*
    Handle<double>  weightH;
    Handle<double>  scaleH;
    iEvent.getByLabel("genEventWeight",weightH);
    iEvent.getByLabel("genEventScale",scaleH);
    w= * weightH.product()      ;
    double s= * scaleH.product()      ;
*/
  }

  // Initialize cut monitors
  for(int i=0;i<40;i++) if(cuts[i]) cuts[i]->newEvent(w);

  // Count the event
  tot+=w;

  // Sel before cuts
  selectedAfterCut[0]+=w;

  // Load the collection of candidates
  Handle<HSCParticleCollection> hscpH;
  iEvent.getByLabel("hscp",hscpH);
  const vector<HSCParticle> & candidates = *hscpH.product();

  // Passing the trigger (exception is thrown if above collection are not filled)
  selectedAfterCut[1]+=w;
  
  // Some counters and flags
  bool selectCandidate     = false;
  bool selectCandidateDedx = false;
  bool selectCandidateTof  = false;
  float ptmax_tkMuons   = 0.;
  float ptmax_staMuons  = 0.;
  float ptmax_CombMuons = 0.;
  bool found1=false; // 1/beta2(tk) > 1.56 using TOF
  bool found2=false; // 1/beta(tof) > 1.25 using TOF
  bool found3=false; // TOF and dedx beta measurements agrees
  bool found4=false; // low error on the mass measurement
  bool found5=false; // 1/beta(tof) > 1.176 using TOF

  // Loop over candidates
  for(vector<HSCParticle>::const_iterator hscpCand = candidates.begin(); hscpCand!=candidates.end(); ++hscpCand) {
    // First, look at muon-related quantities
    if(hscpCand->hasDtInfo()) {
      if(hscpCand->hasMuonTrack()) {
        ptmax_tkMuons = ptmax_tkMuons>hscpCand->muonTrack().pt() ? ptmax_tkMuons : hscpCand->muonTrack().pt();
        h_pt->Fill(hscpCand->muonTrack().pt(),w);
      }
      if(hscpCand->hasMuonStaTrack()) {
        ptmax_staMuons = ptmax_staMuons>hscpCand->staTrack().pt() ? ptmax_staMuons : hscpCand->staTrack().pt();
        h_tofPtSta->Fill(hscpCand->staTrack().pt(),w);
      }
      if(hscpCand->hasMuonCombinedTrack()) {
        ptmax_CombMuons = ptmax_CombMuons>hscpCand->combinedTrack().pt() ? ptmax_CombMuons : hscpCand->combinedTrack().pt();
        h_tofPtComb->Fill(hscpCand->combinedTrack().pt(),w);
      }
      double p = hscpCand->p();
      //double pt = hscpCand->pt();
      //if(pt>100) highptmu=true;
      double invbeta = hscpCand->Dt().second.invBeta;
      double invbetaerr = hscpCand->Dt().second.invBetaErr;
      double mass = hscpCand->massDtBest();
      double mass2 = p*p*(invbeta*invbeta-1);
      // cout << " Muon p: " << p << " invBeta: " << invbeta << " Mass: " << mass << endl;
      if( hscpCand->Dt().second.invBetaErr < 0.07) h_tofBetaCut->Fill(1./invbeta , w);
      h_tofBeta->Fill(invbeta,w);
      h_tofBetaErr->Fill(invbetaerr,w);
      h_tofBetaPull->Fill((invbeta-1.)/invbetaerr,w);
      h_tofBetaPullp->Fill(p,(invbeta-1.)/invbetaerr,w);
      h_tofBetap->Fill(p,invbeta,w);
      h_tofMassp->Fill(p,mass,w);
      h_tofMass->Fill(mass,w);
      h_tofMass2->Fill(mass2,w);
      if(mass>100.) selectCandidateTof = true;
    }
    // Then, look at dedx (and tracker)
    if(hscpCand->hasTkInfo()) {
      double p       = hscpCand->trackerTrack().p();
      double dedxVal = hscpCand->Tk().dedx();
      double mass    = hscpCand->massTk();
      h_dedx->Fill(p, dedxVal,w);   
      h_dedxCtrl->Fill(p, dedxVal,w);   
      h_dedxMass->Fill(mass,w); 
      if(mass>100.) selectCandidateDedx = true;
      // select the MIP region
      if(p > 5 && p < 30 ) {
        h_dedxMIP->Fill( dedxVal,w);   
	/*
        if(dedxVal >3.22) {
          std::cout << track->normalizedChi2() << " " << track->numberOfValidHits() << " " << p <<std::endl;
        }
	*/
        h_dedxMIPbeta->Fill(hscpCand->Tk().beta(),w);
        if(hscpCand->Tk().nDedxHits() >= 12) {
          h_dedxMIPbetaCut->Fill(hscpCand->Tk().beta(),w);
        }
      }
      // select the region of interest
      if(p > 30 && dedxVal > 3.45  ) {  
        h_dedxMassSel->Fill(mass,w); 
      }
      // select the proton region
      if(p < 1.2 && mass > 0.200 ) {
        h_dedxMassProton->Fill(mass,w);
      }
      //FIXME: make it configurable
      double dedxCut[6] = {3.0,3.16,3.24,3.64,4.68,6.2};
      for(int ii=0;ii<6;ii++) {
        if(dedxVal > dedxCut[ii]) {
          dedxSelectionLevel=ii; 
          h_pSpectrumAfterSelection[ii]->Fill(p,w);
          h_massAfterSelection[ii]->Fill(mass,w);      
        }
      }
    }
    // Finally consider the combined information
    if(hscpCand->hasTkInfo() && hscpCand->massTk()>100. &&
       hscpCand->hasDtInfo() && hscpCand->massDtBest()>100.     ) selectCandidate = true;
    if(hscpCand->hasTkInfo() && hscpCand->hasDtInfo() && hscpCand->hasMuonTrack()) {
      h_betaVsBeta->Fill(hscpCand->Dt().second.invBeta,sqrt(hscpCand->Tk().invBeta2()),w);
      selectedAfterCut[19]+=w;
      if(hscpCand->Dt().second.invBeta > 1.176) { 
        found5=true; selectedAfterCut[10]+=w; 
      }
      if(hscpCand->Dt().second.invBeta > 1.25) { 
        found2=true; selectedAfterCut[11]+=w; 
      }
      if(hscpCand->Tk().invBeta2() > 1.56) { 
        found1=true; selectedAfterCut[12]+=w; 
      }
      if(fabs(sqrt(1./hscpCand->Tk().invBeta2()) - 1./hscpCand->Dt().second.invBeta )  < 0.1) { 
        found3=true; 
      }
      if(hscpCand->Dt().second.invBeta > 1.25 && hscpCand->Tk().invBeta2() > 1.56 ) { 
        selectedAfterCut[13]+=w; 
      }
      // compute the average mass and error;
      double avgMass = (hscpCand->massDt()+hscpCand->massTk())/2.;
      h_massVsBeta->Fill(avgMass,2./(sqrt(hscpCand->Tk().invBeta2())+hscpCand->Dt().second.invBeta),w);
      if(2./(sqrt(hscpCand->Tk().invBeta2())+hscpCand->Dt().second.invBeta) < 0.85)
        h_massVsPtError->Fill(avgMass,log10(hscpCand->Tk().track()->ptError()),w);
      double ptMassError=(hscpCand->Tk().track()->ptError()/hscpCand->Tk().track()->pt());
      ptMassError*=ptMassError;  
      double ptMassError2=(hscpCand->Dt().first->track()->ptError()/hscpCand->Dt().first->track()->pt());
      ptMassError2*=ptMassError2;  
      double ib2 = hscpCand->Dt().second.invBeta*hscpCand->Dt().second.invBeta;
      double dtMassError=hscpCand->Dt().second.invBetaErr*(ib2/sqrt(ib2-1)) ;
      dtMassError*= dtMassError;
      double dedxError = 0.2*sqrt(10./hscpCand->Tk().nDedxHits())*0.4/hscpCand->Tk().invBeta2();
      double tkMassError = dedxError/(2.*hscpCand->Tk().invBeta2()-1); 
      tkMassError*=tkMassError;
      double avgMassError=sqrt(ptMassError/4+ptMassError2/4.+dtMassError/4.+tkMassError/4.);
      bool nosel    = hscpCand->hasTkInfo() && hscpCand->hasDtInfo() && hscpCand->hasMuonTrack();
      bool dte07    = hscpCand->Dt().second.invBetaErr < 0.07 ;
      bool dte10    = hscpCand->Dt().second.invBetaErr < 0.10 ;
      bool dt080    = hscpCand->Dt().second.invBeta > 1.25 &&  hscpCand->Dt().second.invBeta < 1000. && dte10;
      bool dt085    = hscpCand->Dt().second.invBeta > 1.176 &&  hscpCand->Dt().second.invBeta < 1000. && dte10;
      bool tk080    = hscpCand->Tk().invBeta2() > 1.56 ;
      bool tkpt100  = hscpCand->Tk().track()->pt() > 100 && 
                      hscpCand->Dt().first->combinedMuon().isNonnull() && hscpCand->Dt().first->combinedMuon()->pt() > 100;
      bool tkm100   = hscpCand->massTk() > 100 ;
      bool tkm200   = hscpCand->massTk() > 200 ;
      bool tkm400   = hscpCand->massTk() > 400 ;
      bool tkhits14 = hscpCand->Tk().nDedxHits() >= 14; 
      bool db01     =  fabs(sqrt(1./hscpCand->Tk().invBeta2())- 1./hscpCand->Dt().second.invBeta )  < 0.1 ;
      if(nosel) cuts[0]->passed(*hscpCand,w);
      if(nosel && dt085 ) cuts[1]->passed(*hscpCand,w);
      if(nosel && dt080 ) cuts[2]->passed(*hscpCand,w);
      if(nosel && tk080 ) cuts[3]->passed(*hscpCand,w);
      if(nosel && dt080 && tk080) cuts[4]->passed(*hscpCand,w);
      if(nosel && db01 )  cuts[5]->passed(*hscpCand,w);
      if(nosel && dt080 && tk080 && db01) cuts[6]->passed(*hscpCand,w);
      if(nosel && dt080 && tk080 && dte07) cuts[7]->passed(*hscpCand,w);
      if(nosel && dt085 && tk080 ) cuts[8]->passed(*hscpCand,w);
      if(nosel && tk080 && tkpt100 ) cuts[9]->passed(*hscpCand,w);
      if(nosel && tk080 && tkpt100 && tkhits14 ) cuts[10]->passed(*hscpCand,w);
      if(nosel && tk080 && tkpt100 && tkm100 ) cuts[11]->passed(*hscpCand,w);
      if(nosel && tk080 && tkpt100 && tkm200 ) cuts[12]->passed(*hscpCand,w);
      if(nosel && tk080 && tkpt100 && tkm400 ) cuts[13]->passed(*hscpCand,w);
      if(nosel && dte10 ) cuts[14]->passed(*hscpCand,w);
      if(nosel && dte07 ) cuts[15]->passed(*hscpCand,w);
      if(avgMassError < 0.05 + 0.1* (hscpCand->massDt()+hscpCand->massTk())/1000.) { 
        found4=true;
      }
      if( hscpCand->Dt().second.invBeta > 1.25 && hscpCand->Tk().invBeta2() > 1.56 && avgMassError < 0.15 ) {
        selectedAfterCut[14]+=w;
        if((hscpCand->massDt()+hscpCand->massTk())/2. > 100)     selectedAfterCut[15]+=w;
        if((hscpCand->massDt()+hscpCand->massTk())/2. > 200)     selectedAfterCut[16]+=w;
        if((hscpCand->massDt()+hscpCand->massTk())/2. > 300)     selectedAfterCut[17]+=w;
        if((hscpCand->massDt()+hscpCand->massTk())/2. > 600)     selectedAfterCut[18]+=w;
      }
      h_massVsMassError->Fill((hscpCand->massDt()+hscpCand->massTk())/2.,avgMassError,w);
      if((hscpCand->Dt().second.invBeta>1.1) || (hscpCand->Tk().invBeta2()>1.3 && hscpCand->hasDtInfo()) ) {
        h_massVsMass->Fill(hscpCand->massDt(),hscpCand->massTk(),w);
        if(hscpCand->hasTkInfo() && 
           hscpCand->hasDtInfo() && 
           hscpCand->hasMuonTrack() && 
           hscpCand->massDt() + hscpCand->massTk() >  280 && 
           fabs(sqrt(1./hscpCand->Tk().invBeta2())- 1./hscpCand->Dt().second.invBeta ) < 0.1 && 
           hscpCand->massDt() > 100 &&  
           hscpCand->massTk() > 100 && 
           hscpCand->Tk().invBeta2() > 1.4 &&  
           hscpCand->Dt().second.invBeta  > 1.11 ) {
          h_massVsMassSel->Fill(hscpCand->massDt(),hscpCand->massTk(),w);
        }
        //cout << "CANDIDATE " <<  hscpCand->massDt() << " " << hscpCand->massTk() 
        //     << " " << hscpCand->Tk().track()->momentum() << " " <<  hscpCand->Dt().first->combinedMuon()->momentum() 
        //     << " " << hscpCand->Dt().first->track()->pt() 
        //     <<" dt beta: " << 1./hscpCand->Dt().second.invBeta << " tk beta : "<< sqrt(1./hscpCand->Tk().invBeta2())
        //     <<" chi &  # hits: " <<  hscpCand->Tk().track()->normalizedChi2() << " " << hscpCand->Tk().track()->numberOfValidHits() 
        //     << "errors: " << avgMassError << " sqrt( " <<ptMassError << "/4 + "
        //     << ptMassError2 << "/4 + "<< dtMassError << "/4 + " << tkMassError << "/4) " << dedxError <<  endl;
      }
    }
    if(hscpCand->hasTkInfo() && 
       hscpCand->hasDtInfo() && 
       hscpCand->massTk() > 100 && 
       hscpCand->Dt().first->standAloneMuon()->pt() > 100 && 
       hscpCand->Tk().invBeta2() > 1.56                         ) {
      h_dedxMassMu->Fill(hscpCand->massTk(),w); 
      h_dedxMassMuVsPtError->Fill(hscpCand->massTk(),hscpCand->Tk().track()->ptError()/hscpCand->Tk().track()->pt(),w);
      double ptMassError=(hscpCand->Tk().track()->ptError()/hscpCand->Tk().track()->pt());
      ptMassError*=ptMassError;
      double dedxError = 0.2*sqrt(10./hscpCand->Tk().nDedxHits())*0.4/hscpCand->Tk().invBeta2();
      double tkBetaMassError = dedxError/(2.*hscpCand->Tk().invBeta2()-1);
      tkBetaMassError*=tkBetaMassError;
      double tkMassError=sqrt(ptMassError+tkBetaMassError);
      h_massVsMassError_tk->Fill(hscpCand->massTk(),tkMassError,w);
      h_massVsBeta_tk->Fill(hscpCand->massTk(),1./sqrt(hscpCand->Tk().invBeta2()),w);
      h_massVsPtError_tk->Fill(hscpCand->massTk(),log10(hscpCand->Tk().track()->ptError()),w);
    }
  } // Loop over candidates

  // store the highest Pt
  if(ptmax_tkMuons>0.1)   h_tkmu_pt->Fill(ptmax_tkMuons,w);
  if(ptmax_staMuons>0.1)  h_stamu_pt->Fill(ptmax_staMuons,w);
  if(ptmax_CombMuons>0.1) h_combmu_pt->Fill(ptmax_CombMuons,w);

  // count candidates
  if(selectCandidate)     selected+=w;
  if(selectCandidateDedx) selectedDedx+=w;
  if(selectCandidateTof)  selectedTOF+=w;
  if(found5) selectedAfterCut[2]+=w;
  if(found2) selectedAfterCut[3]+=w;
  if(found1) selectedAfterCut[4]+=w;
  if(found3) selectedAfterCut[5]+=w;
  if(found1 && found2) selectedAfterCut[6]+=w;
  if(found1 && found2 && found3) selectedAfterCut[7]+=w;
  if(found1 && found2 && found4) selectedAfterCut[8]+=w;

  // Look at simtracks
  if(m_haveSimTracks) {
    Handle<edm::SimTrackContainer> simTracksHandle;
    iEvent.getByLabel("g4SimHits",simTracksHandle);
    const SimTrackContainer simTracks = *(simTracksHandle.product());

    //Loop over simtracks and fill Pt, eta distributions
    SimTrackContainer::const_iterator simTrack;
    for(simTrack = simTracks.begin(); simTrack != simTracks.end(); ++simTrack) {
      // select HSCP
      if(abs((*simTrack).type()) > 1000000) {
        h_simhscp_pt->Fill((*simTrack).momentum().pt(),w);
        h_simhscp_eta->Fill(((*simTrack).momentum().eta()),w);
      }
      // select muons
      if(abs((*simTrack).type()) == 13) {
        h_simmu_pt->Fill((*simTrack).momentum().pt(),w);
        h_simmu_eta->Fill(((*simTrack).momentum().eta()),w);
      }
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void HSCPAnalyzer::beginJob(const edm::EventSetup&)
{
  for(int i=0;i<20;i++) selectedAfterCut[i] = 0;
  for(int i=0;i<40;i++) cuts[i] = 0;
  float minBeta = 0.8l;
  float maxBeta = 3.;

  edm::Service<TFileService> fs;

  cuts[0]  = new CutMonitor("NoSel",fs);
  cuts[1]  = new CutMonitor("DT-085",fs);
  cuts[2]  = new CutMonitor("DT-080",fs);
  cuts[3]  = new CutMonitor("TK-080",fs);
  cuts[4]  = new CutMonitor("TK-080_DT-080",fs);
  cuts[5]  = new CutMonitor("DB-01",fs);
  cuts[6]  = new CutMonitor("TK-080_DT-080_DB-01",fs);
  cuts[7]  = new CutMonitor("TK-080_DT-080_DTE-07",fs);
  cuts[8]  = new CutMonitor("TK-080_DT-085",fs);
  cuts[9]  = new CutMonitor("TK-080_TKPT-100",fs);
  cuts[10] = new CutMonitor("TK-080_TKPT-100_DEDH14",fs);
  cuts[11] = new CutMonitor("TK-080_TKPT-100_M-100",fs);
  cuts[12] = new CutMonitor("TK-080_TKPT-100_M-200",fs);
  cuts[13] = new CutMonitor("TK-080_TKPT-100_M-400",fs);
  cuts[14] = new CutMonitor("DTE-10",fs);
  cuts[15] = new CutMonitor("DTE-07",fs);

  //------------ RECO DEDX ----------------
  TFileDirectory subDir = fs->mkdir( "RecoDeDx" );
  h_pt                  = subDir.make<TH1F>( "mu_pt"  , "p_{t}", 100,  0., 1500. );
  h_dedx                = subDir.make<TH2F>( "dedx_p"  , "\\frac{dE}{dX} vs p", 100,  0., 1500., 100,0,8 );
  h_dedxCtrl            = subDir.make<TH2F>( "dedx_lowp"  , "\\frac{dE}{dX} vs p", 100,  0., 3., 100,0,8 );
  h_dedxMIP             = subDir.make<TH1F>( "dedxMIP"  , "\\frac{dE}{dX}  ",100,0,8 );
  h_dedxMIPbeta         = subDir.make<TH1F>( "dedxMIP_beta"  , "\\frac{dE}{dX}  ",100,0,1);
  h_dedxMIPbetaCut      = subDir.make<TH1F>( "dedxMIP_beta_cut"  , "\\frac{dE}{dX}  ",100,0,1);
  h_dedxMassSel         = subDir.make<TH1F>( "massSel"  , "Mass (dedx), with selection", 100,  0., 1500.);
  h_dedxMass            = subDir.make<TH1F>( "mass"  , "Mass (dedx)", 100,  0., 1500.);
  h_dedxMassProton      = subDir.make<TH1F>( "massProton"  , "Proton Mass (dedx)", 100,  0., 2.);
  h_dedxMassMu          = subDir.make<TH1F>( "massMu"  , "Mass muons (dedx, 1 mu with pt>100 in the event)", 100,  0., 1500.);
  h_dedxMassMuVsPtError = subDir.make<TH2F>( "massMu_vs_PtError"  , "Mass muons vs pt error (dedx, 1 mu with pt>100 in the event)", 100,  0., 1500.,50,0,1);
  
  //------------ RECO TOF ----------------
  TFileDirectory subDirTof = fs->mkdir( "RecoTOF" );
  h_tofBetap        = subDirTof.make<TH2F>("tof_beta_p","1/#beta vs p",100,0,1000,100,minBeta,maxBeta );
  h_tofBetaPullp    = subDirTof.make<TH2F>("tof_beta_pull_p","1/#beta pull vs p",100,0,1000,100,-5.,5 );
  h_tofMassp        = subDirTof.make<TH2F>("tof_mass_p","Mass vs p", 100,0,1000,100,0,1000);
  h_tofMass         = subDirTof.make<TH1F>("tof_mass","Mass from DT TOF",100,0,1000);
  h_tofMass2        = subDirTof.make<TH1F>("tof_mass2","Mass squared from DT TOF",100,-10000,100000);
  h_tofBeta         = subDirTof.make<TH1F>("tof_beta","1/#beta",100,minBeta,maxBeta);
  h_tofBetaErr      = subDirTof.make<TH1F>("tof_beta_err","#Delta 1/#beta",100,0,.5);
  h_tofBetaPull     = subDirTof.make<TH1F>("tof_beta_pull","(1/#beta-1)/(#Delta 1/#beta)",100,-5.,5.);
  h_tofPtSta        = subDirTof.make<TH1F>("tof_pt_sta","StandAlone reconstructed muon p_{t}",100,0,300);
  h_tofPtComb       = subDirTof.make<TH1F>("tof_pt_comb","Global reconstructed muon p_{t}",100,50,150);
  h_tofMassCut      = subDirTof.make<TH1F>("tof_mass_cut","Mass from DT TOF (cut)",100,0,1000);
  h_tofBetaCut      = subDirTof.make<TH1F>("tof_beta_cut","1/#beta (cut)",100,minBeta,maxBeta);
  h_tofBetaPullCut  = subDirTof.make<TH1F>("tof_beta_pull_cut","(1/#beta-1)/(#Delta 1/#beta)",100,-5.,5.);
  h_tofBetapCut     = subDirTof.make<TH2F>("tof_beta_p_cut","1/#beta vs p (cut)",100,0,1000,100,minBeta,maxBeta );
  h_tofBetaPullpCut = subDirTof.make<TH2F>("tof_beta_pull_p_cut","1/#beta pull vs p (cut)",100,0,1000,100,-5.,5 );

  //-------- Analysis ----------------
  TFileDirectory subDirAn =  fs->mkdir( "Analysis" );
  for(int i=0;i<6;i++) {
    h_pSpectrumAfterSelection[i] = subDirAn.make<TH1F>(Form("pSpectrumDedxSel%d",i),Form("P spectrum after selection #%d",i),300,0,1000);
    h_massAfterSelection[i]      = subDirAn.make<TH1F>(Form("massDedxSel%d",i),Form("Mass after selection #%d",i),300,0,1000);
  } 
  h_massVsMass      = subDirAn.make<TH2F>("tof_mass_vs_dedx_mass","Mass tof vs Mass dedx", 100,0,1200,100,0,1200);
  h_massVsMassSel   = subDirAn.make<TH2F>("tof_mass_vs_dedx_mass_sel","Mass tof vs Mass dedx Sel", 100,0,1200,100,0,1200);
  h_betaVsBeta      = subDirAn.make<TH2F>("tof_beta_vs_beta","INVBeta tof vs INVbeta dedx (Pt>30)", 100,0,3,100,0,3);
  h_massVsBeta      = subDirAn.make<TH2F>("avgMass_vs_avgBeta","Mass(avg) vs Beta(avg)", 100,0,1200,50,0,1);
  h_massVsPtError   = subDirAn.make<TH2F>("avgMass_vs_ptError","Mass(avg) vs log(ptError) (beta < 0.85)", 100,0,1200,25,0,10);
  h_massVsMassError = subDirAn.make<TH2F>("avgMass_vs_MassError","Mass(avg) vs log(masstError)", 100,0,1200,100,0,2);
  h_stamu_pt        = subDirAn.make<TH1F>("STA  Muon Pt Distribution", "StandAlone Muon Pt Distribution", 1000, 0, 1000);
  h_combmu_pt       = subDirAn.make<TH1F>("Comb Muon Pt Distribution", "Combined Muon Pt Distribution", 1000, 0, 1000);
  h_tkmu_pt         = subDirAn.make<TH1F>("TK   Muon Pt Distribution", "Track Muon Pt Distribution", 1000, 0, 1000);

  //--------- Analysis tk ------
  TFileDirectory subDirAnTk =  fs->mkdir( "AnalysisTk" );
  h_massVsBeta_tk      = subDirAnTk.make<TH2F>("tkMass_vs_tkBeta","Mass(tk) vs Beta(tk)", 100,0,1200,50,0,1);
  h_massVsPtError_tk   = subDirAnTk.make<TH2F>("tkMass_vs_ptError","Mass(tk) vs log(ptError) (beta < 0.85)", 100,0,1200,25,0,10);
  h_massVsMassError_tk = subDirAnTk.make<TH2F>("tkMass_vs_MassError","Mass(tk) vs mass error", 100,0,1200,100,0,2);

  //------------ SIM ----------------
  TFileDirectory subDir2 = fs->mkdir( "Sim" );
  h_simmu_pt    = subDir2.make<TH1F>( "mu_sim_pt"  , "p_{t} mu", 100,  0., 1500. );
  h_simmu_eta   = subDir2.make<TH1F>( "mu_sim_eta"  , "\\eta mu", 50,  -4., 4. );
  h_simhscp_pt  = subDir2.make<TH1F>( "mu_hscp_pt"  , "p_{t} hscp", 100,  0., 1500. );
  h_simhscp_eta = subDir2.make<TH1F>( "mu_hscp_eta"  , "\\eta hscp" , 50,  -4., 4. );
}

// ------------ method called once each job just after ending the event loop  ------------
void HSCPAnalyzer::endJob() {
  // Final fit and summary
  h_dedxMIP->Fit("gaus","Q");
  h_dedxMassProton->Fit("gaus","Q","",0.8,1.2); 
  float mipMean  = h_dedxMIP->GetFunction("gaus")->GetParameter(1);
  float mipSigma = h_dedxMIP->GetFunction("gaus")->GetParameter(2);
  double effPoints[6] = {0.05,0.01,0.005,0.001,0.0001,0.00001};
  float dedxEff[6];
  float dedxEffP[6];
  float dedxEffM[6];
  for(int i=0;i<6;i++) {
    dedxEff[i]=cutMin(h_dedxMIP,effPoints[i]);
    float n=h_dedxMIP->GetEntries();
    float delta=sqrt(effPoints[i]*n)/n;
    dedxEffM[i]=cutMin(h_dedxMIP,effPoints[i]+delta)-dedxEff[i];
    dedxEffP[i]=cutMin(h_dedxMIP,effPoints[i]-delta)-dedxEff[i];
  }
  cout << "DeDx cuts: " << endl;
  cout << "   dedxSrc de/dx ("<< m_dedxSrc << "):" << endl; 
  cout << "    Proton Mass : " << h_dedxMassProton->GetFunction("gaus")->GetParameter(1) << endl;
  cout << "    MIP  mean : " <<  mipMean << "    sigma : " << mipSigma << endl;
  for(int i=0;i<6;i++)   
    cout << "    " << (1-effPoints[i])*100 << "% @ dedx > " << dedxEff[i] << "+"<< dedxEffP[i] << " -" << dedxEffM[i] << endl;
  cout << endl;
  cout << "Processed events: "     << tot <<  endl;
  cout << "Selected events: "      << selected <<  endl;
  cout << "Selected tof  events: " << selectedTOF <<  endl;
  cout << "Selected dedx events: " << selectedDedx <<  endl;
  cout << "Selection: ";
  for(int i=0; i< 20 ; i++)  
    cout << fixed << setw(7) << setprecision(1) <<  selectedAfterCut[i] << " & " ;
  cout << endl;
  cout << "CutTitle: ";
  for(int i=0;i<40;i++) if(cuts[i]) { 
    cout << " & " << fixed << setw(6) << setprecision(1) ; 
    cuts[i]->printName(); 
  }
  cout << endl << "Cuts: ";
  for(int i=0;i<40;i++) if(cuts[i]) { 
    cout << " & " << fixed << setw(6) << setprecision(1) ; 
    cuts[i]->print(); 
  }
  cout << endl;
  cout << endl << "CutEff: ";
  for(int i=0;i<40;i++) if(cuts[i]) { 
    cout << " & " << fixed << setw(6) << setprecision(1) ; 
    cuts[i]->printEff(); 
  }
  cout << endl;
}

double HSCPAnalyzer::cutMin(TH1F * h, double ci)
{
  //computes the quantile ci of the distribution h
  double sum=0;
  if(h->GetEntries()>0)
  for(int i=h->GetNbinsX();i>=0; i--) {
    sum+=h->GetBinContent(i); 
    if(sum/h->GetEntries()>ci) {
       return h->GetBinCenter(i); 
    }
  } 
  return h->GetBinCenter(0);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPAnalyzer);
