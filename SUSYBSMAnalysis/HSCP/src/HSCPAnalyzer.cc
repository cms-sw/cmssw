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
// $Id: HSCPAnalyzer.cc,v 1.29 2009/05/15 17:52:21 delaer Exp $
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

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

#include "SUSYBSMAnalysis/HSCP/interface/HSCPCandidateFilter.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "Math/GenVector/VectorUtil.h"

#include <TH1F.h>
#include <TH2F.h>
#include <TF1.h>
#include <TTree.h>

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
  // Original plots
  TH2F * h_massVsMass;
  TH2F * h_betaVsBeta;
  TH2F * h_massVsBeta;
  TH2F * h_massVsPtError;
  TH2F * h_massVsMassError;
  TH2F * h_massVsPt;
  TH2F * h_massVsBeta_tk;
  TH2F * h_massVsPtError_tk;
  TH2F * h_massVsMassError_tk;
  TH1F * h_massTk;
  TH1F * h_massDt;
  TH1F * h_massAvg;
  TH1F * h_tofInvBetaErr;
  TH1F * h_dedxHits;
  TH1F * h_dedxBeta;
  TH2F * h_dedxHitsBeta;
  TH1F * h_deltaBeta;
  TH1F * h_deltaBetaInv;
  TH1F * h_tofInvBeta;
  // RECO DEDX
  TH2F * h_dedx;
  TH2F * h_dedxCtrl;
  TH1F * h_dedxMass;
  TH1F * h_dedxMIP;
  TH1F * h_dedxMIPbeta;
  TH1F * h_dedxMIPbetaCut;
  TH1F * h_dedxMassSel;
  TH1F * h_dedxMassProton;
  TH1F * h_pSpectrumAfterSelection[6]; 
  TH1F * h_massAfterSelection[6];
  // RECO TOF
  TH1F * h_pt;
  TH1F * h_tofPtSta;
  TH1F * h_tofPtComb;
  TH1F * h_tofBetaCut;
  TH1F * h_tofBeta;
  TH1F * h_tofBetaErr;
  TH1F * h_tofBetaPull;
  TH2F * h_tofBetaPullp;
  TH2F * h_tofBetap;
  TH2F * h_tofMassp;
  TH1F * h_tofMass;
  TH1F * h_tofMass2;
};

class CutMonitor {
 public:
  CutMonitor(std::string name,edm::Service<TFileService> fs):
    m_plots(fs->mkdir(name)), m_name(name), m_newevent(true),m_evCounter(0),m_candCounter(0),m_tot(0) {}
 
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
    cout << fixed << setprecision(1) << setw(6) <<  m_candCounter 
         << "(" << fixed << setprecision(1) << setw(6) << m_evCounter << ")";
  }
  
  void printEff() {
    cout << fixed << setprecision(3) << setw(5) <<  m_candCounter/m_tot 
         << "(" << fixed << setprecision(3) << setw(5) << m_evCounter/m_tot << ")";
  }
 
  void printName() {
    cout << m_name;
  }
  
  double totalEvents() const { return m_tot; }
  
  double eventCount() const { return m_evCounter; }

  double candidateCount() const { return m_candCounter; }

 public:
  HSCPStandardPlots m_plots;
 private:
  std::string m_name;
  bool m_newevent;
  double m_evCounter;
  double m_candCounter;
  double m_tot;
}; 

HSCPStandardPlots::HSCPStandardPlots(TFileDirectory  subDir) {
  //------------ Analysis ----------------
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
  h_tofInvBeta =          subDir.make<TH1F>("tof_Invbeta"  , " tof beta  ",500,0,5);
  h_tofInvBetaErr =       subDir.make<TH1F>("tof_inv_beta_err"  , " tof beta err  ",100,0,1);
  //------------ RECO TOF ----------------
  h_pt                  = subDir.make<TH1F>("mu_pt"  , "p_{t}", 100,  0., 1500. );
  h_tofPtSta            = subDir.make<TH1F>("tof_pt_sta","StandAlone reconstructed muon p_{t}",100,0,300);
  h_tofPtComb           = subDir.make<TH1F>("tof_pt_comb","Global reconstructed muon p_{t}",100,50,150);
  h_tofBetaCut          = subDir.make<TH1F>("tof_beta_cut","1/#beta (cut)",100,0.81,3.);
  h_tofBeta             = subDir.make<TH1F>("tof_beta","1/#beta",100,0.81,3.);
  h_tofBetaErr          = subDir.make<TH1F>("tof_beta_err","#Delta 1/#beta",100,0,.5);
  h_tofBetaPull         = subDir.make<TH1F>("tof_beta_pull","(1/#beta-1)/(#Delta 1/#beta)",100,-5.,5.);
  h_tofBetaPullp        = subDir.make<TH2F>("tof_beta_pull_p","1/#beta pull vs p",100,0,1000,100,-5.,5 );
  h_tofBetap            = subDir.make<TH2F>("tof_beta_p","1/#beta vs p",100,0,1000,100,0.81,3. );
  h_tofMassp            = subDir.make<TH2F>("tof_mass_p","Mass vs p", 100,0,1000,100,0,1000);
  h_tofMass             = subDir.make<TH1F>("tof_mass","Mass from DT TOF",100,0,1000);
  h_tofMass2            = subDir.make<TH1F>("tof_mass2","Mass squared from DT TOF",100,-10000,100000);
  //------------ RECO DEDX ----------------
  h_dedx                = subDir.make<TH2F>("dedx_p"  , "\\frac{dE}{dX} vs p", 100,  0., 1500., 100,0,8 );
  h_dedxCtrl            = subDir.make<TH2F>("dedx_lowp"  , "\\frac{dE}{dX} vs p", 100,  0., 3., 100,0,8 );
  h_dedxMass            = subDir.make<TH1F>("mass"  , "Mass (dedx)", 100,  0., 1500.);
  h_dedxMIP             = subDir.make<TH1F>("dedxMIP"  , "\\frac{dE}{dX}  ",100,0,8 );
  h_dedxMIPbeta         = subDir.make<TH1F>("dedxMIP_beta"  , "\\frac{dE}{dX}  ",100,0,1);
  h_dedxMIPbetaCut      = subDir.make<TH1F>("dedxMIP_beta_cut"  , "\\frac{dE}{dX}  ",100,0,1);
  h_dedxMassSel         = subDir.make<TH1F>("massSel"  , "Mass (dedx), with selection", 100,  0., 1500.);
  h_dedxMassProton      = subDir.make<TH1F>("massProton"  , "Proton Mass (dedx)", 100,  0., 2.);
  for(int i=0;i<6;i++) {
    h_pSpectrumAfterSelection[i] = subDir.make<TH1F>(Form("pSpectrumDedxSel%d",i),Form("P spectrum after selection #%d",i),300,0,1000);
    h_massAfterSelection[i]      = subDir.make<TH1F>(Form("massDedxSel%d",i),Form("Mass after selection #%d",i),300,0,1000);
  } 
}

void HSCPStandardPlots::fill(const HSCParticle & hscp, double w) {

  // DT information
  if(hscp.hasDtInfo()) {
    if(hscp.hasMuonTrack()) h_pt->Fill(hscp.muonTrack().pt(),w);
    if(hscp.hasMuonStaTrack()) h_tofPtSta->Fill(hscp.staTrack().pt(),w);
    if(hscp.hasMuonCombinedTrack()) h_tofPtComb->Fill(hscp.combinedTrack().pt(),w);
    double p = hscp.p();
    double invbeta = hscp.Dt().second.invBeta;
    double invbetaerr = hscp.Dt().second.invBetaErr;
    double mass = hscp.massDtBest();
    double mass2 = p*p*(invbeta*invbeta-1);
    if( hscp.Dt().second.invBetaErr < 0.07) h_tofBetaCut->Fill(1./invbeta , w);
    h_tofBeta->Fill(invbeta,w);
    h_tofBetaErr->Fill(invbetaerr,w);
    h_tofBetap->Fill(p,invbeta,w);
    if(hscp.Dt().second.invBeta !=0 && hscp.Dt().second.invBetaErr !=0 && hscp.Dt().second.invBetaErr < 1000.) {
      h_tofBetaPull->Fill((invbeta-1.)/invbetaerr,w);
      h_tofBetaPullp->Fill(p,(invbeta-1.)/invbetaerr,w);
    }
    h_tofMassp->Fill(p,mass,w);
    h_tofMass->Fill(mass,w);
    h_tofMass2->Fill(mass2,w);
  }
  // Tracker information
  if(hscp.hasTkInfo()) {
    double p       = hscp.trackerTrack().p();
    double dedxVal = hscp.Tk().dedx();
    double mass    = hscp.massTk();
    h_dedx->Fill(p, dedxVal,w);   
    h_dedxCtrl->Fill(p, dedxVal,w);   
    h_dedxMass->Fill(mass,w); 
    // select the MIP region
    if(p > 5 && p < 30 ) {
      h_dedxMIP->Fill( dedxVal,w);   
      h_dedxMIPbeta->Fill(hscp.Tk().beta(),w);
      if(hscp.Tk().nDedxHits() >= 12) {
        h_dedxMIPbetaCut->Fill(hscp.Tk().beta(),w);
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
    // slices in dedx
    double dedxCut[6] = {3.0,3.16,3.24,3.64,4.68,6.2};
    for(int ii=0;ii<6;ii++) {
      if(dedxVal > dedxCut[ii]) {
        h_pSpectrumAfterSelection[ii]->Fill(p,w);
        h_massAfterSelection[ii]->Fill(mass,w);      
      }
    }
  }
  // combined information
  if(hscp.hasTkInfo() && hscp.hasDtInfo() && hscp.hasMuonTrack()) {
    h_massTk->Fill(hscp.massTk(),w);
    h_massDt->Fill(hscp.massDt(),w);
    h_massAvg->Fill(hscp.massAvg(),w);
    h_dedxHits->Fill(hscp.Tk().nDedxHits(),w);
    h_dedxBeta->Fill(1./sqrt(hscp.Tk().invBeta2()),w); 
    h_dedxHitsBeta->Fill(hscp.Tk().nDedxHits(),1./sqrt(hscp.Tk().invBeta2()),w);
    h_deltaBeta->Fill(1./hscp.Dt().second.invBeta-1./sqrt(hscp.Tk().invBeta2()),w);
    h_deltaBetaInv->Fill(hscp.Dt().second.invBeta-sqrt(hscp.Tk().invBeta2()),w);
    h_massVsMass->Fill(hscp.massDt(),hscp.massTk(),w);
    h_massVsBeta->Fill(hscp.massAvg(),hscp.betaAvg(),w);
    h_massVsPtError->Fill(hscp.massAvg(),log10(hscp.Tk().track()->ptError()),w);
    h_massVsPt->Fill(hscp.massAvg(),hscp.Tk().track()->pt(),w);
    h_massVsMassError->Fill(hscp.massAvg(),hscp.massAvgError(),w);
    h_massVsMassError_tk->Fill(hscp.massTk(),hscp.massTkError(),w);
    h_massVsBeta_tk->Fill(hscp.massTk(),1./sqrt(hscp.Tk().invBeta2()),w);
    h_massVsPtError_tk->Fill(hscp.massTk(),log10(hscp.Tk().track()->ptError()),w);
    h_betaVsBeta->Fill(hscp.Dt().second.invBeta,sqrt(hscp.Tk().invBeta2()),w);
    h_tofInvBetaErr->Fill(hscp.Dt().second.invBetaErr,w);
    h_tofInvBeta->Fill(hscp.Dt().second.invBeta,w);
  }
}

class HSCPStandardTree {
 public:
   HSCPStandardTree(TFileDirectory& subDir);
   void newEvent(const edm::Event&, float w);
   void fill(const HSCParticle & hscp);
 private:
   // the tree itself
   TTree* tree_;
   // variables for branches
   unsigned int event,run;
   float weight;
   float px,py,pz,p,pt;
   float tk_px, tk_py, tk_pz, tk_p, tk_pt;
   float mu_px, mu_py, mu_pz, mu_p, mu_pt;
   float tk_mass, tk_mass_error;
   float dt_mass, dt_mass_error;
   float avg_mass, avg_mass_error;
   float dtcomb_mass, dtsta_mass, dtatk_mass, dtbest_mass, tkacomb_mass;
   float tk_beta, dt_beta, beta_compatibility, ecal_beta, rpc_beta;
   int quality;
};

HSCPStandardTree::HSCPStandardTree(TFileDirectory& subDir)
{
  // creates the tree
  tree_ = subDir.make<TTree>("HSCParticles","HSCParticles");
  // set branches
  tree_->Branch("event",&event,"event/i");
  tree_->Branch("run",&run,"run/i");
  tree_->Branch("weight",&weight,"weight/F");
  tree_->Branch("px",&px,"px/F");
  tree_->Branch("py",&py,"py/F");
  tree_->Branch("pz",&pz,"pz/F");
  tree_->Branch("p",&p,"p/F");
  tree_->Branch("pt",&pt,"pt/F");
  tree_->Branch("tk_px",&tk_px,"tk_px/F");
  tree_->Branch("tk_py",&tk_py,"tk_py/F");
  tree_->Branch("tk_pz",&tk_pz,"tk_pz/F");
  tree_->Branch("tk_p",&tk_p,"tk_p/F");
  tree_->Branch("tk_pt",&tk_pt,"tk_pt/F");
  tree_->Branch("mu_px",&mu_px,"mu_px/F");
  tree_->Branch("mu_py",&mu_py,"mu_py/F");
  tree_->Branch("mu_pz",&mu_pz,"mu_pz/F");
  tree_->Branch("mu_p",&mu_p,"mu_p/F");
  tree_->Branch("mu_pt",&mu_pt,"mu_pt/F");
  tree_->Branch("tk_mass",&tk_mass,"tk_mass/F");
  tree_->Branch("tk_mass_error",&tk_mass_error,"tk_mass_error/F");
  tree_->Branch("dt_mass",&dt_mass,"dt_mass/F");
  tree_->Branch("dt_mass_error",&dt_mass_error,"dt_mass_error/F");
  tree_->Branch("avg_mass",&avg_mass,"avg_mass/F");
  tree_->Branch("avg_mass_error",&avg_mass_error,"avg_mass_error/F");
  tree_->Branch("dtcomb_mass",&dtcomb_mass,"dtcomb_mass/F");
  tree_->Branch("dtsta_mass",&dtsta_mass,"dtsta_mass/F");
  tree_->Branch("dtatk_mass",&dtatk_mass,"dtatk_mass/F");
  tree_->Branch("dtbest_mass",&dtbest_mass,"dtbest_mass/F");
  tree_->Branch("tkacomb_mass",&tkacomb_mass,"tkacomb_mass/F");
  tree_->Branch("tk_beta",&tk_beta,"tk_beta/F");
  tree_->Branch("dt_beta",&dt_beta,"dt_beta/F");
  tree_->Branch("beta_compatibility",&beta_compatibility,"beta_compatibility/F");
  tree_->Branch("ecal_beta",&ecal_beta,"ecal_beta/F");
  tree_->Branch("rpc_beta",&rpc_beta,"rpc_beta/F");
  tree_->Branch("quality",&quality,"quality/I");
}

void HSCPStandardTree::newEvent(const edm::Event& iEvent, float w)
{
  weight = w;
  run = iEvent.id().run();
  event = iEvent.id().event(); 
}

void HSCPStandardTree::fill(const HSCParticle & hscp)
{
  // set the content
  if(hscp.hasMuonCombinedTrack()) {
    px = hscp.combinedTrack().px();
    py = hscp.combinedTrack().py();
    pz = hscp.combinedTrack().pz();
  } else if(hscp.hasMuonStaTrack()) {
    px = hscp.staTrack().px();
    py = hscp.staTrack().py();
    pz = hscp.staTrack().pz();
  } else if(hscp.hasMuonTrack()) {
    px = hscp.muonTrack().px();
    py = hscp.muonTrack().py();
    pz = hscp.muonTrack().pz();
  } else if(hscp.hasMuonTrack()) {
    px = hscp.trackerTrack().px();
    py = hscp.trackerTrack().py();
    pz = hscp.trackerTrack().pz();
  } else {
    px = 0.;
    py = 0.;
    pz = 0.;
  }
  p  = hscp.p();
  pt = hscp.pt();
  tk_px = hscp.hasTrackerTrack() ? hscp.trackerTrack().px() : 0.;
  tk_py = hscp.hasTrackerTrack() ? hscp.trackerTrack().py() : 0.;
  tk_pz = hscp.hasTrackerTrack() ? hscp.trackerTrack().pz() : 0.;
  tk_p  = hscp.hasTrackerTrack() ? hscp.trackerTrack().p()  : 0.;
  tk_pt = hscp.hasTrackerTrack() ? hscp.trackerTrack().pt() : 0.;
  mu_px = hscp.hasMuonTrack() ? hscp.muonTrack().px() : 0.;
  mu_py = hscp.hasMuonTrack() ? hscp.muonTrack().py() : 0.;
  mu_pz = hscp.hasMuonTrack() ? hscp.muonTrack().pz() : 0.;
  mu_p  = hscp.hasMuonTrack() ? hscp.muonTrack().p()  : 0.;
  mu_pt = hscp.hasMuonTrack() ? hscp.muonTrack().pt() : 0.;
  tk_mass = hscp.massTk();
  tk_mass_error = hscp.massTkError();
  dt_mass = hscp.massDt();
  dt_mass_error = hscp.massDtError();
  avg_mass = hscp.massAvg();
  avg_mass_error = hscp.massAvgError();
  dtcomb_mass  = hscp.massDtComb();
  dtsta_mass   = hscp.massDtSta();
  dtatk_mass   = hscp.massDtAssoTk();
  dtbest_mass  = hscp.massDtBest();
  tkacomb_mass = hscp.massTkAssoComb();
  tk_beta = hscp.betaTk();
  dt_beta = hscp.betaDt();
  beta_compatibility = hscp.compatibility();
  ecal_beta = hscp.betaEcal();
  rpc_beta = hscp.betaRpc();
  quality = (hscp.hasTkInfo()<<8)|(hscp.hasTrackerTrack()<<7)
           |(hscp.hasDtInfo()<<6)|(hscp.hasMuonTrack()<<5)
           |(hscp.hasCaloInfo()<<4)|(hscp.hasRpcInfo()<<3);
  // fill
  tree_->Fill();
}

#define NMONITORS 33

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
      // Flags
      bool m_haveSimTracks;
      bool m_useWeights;

      // Counter
      double triggeredCounter;

      // Candidate tree
      HSCPStandardTree* candidateTree;
      
      // Standard plots
      HSCPCandidateFilter * filter;
      CutMonitor * cuts[NMONITORS];

      // SIM plots
      TH1F * h_simmu_pt; 
      TH1F * h_simmu_eta; 
      TH1F * h_simhscp_pt; 
      TH1F * h_simhscp_eta; 
};

//
// constructors and destructor
//
HSCPAnalyzer::HSCPAnalyzer(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  m_haveSimTracks=iConfig.getParameter<bool>("haveSimTracks");
  m_useWeights=iConfig.getParameter<bool>("useWeights");
  triggeredCounter=0;
  filter = new HSCPCandidateFilter;
}


HSCPAnalyzer::~HSCPAnalyzer() 
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   delete filter;
}

//
// member functions
//

// ------------ method called to for each event  ------------
void HSCPAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;

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
    w = *weightHandle;
    if(w>10) cout << "HIGH Weight: " << w 
                  << " Proc id = "   << *procIdHandle 
		  << " eff "         << *effHandle 
		  << " Xsec "        << *xsHandle     << endl;
  }

  // Initialize the tree with event info
  candidateTree->newEvent(iEvent,w);
  // Initialize cut monitors
  for(int i=0;i<NMONITORS;i++) if(cuts[i]) cuts[i]->newEvent(w);

  // Load the collection of candidates
  Handle<HSCParticleCollection> hscpH;
  iEvent.getByLabel("hscp",hscpH);
  const vector<HSCParticle> & candidates = *hscpH.product();

  // Passing the trigger (exception is thrown if above collection are not filled)
  triggeredCounter += w;
  
  // Loop over candidates
  for(vector<HSCParticle>::const_iterator hscpCand = candidates.begin(); hscpCand!=candidates.end(); ++hscpCand) {
    // fill the tree
    candidateTree->fill(*hscpCand);
    // fill the histograms for the different cuts
    for(int i=0;i<NMONITORS;++i)
      if(filter->passCut(*hscpCand,susybsm::HSCPCandidateFilter::cutName(i))) cuts[i]->passed(*hscpCand,w);
  }

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
  edm::Service<TFileService> fs;
  for(int i=0;i<NMONITORS;i++) cuts[i] = new CutMonitor(filter->nameCut(susybsm::HSCPCandidateFilter::cutName(i)),fs);
  candidateTree = new HSCPStandardTree(*fs);

  TFileDirectory subDir2 = fs->mkdir( "Sim" );
  h_simmu_pt    = subDir2.make<TH1F>( "mu_sim_pt"  , "p_{t} mu", 100,  0., 1500. );
  h_simmu_eta   = subDir2.make<TH1F>( "mu_sim_eta"  , "\\eta mu", 50,  -4., 4. );
  h_simhscp_pt  = subDir2.make<TH1F>( "mu_hscp_pt"  , "p_{t} hscp", 100,  0., 1500. );
  h_simhscp_eta = subDir2.make<TH1F>( "mu_hscp_eta"  , "\\eta hscp" , 50,  -4., 4. );
}

// ------------ method called once each job just after ending the event loop  ------------
void HSCPAnalyzer::endJob() {
  // Final summary : Processed and Selected events
  cout << "Processed events: "     << cuts[0]->totalEvents() <<  endl;
  cout << "Selected events: "      << cuts[32]->eventCount() <<  endl;
  cout << "Selected tof  events: " << cuts[31]->eventCount() <<  endl;
  cout << "Selected dedx events: " << cuts[30]->eventCount() <<  endl;
  
  // Final summary : MIP and protons (no selection cut)
  TH1F * h_dedxMIP = cuts[1]->m_plots.h_dedxMIP;
  TH1F * h_dedxMassProton = cuts[1]->m_plots.h_dedxMassProton;
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
  cout << "    Proton Mass : " << h_dedxMassProton->GetFunction("gaus")->GetParameter(1) << endl;
  cout << "    MIP  mean : " <<  mipMean << "    sigma : " << mipSigma << endl;
  for(int i=0;i<6;i++)   
    cout << "    " << (1-effPoints[i])*100 << "% @ dedx > " << dedxEff[i] << "+"<< dedxEffP[i] << " -" << dedxEffM[i] << endl;
  cout << endl;

  // Final summary : Selections
  cout << "CutTitle: ";
  for(int i=0;i<NMONITORS;i++) if(cuts[i]) { 
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
