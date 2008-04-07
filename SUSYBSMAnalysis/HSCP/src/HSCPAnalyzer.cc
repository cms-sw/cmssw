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
// $Id: HSCPAnalyzer.cc,v 1.24 2008/03/17 17:44:56 ptraczyk Exp $
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

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"
#include "Math/GenVector/VectorUtil.h"


#include <iostream>
#include <vector>
#include <TH1F.h>
#include <TH2F.h>
#include <TF1.h>
//
// class decleration
//

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
using namespace susybsm;

using namespace reco;
using namespace std;
using namespace edm;

class PtSorter {
 public:
  template <class T> bool operator() ( const T& a, const T& b ) {
     return ( a->pt() > b->pt() );
  }
};




class HSCPStandardPlots
 {
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


class CutMonitor
{
public:
 CutMonitor(std::string name,edm::Service<TFileService> fs):m_name(name), m_plots(fs->mkdir(name)), m_newevent(true),m_evCounter(0),m_candCounter(0),m_tot(0){ }
 
 void newEvent(float w) {m_newevent=true; m_tot+=w; }
 void passed(const HSCParticle & hscp, double w) 
   {
        m_plots.fill(hscp,w);  
        if(m_newevent) m_evCounter+=w;
        m_candCounter+=w;
        m_newevent=false;
   }
  void print()
  {
   cout << fixed << setprecision(1) << setw(6) <<  m_candCounter << "(" << fixed << setprecision(1) << setw(6) << m_evCounter << ")";
  }
  void printEff()
  {
   cout << fixed << setprecision(3) << setw(5) <<  m_candCounter/m_tot << "(" << fixed << setprecision(3) << setw(5) << m_evCounter/m_tot << ")";
  }
 
  void printName()
  {
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

HSCPStandardPlots::HSCPStandardPlots(TFileDirectory  subDir)
{
  h_massVsMass =          subDir.make<TH2F>("tof_mass_vs_dedx_mass","Mass tof vs Mass dedx", 100,0,1200,100,0,1200);
  h_massVsBeta =          subDir.make<TH2F>("avgMass_vs_avgBeta","Mass(avg) vs Beta(avg)", 100,0,1200,50,0,1);
  h_massVsPtError =       subDir.make<TH2F>("avgMass_vs_ptError","Mass(avg) vs log(ptError) (beta < 0.85)", 100,0,1200,25,0,10);
  h_massVsPt =    subDir.make<TH2F>("mass_vs_pt","Mass vs pt", 100,0,1200,100,0,1500);
  h_massVsMassError =     subDir.make<TH2F>("avgMass_vs_MassError","Mass(avg) vs log(masstError)", 100,0,1200,100,0,2);
  h_massVsBeta_tk =       subDir.make<TH2F>("tkMass_vs_tkBeta","Mass(tk) vs Beta(tk)", 100,0,1200,50,0,1);
  h_massVsPtError_tk =    subDir.make<TH2F>("tkMass_vs_ptError","Mass(tk) vs log(ptError) (beta < 0.85)", 100,0,1200,25,0,10);
  h_massVsMassError_tk =  subDir.make<TH2F>("tkMass_vs_MassError","Mass(tk) vs mass error", 100,0,1200,100,0,2);
  h_betaVsBeta =          subDir.make<TH2F>("tof_beta_vs_beta","INVBeta tof vs INVbeta dedx (Pt>30)", 100,0,3,100,0,3);

  h_massTk =          subDir.make<TH1F>("mass_tk","Mass tk", 100,0,1200);
  h_massDt =          subDir.make<TH1F>("mass_dt","Mass dt", 100,0,1200);
  h_massAvg =          subDir.make<TH1F>("mass_avg","Mass avg", 100,0,1200);

  h_dedxHits = subDir.make<TH1F>("dedx_hits","# hits dedx", 25,-0.5,24.5);
  h_dedxBeta = subDir.make<TH1F>("dedx_beta","Beta tk", 100,0,1);
  h_dedxHitsBeta = subDir.make<TH2F>("dedx_hits_vs_beta","dedx #hits vs beta tk", 25,-0.5,24.5,100,0,1);

  h_deltaBeta = subDir.make<TH1F>("delta_beta","Delta Beta", 200,-1,1);
  h_deltaBetaInv =  subDir.make<TH1F>("delta_betaInv","Delta BetaInv", 200,-3,3);

  h_tofBeta = subDir.make<TH1F>( "tof_beta"  , " tof beta  ",100,0,1);
  h_tofInvBeta = subDir.make<TH1F>( "tof_Invbeta"  , " tof beta  ",500,0,5);
  h_tofInvBetaErr = subDir.make<TH1F>( "tof_inv_beta_err"  , " tof beta err  ",100,0,1);
  h_tofBetaPull = subDir.make<TH1F>( "tof_beta_pull"  , " tof beta pull  ",100,-10,10);

}

void HSCPStandardPlots::fill(const HSCParticle & hscp, double w)
{
  

  double avgMass = (hscp.massDt()+hscp.massTk())/2.;
  double avgMassError;
  double ptMassError=(hscp.tk.track->ptError()/hscp.tk.track->pt());
  ptMassError*=ptMassError;
  double ptMassError2=(hscp.dt.first->track()->ptError()/hscp.dt.first->track()->pt());
  ptMassError2*=ptMassError2;
  double ib2 = hscp.dt.second.invBeta*hscp.dt.second.invBeta;
  double dtMassError=hscp.dt.second.invBetaErr*(ib2/sqrt(ib2-1)) ;
  dtMassError*= dtMassError;
  double dedxError = 0.2*sqrt(10./hscp.tk.nDeDxHits)*0.4/hscp.tk.invBeta2;    //TODO: FIXED IT!!!
  double tkMassError = dedxError/(2.*hscp.tk.invBeta2-1);
  tkMassError*=tkMassError;
  avgMassError=sqrt(ptMassError/4+ptMassError2/4.+dtMassError/4.+tkMassError/4.);

  h_massTk->Fill(hscp.massTk(),w);
  h_massDt->Fill(hscp.massDt(),w);
  h_massAvg->Fill(avgMass,w);
  h_dedxHits->Fill(hscp.tk.nDeDxHits,w);
  h_dedxBeta->Fill(1./sqrt(hscp.tk.invBeta2),w); 
  h_dedxHitsBeta->Fill(hscp.tk.nDeDxHits,1./sqrt(hscp.tk.invBeta2),w);

  h_deltaBeta->Fill(1./hscp.dt.second.invBeta-1./sqrt(hscp.tk.invBeta2),w);
  h_deltaBetaInv->Fill(hscp.dt.second.invBeta-sqrt(hscp.tk.invBeta2),w);

  h_massVsMass->Fill(hscp.massDt(),hscp.massTk(),w);
  h_massVsBeta->Fill(avgMass,2./(sqrt(hscp.tk.invBeta2Fit)+hscp.dt.second.invBeta),w);
  h_massVsPtError->Fill(avgMass,log10(hscp.tk.track->ptError()),w);
  h_massVsPt->Fill(avgMass,hscp.tk.track->pt(),w);
  h_massVsMassError->Fill((hscp.massDt()+hscp.massTk())/2.,avgMassError,w);
  h_massVsMassError_tk->Fill(hscp.massTk(),tkMassError,w);
  h_massVsBeta_tk->Fill(hscp.massTk(),1./sqrt(hscp.tk.invBeta2Fit),w);
  h_massVsPtError_tk->Fill(hscp.massTk(),log10(hscp.tk.track->ptError()),w);
  h_betaVsBeta->Fill(hscp.dt.second.invBeta,sqrt(hscp.tk.invBeta2),w);
  h_tofBeta->Fill(1./hscp.dt.second.invBeta,w);
  h_tofInvBetaErr->Fill(hscp.dt.second.invBetaErr,w);
  h_tofInvBeta->Fill(hscp.dt.second.invBeta,w);
  if(hscp.dt.second.invBeta !=0 && hscp.dt.second.invBetaErr !=0 && hscp.dt.second.invBetaErr < 1000. )  h_tofBetaPull->Fill((hscp.dt.second.invBeta-1)/hscp.dt.second.invBetaErr,w);

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

      edm::InputTag m_dedxSrc;
      bool m_haveSimTracks;
      bool m_useWeights;
       
      TH1F * h_pt;
// RECO DEDX
      TH1F * h_dedxMassSel;
      TH1F * h_dedxMassSelFit;
      TH1F * h_dedxMass;
      TH1F * h_dedxMassMu;
      TH1F * h_dedxMassFit;
      TH1F * h_dedxMassProton;
      TH1F * h_dedxMassProtonFit;
      TH2F * h_dedx;
      TH2F * h_dedxCtrl;
      TH2F * h_dedxFit;
      TH2F * h_dedxFitCtrl;
      TH1F * h_dedxMIP;
      TH1F * h_dedxFitMIP;
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


      CutMonitor * cuts[40];


 // SIM
      TH1F * h_simmu_pt; 
      TH1F * h_simmu_eta; 
      TH1F * h_simhscp_pt; 
      TH1F * h_simhscp_eta; 

/*class DTInfo
  {
    public:
      TrackRef standaloneTrack;
      TrackRef trackerTrack;
      TrackRef combinedTrack;
      float invBeta; 
  };  
 class TKInfo
  {
    public:
      TrackRef track;
      float invBeta2Fit;
      float invBeta2;
  };

 class HSCParticle 
  {
   public:
     DTInfo dt;
     TKInfo tk;
     float massTk() {return tk.track->p()*sqrt(tk.invBeta2-1);}
     float massDt() {return dt.trackerTrack->p()*sqrt(dt.invBeta*dt.invBeta-1);}
     bool hasDt;
     bool hasTk;
  };*/
//vector<HSCParticle> associate(vector<TKInfo> & dts, vector<DTInfo> & tks );

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
void
HSCPAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{


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

//  cout << "auto Xsec: " << auto_cross_section << "nb    precalc Xsev "<< external_cross_section << "pb" << "    filter eff: "<<filter_eff <<endl;

   double w=1.;
   //Event flags
   int dedxSelectionLevel = -1;

   // FIXME: Use candidate + errors 
   vector<float> dedxMass,dedxP,dedxMPV,dedxChi,dedxNHits,dedxMPV2;
   vector<float> tofMass,tofP,tofValue;

   bool highptmu =false; 

   if(m_useWeights)
   {
	Handle<double> xsHandle;
	iEvent.getByLabel ("genEventRunInfo","PreCalculatedCrossSection", xsHandle);
	Handle<double> effHandle;
	iEvent.getByLabel ("genEventRunInfo","FilterEfficiency", effHandle);
	Handle<double> weightHandle;
	iEvent.getByLabel ("weight", weightHandle);
        Handle<int> procIdHandle;
        iEvent.getByLabel("genEventProcID",procIdHandle);
	w = * weightHandle;
  if(w>10)
       cout << "HIGH Weight: " << w << " Proc id = " << *procIdHandle << " eff " << *effHandle << "   Xsec "<< *xsHandle << endl;

   /*    Handle<double>  weightH;
       Handle<double>  scaleH;
       iEvent.getByLabel("genEventWeight",weightH);
       iEvent.getByLabel("genEventScale",scaleH);

       w= * weightH.product()      ;
       double s= * scaleH.product()      ;
*/
   }

  for(int i=0;i<40;i++) if(cuts[i]) cuts[i]->newEvent(w);
  tot+=w;
  
  //Sel before cuts
  selectedAfterCut[0]+=w;

   Handle<reco::MuonCollection> pIn;
   iEvent.getByLabel("muons",pIn);
   const reco::MuonCollection & muons = * pIn.product();

   Handle< MuonTOFCollection >  betaRecoH;
   iEvent.getByLabel("betaFromTOF",betaRecoH);
   const MuonTOFCollection & betaReco = *betaRecoH.product();
 
   //Passing the trigger (exception is thrown if above collection are not filled)
   selectedAfterCut[1]+=w;

   std::vector<TrackRef> tkMuons;
   std::vector<TrackRef> staMuons;
   std::vector<TrackRef> combMuons;

   int i=0;
   reco::MuonCollection::const_iterator muonIt = muons.begin();
   
   for(; muonIt != muons.end() ; ++muonIt) {
     TrackRef tkMuon = muonIt->track();
     TrackRef staMuon = muonIt->standAloneMuon();
     TrackRef combMuon = muonIt->combinedMuon();
     if(tkMuon.isNonnull())     tkMuons.push_back(tkMuon);
     if(staMuon.isNonnull())    staMuons.push_back(staMuon);
     if(combMuon.isNonnull())   combMuons.push_back(combMuon);

     double p=0;
     if(tkMuon.isNonnull()) {
       double pt=tkMuon->pt();
       h_pt->Fill(tkMuon->pt(),w); 
       if(combMuon.isNonnull()) pt=combMuon->pt();
       if(pt>100) highptmu=true;
       p= tkMuon->p();
     }
     if(staMuon.isNonnull()) { 
       p=staMuon->p();
       h_tofPtSta->Fill(p,w);
     }  
     if(combMuon.isNonnull()) {
       //FIXME: tobeused as default
       //p=combMuon->p();
       h_tofPtComb->Fill(combMuon->p(),w);
     }  

     double invbeta = betaReco[i].second.invBeta;
     double invbetaerr = betaReco[i].second.invBetaErr;
     double mass = p*sqrt(invbeta*invbeta-1);
     double mass2 = p*p*(invbeta*invbeta-1);
       
     cout << " Muon p: " << p << " invBeta: " << invbeta << " Mass: " << mass << endl;

     if( betaReco[i].second.invBetaErr < 0.07)  h_tofBetaCut->Fill(1./invbeta , w);

     h_tofBeta->Fill(invbeta,w);
     h_tofBetaErr->Fill(invbetaerr,w);
     h_tofBetaPull->Fill((invbeta-1.)/invbetaerr,w);
     h_tofBetaPullp->Fill(p,(invbeta-1.)/invbetaerr,w);
     h_tofBetap->Fill(p,invbeta,w);
     h_tofMassp->Fill(p,mass,w);
     h_tofMass->Fill(mass,w);
     h_tofMass2->Fill(mass2,w);
     tofMass.push_back(mass);
     tofP.push_back(p);
     tofValue.push_back(invbeta);
     i++;
   }

   std::sort(tkMuons.begin()  , tkMuons.end()  ,PtSorter());
   std::sort(staMuons.begin() , staMuons.end() ,PtSorter());
   std::sort(combMuons.begin(), combMuons.end(),PtSorter());
   if(tkMuons.size  ()>0)  h_tkmu_pt->Fill(tkMuons[0]  ->pt(),w);
   if(staMuons.size ()>0)  h_stamu_pt->Fill(staMuons[0] ->pt(),w);
   if(combMuons.size()>0)  h_combmu_pt->Fill(combMuons[0]->pt(),w);

   cerr << "AFTER Mu" << endl;
   Handle<TrackDeDxEstimateCollection> dedxH;
   Handle<TrackDeDxHitsCollection> dedxHitsH;
//   iEvent.getByLabel("dedxTruncated40",dedxH);
   iEvent.getByLabel("dedxHitsFromRefitter",dedxHitsH);
   iEvent.getByLabel(m_dedxSrc,dedxH);
   Handle<vector<float> >  dedxFitH;
   iEvent.getByLabel("dedxFit",dedxFitH);

   const TrackDeDxEstimateCollection & dedx = *dedxH.product();
   const TrackDeDxHitsCollection & dedxHits = *dedxHitsH.product();

   const vector<float> & dedxFit = *dedxFitH.product();
    cout << "Number of tracks with dedx: " << dedx.size() << endl; 
   for(size_t i=0; i<dedx.size() ; i++)
    {
        int usedhits=0;
        for(reco::DeDxHitCollection::const_iterator it_hits = dedxHits[i].second.begin(); it_hits!=dedxHits[i].second.end();it_hits++)
         {  if(it_hits->subDet() != 1 && it_hits->subDet() != 2 ) usedhits++;       }

      if(dedx[i].first->normalizedChi2() < 5 && dedx[i].first->numberOfValidHits()>8 && usedhits >= 9)
       {
         float dedxVal= dedx[i].second;
         float dedxFitVal= dedxFit[i];
         float p= dedx[i].first->p();

         h_dedx->Fill(p, dedxVal,w);   
         h_dedxCtrl->Fill(p, dedxVal,w);   

         h_dedxFit->Fill(p, dedxFitVal,w);   
         h_dedxFitCtrl->Fill(p, dedxFitVal,w);   
         float k=0.4;  //919/2.75*0.0012;
         float k2=0.432; //919/2.55*0.0012;
         float mass=p*sqrt(k*dedxVal-1);
         float mass2=p*sqrt(k2*dedxFitVal-1);

         if(p > 5 && p < 30 )  
          {
             h_dedxMIP->Fill( dedxVal,w);   
             h_dedxFitMIP->Fill( dedxFitVal,w);   
             if(dedxVal >3.22)
              {
              std::cout << dedx[i].first->normalizedChi2() << " " << dedx[i].first->numberOfValidHits() << " " << p <<std::endl;
              }

         h_dedxMIPbeta->Fill(1./sqrt(k*dedxVal),w);
         if(usedhits >= 12)
         h_dedxMIPbetaCut->Fill(1./sqrt(k*dedxVal),w);
         
         }



         h_dedxMass->Fill(mass,w); 
         h_dedxMassFit->Fill(mass2,w); 
        if(p > 30 && dedxVal > 3.45  ) {  h_dedxMassSel->Fill(mass,w); }
        if(p > 30 && dedxFitVal > 3.3 ) { h_dedxMassSelFit->Fill(mass2,w);     }        

         if(p < 1.2 && mass2 > 0.200 )
          {
           h_dedxMassProton->Fill(mass,w);
          }
         if(p < 1.2 && mass > 0.200 )
          {
           h_dedxMassProtonFit->Fill(mass2,w);
          }

    //Dedx for analsyis
    //FIXME: make it configurable
    double dedxFitCut[6] = {3.0,3.16,3.24,3.64,4.68,6.2};
    for(int ii=0;ii<6;ii++)
      {
        if(dedxFitVal > dedxFitCut[ii]) 
        {
         dedxSelectionLevel=ii; 
         h_pSpectrumAfterSelection[ii]->Fill(p,w);
         h_massAfterSelection[ii]->Fill(mass2,w);      
        }
         dedxMass.push_back(mass2);
         dedxP.push_back(p);
         dedxMPV.push_back(dedxFitVal);
         dedxChi.push_back(dedx[i].first->normalizedChi2());
         dedxNHits.push_back( dedx[i].first->numberOfValidHits());
         dedxMPV2.push_back(dedxVal);
      }


        }
    }

   Handle<HSCParticleCollection> hscpH;
   iEvent.getByLabel("hscp",hscpH);

const vector<HSCParticle> & candidates = *hscpH.product();


if(dedxMass.size() > 0 && tofMass.size() > 0 )
{
 if(find_if(dedxMass.begin(), dedxMass.end(), bind2nd(greater<float>(), 100.))!= dedxMass.end() &&
    find_if(tofMass.begin(), tofMass.end(), bind2nd(greater<float>(), 100.))!= tofMass.end() )
  {
     selected+=w;
  }
  int i=max_element(tofValue.begin(), tofValue.end())-tofValue.begin();
  int j=max_element(dedxMPV.begin(), dedxMPV.end())-dedxMPV.begin();
  if(tofValue[i] > 1.1 || dedxMPV[j] > 3.24)
  if(tofP[i] > 50 && dedxP[j] > 50)
   if((tofMass[i] + dedxMass[j]) > 150)  
   {
       
     cout << "SIGNAL:" <<  " P: " << dedxP[j] <<  " Chi: " << dedxChi[j] << " #hit: " << dedxNHits[j]  << " MPV: " << dedxMPV[j] << " MPV2: " << dedxMPV2[j] << 
         " m1 m2  " << tofMass[i] << " " << dedxMass[j] << " " << tot << endl;

    }
//   h_massVsMass->Fill(tofMass[i],dedxMass[j],w); 

}

if(find_if(dedxMass.begin(), dedxMass.end(), bind2nd(greater<float>(), 100.))!= dedxMass.end()) selectedDedx+=w;
if(find_if(tofMass.begin(), tofMass.end(), bind2nd(greater<float>(), 100.))!= tofMass.end() ) selectedTOF+=w;



for(unsigned int i=0; i < candidates.size();i++)
{
//  if(nosel && dt080 && tk080 &&) cuts[7]->passed(candidates[i],w);
//  if(nosel && dt080 && tk080 &&) cuts[8]->passed(candidates[i],w);



}




bool found=false;
bool found2=false;
bool found3=false;
bool found4=false;
bool found5=false;
for(int i=0; i < candidates.size();i++) 
{
  if(candidates[i].hasTk && candidates[i].hasDt && candidates[i].hasMuonTrack()) selectedAfterCut[19]+=w;
  if(candidates[i].hasTk && candidates[i].hasDt && candidates[i].hasMuonTrack() && candidates[i].dt.second.invBeta > 1.176) { found5=true; selectedAfterCut[10]+=w; }
  if(candidates[i].hasTk && candidates[i].hasDt && candidates[i].hasMuonTrack() && candidates[i].dt.second.invBeta > 1.25) { found2=true; selectedAfterCut[11]+=w; }
  if(candidates[i].hasTk && candidates[i].hasDt && candidates[i].hasMuonTrack() && candidates[i].tk.invBeta2 > 1.56) { found=true; selectedAfterCut[12]+=w; }
  if(candidates[i].hasTk && candidates[i].hasDt && candidates[i].hasMuonTrack() && fabs(sqrt(1./candidates[i].tk.invBeta2Fit)- 1./candidates[i].dt.second.invBeta )  < 0.1) { found3=true;  }
  if(candidates[i].hasTk && candidates[i].hasDt && candidates[i].hasMuonTrack() && candidates[i].dt.second.invBeta > 1.25 && candidates[i].tk.invBeta2 > 1.56 ) {  selectedAfterCut[13]+=w; }
  

}
if(found5)   selectedAfterCut[2]+=w;
if(found2)   selectedAfterCut[3]+=w;
if(found)   selectedAfterCut[4]+=w;
if(found3)   selectedAfterCut[5]+=w;

if(found && found2)   selectedAfterCut[6]+=w;
if(found && found2 && found3)   selectedAfterCut[7]+=w;






for(unsigned int i=0; i < candidates.size();i++)
{
  h_betaVsBeta->Fill(candidates[i].dt.second.invBeta,sqrt(candidates[i].tk.invBeta2),w);


 // cout << candidates[i].massDt() << " " << candidates[i].massTk() << " " << candidates[i].tk.track->momentum() << " " <<  candidates[i].dt.combinedTrack->momentum() << endl;




if(candidates[i].hasTk && candidates[i].hasDt && candidates[i].hasMuonTrack()) // && fabs(sqrt(1./candidates[i].tk.invBeta2Fit)- 1./candidates[i].dt.second.invBeta )  < 0.1)
 { 
  double avgMass = (candidates[i].massDt()+candidates[i].massTk())/2.;
  h_massVsBeta->Fill(avgMass,2./(sqrt(candidates[i].tk.invBeta2Fit)+candidates[i].dt.second.invBeta),w);
  if(2./(sqrt(candidates[i].tk.invBeta2Fit)+candidates[i].dt.second.invBeta) < 0.85)
  h_massVsPtError->Fill(avgMass,log10(candidates[i].tk.track->ptError()),w);

  double avgMassError;

  double ptMassError=(candidates[i].tk.track->ptError()/candidates[i].tk.track->pt());
  ptMassError*=ptMassError;  
  double ptMassError2=(candidates[i].dt.first->track()->ptError()/candidates[i].dt.first->track()->pt());
  ptMassError2*=ptMassError2;  
//double dtMassError=(candidates[i].dt.second.invBetaErr/(2.*candidates[i].dt.second.invBeta-1)) ;
  double ib2 = candidates[i].dt.second.invBeta*candidates[i].dt.second.invBeta;
  double dtMassError=candidates[i].dt.second.invBetaErr*(ib2/sqrt(ib2-1)) ;
  dtMassError*= dtMassError;

  double dedxError = 0.2*sqrt(10./candidates[i].tk.nDeDxHits)*0.4/candidates[i].tk.invBeta2;    //TODO: FIXED IT!!!
  double tkMassError = dedxError/(2.*candidates[i].tk.invBeta2-1); 
  tkMassError*=tkMassError;

  avgMassError=sqrt(ptMassError/4+ptMassError2/4.+dtMassError/4.+tkMassError/4.);



  bool nosel=candidates[i].hasTk && candidates[i].hasDt && candidates[i].hasMuonTrack();
  bool dte07 = candidates[i].dt.second.invBetaErr < 0.07 ;
  bool dte10 = candidates[i].dt.second.invBetaErr < 0.10 ;
  bool dt080  = candidates[i].dt.second.invBeta > 1.25 &&  candidates[i].dt.second.invBeta < 1000. && dte10;
  bool dt085 = candidates[i].dt.second.invBeta > 1.176 &&  candidates[i].dt.second.invBeta < 1000. && dte10;
  bool tk080  = candidates[i].tk.invBeta2 > 1.56 ;
  bool tkpt100  = candidates[i].tk.track->pt() > 100 && candidates[i].dt.first->combinedMuon().isNonnull() && candidates[i].dt.first->combinedMuon()->pt() > 100;
  bool tkm100  = candidates[i].massTk() > 100 ;
  bool tkm200  = candidates[i].massTk() > 200 ;
  bool tkm400  = candidates[i].massTk() > 400 ;
  bool tkhits14 = candidates[i].tk.nDeDxHits >= 14; 

  bool db01  =  fabs(sqrt(1./candidates[i].tk.invBeta2Fit)- 1./candidates[i].dt.second.invBeta )  < 0.1 ;
  bool err015 = avgMassError < 0.15;
  if(nosel) cuts[0]->passed(candidates[i],w);
  if(nosel && dt085 ) cuts[1]->passed(candidates[i],w);
  if(nosel && dt080 ) cuts[2]->passed(candidates[i],w);
  if(nosel && tk080 ) cuts[3]->passed(candidates[i],w);
  if(nosel && dt080 && tk080) cuts[4]->passed(candidates[i],w);
  if(nosel && db01 )  cuts[5]->passed(candidates[i],w);
  if(nosel && dt080 && tk080 && db01) cuts[6]->passed(candidates[i],w);
  if(nosel && dt080 && tk080 && dte07) cuts[7]->passed(candidates[i],w);
  if(nosel && dt085 && tk080 ) cuts[8]->passed(candidates[i],w);
  if(nosel && tk080 && tkpt100 ) cuts[9]->passed(candidates[i],w);
  if(nosel && tk080 && tkpt100 && tkhits14 ) cuts[10]->passed(candidates[i],w);
  if(nosel && tk080 && tkpt100 && tkm100 ) cuts[11]->passed(candidates[i],w);
  if(nosel && tk080 && tkpt100 && tkm200 ) cuts[12]->passed(candidates[i],w);
  if(nosel && tk080 && tkpt100 && tkm400 ) cuts[13]->passed(candidates[i],w);
  if(nosel && dte10 ) cuts[14]->passed(candidates[i],w);
  if(nosel && dte07 ) cuts[15]->passed(candidates[i],w);

/*  if(nosel && dt085 && dte10 ) cuts[10]->passed(candidates[i],w);
  if(nosel && dt080 && dte10 ) cuts[11]->passed(candidates[i],w);
  if(nosel && dt085 && dte10 && tk080 ) cuts[12]->passed(candidates[i],w);
  if(nosel && dt080 && dte10 && tk080 ) cuts[13]->passed(candidates[i],w);
  if(nosel && dt085 && dte10 && tk080 && err015) cuts[14]->passed(candidates[i],w);
  if(nosel && dt080 && dte10 && tk080 && err015) cuts[15]->passed(candidates[i],w);
*/


  if(avgMassError < 0.05 + 0.1* (candidates[i].massDt()+candidates[i].massTk())/1000.) 
  { 
    found4=true;
//    selectedAfterCut[14]+=w;
  }
  
  if( candidates[i].dt.second.invBeta > 1.25 && candidates[i].tk.invBeta2 > 1.56 && avgMassError < 0.15 )
  {
 
    selectedAfterCut[14]+=w;
    if((candidates[i].massDt()+candidates[i].massTk())/2. > 100)     selectedAfterCut[15]+=w;
    if((candidates[i].massDt()+candidates[i].massTk())/2. > 200)     selectedAfterCut[16]+=w;
    if((candidates[i].massDt()+candidates[i].massTk())/2. > 300)     selectedAfterCut[17]+=w;
    if((candidates[i].massDt()+candidates[i].massTk())/2. > 600)     selectedAfterCut[18]+=w;
  }
  
   h_massVsMassError->Fill((candidates[i].massDt()+candidates[i].massTk())/2.,avgMassError,w);

// }

 if((candidates[i].dt.second.invBeta >1.1  )|| ( candidates[i].tk.invBeta2 > 1.3 && candidates[i].hasDt ) )
 {
  h_massVsMass->Fill(candidates[i].massDt(),candidates[i].massTk(),w);
 if(candidates[i].hasTk && candidates[i].hasDt && candidates[i].hasMuonTrack() && candidates[i].massDt() + candidates[i].massTk() >  280 && fabs(sqrt(1./candidates[i].tk.invBeta2Fit)- 1./candidates[i].dt.second.invBeta ) < 0.1 && candidates[i].massDt() > 100 
    &&  candidates[i].massTk() > 100  && candidates[i].tk.invBeta2Fit > 1.4 &&  candidates[i].dt.second.invBeta  > 1.11 )
    {
      h_massVsMassSel->Fill(candidates[i].massDt(),candidates[i].massTk(),w);
    }
//  if(candidates[i].massDt() + candidates[i].massTk() > 150 || (candidates[i].massTk() > 150 && candidates[i].tk.invBeta2Fit > 1.57 ) || candidates[i].massDt() > 150 )
      cout << "CANDIDATE " <<  candidates[i].massDt() << " " << candidates[i].massTk() << " " << candidates[i].tk.track->momentum() << " " <<  candidates[i].dt.first->combinedMuon()->momentum() << " " << candidates[i].dt.first->track()->pt() ;
      cout <<" dt beta: " << 1./candidates[i].dt.second.invBeta << " tk beta : "<< sqrt(1./candidates[i].tk.invBeta2) << " tk beta fit: "<< sqrt(1./candidates[i].tk.invBeta2Fit) <<
      cout <<"chi &  # hits: " <<  candidates[i].tk.track->normalizedChi2() << " " << candidates[i].tk.track->numberOfValidHits() ; 
      cout << "errors: " << avgMassError << " sqrt( " <<ptMassError << "/4 + "<< ptMassError2 << "/4 + "<< dtMassError << "/4 + " << tkMassError << "/4) " << dedxError <<  endl;
 }
 }

//////if(candidates[i].tk.invBeta2 > 1.56 && candidates[i].hasDt && ) 
if(candidates[i].hasTk && candidates[i].hasDt && candidates[i].massTk() > 100 && candidates[i].dt.first->standAloneMuon()->pt() > 100 && candidates[i].tk.invBeta2 > 1.56  )
 {
   h_dedxMassMu->Fill(candidates[i].massTk(),w); 
   h_dedxMassMuVsPtError->Fill(candidates[i].massTk(),candidates[i].tk.track->ptError()/candidates[i].tk.track->pt(),w);

  double ptMassError=(candidates[i].tk.track->ptError()/candidates[i].tk.track->pt());
  ptMassError*=ptMassError;
  double dedxError = 0.2*sqrt(10./candidates[i].tk.nDeDxHits)*0.4/candidates[i].tk.invBeta2;    //TODO: FIX IT!!!
  double tkBetaMassError = dedxError/(2.*candidates[i].tk.invBeta2-1);
  tkBetaMassError*=tkBetaMassError;

  double tkMassError=sqrt(ptMassError+tkBetaMassError);

  h_massVsMassError_tk->Fill(candidates[i].massTk(),tkMassError,w);
  h_massVsBeta_tk->Fill(candidates[i].massTk(),1./sqrt(candidates[i].tk.invBeta2Fit),w);
  h_massVsPtError_tk->Fill(candidates[i].massTk(),log10(candidates[i].tk.track->ptError()),w);


 
 }

}

if(found && found2 &&  found4)   selectedAfterCut[8]+=w;


if(m_haveSimTracks)
 {
  Handle<edm::SimTrackContainer> simTracksHandle;
  iEvent.getByLabel("g4SimHits",simTracksHandle);
  const SimTrackContainer simTracks = *(simTracksHandle.product());

  SimTrackContainer::const_iterator simTrack;
  for (simTrack = simTracks.begin(); simTrack != simTracks.end(); ++simTrack){

      if (abs((*simTrack).type()) > 1000000) {
        h_simhscp_pt->Fill((*simTrack).momentum().pt(),w);
        h_simhscp_eta->Fill(((*simTrack).momentum().eta()),w);
      }

      if (abs((*simTrack).type()) == 13) {
        h_simmu_pt->Fill((*simTrack).momentum().pt(),w);
        h_simmu_eta->Fill(((*simTrack).momentum().eta()),w);
    }
  }
}


}


// ------------ method called once each job just before starting event loop  ------------
void 
HSCPAnalyzer::beginJob(const edm::EventSetup&)
{

  for(int i=0;i<20;i++) selectedAfterCut[i] = 0;
  for(int i=0;i<40;i++) cuts[i] = 0;

  edm::Service<TFileService> fs;

  cuts[0]= new CutMonitor("NoSel",fs);
  cuts[1]= new CutMonitor("DT-085",fs);
  cuts[2]= new CutMonitor("DT-080",fs);
  cuts[3]= new CutMonitor("TK-080",fs);
  cuts[4]= new CutMonitor("TK-080_DT-080",fs);
  cuts[5]= new CutMonitor("DB-01",fs);
  cuts[6]= new CutMonitor("TK-080_DT-080_DB-01",fs);
  cuts[7]= new CutMonitor("TK-080_DT-080_DTE-07",fs);
  cuts[8]= new CutMonitor("TK-080_DT-085",fs);
  cuts[9] = new CutMonitor("TK-080_TKPT-100",fs);
  cuts[10] = new CutMonitor("TK-080_TKPT-100_DEDH14",fs);
  cuts[11] = new CutMonitor("TK-080_TKPT-100_M-100",fs);
  cuts[12] = new CutMonitor("TK-080_TKPT-100_M-200",fs);
  cuts[13] = new CutMonitor("TK-080_TKPT-100_M-400",fs);
  cuts[14] = new CutMonitor("DTE-10",fs);
  cuts[15] = new CutMonitor("DTE-07",fs);

/*cuts[10]= new CutMonitor("DT-085_DTE-10",fs);
  cuts[11]= new CutMonitor("DT-080_DTE-10",fs);
  cuts[12]= new CutMonitor("TK-080_DT-085_DTE-10",fs);
  cuts[13]= new CutMonitor("TK-080_DT-080_DTE-10",fs);
  cuts[14]= new CutMonitor("TK-080_DT-085_DTE-10_ERR-15",fs);
  cuts[15]= new CutMonitor("TK-080_DT-080_DTE-10_ERR-15",fs);
*/



  float minBeta = 0.8l;
  float maxBeta = 3.;

//------------ RECO DEDX ----------------
  TFileDirectory subDir = fs->mkdir( "RecoDeDx" );
  h_pt =  subDir.make<TH1F>( "mu_pt"  , "p_{t}", 100,  0., 1500. );
  h_dedx =  subDir.make<TH2F>( "dedx_p"  , "\\frac{dE}{dX} vs p", 100,  0., 1500., 100,0,8 );
  h_dedxCtrl =  subDir.make<TH2F>( "dedx_lowp"  , "\\frac{dE}{dX} vs p", 100,  0., 3., 100,0,8 );
  h_dedxMIP =  subDir.make<TH1F>( "dedxMIP"  , "\\frac{dE}{dX}  ",100,0,8 );
  h_dedxMIPbeta = subDir.make<TH1F>( "dedxMIP_beta"  , "\\frac{dE}{dX}  ",100,0,1);
  h_dedxMIPbetaCut = subDir.make<TH1F>( "dedxMIP_beta_cut"  , "\\frac{dE}{dX}  ",100,0,1);
  h_dedxFit =  subDir.make<TH2F>( "dedx_p_FIT"  , "\\frac{dE}{dX} vs p", 100,  0., 1500., 100,0,8 );
  h_dedxFitCtrl =  subDir.make<TH2F>( "dedx_lowp_FIT"  , "\\frac{dE}{dX} vs p", 100,  0., 3., 100,0,8 );
  h_dedxFitMIP =  subDir.make<TH1F>( "dedxMIP_FIT"  , "\\frac{dE}{dX}  ",100,0,8 );

  h_dedxMassSel =  subDir.make<TH1F>( "massSel"  , "Mass (dedx), with selection", 100,  0., 1500.);
  h_dedxMassSelFit =  subDir.make<TH1F>( "massSel_FIT"  , "Mass (dedx), with selection", 100,  0., 1500.);
  h_dedxMass =  subDir.make<TH1F>( "mass"  , "Mass (dedx)", 100,  0., 1500.);
  h_dedxMassFit =  subDir.make<TH1F>( "mass_FIT"  , "Mass (dedx)", 100,  0., 1500.);
  h_dedxMassProton =  subDir.make<TH1F>( "massProton"  , "Proton Mass (dedx)", 100,  0., 2.);
  h_dedxMassProtonFit =  subDir.make<TH1F>( "massProton_FIT"  , "Proton Mass (dedx)", 100,  0., 2.);

  h_dedxMassMu =  subDir.make<TH1F>( "massMu"  , "Mass muons (dedx, 1 mu with pt>100 in the event)", 100,  0., 1500.);
  h_dedxMassMuVsPtError =  subDir.make<TH2F>( "massMu_vs_PtError"  , "Mass muons vs pt error (dedx, 1 mu with pt>100 in the event)", 100,  0., 1500.,50,0,1);
//------------ RECO TOF ----------------

  TFileDirectory subDirTof = fs->mkdir( "RecoTOF" );
  h_tofBetap =  subDirTof.make<TH2F>("tof_beta_p","1/#beta vs p",100,0,1000,100,minBeta,maxBeta );
  h_tofBetaPullp =  subDirTof.make<TH2F>("tof_beta_pull_p","1/#beta pull vs p",100,0,1000,100,-5.,5 );
  h_tofMassp =  subDirTof.make<TH2F>("tof_mass_p","Mass vs p", 100,0,1000,100,0,1000);
  h_tofMass =  subDirTof.make<TH1F>("tof_mass","Mass from DT TOF",100,0,1000);
  h_tofMass2 =  subDirTof.make<TH1F>("tof_mass2","Mass squared from DT TOF",100,-10000,100000);
  h_tofBeta =  subDirTof.make<TH1F>("tof_beta","1/#beta",100,minBeta,maxBeta);
  h_tofBetaErr =  subDirTof.make<TH1F>("tof_beta_err","#Delta 1/#beta",100,0,.5);
  h_tofBetaPull = subDirTof.make<TH1F>("tof_beta_pull","(1/#beta-1)/(#Delta 1/#beta)",100,-5.,5.);
  h_tofPtSta =  subDirTof.make<TH1F>("tof_pt_sta","StandAlone reconstructed muon p_{t}",100,0,300);
  h_tofPtComb =  subDirTof.make<TH1F>("tof_pt_comb","Global reconstructed muon p_{t}",100,50,150);

  h_tofMassCut =  subDirTof.make<TH1F>("tof_mass_cut","Mass from DT TOF (cut)",100,0,1000);
  h_tofBetaCut =  subDirTof.make<TH1F>("tof_beta_cut","1/#beta (cut)",100,minBeta,maxBeta);
  h_tofBetaPullCut = subDirTof.make<TH1F>("tof_beta_pull_cut","(1/#beta-1)/(#Delta 1/#beta)",100,-5.,5.);
  h_tofBetapCut =  subDirTof.make<TH2F>("tof_beta_p_cut","1/#beta vs p (cut)",100,0,1000,100,minBeta,maxBeta );
  h_tofBetaPullpCut =  subDirTof.make<TH2F>("tof_beta_pull_p_cut","1/#beta pull vs p (cut)",100,0,1000,100,-5.,5 );


//-------- Analysis ----------------
 TFileDirectory subDirAn =  fs->mkdir( "Analysis" );
 for(int i=0;i<6;i++) 
  {
   stringstream sel;
   sel << i;
   
   h_pSpectrumAfterSelection[i] = subDirAn.make<TH1F>((string("pSpectrumDedxSel")+sel.str()).c_str(),(string("P spectrum after selection #")+sel.str()).c_str(),300,0,1000);
   h_massAfterSelection[i] = subDirAn.make<TH1F>((string("massDedxSel")+sel.str()).c_str(),(string("Mass after selection #")+sel.str()).c_str(),300,0,1000);

  } 
  
  h_massVsMass =  subDirAn.make<TH2F>("tof_mass_vs_dedx_mass","Mass tof vs Mass dedx", 100,0,1200,100,0,1200);
  h_massVsMassSel =  subDirAn.make<TH2F>("tof_mass_vs_dedx_mass_sel","Mass tof vs Mass dedx Sel", 100,0,1200,100,0,1200);
  h_betaVsBeta =  subDirAn.make<TH2F>("tof_beta_vs_beta","INVBeta tof vs INVbeta dedx (Pt>30)", 100,0,3,100,0,3);
  h_massVsBeta =  subDirAn.make<TH2F>("avgMass_vs_avgBeta","Mass(avg) vs Beta(avg)", 100,0,1200,50,0,1);
  h_massVsPtError =  subDirAn.make<TH2F>("avgMass_vs_ptError","Mass(avg) vs log(ptError) (beta < 0.85)", 100,0,1200,25,0,10);
  h_massVsMassError =  subDirAn.make<TH2F>("avgMass_vs_MassError","Mass(avg) vs log(masstError)", 100,0,1200,100,0,2);


  h_stamu_pt =  subDirAn.make<TH1F>("STA  Muon Pt Distribution", "StandAlone Muon Pt Distribution", 1000, 0, 1000);
  h_combmu_pt=  subDirAn.make<TH1F>("Comb Muon Pt Distribution", "Combined Muon Pt Distribution", 1000, 0, 1000);
  h_tkmu_pt =  subDirAn.make<TH1F>("TK   Muon Pt Distribution", "Track Muon Pt Distribution", 1000, 0, 1000);


//--------- Analysis tk ------
 TFileDirectory subDirAnTk =  fs->mkdir( "AnalysisTk" );
  h_massVsBeta_tk =  subDirAnTk.make<TH2F>("tkMass_vs_tkBeta","Mass(tk) vs Beta(tk)", 100,0,1200,50,0,1);
  h_massVsPtError_tk =  subDirAnTk.make<TH2F>("tkMass_vs_ptError","Mass(tk) vs log(ptError) (beta < 0.85)", 100,0,1200,25,0,10);
  h_massVsMassError_tk =  subDirAnTk.make<TH2F>("tkMass_vs_MassError","Mass(tk) vs mass error", 100,0,1200,100,0,2);



//------------ SIM ----------------
  TFileDirectory subDir2 = fs->mkdir( "Sim" );
  h_simmu_pt =  subDir2.make<TH1F>( "mu_sim_pt"  , "p_{t} mu", 100,  0., 1500. );
  h_simmu_eta  =  subDir2.make<TH1F>( "mu_sim_eta"  , "\\eta mu", 50,  -4., 4. );
  h_simhscp_pt = subDir2.make<TH1F>( "mu_hscp_pt"  , "p_{t} hscp", 100,  0., 1500. );
  h_simhscp_eta =  subDir2.make<TH1F>( "mu_hscp_eta"  , "\\eta hscp" , 50,  -4., 4. );

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCPAnalyzer::endJob() {
  h_dedxMIP->Fit("gaus","Q");
  h_dedxFitMIP->Fit("gaus","Q");

  h_dedxMassProton->Fit("gaus","Q","",0.8,1.2); 
  h_dedxMassProtonFit->Fit("gaus","Q","",0.8,1.2); 


  float mipMean = h_dedxMIP->GetFunction("gaus")->GetParameter(1);
  float mipMeanFit = h_dedxFitMIP->GetFunction("gaus")->GetParameter(1);
  float mipSigma = h_dedxMIP->GetFunction("gaus")->GetParameter(2);
  float mipSigmaFit = h_dedxFitMIP->GetFunction("gaus")->GetParameter(2);

  double effPoints[6] = {0.05,0.01,0.005,0.001,0.0001,0.00001};
  float dedxFitEff[6];
  float dedxFitEffP[6];
  float dedxFitEffM[6];
  float dedxStdEff[6];
  float dedxStdEffP[6];
  float dedxStdEffM[6];
  for(int i=0;i<6;i++)
  {
   dedxFitEff[i]=cutMin(h_dedxFitMIP,effPoints[i]);
   float n=h_dedxFitMIP->GetEntries();
   float delta=sqrt(effPoints[i]*n)/n;
   dedxFitEffM[i]=cutMin(h_dedxFitMIP,effPoints[i]+delta)-dedxFitEff[i];
   dedxFitEffP[i]=cutMin(h_dedxFitMIP,effPoints[i]-delta)-dedxFitEff[i];

   dedxStdEff[i]=cutMin(h_dedxMIP,effPoints[i]);

  }
  cout << "DeDx cuts: " << endl;
  cout << "   Unbinned Fit de/dx:"<< endl; 
  cout << "    Proton Mass : " <<  h_dedxMassProtonFit->GetFunction("gaus")->GetParameter(1) << endl;
  cout << "    MIP  mean : " <<  mipMeanFit << "    sigma : " << mipSigmaFit << endl;
  for(int i=0;i<6;i++)   cout << "    " << (1-effPoints[i])*100 << "% @ dedx > " << dedxFitEff[i] << "+"<< dedxFitEffP[i] << " "<< dedxFitEffM[i] <<endl;
  cout << "   dedxSrc de/dx ("<< m_dedxSrc << "):" << endl; 
  cout << "    Proton Mass : " << h_dedxMassProton->GetFunction("gaus")->GetParameter(1) << endl;
  cout << "    MIP  mean : " <<  mipMean << "    sigma : " << mipSigma << endl;
  for(int i=0;i<6;i++)   cout << "    " << (1-effPoints[i])*100 << "% @ dedx > " << dedxStdEff[i] << endl;

  cout << endl;
  cout << "Processed events: " << tot <<  endl;
  cout << "Selected events: " << selected <<  endl;
  cout << "Selected tof  events: " << selectedTOF <<  endl;
  cout << "Selected dedx events: " << selectedDedx <<  endl;
cout << "Selection: ";
for(int i=0; i< 20 ; i++)  cout << fixed << setw(7) << setprecision(1) <<  selectedAfterCut[i] << " & " ;
cout << endl;

cout << "CutTitle: ";
  for(int i=0;i<40;i++) if(cuts[i]) { cout << " & " << fixed << setw(6) << setprecision(1) ; cuts[i]->printName(); }

cout << endl << "Cuts: ";
  for(int i=0;i<40;i++) if(cuts[i]) { cout << " & " << fixed << setw(6) << setprecision(1) ; cuts[i]->print(); }
cout << endl;
cout << endl << "CutEff: ";
  for(int i=0;i<40;i++) if(cuts[i]) { cout << " & " << fixed << setw(6) << setprecision(1) ; cuts[i]->printEff(); }
cout << endl;

}

double HSCPAnalyzer::cutMin(TH1F * h, double ci)
{
 double sum=0;

 if(h->GetEntries()>0)
 for(int i=h->GetNbinsX();i>=0; i--) 
  {
   sum+=h->GetBinContent(i); 
   if(sum/h->GetEntries()>ci)
    {
      return h->GetBinCenter(i); 
    }
  } 
 return h->GetBinCenter(0);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPAnalyzer);
