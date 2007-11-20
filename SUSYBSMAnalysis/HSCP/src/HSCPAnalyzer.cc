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
// $Id: HSCPAnalyzer.cc,v 1.13 2007/11/15 10:23:53 arizzi Exp $
//
//


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
// RECO TOF
      TH2F * h_tofBetap;
      TH2F * h_tofMassp;
      TH1F * h_tofMass;
      TH1F * h_tofMass2;
//ANALYSIS
      TH1F * h_pSpectrumAfterSelection[6]; 
      TH1F * h_massAfterSelection[6];
      TH2F * h_massVsMass;
      TH2F * h_betaVsBeta;
      TH2F * h_massVsMassSel;
//Counters
      double selected;
      double selectedTOF;
      double selectedDedx;
      double tot;
 // SIM
      TH1F * h_simmu_pt; 
      TH1F * h_simmu_eta; 
      TH1F * h_simhscp_pt; 
      TH1F * h_simhscp_eta; 

 class DTInfo
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
  };

vector<HSCParticle> associate(vector<TKInfo> & dts, vector<DTInfo> & tks );

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
   vector<DTInfo> dtInfos;
   vector<TKInfo> tkInfos;

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

   tot+=w;

   Handle<reco::MuonCollection> pIn;
   iEvent.getByLabel("muons",pIn);
   const reco::MuonCollection & muons = * pIn.product();

   Handle<vector<float> >  betaRecoH;
   iEvent.getByLabel("betaFromTOF",betaRecoH);
   const vector<float> & betaReco = *betaRecoH.product();
 
   int i=0;
   reco::MuonCollection::const_iterator muonIt = muons.begin();
   for(; muonIt != muons.end() ; ++muonIt)
    {
      TrackRef tkMuon = muonIt->track();
      TrackRef staMuon = muonIt->standAloneMuon();
      TrackRef combMuon = muonIt->combinedMuon();
      double p=0;
      if(tkMuon.isNonnull())
        {
           double pt=tkMuon->pt();
           h_pt->Fill(tkMuon->pt(),w); 
           if(combMuon.isNonnull()) pt=combMuon->pt();
           if(pt>100) highptmu=true;
            p= tkMuon->p();
     	}
     if(staMuon.isNonnull())        p= staMuon->p();
     //FIXME: tobeused as default
//     if(combMuon.isNonnull()) p=combMuon->p();

      double invbeta = betaReco.at(i);
      double mass = p*sqrt(invbeta*invbeta-1);
      double mass2 = p*p*(invbeta*invbeta-1);
           
      cout << " Muon p: " << p << " invBeta: " << invbeta << " Mass: " << mass << endl;
      
      h_tofBetap->Fill(p,invbeta,w);
      h_tofMassp->Fill(p,mass,w);
      h_tofMass->Fill(mass,w);
      h_tofMass2->Fill(mass2,w);
      tofMass.push_back(mass);
      tofP.push_back(p);
      tofValue.push_back(invbeta);
      i++;

      DTInfo dt;
      dt.trackerTrack=tkMuon;
      dt.combinedTrack=combMuon;
      dt.standaloneTrack=staMuon;
      dt.invBeta = invbeta;
      dtInfos.push_back(dt);

    }


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
      if(dedx[i].first->normalizedChi2() < 5 && dedx[i].first->numberOfValidHits()>8 && dedxHits[i].second.size() >= 10)
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
         float k=0.4;  //919/2.75*0.0012;
         float k2=0.432; //919/2.55*0.0012;
         float mass=p*sqrt(k*dedxVal-1);
         float mass2=p*sqrt(k2*dedxFitVal-1);

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

      TKInfo tk;
      tk.track=dedx[i].first;
      tk.invBeta2 = k*dedxVal;
      tk.invBeta2Fit = k2*dedxFitVal;
      tkInfos.push_back(tk);

       

        }
    }


vector<HSCParticle> candidates = associate(tkInfos,dtInfos);


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


for(int i=0; i < candidates.size();i++)
{
  h_betaVsBeta->Fill(candidates[i].dt.invBeta,sqrt(candidates[i].tk.invBeta2),w);

 // cout << candidates[i].massDt() << " " << candidates[i].massTk() << " " << candidates[i].tk.track->momentum() << " " <<  candidates[i].dt.combinedTrack->momentum() << endl;
 if((candidates[i].dt.invBeta >1.1  )|| ( candidates[i].tk.invBeta2 > 1.3 && candidates[i].hasDt ) )
 {
  h_massVsMass->Fill(candidates[i].massDt(),candidates[i].massTk(),w);
 
 if(candidates[i].massDt() + candidates[i].massTk() >  280 && fabs(sqrt(1./candidates[i].tk.invBeta2Fit)- 1./candidates[i].dt.invBeta ) < 0.1 && candidates[i].massDt() > 100 
    &&  candidates[i].massTk() > 100  && candidates[i].tk.invBeta2Fit > 1.4 &&  candidates[i].dt.invBeta  > 1.11 )
     h_massVsMassSel->Fill(candidates[i].massDt(),candidates[i].massTk(),w);

 

  if(candidates[i].massDt() + candidates[i].massTk() > 150 || (candidates[i].massTk() > 150 && candidates[i].tk.invBeta2Fit > 1.57 ) || candidates[i].massDt() > 150 )
      cout << "CANDIDATE " <<  candidates[i].massDt() << " " << candidates[i].massTk() << " " << candidates[i].tk.track->momentum() << " " <<  candidates[i].dt.combinedTrack->momentum();
      cout <<" dt beta: " << 1./candidates[i].dt.invBeta << " tk beta : "<< sqrt(1./candidates[i].tk.invBeta2) << " tk beta fit: "<< sqrt(1./candidates[i].tk.invBeta2Fit) <<
      cout <<"chi &  # hits: " <<  candidates[i].tk.track->normalizedChi2() << " " << candidates[i].tk.track->numberOfValidHits() << endl;
 }


if(candidates[i].tk.invBeta2 > 1.5 && candidates[i].hasDt && highptmu ) h_dedxMassMu->Fill(candidates[i].massTk(),w); 

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

  float minBeta = 0.8l;
  float maxBeta = 3.;

//------------ RECO DEDX ----------------
  TFileDirectory subDir = fs->mkdir( "RecoDeDx" );
  h_pt =  subDir.make<TH1F>( "mu_pt"  , "p_{t}", 100,  0., 1500. );
  h_dedx =  subDir.make<TH2F>( "dedx_p"  , "\\frac{dE}{dX} vs p", 100,  0., 1500., 100,0,8 );
  h_dedxCtrl =  subDir.make<TH2F>( "dedx_lowp"  , "\\frac{dE}{dX} vs p", 100,  0., 3., 100,0,8 );
  h_dedxMIP =  subDir.make<TH1F>( "dedxMIP"  , "\\frac{dE}{dX}  ",100,0,8 );
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
//------------ RECO TOF ----------------
  TFileDirectory subDirTof = fs->mkdir( "RecoTOF" );
  h_tofBetap =  subDirTof.make<TH2F>("tof_beta_p","1/#beta vs p",100,0,1500,100,minBeta,maxBeta );
  h_tofMassp =  subDirTof.make<TH2F>("tof_mass_p","Mass vs p", 100,0,1500,100,0,1000);
  h_tofMass =  subDirTof.make<TH1F>("tof_mass","Mass from DT TOF",100,0,1000);
  h_tofMass2 =  subDirTof.make<TH1F>("tof_mass2","Mass squared from DT TOF",100,-10000,100000);

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


vector<HSCPAnalyzer::HSCParticle> HSCPAnalyzer::associate(vector<HSCPAnalyzer::TKInfo> & tks, vector<HSCPAnalyzer::DTInfo> & dts)
{
 float minTkP=30;
 float maxTkBeta=0.9;

 float minDtP=30;
 
 float minDR=0.1;
 float maxInvPtDiff=0.005; 

 float minTkInvBeta2=1./(maxTkBeta*maxTkBeta);   
 vector<HSCPAnalyzer::HSCParticle> result; 
 for(int i=0;i<tks.size();i++)
 {
   if( tks[i].track.isNonnull() && tks[i].track->pt() > minTkP && tks[i].invBeta2 >= minTkInvBeta2 )
    {
       cout << "here " <<  tks[i].invBeta2 << endl;
       float min=1000;  
       int found = -1;
       for(int j=0;j<dts.size();j++)
        {
         if(dts[j].combinedTrack.isNonnull())
          {
          float invDT=1./dts[j].combinedTrack->pt();
          float invTK=1./tks[i].track->pt();
          if(fabs(invDT-invTK) > maxInvPtDiff) continue;
          float deltaR=ROOT::Math::VectorUtil::DeltaR(dts[j].combinedTrack->momentum(), tks[i].track->momentum());
          if(deltaR > minDR || deltaR > min) continue;
          min=deltaR;
          found = j;
          }
       }
     HSCParticle candidate; 
     candidate.tk=tks[i];
     candidate.hasDt=false;
     if(found>=0 )
       {
     candidate.hasDt=true;
        candidate.dt=dts[found];
        dts.erase(dts.begin()+found);
       }
      else
	{
//          if( tks[i].invBeta2 >= 1.30)
          cout << "Not found for " << tks[i].track->momentum() << " " << tks[i].track->eta() << endl;
	}
     result.push_back(candidate);
     
    }
 }

 for(int i=0;i<dts.size();i++)
 {
     if(dts[i].combinedTrack.isNonnull() && dts[i].combinedTrack->pt() > minDtP  )
    {
       float min=1000;
       int found = -1;
       for(int j=0;j<tks.size();j++)
        {
         if( tks[j].track.isNonnull() )
         {
          float invDT=1./dts[i].combinedTrack->pt();
          float invTK=1./tks[j].track->pt();
          if(fabs(invDT-invTK) > maxInvPtDiff) continue;
          float deltaR=ROOT::Math::VectorUtil::DeltaR(dts[i].combinedTrack->momentum(), tks[j].track->momentum());
          if(deltaR > minDR || deltaR > min) continue;
          min=deltaR;
          found = j;
          cout << "At least two muons associated to the same track ?" << endl;
         }
       }
     HSCParticle candidate;
     candidate.dt=dts[i];
     candidate.hasTk=false;
     if(found>=0 )
       {
        candidate.hasTk=true;
        candidate.tk=tks[found];
        tks.erase(tks.begin()+found);
       }
     result.push_back(candidate);
    }



 }
 return result;

} 

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPAnalyzer);
