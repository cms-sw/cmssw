namespace reco { class Vertex; class Track; class GenParticle; class DeDxData; class MuonTimeExtra; class PFMET; class HitPattern;}
namespace susybsm { class HSCParticle; class HSCPIsolation; class MuonSegment; class HSCPDeDxInfo;}
namespace fwlite { class ChainEvent;}
namespace trigger { class TriggerEvent;}
namespace edm { class TriggerResults; class TriggerResultsByName; class InputTag; class LumiReWeighting;}
namespace reweight{ class PoissonMeanShifter;}

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/Common/interface/MergeableCounter.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPIsolation.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPDeDxInfo.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/MuonSegment.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "PhysicsTools/Utilities/interface/LumiReWeighting.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

using namespace fwlite;
using namespace reco;
using namespace susybsm;
using namespace std;
using namespace edm;
using namespace trigger;
using namespace reweight;

#endif


//the define here is simply need to load FWLITE code from the include
#define FWLITE
#include "../../ICHEP_Analysis/Analysis_Global.h"
#include "../../ICHEP_Analysis/Analysis_CommonFunction.h"
#include "../../ICHEP_Analysis/Analysis_PlotFunction.h"
#include "../../ICHEP_Analysis/Analysis_PlotStructure.h"
#include "../../ICHEP_Analysis/Analysis_Samples.h"
#include "../../ICHEP_Analysis/tdrstyle.C"

std::vector< float > BgLumiMC; //MC                                                                                                                                                  
std::vector< float > TrueDist;
edm::LumiReWeighting LumiWeightsMC;
reweight::PoissonMeanShifter PShift(0.6);

std::vector<stSample> samples;
vector<std::pair<string,string>> All_triggers;
vector<std::pair<string,string>> AllSA_triggers;

std::string OutputDirectory="pictures/";

class stPlot{
   public:
   TH1D* Histo;
   TH1D* HistoInc;
   TH1D* HistoMatched;
   TH1D* HistoIncMatched;
   TH1D* HistoMatchedSA;
   TH1D* HistoIncMatchedSA;
   TH1D* HistoMatchedGl;
   TH1D* HistoIncMatchedGl;
   TH1D* BetaCount;
   TH1D* BetaTotal;
   TH1D* BetaMuon;
   TH1D* BetaJet;
   TH1D* BetaCountMatched;
   TH1D* BetaTotalMatched;
   TH1D* BetaMuonMatched;
   TH1D* BetaJetMatched;
   TH1D* BetaCountMatchedSA;
   TH1D* BetaTotalMatchedSA;
   TH1D* BetaMuonMatchedSA;
   TH1D* BetaJetMatchedSA;
   TH1D* BetaCountMatchedGl;
   TH1D* BetaTotalMatchedGl;
   TH1D* BetaMuonMatchedGl;
   TH1D* BetaJetMatchedGl;

   stPlot(string SignalName){
      int numberofbins=All_triggers.size()+1;
      Histo             = new TH1D((SignalName + "Abs").c_str(),(SignalName + "Abs").c_str(),numberofbins,0,numberofbins);
      HistoInc          = new TH1D((SignalName + "Inc").c_str(),(SignalName + "Inc").c_str(),numberofbins,0,numberofbins);
      HistoMatched      = new TH1D((SignalName + "AbsMatched").c_str(),(SignalName + "AbsMatched").c_str(),numberofbins,0,numberofbins);
      HistoIncMatched   = new TH1D((SignalName + "IncMatched").c_str(),(SignalName + "IncMatched").c_str(),numberofbins,0,numberofbins);
      HistoMatchedSA    = new TH1D((SignalName + "AbsMatchedSA").c_str(),(SignalName + "AbsMatched").c_str(),numberofbins,0,numberofbins);
      HistoIncMatchedSA = new TH1D((SignalName + "IncMatchedSA").c_str(),(SignalName + "IncMatched").c_str(),numberofbins,0,numberofbins);
      HistoMatchedGl    = new TH1D((SignalName + "AbsMatchedGl").c_str(),(SignalName + "AbsMatched").c_str(),numberofbins,0,numberofbins);
      HistoIncMatchedGl = new TH1D((SignalName + "IncMatchedGl").c_str(),(SignalName + "IncMatched").c_str(),numberofbins,0,numberofbins);

      for(unsigned int i=0;i<All_triggers.size();i++)    { Histo->GetXaxis()->SetBinLabel(i+1,All_triggers[i].second.c_str());   }
      Histo->GetXaxis()->SetBinLabel(numberofbins,"Total");

      for(unsigned int i=0;i<All_triggers.size();i++)    { HistoInc->GetXaxis()->SetBinLabel(i+1,All_triggers[i].second.c_str());   }
      HistoInc->GetXaxis()->SetBinLabel(numberofbins,"Total");

      for(unsigned int i=0;i<All_triggers.size();i++) { HistoMatched->GetXaxis()->SetBinLabel(i+1,All_triggers[i].second.c_str()); }
      HistoMatched->GetXaxis()->SetBinLabel(numberofbins,"Total");

      for(unsigned int i=0;i<All_triggers.size();i++) { HistoIncMatched->GetXaxis()->SetBinLabel(i+1,All_triggers[i].second.c_str()); }
      HistoIncMatched->GetXaxis()->SetBinLabel(numberofbins,"Total");

      for(unsigned int i=0;i<All_triggers.size();i++) { HistoMatchedSA->GetXaxis()->SetBinLabel(i+1,AllSA_triggers[i].second.c_str()); }
      HistoMatchedSA->GetXaxis()->SetBinLabel(numberofbins,"Total");

      for(unsigned int i=0;i<All_triggers.size();i++) { HistoIncMatchedSA->GetXaxis()->SetBinLabel(i+1,AllSA_triggers[i].second.c_str()); }
      HistoIncMatchedSA->GetXaxis()->SetBinLabel(numberofbins,"Total");

      for(unsigned int i=0;i<All_triggers.size();i++) { HistoMatchedGl->GetXaxis()->SetBinLabel(i+1,All_triggers[i].second.c_str()); }
      HistoMatchedGl->GetXaxis()->SetBinLabel(numberofbins,"Total");

      for(unsigned int i=0;i<All_triggers.size();i++) { HistoIncMatchedGl->GetXaxis()->SetBinLabel(i+1,All_triggers[i].second.c_str()); }
      HistoIncMatchedGl->GetXaxis()->SetBinLabel(numberofbins,"Total");

      Histo->Sumw2();
      HistoInc->Sumw2();
      HistoMatched->Sumw2();
      HistoIncMatched->Sumw2();
      HistoMatchedSA->Sumw2();
      HistoIncMatchedSA->Sumw2();
      HistoMatchedGl->Sumw2();
      HistoIncMatchedGl->Sumw2();

      BetaCount    = new TH1D((SignalName + "BetaMuCount").c_str() ,(SignalName + "BetaCount").c_str()   ,20,0,1);   BetaCount  ->Sumw2();
      BetaTotal    = new TH1D((SignalName + "BetaTotal"  ).c_str() ,(SignalName + "BetaTotal").c_str()   ,20,0,1);   BetaTotal  ->Sumw2();
      BetaMuon     = new TH1D((SignalName + "BetaMuon"   ).c_str() ,(SignalName + "BetaMuon" ).c_str()   ,20,0,1);   BetaMuon   ->Sumw2();
      BetaJet      = new TH1D((SignalName + "BetaJet"    ).c_str() ,(SignalName + "BetaJet"  ).c_str()   ,20,0,1);   BetaJet    ->Sumw2();

      BetaCountMatched    = new TH1D((SignalName + "BetaMuCountMatched").c_str() ,(SignalName + "BetaCountMatched").c_str()   ,20,0,1);   BetaCountMatched  ->Sumw2();
      BetaTotalMatched    = new TH1D((SignalName + "BetaTotalMatched"  ).c_str() ,(SignalName + "BetaTotalMatched").c_str()   ,20,0,1);   BetaTotalMatched  ->Sumw2();
      BetaMuonMatched     = new TH1D((SignalName + "BetaMuonMatched"   ).c_str() ,(SignalName + "BetaMuonMatched" ).c_str()   ,20,0,1);   BetaMuonMatched   ->Sumw2();
      BetaJetMatched      = new TH1D((SignalName + "BetaJetMatched"    ).c_str() ,(SignalName + "BetaJetMatched"  ).c_str()   ,20,0,1);   BetaJetMatched    ->Sumw2();

      BetaCountMatchedSA    = new TH1D((SignalName + "BetaMuCountMatchedSA").c_str() ,(SignalName + "BetaCountMatchedSA").c_str()   ,20,0,1);   BetaCountMatchedSA  ->Sumw2();
      BetaTotalMatchedSA    = new TH1D((SignalName + "BetaTotalMatchedSA"  ).c_str() ,(SignalName + "BetaTotalMatchedSA").c_str()   ,20,0,1);   BetaTotalMatchedSA  ->Sumw2();
      BetaMuonMatchedSA     = new TH1D((SignalName + "BetaMuonMatchedSA"   ).c_str() ,(SignalName + "BetaMuonMatchedSA" ).c_str()   ,20,0,1);   BetaMuonMatchedSA   ->Sumw2();
      BetaJetMatchedSA      = new TH1D((SignalName + "BetaJetMatchedSA"    ).c_str() ,(SignalName + "BetaJetMatchedSA"  ).c_str()   ,20,0,1);   BetaJetMatchedSA    ->Sumw2();

      BetaCountMatchedGl    = new TH1D((SignalName + "BetaMuCountMatchedGl").c_str() ,(SignalName + "BetaCountMatchedGl").c_str()   ,20,0,1);   BetaCountMatchedGl  ->Sumw2();
      BetaTotalMatchedGl    = new TH1D((SignalName + "BetaTotalMatchedGl"  ).c_str() ,(SignalName + "BetaTotalMatchedGl").c_str()   ,20,0,1);   BetaTotalMatchedGl  ->Sumw2();
      BetaMuonMatchedGl     = new TH1D((SignalName + "BetaMuonMatchedGl"   ).c_str() ,(SignalName + "BetaMuonMatchedGl" ).c_str()   ,20,0,1);   BetaMuonMatchedGl   ->Sumw2();
      BetaJetMatchedGl      = new TH1D((SignalName + "BetaJetMatchedGl"    ).c_str() ,(SignalName + "BetaJetMatchedGl"  ).c_str()   ,20,0,1);   BetaJetMatchedGl    ->Sumw2();
   }

};


void TriggerStudy_Core(string SignalName, FILE* pFile, stPlot* plot);
double FastestHSCP(const fwlite::ChainEvent& ev);
//bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, double etaCut,int NObjectAboveThreshold, bool averageThreshold=false);
void layout(stPlot** plots, vector<string>& sigs, string name);

void TriggerStudy(string Name="COMPILE", string Sample1="", string Sample2="", string Sample3="", string Sample4="", string Sample5="", string Sample6="")
{
  if(Name=="COMPILE") return;

   system("mkdir pictures");
   std::vector<string> SamplesToRun;
   if(Sample1!="") SamplesToRun.push_back(Sample1);
   if(Sample2!="") SamplesToRun.push_back(Sample2);
   if(Sample3!="") SamplesToRun.push_back(Sample3);
   if(Sample4!="") SamplesToRun.push_back(Sample4);
   if(Sample5!="") SamplesToRun.push_back(Sample5);
   if(Sample6!="") SamplesToRun.push_back(Sample6);

   gStyle->SetCanvasBorderMode(0);
   gStyle->SetCanvasColor(kWhite);
   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.14);
   gStyle->SetPadRightMargin (0.16);
   gStyle->SetPadLeftMargin  (0.14);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.45);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505);

   //initialize LumiReWeighting
  BgLumiMC.clear();
  TrueDist.clear();
#ifdef ANALYSIS2011
   if(Name.find("7TeV")==string::npos){printf("Skim %s because of wrong center of mass energy\n", Name.c_str());return;}

   for(int i=0; i<35; ++i) BgLumiMC.push_back(Pileup_MC_Fall11[i]);
   for(int i=0; i<35; ++i) TrueDist.push_back(TrueDist2011_f[i]);
   SQRTS=7;
#else
   if(Name.find("8TeV")==string::npos){printf("Skim %s because of wrong center of mass energy\n", Name.c_str());return;}

   for(int i=0; i<60; ++i) BgLumiMC.push_back(Pileup_MC_Summer2012[i]);
   for(int i=0; i<60; ++i) TrueDist.push_back(TrueDist2012_f[i]);
   SQRTS=8;
#endif
   LumiWeightsMC = edm::LumiReWeighting(BgLumiMC, TrueDist);

   InitBaseDirectory();
   GetSampleDefinition(samples, "../../ICHEP_Analysis/Analysis_Samples.txt");

   keepOnlySamplesOfNamesXtoY(samples, SamplesToRun);

   ///////////////////////////////////////////////////////
   All_triggers.clear();
   All_triggers.push_back(std::make_pair("HSCPHLTTriggerMuFilter", "Mu40_eta2p1"));
   All_triggers.push_back(std::make_pair("HSCPHLTTriggerPFMetFilter","PFMET150"));
#ifndef ANALYSIS2011
   All_triggers.push_back(std::make_pair("HSCPHLTTriggerL2MuFilter", "L2Muon+Met"));
   All_triggers.push_back(std::make_pair("HSCPHLTTriggerMetDeDxFilter", "Met80+dEdx"));
   All_triggers.push_back(std::make_pair("HSCPHLTTriggerMuDeDxFilter", "Mu40+dEdx"));
   All_triggers.push_back(std::make_pair("HSCPHLTTriggerHtDeDxFilter", "HT+dEdx"));
#endif
   All_triggers.push_back(std::make_pair("HSCPHLTTriggerHtFilter", "HT650" ) );

   AllSA_triggers.clear();
   AllSA_triggers.push_back(std::make_pair("HSCPHLTTriggerPFMetFilter","PFMET150"));
   AllSA_triggers.push_back(std::make_pair("HSCPHLTTriggerMuFilter", "Mu40_eta2p1"));
#ifndef ANALYSIS2011
   AllSA_triggers.push_back(std::make_pair("HSCPHLTTriggerL2MuFilter", "L2Muon+Met"));
   AllSA_triggers.push_back(std::make_pair("HSCPHLTTriggerMetDeDxFilter", "Met80+dEdx"));
   AllSA_triggers.push_back(std::make_pair("HSCPHLTTriggerMuDeDxFilter", "Mu40+dEdx"));
   AllSA_triggers.push_back(std::make_pair("HSCPHLTTriggerHtDeDxFilter", "HT+dEdx"));
#endif
   AllSA_triggers.push_back(std::make_pair("HSCPHLTTriggerHtFilter", "HT650" ) );
   ///////////////////////////////////////////////////////

   FILE* pFile = fopen((OutputDirectory + "Results_" + Name + ".txt").c_str(),"w");

   vector<string> leg;

   stPlot** plots = new stPlot*[samples.size()];  
   for(unsigned int i=0;i<samples.size();i++){
     if(samples[i].Type!=2)continue;
     leg.push_back(samples[i].Legend);
     plots[i] = new stPlot(samples[i].Name);
     TriggerStudy_Core(samples[i].Name, pFile, plots[i]);
   }
   fflush(pFile);
   fclose(pFile);

   if(samples.size()!=0) layout(plots, leg, Name);

/*

//final systmatic calculation
   FILE* pFile1 = fopen("Results.txt","r");

   vector<float> vec_eff_met;
   vector<float> vec_eff_metm;
   vector<float> vec_eff_mu;
   vector<float> vec_eff_tot;
   vector<float> vec_eff_totm;

   for(unsigned int i=0;i<samples.size();i++){
      char str[80];
      float eff_met, eff_mu,eff_tot;
      float meff_met, meff_mu,meff_tot;
      fscanf(pFile1,  "%15s --> MET = %f%% (modified %f%%) Mu = %f%% (modified %f%%) JetMET||Mu = %f%% (%f%%)\n",str, &eff_met, &meff_met, &eff_mu, &meff_mu, &eff_tot, &meff_tot);
      vec_eff_met.push_back(eff_met);
      vec_eff_metm.push_back(meff_met);
      vec_eff_mu.push_back(eff_mu);
      vec_eff_tot.push_back(eff_tot);
      vec_eff_totm.push_back(meff_tot);
   }
   fclose(pFile1);

   FILE* pFile2 = fopen("Results_Systematic.txt","w");
   for(unsigned int i=0;i<samples.size();i=i+2){
      fprintf(pFile2,  "%15s --> Sys_MET_tof = %5.2f%% Sys_MET_jes = %5.2f%% Sys_Mu = %5.2f%% Sys_tot_tof = %5.2f%% Sys_tot_jes = %5.2f%% \n",samples[i].Name.c_str(), fabs(vec_eff_met[i+1]-vec_eff_met[i])/vec_eff_met[i]*100,fabs(vec_eff_metm[i]-vec_eff_met[i])/vec_eff_met[i]*100,fabs(vec_eff_mu[i+1]-vec_eff_mu[i])/vec_eff_mu[i]*100, fabs(vec_eff_tot[i+1]-vec_eff_tot[i])/vec_eff_tot[i]*100,fabs(vec_eff_totm[i]-vec_eff_tot[i])/vec_eff_tot[i]*100);
      fprintf(stdout, "%15s --> Sys_MET_tof = %5.2f%% Sys_MET_jes = %5.2f%% Sys_Mu = %5.2f%% Sys_tot_tof = %5.2f%% Sys_tot_jes = %5.2f%% \n",samples[i].Name.c_str(),fabs(vec_eff_met[i+1]-vec_eff_met[i])/vec_eff_met[i]*100, fabs(vec_eff_metm[i]-vec_eff_met[i])/vec_eff_met[i]*100,fabs(vec_eff_mu[i+1]-vec_eff_mu[i])/vec_eff_mu[i]*100, fabs(vec_eff_tot[i+1]-vec_eff_tot[i])/vec_eff_tot[i]*100,fabs(vec_eff_totm[i]-vec_eff_tot[i])/vec_eff_tot[i]*100);
   }
   fclose(pFile2);

*/

}

void TriggerStudy_Core(string SignalName, FILE* pFile, stPlot* plot)
{

  for (int period=0; period<2; period++) {
    int JobId = JobIdToIndex(SignalName, samples);

    vector<string> fileNames;
    GetInputFiles(samples[JobId], BaseDirectory, fileNames, period);

    string thisname = fileNames[0];
   //bool simhitshifted =0;
   //if(thisname.find("S.",0)<std::string::npos ||thisname.find("SBX1.",0)<std::string::npos) simhitshifted=1 ;

//fileNames.clear();
//      fileNames.push_back("/uscmst1b_scratch/lpc1/lpcphys/jchen/2011Runanalysis/aftereps/cls/CMSSW_4_2_8/src/SUSYBSMAnalysis/HSCP/test/BuildHSCParticles/ShiftSignals/HSCP.root");

    fwlite::ChainEvent ev(fileNames);
    double SampleWeight = 1.0;
    double PUSystFactor;
    int MaxEvent = 10000;

   //get PU reweighted total # MC events.
   double NMCevents=0;
   for(Long64_t ientry=0;ientry<ev.size();ientry++) {
     ev.to(ientry);
     if(MaxEvent>0 && ientry>MaxEvent)break;
     NMCevents += GetPUWeight(ev, samples[JobId].Pileup, PUSystFactor, LumiWeightsMC, PShift);
   }

   SampleWeight = GetSampleWeight  (IntegratedLuminosity,IntegratedLuminosityBeforeTriggerChange,samples[JobId].XSec,NMCevents, period);

   if(SampleWeight==0) continue; //If sample weight 0 don't run, happens Int Lumi before change = 0

   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on %10s        :", SignalName.c_str());
   int TreeStep = ev.size()/50;if(TreeStep==0)TreeStep=1;
   if(MaxEvent<0 || MaxEvent>ev.size())MaxEvent = ev.size();
   for(Long64_t e=0;e<MaxEvent;e++){
      if(e%TreeStep==0){printf(".");fflush(stdout);}
      ev.to(e);

      double Event_Weight = SampleWeight * GetPUWeight(ev, samples[JobId].Pileup, PUSystFactor, LumiWeightsMC, PShift);

      edm::TriggerResultsByName tr = ev.triggerResultsByName("MergeHLT"); 
//      if(simhitshifted) tr= ev.triggerResultsByName("HLTSIMHITSHIFTER");
//      edm::TriggerResultsByName tr = ev.triggerResultsByName("HLT");      if(!tr.isValid())continue;
      //    for(unsigned int i=0;i<tr.size();i++){
      //   printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
      //}fflush(stdout);

      fwlite::Handle< trigger::TriggerEvent > trEvHandle;
      trEvHandle.getByLabel(ev,"hltTriggerSummaryAOD");
      trigger::TriggerEvent trEv = *trEvHandle;

      fwlite::Handle<susybsm::HSCParticleCollection> hscpCollHandle;
      hscpCollHandle.getByLabel(ev,"HSCParticleProducer");
      if(!hscpCollHandle.isValid()){continue;}
      susybsm::HSCParticleCollection hscpColl = *hscpCollHandle;

      //get the collection of generated Particles
      fwlite::Handle< std::vector<reco::GenParticle> > genCollHandle;
      genCollHandle.getByLabel(ev, "genParticles");
      if(!genCollHandle.isValid()){printf("GenParticle Collection NotFound\n");continue;}
      std::vector<reco::GenParticle> genColl = *genCollHandle;

      int NChargedHSCP=HowManyChargedHSCP(genColl);
      Event_Weight*=samples[JobId].GetFGluinoWeight(NChargedHSCP);

      //Match gen HSCP to reco tracks
      bool match=false; double betaMatched=-9999.; 
      bool matchSA=false; double betaMatchedSA=-9999.;
      bool matchGl=false; double betaMatchedGl=-9999.;

	for(unsigned int g=0;g<genColl.size();g++){
	  if(genColl[g].pt()<5)continue;
	  if(genColl[g].status()!=1)continue;
	  int AbsPdg=abs(genColl[g].pdgId());
	  if(AbsPdg<1000000 && AbsPdg!=17)continue;

	  double RMin=9999;
	  double RMinSA = 9999;
          double RMinGl = 9999;
	  for(unsigned int c=0;c<hscpColl.size();c++){
	  //define alias for important variable
	    susybsm::HSCParticle hscp  = hscpColl[c];
	    reco::MuonRef  muon  = hscp.muonRef();

	  //For TOF only analysis use updated stand alone muon track.
	  //Otherwise use inner tracker track
	    reco::TrackRef track;
	    track = hscp.trackRef();

	    reco::TrackRef trackSA;
	    if(trackSA.isNull()) {
	      if(muon.isNull()) continue;
	      trackSA = muon->standAloneMuon();
	    }
	    //skip events without track
	    if(!track.isNull()) {
	      double dR = deltaR(track->eta(), track->phi(), genColl[g].eta(), genColl[g].phi());
	      if(dR<RMin)RMin=dR;
	    }

	    if(!trackSA.isNull()){
	      double dR = deltaR(trackSA->eta(), trackSA->phi(), genColl[g].eta(), genColl[g].phi());
	      if(dR<RMinSA)RMinSA=dR;
	    }
	    if(!muon.isNull() && !track.isNull() && muon->isGlobalMuon()){
              double dR = deltaR(track->eta(), track->phi(), genColl[g].eta(), genColl[g].phi());
              if(dR<RMinGl)RMinGl=dR;
            }
	  }
	  if(RMin<0.3) {
	    match=true;
	    if(genColl[g].p()/genColl[g].energy()>betaMatched) betaMatched=genColl[g].p()/genColl[g].energy();
	  }

	  if(RMinSA<0.3) {
	    matchSA=true;
	    if(genColl[g].p()/genColl[g].energy()>betaMatchedSA) betaMatchedSA=genColl[g].p()/genColl[g].energy();
	  }
          if(RMinGl<0.3) {
            matchGl=true;
            if(genColl[g].p()/genColl[g].energy()>betaMatchedGl) betaMatchedGl=genColl[g].p()/genColl[g].energy();
          }
	}
         
	unsigned int TrIndex_Unknown     = tr.size();

	bool AlreadyAccepted = false;
	bool AcceptMu = false;
	bool AcceptMET = false;
        bool AcceptMuMET = false;

	for(unsigned int i=0;i<All_triggers.size();i++){
	  if(TrIndex_Unknown==tr.triggerIndex(All_triggers[i].first))  {cout << "Trigger " << All_triggers[i].first << " not found" << endl; continue;}
	  bool Accept = tr.accept(All_triggers[i].first.c_str());

/*           if(All_triggers[i]=="HLT_PFMHT150_v2"){
               if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v2")){
                   if(e<MaxPrint)printf("HLT_PFMHT150_v2\n");
                   Accept = tr.accept(tr.triggerIndex("HLT_PFMHT150_v2"));
                }else{
                   if(e<MaxPrint)printf("HLT_PFMHT150_v1\n");
                   Accept = tr.accept(tr.triggerIndex("HLT_PFMHT150_v1"));
                }
               //Accept2 = Accept;
               if(simhitshifted) Accept2 = IncreasedTreshold(trEv, InputTag("hltPFMHT150Filter","","HLTSIMHITSHIFTER"),160 , 100, 1, false);
               else Accept2 = IncreasedTreshold(trEv, InputTag("hltPFMHT150Filter","","HLT"),160 , 100, 1, false);
            }
           else if(All_triggers[i]=="HLT_Mu40_eta2p1_v1"){

              if(simhitshifted) Accept = IncreasedTreshold(trEv, InputTag("hltSingleMu30L3Filtered30","","HLTSIMHITSHIFTER"),40 , 2.1, 1, false);
              else  Accept = IncreasedTreshold(trEv, InputTag("hltSingleMu30L3Filtered30","","HLT"),40 , 2.1, 1, false);              
              Accept2 = Accept;
           }
*/

	  if(Accept){
	    plot->Histo          ->Fill(All_triggers[i].second.c_str(),Event_Weight);
	    if(!AlreadyAccepted) plot->HistoInc       ->Fill(All_triggers[i].second.c_str(),Event_Weight);
	  }
         if(Accept && match){
	   plot->HistoMatched   ->Fill(All_triggers[i].second.c_str(),Event_Weight);
	   if(!AlreadyAccepted) plot->HistoIncMatched->Fill(All_triggers[i].second.c_str(),Event_Weight);
	 }
	 if(Accept && matchGl){
	   plot->HistoMatchedGl  ->Fill(All_triggers[i].second.c_str(),Event_Weight);
	   if(!AlreadyAccepted) plot->HistoIncMatchedGl->Fill(All_triggers[i].second.c_str(),Event_Weight);
	 }
         AlreadyAccepted |= Accept;

	 if(Accept && All_triggers[i].first.find("HSCPHLTTriggerMuFilter")!=string::npos) AcceptMu=true;
         if(Accept && All_triggers[i].first.find("PFMet")!=string::npos) AcceptMET=true;
	 if(Accept && All_triggers[i].first.find("HSCPHLTTriggerL2MuFilter")!=string::npos) AcceptMuMET=true;
	}

      bool AlreadyAcceptedSA = false;
      for(unsigned int i=0;i<AllSA_triggers.size();i++){
	bool Accept = tr.accept(AllSA_triggers[i].first.c_str());

	if(Accept && matchSA){
	  plot->HistoMatchedSA  ->Fill(AllSA_triggers[i].second.c_str(),Event_Weight);
	  if(!AlreadyAcceptedSA) plot->HistoIncMatchedSA->Fill(AllSA_triggers[i].second.c_str(),Event_Weight);
	}
	AlreadyAcceptedSA |= Accept;
      }
   
      fflush(stdout);
   

      if(AlreadyAccepted){
         plot->Histo->Fill("Total",Event_Weight);
         plot->HistoInc->Fill("Total",Event_Weight);
	 if(match) {
	   plot->HistoMatched->Fill("Total",Event_Weight);
	   plot->HistoIncMatched->Fill("Total",Event_Weight);
	 }
	 if(matchGl) {
	   plot->HistoMatchedGl->Fill("Total",Event_Weight);
	   plot->HistoIncMatchedGl->Fill("Total",Event_Weight);
	 }
      }
      if(AlreadyAcceptedSA){
         if(matchSA) {
           plot->HistoMatchedSA->Fill("Total",Event_Weight);
           plot->HistoIncMatchedSA->Fill("Total",Event_Weight);
         }
      }

      double Beta = 1.0;
      if(SignalName!="Data")Beta = FastestHSCP(ev);

      plot->BetaCount->Fill(Beta,Event_Weight);
      if(AcceptMu || AcceptMET)plot->BetaTotal->Fill(Beta,Event_Weight);
      if(AcceptMu)plot->BetaMuon->Fill(Beta,Event_Weight);
      if(AcceptMET)plot->BetaJet->Fill(Beta,Event_Weight);

      if(match) {
	plot->BetaCountMatched->Fill(betaMatched,Event_Weight);
	if(AcceptMu || AcceptMET){
	  plot->BetaTotalMatched->Fill(betaMatched,Event_Weight);
	}
	if(AcceptMu)plot->BetaMuonMatched->Fill(betaMatched,Event_Weight);
	if(AcceptMET)plot->BetaJetMatched->Fill(betaMatched,Event_Weight);
      }
      
      if(matchSA) {
	plot->BetaCountMatchedSA->Fill(betaMatchedSA,Event_Weight);
        if(AcceptMuMET || AcceptMET || AcceptMu)plot->BetaTotalMatchedSA->Fill(betaMatchedSA,Event_Weight);
        if(AcceptMu || AcceptMuMET)plot->BetaMuonMatchedSA->Fill(betaMatchedSA,Event_Weight);
        if(AcceptMET)plot->BetaJetMatchedSA->Fill(betaMatchedSA,Event_Weight);
      }
      if(matchGl) {
        plot->BetaCountMatchedGl->Fill(betaMatchedGl,Event_Weight);
        if(AcceptMu || AcceptMET)plot->BetaTotalMatchedGl->Fill(betaMatchedGl,Event_Weight);
        if(AcceptMu)plot->BetaMuonMatchedGl->Fill(betaMatchedGl,Event_Weight);
        if(AcceptMET)plot->BetaJetMatchedGl->Fill(betaMatchedGl,Event_Weight);
      }

   }printf("\n");
  }

//   printf("Total %i \n",Total);
   plot->Histo->SetStats(0)  ;
   plot->Histo->LabelsOption("v");
   plot->Histo->Scale(100./plot->BetaCount->Integral());


   plot->HistoInc->SetStats(0)  ;
   plot->HistoInc->LabelsOption("v");
   plot->HistoInc->Scale(100./plot->BetaCount->Integral());


   plot->BetaTotal->Divide(plot->BetaCount);
  plot->BetaMuon ->Divide(plot->BetaCount);
   plot->BetaJet  ->Divide(plot->BetaCount);

   plot->BetaTotal->Scale(100.0);
   plot->BetaMuon ->Scale(100.0);
   plot->BetaJet  ->Scale(100.0);

   plot->HistoMatched->SetStats(0)  ;
   plot->HistoMatched->LabelsOption("v");
   plot->HistoMatched->Scale(100./plot->BetaCountMatched->Integral());


   plot->HistoIncMatched->SetStats(0)  ;
   plot->HistoIncMatched->LabelsOption("v");
   plot->HistoIncMatched->Scale(100./plot->BetaCountMatched->Integral());


   plot->BetaTotalMatched->Divide(plot->BetaCountMatched);
   plot->BetaMuonMatched ->Divide(plot->BetaCountMatched);
   plot->BetaJetMatched  ->Divide(plot->BetaCountMatched);

   plot->BetaTotalMatched->Scale(100.0);
   plot->BetaMuonMatched ->Scale(100.0);
   plot->BetaJetMatched  ->Scale(100.0);

   plot->HistoMatchedSA->SetStats(0)  ;
   plot->HistoMatchedSA->LabelsOption("v");
   plot->HistoMatchedSA->Scale(100./plot->BetaCountMatchedSA->Integral());


   plot->HistoIncMatchedSA->SetStats(0)  ;
   plot->HistoIncMatchedSA->LabelsOption("v");
   plot->HistoIncMatchedSA->Scale(100./plot->BetaCountMatchedSA->Integral());

   plot->BetaTotalMatchedSA->Divide(plot->BetaCountMatchedSA);
   plot->BetaMuonMatchedSA ->Divide(plot->BetaCountMatchedSA);
   plot->BetaJetMatchedSA  ->Divide(plot->BetaCountMatchedSA);

   plot->BetaTotalMatchedSA->Scale(100.0);
   plot->BetaMuonMatchedSA ->Scale(100.0);
   plot->BetaJetMatchedSA  ->Scale(100.0);


   plot->HistoMatchedGl->SetStats(0)  ;
   plot->HistoMatchedGl->LabelsOption("v");
   plot->HistoMatchedGl->Scale(100./plot->BetaCountMatchedGl->Integral());

   plot->HistoIncMatchedGl->SetStats(0)  ;
   plot->HistoIncMatchedGl->LabelsOption("v");
   plot->HistoIncMatchedGl->Scale(100./plot->BetaCountMatched->Integral());

   plot->BetaTotalMatchedGl->Divide(plot->BetaCountMatchedGl);
   plot->BetaMuonMatchedGl ->Divide(plot->BetaCountMatchedGl);
   plot->BetaJetMatchedGl  ->Divide(plot->BetaCountMatchedGl);

   plot->BetaTotalMatchedGl->Scale(100.0);
   plot->BetaMuonMatchedGl ->Scale(100.0);
   plot->BetaJetMatchedGl  ->Scale(100.0);

   TH1** Histos = new TH1*[10];
   std::vector<string> legend;
   TCanvas* c1;
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->BetaMuon;                    legend.push_back("Mu triggers");
   Histos[1] = (TH1*)plot->BetaTotal;                   legend.push_back("Mu+Met triggers");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "#beta of the fastest HSCP", "Trigger Efficiency (%)", 0,1, 0,100);
   DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary("Simulation", SQRTS, -1);
   SaveCanvas(c1,OutputDirectory,SignalName);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->BetaMuonMatched;                    legend.push_back("Mu triggers");
   Histos[1] = (TH1*)plot->BetaTotalMatched;                   legend.push_back("Mu+Met triggers");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "#beta of the fastest HSCP", "Trigger Efficiency (%)", 0,1, 0,100);
   DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary("Simulation", SQRTS, -1);
   SaveCanvas(c1,OutputDirectory,SignalName + "Matched");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->BetaMuonMatchedSA;                    legend.push_back("Mu triggers");
   Histos[1] = (TH1*)plot->BetaTotalMatchedSA;                   legend.push_back("Mu+Met triggers");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "#beta of the fastest HSCP", "Trigger Efficiency (%)", 0,1, 0,100);
   DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary("Simulation", SQRTS, -1);
   SaveCanvas(c1,OutputDirectory,SignalName + "MatchedSA");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->BetaMuonMatchedGl;                    legend.push_back("Mu triggers");
   Histos[1] = (TH1*)plot->BetaTotalMatchedGl;                   legend.push_back("Mu+Met triggers");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "#beta of the fastest HSCP", "Trigger Efficiency (%)", 0,1, 0,100);
   DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary("Simulation", SQRTS, -1);
   SaveCanvas(c1,OutputDirectory,SignalName + "MatchedGl");
   delete c1;

  fprintf(pFile,  "Trigger efficiency for %15s\n",SignalName.c_str());
   for(unsigned int i=0; i<All_triggers.size(); i++) {
     fprintf(pFile,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",All_triggers[i].first.c_str(), plot->Histo->GetBinContent(i+1), plot->HistoInc->GetBinContent(i+1), plot->HistoInc->Integral(1, i+1));
   }
   fprintf(pFile,  "\nIn events with reconstructed track\n");

   for(unsigned int i=0; i<All_triggers.size(); i++) {
     fprintf(pFile,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",All_triggers[i].first.c_str(), plot->HistoMatched->GetBinContent(i+1), plot->HistoIncMatched->GetBinContent(i+1), plot->HistoIncMatched->Integral(1, i+1));
   }
   fprintf(pFile,  "\nIn events with reconstructed stand alone muon\n");
   for(unsigned int i=0; i<AllSA_triggers.size(); i++) {
     fprintf(pFile,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",AllSA_triggers[i].first.c_str(), plot->HistoMatchedSA->GetBinContent(i+1), plot->HistoIncMatchedSA->GetBinContent(i+1), plot->HistoIncMatchedSA->Integral(1, i+1));
   }

   fprintf(pFile,  "\nIn events with global muon\n");
   for(unsigned int i=0; i<All_triggers.size(); i++) {
     fprintf(pFile,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",All_triggers[i].first.c_str(), plot->HistoMatchedGl->GetBinContent(i+1), plot->HistoIncMatchedGl->GetBinContent(i+1), plot->HistoIncMatchedGl->Integral(1, i+1));
   }

  fprintf(stdout,  "Trigger efficiency for %15s\n",SignalName.c_str());
   for(unsigned int i=0; i<All_triggers.size(); i++) {
     fprintf(stdout,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",All_triggers[i].first.c_str(), plot->Histo->GetBinContent(i+1), plot->HistoInc->GetBinContent(i+1), plot->HistoInc->Integral(1, i+1));
   }
   fprintf(stdout,  "\nIn events with reconstructed track\n");
   for(unsigned int i=0; i<All_triggers.size(); i++) {
     fprintf(stdout,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",All_triggers[i].first.c_str(), plot->HistoMatched->GetBinContent(i+1), plot->HistoIncMatched->GetBinContent(i+1), plot->HistoIncMatched->Integral(1, i+1));
   }
   fprintf(stdout,  "\nIn events with reconstructed stand alone muon\n");
   for(unsigned int i=0; i<AllSA_triggers.size(); i++) {
     fprintf(stdout,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",AllSA_triggers[i].first.c_str(), plot->HistoMatchedSA->GetBinContent(i+1), plot->HistoIncMatchedSA->GetBinContent(i+1), plot->HistoIncMatchedSA->Integral(1, i+1));
   }
   fprintf(stdout,  "\nIn events with global muon\n");
   for(unsigned int i=0; i<All_triggers.size(); i++) {
     fprintf(stdout,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",All_triggers[i].first.c_str(), plot->HistoMatchedGl->GetBinContent(i+1), plot->HistoIncMatchedGl->GetBinContent(i+1), plot->HistoIncMatchedGl->Integral(1, i+1));
   }
   fprintf(pFile,  "\n\n");
   fprintf(stdout,  "\n\n");
}

void layout(stPlot** plots, vector<string>& sigs, string name){
  //unsigned int NPath   = 0+3;

   std::vector<string> legend;
   TObject** Histos1 = new TObject*[sigs.size()];

//   TLine* line1 = new TLine(plots[0]->Histo->GetBinLowEdge(NPath+1), 0, plots[0]->Histo->GetBinLowEdge(NPath+1), 100);
//   line1->SetLineWidth(2); line1->SetLineStyle(1);
//   TLine* line2 = new TLine(plots[0]->Histo->GetBinLowEdge(NPath+3), 0, plots[0]->Histo->GetBinLowEdge(NPath+3), 100);
//   line2->SetLineWidth(2); line2->SetLineStyle(1);

   TCanvas* c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
   c1->SetGrid();
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<sigs.size();i++){
      Histos1[i]=plots[i]->Histo; legend.push_back(sigs[i]);
   }

//   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);  
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   c1->Update();

   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   c1->Update();

   DrawPreliminary("Simulation", SQRTS, -1);
   c1->Update();
   for(unsigned int i=0;i<sigs.size();i++){
      plots[i]->Histo->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->Histo->SetMarkerSize(0.8);
   }
//   line1->Draw();
//   line2->Draw();
   c1->Update();
   SaveCanvas(c1,OutputDirectory,name);
   delete c1;

   c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
   c1->SetGrid();
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<sigs.size();i++){
      Histos1[i]=plots[i]->HistoInc; legend.push_back(sigs[i]);
   }
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);

   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   DrawPreliminary("Simulation", SQRTS, -1);
   for(unsigned int i=0;i<sigs.size();i++){
      plots[i]->HistoInc->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->HistoInc->SetMarkerSize(0.8);
   }

//   line1->Draw();
//   line2->Draw();
   SaveCanvas(c1,OutputDirectory,name + "_inc");
   delete c1;


   c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
   c1->SetGrid();
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<sigs.size();i++){
      Histos1[i]=plots[i]->HistoMatched; legend.push_back(sigs[i]);
   }
//   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);  
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   DrawPreliminary("Simulation", SQRTS, -1);

   for(unsigned int i=0;i<sigs.size();i++){
      plots[i]->HistoMatched->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->HistoMatched->SetMarkerSize(0.8);
   }
//   line1->Draw();
//   line2->Draw();
   SaveCanvas(c1,OutputDirectory,name + "Matched");
   delete c1;

   c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
   c1->SetGrid();
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<sigs.size();i++){
      Histos1[i]=plots[i]->HistoIncMatched; legend.push_back(sigs[i]);
   }
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);
   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   DrawPreliminary("Simulation", SQRTS, -1);
   for(unsigned int i=0;i<sigs.size();i++){
      plots[i]->HistoIncMatched->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->HistoIncMatched->SetMarkerSize(0.8);
   }

//   line1->Draw();
//   line2->Draw();
   SaveCanvas(c1,OutputDirectory,name + "_incMatched");
   delete c1;



   c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
   c1->SetGrid();
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<sigs.size();i++){
      Histos1[i]=plots[i]->HistoMatchedSA; legend.push_back(sigs[i]);
   }
//   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);  
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   DrawPreliminary("Simulation", SQRTS, -1);

   for(unsigned int i=0;i<sigs.size();i++){
      plots[i]->HistoMatchedSA->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->HistoMatchedSA->SetMarkerSize(0.8);
   }
//   line1->Draw();
//   line2->Draw();
   SaveCanvas(c1,OutputDirectory,name + "MatchedSA");
   delete c1;

   c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
   c1->SetGrid();
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<sigs.size();i++){
      Histos1[i]=plots[i]->HistoIncMatchedSA; legend.push_back(sigs[i]);
   }
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);
   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   DrawPreliminary("Simulation", SQRTS, -1);
   for(unsigned int i=0;i<sigs.size();i++){
      plots[i]->HistoIncMatchedSA->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->HistoIncMatchedSA->SetMarkerSize(0.8);
   }

//   line1->Draw();
//   line2->Draw();
   SaveCanvas(c1,OutputDirectory,name + "_incMatchedSA");
   delete c1;






   c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
   c1->SetGrid();
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<sigs.size();i++){
      Histos1[i]=plots[i]->HistoMatchedGl; legend.push_back(sigs[i]);
   }
//   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);  
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   DrawPreliminary("Simulation", SQRTS, -1);

   for(unsigned int i=0;i<sigs.size();i++){
      plots[i]->HistoMatchedGl->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->HistoMatchedGl->SetMarkerSize(0.8);
   }
//   line1->Draw();
//   line2->Draw();
   SaveCanvas(c1,OutputDirectory,name + "MatchedGl");
   delete c1;

   c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
   c1->SetGrid();
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<sigs.size();i++){
      Histos1[i]=plots[i]->HistoIncMatchedGl; legend.push_back(sigs[i]);
   }
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);
   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   DrawPreliminary("Simulation", SQRTS, -1);
   for(unsigned int i=0;i<sigs.size();i++){
      plots[i]->HistoIncMatchedGl->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->HistoIncMatchedGl->SetMarkerSize(0.8);
   }

//   line1->Draw();
//   line2->Draw();
   SaveCanvas(c1,OutputDirectory,name + "_incMatchedGl");
   delete c1;
}


double FastestHSCP(const fwlite::ChainEvent& ev){
   fwlite::Handle< std::vector<reco::GenParticle> > genCollHandle;
   genCollHandle.getByLabel(ev, "genParticles");
   if(!genCollHandle.isValid()){printf("GenParticle Collection NotFound\n");return -1;}
   std::vector<reco::GenParticle> genColl = *genCollHandle;

   double MaxBeta=-1;
   for(unsigned int g=0;g<genColl.size();g++){
      if(genColl[g].pt()<5)continue;
      if(genColl[g].status()!=1)continue;
      int AbsPdg=abs(genColl[g].pdgId());
      if(AbsPdg<1000000 && AbsPdg!=17)continue;

      double beta=genColl[g].p()/genColl[g].energy();
      if(MaxBeta<beta)MaxBeta=beta;
   }
   return MaxBeta;
}
