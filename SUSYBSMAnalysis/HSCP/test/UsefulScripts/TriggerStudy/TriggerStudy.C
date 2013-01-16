namespace reco { class Vertex; class Track; class GenParticle; class DeDxData; class MuonTimeExtra; class PFMET; class HitPattern;}
namespace susybsm { class HSCParticle; class HSCPIsolation; class MuonSegment; class HSCPDeDxInfo;}
namespace fwlite { class ChainEvent;}
namespace trigger { class TriggerEvent;}
namespace edm { class TriggerResults; class TriggerResultsByName; class InputTag; class LumiReWeighting;}
namespace reweight{ class PoissonMeanShifter;}

#include "TLatex.h"

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
#include "TProfile.h"

std::vector< float > BgLumiMC; //MC                                                                                                                                                  
std::vector< float > TrueDist;
edm::LumiReWeighting LumiWeightsMC;
reweight::PoissonMeanShifter PShift(0.6);

std::vector<stSample> samples;
vector<std::pair<string,string>> All_triggers;

std::string OutputDirectory="pictures/";
std::string HLTName="HLT";

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

  TH1D* DPhiMET;
  TH1D* DPhiMETMatched;
  TH1D* DPhiMETNotMatched;
  TH2D* DPhiMET1vs2NoneCharged;
  TH2D* DPhiMET1vs2OneCharged;
  TH2D* DPhiMET1vs2BothCharged;

  TH1D* DPhiHSCP;
  TH1D* DPhiHSCPMETTrigger;
  TH1D* DPhiHSCPNotMETTrigger;

  TH1D* SystPt;
  TH1D* SystPtMETTrigger;
  TH1D* SystPtNotMETTrigger;

  TH1D* SystPtDiffMET;
  TH2D* SystPtMET;
  TH2D* SystPhiMET;

  TH1D* GenPt;
  TH1D* GenPtMuTrigger;

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

      for(unsigned int i=0;i<All_triggers.size();i++) { HistoMatchedSA->GetXaxis()->SetBinLabel(i+1,All_triggers[i].second.c_str()); }
      HistoMatchedSA->GetXaxis()->SetBinLabel(numberofbins,"Total");

      for(unsigned int i=0;i<All_triggers.size();i++) { HistoIncMatchedSA->GetXaxis()->SetBinLabel(i+1,All_triggers[i].second.c_str()); }
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

      DPhiMET          = new TH1D((SignalName + "DPhiMET"         ).c_str() ,(SignalName + "DPhiMET"         ).c_str(),50,0, 3.14); DPhiMET         ->Sumw2();
      DPhiMETMatched   = new TH1D((SignalName + "DPhiMETMatched"  ).c_str() ,(SignalName + "DPhiMETMatched"  ).c_str(),50,0, 3.14); DPhiMETMatched  ->Sumw2();
      DPhiMETNotMatched = new TH1D((SignalName + "DPhiMETNotMatched").c_str() ,(SignalName + "DPhiMETNotMatched").c_str(),50,0, 3.14); DPhiMETNotMatched->Sumw2();

      DPhiMET1vs2NoneCharged = new TH2D((SignalName + "DPhiMET1vs2NoneCharged").c_str() ,(SignalName).c_str(),50,0, 3.14,50,0, 3.14); DPhiMET1vs2NoneCharged->Sumw2();
      DPhiMET1vs2OneCharged = new TH2D((SignalName + "DPhiMET1vs2OneCharged").c_str() ,(SignalName).c_str(),50,0, 3.14,50,0, 3.14); DPhiMET1vs2OneCharged->Sumw2();
      DPhiMET1vs2BothCharged = new TH2D((SignalName + "DPhiMET1vs2BothCharged").c_str() ,(SignalName).c_str(),50,0, 3.14,50,0, 3.14); DPhiMET1vs2BothCharged->Sumw2();

      DPhiHSCP          = new TH1D((SignalName + "DPhiHSCP"         ).c_str() ,(SignalName + "DPhiHSCP"         ).c_str(),50,0, 3.14); DPhiHSCP         ->Sumw2();
      DPhiHSCPMETTrigger   = new TH1D((SignalName + "DPhiHSCPMETTrigger"  ).c_str() ,(SignalName + "DPhiHSCPMETTrigger"  ).c_str(),50,0, 3.14); DPhiHSCPMETTrigger  ->Sumw2();
      DPhiHSCPNotMETTrigger = new TH1D((SignalName + "DPhiHSCPNotMETTrigger").c_str() ,(SignalName + "DPhiHSCPNotMETTrigger").c_str(),50,0, 3.14); DPhiHSCPNotMETTrigger->Sumw2();

      SystPt          = new TH1D((SignalName + "SystPt"         ).c_str() ,(SignalName + "SystPt"         ).c_str(),50,0, 500); SystPt         ->Sumw2();
      SystPtMETTrigger   = new TH1D((SignalName + "SystPtMETTrigger"  ).c_str() ,(SignalName + "SystPtMETTrigger"  ).c_str(),50,0, 500); SystPtMETTrigger  ->Sumw2();
      SystPtNotMETTrigger = new TH1D((SignalName + "SystPtNotMETTrigger").c_str() ,(SignalName + "SystPtNotMETTrigger").c_str(),50,0, 500); SystPtNotMETTrigger->Sumw2();

      SystPtDiffMET = new TH1D((SignalName + "SystPtDiffMET").c_str() ,(SignalName + "SystPtDiffMET").c_str(),50,-100, 100); SystPtDiffMET         ->Sumw2();
      SystPtMET = new TH2D((SignalName + "SystPtMET").c_str() ,(SignalName).c_str(),50,0, 500,50,0, 500); SystPtMET->Sumw2();
      SystPhiMET = new TH2D((SignalName + "SystPhiMET").c_str() ,(SignalName).c_str(),50,0, 3.14,50,0, 3.14); SystPhiMET->Sumw2();

      GenPt = new TH1D((SignalName + "GenPt").c_str() ,(SignalName + "GenPt").c_str(),600,0, 600); GenPt->Sumw2();
      GenPtMuTrigger = new TH1D((SignalName + "GenPtMuTrigger").c_str() ,(SignalName + "GenPtMuTrigger").c_str(),600,0, 600); GenPtMuTrigger->Sumw2();
   }
};


void TriggerStudy_Core(string SignalName, FILE* pFile, FILE* pTableFile, FILE* pTableFileMatched, FILE* pTableFileMatchedSA, stPlot* plot);
double FastestHSCP(const fwlite::ChainEvent& ev);
//bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, double etaCut,int NObjectAboveThreshold, bool averageThreshold=false);
void layout(stPlot** plots, vector<string>& sigs, string name);
string LatexName(string Name);
double METdPhi(double HSCPphi, const fwlite::ChainEvent& ev);
double GenDr(double HSCPeta, double HSCPphi, const fwlite::ChainEvent& ev);
void TrigTurnOn(string SaveName, string OutputDirectory, TH1D* htop_data, TH1D* hbot_data);

void TriggerStudy(string Name="COMPILE", string Sample1="Sample1", string Sample2="Sample2", string Sample3="Sample3", string Sample4="Sample4", string Sample5="Sample5", string Sample6="Sample6")
{
  if(Name=="COMPILE") return;

   system("mkdir pictures");
   std::vector<string> SamplesToRun;
   if(Sample1!="Sample1") SamplesToRun.push_back(Sample1);
   if(Sample2!="Sample2") SamplesToRun.push_back(Sample2);
   if(Sample3!="Sample3") SamplesToRun.push_back(Sample3);
   if(Sample4!="Sample4") SamplesToRun.push_back(Sample4);
   if(Sample5!="Sample5") SamplesToRun.push_back(Sample5);
   if(Sample6!="Sample6") SamplesToRun.push_back(Sample6);
   /*
   TFile* OutputHisto = new TFile((OutputDirectory + "/Histos_" + Name + ".root").c_str(),"RECREATE");
   stPlot** plots = new stPlot*[6];
   plots[0] = new stPlot(Sample1);
   plots[1] = new stPlot(Sample2);
   plots[2] = new stPlot(Sample3);
   plots[3] = new stPlot(Sample4);
   plots[4] = new stPlot(Sample5);
   plots[5] = new stPlot(Sample6);
   */
   setTDRStyle();
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
   gStyle->SetPadGridY(false);
   gStyle->SetErrorX(0.5);

   InitBaseDirectory();
   GetSampleDefinition(samples, "../../ICHEP_Analysis/Analysis_Samples.txt");

   keepOnlySamplesOfNamesXtoY(samples, SamplesToRun);

   //initialize LumiReWeighting
   BgLumiMC.clear();
   TrueDist.clear();
#ifdef ANALYSIS2011
   if(Name.find("7TeV")==string::npos){printf("Skip %s because of wrong center of mass energy\n", Name.c_str());return;}

   for(int i=0; i<60; ++i) BgLumiMC    .push_back(Pileup_MC_Fall11[i]);
   for(int i=0; i<60; ++i) TrueDist    .push_back(TrueDist2011_f[i]);
   SQRTS=7;
   if(Name.find("LQ")!=string::npos) HLTName="HLTSIMHITSHIFTER";
#else
   if(Name.find("8TeV")==string::npos){printf("Skip %s because of wrong center of mass energy\n", Name.c_str());return;}
   if(samples[0].Pileup=="S10") {for(int i=0; i<60; ++i) BgLumiMC.push_back(Pileup_MC_Summer2012[i]);
   }else{                        for(int i=0; i<60; ++i) BgLumiMC.push_back(Pileup_MC_Fall11[i]);
   }

   for(int i=0; i<60; ++i) TrueDist    .push_back(TrueDist2012_f[i]);

   SQRTS=8;
   if(Name.find("LQ")!=string::npos) HLTName="HLTSIMHITSHIFTER";
#endif
   LumiWeightsMC     = edm::LumiReWeighting(BgLumiMC, TrueDist);

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

   ///////////////////////////////////////////////////////

   FILE* pFile = fopen((OutputDirectory + "Results_" + Name + ".txt").c_str(),"w");
   FILE* pTableFile = fopen((OutputDirectory + "Results_" + Name + "_Table.txt").c_str(),"w");
   FILE* pTableFileMatched = fopen((OutputDirectory + "Results_" + Name + "_TableMatched.txt").c_str(),"w");
   FILE* pTableFileMatchedSA = fopen((OutputDirectory + "Results_" + Name + "_TableMatchedSA.txt").c_str(),"w");

   string OutputDirectory ="pictures/";
#ifdef ANALYSIS2011
   OutputDirectory = "/uscms_data/d2/farrell3/WorkArea/14Aug2012/CMSSW_5_3_3/src/SUSYBSMAnalysis/HSCP/test/UsefulScripts/TriggerStudy/" + OutputDirectory;
#endif


   vector<string> leg;
   stPlot** plots = new stPlot*[samples.size()];  
   for(unsigned int i=0;i<samples.size();i++){
     if(samples[i].Type!=2)continue;
     leg.push_back(samples[i].Legend);
     plots[i] = new stPlot(samples[i].Name);
     string OldHLTName=HLTName;
     TriggerStudy_Core(samples[i].Name, pFile, pTableFile, pTableFileMatched, pTableFileMatchedSA, plots[i]);
     HLTName=OldHLTName;
   }

   fflush(pFile);
   fclose(pFile);

   fflush(pTableFile);
   fclose(pTableFile);

   fflush(pTableFileMatched);
   fclose(pTableFileMatched);

   fflush(pTableFileMatchedSA);
   fclose(pTableFileMatchedSA);
   if(samples.size()!=0) layout(plots, leg, Name);
   /*
   OutputHisto->Write();
   OutputHisto->Close();

   delete plots[0];
   delete plots[1];
   delete plots[2];
   delete plots[3];
   delete plots[4];
   delete plots[5];

   delete plots;
   delete OutputHisto;
   */
}

void TriggerStudy_Core(string SignalName, FILE* pFile, FILE* pTableFile, FILE* pTableFileMatched, FILE* pTableFileMatchedSA, stPlot* plot)
{
  double Total=0;
  //double TotalMatched=0;
  //double TotalMatchedSA=0;
  //double TotalMatchedGl=0;

  int JobId = JobIdToIndex(SignalName, samples);
  for (int period=0; period<2; period++) {
    if(SignalName.find("1o3")!=string::npos && SignalName.find("7TeV")!=string::npos && SignalName.find("M500")!=string::npos && period==1) HLTName="HLT";

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
    int MaxEvent = -1;

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
      Total+=Event_Weight;
      edm::TriggerResultsByName tr = ev.triggerResultsByName("MergeHLT"); 
//      if(simhitshifted) tr= ev.triggerResultsByName("HLTSIMHITSHIFTER");
//      edm::TriggerResultsByName tr = ev.triggerResultsByName("HLT");      if(!tr.isValid())continue;
      //    for(unsigned int i=0;i<tr.size();i++){
      //   printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
      //}fflush(stdout);

      fwlite::Handle< trigger::TriggerEvent > trEvHandle;
      trEvHandle.getByLabel(ev,"hltTriggerSummaryAOD");
      trigger::TriggerEvent trEv = *trEvHandle;

      //get the collection of generated Particles
      fwlite::Handle< std::vector<reco::GenParticle> > genCollHandle;
      genCollHandle.getByLabel(ev, "genParticles");
      if(!genCollHandle.isValid()){printf("GenParticle Collection NotFound\n");continue;}
      std::vector<reco::GenParticle> genColl = *genCollHandle;

      //Have to do this here as HSCParticleCollection not always in event
      double Beta = 1.0;
      if(SignalName!="Data")Beta = FastestHSCP(ev);
      plot->BetaCount->Fill(Beta,Event_Weight);

      for(unsigned int g=0;g<genColl.size();g++){
	if(genColl[g].pt()<5)continue;
	if(genColl[g].status()!=1)continue;
	int AbsPdg=abs(genColl[g].pdgId());
	if(AbsPdg<1000000 && AbsPdg!=17)continue;

	plot->GenPt->Fill(genColl[g].pt(),Event_Weight);
	if(tr.accept("HSCPHLTTriggerMuFilter") && GenDr(genColl[g].eta(), genColl[g].phi(), ev)<0.3) plot->GenPtMuTrigger->Fill(genColl[g].pt(),Event_Weight);
      }

      fwlite::Handle<susybsm::HSCParticleCollection> hscpCollHandle;
      hscpCollHandle.getByLabel(ev,"HSCParticleProducer");
      if(!hscpCollHandle.isValid()){continue;}
      susybsm::HSCParticleCollection hscpColl = *hscpCollHandle;

      int NChargedHSCP=HowManyChargedHSCP(genColl);
      Event_Weight*=samples[JobId].GetFGluinoWeight(NChargedHSCP);

      //Match gen HSCP to reco tracks
      bool match=false; double betaMatched=-9999.; 
      bool matchSA=false; double betaMatchedSA=-9999.;
      bool matchGl=false; double betaMatchedGl=-9999.;
      double dphiHSCP[2]={20,20};
      bool HSCPCharged[2]={false,false};
      int HSCPIndex[2]={-1,-1};

	for(unsigned int g=0;g<genColl.size();g++){
	  if(genColl[g].pt()<5)continue;
	  if(genColl[g].status()!=1)continue;
	  int AbsPdg=abs(genColl[g].pdgId());
	  if(AbsPdg<1000000 && AbsPdg!=17)continue;

	  double dphi = METdPhi(genColl[g].phi(), ev);

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
	    if(!track.isNull() && track->pt()>45) {
	      double dR = deltaR(track->eta(), track->phi(), genColl[g].eta(), genColl[g].phi());
	      if(dR<RMin)RMin=dR;
	    }

	    if(!trackSA.isNull() && trackSA->pt()>80){
	      double dR = deltaR(trackSA->eta(), trackSA->phi(), genColl[g].eta(), genColl[g].phi());
	      if(dR<RMinSA)RMinSA=dR;
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
          if(dphiHSCP[0]==20) {
	    HSCPIndex[0]=g;
	    dphiHSCP[0]=dphi;
	    if(match) HSCPCharged[0]=true;
	  }
          else {
            HSCPIndex[1]=g;
	    dphiHSCP[1]=dphi;
	    if(match) HSCPCharged[1]=true;
	  }
	
	plot->DPhiMET->Fill(dphi,Event_Weight);
	if(match) plot->DPhiMETMatched->Fill(dphi,Event_Weight);
	else plot->DPhiMETNotMatched->Fill(dphi,Event_Weight);         
	}

	if(HSCPIndex[0]!=-1 && HSCPIndex[1]!=-1) {
	  double HSCPdphi=genColl[HSCPIndex[0]].phi()-genColl[HSCPIndex[1]].phi();
	  while (HSCPdphi >   M_PI) HSCPdphi -= 2*M_PI;
	  while (HSCPdphi <= -M_PI) HSCPdphi += 2*M_PI;
	  if(HSCPdphi<0) HSCPdphi = HSCPdphi*-1;

	  plot->DPhiHSCP->Fill(HSCPdphi,Event_Weight);
	  if(tr.accept("HSCPHLTTriggerPFMetFilter")) plot->DPhiHSCPMETTrigger->Fill(HSCPdphi,Event_Weight);
	  else plot->DPhiHSCPNotMETTrigger->Fill(HSCPdphi,Event_Weight);
	
	  double SystemPt=sqrt(pow(genColl[HSCPIndex[0]].px()+genColl[HSCPIndex[1]].px(),2)+pow(genColl[HSCPIndex[0]].py()+genColl[HSCPIndex[1]].py(),2));
	  double SystemPhi=acos((genColl[HSCPIndex[0]].px()+genColl[HSCPIndex[1]].px())/SystemPt);
          while (SystemPhi >   M_PI) SystemPhi -= 2*M_PI;
          while (SystemPhi <= -M_PI) SystemPhi += 2*M_PI;
          if(SystemPhi<0) SystemPhi = SystemPhi*-1;

          plot->SystPt->Fill(SystemPt,Event_Weight);
          if(tr.accept("HSCPHLTTriggerPFMetFilter")) {
	    plot->SystPtMETTrigger->Fill(SystemPt,Event_Weight);

	    unsigned int filterIndex = trEv.filterIndex(InputTag("hltPFMHT150Filter","",HLTName));
	    const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
	    const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
	    const int nI(VIDS.size());
	    const int nK(KEYS.size());
	    assert(nI==nK);
	    const int n(std::max(nI,nK));
	    const trigger::TriggerObjectCollection& TOC(trEv.getObjects());

	    for (int i=0; i!=n; ++i) {
	      if(TOC[KEYS[i]].pt()<150) continue;
	      plot->SystPtDiffMET->Fill(SystemPt-TOC[KEYS[0]].pt(),Event_Weight);
	      plot->SystPtMET->Fill(SystemPt,TOC[KEYS[0]].pt(),Event_Weight);
	      plot->SystPhiMET->Fill(SystemPhi,TOC[KEYS[0]].phi(),Event_Weight);
	    }
	  }
          else plot->SystPtNotMETTrigger->Fill(SystemPt,Event_Weight);

	  if(HSCPCharged[0]==false && HSCPCharged[1]==false) plot->DPhiMET1vs2NoneCharged->Fill(dphiHSCP[0],dphiHSCP[1],Event_Weight);
	  if(HSCPCharged[0]==false && HSCPCharged[1]==true)  plot->DPhiMET1vs2OneCharged ->Fill(dphiHSCP[1],dphiHSCP[0],Event_Weight);
	  if(HSCPCharged[0]==true  && HSCPCharged[1]==false) plot->DPhiMET1vs2OneCharged ->Fill(dphiHSCP[0],dphiHSCP[1],Event_Weight);
	  if(HSCPCharged[0]==true  && HSCPCharged[1]==true)  plot->DPhiMET1vs2BothCharged->Fill(dphiHSCP[0],dphiHSCP[1],Event_Weight);
	}

	unsigned int TrIndex_Unknown     = tr.size();
   
	bool AlreadyAccepted = false;
	bool AcceptMu = false;
	bool AcceptMET = false;
        bool AcceptMuMET = false;

	for(unsigned int i=0;i<All_triggers.size();i++){
	  if(TrIndex_Unknown==tr.triggerIndex(All_triggers[i].first))  {cout << "Trigger " << All_triggers[i].first << " not found" << endl; continue;}
	  bool Accept = tr.accept(All_triggers[i].first.c_str());


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
	 if(Accept && matchSA){
	   plot->HistoMatchedSA  ->Fill(All_triggers[i].second.c_str(),Event_Weight);
	   if(!AlreadyAccepted) plot->HistoIncMatchedSA->Fill(All_triggers[i].second.c_str(),Event_Weight);
	 }

         AlreadyAccepted |= Accept;

	 if(Accept && All_triggers[i].first.find("HSCPHLTTriggerMuFilter")!=string::npos) AcceptMu=true;
         if(Accept && All_triggers[i].first.find("PFMet")!=string::npos) AcceptMET=true;
	 if(Accept && All_triggers[i].first.find("HSCPHLTTriggerL2MuFilter")!=string::npos) AcceptMuMET=true;
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
         if(matchSA) {
           plot->HistoMatchedSA->Fill("Total",Event_Weight);
           plot->HistoIncMatchedSA->Fill("Total",Event_Weight);
         }
      }
   
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

   //plot->BetaTotalMatchedGl->Divide(plot->BetaCountMatchedGl);
   //plot->BetaMuonMatchedGl ->Divide(plot->BetaCountMatchedGl);
   //plot->BetaJetMatchedGl  ->Divide(plot->BetaCountMatchedGl);

   //plot->BetaTotalMatchedGl->Scale(100.0);
   //plot->BetaMuonMatchedGl ->Scale(100.0);
   //plot->BetaJetMatchedGl  ->Scale(100.0);

   TH1** Histos = new TH1*[10];
   std::vector<string> legend;
   TCanvas* c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->BetaCount;                    legend.push_back("All");
   Histos[1] = (TH1*)plot->BetaMuon;                    legend.push_back("Mu triggers");
   Histos[2] = (TH1*)plot->BetaTotal;                   legend.push_back("Mu+Met triggers");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "#beta of the fastest HSCP", "Trigger Efficiency (%)", 0,1, 0,0);
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
   Histos[0] = (TH1*)plot->BetaCountMatchedGl;                    legend.push_back("All");
   Histos[1] = (TH1*)plot->BetaMuonMatchedGl;                    legend.push_back("Mu triggers");
   Histos[2] = (TH1*)plot->BetaTotalMatchedGl;                   legend.push_back("Mu+Met triggers");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "#beta of the fastest HSCP", "Trigger Efficiency (%)", 0,1, 0,0);
   DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary("Simulation", SQRTS, -1);
   SaveCanvas(c1,OutputDirectory,SignalName + "MatchedGl");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->DPhiMET->Clone();  Histos[0]->Scale(1./Histos[0]->Integral());          legend.push_back("All");
   Histos[1] = (TH1*)plot->DPhiMETMatched->Clone();  Histos[1]->Scale(1./Histos[1]->Integral());                 legend.push_back("Matched");
   Histos[2] = (TH1*)plot->DPhiMETNotMatched->Clone(); Histos[2]->Scale(1./Histos[2]->Integral());                  legend.push_back("Not Matched");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "#delta#phi between HSCP and MET", "Normalized Units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary("Simulation", SQRTS, -1);
   SaveCanvas(c1,OutputDirectory,SignalName + "DPhiMET");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->DPhiMET1vs2NoneCharged->Clone();                  legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#delta#phi_1", "#deltaphi_2", 0,0, 0,0, false);
   DrawPreliminary("Simulation", SQRTS, -1);
   c1->SetLogz(1);
   SaveCanvas(c1,OutputDirectory,SignalName + "dphi1vs2NoneCharged", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->DPhiMET1vs2OneCharged->Clone();                  legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#delta#phi_1", "#deltaphi_2", 0,0, 0,0, false);
   DrawPreliminary("Simulation", SQRTS, -1);
   c1->SetLogz(1);
   SaveCanvas(c1,OutputDirectory,SignalName + "dphi1vs2OneCharged", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->DPhiMET1vs2BothCharged->Clone();                  legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "#delta#phi_1", "#deltaphi_2", 0,0, 0,0, false);
   DrawPreliminary("Simulation", SQRTS, -1);
   c1->SetLogz(1);
   SaveCanvas(c1,OutputDirectory,SignalName + "dphi1vs2BothCharged", true);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->DPhiHSCP->Clone();  Histos[0]->Scale(1./Histos[0]->Integral());          legend.push_back("All");
   Histos[1] = (TH1*)plot->DPhiHSCPMETTrigger->Clone();  Histos[1]->Scale(1./Histos[1]->Integral());                 legend.push_back("METTrigger");
   Histos[2] = (TH1*)plot->DPhiHSCPNotMETTrigger->Clone(); Histos[2]->Scale(1./Histos[2]->Integral());                  legend.push_back("Not METTrigger");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "#delta#phi between HSCP", "Normalized Units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary("Simulation", SQRTS, -1);
   SaveCanvas(c1,OutputDirectory,SignalName + "DPhiHSCP");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->SystPt->Clone();  legend.push_back("All");
   Histos[1] = (TH1*)plot->SystPtMETTrigger->Clone();  legend.push_back("METTrigger");
   //Histos[2] = (TH1*)plot->SystPtNotMETTrigger->Clone();  legend.push_back("Not METTrigger");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "HSCP System Pt", "Normalized Units", 0,0, 0,0);
   DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary("Simulation", SQRTS, -1);
   SaveCanvas(c1,OutputDirectory,SignalName + "SystPt");
   delete c1;

   //TrigTurnOn(SignalName + "SystPtEff", OutputDirectory, plot->SystPtMETTrigger, plot->SystPt);

   //c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   //Histos[0] = (TH1*)plot->SystPtMETTrigger->Clone();  legend.push_back("METTrigger");
   //Histos[0]->Divide(plot->SystPt);
   //DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "HSCP System Pt", "Efficiency", 0,0, 0,0);
   //DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   //c1->Modified();
   //DrawPreliminary("Simulation", SQRTS, -1);
   //SaveCanvas(c1,OutputDirectory,SignalName + "SystPtEff");
   //delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->SystPtDiffMET->Clone();  legend.push_back("All");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "HSCP System Pt - HLT MET", "Normalized Units", 0,0, 0,0);
   //DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary("Simulation", SQRTS, -1);
   SaveCanvas(c1,OutputDirectory,SignalName + "SystPtDiffMET");
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->SystPtMET->Clone();                  legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "HSCP System Pt", "HLT PFMET", 0,0, 0,0, false);
   DrawPreliminary("Simulation", SQRTS, -1);
   c1->SetLogz(1);
   SaveCanvas(c1,OutputDirectory,SignalName + "SystPtMET", false);
   delete c1;

   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->SystPhiMET->Clone();                  legend.push_back("Before Cut");
   DrawSuperposedHistos((TH1**)Histos, legend, "COLZ",  "HSCP System #phi", "HLT PFMET #phi", 0,0, 0,0, false);
   DrawPreliminary("Simulation", SQRTS, -1);
   c1->SetLogz(1);
   SaveCanvas(c1,OutputDirectory,SignalName + "SystPhiMET", false);
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
   for(unsigned int i=0; i<All_triggers.size(); i++) {
     fprintf(pFile,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",All_triggers[i].first.c_str(), plot->HistoMatchedSA->GetBinContent(i+1), plot->HistoIncMatchedSA->GetBinContent(i+1), plot->HistoIncMatchedSA->Integral(1, i+1));
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
   for(unsigned int i=0; i<All_triggers.size(); i++) {
     fprintf(stdout,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",All_triggers[i].first.c_str(), plot->HistoMatchedSA->GetBinContent(i+1), plot->HistoIncMatchedSA->GetBinContent(i+1), plot->HistoIncMatchedSA->Integral(1, i+1));
   }
   fprintf(stdout,  "\nIn events with global muon\n");
   for(unsigned int i=0; i<All_triggers.size(); i++) {
     fprintf(stdout,  "Trigger %15s Efficiency = %5.2f%% which adds an incremental efficiency = %5.2f%% Cumulative Efficiency = %5.2f%%\n",All_triggers[i].first.c_str(), plot->HistoMatchedGl->GetBinContent(i+1), plot->HistoIncMatchedGl->GetBinContent(i+1), plot->HistoIncMatchedGl->Integral(1, i+1));
   }
   fprintf(pFile,  "\n\n");
   fprintf(stdout,  "\n\n");

   int Triggers=3;
#ifdef ANALYSIS2011
   Triggers=2;
#endif

   fprintf(pTableFile, "%15s & %4.0f  ",LatexName(samples[JobId].Name).c_str(), samples[JobId].Mass);
   for(int i=0; i<Triggers; i++) {
     fprintf(pTableFile, "& %5.2f      ", plot->Histo->GetBinContent(i+1));
   }
   fprintf(pTableFile, "& %5.2f ", plot->HistoInc->Integral(1, Triggers));
   fprintf(pTableFile, "  \\\\ \n");

   fprintf(pTableFileMatched, "%15s & %4.0f  ",LatexName(samples[JobId].Name).c_str(), samples[JobId].Mass);
   for(int i=0; i<Triggers; i++) {
     fprintf(pTableFileMatched, "& %5.2f      ", plot->HistoMatched->GetBinContent(i+1));
   }
   fprintf(pTableFileMatched, "& %5.2f ", plot->HistoIncMatched->Integral(1, Triggers));
   fprintf(pTableFileMatched, "  \\\\ \n");

   fprintf(pTableFileMatchedSA, "%15s & %4.0f  ",LatexName(samples[JobId].Name).c_str(), samples[JobId].Mass);
   for(int i=0; i<Triggers; i++) {
     fprintf(pTableFileMatchedSA, "& %5.2f      ", plot->HistoMatchedSA->GetBinContent(i+1));
   }
   fprintf(pTableFileMatchedSA, "& %5.2f  ", plot->HistoIncMatchedSA->Integral(1, Triggers));
   fprintf(pTableFileMatchedSA, "  \\\\ \n");
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
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<sigs.size();i++){
      Histos1[i]=plots[i]->Histo; legend.push_back(sigs[i]);
   }

//   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);  
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   DrawPreliminary("Simulation", SQRTS, -1);
   for(unsigned int i=0;i<sigs.size();i++){
      plots[i]->Histo->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->Histo->SetMarkerSize(0.8);
   }
//   line1->Draw();
//   line2->Draw();
   SaveCanvas(c1,OutputDirectory,name);
   delete c1;

   c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
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

string LatexName(string Name) {
  string toReturn="";
  if(Name.find("Gluino")!=string::npos) {
    toReturn+="Gluino";
    if(Name.find("N")!=string::npos) toReturn+="N";

    if(Name.find("f100")!=string::npos) toReturn+=" $f=1.0$";
    else if(Name.find("f50")!=string::npos) toReturn+=" $f=0.5$";
    else if(Name.find("f10")!=string::npos) toReturn+=" $f=0.1$";
  }
  else if(Name.find("Stop")!=string::npos) {
    toReturn+="Stop";
    if(Name.find("N")!=string::npos) toReturn+="N";
  }
  else if(Name.find("GMStau")!=string::npos) toReturn+="GMSB Stau";
  else if(Name.find("PPStau")!=string::npos) toReturn+="PP Stau";
  else if(Name.find("GMStau")!=string::npos) toReturn+="GMSB Stau";
  else if(Name.find("Q1o3")!=string::npos) toReturn+="DY $Q=e/3$";
  else if(Name.find("Q2o3")!=string::npos) toReturn+="DY $Q=2e/3$";
  else if(Name.find("Q1")!=string::npos) toReturn+="DY $Q=e$";
  else if(Name.find("Q2")!=string::npos) toReturn+="DY $Q=2e$";
  else if(Name.find("Q3")!=string::npos) toReturn+="DY $Q=3e$";
  else if(Name.find("Q4")!=string::npos) toReturn+="DY $Q=4e$";
  else if(Name.find("Q5")!=string::npos) toReturn+="DY $Q=5e$";
  return toReturn;
}

double METdPhi(double HSCPphi, const fwlite::ChainEvent& ev) {
  edm::TriggerResultsByName tr = ev.triggerResultsByName("MergeHLT");

  if(!tr.accept("HSCPHLTTriggerPFMetFilter")) return -10;

  fwlite::Handle< trigger::TriggerEvent > trEvHandle;
  trEvHandle.getByLabel(ev,"hltTriggerSummaryAOD");
  trigger::TriggerEvent trEv = *trEvHandle;

  unsigned int filterIndex = trEv.filterIndex(InputTag("hltPFMHT150Filter","",HLTName));

  if (filterIndex<trEv.sizeFilters()){
    const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
    const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
    const int nI(VIDS.size());
    const int nK(KEYS.size());
    assert(nI==nK);
    const int n(std::max(nI,nK));
    const trigger::TriggerObjectCollection& TOC(trEv.getObjects());

    for (int i=0; i!=n; ++i) {
      if(TOC[KEYS[i]].pt()<150) continue;
      double dphi = TOC[KEYS[i]].phi() - HSCPphi;
      while (dphi >   M_PI) dphi -= 2*M_PI;
      while (dphi <= -M_PI) dphi += 2*M_PI;
      if(dphi<0) dphi = dphi*-1;
      return dphi;
    }
  }
  else {
    edm::TriggerResultsByName tr2 = ev.triggerResultsByName(HLTName);

    for(unsigned int i=0;i<tr2.size();i++){
      printf("Path %3i %50s --> %1i\n",i, tr2.triggerName(i).c_str(),tr2.accept(i));
    }fflush(stdout);

    for ( size_t ia = 0; ia < trEv.sizeFilters(); ++ ia) {
      std::string fullname = trEv.filterTag(ia).encode();
      std::string name;
      size_t p = fullname.find_first_of(':');
      if ( p != std::string::npos) {
	name = fullname.substr(0, p);
      }
      else {
	name = fullname;
      }
      //std::cout << "Path name " << name << std::endl;
    }
    //cout << endl << endl;
  }
  return -10;
}

void TrigTurnOn(string SaveName, string OutputDirectory, TH1D* htop_data, TH1D* hbot_data) {
{
  TCanvas *c1;

  //htop_data->Rebin(6);
  //hbot_data->Rebin(6);

  //DD: Set your plots x-axis range:
 float x_min = 0;
 float x_max = 500;

 htop_data->Draw();
 hbot_data->Draw("same");
 TH1D* ratio = (TH1D*)htop_data->Clone();
 ratio->Sumw2();
 ratio->Divide(hbot_data);
  //Fitting with the error function with the form:
  //erf((m-m0)/sigma)+1)/2
  //DD: Change the range for the fitting yourself from 20 - 100 to your own (min,max)
 TF1 * f1 = new TF1("f1","(TMath::Erf((x*[0] -[1])/[2])+1.)/2.",0,6.);
 f1->SetParameter(2,1.);
 ratio->Fit(f1,"R");
  //This is all to make things look nice

 c1 = new TCanvas("c1","Canvas1",0,0,500,500);
 /*
  c1->SetLineColor(0);
  c1->SetFrameFillColor(0);
  c1->SetFillStyle(4000);
  c1->SetFillColor(0);   
  c1->SetBorderMode(0);
  gStyle->SetOptStat(0);    
  c1->SetFillColor(0);
  c1->SetBorderSize(0);
  c1->SetBorderMode(0);
  c1->SetLeftMargin(0.15);
  c1->SetRightMargin(0.12);
  c1->SetTopMargin(0.12);
  c1->SetBottomMargin(0.15);
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000); //This puts in stats box
  gStyle->SetTitleBorderSize(0);
  gStyle->SetTitleX(0.5); // X position of the title box from left
  gStyle->SetTitleAlign(23);
  gStyle->SetTitleY(.975); // Y position of the title box from bottom
  gStyle->SetLabelSize(0.03,"y");
  gStyle->SetStatX(.9);
  gStyle->SetStatY(.9);
  gStyle->SetStatW(0.20);
  gStyle->SetStatFontSize(0.044);
  gStyle->SetStatColor(0);  
 */
 gStyle->SetOptStat(0); 
 gStyle->SetOptStat(0000);

  TGraphAsymmErrors* eff0 = new TGraphAsymmErrors();

  //Really you can divide in any way you'd like, here's the link for the options:
  //http://root.cern.ch/root/html528/TGraphAsymmErrors.html (Wilson, CP, etc)
  eff0->BayesDivide(htop_data,hbot_data);
  //  eff0->BayesDivide(htop_data,hbot_data,"w");

  //
  TH1D *T_Empty = new TH1D("T_Empty", "", 1, x_min, x_max);
  T_Empty->SetMinimum(0.0);
  T_Empty->SetMaximum(1.10);
  T_Empty->SetStats(kTRUE);
  T_Empty->GetXaxis()->SetLabelOffset(0.01);
  T_Empty->GetYaxis()->SetLabelOffset(0.01);
  T_Empty->GetXaxis()->SetLabelSize(0.035);
 T_Empty->GetXaxis()->SetLabelFont(42);
 T_Empty->GetXaxis()->SetTitleSize(0.040);
 T_Empty->GetYaxis()->SetLabelSize(0.035);
 T_Empty->GetYaxis()->SetLabelFont(42);
 T_Empty->GetYaxis()->SetTitleSize(0.040);
 T_Empty->GetXaxis()->SetTitleOffset(1.29);
 T_Empty->GetYaxis()->SetTitleOffset(1.39);
 T_Empty->GetXaxis()->SetTitleColor(1);
 T_Empty->GetYaxis()->SetTitleColor(1);
 T_Empty->GetXaxis()->SetNdivisions(10505);
 T_Empty->GetYaxis()->SetNdivisions(515);
 T_Empty->GetXaxis()->SetTitleFont(42);
 T_Empty->GetYaxis()->SetTitleFont(42);
 //DD:Edit these labels
 T_Empty->GetXaxis()->SetTitle("HSCP System p_{T}");
 T_Empty->GetYaxis()->SetTitle("PFMET150 Efficiency");
 T_Empty->Draw("AXIS");
 
 eff0->SetMarkerStyle(20);
 eff0->SetMarkerSize(1.0);
 eff0->SetMarkerColor(1);
 eff0->SetLineWidth(2);

 eff0->Draw("e1pZ");

 TLegend *leg = new TLegend(0.40,0.42,0.82,0.57,NULL,"brNDC");
 leg->SetTextFont(42);
 leg->SetTextSize(0.030);
 leg->SetLineColor(1);
 leg->SetLineStyle(1);
 leg->SetLineWidth(1);
 leg->SetFillStyle(1001);
 //DD:Edit these labels and positions
 leg->AddEntry(eff0,"Any labels you'd like to add to make it nice","p");
 leg->SetBorderSize(0);
 leg->SetFillColor(0);
 //leg->Draw();

 TPaveText* T = new TPaveText(0.4,0.995,0.82,0.945, "NDC");
 T->SetFillColor(0);
 T->SetTextAlign(22);
 char tmp[2048];
 sprintf(tmp,"CMS Preliminary   #sqrt{s} = %1.0f TeV",8.0);
 T->AddText(tmp);
 T->Draw("same");

 TLatex* tex = new TLatex();
 tex->SetTextColor(1);
 tex->SetTextSize(0.030);
 tex->SetLineWidth(2);
 tex->SetTextFont(42);
 //DD:Edit these labels and positions
 //tex->DrawLatex(1000., 0.65, "More labels!");
 //tex->Draw();

 SaveCanvas(c1,OutputDirectory,SaveName);

 //DD:Edit these labels
 //99% efficiency line.
 float y_min = 0.9;
 float y_max = 0.9;
 TLine *line = new TLine(x_min, y_min, x_max, y_max);
 line->SetLineColor(kRed);
 line->SetLineWidth(2);
 line->SetLineStyle(3);
 //line->Draw("same");
 //I use this line to show on the x-axis at what x I become 99% efficient
 //but this I leave up to you.
 TLine *line2 = new TLine(900., 0.5, 900., 1.0);
 line2->SetLineColor(kBlue);
 line2->SetLineWidth(3);
 line2->SetLineStyle(2);
 //line2->Draw("same");
 f1->SetLineColor(4);
 f1->Draw("same");

 DrawPreliminary("Simulation", SQRTS, -1);
 //SaveCanvas(c1,OutputDirectory,SaveName+"_Fit");

 delete c1;
  }
 }



double GenDr(double HSCPeta, double HSCPphi, const fwlite::ChainEvent& ev) {
  edm::TriggerResultsByName tr = ev.triggerResultsByName("MergeHLT");

  if(!tr.accept("HSCPHLTTriggerMuFilter")) return -10;
  fwlite::Handle< trigger::TriggerEvent > trEvHandle;
  trEvHandle.getByLabel(ev,"hltTriggerSummaryAOD");
  trigger::TriggerEvent trEv = *trEvHandle;
  unsigned int filterIndex = trEv.filterIndex(InputTag("hltL3fL1sMu16Eta2p1L1f0L2f16QL3Filtered40Q","",HLTName));
  if (filterIndex>=trEv.sizeFilters()) filterIndex = trEv.filterIndex(InputTag("hltSingleMu30L2QualL3Filtered30","",HLTName));
  if (filterIndex>=trEv.sizeFilters()) filterIndex = trEv.filterIndex(InputTag("hltSingleMu30L3Filtered30","",HLTName));

  if (filterIndex>=trEv.sizeFilters()) {
    edm::TriggerResultsByName tr2 = ev.triggerResultsByName(HLTName);
    for(unsigned int i=0;i<tr2.size();i++){
      if(tr2.triggerName(i).find("Mu40_eta2p1")!=string::npos && !tr2.accept(i)) return -10; 
    }
  }

  if (filterIndex<trEv.sizeFilters()){

    const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
    const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
    const int nI(VIDS.size());
    const int nK(KEYS.size());
    assert(nI==nK);
    const int n(std::max(nI,nK));
    const trigger::TriggerObjectCollection& TOC(trEv.getObjects());

    double minDr=9999.;
    for (int i=0; i!=n; ++i) {
      if(TOC[KEYS[i]].pt()<40 || fabs(TOC[KEYS[i]].eta())>2.1) continue;
      double dR=deltaR(HSCPeta, HSCPphi, TOC[KEYS[i]].eta(), TOC[KEYS[i]].phi());
      if(dR<minDr) minDr=dR;
    }
    return minDr;
  }
  else {
    edm::TriggerResultsByName tr2 = ev.triggerResultsByName(HLTName);

    for(unsigned int i=0;i<tr2.size();i++){
    printf("Path %3i %50s --> %1i\n",i, tr2.triggerName(i).c_str(),tr2.accept(i));
    }fflush(stdout);

    for ( size_t ia = 0; ia < trEv.sizeFilters(); ++ ia) {
      std::string fullname = trEv.filterTag(ia).encode();
      std::string name;
      size_t p = fullname.find_first_of(':');
      if ( p != std::string::npos) {
	name = fullname.substr(0, p);
      }
      else {
	name = fullname;
      }
      std::cout << "Path name " << name << std::endl;
      if(name=="hltSingleMu30L3Filtered30") {cout << endl <<  "Found my path" << endl << endl; assert(1==0);}
    }
    cout << endl << endl;
    assert(1==0);
  }
  return -10;
}
