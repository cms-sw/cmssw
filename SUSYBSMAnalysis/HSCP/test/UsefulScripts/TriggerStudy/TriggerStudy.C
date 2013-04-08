
#include <exception>
#include <vector>
#include <string>

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TGraphAsymmErrors.h"
#include "TPaveText.h"
#include "tdrstyle.C"

 class stSignal;
 namespace edm {class TriggerResults; class TriggerResultsByName; class InputTag;}
 namespace reco { class Vertex; class Track; class GenParticle;}
 namespace susybsm {class HSCParticle;}
 namespace fwlite  { class ChainEvent;}
 namespace trigger {class TriggerEvent;}


#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"


#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"


 using namespace fwlite;
 using namespace reco;
 using namespace edm;
 using namespace std;
 using namespace trigger;

#include "Analysis_Samples.h"
#include "../../ICHEP_Analysis/Analysis_PlotFunction.h"


#endif

std::vector<stSignal> signals;
vector<string> JetMetSD_triggers;
vector<string> MuSD_triggers;
vector<string> All_triggers;
map<string,bool> All_mask;

class stPlot{
   public:
   TH1D* Histo;
   TH1D* HistoInc;
   TH1D* BetaCount;
   TH1D* BetaTotal;
   TH1D* BetaMuon;
   TH1D* BetaJet;

   stPlot(string SignalName){
      int numberofbins=JetMetSD_triggers.size()+MuSD_triggers.size()+1;
      Histo    = new TH1D((SignalName + "Abs").c_str(),(SignalName + "Abs").c_str(),numberofbins,0,numberofbins);
      HistoInc = new TH1D((SignalName + "Inc").c_str(),(SignalName + "Inc").c_str(),numberofbins,0,numberofbins);

      for(unsigned int i=0;i<MuSD_triggers.size();i++)    { Histo->GetXaxis()->SetBinLabel(i+1,MuSD_triggers[i].c_str());   }
      for(unsigned int i=0;i<JetMetSD_triggers.size();i++){ Histo->GetXaxis()->SetBinLabel(MuSD_triggers.size()+1+i,JetMetSD_triggers[i].c_str());   }
//      Histo->GetXaxis()->SetBinLabel(numberofbins-2,"Mu Paths");
//      Histo->GetXaxis()->SetBinLabel(numberofbins-1,"JetMET Paths");
      Histo->GetXaxis()->SetBinLabel(numberofbins,"Total");

      for(unsigned int i=0;i<MuSD_triggers.size();i++)    { HistoInc->GetXaxis()->SetBinLabel(i+1,MuSD_triggers[i].c_str());   }
      for(unsigned int i=0;i<JetMetSD_triggers.size();i++){ HistoInc->GetXaxis()->SetBinLabel(MuSD_triggers.size()+1+i,JetMetSD_triggers[i].c_str());   }
//      HistoInc->GetXaxis()->SetBinLabel(numberofbins-2,"Mu Paths");
//      HistoInc->GetXaxis()->SetBinLabel(numberofbins-1,"JetMET Paths");
      HistoInc->GetXaxis()->SetBinLabel(numberofbins,"Total");

      Histo->Sumw2();
      HistoInc->Sumw2();

      BetaCount    = new TH1D((SignalName + "BetaMuCount").c_str() ,(SignalName + "BetaCount").c_str()   ,20,0,1);   BetaCount  ->Sumw2();
      BetaTotal    = new TH1D((SignalName + "BetaTotal"  ).c_str() ,(SignalName + "BetaTotal").c_str()   ,20,0,1);   BetaTotal  ->Sumw2();
      BetaMuon     = new TH1D((SignalName + "BetaMuon"   ).c_str() ,(SignalName + "BetaMuon" ).c_str()   ,20,0,1);   BetaMuon   ->Sumw2();
      BetaJet      = new TH1D((SignalName + "BetaJet"    ).c_str() ,(SignalName + "BetaJet"  ).c_str()   ,20,0,1);   BetaJet    ->Sumw2();
   }

};


void TriggerStudy_Core(string SignalName, FILE* pFile, stPlot* plot);
double FastestHSCP(const fwlite::ChainEvent& ev);
bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, double etaCut,int NObjectAboveThreshold, bool averageThreshold=false);
void layout(vector<stPlot*>& plots, vector<string>& sigs, string name);
int JobIdToIndex(string JobId);
void SetWeight(const double& IntegratedLuminosityInPb=-1, const double& IntegratedLuminosityInPbBeforeTriggerChange=-1, const double& CrossSection=0, const double& MCEvents=0, int period=0);


void SetWeight(const double& IntegratedLuminosityInPb, const double& IntegratedLuminosityInPbBeforeTriggerChange, const double& CrossSection, const double& MCEvents, int period){
  if(IntegratedLuminosityInPb>=IntegratedLuminosityInPbBeforeTriggerChange && IntegratedLuminosityInPb>0){
    double NMCEvents = MCEvents;
    if(MaxEntry>0)NMCEvents=std::min(MCEvents,(double)MaxEntry);
    if (period==0) Event_Weight = (CrossSection * IntegratedLuminosityInPbBeforeTriggerChange) / NMCEvents;
    else if (period==1)Event_Weight = (CrossSection * (IntegratedLuminosityInPb-IntegratedLuminosityInPbBeforeTriggerChange)) / NMCEvents;
  }else{
    Event_Weight=1;
  }
}

 

void TriggerStudy()
{
   system("mkdir pictures");

   setTDRStyle();
   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.10);
   gStyle->SetPadRightMargin (0.18);
   gStyle->SetPadLeftMargin  (0.14);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505,"X");

//   std::vector<stSignal> signals;
   GetSignalDefinition(signals);


   ///////////////////////////////////////////////////////
   JetMetSD_triggers.push_back("HLT_PFMHT150_v2");
//   JetMetSD_triggers.push_back("HLT_MET100_v1");

   MuSD_triggers.push_back("HLT_Mu40_eta2p1_v1");
//   MuSD_triggers.push_back("HLT_DoubleMu7_v1");

   All_triggers.clear();
   for(unsigned int i=0;i<MuSD_triggers.size();i++)All_triggers.push_back(MuSD_triggers[i]);
   for(unsigned int i=0;i<JetMetSD_triggers.size();i++)All_triggers.push_back(JetMetSD_triggers[i]);
   for(unsigned int i=0;i<All_triggers.size();i++)All_mask[All_triggers[i]] = true;
   ///////////////////////////////////////////////////////
   FILE* pFile = fopen("Results.txt","w");

   stPlot** plots = new stPlot*[signals.size()];  
   for(unsigned int i=0;i<signals.size();i++){
      plots[i] = new stPlot(signals[i].Name);
//      if(signals[i].Name!="GMStau100" && signals[i].Name!="GMStau200" && signals[i].Name!="GMStau308")continue;
      TriggerStudy_Core(signals[i].Name, pFile, plots[i]);
   }

   int Id;                              vector<stPlot*> objs;        vector<string> leg;

                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("Gluino300");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("Gluino600");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("Gluino1100");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   layout(objs, leg, "summary_Gluino");

                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("Gluino300S");     objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("Gluino600S");     objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("Gluino1100S");     objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   layout(objs, leg, "summary_GluinoS");


                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("GMStau100");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("GMStau200");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("GMStau308");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   layout(objs, leg, "summary_GMStau");

                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("GMStau100S");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("GMStau200S");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("GMStau308S");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   layout(objs, leg, "summary_GMStauS");



                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("PPStau100");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("PPStau200");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("PPStau308");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   layout(objs, leg, "summary_PPStau");

                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("PPStau100S");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("PPStau200S");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("PPStau308S");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   layout(objs, leg, "summary_PPStauS");



/*

                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("DCStau121");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("DCStau242");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("DCStau302");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   layout(objs, leg, "summary_DCStau");

   int Id;                              vector<stPlot*> objs;        vector<string> leg;

                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("Gluino300");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("Gluino500");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("Gluino800");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("Gluino900");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("Gluino1000");     objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   layout(objs, leg, "summary_Gluino");



                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("Gluino600");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
                                        objs.push_back(plotsG600Z2);leg.push_back("Gluino600 Z2");
   Id = JobIdToIndex("Stop300");        objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   Id = JobIdToIndex("GMStau126");      objs.push_back(plots[Id]);   leg.push_back(signals[Id].Name);
   layout(objs, leg, "summary_Mixed");
*/

   fflush(pFile);
   fclose(pFile);

}

void TriggerStudy_Core(string SignalName, FILE* pFile, stPlot* plot)
{

   double Total       = 0;
   double SDJetMET    = 0;
   double SDMu        = 0;
   double SDBoth      = 0;
   double SDJetMETInc = 0;
   double SDMuInc     = 0;
   double TrJetMET    = 0;
   double TrMu        = 0;
   double TrBoth      = 0;

   int MaxPrint = 0;
   for (int period=0; period<RunningPeriods; period++) {

   vector<string> fileNames;
   GetInputFiles(fileNames,SignalName, period);
   string thisname = fileNames[0];
   bool simhitshifted =0;
   if(thisname.find("S.",0)<std::string::npos ||thisname.find("SBX1.",0)<std::string::npos) simhitshifted=1 ;
//   cout<<thisname<<simhitshifted<<endl;

//fileNames.clear();
//      fileNames.push_back("/uscmst1b_scratch/lpc1/lpcphys/jchen/2011Runanalysis/aftereps/cls/CMSSW_4_2_8/src/SUSYBSMAnalysis/HSCP/test/BuildHSCParticles/ShiftSignals/HSCP.root");

   fwlite::ChainEvent ev(fileNames);

   int JobId = JobIdToIndex(SignalName);
   SetWeight(IntegratedLuminosity,IntegratedLuminosityBeforeTriggerChange,signals[JobId].XSec,(double)ev.size(), period);

   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on %10s        :", SignalName.c_str());
   int TreeStep = ev.size()/50;if(TreeStep==0)TreeStep=1;
   for(Long64_t e=0;e<ev.size();e++){
      if(e%TreeStep==0){printf(".");fflush(stdout);}
      if(MaxEntry>0 && e>MaxEntry)break;
      ev.to(e);
      edm::TriggerResultsByName tr = ev.triggerResultsByName("HLT"); 
      if(simhitshifted) tr= ev.triggerResultsByName("HLTSIMHITSHIFTER");
//      edm::TriggerResultsByName tr = ev.triggerResultsByName("HLT");      if(!tr.isValid())continue;
      //    for(unsigned int i=0;i<tr.size();i++){
      //   printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
      //}fflush(stdout);

      fwlite::Handle< trigger::TriggerEvent > trEvHandle;
      trEvHandle.getByLabel(ev,"hltTriggerSummaryAOD");
      trigger::TriggerEvent trEv = *trEvHandle;

      //for(unsigned int i=0;i<trEvHandle->sizeFilters();i++){
      //   if(strncmp(trEvHandle->filterTag(i).label().c_str(),"hltL1",5)==0)continue;
      //   printf("%i - %s\n",i,trEvHandle->filterTag(i).label().c_str());
      //}


      bool JetMetSD    = false;
      bool MuSD        = false;
      bool JetMetSDInc = false;
      bool MuSDInc     = false;
      bool JetMetTr    = false;
      bool MuTr        = false;


      unsigned int TrIndex_Unknown     = tr.size();

      bool AlreadyAccepted = false;

      for(unsigned int i=0;i<All_triggers.size();i++){
         vector<string>::iterator whereMuSD     = find(MuSD_triggers    .begin(), MuSD_triggers    .end(),All_triggers[i].c_str() );
         vector<string>::iterator whereJetMetSD = find(JetMetSD_triggers.begin(), JetMetSD_triggers.end(),All_triggers[i].c_str() );
  
 
         bool Accept = false;
         bool Accept2 = false;

           if(All_triggers[i]=="HLT_PFMHT150_v2"){
               if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMHT150_v2")){
                   if(e<MaxPrint)printf("HLT_PFMHT150_v2\n");
                   Accept = tr.accept(tr.triggerIndex("HLT_PFMHT150_v2"));
                }else{
                   if(e<MaxPrint)printf("HLT_PFMHT150_v1\n");
                   Accept = tr.accept(tr.triggerIndex("HLT_PFMHT150_v1"));
                }
               Accept2 = Accept;
               //Accept2 = IncreasedTreshold(trEv, InputTag("hltPFMHT150Filter","","HLT"),160 , 2.4, 1, false);

            }
           else if(All_triggers[i]=="HLT_Mu40_eta2p1_v1"){

              if(simhitshifted) Accept = IncreasedTreshold(trEv, InputTag("hltSingleMu30L3Filtered30","","HLTSIMHITSHIFTER"),40 , 2.1, 1, false);
              else  Accept = IncreasedTreshold(trEv, InputTag("hltSingleMu30L3Filtered30","","HLT"),40 , 2.1, 1, false);              
              Accept2 = Accept;
           }
           else{
               Accept = tr.accept(All_triggers[i].c_str());
               Accept2 = Accept;
            }

         if(Accept                    ){plot->Histo   ->Fill(All_triggers[i].c_str(),Event_Weight);}       
         if(Accept && !AlreadyAccepted){plot->HistoInc->Fill(All_triggers[i].c_str(),Event_Weight);}

         if     (whereJetMetSD!=JetMetSD_triggers.end()){ JetMetSD |= Accept; if(!AlreadyAccepted)JetMetSDInc |= Accept;}
         else if(whereMuSD    !=MuSD_triggers.end())    { MuSD     |= Accept; if(!AlreadyAccepted)MuSDInc     |= Accept;}

         if     (whereJetMetSD!=JetMetSD_triggers.end()){ JetMetTr |= Accept2; }
         else if(whereMuSD    !=MuSD_triggers.end())    { MuTr     |= Accept2; }


         AlreadyAccepted |= Accept;
      }       
      fflush(stdout);


      if(JetMetSD||MuSD){
         plot->Histo->Fill("Total",Event_Weight);
         plot->HistoInc->Fill("Total",Event_Weight);
      }

//      JetMetTr = JetMetSD & ((rand()%100)<90);
//      MuTr     = MuSD     & ((rand()%100)<90);  

      Total+=Event_Weight;
      if(JetMetSD)SDJetMET+=Event_Weight;
      if(MuSD)SDMu+=Event_Weight;
      if(JetMetSDInc)SDJetMETInc+=Event_Weight;
      if(MuSDInc)SDMuInc+=Event_Weight;
      if(JetMetSD||MuSD)SDBoth+=Event_Weight;
      if(JetMetTr)TrJetMET+=Event_Weight;
      if(MuTr)TrMu+=Event_Weight;
      if(JetMetTr||MuTr)TrBoth+=Event_Weight;

      double Beta = 1.0;
      if(SignalName!="Data")Beta = FastestHSCP(ev);
      plot->BetaCount->Fill(Beta,Event_Weight);
      if(MuSD||JetMetSD)plot->BetaTotal->Fill(Beta,Event_Weight);
      if(MuSD)plot->BetaMuon->Fill(Beta,Event_Weight);
      if(JetMetSD)plot->BetaJet->Fill(Beta,Event_Weight);

   }printf("\n");
   }

//   fprintf(pFile,  "%15s --> JetMET = %5.2f%% (was %5.2f%%) Mu = %5.2f%% (was %5.2f%%) JetMET||Mu = %5.2f%% (%5.2f%%)\n",SignalName.c_str(), (100.0*TrJetMET)/Total, (100.0*SDJetMET)/Total, (100.0*TrMu)/Total, (100.0*SDMu)/Total, (100.0*TrBoth)/Total, (100.0*SDBoth)/Total);
//   fprintf(stdout, "%15s --> JetMET = %5.2f%% (was %5.2f%%) Mu = %5.2f%% (was %5.2f%%) JetMET||Mu = %5.2f%% (%5.2f%%)\n",SignalName.c_str(), (100.0*TrJetMET)/Total, (100.0*SDJetMET)/Total, (100.0*TrMu)/Total, (100.0*SDMu)/Total, (100.0*TrBoth)/Total, (100.0*SDBoth)/Total);


   fprintf(pFile,  "%15s --> MET = %5.2f%% (modified %5.2f%%) Mu = %5.2f%% (modified %5.2f%%) JetMET||Mu = %5.2f%% (%5.2f%%)\n",SignalName.c_str(), (100.0*SDJetMET)/Total, (100.0*TrJetMET)/Total, (100.0*SDMu)/Total, (100.0*TrMu)/Total, (100.0*SDBoth)/Total, (100.0*TrBoth)/Total);
   fprintf(stdout, "%15s --> MET = %5.2f%% (modified %5.2f%%) Mu = %5.2f%% (modified %5.2f%%) JetMET||Mu = %5.2f%% (%5.2f%%)\n",SignalName.c_str(), (100.0*SDJetMET)/Total, (100.0*TrJetMET)/Total, (100.0*SDMu)/Total, (100.0*TrMu)/Total, (100.0*SDBoth)/Total, (100.0*TrBoth)/Total);



//   printf("Total %i \n",Total);
   plot->Histo->SetStats(0)  ;
   plot->Histo->LabelsOption("v");
   plot->Histo->Scale(100./Total);


   plot->HistoInc->SetStats(0)  ;
   plot->HistoInc->LabelsOption("v");
   plot->HistoInc->Scale(100./Total);


   plot->BetaTotal->Divide(plot->BetaCount);
   plot->BetaMuon ->Divide(plot->BetaCount);
   plot->BetaJet  ->Divide(plot->BetaCount);

   plot->BetaTotal->Scale(100.0);
   plot->BetaMuon ->Scale(100.0);
   plot->BetaJet  ->Scale(100.0);

   TH1** Histos = new TH1*[10];
   std::vector<string> legend;
   TCanvas* c1;
   
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = (TH1*)plot->BetaMuon;                    legend.push_back("Muon");
   Histos[1] = (TH1*)plot->BetaTotal;                   legend.push_back("Overall");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "#beta of the fastest HSCP", "Trigger Efficiency (%)", 0,1, 0,100);
   DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary(-1);
   SaveCanvas(c1,"pictures/",SignalName);
   delete c1;
}

void layout(vector<stPlot*>& plots, vector<string>& sigs, string name){
   unsigned int NPath   = 0+3;

   std::vector<string> legend;
   TObject** Histos1 = new TObject*[plots.size()];


//   TLine* line1 = new TLine(plots[0]->Histo->GetBinLowEdge(NPath+1), 0, plots[0]->Histo->GetBinLowEdge(NPath+1), 100);
//   line1->SetLineWidth(2); line1->SetLineStyle(1);
//   TLine* line2 = new TLine(plots[0]->Histo->GetBinLowEdge(NPath+3), 0, plots[0]->Histo->GetBinLowEdge(NPath+3), 100);
//   line2->SetLineWidth(2); line2->SetLineStyle(1);

   TCanvas* c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
   c1->SetGrid();
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<plots.size();i++){
      Histos1[i]=plots[i]->Histo; legend.push_back(sigs[i]);
   }
//   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);  
   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,30);
   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.98, 0.90, 0.13, 0.07);
   DrawPreliminary(-1);

   for(unsigned int i=0;i<plots.size();i++){
      plots[i]->Histo->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->Histo->SetMarkerSize(0.8);
   }
//   line1->Draw();
//   line2->Draw();
   SaveCanvas(c1,"pictures/",name);
   delete c1;

   c1 = new TCanvas("MyC","Histo",600,600);
   legend.clear();
   c1->SetGrid();
   c1->SetBottomMargin(0.3);

   for(unsigned int i=0;i<plots.size();i++){
      Histos1[i]=plots[i]->HistoInc; legend.push_back(sigs[i]);
   }
   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,30);
   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.98, 0.90, 0.13, 0.07);
   DrawPreliminary(-1);

   for(unsigned int i=0;i<plots.size();i++){
      plots[i]->HistoInc->GetYaxis()->SetTitleOffset(1.55);
      plots[i]->HistoInc->SetMarkerSize(0.8);
   }

//   line1->Draw();
//   line2->Draw();
   SaveCanvas(c1,"pictures/",name + "_inc");
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
      if(AbsPdg<1000000)continue;    

      double beta=genColl[g].p()/genColl[g].energy();
      if(MaxBeta<beta)MaxBeta=beta;
   }
   return MaxBeta;
}

bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, double etaCut, int NObjectAboveThreshold, bool averageThreshold)
{
   unsigned int filterIndex = trEv.filterIndex(InputPath);
   //if(filterIndex<trEv.sizeFilters())printf("SELECTED INDEX =%i --> %s    XXX   %s\n",filterIndex,trEv.filterTag(filterIndex).label().c_str(), trEv.filterTag(filterIndex).process().c_str());

   if (filterIndex<trEv.sizeFilters()){
      const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
      const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
      const int nI(VIDS.size());
      const int nK(KEYS.size());
      assert(nI==nK);
      const int n(std::max(nI,nK));
      const trigger::TriggerObjectCollection& TOC(trEv.getObjects());


      if(!averageThreshold){
         int NObjectAboveThresholdObserved = 0;
         for (int i=0; i!=n; ++i) {
            if(TOC[KEYS[i]].pt()> NewThreshold && fabs(TOC[KEYS[i]].eta())<etaCut) NObjectAboveThresholdObserved++;
            //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "<< TOC[KEYS[i]].id() << " " << TOC[KEYS[i]].pt() << " " << TOC[KEYS[i]].eta() << " " << TOC[KEYS[i]].phi() << " " << TOC[KEYS[i]].mass()<< endl;
         }
         if(NObjectAboveThresholdObserved>=NObjectAboveThreshold)return true;

      }else{
         std::vector<double> ObjPt;

         for (int i=0; i!=n; ++i) {
            ObjPt.push_back(TOC[KEYS[i]].pt());
            //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "<< TOC[KEYS[i]].id() << " " << TOC[KEYS[i]].pt() << " " << TOC[KEYS[i]].eta() << " " << TOC[KEYS[i]].phi() << " " << TOC[KEYS[i]].mass()<< endl;
         }
         if((int)(ObjPt.size())<NObjectAboveThreshold)return false;
         std::sort(ObjPt.begin(), ObjPt.end());

         double Average = 0;
         for(int i=0; i<NObjectAboveThreshold;i++){
            Average+= ObjPt[ObjPt.size()-1-i];
         }Average/=NObjectAboveThreshold;
         //cout << "AVERAGE = " << Average << endl;

         if(Average>NewThreshold)return true;
      }
   }
   return false;
}

int JobIdToIndex(string JobId){
   for(unsigned int s=0;s<signals.size();s++){
      if(signals[s].Name==JobId)return s;
   }return -1;
}


