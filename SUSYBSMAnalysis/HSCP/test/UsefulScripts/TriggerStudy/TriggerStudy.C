
namespace reco    { class Vertex; class Track; class GenParticle; class DeDxData; class MuonTimeExtra; class PFMET; class HitPattern;}
namespace susybsm { class HSCParticle; class HSCPIsolation; class MuonSegment;}
namespace fwlite  { class ChainEvent;}
namespace trigger { class TriggerEvent;}
namespace edm     { class TriggerResults; class TriggerResultsByName; class InputTag; class LumiReWeighting;}
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



std::vector<stSample> samples;
vector<std::pair<string,string>> JetMetSD_triggers;
vector<std::pair<string,string>> MuSD_triggers;
vector<std::pair<string,string>> OthersSD_triggers;
vector<std::pair<string,string>> All_triggers;
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
      int numberofbins=All_triggers.size()+1;
      Histo    = new TH1D((SignalName + "Abs").c_str(),(SignalName + "Abs").c_str(),numberofbins,0,numberofbins);
      HistoInc = new TH1D((SignalName + "Inc").c_str(),(SignalName + "Inc").c_str(),numberofbins,0,numberofbins);

      for(unsigned int i=0;i<All_triggers.size();i++)    { Histo->GetXaxis()->SetBinLabel(i+1,All_triggers[i].second.c_str());   }
      Histo->GetXaxis()->SetBinLabel(numberofbins,"Total");

      for(unsigned int i=0;i<All_triggers.size();i++)    { HistoInc->GetXaxis()->SetBinLabel(i+1,All_triggers[i].second.c_str());   }
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
//bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, double etaCut,int NObjectAboveThreshold, bool averageThreshold=false);
void layout(vector<stPlot*>& plots, vector<string>& sigs, string name);

void TriggerStudy()
{
   system("mkdir pictures");

   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.14);
   gStyle->SetPadRightMargin (0.16);
   gStyle->SetPadLeftMargin  (0.14);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.45);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505);


   InitBaseDirectory();
   GetSampleDefinition(samples, "../../ICHEP_Analysis/Analysis_Samples.txt");
   keepOnlySamplesAt7and8TeVX(samples, SQRTS);

   ///////////////////////////////////////////////////////

   MuSD_triggers.push_back(std::make_pair("HSCPHLTTriggerMuFilter", "Mu40_eta2p1"));
   JetMetSD_triggers.push_back(std::make_pair("HSCPHLTTriggerPFMetFilter","PFMET150"));

   OthersSD_triggers.push_back(std::make_pair("HSCPHLTTriggerL2MuFilter", "L2Muon+Met"));
   OthersSD_triggers.push_back(std::make_pair("HSCPHLTTriggerMetDeDxFilter", "Met80+dEdx"));
   OthersSD_triggers.push_back(std::make_pair("HSCPHLTTriggerMuDeDxFilter", "Mu40+dEdx"));
   OthersSD_triggers.push_back(std::make_pair("HSCPHLTTriggerHtDeDxFilter", "HT+dEdx"));
   OthersSD_triggers.push_back(std::make_pair("HSCPHLTTriggerHtFilter", "HT650" ) );
//   OthersSD_triggers.push_back("HSCPHLTTriggerMetFilter");


   All_triggers.clear();
   for(unsigned int i=0;i<MuSD_triggers.size();i++)All_triggers.push_back(MuSD_triggers[i]);
   for(unsigned int i=0;i<JetMetSD_triggers.size();i++)All_triggers.push_back(JetMetSD_triggers[i]);
   for(unsigned int i=0;i<OthersSD_triggers.size();i++)All_triggers.push_back(OthersSD_triggers[i]);
   for(unsigned int i=0;i<All_triggers.size();i++)All_mask[All_triggers[i].first] = true;
   ///////////////////////////////////////////////////////

   FILE* pFile = fopen("Results.txt","w");

   stPlot** plots = new stPlot*[samples.size()];  
   for(unsigned int i=0;i<samples.size();i++){
      if(samples[i].Type!=2)continue;
      if(samples[i].Name != "Gluino_8TeV_M300_f10" && samples[i].Name != "Gluino_8TeV_M600_f10" && samples[i].Name != "Gluino_8TeV_M1100_f10"
      && samples[i].Name != "GMStau_8TeV_M100"     && samples[i].Name != "GMStau_8TeV_M200"     && samples[i].Name != "GMStau_8TeV_M308"     
      && samples[i].Name != "PPStau_8TeV_M100"     && samples[i].Name != "PPStau_8TeV_M200"     && samples[i].Name != "PPStau_8TeV_M308"
      && samples[i].Name != "DY_8TeV_M100_Q1o3"    && samples[i].Name != "DY_8TeV_M600_Q1o3"    && samples[i].Name != "DY_8TeV_M100_Q2o3"     && samples[i].Name != "DY_8TeV_M600_Q2o3"    
      && samples[i].Name != "DY_8TeV_M100_Q2"      && samples[i].Name != "DY_8TeV_M600_Q2"      && samples[i].Name != "DY_8TeV_M100_Q5"       && samples[i].Name != "DY_8TeV_M600_Q5" )continue;
      plots[i] = new stPlot(samples[i].Name);
      TriggerStudy_Core(samples[i].Name, pFile, plots[i]);
   }
   fflush(pFile);
   fclose(pFile);


   int Id;                                                  vector<stPlot*> objs;        vector<string> leg;

                                                            objs.clear();                leg.clear();
   Id = JobIdToIndex("Gluino_8TeV_M300_f10", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("Gluino_8TeV_M600_f10", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("Gluino_8TeV_M1100_f10",samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   layout(objs, leg, "summary_8TeV_Gluino");

                                                         objs.clear();                leg.clear();
   Id = JobIdToIndex("GMStau_8TeV_M100", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("GMStau_8TeV_M200", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("GMStau_8TeV_M308", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   layout(objs, leg, "summary_8TeV_GMStau");

                                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("PPStau_8TeV_M100", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("PPStau_8TeV_M200", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("PPStau_8TeV_M308", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   layout(objs, leg, "summary_8TeV_PPStau");

                                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("DY_8TeV_M100_Q1o3", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("DY_8TeV_M600_Q1o3", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("DY_8TeV_M100_Q2o3", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("DY_8TeV_M600_Q2o3", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   layout(objs, leg, "summary_8TeV_DYLQ");

                                                        objs.clear();                leg.clear();
   Id = JobIdToIndex("DY_8TeV_M100_Q2", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("DY_8TeV_M600_Q2", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("DY_8TeV_M100_Q5", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   Id = JobIdToIndex("DY_8TeV_M600_Q5", samples);      objs.push_back(plots[Id]);   leg.push_back(samples[Id].Legend);
   layout(objs, leg, "summary_8TeV_DYHQ");

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

   int JobId = JobIdToIndex(SignalName, samples);


   vector<string> fileNames;
   GetInputFiles(samples[JobId], BaseDirectory, fileNames, period);

   string thisname = fileNames[0];
   bool simhitshifted =0;
   if(thisname.find("S.",0)<std::string::npos ||thisname.find("SBX1.",0)<std::string::npos) simhitshifted=1 ;
//   cout<<thisname<<simhitshifted<<endl;

//fileNames.clear();
//      fileNames.push_back("/uscmst1b_scratch/lpc1/lpcphys/jchen/2011Runanalysis/aftereps/cls/CMSSW_4_2_8/src/SUSYBSMAnalysis/HSCP/test/BuildHSCParticles/ShiftSignals/HSCP.root");

   fwlite::ChainEvent ev(fileNames);

   //double SampleWeight = GetSampleWeigh(IntegratedLuminosity,IntegratedLuminosityBeforeTriggerChange,samples[JobId].XSec,0, period);

   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on %10s        :", SignalName.c_str());
   int TreeStep = ev.size()/50;if(TreeStep==0)TreeStep=1;
   int MaxEvent = 100; 
   if(MaxEvent<0 || MaxEvent>ev.size())MaxEvent = ev.size();
   for(Long64_t e=0;e<MaxEvent;e++){
      if(e%TreeStep==0){printf(".");fflush(stdout);}
      ev.to(e);

      double Event_Weight = 1.0;//SampleWeight * GetPUWeight(ev, samples[s].Pileup, PUSystFactor, LumiWeightsMC, PShift);


      edm::TriggerResultsByName tr = ev.triggerResultsByName("MergeHLT"); 
//      if(simhitshifted) tr= ev.triggerResultsByName("HLTSIMHITSHIFTER");
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
      bool OtherSD     = false;
      bool JetMetSDInc = false;
      bool MuSDInc     = false;
      bool OtherSDInc  = false;
      bool JetMetTr    = false;
      bool MuTr        = false;
      bool OtherTr     = false;


      unsigned int TrIndex_Unknown     = tr.size();

      bool AlreadyAccepted = false;

      for(unsigned int i=0;i<All_triggers.size();i++){
         vector<std::pair<string,string> >::iterator whereMuSD     = find(MuSD_triggers    .begin(), MuSD_triggers    .end(),All_triggers[i] );
         vector<std::pair<string,string> >::iterator whereJetMetSD = find(JetMetSD_triggers.begin(), JetMetSD_triggers.end(),All_triggers[i] );
         vector<std::pair<string,string> >::iterator whereOtherSD  = find(OthersSD_triggers.begin(), OthersSD_triggers.end(),All_triggers[i] );

 
         bool Accept = false;
         bool Accept2 = false;

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
           else{*/
               Accept = tr.accept(All_triggers[i].first.c_str());
               Accept2 = Accept;
//            }

         if(Accept                    ){plot->Histo   ->Fill(All_triggers[i].second.c_str(),Event_Weight);}       
         if(Accept && !AlreadyAccepted){plot->HistoInc->Fill(All_triggers[i].second.c_str(),Event_Weight);}

         if     (whereJetMetSD!=JetMetSD_triggers.end()){ JetMetSD |= Accept; if(!AlreadyAccepted)JetMetSDInc |= Accept;}
         else if(whereMuSD    !=MuSD_triggers.end())    { MuSD     |= Accept; if(!AlreadyAccepted)MuSDInc     |= Accept;}
         else if(whereOtherSD !=OthersSD_triggers.end()){ OtherSD  |= Accept; if(!AlreadyAccepted)OtherSDInc  |= Accept;}


         if     (whereJetMetSD!=JetMetSD_triggers.end()){ JetMetTr |= Accept2; }
         else if(whereMuSD    !=MuSD_triggers.end())    { MuTr     |= Accept2; }
         else if(whereOtherSD !=OthersSD_triggers.end()) { OtherTr  |= Accept2; }

         AlreadyAccepted |= Accept;
      }       
      fflush(stdout);


      if(JetMetSD||MuSD||OtherSD){
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
   Histos[0] = (TH1*)plot->BetaMuon;                    legend.push_back("Mu triggers");
   Histos[1] = (TH1*)plot->BetaTotal;                   legend.push_back("Mu+Met triggers");
   DrawSuperposedHistos((TH1**)Histos, legend, "HIST E1",  "#beta of the fastest HSCP", "Trigger Efficiency (%)", 0,1, 0,100);
   DrawLegend((TObject**)Histos,legend,"Trigger:","LP",0.35, 0.93, 0.18, 0.04);
   c1->Modified();
   DrawPreliminary("Simulation", SQRTS, -1);
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
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   DrawPreliminary("Simulation", SQRTS, -1);

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
//   if(name=="summary_Gluino")DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,30);
//   else                      DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);
   DrawSuperposedHistos((TH1**)Histos1, legend, "E1",  "", "Incremental Efficiency (%)", 0,0, 0,100);
   DrawLegend(Histos1,legend,"","P", 0.58, 0.90, 0.13, 0.07);
   DrawPreliminary("Simulation", SQRTS, -1);
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

/*
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
*/

