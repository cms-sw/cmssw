// Original Author:  Loic Quertenmont


namespace reco    { class Vertex; class Track; class GenParticle; class DeDxData; class MuonTimeExtra; class PFMET; class HitPattern;}
namespace susybsm { class HSCParticle; class HSCPIsolation; class MuonSegment; class HSCPDeDxInfo;}
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
#include "Analysis_Global.h"
#include "Analysis_CommonFunction.h"
#include "Analysis_PlotFunction.h"
#include "Analysis_PlotStructure.h"
#include "Analysis_Samples.h"
#include "tdrstyle.C"

/////////////////////////// FUNCTION DECLARATION /////////////////////////////

void InitHistos(stPlots* st=NULL);
void Analysis_Step3(char* SavePath);

bool PassTrigger(const fwlite::ChainEvent& ev, bool isData, bool isCosmic=false);
bool   PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData* dedxSObj, const reco::DeDxData* dedxMObj, const reco::MuonTimeExtra* tof, const reco::MuonTimeExtra* dttof, const reco::MuonTimeExtra* csctof, const fwlite::ChainEvent& ev, stPlots* st=NULL, const double& GenBeta=-1, bool RescaleP=false, const double& RescaleI=0.0, const double& RescaleT=0.0);
bool PassSelection(const susybsm::HSCParticle& hscp,  const reco::DeDxData* dedxSObj, const reco::DeDxData* dedxMObj, const reco::MuonTimeExtra* tof, const fwlite::ChainEvent& ev, const int& CutIndex=0, stPlots* st=NULL, const bool isFlip=false, const double& GenBeta=-1, bool RescaleP=false, const double& RescaleI=0.0, const double& RescaleT=0.0);
void Analysis_FillControlAndPredictionHist(const susybsm::HSCParticle& hscp, const reco::DeDxData* dedxSObj, const reco::DeDxData* dedxMObj, const reco::MuonTimeExtra* tof, stPlots* st=NULL);
double SegSep(const susybsm::HSCParticle& hscp, const fwlite::ChainEvent& ev, double& minPhi, double& minEta);
double RescaledPt(const double& pt, const double& eta, const double& phi, const int& charge);
int  muonStations(reco::HitPattern hitPattern);
double scaleFactor(double eta);
/////////////////////////// VARIABLE DECLARATION /////////////////////////////

float Event_Weight = 1;
int   MaxEntry = -1;

TFile* HistoFile;

std::vector<double>  CutPt ;
std::vector<double>  CutI  ;
std::vector<double>  CutTOF;

TProfile*  HCuts_Pt;
TProfile*  HCuts_I;
TProfile*  HCuts_TOF;

//The cuts used for the check on the background prediction by tracks with TOF<1
std::vector<double>  CutPt_Flip ;
std::vector<double>  CutI_Flip  ;
std::vector<double>  CutTOF_Flip;

TProfile*  HCuts_Pt_Flip;
TProfile*  HCuts_I_Flip;
TProfile*  HCuts_TOF_Flip;

std::vector<stSample> samples;
std::map<std::string, stPlots> plotsMap;

std::vector< float > BgLumiMC; //MC                                           
std::vector< float > TrueDist;                                    
edm::LumiReWeighting LumiWeightsMC;
reweight::PoissonMeanShifter PShift(0.6);//0.6 for upshift, -0.6 for downshift


TH3F* dEdxTemplates = NULL;

/////////////////////////// CODE PARAMETERS /////////////////////////////

void Analysis_Step3(string MODE="COMPILE", int TypeMode_=0, string dEdxSel_=dEdxS_Label, string dEdxMass_=dEdxM_Label, string TOF_Label_=TOF_Label, double CutPt_=-1.0, double CutI_=-1, double CutTOF_=-1, float MinPt_=GlobalMinPt, float MaxEta_=GlobalMaxEta, float MaxDZ_=GlobalMaxDZ, float MaxDXY_=GlobalMaxDXY)
{
   if(MODE=="COMPILE")return;

   //setup ROOT global variables (mostly cosmetic and histo in file treatment)
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.05);
   gStyle->SetPadBottomMargin(0.10);
   gStyle->SetPadRightMargin (0.18);
   gStyle->SetPadLeftMargin  (0.13);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505);
   TH1::AddDirectory(kTRUE);

   // redefine global variable dependent on the arguments given to the function
   dEdxS_Label    = dEdxSel_;
   dEdxM_Label    = dEdxMass_;
   TOF_Label      = TOF_Label_;
   InitdEdx(dEdxS_Label);
   TypeMode       = TypeMode_;
   GlobalMaxEta   = MaxEta_;
   GlobalMinPt    = MinPt_;
   GlobalMaxDZ    = MaxDZ_;
   GlobalMaxDXY    = MaxDXY_;

   if(TypeMode<2){      GlobalMinNDOF   = 0; 
                         GlobalMinTOF    = 0;
   }else if(TypeMode==2) { //GlobalMaxTIsol *= 2;
                          // GlobalMaxEIsol *= 2;
   }else if(TypeMode==3){
     GlobalMaxV3D     =  999999;
     GlobalMinIs      =   -1;
     IPbound=150;
     PredBins=6;
     //SA Muon trigger only existed for part of 2011 running
#ifdef ANALYSIS2011
     IntegratedLuminosityBeforeTriggerChange = 0;
     IntegratedLuminosity = 4100;
#endif
   }else if(TypeMode==4){
//         GlobalMaxTIsol   =  999999;      // cut on tracker isolation (SumPt)
//         GlobalMaxRelTIsol   =  0.10; // cut on relative tracker isolation (SumPt/Pt)
         GlobalMaxEIsol   =  999999;   // cut on calorimeter isolation (E/P)
   } else if(TypeMode==5){
     IPbound=4.5;
     GlobalMinIm   = 2.8; //is actually dEdx max at skim level (reverse logic for type5)
     GlobalMinNDOF = 0; //tkOnly analysis --> comment these 2 lines to use only global muon tracks
     GlobalMinTOF  = 0;
   }
   
   // define the selection to be considered later for the optimization
   // WARNING: recall that this has a huge impact on the analysis time AND on the output file size --> be carefull with your choice
   CutPt .push_back(GlobalMinPt);   CutI  .push_back(GlobalMinIs);  CutTOF.push_back(GlobalMinTOF);
   CutPt_Flip .push_back(GlobalMinPt);   CutI_Flip  .push_back(GlobalMinIs);  CutTOF_Flip.push_back(GlobalMinTOF);

   if(TypeMode<2){   
      for(double Pt =GlobalMinPt+5 ; Pt <200;Pt+=5){
      for(double I  =GlobalMinIs+0.025  ; I  <0.45 ;I+=0.025){
         CutPt .push_back(Pt);   CutI  .push_back(I);  CutTOF.push_back(-1);
      }}
   }else if(TypeMode==2){
      for(double Pt =GlobalMinPt+5 ; Pt <120;  Pt+=5){
      if(Pt>80 && ((int)Pt)%10!=0)continue;
      for(double I  =GlobalMinIs +0.025; I  <0.40;  I+=0.025){
      for(double TOF=GlobalMinTOF+0.025; TOF<1.35;TOF+=0.025){
	   CutPt .push_back(Pt);   CutI  .push_back(I);  CutTOF.push_back(TOF);
      }}}
      for(double Pt =GlobalMinPt+10 ; Pt <90;  Pt+=30){
      for(double I  =GlobalMinIs +0.1; I  <0.30;  I+=0.1){
      for(double TOF=GlobalMinTOF-0.05; TOF>0.65;TOF-=0.05){
	   CutPt_Flip .push_back(Pt);   CutI_Flip  .push_back(I);  CutTOF_Flip.push_back(TOF);
      }}}
   }else if(TypeMode==3){
      for(double Pt =GlobalMinPt+30 ; Pt <450;  Pt+=30){
      for(double TOF=GlobalMinTOF+0.025; TOF<1.5;TOF+=0.025){
         CutPt .push_back(Pt);   CutI  .push_back(-1);  CutTOF.push_back(TOF);
      }}
      for(double Pt =GlobalMinPt+30 ; Pt <450;  Pt+=60){
      for(double TOF=GlobalMinTOF-0.025; TOF>0.5;TOF-=0.025){
         CutPt_Flip .push_back(Pt);   CutI_Flip  .push_back(-1);  CutTOF_Flip.push_back(TOF);
      }}
   }else if(TypeMode==4){
     for(double I  =GlobalMinIs +0.025; I  <0.45;  I+=0.025){
       for(double TOF=GlobalMinTOF+0.025; TOF<1.46;TOF+=0.025){
 	 CutPt .push_back(-1);   CutI  .push_back(I);  CutTOF.push_back(TOF);
       }}
      for(double I  =GlobalMinIs +0.025; I  <0.45;  I+=0.025){
      for(double TOF=GlobalMinTOF-0.025; TOF>0.54;TOF-=0.025){
	 CutPt_Flip .push_back(-1);   CutI_Flip  .push_back(I);  CutTOF_Flip.push_back(TOF);
       }}
   }else if(TypeMode==5){   
      for(double Pt =75 ; Pt <=150;Pt+=25){
      for(double I  =0.0; I  <=0.45 ;I+=0.025){
//         if(I<0.85 && int(I*1000)%5!=0)continue;
         CutPt     .push_back(Pt);   CutI     .push_back(I);  CutTOF     .push_back(-1);
         CutPt_Flip.push_back(Pt);   CutI_Flip.push_back(I);  CutTOF_Flip.push_back(-1);
     }}
   }

   printf("%i Different Final Selection will be tested\n",(int)CutPt.size());
   printf("%i Different Final Selection will be tested for background uncertainty\n",(int)CutPt_Flip.size());
   //   for (int CutIndex = 0; CutIndex < CutPt.size(); ++CutIndex)      printf("%4.0i  %3.0f   %3.3f   %3.3f\n", CutIndex+1, CutPt[CutIndex], CutI[CutIndex],  CutTOF[CutIndex]);
   //   for (int CutIndex = 0; CutIndex < CutPt_Flip.size(); ++CutIndex) printf("%4.0i  %3.0f   %3.3f   %3.3f\n", CutIndex+1, CutPt_Flip[CutIndex], CutI_Flip[CutIndex],  CutTOF_Flip[CutIndex]);

   //initialize LumiReWeighting
#ifdef ANALYSIS2011
   for(int i=0; i<35; ++i) BgLumiMC.push_back(Pileup_MC_Fall11[i]);
   for(int i=0; i<35; ++i) TrueDist.push_back(TrueDist2011_f[i]);
#else
   for(int i=0; i<60; ++i) BgLumiMC.push_back(Pileup_MC_Summer2012[i]);
   for(int i=0; i<60; ++i) TrueDist.push_back(TrueDist2012_f[i]);
#endif
   LumiWeightsMC = edm::LumiReWeighting(BgLumiMC, TrueDist);

   //make the directory structure corresponding to this analysis (depends on dEdx/TOF estimator being used, Eta/Pt cuts and Mode of the analysis)
   char Buffer[2048], Command[2048];
   //sprintf(Buffer,"Results/%s/%s/Eta%02.0f/PtMin%02.0f/Type%i/", dEdxS_Label.c_str(), TOF_Label.c_str(), 10.0*GlobalMaxEta, GlobalMinPt, TypeMode);
   sprintf(Buffer,"Results/Type%i/", TypeMode);
   sprintf(Command,"mkdir -p %s",Buffer); system(Command);

   // get all the samples and clean the list to keep only the one we want to run on... Also initialize the BaseDirectory
   InitBaseDirectory();
   GetSampleDefinition(samples);
   if(MODE.find("ANALYSE_")==0){
      int sampleIdStart, sampleIdEnd; sscanf(MODE.c_str(),"ANALYSE_%d_to_%d",&sampleIdStart, &sampleIdEnd);
      keepOnlyTheXtoYSamples(samples,sampleIdStart,sampleIdEnd);
      printf("----------------------------------------------------------------------------------------------------------------------------------------------------\n");
      printf("Run on the following samples:\n");
      for(unsigned int s=0;s<samples.size();s++){samples[s].print();}
      printf("----------------------------------------------------------------------------------------------------------------------------------------------------\n\n");
   }else{
      printf("You must select a MODE:\n");
      printf("MODE='ANALYSE_X_to_Y'   : Will run the analysis on the samples with index in the range [X,Y]\n"); 
      return;
   }

   //create histogram file and run the analyis
   HistoFile = new TFile((string(Buffer)+"/Histos_"+samples[0].Name+"_"+samples[0].FileName+".root").c_str(),"RECREATE");
   Analysis_Step3(Buffer);
   HistoFile->Write();
   HistoFile->Close();
   return;
}


// check if the event is passing trigger or not --> note that the function has two part (one for 2011 analysis and the other one for 2012)
bool PassTrigger(const fwlite::ChainEvent& ev, bool isData, bool isCosmic)
{
      edm::TriggerResultsByName tr = ev.triggerResultsByName("MergeHLT");
      if(!tr.isValid())return false;

      #ifdef ANALYSIS2011
          if(tr.accept(tr.triggerIndex("HSCPHLTTriggerMuFilter")))return true;
          else if(tr.accept(tr.triggerIndex("HSCPHLTTriggerPFMetFilter"))){
             if(!isData) Event_Weight=Event_Weight*0.96;
             return true;
          }
	  if(TypeMode==3) {
	  if(tr.size()== tr.triggerIndex("HSCPHLTTriggerL2MuFilter")) return false;
	  if(tr.accept(tr.triggerIndex("HSCPHLTTriggerL2MuFilter"))) return true;
	  }
      #else
//	   if(tr.accept("HSCPHLTTriggerMetDeDxFilter"))return true;
//	   if(tr.accept("HSCPHLTTriggerMuDeDxFilter"))return true;
	   if(tr.accept("HSCPHLTTriggerMuFilter"))return true;
           if(tr.accept("HSCPHLTTriggerPFMetFilter"))return true;

	   //Could probably use this trigger for the other analyses as well
	   if(TypeMode==3) {
           if(tr.size()== tr.triggerIndex("HSCPHLTTriggerL2MuFilter")) return false;
	   if(tr.accept(tr.triggerIndex("HSCPHLTTriggerL2MuFilter"))) {
	     if(!isData) {
	       //First 700 pb-1 taken with MET threshold at 65, for events with 55 < MET < 65 correct weight for luminosity actually taken
	       fwlite::Handle< trigger::TriggerEvent > trEvHandle;
	       trEvHandle.getByLabel(ev, "hltTriggerSummaryAOD");
	       trigger::TriggerEvent trEv = *trEvHandle;
	       if(!IncreasedTreshold(trEv, InputTag("hltPFMHT55Filter","","HLT"), 65, 99999., 1, false)) {
		 Event_Weight = Event_Weight * (IntegratedLuminosity - IntegratedLuminosityHigherMETThreshold)/IntegratedLuminosity;}
	     }
	     return true;
	   }

	 //Only accepted if looking for cosmic events
	 if(isCosmic) {
           if(tr.size()== tr.triggerIndex("HSCPHLTTriggerCosmicFilter")) return false;
	   if(tr.accept(tr.triggerIndex("HSCPHLTTriggerCosmicFilter"))) return true;
	 }
	   }
      #endif
      return false;
}

// check if one HSCP candidate is passing the preselection (the function also has many more arguments because it is used to fill some histograms AND to evaluate the systematics
double OpenAngle = -1; //global variable needed by PassPreselection... Ugly isn't it?!
double TreeDXY = -1;
double TreeDZ = -1;
bool isCosmicSB = false;
bool isSemiCosmicSB = false;
bool PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData* dedxSObj, const reco::DeDxData* dedxMObj, const reco::MuonTimeExtra* tof, const reco::MuonTimeExtra* dttof, const reco::MuonTimeExtra* csctof, const fwlite::ChainEvent& ev, stPlots* st, const double& GenBeta, bool RescaleP, const double& RescaleI, const double& RescaleT)
{
   if(TypeMode==1 && !(hscp.type() == HSCParticleType::trackerMuon || hscp.type() == HSCParticleType::globalMuon))return false;
   if( (TypeMode==2 || TypeMode==4) && hscp.type() != HSCParticleType::globalMuon)return false;

   reco::TrackRef   track;
   reco::MuonRef muon = hscp.muonRef();

   if(TypeMode!=3) track = hscp.trackRef();
   else {
     if(muon.isNull()) return false;
     track = muon->standAloneMuon();
   }
   if(track.isNull())return false;

   if(st){st->Total->Fill(0.0,Event_Weight);
     if(GenBeta>=0)st->Beta_Matched->Fill(GenBeta, Event_Weight);
     st->BS_Eta->Fill(track->eta(),Event_Weight);
   }

   if(fabs(track->eta())>GlobalMaxEta) return false;

   //Cut on number of matched muon stations
   int count = muonStations(track->hitPattern());
   if(st) {
     st->BS_MatchedStations->Fill(count, Event_Weight);
   }

   if(TypeMode==3 && count<minMuStations) return false;
   if(st) st->Stations->Fill(0.0, Event_Weight);

   if(st){st->BS_TNOH->Fill(track->found(),Event_Weight);
          st->BS_TNOHFraction->Fill(track->validFraction(),Event_Weight);
	  st->BS_TNOPH->Fill(track->hitPattern().numberOfValidPixelHits(),Event_Weight);
   }

   if(TypeMode!=3 && track->found()<GlobalMinNOH)return false;

   if(TypeMode!=3 && track->hitPattern().numberOfValidPixelHits()<GlobalMinNOPH)return false;
   if(TypeMode!=3 && track->validFraction()<GlobalMinFOVH)return false;

   if(st){st->TNOH  ->Fill(0.0,Event_Weight);
     if(dedxSObj) st->BS_TNOM->Fill(dedxSObj->numberOfMeasurements(),Event_Weight);
   }
   if(dedxSObj) if(dedxSObj->numberOfMeasurements()<GlobalMinNOM)return false;
   if(st){st->TNOM  ->Fill(0.0,Event_Weight);}

   if(tof){
   if(st){st->BS_nDof->Fill(tof->nDof(),Event_Weight);}
   if((TypeMode>1  && TypeMode!=5) && tof->nDof()<GlobalMinNDOF && (dttof->nDof()<GlobalMinNDOFDT || csctof->nDof()<GlobalMinNDOFCSC) )return false;
   }

   if(st){st->nDof  ->Fill(0.0,Event_Weight);
          st->BS_Qual->Fill(track->qualityMask(),Event_Weight);
   }

   if(TypeMode!=3 && track->qualityMask()<GlobalMinQual )return false;
   if(st){st->Qual  ->Fill(0.0,Event_Weight);
          st->BS_Chi2->Fill(track->chi2()/track->ndof(),Event_Weight);
   }
   if(TypeMode!=3 && track->chi2()/track->ndof()>GlobalMaxChi2 )return false;
   if(st){st->Chi2  ->Fill(0.0,Event_Weight);}

   if(st && GenBeta>=0)st->Beta_PreselectedA->Fill(GenBeta, Event_Weight);

   if(st){st->BS_MPt ->Fill(track->pt(),Event_Weight);}
   if(RescaleP){ if(RescaledPt(track->pt(),track->eta(),track->phi(),track->charge())<GlobalMinPt)return false;
   }else{        if(track->pt()<GlobalMinPt)return false;   }

   if(st){st->MPt   ->Fill(0.0,Event_Weight);
     if(dedxSObj) st->BS_MIs->Fill(dedxSObj->dEdx(),Event_Weight);
     if(dedxMObj) st->BS_MIm->Fill(dedxMObj->dEdx(),Event_Weight);
   }

   if(dedxSObj && dedxSObj->dEdx()+RescaleI<GlobalMinIs)return false;
   if(dedxMObj && ((TypeMode!=5 && dedxMObj->dEdx()<GlobalMinIm) || (TypeMode==5 && dedxMObj->dEdx()>GlobalMinIm)) )return false;
   if(st){st->MI   ->Fill(0.0,Event_Weight);}

   if(tof){
   if(st){st->BS_MTOF ->Fill(tof->inverseBeta(),Event_Weight);}
   //This cut is no longer applied here but rather in the PassSelection part to use the region
   //with TOF<GlobalMinTOF as a background check
   //if(TypeMode>1 && tof->inverseBeta()+RescaleT<GlobalMinTOF)return false;

   if(st)st->BS_TOFError->Fill(tof->inverseBetaErr(),Event_Weight);
   if((TypeMode>1  && TypeMode!=5) && tof->inverseBetaErr()>GlobalMaxTOFErr)return false;

   if(st) st->BS_TimeAtIP->Fill(tof->timeAtIpInOut(),Event_Weight);
   if(min(min(fabs(tof->timeAtIpInOut()-100), fabs(tof->timeAtIpInOut()-50)), min(fabs(tof->timeAtIpInOut()+100), fabs(tof->timeAtIpInOut()+50)))<5) return false;
   }

   if(st){st->MTOF ->Fill(0.0,Event_Weight);
      if(GenBeta>=0)st->Beta_PreselectedB->Fill(GenBeta, Event_Weight);
   }

   fwlite::Handle< std::vector<reco::Vertex> > vertexCollHandle;
   vertexCollHandle.getByLabel(ev,"offlinePrimaryVertices");
   if(!vertexCollHandle.isValid()){printf("Vertex Collection NotFound\n");return false;}
   const std::vector<reco::Vertex>& vertexColl = *vertexCollHandle;
   if(st){st->BS_NVertex->Fill(vertexColl.size(), Event_Weight);
     st->BS_NVertex_NoEventWeight->Fill(vertexColl.size());
   }
   if(vertexColl.size()<1){printf("NO VERTEX\n"); return false;}

   double dz  = track->dz (vertexColl[0].position());
   double dxy = track->dxy(vertexColl[0].position());
   int goodVerts=0;
   for(unsigned int i=0;i<vertexColl.size();i++){
     if(st) st->BS_dzAll->Fill( track->dz (vertexColl[i].position()),Event_Weight);
     if(st) st->BS_dxyAll->Fill(track->dxy(vertexColl[i].position()),Event_Weight);
     if(fabs(vertexColl[i].z())<15 && sqrt(vertexColl[i].x()*vertexColl[i].x()+vertexColl[i].y()*vertexColl[i].y())<2 && vertexColl[i].ndof()>3){ goodVerts++;}else{continue;} //only consider good vertex
     if(fabs(track->dz (vertexColl[i].position())) < fabs(dz) ){
       dz  = track->dz (vertexColl[i].position());
       dxy = track->dxy(vertexColl[i].position());
     }
   }

   if(st) st->BS_dzMinv3d->Fill(dz,Event_Weight);
   if(st) st->BS_dxyMinv3d->Fill(dxy,Event_Weight);
   if(st) st->BS_PV->Fill(goodVerts,Event_Weight);
   if(st) st->BS_PV_NoEventWeight->Fill(goodVerts);

   //Require at least one good vertex except if cosmic event
   if(TypeMode==3 && goodVerts<1 && (!st || st->Name.find("Cosmic")==string::npos)) return false;

   //For TOF only analysis match to a SA track without vertex constraint for IP cuts
   if(TypeMode==3) {
     fwlite::Handle< std::vector<reco::Track> > noVertexTrackCollHandle;
     noVertexTrackCollHandle.getByLabel(ev,"refittedStandAloneMuons", "");

     //Find closest NV track
     const std::vector<reco::Track>& noVertexTrackColl = *noVertexTrackCollHandle;
     reco::Track NVTrack;
     double minDr=15;
     for(unsigned int i=0;i<noVertexTrackColl.size();i++){
       double dR = deltaR(track->eta(), track->phi(), noVertexTrackColl[i].eta(), noVertexTrackColl[i].phi());
       if(dR<minDr) {minDr=dR;
	 NVTrack=noVertexTrackColl[i];}
     }
     if(st) st->BS_dR_NVTrack->Fill(minDr,Event_Weight);
     if(minDr>0.4) return false;
     if(st)st->NVTrack->Fill(0.0,Event_Weight);

     //Find displacement of tracks with respect to beam spot
     fwlite::Handle<reco::BeamSpot> beamSpotCollHandle;
     beamSpotCollHandle.getByLabel(ev,"offlineBeamSpot");
     if(!beamSpotCollHandle.isValid()){printf("Beam Spot Collection NotFound\n");return false;}
     const reco::BeamSpot& beamSpotColl = *beamSpotCollHandle;

     dz  = NVTrack.dz (beamSpotColl.position());
     dxy = NVTrack.dxy(beamSpotColl.position());
     if(muonStations(NVTrack.hitPattern())<minMuStations) return false;
   }

   double v3d = sqrt(dz*dz+dxy*dxy);

   if(st){st->BS_V3D->Fill(v3d,Event_Weight);}
   if(v3d>GlobalMaxV3D )return false;

   if(st)st->BS_Dxy->Fill(dxy, Event_Weight);

   TreeDXY = dxy;   
   bool DXYSB = false;
   if(TypeMode!=5 && fabs(dxy)>GlobalMaxDXY)return false;
   if(TypeMode==5 && fabs(dxy)>4)return false;
   if(TypeMode==5 && fabs(dxy)>GlobalMaxDXY) DXYSB = true;

   if(st){st->V3D  ->Fill(0.0,Event_Weight);}

   if(TypeMode!=3) {
     fwlite::Handle<HSCPIsolationValueMap> IsolationH;
     IsolationH.getByLabel(ev, "HSCPIsolation03");
     if(!IsolationH.isValid()){printf("Invalid IsolationH\n");return false;}
     const ValueMap<HSCPIsolation>& IsolationMap = *IsolationH.product();

     HSCPIsolation hscpIso = IsolationMap.get((size_t)track.key());
     if(st){st->BS_TIsol ->Fill(hscpIso.Get_TK_SumEt(),Event_Weight);}
//     if(TypeMode!=4){       if(hscpIso.Get_TK_SumEt()>GlobalMaxTIsol)return false;     }
      if(hscpIso.Get_TK_SumEt()>GlobalMaxTIsol)return false;
     if(st){st->TIsol   ->Fill(0.0,Event_Weight);}

     double EoP = (hscpIso.Get_ECAL_Energy() + hscpIso.Get_HCAL_Energy())/track->p();
     if(st){st->BS_EIsol ->Fill(EoP,Event_Weight);}
//     if(TypeMode!=4){       if(EoP>GlobalMaxEIsol)return false;     }
     if(EoP>GlobalMaxEIsol)return false;
     if(st){st->EIsol   ->Fill(0.0,Event_Weight);}
     
     // relative tracker isolation
     if (st) {  st->BS_SumpTOverpT->Fill(hscpIso.Get_TK_SumEt()/track->pt(), Event_Weight); }
//     if(TypeMode==4) { if(hscpIso.Get_TK_SumEt()/track->pt()>GlobalMaxRelTIsol)return false;   }
     if(hscpIso.Get_TK_SumEt()/track->pt()>GlobalMaxRelTIsol)return false;
     if (st) {  st->SumpTOverpT   ->Fill(0.0,Event_Weight);} 
   }

   if(st){st->BS_Pterr ->Fill(track->ptError()/track->pt(),Event_Weight);}
   if(TypeMode!=3 && (track->ptError()/track->pt())>GlobalMaxPterr)return false;

   if(std::max(0.0,track->pt())<GlobalMinPt)return false;
   if(st){st->Pterr   ->Fill(0.0,Event_Weight);}

   //Find distance to nearest segment on opposite side of detector
   double minPhi, minEta;
   double segSep=SegSep(hscp, ev, minPhi, minEta);

   if(st){
     st->BS_SegSep->Fill(segSep, Event_Weight);
     st->BS_SegMinPhiSep->Fill(minPhi, Event_Weight);
     st->BS_SegMinEtaSep->Fill(minEta, Event_Weight);
     //Plotting segment separation depending on whether track passed dz cut
     if(fabs(dz)>GlobalMaxDZ) {
       st->BS_SegMinEtaSep_FailDz->Fill(minEta, Event_Weight);
     }
     else {
       st->BS_SegMinEtaSep_PassDz->Fill(minEta, Event_Weight);
     }
     //Plots for tracking failing Eta Sep cut
     if(fabs(minEta)<minSegEtaSep) {
       //Needed to compare dz distribution of cosmics in pure cosmic and main sample
       st->BS_Dz_FailSep->Fill(dz);
     }
   }



   //Now cut Eta separation
   //if(TypeMode==3 && fabs(minEta)<minSegEtaSep) return false;
   if(st){st->SegSep->Fill(0.0,Event_Weight);}

   if(st) {
     //Plots for tracks in dz control region
     if(fabs(dz)>CosmicMinDz && fabs(dz)<CosmicMaxDz && !muon->isGlobalMuon()) {
       st->BS_Pt_FailDz->Fill(track->pt(), Event_Weight);
       st->BS_TOF_FailDz->Fill(tof->inverseBeta(), Event_Weight);
       if(fabs(track->eta())>CSCRegion) {
	 st->BS_TOF_FailDz_CSC->Fill(tof->inverseBeta(), Event_Weight);
	 st->BS_Pt_FailDz_CSC->Fill(track->pt(), Event_Weight);
       }
       else if(fabs(track->eta())<DTRegion) {
	 st->BS_TOF_FailDz_DT->Fill(tof->inverseBeta(), Event_Weight);
	 st->BS_Pt_FailDz_DT->Fill(track->pt(), Event_Weight);
       }
     }
     //Plots of dz
     st->BS_Dz->Fill(dz, Event_Weight);
     if(fabs(track->eta())>CSCRegion) st->BS_Dz_CSC->Fill(dz,Event_Weight);
     else if(fabs(track->eta())<DTRegion) st->BS_Dz_DT->Fill(dz,Event_Weight);
     st->BS_EtaDz->Fill(track->eta(),dz,Event_Weight);
   }


   //Split into different dz regions, each different region used to predict cosmic background and find systematic
   if(TypeMode==3 && !muon->isGlobalMuon() && st) {
     int DzType=-1;
     if(fabs(dz)<GlobalMaxDZ) DzType=0;
     else if(fabs(dz)<30) DzType=1;
     else if(fabs(dz)<50) DzType=2;
     else if(fabs(dz)<70) DzType=3;
     if(fabs(dz)>CosmicMinDz && fabs(dz)<CosmicMaxDz) DzType=4;
     if(fabs(dz)>CosmicMaxDz) DzType=5;

     //Count number of tracks in dz sidebands passing the TOF cut
     //The pt cut is not applied to increase statistics
     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
       if(tof->inverseBeta()>=CutTOF[CutIndex]) {
	 st->H_D_DzSidebands->Fill(CutIndex, DzType);
       }
     }
   }

   TreeDZ = dz;
   bool DZSB = false;
   if(TypeMode!=5 && fabs(dz)>GlobalMaxDZ) return false;
   if(TypeMode==5 && fabs(dz)>4) return false;
   if(TypeMode==5 && fabs(dz)>GlobalMaxDZ) DZSB = true;


   if(TypeMode==3 && fabs(minEta)<minSegEtaSep) return false;
   if(st)st->BS_Phi->Fill(track->phi(),Event_Weight);
   if(TypeMode==3 && fabs(track->phi())>1.2 && fabs(track->phi())<1.9) return false;

    //skip HSCP that are compatible with cosmics.
    if(st)st->BS_OpenAngle->Fill(OpenAngle,Event_Weight);

    bool OASB = false;
    if(TypeMode==5 && OpenAngle>=2.8)OASB = true;

   isCosmicSB = DXYSB && DZSB && OASB;
   isSemiCosmicSB = (!isCosmicSB && (DXYSB || DZSB || OASB));
 
   if(st){if(dedxSObj) st->BS_EtaIs->Fill(track->eta(),dedxSObj->dEdx(),Event_Weight);
          if(dedxMObj) st->BS_EtaIm->Fill(track->eta(),dedxMObj->dEdx(),Event_Weight);
          st->BS_EtaP ->Fill(track->eta(),track->p(),Event_Weight);
          st->BS_EtaPt->Fill(track->eta(),track->pt(),Event_Weight);
          if(tof)st->BS_EtaTOF->Fill(track->eta(),tof->inverseBeta(),Event_Weight);
   }

   if(st){if(GenBeta>=0)st->Beta_PreselectedC->Fill(GenBeta, Event_Weight);
          if(DZSB  && OASB)st->BS_Dxy_Cosmic->Fill(dxy, Event_Weight);
          if(DXYSB && OASB)st->BS_Dz_Cosmic->Fill(dz, Event_Weight);
          if(DXYSB && DZSB)st->BS_OpenAngle_Cosmic->Fill(OpenAngle,Event_Weight);

          st->BS_P  ->Fill(track->p(),Event_Weight);
          st->BS_Pt ->Fill(track->pt(),Event_Weight);
          if(DXYSB && DZSB && OASB) st->BS_Pt_Cosmic->Fill(track->pt(),Event_Weight);

	  if(fabs(track->eta())<DTRegion) st->BS_Pt_DT->Fill(track->pt(),Event_Weight);
	  else st->BS_Pt_CSC->Fill(track->pt(),Event_Weight);

          double RecoQoPt = track->charge()/track->pt();
          if(!hscp.trackRef().isNull() && hscp.trackRef()->pt()>200) {
            double InnerRecoQoPt = hscp.trackRef()->charge()/hscp.trackRef()->pt();
            st->BS_InnerInvPtDiff->Fill((RecoQoPt-InnerRecoQoPt)/InnerRecoQoPt,Event_Weight);
          }

          if(dedxSObj) st->BS_Is ->Fill(dedxSObj->dEdx(),Event_Weight);
          if(dedxSObj && DXYSB && DZSB && OASB) st->BS_Is_Cosmic->Fill(dedxSObj->dEdx(),Event_Weight);
          if(dedxSObj) st->BS_Im ->Fill(dedxMObj->dEdx(),Event_Weight);
          if(tof) {
	    st->BS_TOF->Fill(tof->inverseBeta(),Event_Weight);
	    if(dttof->nDof()>6) st->BS_TOF_DT->Fill(dttof->inverseBeta(),Event_Weight);
            if(csctof->nDof()>6) st->BS_TOF_CSC->Fill(csctof->inverseBeta(),Event_Weight);
            st->BS_PtTOF->Fill(track->pt() ,tof->inverseBeta(),Event_Weight);
	  }
          if(dedxSObj) {
	    st->BS_PIs  ->Fill(track->p()  ,dedxSObj->dEdx(),Event_Weight);
            st->BS_PIm  ->Fill(track->p()  ,dedxMObj->dEdx(),Event_Weight);
            st->BS_PtIs ->Fill(track->pt() ,dedxSObj->dEdx(),Event_Weight);
            st->BS_PtIm ->Fill(track->pt() ,dedxMObj->dEdx(),Event_Weight);
	  }
          if(tof && dedxSObj)st->BS_TOFIs->Fill(tof->inverseBeta(),dedxSObj->dEdx(),Event_Weight);
          if(tof && dedxSObj)st->BS_TOFIm->Fill(tof->inverseBeta(),dedxMObj->dEdx(),Event_Weight);

	  //Muon only prediction binned depending on where in the detector the track is and how many muon stations it has
	  //Binning not used for other analyses
	  int bin=-1;
	  if(TypeMode==3) {
	    if(fabs(track->eta())<DTRegion) bin=muonStations(track->hitPattern())-2;
	    else bin=muonStations(track->hitPattern())+1;
	    st->BS_Pt_Binned[bin] ->Fill(track->pt(),Event_Weight);
	    if(tof) st->BS_TOF_Binned[bin]->Fill(tof->inverseBeta(),Event_Weight);
	  }
   }

   return true;
}

// check if one HSCP candidate is passing the selection (the function also has many more arguments because it is used to fill some histograms AND to evaluate the systematics
bool PassSelection(const susybsm::HSCParticle& hscp,  const reco::DeDxData* dedxSObj, const reco::DeDxData* dedxMObj, const reco::MuonTimeExtra* tof, const fwlite::ChainEvent& ev, const int& CutIndex, stPlots* st, const bool isFlip, const double& GenBeta, bool RescaleP, const double& RescaleI, const double& RescaleT){

   reco::TrackRef   track;
   if(TypeMode!=3) track = hscp.trackRef();
   else {
     reco::MuonRef muon = hscp.muonRef();
     if(muon.isNull()) return false;
     track = muon->standAloneMuon();
   }
   if(track.isNull())return false;

   double MuonTOF = GlobalMinTOF;
   if(tof){
      MuonTOF = tof->inverseBeta();
   }

   double Is=0;
   if(dedxSObj) Is=dedxSObj->dEdx();
   double Ih=0;
   if(dedxMObj) Ih=dedxMObj->dEdx();

   double PtCut=CutPt[CutIndex];
   double ICut=CutI[CutIndex];
   double TOFCut=CutTOF[CutIndex];
   if(isFlip) {
     PtCut=CutPt_Flip[CutIndex];
     ICut=CutI_Flip[CutIndex];
     TOFCut=CutTOF_Flip[CutIndex];
   }

   if(RescaleP){
     if(RescaledPt(track->pt(),track->eta(),track->phi(),track->charge())<PtCut)return false;
     //if(std::max(0.0,RescaledPt(track->pt() - track->ptError(),track->eta(),track->phi(),track->charge()))<CutPt[CutIndex])return false;
   }else{
     if(track->pt()<PtCut)return false;
     //if(std::max(0.0,(track->pt() - track->ptError()))<CutPt[CutIndex])return false;
   } 
   if(st){st->Pt    ->Fill(CutIndex,Event_Weight);
          if(GenBeta>=0)st->Beta_SelectedP->Fill(CutIndex,GenBeta, Event_Weight);
   }

   if(TypeMode!=3 && Is+RescaleI<ICut)return false;

   if(st){st->I    ->Fill(CutIndex,Event_Weight);
          if(GenBeta>=0)st->Beta_SelectedI->Fill(CutIndex, GenBeta, Event_Weight);
   }

   if((TypeMode>1  && TypeMode!=5) && !isFlip && MuonTOF+RescaleT<TOFCut)return false;
   if((TypeMode>1  && TypeMode!=5) && isFlip && MuonTOF+RescaleT>TOFCut)return false;

   if(st){st->TOF  ->Fill(CutIndex,Event_Weight);
          if(GenBeta>=0)st->Beta_SelectedT->Fill(CutIndex, GenBeta, Event_Weight);
          st->AS_P  ->Fill(CutIndex,track->p(),Event_Weight);
          st->AS_Pt ->Fill(CutIndex,track->pt(),Event_Weight);
          st->AS_Is ->Fill(CutIndex,Is,Event_Weight);
          st->AS_Im ->Fill(CutIndex,Ih,Event_Weight);
          st->AS_TOF->Fill(CutIndex,MuonTOF,Event_Weight);
//        st->AS_EtaIs->Fill(CutIndex,track->eta(),Is,Event_Weight);
//        st->AS_EtaIm->Fill(CutIndex,track->eta(),Ih,Event_Weight);
//        st->AS_EtaP ->Fill(CutIndex,track->eta(),track->p(),Event_Weight);
//        st->AS_EtaPt->Fill(CutIndex,track->eta(),track->pt(),Event_Weight);
          st->AS_PIs  ->Fill(CutIndex,track->p()  ,Is,Event_Weight);
          st->AS_PIm  ->Fill(CutIndex,track->p()  ,Ih,Event_Weight);
          st->AS_PtIs ->Fill(CutIndex,track->pt() ,Is,Event_Weight);
          st->AS_PtIm ->Fill(CutIndex,track->pt() ,Ih,Event_Weight);
          st->AS_TOFIs->Fill(CutIndex,MuonTOF     ,Is,Event_Weight);
          st->AS_TOFIm->Fill(CutIndex,MuonTOF     ,Ih,Event_Weight);
   }

   return true;
}

// all code for the filling of the ABCD related histograms --> this information will be used later in Step4 for the actual datadriven prediction
void Analysis_FillControlAndPredictionHist(const susybsm::HSCParticle& hscp, const reco::DeDxData* dedxSObj, const reco::DeDxData* dedxMObj, const reco::MuonTimeExtra* tof, stPlots* st){
	 reco::TrackRef   track;
         if(TypeMode!=3) track = hscp.trackRef();
         else {
	   reco::MuonRef muon = hscp.muonRef();
           if(muon.isNull()) return;
           track = muon->standAloneMuon();
         }

         double MuonTOF = GlobalMinTOF;
         if(tof){MuonTOF = tof->inverseBeta(); }

	 double Is=0;
	 if(dedxSObj) Is=dedxSObj->dEdx();
	 double Ih=0;
	 if(dedxMObj) Ih=dedxMObj->dEdx();

         if(!isCosmicSB){
	 st->Hist_Pt->Fill(track->pt(),Event_Weight);
         st->Hist_Is->Fill(Is,Event_Weight);
         st->Hist_TOF->Fill(MuonTOF,Event_Weight);
         }

//          /\ I
//       /\  |----------------------------
//        |  |   |           |             |
//        |  |   |           |             |
//        |  |   |    B      |     D       |
//        |  |   |           |             |
//        |  ------------------------------
//        |  |   |           |             |
//        |  |   |    A      |     C       |
//        |  |   |           |             |
//        |  |---|-----------|-------------|
//        |  |   |           |             |
//        |  /--------------------------------> PT
//        | /       E       /    G  
//         /------------------------------->
//        /
//      TOF

         //Use different pt regions if using momentum from Stand Alone Muons
	 std::vector<double> PtLimits;
	 if(TypeMode!=3) {
	   PtLimits.push_back(100);
           PtLimits.push_back(80);
           PtLimits.push_back(60);
	 }
	 else {
           PtLimits.push_back(240);
           PtLimits.push_back(170);
           PtLimits.push_back(120);
	 }

	    //Muon only prediction binned depending on where in the detector the track is and how many muon stations it has
	    //Binning not used for other analyses
	    int bin=-1;
	    if(TypeMode==3) {
	      if(fabs(track->eta())<DTRegion) bin=muonStations(track->hitPattern())-2;
	      else bin=muonStations(track->hitPattern())+1;
	    }

         if(!isCosmicSB){
            if(track->pt()>PtLimits[0]){
               st->CtrlPt_S4_Is->Fill(Is, Event_Weight);
               st->CtrlPt_S4_Im->Fill(Ih, Event_Weight);
               if(tof)st->CtrlPt_S4_TOF->Fill(MuonTOF, Event_Weight);
               if(tof && bin>=0 && bin<MaxPredBins)st->CtrlPt_S4_TOF_Binned[bin]->Fill(MuonTOF, Event_Weight);
            }else if(track->pt()>PtLimits[1]){
               st->CtrlPt_S3_Is->Fill(Is, Event_Weight);
               st->CtrlPt_S3_Im->Fill(Ih, Event_Weight);
               if(tof)st->CtrlPt_S3_TOF->Fill(MuonTOF, Event_Weight);
               if(tof && bin>=0 && bin<MaxPredBins)st->CtrlPt_S3_TOF_Binned[bin]->Fill(MuonTOF, Event_Weight);
            }else if(track->pt()>PtLimits[2]){
               st->CtrlPt_S2_Is->Fill(Is, Event_Weight);
               st->CtrlPt_S2_Im->Fill(Ih, Event_Weight);
               if(tof)st->CtrlPt_S2_TOF->Fill(MuonTOF, Event_Weight);
               if(tof && bin>=0 && bin<MaxPredBins)st->CtrlPt_S2_TOF_Binned[bin]->Fill(MuonTOF, Event_Weight);
            }else{
               st->CtrlPt_S1_Is->Fill(Is, Event_Weight);
               st->CtrlPt_S1_Im->Fill(Ih, Event_Weight);
               if(tof)st->CtrlPt_S1_TOF->Fill(MuonTOF, Event_Weight);
               if(tof && bin>=0 && bin<MaxPredBins)st->CtrlPt_S1_TOF_Binned[bin]->Fill(MuonTOF, Event_Weight);
            }

            if(Is>0.2){           if(tof)st->CtrlIs_S4_TOF->Fill(MuonTOF, Event_Weight);
            }else if(Is>0.1){     if(tof)st->CtrlIs_S3_TOF->Fill(MuonTOF, Event_Weight);
            }else if(Is>0.05){    if(tof)st->CtrlIs_S2_TOF->Fill(MuonTOF, Event_Weight);
            }else{                if(tof)st->CtrlIs_S1_TOF->Fill(MuonTOF, Event_Weight);
            }

            if(Ih>4.4){           if(tof)st->CtrlIm_S4_TOF->Fill(MuonTOF, Event_Weight);
            }else if(Ih>4.1){     if(tof)st->CtrlIm_S3_TOF->Fill(MuonTOF, Event_Weight);
            }else if(Ih>3.8){     if(tof)st->CtrlIm_S2_TOF->Fill(MuonTOF, Event_Weight);
            }else{                if(tof)st->CtrlIm_S1_TOF->Fill(MuonTOF, Event_Weight);
            }
         }


         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
 	    if(MuonTOF<GlobalMinTOF) continue;
            if(TypeMode==5 && isCosmicSB)continue;
            bool PassPtCut  = track->pt()>=CutPt[CutIndex];
            bool PassICut   = (Is>=CutI[CutIndex]);
            bool PassTOFCut = MuonTOF>=CutTOF[CutIndex];
            if(       PassTOFCut &&  PassPtCut &&  PassICut){   //Region D
               st->H_D      ->Fill(CutIndex,                Event_Weight);
               if(bin>-1 && bin<MaxPredBins) st->H_D_Binned[bin]->Fill(CutIndex,                Event_Weight);
               st->RegionD_P  ->Fill(CutIndex,track->p(),     Event_Weight);
               st->RegionD_I  ->Fill(CutIndex,Ih,Event_Weight);
	       st->RegionD_Ias->Fill(CutIndex,Is,Event_Weight);
               st->RegionD_TOF->Fill(CutIndex,MuonTOF,        Event_Weight);
	       st->AS_Eta_RegionD->Fill(CutIndex,track->eta());
            }else if( PassTOFCut &&  PassPtCut && !PassICut){   //Region C
               st->H_C     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode<2)st->Pred_EtaP  ->Fill(CutIndex,track->eta(), track->p(),     Event_Weight);
//               Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);
               st->AS_Eta_RegionC->Fill(CutIndex,track->eta());
            }else if( PassTOFCut && !PassPtCut &&  PassICut){   //Region B
               st->H_B     ->Fill(CutIndex,                 Event_Weight);
               if(bin>-1 && bin<MaxPredBins) st->H_B_Binned[bin]->Fill(CutIndex,                Event_Weight);
               if(TypeMode<2)st->Pred_I  ->Fill(CutIndex,Ih, Event_Weight);
               if(TypeMode<2)st->Pred_EtaS->Fill(CutIndex,track->eta(),         Event_Weight);
//               Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);
               st->AS_Eta_RegionB->Fill(CutIndex,track->eta());
            }else if( PassTOFCut && !PassPtCut && !PassICut){   //Region A
               st->H_A     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)st->Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);
               if(TypeMode<2)st->Pred_EtaB->Fill(CutIndex,track->eta(),         Event_Weight);
               if(TypeMode==2)st->Pred_EtaS2->Fill(CutIndex,track->eta(),        Event_Weight);
               st->AS_Eta_RegionA->Fill(CutIndex,track->eta());
            }else if(!PassTOFCut &&  PassPtCut &&  PassICut){   //Region H
               st->H_H   ->Fill(CutIndex,          Event_Weight);
               if(bin>-1 && bin<MaxPredBins) st->H_H_Binned[bin]->Fill(CutIndex,                Event_Weight);
	       st->RegionH_Ias->Fill(CutIndex,Is,Event_Weight);
//               Pred_P->Fill(CutIndex,track->p(),        Event_Weight);
//               Pred_I->Fill(CutIndex,Ih,   Event_Weight);
               st->AS_Eta_RegionH->Fill(CutIndex,track->eta());
            }else if(!PassTOFCut &&  PassPtCut && !PassICut){   //Region G
               st->H_G     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)st->Pred_EtaP  ->Fill(CutIndex,track->eta(),track->p(),     Event_Weight);
               st->AS_Eta_RegionG->Fill(CutIndex,track->eta());
            }else if(!PassTOFCut && !PassPtCut &&  PassICut){   //Region F
               st->H_F     ->Fill(CutIndex,                 Event_Weight);
               if(bin>-1 && bin<MaxPredBins) st->H_F_Binned[bin]->Fill(CutIndex,                Event_Weight);
               if(TypeMode==2)st->Pred_I  ->Fill(CutIndex,Ih, Event_Weight);
               if(TypeMode==2)st->Pred_EtaS->Fill(CutIndex,track->eta(),         Event_Weight);
               st->AS_Eta_RegionF->Fill(CutIndex,track->eta());
            }else if(!PassTOFCut && !PassPtCut && !PassICut){   //Region E
               st->H_E     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)st->Pred_EtaB->Fill(CutIndex,track->eta(),         Event_Weight);
               st->AS_Eta_RegionE->Fill(CutIndex,track->eta());
            }
         }

	 //Use events with low TOF to check accuracy of background prediction
         for(unsigned int CutIndex=0;CutIndex<CutPt_Flip.size();CutIndex++){
            if(TypeMode!=5 && MuonTOF>=GlobalMinTOF) continue;
            if(TypeMode==5 && !isCosmicSB)continue;

            bool PassPtCut  = track->pt()>=CutPt_Flip[CutIndex];
            bool PassICut   = (Is>=CutI_Flip[CutIndex]);
            bool PassTOFCut = MuonTOF<=CutTOF_Flip[CutIndex]; 
            if(TypeMode==5)PassTOFCut=true;

            if(       PassTOFCut &&  PassPtCut &&  PassICut){   //Region D
	      st->RegionD_P_Flip  ->Fill(CutIndex,track->p(),     Event_Weight);
	      st->RegionD_I_Flip  ->Fill(CutIndex,Ih,Event_Weight);
	      st->RegionD_Ias_Flip  ->Fill(CutIndex,Is,Event_Weight);
	      st->RegionD_TOF_Flip->Fill(CutIndex,MuonTOF,        Event_Weight);
               st->H_D_Flip->Fill(CutIndex,                Event_Weight);
               if(bin>-1 && bin<MaxPredBins) st->H_D_Binned_Flip[bin]->Fill(CutIndex,                Event_Weight);
            }else if( PassTOFCut &&  PassPtCut && !PassICut){   //Region C
               st->H_C_Flip->Fill(CutIndex,                 Event_Weight);
               if(TypeMode<2)st->Pred_EtaP_Flip->Fill(CutIndex,track->eta(), track->p(),     Event_Weight);
//               Pred_TOF_Flip->Fill(CutIndex,MuonTOF,         Event_Weight);
            }else if( PassTOFCut && !PassPtCut &&  PassICut){   //Region B
               st->H_B_Flip->Fill(CutIndex,                 Event_Weight);
               if(bin>-1 && bin<MaxPredBins) st->H_B_Binned_Flip[bin]->Fill(CutIndex,                Event_Weight);
               if(TypeMode<2)st->Pred_I_Flip->Fill(CutIndex,Ih, Event_Weight);
               if(TypeMode<2)st->Pred_EtaS_Flip->Fill(CutIndex,track->eta(),         Event_Weight);
//               Pred_TOF_Flip->Fill(CutIndex,MuonTOF,         Event_Weight);
            }else if( PassTOFCut && !PassPtCut && !PassICut){   //Region A
               st->H_A_Flip->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)st->Pred_TOF_Flip->Fill(CutIndex,MuonTOF,         Event_Weight);
               if(TypeMode<2)st->Pred_EtaB_Flip->Fill(CutIndex,track->eta(),         Event_Weight);
               if(TypeMode==2)st->Pred_EtaS2_Flip->Fill(CutIndex,track->eta(),        Event_Weight);
            }else if(!PassTOFCut &&  PassPtCut &&  PassICut){   //Region H
               st->H_H_Flip->Fill(CutIndex,          Event_Weight);
               if(bin>-1 && bin<MaxPredBins) st->H_H_Binned_Flip[bin]->Fill(CutIndex,                Event_Weight);
	       st->RegionH_Ias_Flip  ->Fill(CutIndex,Is,Event_Weight);
	       //   Pred_P_Flip->Fill(CutIndex,track->p(),        Event_Weight);
	       //               Pred_I_Flip->Fill(CutIndex,Ih,   Event_Weight);
            }else if(!PassTOFCut &&  PassPtCut && !PassICut){   //Region G
               st->H_G_Flip->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)st->Pred_EtaP_Flip->Fill(CutIndex,track->eta(),track->p(),     Event_Weight);
            }else if(!PassTOFCut && !PassPtCut &&  PassICut){   //Region F
               st->H_F_Flip->Fill(CutIndex,                 Event_Weight);
               if(bin>-1 && bin<MaxPredBins) st->H_F_Binned_Flip[bin]->Fill(CutIndex,                Event_Weight);
               if(TypeMode==2)st->Pred_I_Flip->Fill(CutIndex,Ih, Event_Weight);
               if(TypeMode==2)st->Pred_EtaS_Flip->Fill(CutIndex,track->eta(),         Event_Weight);
            }else if(!PassTOFCut && !PassPtCut && !PassICut){   //Region E
               st->H_E_Flip->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)st->Pred_EtaB_Flip->Fill(CutIndex,track->eta(),         Event_Weight);
            }
         }
}

// Looping on all events, tracks, selection and check how many events are entering the mass distribution
void Analysis_Step3(char* SavePath)
{
   //Initialize a RandomNumberGenerator
   TRandom3* RNG = new TRandom3();

   //Initialize histo common to all samples
   InitHistos(NULL);

   for(unsigned int s=0;s<samples.size();s++){
      bool isData   = (samples[s].Type==0);
      bool isMC     = (samples[s].Type==1);
      bool isSignal = (samples[s].Type>=2);

      #ifdef ANALYSIS2011
         dEdxTemplates = NULL;
      #else
         if(isData){
//            dEdxTemplates = loadDeDxTemplate("../../data/Discrim_Templates_Data_2012.root");
            dEdxTemplates = NULL;
         }else{
            dEdxTemplates = NULL;
         }
      #endif



      //check that the plot container exist for this sample, otherwise create it
      if(plotsMap.find(samples[s].Name)==plotsMap.end()){plotsMap[samples[s].Name] = stPlots();}
      //For data and MCTr only initialize prediction histograms
      if(isData) stPlots_Init(HistoFile,plotsMap[samples[s].Name],samples[s].Name, CutPt.size(), false, false, CutPt_Flip.size());
      else stPlots_Init(HistoFile,plotsMap[samples[s].Name],samples[s].Name, CutPt.size());
      stPlots* SamplePlots = &plotsMap[samples[s].Name];
      SamplePlots->IntLumi->Fill(0.0,IntegratedLuminosity);

      string MCTrDirName = "MCTr_8TeV";
      if(isMC){
	if(samples[s].Name.find("7TeV")!=string::npos) MCTrDirName = "MCTr_7TeV";
         if(plotsMap.find(MCTrDirName)==plotsMap.end()){plotsMap[MCTrDirName] = stPlots();}
         stPlots_Init(HistoFile,plotsMap[MCTrDirName],MCTrDirName, CutPt.size(), false, false, CutPt_Flip.size());
      }stPlots* MCTrPlots = &plotsMap[MCTrDirName];

      //Initialize plot container for pure cosmic sample
      //Cosmic sample is contained in data file so for TOF-Only search
      //need a new set of plots for these events
      string CosmicName="";
      if(isData && TypeMode==3) {
	if(samples[s].Name.find("8TeV")!=string::npos) CosmicName="Cosmic8TeV";
	else CosmicName="Cosmic7TeV";
	if(plotsMap.find(CosmicName)==plotsMap.end()){plotsMap[CosmicName] = stPlots();}
	stPlots_Init(HistoFile,plotsMap[CosmicName],CosmicName, CutPt.size(), false, false, CutPt_Flip.size());
      }

      //needed for bookeeping
      bool* HSCPTk          = new bool[CutPt.size()];
      bool* HSCPTk_SystP    = new bool[CutPt.size()];
      bool* HSCPTk_SystI    = new bool[CutPt.size()];
      bool* HSCPTk_SystT    = new bool[CutPt.size()];
      bool* HSCPTk_SystM    = new bool[CutPt.size()];
      bool* HSCPTk_SystPU   = new bool[CutPt.size()];
      double* MaxMass       = new double[CutPt.size()];
      double* MaxMass_SystP = new double[CutPt.size()];
      double* MaxMass_SystI = new double[CutPt.size()];
      double* MaxMass_SystT = new double[CutPt.size()];
      double* MaxMass_SystM = new double[CutPt.size()];
      double* MaxMass_SystPU= new double[CutPt.size()];

      //do two loops through signal for samples with and without trigger changes.
      for (int period=0; period<(samples[s].Type>=2?2:1); period++){
         //load the files corresponding to this sample
         std::vector<string> FileName;
	 GetInputFiles(samples[s], BaseDirectory, FileName, period);
         fwlite::ChainEvent ev(FileName);

         DuplicatesClass duplicateChecker; 
         duplicateChecker.Clear();
         bool checkDuplicates = isData && FileName.size()>1;
         if(checkDuplicates){printf("Duplicated events will be removed\n");}

         //compute sample global weight
         Event_Weight = 1.0;
         double SampleWeight = 1.0;
         double PUSystFactor;

         if(samples[s].Type>0){           
            //get PU reweighted total # MC events.
            double NMCevents=0;
            for(Long64_t ientry=0;ientry<ev.size();ientry++){
              ev.to(ientry);
              if(MaxEntry>0 && ientry>MaxEntry)break;
              NMCevents += GetPUWeight(ev, samples[s].Pileup, PUSystFactor, LumiWeightsMC, PShift);
            }
            if(samples[s].Type==1)SampleWeight = GetSampleWeightMC(IntegratedLuminosity,FileName, samples[s].XSec, ev.size(), NMCevents);
            else                  SampleWeight = GetSampleWeight  (IntegratedLuminosity,IntegratedLuminosityBeforeTriggerChange,samples[s].XSec,NMCevents, period);
         }

	 if(SampleWeight==0) continue; //If sample weight 0 don't run, happens Int Lumi before change = 0

         //Loop on the events
         printf("Progressing Bar                   :0%%       20%%       40%%       60%%       80%%       100%%\n");
         printf("Building Mass for %10s (%1i/%1i) :",samples[s].Name.c_str(),period+1,(samples[s].Type>=2?RunningPeriods:1));
         int TreeStep = ev.size()/50;if(TreeStep==0)TreeStep=1;
         for(Long64_t ientry=0;ientry<ev.size();ientry++){
            ev.to(ientry);
            if(MaxEntry>0 && ientry>MaxEntry)break;
            if(ientry%TreeStep==0){printf(".");fflush(stdout);}
            if(checkDuplicates && duplicateChecker.isDuplicate(ev.eventAuxiliary().run(), ev.eventAuxiliary().event()))continue;

            //compute event weight
            if(samples[s].Type>0){Event_Weight = SampleWeight * GetPUWeight(ev, samples[s].Pileup, PUSystFactor, LumiWeightsMC, PShift);}else{Event_Weight = 1;}
            std::vector<reco::GenParticle> genColl;
            double HSCPGenBeta1=-1, HSCPGenBeta2=-1;
            if(isSignal){
               //get the collection of generated Particles
               fwlite::Handle< std::vector<reco::GenParticle> > genCollHandle;
               genCollHandle.getByLabel(ev, "genParticles");
               if(!genCollHandle.isValid()){printf("GenParticle Collection NotFound\n");continue;}
               genColl = *genCollHandle;
               int NChargedHSCP=HowManyChargedHSCP(genColl);

	       //if (NChargedHSCP > 2)   continue;   DONT THINK THIS IS NEEDED

               //skip event wich does not have the right number of charged HSCP --> DEPRECATED
               //if(samples[s].NChargedHSCP>=0 && samples[s].NChargedHSCP!=NChargedHSCP)continue;
               //NEW: reweight the events based on the number of charged HSCP directly at Analysis_step23.C (instead of Step6.C as in the past)
               Event_Weight*=samples[s].GetFGluinoWeight(NChargedHSCP);

               GetGenHSCPBeta(genColl,HSCPGenBeta1,HSCPGenBeta2,false);
               if(HSCPGenBeta1>=0)SamplePlots->Beta_Gen      ->Fill(HSCPGenBeta1, Event_Weight);  if(HSCPGenBeta2>=0)SamplePlots->Beta_Gen       ->Fill(HSCPGenBeta2, Event_Weight);
               GetGenHSCPBeta(genColl,HSCPGenBeta1,HSCPGenBeta2,true);
               if(HSCPGenBeta1>=0)SamplePlots->Beta_GenCharged->Fill(HSCPGenBeta1, Event_Weight); if(HSCPGenBeta2>=0)SamplePlots->Beta_GenCharged->Fill(HSCPGenBeta2, Event_Weight);
            }
            //check if the event is passing trigger
            SamplePlots      ->TotalE  ->Fill(0.0,Event_Weight);  
            if(isMC)MCTrPlots->TotalE  ->Fill(0.0,Event_Weight);
            SamplePlots      ->TotalEPU->Fill(0.0,Event_Weight*PUSystFactor);
            if(isMC)MCTrPlots->TotalEPU->Fill(0.0,Event_Weight*PUSystFactor);
	    //See if event passed signal triggers
            if(!PassTrigger(ev, isData) ) {
	      //For TOF only analysis if the event doesn't pass the signal triggers check if it was triggered by the no BPTX cosmic trigger
	      //If not TOF only then move to next event
	      if(TypeMode!=3) continue;
	      if(!PassTrigger(ev, isData, true)) continue;

	      //If is cosmic event then switch plots to use to the ones for cosmics
	      SamplePlots=&plotsMap[CosmicName];
	    }else if(TypeMode==3) {
	      SamplePlots = &plotsMap[samples[s].Name];
	    }

            SamplePlots       ->TotalTE->Fill(0.0,Event_Weight);
            if(isMC)MCTrPlots ->TotalTE->Fill(0.0,Event_Weight);

            //keep beta distribution for signal
            if(isSignal){if(HSCPGenBeta1>=0)SamplePlots->Beta_Triggered->Fill(HSCPGenBeta1, Event_Weight); if(HSCPGenBeta2>=0)SamplePlots->Beta_Triggered->Fill(HSCPGenBeta2, Event_Weight);}

            //load all event collection that will be used later on (HSCP COll, dEdx and TOF)
            fwlite::Handle<susybsm::HSCParticleCollection> hscpCollHandle;
            hscpCollHandle.getByLabel(ev,"HSCParticleProducer");
            //if(!hscpCollHandle.isValid()){printf("HSCP Collection NotFound\n");continue;}
            if(!hscpCollHandle.isValid())continue;
            const susybsm::HSCParticleCollection& hscpColl = *hscpCollHandle;

            fwlite::Handle<DeDxDataValueMap> dEdxSCollH;
            dEdxSCollH.getByLabel(ev, dEdxS_Label.c_str());
            if(!dEdxSCollH.isValid()){printf("Invalid dEdx Selection collection\n");continue;}

            fwlite::Handle<DeDxDataValueMap> dEdxMCollH;
            dEdxMCollH.getByLabel(ev, dEdxM_Label.c_str());
            if(!dEdxMCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}

            fwlite::Handle<MuonTimeExtraMap> TOFCollH;
            TOFCollH.getByLabel(ev, "muontiming",TOF_Label.c_str());
            if(!TOFCollH.isValid()){printf("Invalid TOF collection\n");return;}

            fwlite::Handle<MuonTimeExtraMap> TOFDTCollH;
            TOFDTCollH.getByLabel(ev, "muontiming",TOFdt_Label.c_str());
            if(!TOFDTCollH.isValid()){printf("Invalid DT TOF collection\n");return;}

            fwlite::Handle<MuonTimeExtraMap> TOFCSCCollH;
            TOFCSCCollH.getByLabel(ev, "muontiming",TOFcsc_Label.c_str());
            if(!TOFCSCCollH.isValid()){printf("Invalid CSC TOF collection\n");return;}

            //reinitialize the bookeeping array for each event
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk        [CutIndex] = false;   }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk_SystP  [CutIndex] = false;   }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk_SystI  [CutIndex] = false;   }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk_SystT  [CutIndex] = false;   }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk_SystM  [CutIndex] = false;   }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk_SystPU [CutIndex] = false; }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass       [CutIndex] = -1; }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass_SystP [CutIndex] = -1; }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass_SystI [CutIndex] = -1; }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass_SystT [CutIndex] = -1; }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass_SystM [CutIndex] = -1; }
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass_SystPU[CutIndex] = -1; }

            //loop on HSCP candidates
            for(unsigned int c=0;c<hscpColl.size();c++){
               //define alias for important variable
               susybsm::HSCParticle hscp  = hscpColl[c];
               reco::MuonRef  muon  = hscp.muonRef();

	       //For TOF only analysis use updated stand alone muon track.
	       //Otherwise use inner tracker track
	       reco::TrackRef track;
	       if(TypeMode!=3) track = hscp.trackRef();
	       else {
		 if(muon.isNull()) continue;
		 track = muon->standAloneMuon();
	       }
               //skip events without track
	       if(track.isNull())continue;

	       //Apply a scale factor to muon only analysis to account for differences seen in data/MC preselection efficiency
	       //For eta regions where Data > MC no correction to be conservative
	       if(!isData && TypeMode==3 && scaleFactor(track->eta())<RNG->Uniform(0, 1)) continue;

               //for signal only, make sure that the candidate is associated to a true HSCP
               int ClosestGen;
               if(isSignal && DistToHSCP(hscp, genColl, ClosestGen)>0.03)continue;

               //load quantity associated to this track (TOF and dEdx)
	       const DeDxData* dedxSObj = NULL;
	       const DeDxData* dedxMObj = NULL;
	       if(TypeMode!=3 && !track.isNull()) {
		 dedxSObj  = &dEdxSCollH->get(track.key());
		 dedxMObj  = &dEdxMCollH->get(track.key());
	       }

               const reco::MuonTimeExtra* tof = NULL;
               const reco::MuonTimeExtra* dttof = NULL;
               const reco::MuonTimeExtra* csctof = NULL;
              if(TypeMode>1 && !hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); dttof = &TOFDTCollH->get(hscp.muonRef().key());  csctof = &TOFCSCCollH->get(hscp.muonRef().key());}

               //Recompute dE/dx on the fly
               if(dedxSObj){
                  dedxSObj = dEdxOnTheFly(ev, track, dedxSObj, dEdxTemplates, TypeMode==5);

                  if(TypeMode==5)OpenAngle = deltaROpositeTrack(hscpColl, hscp); //OpenAngle is a global variable... that's uggly C++, but that's the best I found so far
               }

               //compute systematic uncertainties on signal
               if(isSignal){
#ifdef ANALYSIS2011
                  bool   PRescale = true;
                  double IRescale = RNG->Gaus(0, 0.083)+0.015; // added to the Ias value
                  double MRescale = 1.036;
		  double TRescale = -0.02; // added to the 1/beta value
		  if(tof) if(csctof->nDof()==0) TRescale = -0.003;
#else
                  bool   PRescale = true;
                  double IRescale = RNG->Gaus(0, 0.065)+0.020; // added to the Ias value
                  double MRescale = 1.041;
		  double TRescale = -0.005; // added to the 1/beta value
#endif
		  
		  double genpT = -1.0;
		  for(unsigned int g=0;g<genColl.size();g++) {
		    if(genColl[g].pt()<5)continue;
		    if(genColl[g].status()!=1)continue;
		    int AbsPdg=abs(genColl[g].pdgId());
		    if(AbsPdg!=17)continue;
		    
		    double separation = deltaR(track->eta(), track->phi(), genColl[g].eta(), genColl[g].phi());
		    if (separation > 0.03) continue;
		    genpT = genColl[g].pt();
		    break;    
		  }
                  if (genpT>0) {  SamplePlots->genrecopT->Fill(genpT, track->pt()); }
		  
                  // compute systematic due to momentum scale
                  if(PassPreselection( hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, ev,  NULL, -1,   PRescale, 0, 0)){
 		     double Mass     = -1; if(dedxMObj) Mass=GetMass(track->p()*PRescale,dedxMObj->dEdx(),!isData);
		     double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p()*PRescale,tof->inverseBeta());
		     double MassComb = -1;
		     if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5);
		     else if(dedxMObj) MassComb = Mass;
		     if(tof) MassComb=MassTOF;

                     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
		       if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, false, -1,   PRescale, 0, 0)){
                           HSCPTk_SystP[CutIndex] = true;
                           if(Mass>MaxMass_SystP[CutIndex]) MaxMass_SystP[CutIndex]=Mass;
                           SamplePlots->Mass_SystP->Fill(CutIndex, Mass,Event_Weight);
                           if(tof){
                              SamplePlots->MassTOF_SystP ->Fill(CutIndex, MassTOF , Event_Weight);
                           }
                           SamplePlots->MassComb_SystP->Fill(CutIndex, MassComb, Event_Weight);
                        }
                     }
                  }

                  // compute systematic due to dEdx (both Ias and Ih)
                  if(PassPreselection( hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, ev,  NULL, -1,   0, IRescale, 0)){
		     double Mass     = -1; if(dedxMObj) Mass=GetMass(track->p(),dedxMObj->dEdx()*MRescale,!isData);
		     double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta());
		     double MassComb = -1;
		     if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5);
		     else if(dedxMObj) MassComb = Mass;
		     if(tof) MassComb=MassTOF;

                     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
		       if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, false, -1,   0, IRescale, 0)){
                           HSCPTk_SystI[CutIndex] = true;
                           if(Mass>MaxMass_SystI[CutIndex]) MaxMass_SystI[CutIndex]=Mass;
                           SamplePlots->Mass_SystI->Fill(CutIndex, Mass,Event_Weight);
                           if(tof){
                              SamplePlots->MassTOF_SystI ->Fill(CutIndex, MassTOF , Event_Weight);
                           }
                           SamplePlots->MassComb_SystI->Fill(CutIndex, MassComb, Event_Weight);
                        }
                     }
                  }

                  // compute systematic due to Mass shift
                  if(PassPreselection( hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, ev,  NULL, -1,   0, 0, 0)){
		     double Mass     = -1; if(dedxMObj) Mass=GetMass(track->p(),dedxMObj->dEdx()*MRescale,!isData);
		     double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta());
		     double MassComb = -1;
		     if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5);
		     else if(dedxMObj) MassComb = Mass;
		     if(tof) MassComb=MassTOF;

                     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
		       if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, false, -1,   0, 0, 0)){
                           HSCPTk_SystM[CutIndex] = true;
                           if(Mass>MaxMass_SystM[CutIndex]) MaxMass_SystM[CutIndex]=Mass;
                           SamplePlots->Mass_SystM->Fill(CutIndex, Mass,Event_Weight);
                           if(tof){
                              SamplePlots->MassTOF_SystM ->Fill(CutIndex, MassTOF , Event_Weight);
                           }
                           SamplePlots->MassComb_SystM->Fill(CutIndex, MassComb, Event_Weight);
                        }
                     }
                  }

                  // compute systematic due to TOF
                  if(PassPreselection( hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, ev,  NULL, -1,   0, 0, TRescale)){
 		     double Mass     = -1; if(dedxMObj) Mass=GetMass(track->p(),dedxMObj->dEdx(),!isData);
		     double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),(tof->inverseBeta()+TRescale));
		     double MassComb = -1;
		     if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5);
		     else if(dedxMObj) MassComb = Mass;
		     if(tof) MassComb=MassTOF;

                     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
		       if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, false, -1,   0, 0, TRescale)){
                           HSCPTk_SystT[CutIndex] = true;
                           if(Mass>MaxMass_SystT[CutIndex]) MaxMass_SystT[CutIndex]=Mass;
                           SamplePlots->Mass_SystT->Fill(CutIndex, Mass,Event_Weight);
                           if(tof){
                              SamplePlots->MassTOF_SystT ->Fill(CutIndex, MassTOF , Event_Weight);
                           }
                           SamplePlots->MassComb_SystT->Fill(CutIndex, MassComb, Event_Weight);
                        }
                     }
                  }

                  // compute systematics due to PU
                  if(PassPreselection( hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, ev,  NULL, -1,   PRescale, 0, 0)){
		     double Mass     = -1; if(dedxMObj) Mass=GetMass(track->p(),dedxMObj->dEdx(),!isData);
		     double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta());
		     double MassComb = -1;
		     if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5);
		     else if(dedxMObj) MassComb = Mass;
		     if(tof) MassComb=MassTOF;

                     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
		       if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, false, -1,   PRescale, 0, 0)){
                           HSCPTk_SystPU[CutIndex] = true;
                           if(Mass>MaxMass_SystPU[CutIndex]) MaxMass_SystPU[CutIndex]=Mass;
                           SamplePlots->Mass_SystPU->Fill(CutIndex, Mass,Event_Weight*PUSystFactor);
                           if(tof){
                              SamplePlots->MassTOF_SystPU ->Fill(CutIndex, MassTOF , Event_Weight*PUSystFactor);
                           }
                           SamplePlots->MassComb_SystPU->Fill(CutIndex, MassComb, Event_Weight*PUSystFactor);
                        }
                     }
                  }
               }//End of systematic computation for signal

               //check if the canddiate pass the preselection cuts
               if(isMC)PassPreselection( hscp, dedxSObj, dedxMObj, tof, dttof, csctof, ev, MCTrPlots   );
               if(    !PassPreselection( hscp, dedxSObj, dedxMObj, tof, dttof, csctof, ev, SamplePlots, isSignal?genColl[ClosestGen].p()/genColl[ClosestGen].energy():-1)) continue;
               //stPlots_FillTree(SamplePlots, ev.eventAuxiliary().run(),ev.eventAuxiliary().event(), c, track->pt(), dedxSObj ? dedxSObj->dEdx() : -1, tof ? tof->inverseBeta() : -1, -1, TreeDZ, TreeDXY, OpenAngle, track->eta(), track->phi(), -1);
               if(TypeMode==5 && isSemiCosmicSB)continue;

               //fill the ABCD histograms and a few other control plots
               if(isData)Analysis_FillControlAndPredictionHist(hscp, dedxSObj, dedxMObj, tof, SamplePlots);
	       else if(isMC) Analysis_FillControlAndPredictionHist(hscp, dedxSObj, dedxMObj, tof, MCTrPlots);

               if(TypeMode==5 && isCosmicSB)continue; 

	       //Find the number of tracks passing selection for TOF<1 that will be used to check the background prediction
	       if(isMC || isData) {
               //compute the mass of the candidate, for TOF mass flip the TOF over 1 to get the mass, so 0.8->1.2
		 double Mass     = -1; if(dedxMObj) Mass = GetMass(track->p(),dedxMObj->dEdx(),!isData);
		 double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),(2-tof->inverseBeta()));
		 double MassComb = -1;
		 if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p(), (GetIBeta(dedxMObj->dEdx(),!isData) + (1/(2-tof->inverseBeta())))*0.5 ) ;
		 if(dedxMObj) MassComb = Mass;
		 if(tof)MassComb=GetMassFromBeta(track->p(),(1/(2-tof->inverseBeta())));

		 for(unsigned int CutIndex=0;CutIndex<CutPt_Flip.size();CutIndex++){
		 //Background check looking at region with TOF<1
		   if(!PassSelection   (hscp, dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, true)) continue;
                  //Fill Mass Histograms

                  if(isMC)MCTrPlots->Mass_Flip->Fill(CutIndex, Mass,Event_Weight);
                  SamplePlots      ->Mass_Flip->Fill(CutIndex, Mass,Event_Weight);
                  if(tof){
                  if(isMC)MCTrPlots->MassTOF_Flip->Fill(CutIndex, MassTOF, Event_Weight);
                     SamplePlots   ->MassTOF_Flip->Fill(CutIndex, MassTOF, Event_Weight);
                  }
                  if(isMC)MCTrPlots->MassComb_Flip->Fill(CutIndex, MassComb, Event_Weight);
                  SamplePlots      ->MassComb_Flip->Fill(CutIndex, MassComb, Event_Weight);
		 }
	       }

               //compute the mass of the candidate
	       double Mass     = -1; if(dedxMObj) Mass = GetMass(track->p(),dedxMObj->dEdx(),!isData);
	       double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta());
	       double MassComb = -1;
	       if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p(), (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5 ) ;
	       if(dedxMObj) MassComb = Mass;
	       if(tof)MassComb=GetMassFromBeta(track->p(),(1/tof->inverseBeta()));
               bool PassNonTrivialSelection=false;

               //loop on all possible selection (one of them, the optimal one, will be used later)
               for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                  //Full Selection
		 if(isMC)PassSelection   (hscp, dedxSObj, dedxMObj, tof, ev, CutIndex, MCTrPlots);
		 if(    !PassSelection   (hscp, dedxSObj, dedxMObj, tof, ev, CutIndex, SamplePlots, false, isSignal?genColl[ClosestGen].p()/genColl[ClosestGen].energy():-1))continue;

                  if(CutIndex!=0)PassNonTrivialSelection=true;
                  HSCPTk[CutIndex] = true;
                  if(Mass>MaxMass[CutIndex]) MaxMass[CutIndex]=Mass;

                  //Fill Mass Histograms
                  if(isMC)MCTrPlots->Mass->Fill(CutIndex, Mass,Event_Weight);
                  SamplePlots      ->Mass->Fill(CutIndex, Mass,Event_Weight);
                  if(tof){
                  if(isMC)MCTrPlots->MassTOF->Fill(CutIndex, MassTOF, Event_Weight);
                     SamplePlots   ->MassTOF->Fill(CutIndex, MassTOF, Event_Weight);
                  }
                  if(isMC)MCTrPlots->MassComb->Fill(CutIndex, MassComb, Event_Weight);
                  SamplePlots      ->MassComb->Fill(CutIndex, MassComb, Event_Weight);
               } //end of Cut loop
               if(PassNonTrivialSelection) stPlots_FillTree(SamplePlots, ev.eventAuxiliary().run(),ev.eventAuxiliary().event(), c, track->pt(), dedxSObj ? dedxSObj->dEdx() : -1, tof ? tof->inverseBeta() : -1, Mass, TreeDZ, TreeDXY, OpenAngle, track->eta(), track->phi(), -1);
            }// end of Track Loop

            //save event dependent information thanks to the bookkeeping
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
              if(HSCPTk[CutIndex]){
                 SamplePlots->HSCPE             ->Fill(CutIndex,Event_Weight);
                 SamplePlots->MaxEventMass      ->Fill(CutIndex,MaxMass[CutIndex], Event_Weight);
                 if(isMC){
                 MCTrPlots->HSCPE               ->Fill(CutIndex,Event_Weight);
                 MCTrPlots->MaxEventMass        ->Fill(CutIndex,MaxMass[CutIndex], Event_Weight);
                 }
              }
              if(HSCPTk_SystP[CutIndex]){
                 SamplePlots->HSCPE_SystP       ->Fill(CutIndex,Event_Weight);
                 SamplePlots->MaxEventMass_SystP->Fill(CutIndex,MaxMass_SystP[CutIndex], Event_Weight);
              }
              if(HSCPTk_SystI[CutIndex]){
                 SamplePlots->HSCPE_SystI       ->Fill(CutIndex,Event_Weight);
                 SamplePlots->MaxEventMass_SystI->Fill(CutIndex,MaxMass_SystI[CutIndex], Event_Weight);
              }
              if(HSCPTk_SystM[CutIndex]){
                 SamplePlots->HSCPE_SystM       ->Fill(CutIndex,Event_Weight);
                 SamplePlots->MaxEventMass_SystM->Fill(CutIndex,MaxMass_SystM[CutIndex], Event_Weight);
              }
              if(HSCPTk_SystT[CutIndex]){
                 SamplePlots->HSCPE_SystT       ->Fill(CutIndex,Event_Weight);
                 SamplePlots->MaxEventMass_SystT->Fill(CutIndex,MaxMass_SystT[CutIndex], Event_Weight);
              }
              if(HSCPTk_SystPU[CutIndex]){
                 SamplePlots->HSCPE_SystPU       ->Fill(CutIndex,Event_Weight);
                 SamplePlots->MaxEventMass_SystPU->Fill(CutIndex,MaxMass_SystPU[CutIndex], Event_Weight);
              }
           }
         }printf("\n");// end of Event Loop
      }//end of period loop
      delete [] HSCPTk;
      delete [] HSCPTk_SystP;
      delete [] HSCPTk_SystI;
      delete [] HSCPTk_SystT;
      delete [] HSCPTk_SystM;
      delete [] HSCPTk_SystPU;
      delete [] MaxMass;
      delete [] MaxMass_SystP;
      delete [] MaxMass_SystI;
      delete [] MaxMass_SystT;
      delete [] MaxMass_SystM;
      delete [] MaxMass_SystPU;

      stPlots_Clear(SamplePlots, true);
      if(isMC)stPlots_Clear(MCTrPlots, true);
   }// end of sample loop
   delete RNG;
}

void InitHistos(stPlots* st){   
   //Initialization of variables that are common to all samples
   HistoFile->cd();
   HCuts_Pt  = new TProfile("HCuts_Pt" ,"HCuts_Pt" ,CutPt.size(),0,CutPt.size());
   HCuts_I   = new TProfile("HCuts_I"  ,"HCuts_I"  ,CutPt.size(),0,CutPt.size());
   HCuts_TOF = new TProfile("HCuts_TOF","HCuts_TOF",CutPt.size(),0,CutPt.size());
   for(unsigned int i=0;i<CutPt.size();i++){  HCuts_Pt->Fill(i,CutPt[i]);     HCuts_I->Fill(i,CutI[i]);    HCuts_TOF->Fill(i,CutTOF[i]); }

   HCuts_Pt_Flip  = new TProfile("HCuts_Pt_Flip" ,"HCuts_Pt_Flip" ,CutPt_Flip.size(),0,CutPt_Flip.size());
   HCuts_I_Flip   = new TProfile("HCuts_I_Flip"  ,"HCuts_I_Flip"  ,CutPt_Flip.size(),0,CutPt_Flip.size());
   HCuts_TOF_Flip = new TProfile("HCuts_TOF_Flip","HCuts_TOF_Flip",CutPt_Flip.size(),0,CutPt_Flip.size());
   for(unsigned int i=0;i<CutPt_Flip.size();i++){  HCuts_Pt_Flip->Fill(i,CutPt_Flip[i]);     HCuts_I_Flip->Fill(i,CutI_Flip[i]);    HCuts_TOF_Flip->Fill(i,CutTOF_Flip[i]); }
}

// code needed for the evaluation of the systematics related to pt measurement
double RescaledPt(const double& pt, const double& eta, const double& phi, const int& charge)
{
  if(TypeMode!=3) {
    double newInvPt = 1/pt+0.000236-0.000135*pow(eta,2)+charge*0.000282*TMath::Sin(phi-1.337);
    return 1/newInvPt;
  }
  else {
    double newInvPt = (1./pt)*1.1;
    return 1/newInvPt;
  }
}

double SegSep(const susybsm::HSCParticle& hscp, const fwlite::ChainEvent& ev, double& minPhi, double& minEta) {
  if(TypeMode!=3)return -1;

  reco::MuonRef muon = hscp.muonRef();
  if(muon.isNull()) return false;
  reco::TrackRef  track = muon->standAloneMuon();
  if(track.isNull())return false;

  fwlite::Handle<MuonSegmentCollection> SegCollHandle;
  SegCollHandle.getByLabel(ev, "MuonSegmentProducer");
  if(!SegCollHandle.isValid()){printf("Segment Collection Not Found\n"); return -1;}
  MuonSegmentCollection SegCollection = *SegCollHandle;

  double minDr=10;
  minPhi=10;
  minEta=10;

  //Look for segment on opposite side of detector from track
  for (MuonSegmentCollection::const_iterator segment = SegCollection.begin(); segment!=SegCollection.end();++segment) {  
    GlobalPoint gp = segment->getGP();

    //Flip HSCP to opposite side of detector
    double eta_hscp = -1*track->eta();
    double phi_hscp= track->phi()+M_PI;

    double deta = gp.eta() - eta_hscp;
    double dphi = gp.phi() - phi_hscp;
    while (dphi >   M_PI) dphi -= 2*M_PI;
    while (dphi <= -M_PI) dphi += 2*M_PI;

    //Find segment most opposite in eta
    //Require phi difference of 0.5 so it doesn't match to own segment
    if(fabs(deta)<fabs(minEta) && fabs(dphi)<(M_PI-0.5)) {
      minEta=deta;
    }
    //Find segment most opposite in phi
    if(fabs(dphi)<fabs(minPhi)) {
      minPhi=dphi;
    }
    //Find segment most opposite in Eta-Phi
    double dR=sqrt(deta*deta+dphi*dphi);
    if(dR<minDr) minDr=dR;
  }
  return minDr;
}

//Counts the number of muon stations used in track fit only counting DT and CSC stations.
int  muonStations(reco::HitPattern hitPattern) {
  int stations[4] = { 0,0,0,0 };

  for (int i=0; i<hitPattern.numberOfHits(); i++) {
    uint32_t pattern = hitPattern.getHitPattern(i);
    if (pattern == 0) break;
    if (hitPattern.muonHitFilter(pattern) &&
	(int(hitPattern.getSubStructure(pattern)) == 1 ||
	 int(hitPattern.getSubStructure(pattern)) == 2) &&
	hitPattern.getHitType(pattern) == 0) {
      stations[hitPattern.getMuonStation(pattern)-1] = 1;
    }
  }

  return stations[0]+stations[1]+stations[2]+stations[3];
}

double scaleFactor(double eta) {
  double etaBins[15]   = {-2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0 , 0.3 , 0.6 , 0.9 , 1.2 ,1.5 , 1.8 , 2.1 };
  double scaleBins[15] = {0,    0.97, 1.06, 1.00, 0.89, 0.91, 0.93, 0.93, 0.92, 0.92, 0.91, 0.89,1.00, 1.06, 0.99};
  for (int i=0; i<15; i++) if(eta<etaBins[i]) return scaleBins[i];
  return 0;
}
