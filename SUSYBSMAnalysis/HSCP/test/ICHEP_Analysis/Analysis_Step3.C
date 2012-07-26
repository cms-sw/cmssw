// Original Author:  Loic Quertenmont


namespace reco    { class Vertex; class Track; class GenParticle; class DeDxData; class MuonTimeExtra; class PFMET;}
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
bool PassSelection(const susybsm::HSCParticle& hscp,  const reco::DeDxData* dedxSObj, const reco::DeDxData* dedxMObj, const reco::MuonTimeExtra* tof, const fwlite::ChainEvent& ev, const int& CutIndex=0, stPlots* st=NULL, const double& GenBeta=-1, bool RescaleP=false, const double& RescaleI=0.0, const double& RescaleT=0.0);
void Analysis_FillControlAndPredictionHist(const susybsm::HSCParticle& hscp, const reco::DeDxData* dedxSObj, const reco::DeDxData* dedxMObj, const reco::MuonTimeExtra* tof, stPlots* st=NULL);
double SegSep(const susybsm::HSCParticle& hscp, const fwlite::ChainEvent& ev, double& minPhi, double& minEta);
double RescaledPt       (const double& pt, const double& eta, const double& phi, const int& charge);
/////////////////////////// VARIABLE DECLARATION /////////////////////////////

float Event_Weight = 1;
int   MaxEntry = -1;

TFile* HistoFile;

TH1D*  Hist_Pt ;
TH1D*  Hist_Is  ;
TH1D*  Hist_TOF;

TH2D*  Pred_Mass;
TH2D*  Pred_MassTOF;
TH2D*  Pred_MassComb;

TH1D* H_A;
TH1D* H_B;
TH1D* H_C;
TH1D* H_D;
TH1D* H_E;
TH1D* H_F;
TH1D* H_G;
TH1D* H_H;
TH1D* H_P;

TH2D* H_D_DzSidebands;
TH2D* H_D_DzSidebands_DT;
TH2D* H_D_DzSidebands_CSC;

std::vector<double>  CutPt ;
std::vector<double>  CutI  ;
std::vector<double>  CutTOF;

TProfile*  HCuts_Pt;
TProfile*  HCuts_I;
TProfile*  HCuts_TOF;

TH3D*  Pred_EtaP ;
TH2D*  Pred_I    ;
TH2D*  Pred_TOF  ;
TH2D*  Pred_EtaB;
TH2D*  Pred_EtaS;
TH2D*  Pred_EtaS2;

TH2D*  RegionD_P   ;
TH2D*  RegionD_I   ;
TH2D*  RegionD_TOF  ;

TH1D*  CtrlPt_S1_Is;
TH1D*  CtrlPt_S1_Im;
TH1D*  CtrlPt_S1_TOF;
TH1D*  CtrlPt_S2_Is;
TH1D*  CtrlPt_S2_Im;
TH1D*  CtrlPt_S2_TOF;
TH1D*  CtrlPt_S3_Is;
TH1D*  CtrlPt_S3_Im;
TH1D*  CtrlPt_S3_TOF;
TH1D*  CtrlPt_S4_Is;
TH1D*  CtrlPt_S4_Im;
TH1D*  CtrlPt_S4_TOF;
TH1D*  CtrlIs_S1_TOF;
TH1D*  CtrlIs_S2_TOF;
TH1D*  CtrlIs_S3_TOF;
TH1D*  CtrlIs_S4_TOF;
TH1D*  CtrlIm_S1_TOF;
TH1D*  CtrlIm_S2_TOF;
TH1D*  CtrlIm_S3_TOF;
TH1D*  CtrlIm_S4_TOF;

std::vector<stSample> samples;
std::map<std::string, stPlots> plotsMap;

//for initializing PileupReweighting utility.
const   float TrueDist2011_f[35] = {0.00285942, 0.0125603, 0.0299631, 0.051313, 0.0709713, 0.0847864, 0.0914627, 0.0919255, 0.0879994, 0.0814127, 0.0733995, 0.0647191, 0.0558327, 0.0470663, 0.0386988, 0.0309811, 0.0241175, 0.018241, 0.0133997, 0.00956071, 0.00662814, 0.00446735, 0.00292946, 0.00187057, 0.00116414, 0.000706805, 0.000419059, 0.000242856, 0.0001377, 7.64582e-05, 4.16101e-05, 2.22135e-05, 1.16416e-05, 5.9937e-06, 5.95542e-06};//from 2011 Full dataset

const   float Pileup_MC[35]= {1.45346E-01, 6.42802E-02, 6.95255E-02, 6.96747E-02, 6.92955E-02, 6.84997E-02, 6.69528E-02, 6.45515E-02, 6.09865E-02, 5.63323E-02, 5.07322E-02, 4.44681E-02, 3.79205E-02, 3.15131E-02, 2.54220E-02, 2.00184E-02, 1.53776E-02, 1.15387E-02, 8.47608E-03, 6.08715E-03, 4.28255E-03, 2.97185E-03, 2.01918E-03, 1.34490E-03, 8.81587E-04, 5.69954E-04, 3.61493E-04, 2.28692E-04, 1.40791E-04, 8.44606E-05, 5.10204E-05, 3.07802E-05, 1.81401E-05, 1.00201E-05, 5.80004E-06};

std::vector< float > BgLumiMC; //MC                                           
std::vector< float > TrueDist2011;                                    
edm::LumiReWeighting LumiWeightsMC;
reweight::PoissonMeanShifter PShift(0.6);//0.6 for upshift, -0.6 for downshift

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
   }else if(TypeMode==2) { GlobalMaxTIsol *= 2;
                           GlobalMaxEIsol *= 2;
   }
   else if(TypeMode==3){
     GlobalMinIs      =   -1;
   }

   // define the selection to be considered later for the optimization
   // WARNING: recall that this has a huge impact on the analysis time AND on the output file size --> be carefull with your choice
   CutPt .push_back(GlobalMinPt);   CutI  .push_back(GlobalMinIs);  CutTOF.push_back(GlobalMinTOF);

   if(TypeMode<2){   
      for(double Pt =GlobalMinPt+5 ; Pt <200;Pt+=5){
      for(double I  =GlobalMinIs+0.025  ; I  <0.45 ;I+=0.025){
         CutPt .push_back(Pt);   CutI  .push_back(I);  CutTOF.push_back(-1);
      }}
   }else if(TypeMode==2){
      for(double Pt =GlobalMinPt+5 ; Pt <120;  Pt+=5){
      for(double I  =GlobalMinIs +0.025; I  <0.40;  I+=0.025){
      for(double TOF=GlobalMinTOF+0.025; TOF<1.35;TOF+=0.025){
         CutPt .push_back(Pt);   CutI  .push_back(I);  CutTOF.push_back(TOF);
      }}}
   }else if(TypeMode==3){
     for(double Pt =GlobalMinPt+30 ; Pt <550;  Pt+=30){
       for(double TOF=GlobalMinTOF+0.025; TOF<1.4;TOF+=0.01){
         CutPt .push_back(Pt);   CutI  .push_back(-1);  CutTOF.push_back(TOF);
       }}
   }
   printf("%i Different Final Selection will be tested\n",(int)CutPt.size());

   //initialize LumiReWeighting
   for(int i=0; i<35; ++i) BgLumiMC.push_back(Pileup_MC[i]);
   for(int i=0; i<35; ++i) TrueDist2011.push_back(TrueDist2011_f[i]);
   LumiWeightsMC = edm::LumiReWeighting(BgLumiMC, TrueDist2011);

   //make the directory structure corresponding to this analysis (depends on dEdx/TOF estimator being used, Eta/Pt cuts and Mode of the analysis)
   char Buffer[2048], Command[2048];
   sprintf(Buffer,"Results/%s/%s/Eta%02.0f/PtMin%02.0f/Type%i/", dEdxS_Label.c_str(), TOF_Label.c_str(), 10.0*GlobalMaxEta, GlobalMinPt, TypeMode);
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
         if(tr.accept(tr.triggerIndex("HscpPathSingleMu")))return true;
         else if(tr.accept(tr.triggerIndex("HscpPathPFMet"))){
           if(!isData) Event_Weight=Event_Weight*0.96;
           return true;
         }
	 if(TypeMode==3) {
	   fwlite::Handle<reco::PFMETCollection> pfMETCollection;
	   pfMETCollection.getByLabel(ev,"pfMet");
	   if(tr.accept(tr.triggerIndex("HSCPPathSAMU")) && pfMETCollection->begin()->et()>60) return true;
	 }
      #else
	 if(TypeMode!=3) {
	   if(tr.accept("HSCPHLTTriggerMetDeDxFilter"))return true;
	   if(tr.accept("HSCPHLTTriggerMuDeDxFilter"))return true;
	   if(tr.accept("HSCPHLTTriggerMuFilter"))return true;
	 }
	 //Only for the TOF only analysis
	 if(TypeMode==3) if(tr.accept(tr.triggerIndex("HSCPHLTTriggerL2MuFilter"))) return true;
	 //Only accepted if looking for cosmic events
	 if(isCosmic) if(tr.accept(tr.triggerIndex("HSCPHLTTriggerCosmicFilter"))) return true;
      #endif
      return false;
}

// check if one HSCP candidate is passing the preselection (the function also has many more arguments because it is used to fill some histograms AND to evaluate the systematics
bool PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData* dedxSObj, const reco::DeDxData* dedxMObj, const reco::MuonTimeExtra* tof, const reco::MuonTimeExtra* dttof, const reco::MuonTimeExtra* csctof, const fwlite::ChainEvent& ev, stPlots* st, const double& GenBeta, bool RescaleP, const double& RescaleI, const double& RescaleT)
{
   if(TypeMode==1 && !(hscp.type() == HSCParticleType::trackerMuon || hscp.type() == HSCParticleType::globalMuon))return false;
   if(TypeMode==2 && hscp.type() != HSCParticleType::globalMuon)return false;

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

   if(st){st->BS_TNOH->Fill(track->found(),Event_Weight);
          st->BS_TNOHFraction->Fill(track->validFraction(),Event_Weight);
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
   if(TypeMode>1 && tof->nDof()<GlobalMinNDOF && (dttof->nDof()<GlobalMinNDOFDT || csctof->nDof()<GlobalMinNDOFCSC) )return false;
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
     if(dedxSObj) st->BS_MIm->Fill(dedxMObj->dEdx(),Event_Weight);
   }
   if(dedxSObj) if(dedxSObj->dEdx()+RescaleI<GlobalMinIs)return false;
   if(dedxSObj) if(dedxMObj->dEdx()<GlobalMinIm)return false;
   if(st){st->MI   ->Fill(0.0,Event_Weight);}
   if(tof){
   if(st){st->BS_MTOF ->Fill(tof->inverseBeta(),Event_Weight);}
   if(TypeMode>1 && tof->inverseBeta()+RescaleT<GlobalMinTOF)return false;
   if(TypeMode>1 && tof->inverseBetaErr()>GlobalMaxTOFErr)return false;
   }
   if(st){st->MTOF ->Fill(0.0,Event_Weight);
      if(GenBeta>=0)st->Beta_PreselectedB->Fill(GenBeta, Event_Weight);
   }

   fwlite::Handle< std::vector<reco::Vertex> > vertexCollHandle;
   vertexCollHandle.getByLabel(ev,"offlinePrimaryVertices");
   if(!vertexCollHandle.isValid()){printf("Vertex Collection NotFound\n");return false;}
   const std::vector<reco::Vertex>& vertexColl = *vertexCollHandle;
   if(vertexColl.size()<1){printf("NO VERTEX\n"); return false;}

   double dz  = track->dz (vertexColl[0].position());
   double dxy = track->dxy(vertexColl[0].position());
   int goodVerts=0;
   for(unsigned int i=0;i<vertexColl.size();i++){
     if(fabs(track->dz (vertexColl[i].position())) < fabs(dz) ){
       dz  = track->dz (vertexColl[i].position());
       dxy = track->dxy(vertexColl[i].position());
       if(fabs(vertexColl[i].z())<15 && sqrt(vertexColl[i].x()*vertexColl[i].x()+vertexColl[i].y()*vertexColl[i].y())<2 && vertexColl[i].ndof()>3) goodVerts++;
     }
   }

   if(st) st->BS_PV->Fill(goodVerts,Event_Weight);
   if(TypeMode==3 && goodVerts<1) return false;

   //For TOF only analysis match to a SA track without vertex constraint for IP cuts
   if(TypeMode==3) {
     fwlite::Handle< std::vector<reco::Track> > noVertexTrackCollHandle;
     noVertexTrackCollHandle.getByLabel(ev,"RefitSAMuons", "");

     //To be cleaned up when new EDM files created, different track names exist in different files
     if(!noVertexTrackCollHandle.isValid()){
       noVertexTrackCollHandle.getByLabel(ev,"refittedStandAloneMuons", "");
       if(!noVertexTrackCollHandle.isValid()){
	 noVertexTrackCollHandle.getByLabel(ev,"RefitMTSAMuons", "");
	 if(!noVertexTrackCollHandle.isValid()){
	   printf("No Vertex Track Collection Not Found\n");
	   return false;
	 }
       }
     }
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
   }

   double v3d = sqrt(dz*dz+dxy*dxy);

   if(st){st->BS_V3D->Fill(v3d,Event_Weight);}
   if(TypeMode!=3 && v3d>GlobalMaxV3D )return false;
   if(TypeMode==3 && fabs(dxy)>GlobalMaxDXY) return false;
   if(st){st->V3D  ->Fill(0.0,Event_Weight);}

   if(TypeMode!=3) {
     fwlite::Handle<HSCPIsolationValueMap> IsolationH;
     IsolationH.getByLabel(ev, "HSCPIsolation03");
     if(!IsolationH.isValid()){printf("Invalid IsolationH\n");return false;}
     const ValueMap<HSCPIsolation>& IsolationMap = *IsolationH.product();

     HSCPIsolation hscpIso = IsolationMap.get((size_t)track.key());

     if(st){st->BS_TIsol ->Fill(hscpIso.Get_TK_SumEt(),Event_Weight);}
     if(hscpIso.Get_TK_SumEt()>GlobalMaxTIsol)return false;
     if(st){st->TIsol   ->Fill(0.0,Event_Weight);}

     double EoP = (hscpIso.Get_ECAL_Energy() + hscpIso.Get_HCAL_Energy())/track->p();
     if(st){st->BS_EIsol ->Fill(EoP,Event_Weight);}
     if(EoP>GlobalMaxEIsol)return false;
     if(st){st->EIsol   ->Fill(0.0,Event_Weight);}
   }

   if(st){st->BS_Pterr ->Fill(track->ptError()/track->pt(),Event_Weight);}
   if(TypeMode!=3 && (track->ptError()/track->pt())>GlobalMaxPterr)return false;

   if(std::max(0.0,track->pt())<GlobalMinPt)return false;
   if(st){st->Pterr   ->Fill(0.0,Event_Weight);}

   //Cut on number of matched muon stations
   int count=track->hitPattern().muonStationsWithValidHits();
   if(st) {
     st->BS_MatchedStations->Fill(count, Event_Weight);  ;
   }
   if(TypeMode==3 && count<2) return false;
   if(st) st->Stations->Fill(0.0, Event_Weight);

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
   if(TypeMode==3 && fabs(minEta)<minSegEtaSep) return false;
   if(st){st->SegSep->Fill(0.0,Event_Weight);}

   if(st) {
     //Plots for tracks in dz control region
     if(fabs(dz)>CosmicMinDz && fabs(dz)<CosmicMaxDz) {
       st->BS_Pt_FailDz->Fill(track->pt(), Event_Weight);
       st->BS_TOF_FailDz->Fill(tof->inverseBeta(), Event_Weight);
       st->BS_Eta_FailDz->Fill(track->eta(), Event_Weight);
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
	 H_D_DzSidebands->Fill(CutIndex, DzType);
	 if(fabs(track->eta())<DTRegion) H_D_DzSidebands_DT->Fill(CutIndex, DzType);
         else H_D_DzSidebands_CSC->Fill(CutIndex, DzType);
       }
     }
   }

   //Cut on dz for TOF only analysis
   if(TypeMode==3 && fabs(dz)>GlobalMaxDZ) return false;

   if(st){if(dedxSObj) st->BS_EtaIs->Fill(track->eta(),dedxSObj->dEdx(),Event_Weight);
          if(dedxMObj) st->BS_EtaIm->Fill(track->eta(),dedxMObj->dEdx(),Event_Weight);
          st->BS_EtaP ->Fill(track->eta(),track->p(),Event_Weight);
          st->BS_EtaPt->Fill(track->eta(),track->pt(),Event_Weight);
          if(tof)st->BS_EtaTOF->Fill(track->eta(),tof->inverseBeta(),Event_Weight);
   }

   if(st){if(GenBeta>=0)st->Beta_PreselectedC->Fill(GenBeta, Event_Weight);
          st->BS_P  ->Fill(track->p(),Event_Weight);
          st->BS_Pt ->Fill(track->pt(),Event_Weight);
          if(dedxSObj) st->BS_Is ->Fill(dedxSObj->dEdx(),Event_Weight);
          if(dedxSObj) st->BS_Im ->Fill(dedxMObj->dEdx(),Event_Weight);
          if(tof) {
	    st->BS_TOF->Fill(tof->inverseBeta(),Event_Weight);
	    if(dttof->nDof()>6) st->BS_TOF_DT->Fill(dttof->inverseBeta(),Event_Weight);
            if(csctof->nDof()>6) st->BS_TOF_CSC->Fill(csctof->inverseBeta(),Event_Weight);
	  }
          if(dedxSObj) {
	    st->BS_PIs  ->Fill(track->p()  ,dedxSObj->dEdx(),Event_Weight);
            st->BS_PIm  ->Fill(track->p()  ,dedxMObj->dEdx(),Event_Weight);
            st->BS_PtIs ->Fill(track->pt() ,dedxSObj->dEdx(),Event_Weight);
            st->BS_PtIm ->Fill(track->pt() ,dedxMObj->dEdx(),Event_Weight);
	  }
          if(tof && dedxSObj)st->BS_TOFIs->Fill(tof->inverseBeta(),dedxSObj->dEdx(),Event_Weight);
          if(tof && dedxSObj)st->BS_TOFIm->Fill(tof->inverseBeta(),dedxMObj->dEdx(),Event_Weight);
   }
   return true;
}

// check if one HSCP candidate is passing the selection (the function also has many more arguments because it is used to fill some histograms AND to evaluate the systematics
bool PassSelection(const susybsm::HSCParticle& hscp,  const reco::DeDxData* dedxSObj, const reco::DeDxData* dedxMObj, const reco::MuonTimeExtra* tof, const fwlite::ChainEvent& ev, const int& CutIndex, stPlots* st, const double& GenBeta, bool RescaleP, const double& RescaleI, const double& RescaleT){

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

   if(RescaleP){
     if(RescaledPt(track->pt(),track->eta(),track->phi(),track->charge())<CutPt[CutIndex])return false;
     //if(std::max(0.0,RescaledPt(track->pt() - track->ptError(),track->eta(),track->phi(),track->charge()))<CutPt[CutIndex])return false;
   }else{
     if(track->pt()<CutPt[CutIndex])return false;
     //if(std::max(0.0,(track->pt() - track->ptError()))<CutPt[CutIndex])return false;
   } 
   if(st){st->Pt    ->Fill(CutIndex,Event_Weight);
          if(GenBeta>=0)st->Beta_SelectedP->Fill(CutIndex,GenBeta, Event_Weight);
   }

   if(TypeMode!=3 && Is+RescaleI<CutI[CutIndex])return false;
   if(st){st->I    ->Fill(CutIndex,Event_Weight);
          if(GenBeta>=0)st->Beta_SelectedI->Fill(CutIndex, GenBeta, Event_Weight);
   }

   if(TypeMode>1 && MuonTOF+RescaleT<CutTOF[CutIndex])return false;
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

	 Hist_Pt->Fill(track->pt(),Event_Weight);
         Hist_Is->Fill(Is,Event_Weight);
         Hist_TOF->Fill(MuonTOF,Event_Weight);


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

            if(track->pt()>100){
               CtrlPt_S4_Is->Fill(Is, Event_Weight);
               CtrlPt_S4_Im->Fill(Ih, Event_Weight);
               if(tof)CtrlPt_S4_TOF->Fill(MuonTOF, Event_Weight);
            }else if(track->pt()>80){
               CtrlPt_S3_Is->Fill(Is, Event_Weight);
               CtrlPt_S3_Im->Fill(Ih, Event_Weight);
               if(tof)CtrlPt_S3_TOF->Fill(MuonTOF, Event_Weight);
            }else if(track->pt()>60){
               CtrlPt_S2_Is->Fill(Is, Event_Weight);
               CtrlPt_S2_Im->Fill(Ih, Event_Weight);
               if(tof)CtrlPt_S2_TOF->Fill(MuonTOF, Event_Weight);
            }else{
               CtrlPt_S1_Is->Fill(Is, Event_Weight);
               CtrlPt_S1_Im->Fill(Ih, Event_Weight);
               if(tof)CtrlPt_S1_TOF->Fill(MuonTOF, Event_Weight);
            }

            if(Is>0.2){           if(tof)CtrlIs_S4_TOF->Fill(MuonTOF, Event_Weight);
            }else if(Is>0.1){     if(tof)CtrlIs_S3_TOF->Fill(MuonTOF, Event_Weight);
            }else if(Is>0.05){     if(tof)CtrlIs_S2_TOF->Fill(MuonTOF, Event_Weight);
            }else{                             if(tof)CtrlIs_S1_TOF->Fill(MuonTOF, Event_Weight);
            }

            if(Ih>4.4){           if(tof)CtrlIm_S4_TOF->Fill(MuonTOF, Event_Weight);
            }else if(Ih>4.1){     if(tof)CtrlIm_S3_TOF->Fill(MuonTOF, Event_Weight);
            }else if(Ih>3.8){     if(tof)CtrlIm_S2_TOF->Fill(MuonTOF, Event_Weight);
            }else{                             if(tof)CtrlIm_S1_TOF->Fill(MuonTOF, Event_Weight);
            }


         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){

            bool PassPtCut  = track->pt()>=CutPt[CutIndex];
            bool PassICut   = (Is>=CutI[CutIndex]);
            bool PassTOFCut = MuonTOF>=CutTOF[CutIndex];
            if(       PassTOFCut &&  PassPtCut &&  PassICut){   //Region D
               H_D      ->Fill(CutIndex,                Event_Weight);
               RegionD_P  ->Fill(CutIndex,track->p(),     Event_Weight);
               RegionD_I  ->Fill(CutIndex,Ih,Event_Weight);
               RegionD_TOF->Fill(CutIndex,MuonTOF,        Event_Weight);
	       st->AS_Eta_RegionD->Fill(CutIndex,track->eta());
            }else if( PassTOFCut &&  PassPtCut && !PassICut){   //Region C
               H_C     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode!=2)Pred_EtaP  ->Fill(CutIndex,track->eta(), track->p(),     Event_Weight);
//               Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);      //Not used but a priori uncorrelated so could be used (TO BE CHECK)
               st->AS_Eta_RegionC->Fill(CutIndex,track->eta());
            }else if( PassTOFCut && !PassPtCut &&  PassICut){   //Region B
               H_B     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode!=2)Pred_I  ->Fill(CutIndex,Ih, Event_Weight);
               if(TypeMode!=2)Pred_EtaS->Fill(CutIndex,track->eta(),         Event_Weight);
//               Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);      //Not used but a priori uncorrelated so could be used (TO BE CHECK)
               st->AS_Eta_RegionB->Fill(CutIndex,track->eta());
            }else if( PassTOFCut && !PassPtCut && !PassICut){   //Region A
               H_A     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);
               if(TypeMode!=2)Pred_EtaB->Fill(CutIndex,track->eta(),         Event_Weight);
               if(TypeMode==2)Pred_EtaS2->Fill(CutIndex,track->eta(),        Event_Weight);
               st->AS_Eta_RegionA->Fill(CutIndex,track->eta());
            }else if(!PassTOFCut &&  PassPtCut &&  PassICut){   //Region H
               H_H   ->Fill(CutIndex,          Event_Weight);
//               Pred_I->Fill(CutIndex,Ih,   Event_Weight);      //Not used but a priori uncorrelated so could be used (TO BE CHECK)
               st->AS_Eta_RegionH->Fill(CutIndex,track->eta());
            }else if(!PassTOFCut &&  PassPtCut && !PassICut){   //Region G
               H_G     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)Pred_EtaP  ->Fill(CutIndex,track->eta(),track->p(),     Event_Weight);
               st->AS_Eta_RegionG->Fill(CutIndex,track->eta());
            }else if(!PassTOFCut && !PassPtCut &&  PassICut){   //Region F
               H_F     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)Pred_I  ->Fill(CutIndex,Ih, Event_Weight);
               if(TypeMode==2)Pred_EtaS->Fill(CutIndex,track->eta(),         Event_Weight);
               st->AS_Eta_RegionF->Fill(CutIndex,track->eta());
            }else if(!PassTOFCut && !PassPtCut && !PassICut){   //Region E
               H_E     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)Pred_EtaB->Fill(CutIndex,track->eta(),         Event_Weight);
               st->AS_Eta_RegionE->Fill(CutIndex,track->eta());
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
      bool isSignal = (samples[s].Type==2);
 
      //check that the plot container exist for this sample, otherwise create it
      if(plotsMap.find(samples[s].Name)==plotsMap.end()){plotsMap[samples[s].Name] = stPlots();}
      stPlots_Init(HistoFile,plotsMap[samples[s].Name],samples[s].Name, CutPt.size());
      stPlots& SamplePlots = plotsMap[samples[s].Name];
      SamplePlots.IntLumi->Fill(0.0,IntegratedLuminosity);

      if(isMC){
         if(plotsMap.find("MCTr")==plotsMap.end()){plotsMap["MCTr"] = stPlots();}
         stPlots_Init(HistoFile,plotsMap["MCTr"],"MCTr", CutPt.size());
      }stPlots& MCTrPlots = plotsMap["MCTr"];

      //Initialize plot container for pure cosmic sample
      //Cosmic sample is contained in data file so for TOF-Only search
      //need a new set of plots for these events
      string CosmicName="";
      if(isData && TypeMode==3) {
	string Name;
	if(samples[s].Name.find("12")!=string::npos) CosmicName="Cosmic12";
	else CosmicName="Cosmic11";
	if(plotsMap.find(CosmicName)==plotsMap.end()){plotsMap[CosmicName] = stPlots();}
	stPlots_Init(HistoFile,plotsMap[CosmicName],CosmicName, CutPt.size());
      }stPlots& CosmicPlots = plotsMap[CosmicName];

      //Initialize histo specific to the sample (mostly related to ABCD --> done only for MCTr or Data);
      if(isData)InitHistos(&SamplePlots);
      if(isMC)  InitHistos(&MCTrPlots);

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
      for (int period=0; period<(samples[s].Type==2?RunningPeriods:1); period++){
         //load the files corresponding to this sample
         std::vector<string> FileName;
         GetInputFiles(samples[s], BaseDirectory, FileName, period);
         fwlite::ChainEvent ev(FileName);
    
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
         //Loop on the events
         printf("Progressing Bar                   :0%%       20%%       40%%       60%%       80%%       100%%\n");
         printf("Building Mass for %10s (%1i/%1i) :",samples[s].Name.c_str(),period+1,(samples[s].Type==2?RunningPeriods:1));
         int TreeStep = ev.size()/50;if(TreeStep==0)TreeStep=1;
         for(Long64_t ientry=0;ientry<ev.size();ientry++){
            ev.to(ientry);
            if(MaxEntry>0 && ientry>MaxEntry)break;
            if(ientry%TreeStep==0){printf(".");fflush(stdout);}

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

                 
               //skip event wich does not have the right number of charged HSCP --> DEPRECATED
               //if(samples[s].NChargedHSCP>=0 && samples[s].NChargedHSCP!=NChargedHSCP)continue;
               //NEW: reweight the events based on the number of charged HSCP directly at Analysis_step23.C (instead of Step6.C as in the past)
               Event_Weight*=samples[s].GetFGluinoWeight(NChargedHSCP);

               GetGenHSCPBeta(genColl,HSCPGenBeta1,HSCPGenBeta2,false);
               if(HSCPGenBeta1>=0)SamplePlots.Beta_Gen      ->Fill(HSCPGenBeta1, Event_Weight);  if(HSCPGenBeta2>=0)SamplePlots.Beta_Gen       ->Fill(HSCPGenBeta2, Event_Weight);
               GetGenHSCPBeta(genColl,HSCPGenBeta1,HSCPGenBeta2,true);
               if(HSCPGenBeta1>=0)SamplePlots.Beta_GenCharged->Fill(HSCPGenBeta1, Event_Weight); if(HSCPGenBeta2>=0)SamplePlots.Beta_GenCharged->Fill(HSCPGenBeta2, Event_Weight);
            }

            //check if the event is passing trigger
            SamplePlots      .TotalE  ->Fill(0.0,Event_Weight);  
            if(isMC)MCTrPlots.TotalE  ->Fill(0.0,Event_Weight);
            SamplePlots      .TotalEPU->Fill(0.0,Event_Weight*PUSystFactor);
            if(isMC)MCTrPlots.TotalEPU->Fill(0.0,Event_Weight*PUSystFactor);
	    //See if event passed signal triggers
            if(!PassTrigger(ev, isData) ) {
	      //For TOF only analysis if the event doesn't pass the signal triggers check if it was triggered by the no BPTX cosmic trigger
	      //If not TOF only then move to next event
	      if(TypeMode!=3) continue;
	      if(!PassTrigger(ev, isData, true)) continue;
	      //If is cosmic event then switch plots to use to the ones for cosmics
	      SamplePlots=CosmicPlots;
	    }
            SamplePlots       .TotalTE->Fill(0.0,Event_Weight);
            if(isMC)MCTrPlots .TotalTE->Fill(0.0,Event_Weight);

            //keep beta distribution for signal
            if(isSignal){if(HSCPGenBeta1>=0)SamplePlots.Beta_Triggered->Fill(HSCPGenBeta1, Event_Weight); if(HSCPGenBeta2>=0)SamplePlots.Beta_Triggered->Fill(HSCPGenBeta2, Event_Weight);}

            //load all event collection that will be used later on (HSCP COll, dEdx and TOF)
            fwlite::Handle<susybsm::HSCParticleCollection> hscpCollHandle;
            hscpCollHandle.getByLabel(ev,"HSCParticleProducer");
            if(!hscpCollHandle.isValid()){printf("HSCP Collection NotFound\n");continue;}
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
		 if(!muon->combinedQuality().updatedSta) continue;
		 track = muon->standAloneMuon();
	       }

               //skip events without track
	       if(track.isNull())continue;

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

               //compute systematic uncertainties on signal
               if(isSignal){
                  bool   PRescale = true;
                  double IRescale = RNG->Gaus(0, 0.083)+0.015; // added to the Ias value
                  double MRescale = 1.036;
                  double TRescale = -0.02; // added to the 1/beta value
                  if(tof) if(csctof->nDof()==0) TRescale = -0.003;

                  // compute systematic due to momentum scale
                  if(PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, ev,  NULL, -1,   PRescale, 0, 0)){
 		     double Mass     = -1; if(dedxMObj) Mass=GetMass(track->p()*PRescale,dedxMObj->dEdx(),!isData);
		     double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p()*PRescale,tof->inverseBeta());
		     double MassComb = -1;
		     if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5);
		     else if(dedxMObj) MassComb = Mass;
		     if(tof) MassComb=MassTOF;

                     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                        if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, -1,   PRescale, 0, 0)){
                           HSCPTk_SystP[CutIndex] = true;
                           if(Mass>MaxMass_SystP[CutIndex]) MaxMass_SystP[CutIndex]=Mass;
                           SamplePlots.Mass_SystP->Fill(CutIndex, Mass,Event_Weight);
                           if(tof){
                              SamplePlots.MassTOF_SystP ->Fill(CutIndex, MassTOF , Event_Weight);
                           }
                           SamplePlots.MassComb_SystP->Fill(CutIndex, MassComb, Event_Weight);
                        }
                     }
                  }

                  // compute systematic due to dEdx (both Ias and Ih)
                  if(PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, ev,  NULL, -1,   0, IRescale, 0)){
		     double Mass     = -1; if(dedxMObj) Mass=GetMass(track->p(),dedxMObj->dEdx()*MRescale,!isData);
		     double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta());
		     double MassComb = -1;
		     if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5);
		     else if(dedxMObj) MassComb = Mass;
		     if(tof) MassComb=MassTOF;

                     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                        if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, -1,   0, IRescale, 0)){
                           HSCPTk_SystI[CutIndex] = true;
                           if(Mass>MaxMass_SystI[CutIndex]) MaxMass_SystI[CutIndex]=Mass;
                           SamplePlots.Mass_SystI->Fill(CutIndex, Mass,Event_Weight);
                           if(tof){
                              SamplePlots.MassTOF_SystI ->Fill(CutIndex, MassTOF , Event_Weight);
                           }
                           SamplePlots.MassComb_SystI->Fill(CutIndex, MassComb, Event_Weight);
                        }
                     }
                  }

                  // compute systematic due to Mass shift
                  if(PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, ev,  NULL, -1,   0, 0, 0)){
		     double Mass     = -1; if(dedxMObj) Mass=GetMass(track->p(),dedxMObj->dEdx()*MRescale,!isData);
		     double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta());
		     double MassComb = -1;
		     if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5);
		     else if(dedxMObj) MassComb = Mass;
		     if(tof) MassComb=MassTOF;

                     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                        if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, -1,   0, 0, 0)){
                           HSCPTk_SystM[CutIndex] = true;
                           if(Mass>MaxMass_SystM[CutIndex]) MaxMass_SystM[CutIndex]=Mass;
                           SamplePlots.Mass_SystM->Fill(CutIndex, Mass,Event_Weight);
                           if(tof){
                              SamplePlots.MassTOF_SystM ->Fill(CutIndex, MassTOF , Event_Weight);
                           }
                           SamplePlots.MassComb_SystM->Fill(CutIndex, MassComb, Event_Weight);
                        }
                     }
                  }

                  // compute systematic due to TOF
                  if(PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, ev,  NULL, -1,   0, 0, TRescale)){
 		     double Mass     = -1; if(dedxMObj) Mass=GetMass(track->p(),dedxMObj->dEdx(),!isData);
		     double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),(tof->inverseBeta()+TRescale));
		     double MassComb = -1;
		     if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5);
		     else if(dedxMObj) MassComb = Mass;
		     if(tof) MassComb=MassTOF;

                     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                        if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, -1,   0, 0, TRescale)){
                           HSCPTk_SystT[CutIndex] = true;
                           if(Mass>MaxMass_SystT[CutIndex]) MaxMass_SystT[CutIndex]=Mass;
                           SamplePlots.Mass_SystT->Fill(CutIndex, Mass,Event_Weight);
                           if(tof){
                              SamplePlots.MassTOF_SystT ->Fill(CutIndex, MassTOF , Event_Weight);
                           }
                           SamplePlots.MassComb_SystT->Fill(CutIndex, MassComb, Event_Weight);
                        }
                     }
                  }

                  // compute systematics due to PU
                  if(PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, ev,  NULL, -1,   PRescale, 0, 0)){
		     double Mass     = -1; if(dedxMObj) Mass=GetMass(track->p(),dedxMObj->dEdx(),!isData);
		     double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta());
		     double MassComb = -1;
		     if(tof && dedxMObj)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj->dEdx(),!isData) + (1/tof->inverseBeta()))*0.5);
		     else if(dedxMObj) MassComb = Mass;
		     if(tof) MassComb=MassTOF;

                     for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                        if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, ev, CutIndex, NULL, -1,   PRescale, 0, 0)){
                           HSCPTk_SystPU[CutIndex] = true;
                           if(Mass>MaxMass_SystPU[CutIndex]) MaxMass_SystPU[CutIndex]=Mass;
                           SamplePlots.Mass_SystPU->Fill(CutIndex, Mass,Event_Weight*PUSystFactor);
                           if(tof){
                              SamplePlots.MassTOF_SystPU ->Fill(CutIndex, MassTOF , Event_Weight*PUSystFactor);
                           }
                           SamplePlots.MassComb_SystPU->Fill(CutIndex, MassComb, Event_Weight*PUSystFactor);
                        }
                     }
                  }
               }//End of systematic computation for signal

               //check if the canddiate pass the preselection cuts
               if(isMC)PassPreselection(hscp, dedxSObj, dedxMObj, tof, dttof, csctof, ev, &MCTrPlots   );
               if(    !PassPreselection(hscp, dedxSObj, dedxMObj, tof, dttof, csctof, ev, &SamplePlots, isSignal?genColl[ClosestGen].p()/genColl[ClosestGen].energy():-1))continue;

               //fill the ABCD histograms and a few other control plots
               if(isData || isMC)Analysis_FillControlAndPredictionHist(hscp, dedxSObj, dedxMObj, tof, &SamplePlots);

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
                  if(isMC)PassSelection   (hscp, dedxSObj, dedxMObj, tof, ev, CutIndex, &MCTrPlots);
                  if(    !PassSelection   (hscp, dedxSObj, dedxMObj, tof, ev, CutIndex, &SamplePlots, isSignal?genColl[ClosestGen].p()/genColl[ClosestGen].energy():-1))continue;

                  if(CutIndex!=0)PassNonTrivialSelection=true;
                  HSCPTk[CutIndex] = true;
                  if(Mass>MaxMass[CutIndex]) MaxMass[CutIndex]=Mass;

                  //Fill Mass Histograms
                  if(isMC)MCTrPlots.Mass->Fill(CutIndex, Mass,Event_Weight);
                  SamplePlots      .Mass->Fill(CutIndex, Mass,Event_Weight);
                  if(tof){
                  if(isMC)MCTrPlots.MassTOF->Fill(CutIndex, MassTOF, Event_Weight);
                     SamplePlots   .MassTOF->Fill(CutIndex, MassTOF, Event_Weight);
                  }
                  if(isMC)MCTrPlots.MassComb->Fill(CutIndex, MassComb, Event_Weight);
                  SamplePlots      .MassComb->Fill(CutIndex, MassComb, Event_Weight);
               } //end of Cut loop
               if(PassNonTrivialSelection) stPlots_FillTree(SamplePlots, ev.eventAuxiliary().run(),ev.eventAuxiliary().event(), c, track->pt(), dedxSObj ? dedxSObj->dEdx() : -1, tof ? tof->inverseBeta() : -1, Mass, -1);
            }// end of Track Loop

            //save event dependent information thanks to the bookkeeping
            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
              if(HSCPTk[CutIndex]){
                 SamplePlots.HSCPE             ->Fill(CutIndex,Event_Weight);
                 SamplePlots.MaxEventMass      ->Fill(CutIndex,MaxMass[CutIndex], Event_Weight);
                 if(isMC){
                 MCTrPlots.HSCPE               ->Fill(CutIndex,Event_Weight);
                 MCTrPlots.MaxEventMass        ->Fill(CutIndex,MaxMass[CutIndex], Event_Weight);
                 }
              }
              if(HSCPTk_SystP[CutIndex]){
                 SamplePlots.HSCPE_SystP       ->Fill(CutIndex,Event_Weight);
                 SamplePlots.MaxEventMass_SystP->Fill(CutIndex,MaxMass_SystP[CutIndex], Event_Weight);
              }
              if(HSCPTk_SystI[CutIndex]){
                 SamplePlots.HSCPE_SystI       ->Fill(CutIndex,Event_Weight);
                 SamplePlots.MaxEventMass_SystI->Fill(CutIndex,MaxMass_SystI[CutIndex], Event_Weight);
              }
              if(HSCPTk_SystM[CutIndex]){
                 SamplePlots.HSCPE_SystM       ->Fill(CutIndex,Event_Weight);
                 SamplePlots.MaxEventMass_SystM->Fill(CutIndex,MaxMass_SystM[CutIndex], Event_Weight);
              }
              if(HSCPTk_SystT[CutIndex]){
                 SamplePlots.HSCPE_SystT       ->Fill(CutIndex,Event_Weight);
                 SamplePlots.MaxEventMass_SystT->Fill(CutIndex,MaxMass_SystT[CutIndex], Event_Weight);
              }
              if(HSCPTk_SystPU[CutIndex]){
                 SamplePlots.HSCPE_SystPU       ->Fill(CutIndex,Event_Weight);
                 SamplePlots.MaxEventMass_SystPU->Fill(CutIndex,MaxMass_SystPU[CutIndex], Event_Weight);
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

   }
   delete RNG;
}

void InitHistos(stPlots* st){   
   char Name   [1024];

   //Initialization of variables that are common to all samples
   if(st==NULL){
      HistoFile->cd();
      HCuts_Pt  = new TProfile("HCuts_Pt" ,"HCuts_Pt" ,CutPt.size(),0,CutPt.size());
      HCuts_I   = new TProfile("HCuts_I"  ,"HCuts_I"  ,CutPt.size(),0,CutPt.size());
      HCuts_TOF = new TProfile("HCuts_TOF","HCuts_TOF",CutPt.size(),0,CutPt.size());
      for(unsigned int i=0;i<CutPt.size();i++){  HCuts_Pt->Fill(i,CutPt[i]);     HCuts_I->Fill(i,CutI[i]);    HCuts_TOF->Fill(i,CutTOF[i]);   }

   //Initialization of variables that exist for each different samples
   }else{
      st->Directory->cd();

      sprintf(Name,"Pred_Mass");    Pred_Mass     = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),MassNBins,0,MassHistoUpperBound);                   Pred_Mass->Sumw2();
      sprintf(Name,"Pred_MassTOF"); Pred_MassTOF  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),MassNBins,0,MassHistoUpperBound);                   Pred_MassTOF->Sumw2();
      sprintf(Name,"Pred_MassComb");Pred_MassComb = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),MassNBins,0,MassHistoUpperBound);                   Pred_MassComb->Sumw2();

      H_A = new TH1D("H_A" ,"H_A" ,CutPt.size(),0,CutPt.size());
      H_B = new TH1D("H_B" ,"H_B" ,CutPt.size(),0,CutPt.size());
      H_C = new TH1D("H_C" ,"H_C" ,CutPt.size(),0,CutPt.size());
      H_D = new TH1D("H_D" ,"H_D" ,CutPt.size(),0,CutPt.size());
      H_E = new TH1D("H_E" ,"H_E" ,CutPt.size(),0,CutPt.size());
      H_F = new TH1D("H_F" ,"H_F" ,CutPt.size(),0,CutPt.size());
      H_G = new TH1D("H_G" ,"H_G" ,CutPt.size(),0,CutPt.size());
      H_H = new TH1D("H_H" ,"H_H" ,CutPt.size(),0,CutPt.size());
      H_P = new TH1D("H_P" ,"H_P" ,CutPt.size(),0,CutPt.size());

      H_D_DzSidebands = new TH2D("H_D_DzSidebands" ,"H_D_DzSidebands" ,CutPt.size(),0,CutPt.size(), DzRegions, 0, DzRegions); H_D_DzSidebands->Sumw2();
      H_D_DzSidebands_DT = new TH2D("H_D_DzSidebands_DT" ,"H_D_DzSidebands_DT" ,CutPt.size(),0,CutPt.size(), DzRegions, 0, DzRegions); H_D_DzSidebands_DT->Sumw2();
      H_D_DzSidebands_CSC = new TH2D("H_D_DzSidebands_CSC" ,"H_D_DzSidebands_CSC" ,CutPt.size(),0,CutPt.size(), DzRegions, 0, DzRegions); H_D_DzSidebands_CSC->Sumw2();

      CtrlPt_S1_Is   = new TH1D("CtrlPt_S1_Is" ,"CtrlPt_S1_Is" ,200,0,dEdxS_UpLim);  CtrlPt_S1_Is ->Sumw2();
      CtrlPt_S1_Im   = new TH1D("CtrlPt_S1_Im" ,"CtrlPt_S1_Im" ,200,0,dEdxM_UpLim);  CtrlPt_S1_Im ->Sumw2();
      CtrlPt_S1_TOF  = new TH1D("CtrlPt_S1_TOF","CtrlPt_S1_TOF",200,0,5);            CtrlPt_S1_TOF->Sumw2();
      CtrlPt_S2_Is   = new TH1D("CtrlPt_S2_Is" ,"CtrlPt_S2_Is" ,200,0,dEdxS_UpLim);  CtrlPt_S2_Is ->Sumw2();
      CtrlPt_S2_Im   = new TH1D("CtrlPt_S2_Im" ,"CtrlPt_S2_Im" ,200,0,dEdxM_UpLim);  CtrlPt_S2_Im ->Sumw2();
      CtrlPt_S2_TOF  = new TH1D("CtrlPt_S2_TOF","CtrlPt_S2_TOF",200,0,5);            CtrlPt_S2_TOF->Sumw2();
      CtrlPt_S3_Is   = new TH1D("CtrlPt_S3_Is" ,"CtrlPt_S3_Is" ,200,0,dEdxS_UpLim);  CtrlPt_S3_Is ->Sumw2();
      CtrlPt_S3_Im   = new TH1D("CtrlPt_S3_Im" ,"CtrlPt_S3_Im" ,200,0,dEdxM_UpLim);  CtrlPt_S3_Im ->Sumw2();
      CtrlPt_S3_TOF  = new TH1D("CtrlPt_S3_TOF","CtrlPt_S3_TOF",200,0,5);            CtrlPt_S3_TOF->Sumw2();
      CtrlPt_S4_Is   = new TH1D("CtrlPt_S4_Is" ,"CtrlPt_S4_Is" ,200,0,dEdxS_UpLim);  CtrlPt_S4_Is ->Sumw2();
      CtrlPt_S4_Im   = new TH1D("CtrlPt_S4_Im" ,"CtrlPt_S4_Im" ,200,0,dEdxM_UpLim);  CtrlPt_S4_Im ->Sumw2();
      CtrlPt_S4_TOF  = new TH1D("CtrlPt_S4_TOF","CtrlPt_S4_TOF",200,0,5);            CtrlPt_S4_TOF->Sumw2();

      CtrlIs_S1_TOF  = new TH1D("CtrlIs_S1_TOF","CtrlIs_S1_TOF",200,0,5);            CtrlIs_S1_TOF->Sumw2();
      CtrlIs_S2_TOF  = new TH1D("CtrlIs_S2_TOF","CtrlIs_S2_TOF",200,0,5);            CtrlIs_S2_TOF->Sumw2();
      CtrlIs_S3_TOF  = new TH1D("CtrlIs_S3_TOF","CtrlIs_S3_TOF",200,0,5);            CtrlIs_S3_TOF->Sumw2();
      CtrlIs_S4_TOF  = new TH1D("CtrlIs_S4_TOF","CtrlIs_S4_TOF",200,0,5);            CtrlIs_S4_TOF->Sumw2();

      CtrlIm_S1_TOF  = new TH1D("CtrlIm_S1_TOF","CtrlIm_S1_TOF",200,0,5);            CtrlIm_S1_TOF->Sumw2();
      CtrlIm_S2_TOF  = new TH1D("CtrlIm_S2_TOF","CtrlIm_S2_TOF",200,0,5);            CtrlIm_S2_TOF->Sumw2();
      CtrlIm_S3_TOF  = new TH1D("CtrlIm_S3_TOF","CtrlIm_S3_TOF",200,0,5);            CtrlIm_S3_TOF->Sumw2();
      CtrlIm_S4_TOF  = new TH1D("CtrlIm_S4_TOF","CtrlIm_S4_TOF",200,0,5);            CtrlIm_S4_TOF->Sumw2();

      sprintf(Name,"Hist_Is");      Hist_Is       = new TH1D(Name,Name, 200,0,dEdxS_UpLim);                                                            Hist_Is    ->Sumw2(); 
      sprintf(Name,"Hist_Pt");      Hist_Pt       = new TH1D(Name,Name,200,0,PtHistoUpperBound);                                                       Hist_Pt    ->Sumw2();
      sprintf(Name,"Hist_TOF");     Hist_TOF      = new TH1D(Name,Name,200,-10,20);                                                                    Hist_TOF   ->Sumw2();
      sprintf(Name,"Pred_I");       Pred_I        = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   200,GlobalMinIm,dEdxM_UpLim);                    Pred_I     ->Sumw2();
      sprintf(Name,"Pred_EtaB");    Pred_EtaB     = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   50,-3,3);                                        Pred_EtaB  ->Sumw2();
      sprintf(Name,"Pred_EtaS");    Pred_EtaS     = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   50,-3,3);                                        Pred_EtaS  ->Sumw2();
      sprintf(Name,"Pred_EtaS2");   Pred_EtaS2    = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   50,-3,3);                                        Pred_EtaS2 ->Sumw2();
      sprintf(Name,"Pred_EtaP");    Pred_EtaP     = new TH3D(Name,Name,CutPt.size(),0,CutPt.size(),   50, -3, 3, 200,GlobalMinPt,PtHistoUpperBound);   Pred_EtaP  ->Sumw2();
      sprintf(Name,"Pred_TOF");     Pred_TOF      = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   200,GlobalMinTOF,5);                             Pred_TOF   ->Sumw2();
      sprintf(Name,"RegionD_I");    RegionD_I     = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   200,GlobalMinIm,dEdxM_UpLim);                    RegionD_I  ->Sumw2();
      sprintf(Name,"RegionD_P");    RegionD_P     = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   200,GlobalMinPt,PtHistoUpperBound);              RegionD_P  ->Sumw2();
      sprintf(Name,"RegionD_TOF");  RegionD_TOF   = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   200,GlobalMinTOF,5);                             RegionD_TOF->Sumw2();

      HistoFile->cd();
   }
}

// code needed for the evaluation of the systematics related to pt measurement
double RescaledPt(const double& pt, const double& eta, const double& phi, const int& charge)
{
   double newInvPt = 1/pt+0.000236-0.000135*pow(eta,2)+charge*0.000282*TMath::Sin(phi-1.337);
   return 1/newInvPt;
}

double SegSep(const susybsm::HSCParticle& hscp, const fwlite::ChainEvent& ev, double& minPhi, double& minEta) {
  reco::TrackRef   track;
  if(TypeMode!=3) track = hscp.trackRef();
  else {
    reco::MuonRef muon = hscp.muonRef();
    if(muon.isNull()) return false;
    track = muon->standAloneMuon();
  }
  if(track.isNull())return false;

  fwlite::Handle<MuonSegmentCollection> SegCollHandle;
  SegCollHandle.getByLabel(ev, "MuonSegmentProducer");
  if(!SegCollHandle.isValid()){printf("Segment Collection Not Found\n");return 0;}
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
