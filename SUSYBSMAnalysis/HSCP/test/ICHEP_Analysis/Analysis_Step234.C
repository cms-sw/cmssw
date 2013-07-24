#include <vector>

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
#include "TRandom3.h"
#include "TProfile.h"
#include "TDirectory.h"

namespace reco    { class Vertex; class Track; class GenParticle; class DeDxData; class MuonTimeExtra;}
namespace susybsm { class HSCParticle; class HSCPIsolation;}
namespace fwlite  { class ChainEvent;}
namespace trigger { class TriggerEvent;}
namespace edm     {class TriggerResults; class TriggerResultsByName; class InputTag; class LumiReWeighting;}

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/Common/interface/MergeableCounter.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPIsolation.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

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
#endif

#include "Analysis_Global.h"
#include "Analysis_CommonFunction.h"
#include "Analysis_PlotFunction.h"
#include "Analysis_PlotStructure.h"
#include "Analysis_Samples.h"

/////////////////////////// FUNCTION DECLARATION /////////////////////////////

void Analysis_Step3(char* SavePath);
void Analysis_Step4(char* SavePath);

void InitHistos();

double DistToHSCP      (const susybsm::HSCParticle& hscp, const std::vector<reco::GenParticle>& genColl, int& IndexOfClosest);
int HowManyChargedHSCP (const std::vector<reco::GenParticle>& genColl);
void  GetGenHSCPBeta   (const std::vector<reco::GenParticle>& genColl, double& beta1, double& beta2, bool onlyCharged=true);
bool   PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const reco::MuonTimeExtra* dttof, const reco::MuonTimeExtra* csctof, const fwlite::ChainEvent& ev, stPlots* st=NULL, const double& GenBeta=-1, bool RescaleP=false, const double& RescaleI=0.0, const double& RescaleT=0.0);
bool PassSelection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const fwlite::ChainEvent& ev, const int& CutIndex=0, stPlots* st=NULL, const double& GenBeta=-1, bool RescaleP=false, const double& RescaleI=0.0, const double& RescaleT=0.0);

bool PassTrigger      (const fwlite::ChainEvent& ev);
bool hasGoodPtHat     (const fwlite::ChainEvent& ev, const double& PtMax);

double GetPUWeight(const fwlite::ChainEvent& ev, const bool& Iss4pileup);
double GetSampleWeight(const double& IntegratedLuminosityInPb=-1, const double& IntegratedLuminosityInPbBeforeTriggerChange=-1, const double& CrossSection=0, const double& MCEvents=0, int period=0);
double GetSampleWeightMC(const double& IntegratedLuminosityInPb, const std::vector<string> fileNames, const double& XSection, const double& SampleSize, double MaxEvent);
double RescaledPt(const double& pt, const double& eta, const double& phi, const int& charge);
unsigned long GetInitialNumberOfMCEvent(const vector<string>& fileNames);
/////////////////////////// VARIABLE DECLARATION /////////////////////////////

class DuplicatesClass{
   private :
      typedef std::map<std::pair<unsigned int, unsigned int>, bool > RunEventHashMap;
      RunEventHashMap map;
   public :
        DuplicatesClass(){}
        ~DuplicatesClass(){}
        void Clear(){map.clear();}
        bool isDuplicate(unsigned int Run, unsigned int Event){
	   RunEventHashMap::iterator it = map.find(std::make_pair(Run,Event));
           if(it==map.end()){
   	      map[std::make_pair(Run,Event)] = true;
              return false;
           }
           return true;
        }
};


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

std::vector<double>  CutPt ;
std::vector<double>  CutI  ;
std::vector<double>  CutTOF;

TH1D*  HCuts_Pt;
TH1D*  HCuts_I;
TH1D*  HCuts_TOF;

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

std::vector<stSignal> signals;
std::vector<stMC>     MCsample;
std::vector<string>   DataFileName;

stPlots              DataPlots;  
std::vector<stPlots> SignPlots; 
std::vector<stPlots> MCPlots;  
stPlots              MCTrPlots;
//for initializing PileUpReweighting utility.
const   float TrueDist2011_f[35] = {0.00285942, 0.0125603, 0.0299631, 0.051313, 0.0709713, 0.0847864, 0.0914627, 0.0919255, 0.0879994, 0.0814127, 0.0733995, 0.0647191, 0.0558327, 0.0470663, 0.0386988, 0.0309811, 0.0241175, 0.018241, 0.0133997, 0.00956071, 0.00662814, 0.00446735, 0.00292946, 0.00187057, 0.00116414, 0.000706805, 0.000419059, 0.000242856, 0.0001377, 7.64582e-05, 4.16101e-05, 2.22135e-05, 1.16416e-05, 5.9937e-06, 5.95542e-06};//from 2011 Full dataset

const   float Pileup_MC[35]= {1.45346E-01, 6.42802E-02, 6.95255E-02, 6.96747E-02, 6.92955E-02, 6.84997E-02, 6.69528E-02, 6.45515E-02, 6.09865E-02, 5.63323E-02, 5.07322E-02, 4.44681E-02, 3.79205E-02, 3.15131E-02, 2.54220E-02, 2.00184E-02, 1.53776E-02, 1.15387E-02, 8.47608E-03, 6.08715E-03, 4.28255E-03, 2.97185E-03, 2.01918E-03, 1.34490E-03, 8.81587E-04, 5.69954E-04, 3.61493E-04, 2.28692E-04, 1.40791E-04, 8.44606E-05, 5.10204E-05, 3.07802E-05, 1.81401E-05, 1.00201E-05, 5.80004E-06};


edm::LumiReWeighting LumiWeightsMC_;
std::vector< float > BgLumiMC; //MC                                           
std::vector< float > TrueDist2011;                                    

/////////////////////////// CODE PARAMETERS /////////////////////////////

void Analysis_Step234(string MODE="COMPILE", int TypeMode_=0, string dEdxSel_="dedxASmi", string dEdxMass_="dedxHarm2", string TOF_Label_="combined", double CutPt_=-1.0, double CutI_=-1, double CutTOF_=-1, float MinPt_=GlobalMinPt, float MaxEta_=GlobalMaxEta, float MaxPtErr_=GlobalMaxPterr)
{
   if(MODE=="COMPILE")return;

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

   GetSignalDefinition(signals);
   GetMCDefinition(MCsample);

   char Buffer[2048];   
   char Command[2048];
   DataFileName.clear();
   GetInputFiles(DataFileName, "Data");

   dEdxS_Label = dEdxSel_;
   dEdxM_Label = dEdxMass_;
   TOF_Label   = TOF_Label_;
   InitdEdx(dEdxS_Label);

   TypeMode  = TypeMode_;
   GlobalMaxEta = MaxEta_;
   GlobalMaxPterr = MaxPtErr_;
   GlobalMinPt    = MinPt_;

   if(TypeMode!=2){
      GlobalMinNDOF   = 0; 
      GlobalMinTOF    = 0;
   }else{
      GlobalMaxTIsol *= 2;
      GlobalMaxEIsol *= 2;
   }

   CutPt .push_back(GlobalMinPt);   CutI  .push_back(GlobalMinIs);  CutTOF.push_back(GlobalMinTOF);

   if(TypeMode!=2){   
      for(double Pt =GlobalMinPt+5 ; Pt <200;Pt+=5){
      for(double I  =GlobalMinIs+0.025  ; I  <0.45 ;I+=0.025){
         CutPt .push_back(Pt);   CutI  .push_back(I);  CutTOF.push_back(-1);
      }}
   }else{
      for(double Pt =GlobalMinPt+5 ; Pt <120;  Pt+=5){
      for(double I  =GlobalMinIs +0.025; I  <0.40;  I+=0.025){
      for(double TOF=GlobalMinTOF+0.025; TOF<1.35;TOF+=0.025){
         CutPt .push_back(Pt);   CutI  .push_back(I);  CutTOF.push_back(TOF);
      }}}
   }
   printf("%i Different Final Selection will be tested\n",(int)CutPt.size());

   //initialize LumiReWeighting
   for(int i=0; i<35; ++i)   BgLumiMC.push_back(Pileup_MC[i]);
   for(int i=0; i<35; ++i)    TrueDist2011.push_back(TrueDist2011_f[i]);
   LumiWeightsMC_ = edm::LumiReWeighting(BgLumiMC, TrueDist2011);

   sprintf(Buffer,"Results/"       );                                          sprintf(Command,"mkdir %s",Buffer); system(Command);
   sprintf(Buffer,"%s%s/"         ,Buffer,dEdxS_Label.c_str());                sprintf(Command,"mkdir %s",Buffer); system(Command);
   sprintf(Buffer,"%s%s/"         ,Buffer,TOF_Label.c_str());                  sprintf(Command,"mkdir %s",Buffer); system(Command);
   sprintf(Buffer,"%sEta%02.0f/"  ,Buffer,10.0*GlobalMaxEta);                  sprintf(Command,"mkdir %s",Buffer); system(Command);
   sprintf(Buffer,"%sPtMin%02.0f/",Buffer,GlobalMinPt);                        sprintf(Command,"mkdir %s",Buffer); system(Command);
   sprintf(Buffer,"%sType%i/"     ,Buffer,TypeMode);                           sprintf(Command,"mkdir %s",Buffer); system(Command);
//   sprintf(Buffer,"%sPt%03.0f/"   ,Buffer,CutPt [0]);		               sprintf(Command,"mkdir %s",Buffer); system(Command);
//   sprintf(Buffer,"%sI%05.2f/"    ,Buffer,CutI  [0]);                          sprintf(Command,"mkdir %s",Buffer); system(Command);
//   sprintf(Buffer,"%sTOF%05.2f/"  ,Buffer,CutTOF[0]);                          sprintf(Command,"mkdir %s",Buffer); system(Command);

   time_t start = time(NULL);
   if(MODE=="ANALYSE_DATA"){
      signals.clear();  //Remove all signal samples
      MCsample.clear();
      HistoFile = new TFile((string(Buffer) + "/Histos_Data.root").c_str(),"RECREATE");
   }else if(MODE=="ANALYSE_SIGNAL"){
      DataFileName.clear();  //Remove all data files
      MCsample.clear();
      HistoFile = new TFile((string(Buffer) + "/Histos.root").c_str(),"RECREATE");
   }else if(MODE=="ANALYSE_MC"){
      DataFileName.clear();  //Remove all data files
      signals.clear();  //Remove all signal samples
      HistoFile = new TFile((string(Buffer) + "/Histos_MC.root").c_str(),"RECREATE");
   }else{
      printf("You must select a MODE:\n");
      printf("MODE='ANALYSE_DATA'   : Will run the analysis on Data\n"); 
      printf("MODE='ANALYSE_SIGNAL' : Will run the analysis on Signal MC\n");
      printf("MODE='ANALYSE_MC'     : Will run the analysis on Background MC\n");
      return;
   }

   InitHistos();
   Analysis_Step3(Buffer);
   Analysis_Step4(Buffer);
   HistoFile->Write();
   HistoFile->Close();
   time_t end = time(NULL);
   printf("RUN TIME = %i sec\n",(int)(end-start));
   return;
}

bool hasGoodPtHat(const fwlite::ChainEvent& ev, const double& PtMax){
   if(PtMax<0)return true;
   fwlite::Handle< GenEventInfoProduct > genInfo;
   genInfo.getByLabel(ev, "generator");
   if(!genInfo.isValid()){printf("genInfo NotFound\n");return false;}
   if((genInfo->binningValues()[0])<PtMax)return true;
   return false;
}

bool PassTrigger(const fwlite::ChainEvent& ev)
{
      edm::TriggerResultsByName tr = ev.triggerResultsByName("MergeHLT");
      if(!tr.isValid())return false;

      if(tr.accept(tr.triggerIndex("HscpPathSingleMu")))return true;
//      if(tr.accept(tr.triggerIndex("HscpPathDoubleMu")))return true;
      if(tr.accept(tr.triggerIndex("HscpPathPFMet")))return true;
//      if(tr.accept(tr.triggerIndex("HscpPathCaloMet")))return true;
      return false;
}

bool PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const reco::MuonTimeExtra* dttof, const reco::MuonTimeExtra* csctof, const fwlite::ChainEvent& ev, stPlots* st, const double& GenBeta, bool RescaleP, const double& RescaleI, const double& RescaleT)
{
   if(TypeMode==1 && !(hscp.type() == HSCParticleType::trackerMuon || hscp.type() == HSCParticleType::globalMuon))return false;
   if(TypeMode==2 && hscp.type() != HSCParticleType::globalMuon)return false;
   reco::TrackRef   track = hscp.trackRef(); if(track.isNull())return false;

   if(st){st->Total->Fill(0.0,Event_Weight);
          if(GenBeta>=0)st->Beta_Matched->Fill(GenBeta, Event_Weight);
          st->BS_TNOH->Fill(track->found(),Event_Weight);
          st->BS_TNOHFraction->Fill(track->validFraction(),Event_Weight);
   }

   if(track->found()<GlobalMinNOH)return false;
   if(track->validFraction()<0.80)return false;
   if(track->hitPattern().numberOfValidPixelHits()<2)return false;

   if(st){st->TNOH  ->Fill(0.0,Event_Weight);
          st->BS_TNOM->Fill(dedxSObj.numberOfMeasurements(),Event_Weight);
   }
   if(dedxSObj.numberOfMeasurements()<GlobalMinNOM)return false;
   if(st){st->TNOM  ->Fill(0.0,Event_Weight);}

   if(tof){
   if(st){st->BS_nDof->Fill(tof->nDof(),Event_Weight);}
   if(TypeMode==2 && tof->nDof()<GlobalMinNDOF && (dttof->nDof()<GlobalMinNDOFDT || csctof->nDof()<GlobalMinNDOFCSC) )return false;
   }

   if(st){st->nDof  ->Fill(0.0,Event_Weight);
          st->BS_Qual->Fill(track->qualityMask(),Event_Weight);
   }

   if(track->qualityMask()<GlobalMinQual )return false;
   if(st){st->Qual  ->Fill(0.0,Event_Weight);
          st->BS_Chi2->Fill(track->chi2()/track->ndof(),Event_Weight);
   }
   if(track->chi2()/track->ndof()>GlobalMaxChi2 )return false;
   if(st){st->Chi2  ->Fill(0.0,Event_Weight);}

   if(st && GenBeta>=0)st->Beta_PreselectedA->Fill(GenBeta, Event_Weight);

   if(st){st->BS_MPt ->Fill(track->pt(),Event_Weight);}
   if(RescaleP){ if(RescaledPt(track->pt(),track->eta(),track->phi(),track->charge())<GlobalMinPt)return false;
   }else{        if(track->pt()<GlobalMinPt)return false;   }

   if(st){st->MPt   ->Fill(0.0,Event_Weight);
          st->BS_MIs->Fill(dedxSObj.dEdx(),Event_Weight);
          st->BS_MIm->Fill(dedxMObj.dEdx(),Event_Weight);
   }
   if(dedxSObj.dEdx()+RescaleI<GlobalMinIs)return false;
   if(dedxMObj.dEdx()<GlobalMinIm)return false;
   if(st){st->MI   ->Fill(0.0,Event_Weight);}
   if(tof){
   if(st){st->BS_MTOF ->Fill(tof->inverseBeta(),Event_Weight);}
   if(TypeMode==2 && tof->inverseBeta()+RescaleT<GlobalMinTOF)return false;
   if(TypeMode==2 && tof->inverseBetaErr()>GlobalMaxTOFErr)return false;
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
   for(unsigned int i=1;i<vertexColl.size();i++){
      if(fabs(track->dz (vertexColl[i].position())) < fabs(dz) ){
         dz  = track->dz (vertexColl[i].position());
         dxy = track->dxy(vertexColl[i].position());
      }
   }
   double v3d = sqrt(dz*dz+dxy*dxy);

   if(st){st->BS_V3D->Fill(v3d,Event_Weight);}
   if(v3d>GlobalMaxV3D )return false;
   if(st){st->V3D  ->Fill(0.0,Event_Weight);}

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

   if(st){st->BS_Pterr ->Fill(track->ptError()/track->pt(),Event_Weight);}
   if((track->ptError()/track->pt())>GlobalMaxPterr)return false;

   if(std::max(0.0,track->pt())<GlobalMinPt)return false;
   if(st){st->Pterr   ->Fill(0.0,Event_Weight);}

   if(st){st->BS_EtaIs->Fill(track->eta(),dedxSObj.dEdx(),Event_Weight);
          st->BS_EtaIm->Fill(track->eta(),dedxMObj.dEdx(),Event_Weight);
          st->BS_EtaP ->Fill(track->eta(),track->p(),Event_Weight);
          st->BS_EtaPt->Fill(track->eta(),track->pt(),Event_Weight);
          if(tof)st->BS_EtaTOF->Fill(track->eta(),tof->inverseBeta(),Event_Weight);
          st->BS_Eta->Fill(track->eta(),Event_Weight);
   }
   if(fabs(track->eta())>GlobalMaxEta) return false;

   if(st){if(GenBeta>=0)st->Beta_PreselectedC->Fill(GenBeta, Event_Weight);
          st->BS_P  ->Fill(track->p(),Event_Weight);
          st->BS_Pt ->Fill(track->pt(),Event_Weight);
          st->BS_Is ->Fill(dedxSObj.dEdx(),Event_Weight);
          st->BS_Im ->Fill(dedxMObj.dEdx(),Event_Weight);
          if(tof)st->BS_TOF->Fill(tof->inverseBeta(),Event_Weight);
          st->BS_PIs  ->Fill(track->p()  ,dedxSObj.dEdx(),Event_Weight);
          st->BS_PIm  ->Fill(track->p()  ,dedxMObj.dEdx(),Event_Weight);
          st->BS_PtIs ->Fill(track->pt() ,dedxSObj.dEdx(),Event_Weight);
          st->BS_PtIm ->Fill(track->pt() ,dedxMObj.dEdx(),Event_Weight);
          if(tof)st->BS_TOFIs->Fill(tof->inverseBeta(),dedxSObj.dEdx(),Event_Weight);
          if(tof)st->BS_TOFIm->Fill(tof->inverseBeta(),dedxMObj.dEdx(),Event_Weight);
   }

   return true;
}

bool PassSelection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const fwlite::ChainEvent& ev, const int& CutIndex, stPlots* st, const double& GenBeta, bool RescaleP, const double& RescaleI, const double& RescaleT){
   reco::TrackRef   track = hscp.trackRef(); if(track.isNull())return false;

   double MuonTOF = GlobalMinTOF;
   double NDOF     = 9999;
   if(tof){
      MuonTOF = tof->inverseBeta();
      NDOF = tof->nDof();
   }

   if(RescaleP)
   {
     if(RescaledPt(track->pt(),track->eta(),track->phi(),track->charge())<CutPt[CutIndex])return false;
     if(std::max(0.0,RescaledPt(track->pt() - track->ptError(),track->eta(),track->phi(),track->charge()))<CutPt[CutIndex])return false;
   }
   else
   {
     if(track->pt()<CutPt[CutIndex])return false;
     if(std::max(0.0,(track->pt() - track->ptError()))<CutPt[CutIndex])return false;
   } 
   if(st){st->Pt    ->Fill(CutIndex,Event_Weight);
          if(GenBeta>=0)st->Beta_SelectedP->Fill(CutIndex,GenBeta, Event_Weight);
   }

   if(dedxSObj.dEdx()+RescaleI<CutI[CutIndex])return false;
   if(st){st->I    ->Fill(CutIndex,Event_Weight);
          if(GenBeta>=0)st->Beta_SelectedI->Fill(CutIndex, GenBeta, Event_Weight);
   }

   if(TypeMode==2 && MuonTOF+RescaleT<CutTOF[CutIndex])return false;
   if(st){st->TOF  ->Fill(CutIndex,Event_Weight);
          if(GenBeta>=0)st->Beta_SelectedT->Fill(CutIndex, GenBeta, Event_Weight);
          st->AS_P  ->Fill(CutIndex,track->p(),Event_Weight);
          st->AS_Pt ->Fill(CutIndex,track->pt(),Event_Weight);
          st->AS_Is ->Fill(CutIndex,dedxSObj.dEdx(),Event_Weight);
          st->AS_Im ->Fill(CutIndex,dedxMObj.dEdx(),Event_Weight);
          st->AS_TOF->Fill(CutIndex,MuonTOF,Event_Weight);
//        st->AS_EtaIs->Fill(CutIndex,track->eta(),dedxSObj.dEdx(),Event_Weight);
//        st->AS_EtaIm->Fill(CutIndex,track->eta(),dedxMObj.dEdx(),Event_Weight);
//        st->AS_EtaP ->Fill(CutIndex,track->eta(),track->p(),Event_Weight);
//        st->AS_EtaPt->Fill(CutIndex,track->eta(),track->pt(),Event_Weight);
          st->AS_PIs  ->Fill(CutIndex,track->p()  ,dedxSObj.dEdx(),Event_Weight);
          st->AS_PIm  ->Fill(CutIndex,track->p()  ,dedxMObj.dEdx(),Event_Weight);
          st->AS_PtIs ->Fill(CutIndex,track->pt() ,dedxSObj.dEdx(),Event_Weight);
          st->AS_PtIm ->Fill(CutIndex,track->pt() ,dedxMObj.dEdx(),Event_Weight);
          st->AS_TOFIs->Fill(CutIndex,MuonTOF     ,dedxSObj.dEdx(),Event_Weight);
          st->AS_TOFIm->Fill(CutIndex,MuonTOF     ,dedxMObj.dEdx(),Event_Weight);
   }

   return true;
}

void Analysis_FillControlAndPredictionHist(const susybsm::HSCParticle& hscp, const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof){
         reco::TrackRef   track = hscp.trackRef(); if(track.isNull())return;

         double MuonTOF = GlobalMinTOF;
         if(tof){MuonTOF = tof->inverseBeta(); }

	 Hist_Pt->Fill(track->pt(),Event_Weight);
         Hist_Is->Fill(dedxSObj.dEdx(),Event_Weight);
         Hist_TOF->Fill(MuonTOF,Event_Weight);


//          /\ I
//       /\  |----------------------------
//        |  |   |           |             |
//        |  |   |           |             |
//        |  |   |    B      |     D       |
//        |  |   |           |             |
//        |  ------------------------------
//        |  |   |           |             |
//        |  |   |   A       |    C        |
//        |  |   |           |             |
//        |  |---|-----------|-------------|
//        |  |   |           |             |
//        |  /---15---------------------------> PT
//        | /
//         /------------------------------->
//        /
//      TOF

            if(track->pt()>100){
               CtrlPt_S4_Is->Fill(dedxSObj.dEdx(), Event_Weight);
               CtrlPt_S4_Im->Fill(dedxMObj.dEdx(), Event_Weight);
               if(tof)CtrlPt_S4_TOF->Fill(MuonTOF, Event_Weight);
            }else if(track->pt()>60){
               CtrlPt_S3_Is->Fill(dedxSObj.dEdx(), Event_Weight);
               CtrlPt_S3_Im->Fill(dedxMObj.dEdx(), Event_Weight);
               if(tof)CtrlPt_S3_TOF->Fill(MuonTOF, Event_Weight);
            }else if(track->pt()>45){
               CtrlPt_S2_Is->Fill(dedxSObj.dEdx(), Event_Weight);
               CtrlPt_S2_Im->Fill(dedxMObj.dEdx(), Event_Weight);
               if(tof)CtrlPt_S2_TOF->Fill(MuonTOF, Event_Weight);
            }else{
               CtrlPt_S1_Is->Fill(dedxSObj.dEdx(), Event_Weight);
               CtrlPt_S1_Im->Fill(dedxMObj.dEdx(), Event_Weight);
               if(tof)CtrlPt_S1_TOF->Fill(MuonTOF, Event_Weight);
            }

            if(dedxSObj.dEdx()>0.4){           if(tof)CtrlIs_S4_TOF->Fill(MuonTOF, Event_Weight);
            }else if(dedxSObj.dEdx()>0.3){     if(tof)CtrlIs_S3_TOF->Fill(MuonTOF, Event_Weight);
            }else if(dedxSObj.dEdx()>0.2){     if(tof)CtrlIs_S2_TOF->Fill(MuonTOF, Event_Weight);
            }else{                             if(tof)CtrlIs_S1_TOF->Fill(MuonTOF, Event_Weight);
            }


         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){

            bool PassPtCut  = track->pt()>=CutPt[CutIndex];
            bool PassICut   = (dedxSObj.dEdx()>=CutI[CutIndex]);
            bool PassTOFCut = MuonTOF>=CutTOF[CutIndex];
            if(       PassTOFCut &&  PassPtCut &&  PassICut){   //Region D
               H_D      ->Fill(CutIndex,                Event_Weight);
               RegionD_P  ->Fill(CutIndex,track->p(),     Event_Weight);
               RegionD_I  ->Fill(CutIndex,dedxMObj.dEdx(),Event_Weight);
               RegionD_TOF->Fill(CutIndex,MuonTOF,        Event_Weight);
            }else if( PassTOFCut &&  PassPtCut && !PassICut){   //Region C
               H_C     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode!=2)Pred_EtaP  ->Fill(CutIndex,track->eta(), track->p(),     Event_Weight);
//               Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);
            }else if( PassTOFCut && !PassPtCut &&  PassICut){   //Region B
               H_B     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode!=2)Pred_I  ->Fill(CutIndex,dedxMObj.dEdx(), Event_Weight);
               if(TypeMode!=2)Pred_EtaS->Fill(CutIndex,track->eta(),         Event_Weight);
//               Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);
            }else if( PassTOFCut && !PassPtCut && !PassICut){   //Region A
               H_A     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);
               if(TypeMode!=2)Pred_EtaB->Fill(CutIndex,track->eta(),         Event_Weight);
               if(TypeMode==2)Pred_EtaS2->Fill(CutIndex,track->eta(),        Event_Weight);
            }else if(!PassTOFCut &&  PassPtCut &&  PassICut){   //Region H
               H_H   ->Fill(CutIndex,          Event_Weight);
//               Pred_P->Fill(CutIndex,track->p(),        Event_Weight);
//               Pred_I->Fill(CutIndex,dedxMObj.dEdx(),   Event_Weight);
            }else if(!PassTOFCut &&  PassPtCut && !PassICut){   //Region G
               H_G     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)Pred_EtaP  ->Fill(CutIndex,track->eta(),track->p(),     Event_Weight);
            }else if(!PassTOFCut && !PassPtCut &&  PassICut){   //Region F
               H_F     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)Pred_I  ->Fill(CutIndex,dedxMObj.dEdx(), Event_Weight);
               if(TypeMode==2)Pred_EtaS->Fill(CutIndex,track->eta(),         Event_Weight);
            }else if(!PassTOFCut && !PassPtCut && !PassICut){   //Region E
               H_E     ->Fill(CutIndex,                 Event_Weight);
               if(TypeMode==2)Pred_EtaB->Fill(CutIndex,track->eta(),         Event_Weight);
            }
         }

}





void Analysis_Step3(char* SavePath)
{
   printf("Step3: Building Mass Spectrum for B and S\n");

   int TreeStep;
   //////////////////////////////////////////////////     BUILD BACKGROUND MASS SPECTRUM

   if(DataFileName.size())stPlots_Init(HistoFile, DataPlots,"Data", CutPt.size());
   HistoFile->cd();

   DuplicatesClass Duplicates;
   Duplicates.Clear();

   fwlite::ChainEvent treeD(DataFileName);
   double SampleWeight = GetSampleWeight(-1);
   Event_Weight = SampleWeight;
   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Building Mass Spectrum for D :");
   TreeStep = treeD.size()/50;if(TreeStep==0)TreeStep=1;

   bool* HSCPTk = new bool[CutPt.size()]; 
   double* MaxMass = new double[CutPt.size()];
   for(Long64_t ientry=0;ientry<treeD.size();ientry++){
      treeD.to(ientry);
      if(MaxEntry>0 && ientry>MaxEntry)break;
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}

      if(Duplicates.isDuplicate(treeD.eventAuxiliary().run(),treeD.eventAuxiliary().event())){continue;}

      DataPlots.TotalE->Fill(0.0,Event_Weight);  
      if(!PassTrigger(treeD) )continue;
      DataPlots.TotalTE->Fill(0.0,Event_Weight);

      fwlite::Handle<susybsm::HSCParticleCollection> hscpCollHandle;
      hscpCollHandle.getByLabel(treeD,"HSCParticleProducer");
      if(!hscpCollHandle.isValid()){printf("HSCP Collection NotFound\n");continue;}
      const susybsm::HSCParticleCollection& hscpColl = *hscpCollHandle;

      fwlite::Handle<DeDxDataValueMap> dEdxSCollH;
      dEdxSCollH.getByLabel(treeD, dEdxS_Label.c_str());
      if(!dEdxSCollH.isValid()){printf("Invalid dEdx Selection collection\n");continue;}

      fwlite::Handle<DeDxDataValueMap> dEdxMCollH;
      dEdxMCollH.getByLabel(treeD, dEdxM_Label.c_str());
      if(!dEdxMCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}

      fwlite::Handle<MuonTimeExtraMap> TOFCollH;
      TOFCollH.getByLabel(treeD, "muontiming",TOF_Label.c_str());
      if(!TOFCollH.isValid()){printf("Invalid TOF collection\n");return;}

      fwlite::Handle<MuonTimeExtraMap> TOFDTCollH;
      TOFDTCollH.getByLabel(treeD, "muontiming",TOFdt_Label.c_str());
      if(!TOFDTCollH.isValid()){printf("Invalid DT TOF collection\n");return;}

      fwlite::Handle<MuonTimeExtraMap> TOFCSCCollH;
      TOFCSCCollH.getByLabel(treeD, "muontiming",TOFcsc_Label.c_str());
      if(!TOFCSCCollH.isValid()){printf("Invalid CSC TOF collection\n");return;}
      
      for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk[CutIndex] = false;   }
      for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass[CutIndex] = -1; }
      for(unsigned int c=0;c<hscpColl.size();c++){
         susybsm::HSCParticle hscp  = hscpColl[c];
         reco::MuonRef  muon  = hscp.muonRef();
         reco::TrackRef track = hscp.trackRef();
         if(track.isNull())continue;

         const DeDxData& dedxSObj  = dEdxSCollH->get(track.key());
         const DeDxData& dedxMObj  = dEdxMCollH->get(track.key());
         const reco::MuonTimeExtra* tof = NULL;
         const reco::MuonTimeExtra* dttof = NULL;
         const reco::MuonTimeExtra* csctof = NULL;
        if(TypeMode==2 && !hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); dttof = &TOFDTCollH->get(hscp.muonRef().key());  csctof = &TOFCSCCollH->get(hscp.muonRef().key());}


         double MuonTOF = GlobalMinTOF;
         if(tof){MuonTOF = tof->inverseBeta(); }
 
         if(!PassPreselection(hscp, dedxSObj, dedxMObj, tof, dttof, csctof, treeD, &DataPlots))continue;

         Analysis_FillControlAndPredictionHist(hscp, dedxSObj, dedxMObj, tof);


         double Mass     = GetMass(track->p(),dedxMObj.dEdx());
         double MassTOF  = -1;  if(tof)MassTOF=GetTOFMass(track->p(),tof->inverseBeta());
         double MassComb = Mass;if(tof)MassComb=GetMassFromBeta(track->p(), (GetIBeta(dedxMObj.dEdx()) + (1/tof->inverseBeta()))*0.5 ) ;
         bool PassNonTrivialSelection=false;
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
            //Full Selection
            if(!PassSelection   (hscp, dedxSObj, dedxMObj, tof, treeD, CutIndex, &DataPlots))continue;
            if(CutIndex!=0)PassNonTrivialSelection=true;
            HSCPTk[CutIndex] = true;
	    if(Mass>MaxMass[CutIndex]) MaxMass[CutIndex]=Mass;

      	    DataPlots.Mass->Fill(CutIndex, Mass,Event_Weight);
            if(tof){
               DataPlots.MassTOF ->Fill(CutIndex, MassTOF , Event_Weight);
            }
            DataPlots.MassComb->Fill(CutIndex, MassComb, Event_Weight);
         } //end of Cut loop
//         if(track->pt()>40 && Mass>75)stPlots_FillTree(DataPlots, treeD.eventAuxiliary().run(),treeD.eventAuxiliary().event(), c, track->pt(), dedxSObj.dEdx(), tof ? tof->inverseBeta() : -1);
         if (PassNonTrivialSelection) stPlots_FillTree(DataPlots, treeD.eventAuxiliary().run(),treeD.eventAuxiliary().event(), c, track->pt(), dedxSObj.dEdx(), tof ? tof->inverseBeta() : -1, Mass, -1);
      } // end of Track Loop
      for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  if(HSCPTk[CutIndex]){DataPlots.HSCPE->Fill(CutIndex,Event_Weight); DataPlots.MaxEventMass->Fill(CutIndex,MaxMass[CutIndex],Event_Weight);} }
   }// end of Event Loop
   delete [] HSCPTk;
   delete [] MaxMass;
   printf("\n");
   if(DataFileName.size())stPlots_Clear(DataPlots, true);


   //////////////////////////////////////////////////     BUILD MCTRUTH MASS SPECTRUM
   if(MCsample.size())stPlots_Init(HistoFile, MCTrPlots,"MCTr", CutPt.size());

   for(unsigned int m=0;m<MCsample.size();m++){
      stPlots_Init(HistoFile,MCPlots[m],MCsample[m].Name, CutPt.size());

      std::vector<string> FileName;
      GetInputFiles(FileName, MCsample[m].Name);

      fwlite::ChainEvent treeM(FileName);
      double SampleWeight = GetSampleWeightMC(IntegratedLuminosity,FileName, MCsample[m].XSection, treeM.size(), MCsample[m].MaxEvent);

      printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
      printf("Building Mass for %10s :",MCsample[m].Name.c_str());
      TreeStep = treeM.size()/50;if(TreeStep==0)TreeStep=1;

      bool* HSCPTk = new bool[CutPt.size()]; 
      double* MaxMass = new double[CutPt.size()];
      for(Long64_t ientry=0;ientry<treeM.size();ientry++){       
          treeM.to(ientry);
         if(MaxEntry>0 && ientry>MaxEntry)break;
         if(MCsample[m].MaxEvent>0 && ientry>MCsample[m].MaxEvent)break;
         if(ientry%TreeStep==0){printf(".");fflush(stdout);}

         if(!hasGoodPtHat(treeM, MCsample[m].MaxPtHat)){continue;}
         Event_Weight = SampleWeight * GetPUWeight(treeM, MCsample[m].IsS4PileUp);

         MCTrPlots .TotalE->Fill(0.0,Event_Weight);
         MCPlots[m].TotalE->Fill(0.0,Event_Weight);
         if(!PassTrigger(treeM) )continue;
         MCTrPlots .TotalTE->Fill(0.0,Event_Weight);
         MCPlots[m].TotalTE->Fill(0.0,Event_Weight);

         fwlite::Handle<susybsm::HSCParticleCollection> hscpCollHandle;
         hscpCollHandle.getByLabel(treeM,"HSCParticleProducer");
         if(!hscpCollHandle.isValid()){printf("HSCP Collection NotFound\n");continue;}
         const susybsm::HSCParticleCollection& hscpColl = *hscpCollHandle;

         fwlite::Handle<DeDxDataValueMap> dEdxSCollH;
         dEdxSCollH.getByLabel(treeM, dEdxS_Label.c_str());
         if(!dEdxSCollH.isValid()){printf("Invalid dEdx Selection collection\n");continue;}

         fwlite::Handle<DeDxDataValueMap> dEdxMCollH;
         dEdxMCollH.getByLabel(treeM, dEdxM_Label.c_str());
         if(!dEdxMCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}

         fwlite::Handle<MuonTimeExtraMap> TOFCollH;
         TOFCollH.getByLabel(treeM, "muontiming",TOF_Label.c_str());
         if(!TOFCollH.isValid()){printf("Invalid TOF collection\n");continue;}
         
         fwlite::Handle<MuonTimeExtraMap> TOFDTCollH;
         TOFDTCollH.getByLabel(treeM, "muontiming",TOFdt_Label.c_str());
         if(!TOFDTCollH.isValid()){printf("Invalid DT TOF collection\n");continue;}

         fwlite::Handle<MuonTimeExtraMap> TOFCSCCollH;
         TOFCSCCollH.getByLabel(treeM, "muontiming",TOFcsc_Label.c_str());
         if(!TOFCSCCollH.isValid()){printf("Invalid CSCTOF collection\n");continue;}

         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk[CutIndex] = false;   }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass[CutIndex] = -1; }
         for(unsigned int c=0;c<hscpColl.size();c++){
            susybsm::HSCParticle hscp  = hscpColl[c];
            reco::MuonRef  muon  = hscp.muonRef();
            reco::TrackRef track = hscp.trackRef();
            if(track.isNull())continue;

            const DeDxData& dedxSObj  = dEdxSCollH->get(track.key());
            const DeDxData& dedxMObj  = dEdxMCollH->get(track.key());
            const reco::MuonTimeExtra* tof = NULL;
            const reco::MuonTimeExtra* dttof = NULL;
            const reco::MuonTimeExtra* csctof = NULL;
            if(TypeMode==2 && !hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); dttof  = &TOFDTCollH->get(hscp.muonRef().key()); csctof  = &TOFCSCCollH->get(hscp.muonRef().key());}

                PassPreselection(hscp, dedxSObj, dedxMObj, tof, dttof, csctof, treeM,           &MCPlots[m]);
            if(!PassPreselection(hscp, dedxSObj, dedxMObj, tof, dttof, csctof, treeM,           &MCTrPlots))continue;
            Analysis_FillControlAndPredictionHist(hscp, dedxSObj, dedxMObj, tof);

            double Mass     = GetMass(track->p(),dedxMObj.dEdx());
            double MassTOF  = -1;   if(tof)MassTOF  = GetTOFMass(track->p(),tof->inverseBeta());
            double MassComb = Mass;if(tof)MassComb=GetMassFromBeta(track->p(), (GetIBeta(dedxMObj.dEdx()) + (1/tof->inverseBeta()))*0.5 ) ;


            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){

                   PassSelection   (hscp, dedxSObj, dedxMObj, tof, treeM, CutIndex, &MCPlots[m]);
               if(!PassSelection   (hscp, dedxSObj, dedxMObj, tof, treeM, CutIndex, &MCTrPlots))continue;
               HSCPTk[CutIndex] = true;
	       if(Mass>MaxMass[CutIndex]) MaxMass[CutIndex]=Mass;

               MCTrPlots .Mass->Fill(CutIndex , Mass,Event_Weight);
               MCPlots[m].Mass->Fill(CutIndex, Mass,Event_Weight);

               if(tof){
                  MCTrPlots .MassTOF ->Fill(CutIndex, MassTOF , Event_Weight);
                  MCPlots[m].MassTOF ->Fill(CutIndex, MassTOF , Event_Weight);
               }
               MCTrPlots .MassComb->Fill(CutIndex, MassComb, Event_Weight);
               MCPlots[m].MassComb->Fill(CutIndex, MassComb, Event_Weight);
         } //end of Cut loo
	    if(track->pt()>35)stPlots_FillTree(MCTrPlots , treeM.eventAuxiliary().run(),treeM.eventAuxiliary().event(), c, track->pt(), dedxSObj.dEdx(), tof ? tof->inverseBeta() : -1, Mass);
	    if(track->pt()>35)stPlots_FillTree(MCPlots[m], treeM.eventAuxiliary().run(),treeM.eventAuxiliary().event(), c, track->pt(), dedxSObj.dEdx(), tof ? tof->inverseBeta() : -1, Mass);

         } // end of Track Loop 
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  if(HSCPTk[CutIndex]){
	     MCTrPlots .HSCPE->Fill(CutIndex,Event_Weight);MCPlots[m].HSCPE->Fill(CutIndex,Event_Weight);
	     MCTrPlots.MaxEventMass->Fill(CutIndex,MaxMass[CutIndex],Event_Weight);MCPlots[m].MaxEventMass->Fill(CutIndex,MaxMass[CutIndex],Event_Weight); 
	   } }
      }// end of Event Loop
      delete [] HSCPTk;
      delete [] MaxMass;
      stPlots_Clear(MCPlots[m], true);
      printf("\n");
   }
   if(MCsample.size())stPlots_Clear(MCTrPlots, true);


   //////////////////////////////////////////////////     BUILD SIGNAL MASS SPECTRUM

   for(unsigned int s=0;s<signals.size();s++){
      stPlots_Init(HistoFile,SignPlots[4*s+0],signals[s].Name       , CutPt.size());
      stPlots_Init(HistoFile,SignPlots[4*s+1],signals[s].Name+"_NC0", CutPt.size());//, true);
      stPlots_Init(HistoFile,SignPlots[4*s+2],signals[s].Name+"_NC1", CutPt.size());//, true);
      stPlots_Init(HistoFile,SignPlots[4*s+3],signals[s].Name+"_NC2", CutPt.size());//, true);

      bool* HSCPTk          = new bool[CutPt.size()];
      bool* HSCPTk_SystP    = new bool[CutPt.size()];
      bool* HSCPTk_SystI    = new bool[CutPt.size()];
      bool* HSCPTk_SystT    = new bool[CutPt.size()];
      bool* HSCPTk_SystM    = new bool[CutPt.size()];
      double* MaxMass       = new double[CutPt.size()];
      double* MaxMass_SystP = new double[CutPt.size()];
      double* MaxMass_SystI = new double[CutPt.size()];
      double* MaxMass_SystT = new double[CutPt.size()];
      double* MaxMass_SystM = new double[CutPt.size()];

      printf("Progressing Bar                                    :0%%       20%%       40%%       60%%       80%%       100%%\n");
      //Do two loops through signal for samples with and without trigger change.  Period before has 325 1/pb and rest of luminosity is after
      for (int period=0; period<RunningPeriods; period++) {

      std::vector<string> SignFileName;
//      GetInputFiles(SignFileName, signals[s].FileName, period);
      GetInputFiles(SignFileName, signals[s].Name, period);

      fwlite::ChainEvent treeS(SignFileName);

      if (period==0) printf("Building Mass for %10s for before RPC change :",signals[s].Name.c_str());
      if (period==1) printf("\nBuilding Mass for %10s for after RPC change  :",signals[s].Name.c_str());
      TreeStep = treeS.size()/50;if(TreeStep==0)TreeStep=1;

      double SampleWeight = GetSampleWeight(IntegratedLuminosity,IntegratedLuminosityBeforeTriggerChange,signals[s].XSec,(double)treeS.size(), period);
      for(Long64_t ientry=0;ientry<treeS.size();ientry++){
         treeS.to(ientry);
         if(MaxEntry>0 && ientry>MaxEntry)break;
         if(ientry%TreeStep==0){printf(".");fflush(stdout);}
         Event_Weight = SampleWeight * GetPUWeight(treeS, signals[s].IsS4PileUp);

         fwlite::Handle< std::vector<reco::GenParticle> > genCollHandle;
         genCollHandle.getByLabel(treeS, "genParticles");
         if(!genCollHandle.isValid()){printf("GenParticle Collection NotFound\n");continue;}
         std::vector<reco::GenParticle> genColl = *genCollHandle;
         int NChargedHSCP=HowManyChargedHSCP(genColl);

         double HSCPGenBeta1, HSCPGenBeta2;
         GetGenHSCPBeta(genColl,HSCPGenBeta1,HSCPGenBeta2,false);
         if(HSCPGenBeta1>=0)SignPlots[4*s].Beta_Gen->Fill(HSCPGenBeta1, Event_Weight);        if(HSCPGenBeta2>=0)SignPlots[4*s].Beta_Gen->Fill(HSCPGenBeta2, Event_Weight);
         GetGenHSCPBeta(genColl,HSCPGenBeta1,HSCPGenBeta2,true);
         if(HSCPGenBeta1>=0)SignPlots[4*s].Beta_GenCharged->Fill(HSCPGenBeta1, Event_Weight); if(HSCPGenBeta2>=0)SignPlots[4*s].Beta_GenCharged->Fill(HSCPGenBeta2, Event_Weight);

         SignPlots[4*s]               .TotalE ->Fill(0.0,Event_Weight);
         SignPlots[4*s+NChargedHSCP+1].TotalE ->Fill(0.0,Event_Weight);
         if(!PassTrigger(treeS) )continue;
         SignPlots[4*s]               .TotalTE->Fill(0.0,Event_Weight);
         SignPlots[4*s+NChargedHSCP+1].TotalTE->Fill(0.0,Event_Weight);

         if(HSCPGenBeta1>=0)SignPlots[4*s].Beta_Triggered->Fill(HSCPGenBeta1, Event_Weight); if(HSCPGenBeta2>=0)SignPlots[4*s].Beta_Triggered->Fill(HSCPGenBeta2, Event_Weight);

         fwlite::Handle<susybsm::HSCParticleCollection> hscpCollHandle;
         hscpCollHandle.getByLabel(treeS,"HSCParticleProducer");
         if(!hscpCollHandle.isValid()){printf("HSCP Collection NotFound\n");continue;}
         const susybsm::HSCParticleCollection& hscpColl = *hscpCollHandle;

         fwlite::Handle<DeDxDataValueMap> dEdxSCollH;
         dEdxSCollH.getByLabel(treeS, dEdxS_Label.c_str());
         if(!dEdxSCollH.isValid()){printf("Invalid dEdx Selection collection\n");continue;}

         fwlite::Handle<DeDxDataValueMap> dEdxMCollH;
         dEdxMCollH.getByLabel(treeS, dEdxM_Label.c_str());
         if(!dEdxMCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}

         fwlite::Handle<MuonTimeExtraMap> TOFCollH;
         TOFCollH.getByLabel(treeS, "muontiming",TOF_Label.c_str());
         if(!TOFCollH.isValid()){printf("Invalid TOF collection\n");continue;}

         fwlite::Handle<MuonTimeExtraMap> TOFDTCollH;
         TOFDTCollH.getByLabel(treeS, "muontiming",TOFdt_Label.c_str());
         if(!TOFDTCollH.isValid()){printf("Invalid DT TOF collection\n");continue;}

         fwlite::Handle<MuonTimeExtraMap> TOFCSCCollH;
         TOFCSCCollH.getByLabel(treeS, "muontiming",TOFcsc_Label.c_str());
         if(!TOFCSCCollH.isValid()){printf("Invalid CSC TOF collection\n");continue;}

         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk       [CutIndex] = false;   }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk_SystP [CutIndex] = false;   }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk_SystI [CutIndex] = false;   }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk_SystT [CutIndex] = false;   }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk_SystM [CutIndex] = false;   }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass      [CutIndex] = -1; }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass_SystP[CutIndex] = -1; }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass_SystI[CutIndex] = -1; }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass_SystT[CutIndex] = -1; }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  MaxMass_SystM[CutIndex] = -1; }
         for(unsigned int c=0;c<hscpColl.size();c++){
            susybsm::HSCParticle hscp  = hscpColl[c];
            reco::MuonRef  muon  = hscp.muonRef();
            reco::TrackRef track = hscp.trackRef();
            if(track.isNull())continue;

            int ClosestGen;
            if(DistToHSCP(hscp, genColl, ClosestGen)>0.03)continue;

            const DeDxData& dedxSObj  = dEdxSCollH->get(track.key());
            const DeDxData& dedxMObj  = dEdxMCollH->get(track.key());
            const reco::MuonTimeExtra* tof = NULL;
            const reco::MuonTimeExtra* dttof = NULL;
            const reco::MuonTimeExtra* csctof = NULL;
            if(TypeMode==2 && !hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); dttof  = &TOFDTCollH->get(hscp.muonRef().key()); csctof  = &TOFCSCCollH->get(hscp.muonRef().key()); }


            ///////////// START COMPUTATION OF THE SYSTEMATIC //////////
            bool PRescale = true;
            double IRescale = -0.0438; // added to the Ias value
            double MRescale = 0.97;
            double TRescale = -0.00694; // added to the 1/beta value

            // Systematic on P
            if(PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, treeS,  NULL, -1,   PRescale, 0, 0)){
               double Mass     = GetMass(track->p()*PRescale,dedxMObj.dEdx());
               double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p()*PRescale,tof->inverseBeta());
               double MassComb = Mass;if(tof)MassComb=GetMassFromBeta(track->p()*PRescale, (GetIBeta(dedxMObj.dEdx()) + (1/tof->inverseBeta()))*0.5 ) ;

               for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                  if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, treeS, CutIndex, NULL, -1,   PRescale, 0, 0)){
                     HSCPTk_SystP[CutIndex] = true;
		     if(Mass>MaxMass_SystP[CutIndex]) MaxMass_SystP[CutIndex]=Mass;
                     SignPlots[4*s               ].Mass_SystP->Fill(CutIndex, Mass,Event_Weight);
                     SignPlots[4*s+NChargedHSCP+1].Mass_SystP->Fill(CutIndex, Mass,Event_Weight);
                     if(tof){
                        SignPlots[4*s               ].MassTOF_SystP ->Fill(CutIndex, MassTOF , Event_Weight);
                        SignPlots[4*s+NChargedHSCP+1].MassTOF_SystP ->Fill(CutIndex, MassTOF , Event_Weight);
                     }
                     SignPlots[4*s               ].MassComb_SystP->Fill(CutIndex, MassComb, Event_Weight);
                     SignPlots[4*s+NChargedHSCP+1].MassComb_SystP->Fill(CutIndex, MassComb, Event_Weight);
                  }
               }
            }

            // Systematic on I
            if(PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, treeS,  NULL, -1,   0, IRescale, 0)){
               double Mass     = GetMass(track->p(),dedxMObj.dEdx());
               double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta());
               double MassComb = Mass;if(tof)MassComb=GetMassFromBeta(track->p(), (GetIBeta(dedxMObj.dEdx()) + (1/tof->inverseBeta()))*0.5 ) ;

               for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                  if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, treeS, CutIndex, NULL, -1,   0, IRescale, 0)){
                     HSCPTk_SystI[CutIndex] = true;
                     if(Mass>MaxMass_SystI[CutIndex]) MaxMass_SystI[CutIndex]=Mass;
                     SignPlots[4*s               ].Mass_SystI->Fill(CutIndex, Mass,Event_Weight);
                     SignPlots[4*s+NChargedHSCP+1].Mass_SystI->Fill(CutIndex, Mass,Event_Weight);
                     if(tof){
                        SignPlots[4*s               ].MassTOF_SystI ->Fill(CutIndex, MassTOF , Event_Weight);
                        SignPlots[4*s+NChargedHSCP+1].MassTOF_SystI ->Fill(CutIndex, MassTOF , Event_Weight);
                     }
                     SignPlots[4*s               ].MassComb_SystI->Fill(CutIndex, MassComb, Event_Weight);
                     SignPlots[4*s+NChargedHSCP+1].MassComb_SystI->Fill(CutIndex, MassComb, Event_Weight);
                  }
               }
            }


            // Systematic on M
            if(PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, treeS,  NULL, -1,   0, 0, 0)){
               double Mass     = GetMass(track->p(),dedxMObj.dEdx()*MRescale);
               double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta());
               double MassComb = Mass;if(tof)MassComb=GetMassFromBeta(track->p(), (GetIBeta(dedxMObj.dEdx()*MRescale) + (1/tof->inverseBeta()))*0.5 ) ;

               for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                  if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, treeS, CutIndex, NULL, -1,   0, 0, 0)){
                     HSCPTk_SystM[CutIndex] = true;
                     if(Mass>MaxMass_SystM[CutIndex]) MaxMass_SystM[CutIndex]=Mass;
                     SignPlots[4*s               ].Mass_SystM->Fill(CutIndex, Mass,Event_Weight);
                     SignPlots[4*s+NChargedHSCP+1].Mass_SystM->Fill(CutIndex, Mass,Event_Weight);
                     if(tof){
                        SignPlots[4*s               ].MassTOF_SystM ->Fill(CutIndex, MassTOF , Event_Weight);
                        SignPlots[4*s+NChargedHSCP+1].MassTOF_SystM ->Fill(CutIndex, MassTOF , Event_Weight);
                     }
                     SignPlots[4*s               ].MassComb_SystM->Fill(CutIndex, MassComb, Event_Weight);
                     SignPlots[4*s+NChargedHSCP+1].MassComb_SystM->Fill(CutIndex, MassComb, Event_Weight);
                  }
               }
            }


            // Systematic on T
            if(PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, treeS,  NULL, -1,   0, 0, TRescale)){
               double Mass     = GetMass(track->p(),dedxMObj.dEdx());
               double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta()*TRescale);
               double MassComb = Mass;if(tof)MassComb=GetMassFromBeta(track->p(), (GetIBeta(dedxMObj.dEdx()) + ((1/tof->inverseBeta())*TRescale ))*0.5 ) ;

               for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                  if(PassSelection(hscp,  dedxSObj, dedxMObj, tof, treeS, CutIndex, NULL, -1,   0, 0, TRescale)){
                     HSCPTk_SystT[CutIndex] = true;
                     if(Mass>MaxMass_SystT[CutIndex]) MaxMass_SystT[CutIndex]=Mass;
                     SignPlots[4*s               ].Mass_SystT->Fill(CutIndex, Mass,Event_Weight);
                     SignPlots[4*s+NChargedHSCP+1].Mass_SystT->Fill(CutIndex, Mass,Event_Weight);
                     if(tof){
                        SignPlots[4*s               ].MassTOF_SystT ->Fill(CutIndex, MassTOF , Event_Weight);
                        SignPlots[4*s+NChargedHSCP+1].MassTOF_SystT ->Fill(CutIndex, MassTOF , Event_Weight);
                     }
                     SignPlots[4*s               ].MassComb_SystT->Fill(CutIndex, MassComb, Event_Weight);
                     SignPlots[4*s+NChargedHSCP+1].MassComb_SystT->Fill(CutIndex, MassComb, Event_Weight);
                  }
               }
            }

            ///////////// END   COMPUTATION OF THE SYSTEMATIC //////////



               PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, treeS,           &SignPlots[4*s+NChargedHSCP+1], genColl[ClosestGen].p()/genColl[ClosestGen].energy());
            if(!PassPreselection(hscp,  dedxSObj, dedxMObj, tof, dttof, csctof, treeS,           &SignPlots[4*s               ], genColl[ClosestGen].p()/genColl[ClosestGen].energy()))continue;         

            double Mass     = GetMass(track->p(),dedxMObj.dEdx());
            double MassTOF  = -1; if(tof)MassTOF = GetTOFMass(track->p(),tof->inverseBeta());
            double MassComb = Mass;if(tof)MassComb=GetMassFromBeta(track->p(), (GetIBeta(dedxMObj.dEdx()) + (1/tof->inverseBeta()))*0.5 ) ;


            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                   PassSelection   (hscp,  dedxSObj, dedxMObj, tof, treeS, CutIndex, &SignPlots[4*s+NChargedHSCP+1], genColl[ClosestGen].p()/genColl[ClosestGen].energy());
               if(!PassSelection   (hscp,  dedxSObj, dedxMObj, tof, treeS, CutIndex, &SignPlots[4*s               ], genColl[ClosestGen].p()/genColl[ClosestGen].energy()))continue;    

               HSCPTk[CutIndex] = true;
	       if(Mass>MaxMass[CutIndex]) MaxMass[CutIndex]=Mass;

               SignPlots[4*s               ].Mass->Fill(CutIndex, Mass,Event_Weight);
               SignPlots[4*s+NChargedHSCP+1].Mass->Fill(CutIndex, Mass,Event_Weight);
               if(tof){
                  SignPlots[4*s               ].MassTOF ->Fill(CutIndex, MassTOF , Event_Weight);
                  SignPlots[4*s+NChargedHSCP+1].MassTOF ->Fill(CutIndex, MassTOF , Event_Weight);
               }
               SignPlots[4*s               ].MassComb->Fill(CutIndex, MassComb, Event_Weight);
               SignPlots[4*s+NChargedHSCP+1].MassComb->Fill(CutIndex, MassComb, Event_Weight);
            } //end of Cut loop
            if(track->pt()>35 && Mass>35)stPlots_FillTree(SignPlots[4*s               ] , treeS.eventAuxiliary().run(),treeS.eventAuxiliary().event(), c, track->pt(), dedxSObj.dEdx(), tof ? tof->inverseBeta() : -1, Mass);
         } // end of Track Loop 
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
	   if(HSCPTk[CutIndex]){
	     SignPlots[4*s               ].HSCPE             ->Fill(CutIndex,Event_Weight);
	     SignPlots[4*s+NChargedHSCP+1].HSCPE             ->Fill(CutIndex,Event_Weight);
             SignPlots[4*s               ].MaxEventMass      ->Fill(CutIndex,MaxMass[CutIndex],Event_Weight);
             SignPlots[4*s+NChargedHSCP+1].MaxEventMass      ->Fill(CutIndex,MaxMass[CutIndex],Event_Weight); } }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
           if(HSCPTk_SystP[CutIndex]){
             SignPlots[4*s               ].HSCPE_SystP       ->Fill(CutIndex,Event_Weight);
             SignPlots[4*s+NChargedHSCP+1].HSCPE_SystP       ->Fill(CutIndex,Event_Weight);
             SignPlots[4*s               ].MaxEventMass_SystP->Fill(CutIndex,MaxMass_SystP[CutIndex],Event_Weight);
             SignPlots[4*s+NChargedHSCP+1].MaxEventMass_SystP->Fill(CutIndex,MaxMass_SystP[CutIndex],Event_Weight);  } }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
           if(HSCPTk_SystI[CutIndex]){
             SignPlots[4*s               ].HSCPE_SystI       ->Fill(CutIndex,Event_Weight);
             SignPlots[4*s+NChargedHSCP+1].HSCPE_SystI       ->Fill(CutIndex,Event_Weight);
             SignPlots[4*s               ].MaxEventMass_SystI->Fill(CutIndex,MaxMass_SystI[CutIndex],Event_Weight);
             SignPlots[4*s+NChargedHSCP+1].MaxEventMass_SystI->Fill(CutIndex,MaxMass_SystI[CutIndex],Event_Weight); } }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
           if(HSCPTk_SystM[CutIndex]){
             SignPlots[4*s               ].HSCPE_SystM       ->Fill(CutIndex,Event_Weight);
             SignPlots[4*s+NChargedHSCP+1].HSCPE_SystM       ->Fill(CutIndex,Event_Weight);
             SignPlots[4*s               ].MaxEventMass_SystM->Fill(CutIndex,MaxMass_SystM[CutIndex],Event_Weight);
             SignPlots[4*s+NChargedHSCP+1].MaxEventMass_SystM->Fill(CutIndex,MaxMass_SystM[CutIndex],Event_Weight); } }
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
           if(HSCPTk_SystT[CutIndex]){
             SignPlots[4*s               ].HSCPE_SystT       ->Fill(CutIndex,Event_Weight);
             SignPlots[4*s+NChargedHSCP+1].HSCPE_SystT       ->Fill(CutIndex,Event_Weight);
             SignPlots[4*s               ].MaxEventMass_SystT->Fill(CutIndex,MaxMass_SystT[CutIndex],Event_Weight);
             SignPlots[4*s+NChargedHSCP+1].MaxEventMass_SystT->Fill(CutIndex,MaxMass_SystT[CutIndex],Event_Weight); } }

      }// end of Event Loop
      }
      printf("\n");
      delete [] HSCPTk;
      delete [] HSCPTk_SystP;
      delete [] HSCPTk_SystI;
      delete [] HSCPTk_SystT;
      delete [] HSCPTk_SystM;
      delete [] MaxMass;
      delete [] MaxMass_SystP;
      delete [] MaxMass_SystI;
      delete [] MaxMass_SystT;
      delete [] MaxMass_SystM;


      stPlots_Clear(SignPlots[4*s+0], true);
      stPlots_Clear(SignPlots[4*s+1], true);
      stPlots_Clear(SignPlots[4*s+2], true);
      stPlots_Clear(SignPlots[4*s+3], true);
   }// end of signal Type loop
}

TH1D* GetPDF(TH1D* pdf){
   char NewName[2048];
   sprintf(NewName,"%s_PDF", pdf->GetName());

   TH1D* PDF = new TH1D(NewName,NewName,pdf->GetNbinsX(),pdf->GetXaxis()->GetXmin(),pdf->GetXaxis()->GetXmax());
   for(int i=0;i<=pdf->GetNbinsX();i++){
      if(i==0){
         PDF->SetBinContent(i, pdf->GetBinContent(i) );
      }else{
         PDF->SetBinContent(i, pdf->GetBinContent(i)+PDF->GetBinContent(i-1) );
      }
   }
   PDF->Scale(1.0/PDF->GetBinContent(PDF->GetNbinsX()));
   return PDF;
}

double GetRandValue(TH1D* PDF){
   int randNumber = rand();
   double uniform = randNumber / (double)RAND_MAX;
   for(int i=1;i<=PDF->GetNbinsX();i++){
      if(PDF->GetBinContent(i)>uniform){
         return PDF->GetXaxis()->GetBinUpEdge(i);
//         return PDF->GetXaxis()->GetBinUpEdge(i-1)+(rand()/(double)RAND_MAX)*PDF->GetXaxis()->GetBinWidth(i-1);
      }
   }
   return PDF->GetXaxis()->GetBinLowEdge(PDF->GetNbinsX());
}

void Analysis_Step4(char* SavePath)
{
   if(! (DataFileName.size() || MCsample.size()))return; 
   printf("Step4: Doing final computations\n");

   //////////////////////////////////////////////////      MAKING THE PREDICTION
   for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){

      const double& A=H_A->GetBinContent(CutIndex+1);
      const double& B=H_B->GetBinContent(CutIndex+1);
      const double& C=H_C->GetBinContent(CutIndex+1);
      const double& D=H_D->GetBinContent(CutIndex+1);
      const double& E=H_E->GetBinContent(CutIndex+1);
      const double& F=H_F->GetBinContent(CutIndex+1);
      const double& G=H_G->GetBinContent(CutIndex+1);
      const double& H=H_H->GetBinContent(CutIndex+1);
      double P=0;
      double Perr=0;

      printf("%4i --> Pt>%7.2f  I>%6.2f  TOF>%+5.2f --> A=%6.2E B=%6.E C=%6.2E D=%6.2E E=%6.2E F=%6.2E G=%6.2E H=%6.2E\n",CutIndex,CutPt[CutIndex], CutI[CutIndex], CutTOF[CutIndex],A, B, C, D, E, F, G, H );

      if(E>0){
         P    = (A*F*G)/(E*E);
         Perr = sqrt( ((pow(F*G,2)* A + pow(A*G,2)*F + pow(A*F,2)*G)/pow(E,4)) + (pow((2*A*F*G)/pow(E,3),2)*E));
      }else if(A>0){
         P    = ((C*B)/A);
         Perr = sqrt( (pow(B/A,2)*C) + (pow(C/A,2)*B) + (pow((B*(C)/(A*A)),2)*A) );
      }

      H_P->SetBinContent(CutIndex+1,P);
      H_P->SetBinError  (CutIndex+1,Perr);
      if(P==0 || isnan(P))continue; //Skip this CutIndex --> No Prediction possible

      printf("%4i --> Pt>%7.2f  I>%6.2f  TOF>%+5.2f --> D=%6.2E vs Pred = %6.2E +- %6.2E (%6.2E%%)\n", CutIndex,CutPt[CutIndex], CutI[CutIndex], CutTOF[CutIndex],D, P,  Perr, 100.0*Perr/P );

      TH1D* Pred_EtaB_Proj = Pred_EtaB->ProjectionY("ProjEtaB",CutIndex+1,CutIndex+1);  // Pred_EtaB_Proj->Scale(1.0/Pred_EtaB_Proj->Integral());
      TH1D* Pred_EtaS_Proj = Pred_EtaS->ProjectionY("ProjEtaS",CutIndex+1,CutIndex+1); //  Pred_EtaS_Proj->Scale(1.0/Pred_EtaS_Proj->Integral());
      TH1D* Pred_EtaS2_Proj = Pred_EtaS2->ProjectionY("ProjEtaS2",CutIndex+1,CutIndex+1);//   Pred_EtaS2_Proj->Scale(1.0/Pred_EtaS2_Proj->Integral());
      TH1D* Pred_EtaB_Proj_PE  = (TH1D*)Pred_EtaB_Proj->Clone("Pred_EtaB_Proj_PE");   Pred_EtaB_Proj_PE->Reset();
      TH1D* Pred_EtaS_Proj_PE  = (TH1D*)Pred_EtaS_Proj->Clone("Pred_EtaS_Proj_PE");   Pred_EtaS_Proj_PE->Reset();
      TH1D* Pred_EtaS2_Proj_PE = (TH1D*)Pred_EtaS2_Proj->Clone("Pred_EtaS2_Proj_PE"); Pred_EtaS2_Proj_PE->Reset();

      Pred_EtaP->GetXaxis()->SetRange(CutIndex+1,CutIndex+1);
      TH2D* Pred_EtaPWeighted = (TH2D*)Pred_EtaP->Project3D("zy");
      TH2D* Pred_EtaPWeighted_PE = (TH2D*)Pred_EtaPWeighted->Clone("Pred_EtaPWeightedPE");   Pred_EtaPWeighted_PE->Reset();

/*
      for(int x=0;x<=Pred_EtaPWeighted->GetXaxis()->GetNbins();x++){
         double WeightP = 0.0;
         if(Pred_EtaB_Proj->GetBinContent(x)>0){
            WeightP = Pred_EtaS_Proj->GetBinContent(x)/Pred_EtaB_Proj->GetBinContent(x);
            if(TypeMode==2)WeightP*= Pred_EtaS2_Proj->GetBinContent(x)/Pred_EtaB_Proj->GetBinContent(x);
         }

         for(int y=0;y<=Pred_EtaPWeighted->GetYaxis()->GetNbins();y++){
            Pred_EtaPWeighted->SetBinContent(x,y,Pred_EtaPWeighted->GetBinContent(x,y)*WeightP);
         }
      }
*/
//      TH1D* Pred_P_Proj = Pred_EtaPWeighted->ProjectionY("ProjP");
      TH1D* Pred_I_Proj = Pred_I->ProjectionY("ProjI",CutIndex+1,CutIndex+1);
      TH1D* Pred_T_Proj = Pred_TOF->ProjectionY("ProjT",CutIndex+1,CutIndex+1);
      TH1D* Pred_I_ProjPE = (TH1D*) Pred_I_Proj->Clone("Pred_I_ProjPE"); Pred_I_ProjPE->Reset();
      TH1D* Pred_T_ProjPE = (TH1D*) Pred_T_Proj->Clone("Pred_T_ProjPE"); Pred_T_ProjPE->Reset();


//      TH1D* Pred_P_PDF = GetPDF(Pred_P_Proj);
//      TH1D* Pred_I_PDF = GetPDF(Pred_I_Proj);
//      TH1D* Pred_T_PDF = GetPDF(Pred_T_Proj);

      TH2D* Pred_Prof_Mass     =  new TH2D("Pred_Prof_Mass"    ,"Pred_Prof_Mass"    ,MassNBins,0,MassHistoUpperBound, 100, 0, 100); 
      TH2D* Pred_Prof_MassTOF  =  new TH2D("Pred_Prof_MassTOF" ,"Pred_Prof_MassTOF" ,MassNBins,0,MassHistoUpperBound, 100, 0, 100);  
      TH2D* Pred_Prof_MassComb =  new TH2D("Pred_Prof_MassComb","Pred_Prof_MassComb",MassNBins,0,MassHistoUpperBound, 100, 0, 100);


    for(int x=0;x<Pred_Mass->GetNbinsY()+1;x++){
       for(unsigned int pe=0;pe<100;pe++){
          Pred_Prof_Mass    ->SetBinContent(x, pe, 0);
          Pred_Prof_MassTOF ->SetBinContent(x, pe, 0);
          Pred_Prof_MassComb->SetBinContent(x, pe, 0);
       }
    }



      TRandom3* RNG = new TRandom3();
      printf("Predicting (%4i / %4i)     :",CutIndex+1,(int)CutPt.size());
      int TreeStep = 100/50;if(TreeStep==0)TreeStep=1;
      for(unsigned int pe=0;pe<100;pe++){    
      if(pe%TreeStep==0){printf(".");fflush(stdout);}

      TH1D* tmpH_Mass     =  new TH1D("tmpH_Mass"    ,"tmpH_Mass"    ,MassNBins,0,MassHistoUpperBound);
      TH1D* tmpH_MassTOF  =  new TH1D("tmpH_MassTOF" ,"tmpH_MassTOF" ,MassNBins,0,MassHistoUpperBound);
      TH1D* tmpH_MassComb =  new TH1D("tmpH_MassComb","tmpH_MassComb",MassNBins,0,MassHistoUpperBound);


      double PE_A=RNG->Poisson(A);
      double PE_B=RNG->Poisson(B);
      double PE_C=RNG->Poisson(C);
      //double PE_D=RNG->Poisson(D);
      double PE_E=RNG->Poisson(E);
      double PE_F=RNG->Poisson(F);
      double PE_G=RNG->Poisson(G);
      //double PE_H=RNG->Poisson(H);
      double PE_P = 0;

      if(E>0){
         PE_P    = (PE_E>0 ? (PE_A*PE_F*PE_G)/(PE_E*PE_E) : 0);
      }else if(A>0){
         PE_P    = (PE_A>0 ? ((PE_C*PE_B)/PE_A) : 0);
      }

      for(int i=0;i<Pred_EtaB_Proj_PE->GetNbinsX()+1;i++){Pred_EtaB_Proj_PE->SetBinContent(i,RNG->Poisson(Pred_EtaB_Proj->GetBinContent(i)) );}    Pred_EtaB_Proj_PE->Scale(1.0/Pred_EtaB_Proj_PE->Integral());
      for(int i=0;i<Pred_EtaS_Proj_PE->GetNbinsX()+1;i++){Pred_EtaS_Proj_PE->SetBinContent(i,RNG->Poisson(Pred_EtaS_Proj->GetBinContent(i)) );}    Pred_EtaS_Proj_PE->Scale(1.0/Pred_EtaS_Proj_PE->Integral());
      for(int i=0;i<Pred_EtaS2_Proj_PE->GetNbinsX()+1;i++){Pred_EtaS2_Proj_PE->SetBinContent(i,RNG->Poisson(Pred_EtaS2_Proj->GetBinContent(i)) );} Pred_EtaS2_Proj_PE->Scale(1.0/Pred_EtaS2_Proj_PE->Integral());


      for(int i=0;i<Pred_EtaPWeighted_PE->GetNbinsX()+1;i++){
      for(int j=0;j<Pred_EtaPWeighted_PE->GetNbinsY()+1;j++){
         Pred_EtaPWeighted_PE->SetBinContent(i,j,RNG->Poisson(Pred_EtaPWeighted->GetBinContent(i,j)));
      }}

      double WeightP = 0.0;
      for(int x=0;x<=Pred_EtaPWeighted_PE->GetXaxis()->GetNbins();x++){
         WeightP = 0.0;
         if(Pred_EtaB_Proj_PE->GetBinContent(x)>0){
                           WeightP = Pred_EtaS_Proj_PE ->GetBinContent(x)/Pred_EtaB_Proj_PE->GetBinContent(x);
            if(TypeMode==2)WeightP*= Pred_EtaS2_Proj_PE->GetBinContent(x)/Pred_EtaB_Proj_PE->GetBinContent(x);
         }

         for(int y=0;y<=Pred_EtaPWeighted_PE->GetYaxis()->GetNbins();y++){
            Pred_EtaPWeighted_PE->SetBinContent(x,y,Pred_EtaPWeighted_PE->GetBinContent(x,y)*WeightP);
         }
      }
      TH1D* Pred_P_ProjPE = Pred_EtaPWeighted_PE->ProjectionY("Pred_P_ProjPE");                                                        Pred_P_ProjPE->Scale(1.0/Pred_P_ProjPE->Integral());
      for(int i=0;i<Pred_I_ProjPE->GetNbinsX()+1;i++){Pred_I_ProjPE->SetBinContent(i,RNG->Poisson(Pred_I_Proj->GetBinContent(i)) );}   Pred_I_ProjPE->Scale(1.0/Pred_I_ProjPE->Integral());
      for(int i=0;i<Pred_T_ProjPE->GetNbinsX()+1;i++){Pred_T_ProjPE->SetBinContent(i,RNG->Poisson(Pred_T_Proj->GetBinContent(i)) );}   Pred_T_ProjPE->Scale(1.0/Pred_T_ProjPE->Integral());

      double Proba, MI, MComb, MT=0, ProbaT=0;
      for(int x=0;x<Pred_P_ProjPE->GetNbinsX()+1;x++){    if(Pred_P_ProjPE->GetBinContent(x)<=0.0){continue;}  const double& p = Pred_P_ProjPE->GetBinCenter(x);
      for(int y=0;y<Pred_I_ProjPE->GetNbinsX()+1;y++){    if(Pred_I_ProjPE->GetBinContent(y)<=0.0){continue;}  const double& i = Pred_I_ProjPE->GetBinCenter(y);
         Proba = Pred_P_ProjPE->GetBinContent(x) * Pred_I_ProjPE->GetBinContent(y);  if(Proba<=0 || isnan(Proba))continue;
         MI = GetMass(p,i);
         MComb = MI;
         tmpH_Mass->Fill(MI,Proba);

//         if(TypeMode==2){
//         for(int z=0;z<Pred_T_ProjPE->GetNbinsX()+1;z++){   if(Pred_T_ProjPE->GetBinContent(z)<=0.0){continue;}   const double& t = Pred_T_ProjPE->GetBinCenter(z);
//            ProbaT = Proba * Pred_T_ProjPE->GetBinContent(z);  if(ProbaT<=0 || isnan(ProbaT))continue;
//            MT = GetTOFMass(p,t);
//            tmpH_MassTOF->Fill(MT,ProbaT);
//            MComb = GetMassFromBeta(p, (GetIBeta(i) + (1/t))*0.5 );        
//            tmpH_MassComb->Fill(MComb,ProbaT);
//         }}else{
            tmpH_MassComb->Fill(MComb,Proba);
//         }
      }}

//      printf("PE_P = %f\n",PE_P);

      for(int x=0;x<tmpH_Mass->GetNbinsX()+1;x++){
         //const double& M = tmpH_Mass->GetXaxis()->GetBinCenter(x);
         Pred_Prof_Mass    ->SetBinContent(x, pe, tmpH_Mass    ->GetBinContent(x) * PE_P);
         Pred_Prof_MassTOF ->SetBinContent(x, pe, tmpH_MassTOF ->GetBinContent(x) * PE_P);
         Pred_Prof_MassComb->SetBinContent(x, pe, tmpH_MassComb->GetBinContent(x) * PE_P);
         if(isnan(tmpH_Mass    ->GetBinContent(x) * PE_P)){printf("%f x %f\n",tmpH_Mass    ->GetBinContent(x),PE_P); fflush(stdout);exit(0);}
      }
     
      delete Pred_P_ProjPE;
      delete tmpH_Mass;
      delete tmpH_MassTOF;
      delete tmpH_MassComb;
     }printf("\n");

    for(int x=0;x<Pred_Mass->GetNbinsY()+1;x++){
//       Pred_Mass    ->SetBinContent(CutIndex+1,x,Pred_Prof_Mass    ->GetBinContent(x)); Pred_Mass      ->SetBinError(CutIndex+1,x,sqrt(pow(Pred_Prof_Mass    ->GetBinError(x),2) + Pred_Prof_Mass    ->GetBinContent(x)*(Perr/P)));
//       Pred_MassTOF ->SetBinContent(CutIndex+1,x,Pred_Prof_MassTOF ->GetBinContent(x)); Pred_MassTOF   ->SetBinError(CutIndex+1,x,sqrt(pow(Pred_Prof_MassTOF ->GetBinError(x),2) + Pred_Prof_MassTOF ->GetBinContent(x)*(Perr/P)));
//       Pred_MassComb->SetBinContent(CutIndex+1,x,Pred_Prof_MassComb->GetBinContent(x)); Pred_MassComb  ->SetBinError(CutIndex+1,x,sqrt(pow(Pred_Prof_MassComb->GetBinError(x),2) + Pred_Prof_MassComb->GetBinContent(x)*(Perr/P)));

       double Mean=0, MeanTOF=0, MeanComb=0;
       for(unsigned int pe=0;pe<100;pe++){
	 //if(CutIndex==4){printf("Bin=%4i pe=%3i --> BinCOntent=%f\n",x,pe,Pred_Prof_Mass    ->GetBinContent(x, pe));}
          Mean     += Pred_Prof_Mass    ->GetBinContent(x, pe);
          MeanTOF  += Pred_Prof_MassTOF ->GetBinContent(x, pe);
          MeanComb += Pred_Prof_MassComb->GetBinContent(x, pe);
       }Mean/=100.0; MeanTOF/=100.0;  MeanComb/=100.0;

       //if(CutIndex==4){printf("MEAN = %f\n",Mean);}


       double Err=0, ErrTOF=0, ErrComb=0;
       for(unsigned int pe=0;pe<100;pe++){
	  //if(CutIndex==4){printf("Bin=%4i pe=%3i --> DeltaM=%f\n",x,pe,sqrt(pow(Mean     - Pred_Prof_Mass    ->GetBinContent(x, pe),2)));}
          Err     += pow(Mean     - Pred_Prof_Mass    ->GetBinContent(x, pe),2);
          ErrTOF  += pow(MeanTOF  - Pred_Prof_MassTOF ->GetBinContent(x, pe),2);
          ErrComb += pow(MeanComb - Pred_Prof_MassComb->GetBinContent(x, pe),2);
       }Err=sqrt(Err/99.0); ErrTOF=sqrt(ErrTOF/99.0);  ErrComb=sqrt(ErrComb/99.0);
       //if(CutIndex==4){printf("ERROR = %f\n",Err);}


       Pred_Mass    ->SetBinContent(CutIndex+1,x,Mean    ); Pred_Mass      ->SetBinError(CutIndex+1,x,Err    );
       Pred_MassTOF ->SetBinContent(CutIndex+1,x,MeanTOF ); Pred_MassTOF   ->SetBinError(CutIndex+1,x,ErrTOF );
       Pred_MassComb->SetBinContent(CutIndex+1,x,MeanComb); Pred_MassComb  ->SetBinError(CutIndex+1,x,ErrComb);
    }
//    printf("MassInt %f\n",Pred_Prof_Mass->Integral());


    delete Pred_EtaB_Proj_PE;
    delete Pred_EtaS_Proj_PE;
    delete Pred_EtaS2_Proj_PE;

    delete Pred_Prof_Mass;
    delete Pred_Prof_MassTOF;
    delete Pred_Prof_MassComb;
    delete Pred_EtaPWeighted_PE;
    delete Pred_I_ProjPE;
    delete Pred_T_ProjPE;

//    delete Pred_P_PDF;
//    delete Pred_I_PDF;
//    delete Pred_T_PDF;
//    delete Pred_P_Proj;
    delete Pred_I_Proj;
    delete Pred_T_Proj;
    delete Pred_EtaB_Proj;
    delete Pred_EtaS_Proj;
    delete Pred_EtaS2_Proj;
    delete Pred_EtaPWeighted;
   }


   //////////////////////////////////////////////////     DUMP USEFUL INFORMATION
   if(DataFileName.size()>0 || MCsample.size()){  //Dump info only if we are looking at some datasamples.
   char Buffer[2048];
   if(DataFileName.size()>0){sprintf(Buffer,"%s/Info.txt",SavePath);
   }else{                    sprintf(Buffer,"%s/Info_MC.txt",SavePath);}
   FILE* pFile = fopen(Buffer,"w");
   fprintf(pFile,"Selection      = %s\n",dEdxS_Label.c_str());
   fprintf(pFile,"Mass           = %s\n",dEdxM_Label.c_str());
   fprintf(pFile,"TOF            = %s\n",TOF_Label.c_str());
   fprintf(pFile,"|eta|          < %f\n",GlobalMaxEta);
   fprintf(pFile,"pT_err/pT      < %f\n",GlobalMaxPterr);
   fprintf(pFile,"#Hit           > %02i\n",GlobalMinNOH);
   fprintf(pFile,"#dEdx Hit      > %02i\n",GlobalMinNOM);
   fprintf(pFile,"nDoF           > %02i\n",GlobalMinNOH);
   fprintf(pFile,"Chi2/ndf       < %6.2f\n",GlobalMaxChi2);
   fprintf(pFile,"SumPt          < %6.2f\n",GlobalMaxTIsol);
   fprintf(pFile,"E/p            < %6.2f\n",GlobalMaxEIsol);

   for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
      const double& A=H_A->GetBinContent(CutIndex+1);
      const double& B=H_B->GetBinContent(CutIndex+1);
      const double& C=H_C->GetBinContent(CutIndex+1);
      const double& D=H_D->GetBinContent(CutIndex+1);
      const double& E=H_E->GetBinContent(CutIndex+1);
      const double& F=H_F->GetBinContent(CutIndex+1);
      const double& G=H_G->GetBinContent(CutIndex+1);
      const double& H=H_H->GetBinContent(CutIndex+1);

      fprintf(pFile  ,"CutIndex=%4i --> (Pt>%6.2f I>%6.3f TOF>%6.3f) Ndata=%+6.2E  NPred=%6.3E+-%6.3E <--> A=%6.2E B=%6.E C=%6.2E D=%6.2E E=%6.2E F=%6.2E G=%6.2E H=%6.2E\n",CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), D,H_P->GetBinContent(CutIndex+1),H_P->GetBinError(CutIndex+1) ,A, B, C, D, E, F, G, H);
   }
   fprintf(pFile,"--------------------\n");
   fclose(pFile);
   } 
   //////////////////////////////////////////////////     CREATE EFFICIENCY FILE

   fflush(stdout);
}


void InitHistos(){
   for(unsigned int m=0;m<MCsample.size();m++){
      stPlots tmp;
      MCPlots.push_back(tmp);
   }

   for(unsigned int s=0;s<signals.size();s++){
   for(int NC=0;NC<4;NC++){
      stPlots tmp;
      if(NC==0){
      }else{
         char buffer[256];sprintf(buffer,"_NC%i",NC-1);
      }
      SignPlots.push_back(tmp);
   }}
   HistoFile->cd();

   HCuts_Pt  = new TH1D("HCuts_Pt" ,"HCuts_Pt" ,CutPt.size(),0,CutPt.size());
   HCuts_I   = new TH1D("HCuts_I"  ,"HCuts_I"  ,CutPt.size(),0,CutPt.size());
   HCuts_TOF = new TH1D("HCuts_TOF","HCuts_TOF",CutPt.size(),0,CutPt.size());
   for(unsigned int i=0;i<CutPt.size();i++){  HCuts_Pt->Fill(i,CutPt[i]);     HCuts_I->Fill(i,CutI[i]);    HCuts_TOF->Fill(i,CutTOF[i]);   }

   if(DataFileName.size() || MCsample.size()){
      H_A = new TH1D("H_A" ,"H_A" ,CutPt.size(),0,CutPt.size());
      H_B = new TH1D("H_B" ,"H_B" ,CutPt.size(),0,CutPt.size());
      H_C = new TH1D("H_C" ,"H_C" ,CutPt.size(),0,CutPt.size());
      H_D = new TH1D("H_D" ,"H_D" ,CutPt.size(),0,CutPt.size());
      H_E = new TH1D("H_E" ,"H_E" ,CutPt.size(),0,CutPt.size());
      H_F = new TH1D("H_F" ,"H_F" ,CutPt.size(),0,CutPt.size());
      H_G = new TH1D("H_G" ,"H_G" ,CutPt.size(),0,CutPt.size());
      H_H = new TH1D("H_H" ,"H_H" ,CutPt.size(),0,CutPt.size());
      H_P = new TH1D("H_P" ,"H_P" ,CutPt.size(),0,CutPt.size());

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

      char Name   [1024];
      sprintf(Name,"Is");
      Hist_Is         = new TH1D(Name,Name, 200,0,dEdxS_UpLim);
      Hist_Is->Sumw2(); 

      sprintf(Name,"Pt");
      Hist_Pt       = new TH1D(Name,Name,200,0,PtHistoUpperBound);
      Hist_Pt->Sumw2();

      sprintf(Name,"TOF");
      Hist_TOF       = new TH1D(Name,Name,200,-10,20);
      Hist_TOF->Sumw2();

      sprintf(Name,"Pred_Mass");
      Pred_Mass = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),MassNBins,0,MassHistoUpperBound);
      Pred_Mass->Sumw2();

      sprintf(Name,"Pred_MassTOF");
      Pred_MassTOF = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(), MassNBins,0,MassHistoUpperBound);
      Pred_MassTOF->Sumw2();

      sprintf(Name,"Pred_MassComb");
      Pred_MassComb = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),MassNBins,0,MassHistoUpperBound);
      Pred_MassComb->Sumw2();

      sprintf(Name,"Pred_I");
      Pred_I  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   200,GlobalMinIm,dEdxM_UpLim);
      Pred_I->Sumw2();

      sprintf(Name,"Pred_EtaB");
      Pred_EtaB  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   50,-3,3);
      Pred_EtaB->Sumw2();

      sprintf(Name,"Pred_EtaS");
      Pred_EtaS  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   50,-3,3);
      Pred_EtaS->Sumw2();

      sprintf(Name,"Pred_EtaS2");
      Pred_EtaS2  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   50,-3,3);
      Pred_EtaS2->Sumw2();


      sprintf(Name,"Pred_EtaP");
      Pred_EtaP  = new TH3D(Name,Name,CutPt.size(),0,CutPt.size(),   50, -3, 3, 200,GlobalMinPt,PtHistoUpperBound);
      Pred_EtaP->Sumw2();

      sprintf(Name,"Pred_TOF");
      Pred_TOF  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   200,GlobalMinTOF,5);
      Pred_TOF->Sumw2();


      sprintf(Name,"RegionD_I");
      RegionD_I  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   200,GlobalMinIm,dEdxM_UpLim);
      RegionD_I->Sumw2();

      sprintf(Name,"RegionD_P");
      RegionD_P  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   200,GlobalMinPt,PtHistoUpperBound);
      RegionD_P->Sumw2();

      sprintf(Name,"RegionD_TOF");
      RegionD_TOF  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),   200,GlobalMinTOF,5);
      RegionD_TOF->Sumw2();
   } 
}


double DistToHSCP (const susybsm::HSCParticle& hscp, const std::vector<reco::GenParticle>& genColl, int& IndexOfClosest){
   reco::TrackRef   track = hscp.trackRef(); if(track.isNull())return false;

   double RMin = 9999; IndexOfClosest=-1;
   for(unsigned int g=0;g<genColl.size();g++){
      if(genColl[g].pt()<5)continue;
      if(genColl[g].status()!=1)continue;
      int AbsPdg=abs(genColl[g].pdgId());
      if(AbsPdg<1000000)continue;    

      double dR = deltaR(track->eta(), track->phi(), genColl[g].eta(), genColl[g].phi());
      if(dR<RMin){RMin=dR;IndexOfClosest=g;}
   }
   return RMin;
}

double GetSampleWeight(const double& IntegratedLuminosityInPb, const double& IntegratedLuminosityInPbBeforeTriggerChange, const double& CrossSection, const double& MCEvents, int period){
  double Weight = 1.0;
  if(IntegratedLuminosityInPb>=IntegratedLuminosityInPbBeforeTriggerChange && IntegratedLuminosityInPb>0){
    double NMCEvents = MCEvents;
    if(MaxEntry>0)NMCEvents=std::min(MCEvents,(double)MaxEntry);
    if      (period==0)Weight = (CrossSection * IntegratedLuminosityInPbBeforeTriggerChange) / NMCEvents;
    else if (period==1)Weight = (CrossSection * (IntegratedLuminosityInPb-IntegratedLuminosityInPbBeforeTriggerChange)) / NMCEvents;
  }
  return Weight;
}


double GetSampleWeightMC(const double& IntegratedLuminosityInPb, const std::vector<string> fileNames, const double& XSection, const double& SampleSize, double MaxEvent){
  double Weight = 1.0;
   unsigned long InitNumberOfEvents = GetInitialNumberOfMCEvent(fileNames); 
   double SampleEquivalentLumi = InitNumberOfEvents / XSection;
   if(MaxEvent<0)MaxEvent=SampleSize;
   printf("GetSampleWeight MC: IntLumi = %6.2E  SampleLumi = %6.2E --> EventWeight = %6.2E --> ",IntegratedLuminosityInPb,SampleEquivalentLumi, IntegratedLuminosityInPb/SampleEquivalentLumi);
//   printf("Sample NEvent = %6.2E   SampleEventUsed = %6.2E --> Weight Rescale = %6.2E\n",SampleSize, MaxEvent, SampleSize/MaxEvent);
   Weight = (IntegratedLuminosityInPb/SampleEquivalentLumi) * (SampleSize/MaxEvent);
   printf("FinalWeight = %6.2f\n",Weight);
   return Weight;
}

double GetPUWeight(const fwlite::ChainEvent& ev, const bool& Iss4pileup){
   //get pile up weight for this event
   fwlite::Handle<std::vector<PileupSummaryInfo> > PupInfo;
   PupInfo.getByLabel(ev, "addPileupInfo");
   if(!PupInfo.isValid()){printf("PileupSummaryInfo Collection NotFound\n");return 1.0;}
   double PUWeight_thisevent=1;
   std::vector<PileupSummaryInfo>::const_iterator PVI;
   int npv = -1;
   if(Iss4pileup){
      float sum_nvtx = 0;
      for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
         npv = PVI->getPU_NumInteractions();
         sum_nvtx += float(npv);
      }
      float ave_nvtx = sum_nvtx/3.;
      PUWeight_thisevent = LumiWeightsMC_.weight3BX( ave_nvtx );
   }else{
      for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
         int BX = PVI->getBunchCrossing();
         if(BX == 0) {
            npv = PVI->getPU_NumInteractions();
            continue;
         }
      }
      PUWeight_thisevent = LumiWeightsMC_.weight( npv );
   }
   return PUWeight_thisevent;
}


int HowManyChargedHSCP (const std::vector<reco::GenParticle>& genColl){
   int toReturn = 0;
   for(unsigned int g=0;g<genColl.size();g++){
      if(genColl[g].pt()<5)continue;
      if(genColl[g].status()!=1)continue;
      int AbsPdg=abs(genColl[g].pdgId());
      if(AbsPdg<1000000)continue;
      if(AbsPdg==1000993 || AbsPdg==1009313 || AbsPdg==1009113 || AbsPdg==1009223 || AbsPdg==1009333 || AbsPdg==1092114 || AbsPdg==1093214 || AbsPdg==1093324)continue; //Skip neutral gluino RHadrons
      if(AbsPdg==1000622 || AbsPdg==1000642 || AbsPdg==1006113 || AbsPdg==1006311 || AbsPdg==1006313 || AbsPdg==1006333)continue;  //skip neutral stop RHadrons
      toReturn++;
   }
   return toReturn;
}


void  GetGenHSCPBeta (const std::vector<reco::GenParticle>& genColl, double& beta1, double& beta2, bool onlyCharged){
   beta1=-1; beta2=-1;
   for(unsigned int g=0;g<genColl.size();g++){
      if(genColl[g].pt()<5)continue;
      if(genColl[g].status()!=1)continue;
      int AbsPdg=abs(genColl[g].pdgId());
      if(AbsPdg<1000000)continue;
      if(onlyCharged && (AbsPdg==1000993 || AbsPdg==1009313 || AbsPdg==1009113 || AbsPdg==1009223 || AbsPdg==1009333 || AbsPdg==1092114 || AbsPdg==1093214 || AbsPdg==1093324))continue; //Skip neutral gluino RHadrons
      if(onlyCharged && (AbsPdg==1000622 || AbsPdg==1000642 || AbsPdg==1006113 || AbsPdg==1006311 || AbsPdg==1006313 || AbsPdg==1006333))continue;  //skip neutral stop RHadrons
      if(beta1<0){beta1=genColl[g].p()/genColl[g].energy();}else if(beta2<0){beta2=genColl[g].p()/genColl[g].energy();return;}
   }
}

double RescaledPt(const double& pt, const double& eta, const double& phi, const int& charge)
{
   double newInvPt = 1/pt+0.000236-0.000135*pow(eta,2)+charge*0.000282*TMath::Sin(phi-1.337);
   return 1/newInvPt;
}

unsigned long GetInitialNumberOfMCEvent(const vector<string>& fileNames)
{
   unsigned long Total = 0;
   fwlite::ChainEvent tree(fileNames);

   for(unsigned int f=0;f<fileNames.size();f++){
      TFile file(fileNames[f].c_str() );
      fwlite::LuminosityBlock ls( &file );
      for(ls.toBegin(); !ls.atEnd(); ++ls){
         fwlite::Handle<edm::MergeableCounter> nEventsTotalCounter;
         nEventsTotalCounter.getByLabel(ls,"nEventsBefSkim");
         if(!nEventsTotalCounter.isValid()){printf("Invalid nEventsTotalCounterH\n");continue;}
         Total+= nEventsTotalCounter->value;
      }
   }
   return Total;
}
