
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
namespace edm     {class TriggerResults; class TriggerResultsByName; class InputTag;}


#if !defined(__CINT__) && !defined(__MAKECINT__)
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPIsolation.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"

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
bool   PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const fwlite::ChainEvent& ev, stPlots* st=NULL, double GenBeta=-1);
bool PassSelection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const fwlite::ChainEvent& ev, const int& CutIndex=0, stPlots* st=NULL, double GenBeta=-1, const double& PtRescale=1.0, const double& IRescale=1.0);

void DumpCandidateInfo(const susybsm::HSCParticle& hscp, const fwlite::ChainEvent& ev, FILE* pFile);
bool PassTrigger      (const fwlite::ChainEvent& ev);
bool hasGoodPtHat     (const fwlite::ChainEvent& ev, const double& PtMax);

void SetWeight(const double& IntegratedLuminosityInPb=-1, const double& CrossSection=0, const double& MCEvents=0);
void SetWeightMC(const double& IntegratedLuminosityInPb, const double& SampleEquivalentLumi, const double& SampleSize, double MaxEvent);

/////////////////////////// VARIABLE DECLARATION /////////////////////////////

TFile* HistoFile;

TH1D*  Data_Pt ;
TH1D*  Data_I  ;
TH1D*  Data_TOF;

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

TH2D*  Pred_P    ;
TH2D*  Pred_I    ;
TH2D*  Pred_TOF  ;
TH2D*  DataD_P   ;
TH2D*  DataD_I   ;
TH2D*  DataD_TOF  ;

TH1D*  CtrlPt_BckgIs;
TH1D*  CtrlPt_BckgIm;
TH1D*  CtrlPt_BckgTOF;
TH1D*  CtrlPt_SignIs;
TH1D*  CtrlPt_SignIm;
TH1D*  CtrlPt_SignTOF;

TH1D*  CtrlIs_BckgPt;
TH1D*  CtrlIs_BckgTOF;
TH1D*  CtrlIs_SignPt;
TH1D*  CtrlIs_SignTOF;

TH1D*  CtrlTOF_BckgPt;
TH1D*  CtrlTOF_BckgIs;
TH1D*  CtrlTOF_SignPt;
TH1D*  CtrlTOF_SignIs;


std::vector<stSignal> signals;
std::vector<stMC>     MCsample;
std::vector<string>   DataFileName;

stPlots              DataPlots;  
std::vector<stPlots> SignPlots; 
std::vector<stPlots> MCPlots;  
stPlots              MCTrPlots;


/////////////////////////// CODE PARAMETERS /////////////////////////////



void Analysis_Step234(string MODE="COMPILE", int TypeMode_=0, string dEdxSel_="dedxASmi", string dEdxMass_="dedxHarm2", string TOF_Label_="combined", double CutPt_=-1.0, double CutI_=-1, double CutTOF_=-1, float MinPt_=GlobalMinPt, float MaxEta_=GlobalMaxEta, float MaxPtErr_=GlobalMaxPterr)
{
   if(MODE=="COMPILE")return;

   //////////////////////////////////////////////////     GLOBAL INIT
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


   if(TypeMode!=2){   
      for(double Pt =GlobalMinPt ; Pt <200;Pt+=10){
      for(double I  =GlobalMinI  ; I  <0.5 ;I+=0.025){
         CutPt .push_back(Pt);   CutI  .push_back(I);  CutTOF.push_back(-1);
      }}
   }else{
      for(double Pt =GlobalMinPt ; Pt <100;  Pt+=15){
      for(double I  =GlobalMinI  ; I  <0.35;  I+=0.05){
      for(double TOF=GlobalMinTOF; TOF<1.35;TOF+=0.05){
         CutPt .push_back(Pt);   CutI  .push_back(I);  CutTOF.push_back(TOF);
      }}}
   }
   std::cout << "CUT VECTOR SIZE = "<< CutPt.size() << endl;



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
   HistoFile = new TFile((string(Buffer) + "/Histos.root").c_str(),"RECREATE");
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


bool PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const fwlite::ChainEvent& ev, stPlots* st, double GenBeta)
{
   if(TypeMode==1 && !(hscp.type() == HSCParticleType::trackerMuon || hscp.type() == HSCParticleType::globalMuon))return false;
   if(TypeMode==2 && hscp.type() != HSCParticleType::globalMuon)return false;
   reco::TrackRef   track = hscp.trackRef(); if(track.isNull())return false;
   if(st){st->Total->Fill(0.0,Event_Weight);}

   if(st && GenBeta>=0)st->Beta_Matched->Fill(GenBeta, Event_Weight);

   if(fabs(track->eta())>GlobalMaxEta) return false;

   if(st){st->BS_Hits->Fill(track->found(),Event_Weight);}
   if(track->found()<GlobalMinNOH)return false;
   if(dedxSObj.numberOfMeasurements()<GlobalMinNOM)return false;
   if(st){st->Hits  ->Fill(0.0,Event_Weight);}

   double MuonTOF = GlobalMinTOF;
   double NDOF     = 9999;
   if(tof){
      MuonTOF = tof->inverseBeta();
      NDOF = tof->nDof();
   }

   if(st){st->BS_nDof->Fill(NDOF,Event_Weight);}
   if(NDOF<GlobalMinNDOF)return false;
   if(st){st->nDof  ->Fill(0.0,Event_Weight);}

   if(st){st->BS_Qual->Fill(track->qualityMask(),Event_Weight);}
   if(track->qualityMask()<GlobalMinQual )return false;
   if(st){st->Qual  ->Fill(0.0,Event_Weight);}

   if(st){st->BS_Chi2->Fill(track->chi2()/track->ndof(),Event_Weight);}
   if(track->chi2()/track->ndof()>GlobalMaxChi2 )return false;
   if(st){st->Chi2  ->Fill(0.0,Event_Weight);}

   if(st && GenBeta>=0)st->Beta_PreselectedA->Fill(GenBeta, Event_Weight);


   if(st){st->BS_MPt ->Fill(track->pt(),Event_Weight);}
   if(track->pt()<GlobalMinPt)return false;
   if(st){st->MPt   ->Fill(0.0,Event_Weight);}

   if(st){st->BS_MIs->Fill(dedxSObj.dEdx(),Event_Weight);}
   if(st){st->BS_MIm->Fill(dedxMObj.dEdx(),Event_Weight);}
   if(dedxSObj.dEdx()<GlobalMinI)return false;
   if(dedxMObj.dEdx()<3.2)return false;
   if(st){st->MI   ->Fill(0.0,Event_Weight);}

   if(st){st->BS_MTOF ->Fill(MuonTOF,Event_Weight);}
   if(MuonTOF<GlobalMinTOF)return false;
   if(tof && tof->inverseBetaErr()/tof->inverseBeta()>0.2)return false;
   if(st){st->MTOF ->Fill(0.0,Event_Weight);}

   if(st && GenBeta>=0)st->Beta_PreselectedB->Fill(GenBeta, Event_Weight);

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
   if(std::max(0.0,track->pt() - track->ptError())<GlobalMinPt)return false;
   if(st){st->Pterr   ->Fill(0.0,Event_Weight);}

   if(st && GenBeta>=0)st->Beta_PreselectedC->Fill(GenBeta, Event_Weight);

   if(st){st->BS_P  ->Fill(track->p(),Event_Weight);}
   if(st){st->BS_Pt ->Fill(track->pt(),Event_Weight);}
   if(st){st->BS_Is ->Fill(dedxSObj.dEdx(),Event_Weight);}
   if(st){st->BS_Im ->Fill(dedxMObj.dEdx(),Event_Weight);}
   if(st){st->BS_TOF->Fill(MuonTOF,Event_Weight);}

   if(st){st->BS_EtaIs->Fill(track->eta(),dedxSObj.dEdx(),Event_Weight);}
   if(st){st->BS_EtaIm->Fill(track->eta(),dedxMObj.dEdx(),Event_Weight);}
   if(st){st->BS_EtaP ->Fill(track->eta(),track->p(),Event_Weight);}
   if(st){st->BS_EtaPt->Fill(track->eta(),track->pt(),Event_Weight);}
   if(st){st->BS_PIs  ->Fill(track->p()  ,dedxSObj.dEdx(),Event_Weight);}
   if(st){st->BS_PIm  ->Fill(track->p()  ,dedxMObj.dEdx(),Event_Weight);}
   if(st){st->BS_PtIs ->Fill(track->pt() ,dedxSObj.dEdx(),Event_Weight);}
   if(st){st->BS_PtIm ->Fill(track->pt() ,dedxMObj.dEdx(),Event_Weight);}
   if(st){st->BS_TOFIs->Fill(MuonTOF     ,dedxSObj.dEdx(),Event_Weight);}
   if(st){st->BS_TOFIm->Fill(MuonTOF     ,dedxMObj.dEdx(),Event_Weight);}

   return true;
}

bool PassSelection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const fwlite::ChainEvent& ev, const int& CutIndex, stPlots* st, double GenBeta, const double& PtRescale, const double& IRescale){
   reco::TrackRef   track = hscp.trackRef(); if(track.isNull())return false;

   double MuonTOF = GlobalMinTOF;
   double NDOF     = 9999;
   if(tof){
      MuonTOF = tof->inverseBeta();
      NDOF = tof->nDof();
   }

   if(track->pt()*PtRescale<CutPt[CutIndex])return false;
   if(std::max(0.0,track->pt() - track->ptError())<CutPt[CutIndex])return false;
   if(st){st->Pt    ->Fill(CutIndex,Event_Weight);}
   if(st && GenBeta>=0)st->Beta_SelectedP->Fill(CutIndex,GenBeta, Event_Weight);

   if(dedxSObj.dEdx()*IRescale<CutI[CutIndex])return false;
   if(st){st->I    ->Fill(CutIndex,Event_Weight);}
   if(st && GenBeta>=0)st->Beta_SelectedI->Fill(CutIndex, GenBeta, Event_Weight);

   if(MuonTOF<CutTOF[CutIndex])return false;
   if(st){st->TOF  ->Fill(CutIndex,Event_Weight);}
   if(st && GenBeta>=0)st->Beta_SelectedT->Fill(CutIndex, GenBeta, Event_Weight);

   if(st){st->AS_P  ->Fill(CutIndex,track->p(),Event_Weight);}
   if(st){st->AS_Pt ->Fill(CutIndex,track->pt(),Event_Weight);}
   if(st){st->AS_Is ->Fill(CutIndex,dedxSObj.dEdx(),Event_Weight);}
   if(st){st->AS_Im ->Fill(CutIndex,dedxMObj.dEdx(),Event_Weight);}
   if(st){st->AS_TOF->Fill(CutIndex,MuonTOF,Event_Weight);}

   if(st){st->AS_EtaIs->Fill(CutIndex,track->eta(),dedxSObj.dEdx(),Event_Weight);}
   if(st){st->AS_EtaIm->Fill(CutIndex,track->eta(),dedxMObj.dEdx(),Event_Weight);}
   if(st){st->AS_EtaP ->Fill(CutIndex,track->eta(),track->p(),Event_Weight);}
   if(st){st->AS_EtaPt->Fill(CutIndex,track->eta(),track->pt(),Event_Weight);}
   if(st){st->AS_PIs  ->Fill(CutIndex,track->p()  ,dedxSObj.dEdx(),Event_Weight);}
   if(st){st->AS_PIm  ->Fill(CutIndex,track->p()  ,dedxMObj.dEdx(),Event_Weight);}
   if(st){st->AS_PtIs ->Fill(CutIndex,track->pt() ,dedxSObj.dEdx(),Event_Weight);}
   if(st){st->AS_PtIm ->Fill(CutIndex,track->pt() ,dedxMObj.dEdx(),Event_Weight);}
   if(st){st->AS_TOFIs->Fill(CutIndex,MuonTOF     ,dedxSObj.dEdx(),Event_Weight);}
   if(st){st->AS_TOFIm->Fill(CutIndex,MuonTOF     ,dedxMObj.dEdx(),Event_Weight);}

   return true;
}

void DumpCandidateInfo(const susybsm::HSCParticle& hscp, const fwlite::ChainEvent& ev, FILE* pFile)
{
   reco::MuonRef  muon  = hscp.muonRef();
   reco::TrackRef track = hscp.trackRef();
   if(track.isNull())return;

   fwlite::Handle< std::vector<reco::Vertex> > vertexCollHandle;
   vertexCollHandle.getByLabel(ev,"offlinePrimaryVertices");
   if(!vertexCollHandle.isValid()){printf("Vertex Collection NotFound\n");return;}
   std::vector<reco::Vertex> vertexColl = *vertexCollHandle;
   if(vertexColl.size()<1){printf("NO VERTEX\n"); return;}
   const reco::Vertex& vertex = vertexColl[0];

   fwlite::Handle<DeDxDataValueMap> dEdxSCollH;
   dEdxSCollH.getByLabel(ev, dEdxS_Label.c_str());
   if(!dEdxSCollH.isValid()){printf("Invalid dEdx Selection collection\n");return;}
   DeDxData dedxSObj  = dEdxSCollH->get(track.key());

   fwlite::Handle<DeDxDataValueMap> dEdxMCollH;
   dEdxMCollH.getByLabel(ev, dEdxM_Label.c_str());
   if(!dEdxMCollH.isValid()){printf("Invalid dEdx Mass collection\n");return;}
   DeDxData dedxMObj  = dEdxMCollH->get(track.key());


   double PBinned = Pred_P->GetXaxis()->GetBinCenter(Pred_P->GetXaxis()->FindBin(track->p()));
   double IBinned = Pred_I->GetXaxis()->GetBinCenter(Pred_I->GetXaxis()->FindBin(dedxMObj.dEdx()));
//   double TBinned = Pred_TOF[0]->GetXaxis()->GetBinCenter(Pred_TOF[0]->GetXaxis()->FindBin());

   double Mass = GetMass(PBinned,IBinned);   
   double MassExact = GetMass(track->p(),dedxMObj.dEdx(), true);
   double dz  = track->dz (vertex.position());
   double dxy = track->dxy(vertex.position());

   fprintf(pFile,"\n");
   fprintf(pFile,"---------------------------------------------------------------------------------------------------\n");
   fprintf(pFile,"Candidate Type = %i --> Mass (Binned): %7.2f GeV  Mass (UnBinned): %7.2f\n",hscp.type(),Mass, MassExact);
   fprintf(pFile,"------------------------------------------ EVENT INFO ---------------------------------------------\n");
   fprintf(pFile,"Run=%i Lumi=%i Event=%i BX=%i  Orbit=%i Store=%i\n",ev.eventAuxiliary().run(),ev.eventAuxiliary().luminosityBlock(),ev.eventAuxiliary().event(),ev.eventAuxiliary().luminosityBlock(),ev.eventAuxiliary().orbitNumber(),ev.eventAuxiliary().storeNumber());
   fprintf(pFile,"------------------------------------------ INNER TRACKER ------------------------------------------\n");
   fprintf(pFile,"Quality = %i Chi2/NDF=%6.2f dz=+%6.2f dxy=%+6.2f charge:%+i\n",track->qualityMask(), track->chi2()/track->ndof(), dz, dxy, track->charge());
   fprintf(pFile,"P=%7.2f  Pt=%7.2f+-%6.2f (Cut=%6.2f) Eta=%+6.2f  Phi=%+6.2f  NOH=%2i\n",track->p(),track->pt(), track->ptError(), CutPt[0], track->eta(), track->phi(), track->found() );

   fprintf(pFile,"------------------------------------------ DEDX INFO ----------------------------------------------\n");
   fprintf(pFile,"dEdx for selection:%6.2f (Cut=%6.2f) NOM %2i NOS %2i\n",dedxSObj.dEdx(),CutI[0],dedxSObj.numberOfMeasurements(),dedxSObj.numberOfSaturatedMeasurements());
   fprintf(pFile,"dEdx for mass reco:%6.2f             NOM %2i NOS %2i\n",dedxMObj.dEdx(),dedxMObj.numberOfMeasurements(),dedxMObj.numberOfSaturatedMeasurements());
   if(!muon.isNull()){
      fprintf(pFile,"------------------------------------------ MUON INFO ----------------------------------------------\n");
      fwlite::Handle<MuonTimeExtraMap> TOFDTCollH;
      TOFDTCollH.getByLabel(ev, "muontiming","dt");
      if(!TOFDTCollH.isValid()){printf("Invalid TOF DT collection\n");return;}
      MuonTimeExtra tofDT      = TOFDTCollH->get(hscp.muonRef().key());

      fwlite::Handle<MuonTimeExtraMap> TOFCSCCollH;
      TOFCSCCollH.getByLabel(ev, "muontiming","csc");
      if(!TOFDTCollH.isValid()){printf("Invalid TOF CSC collection\n");return;}
      MuonTimeExtra tofCSC      = TOFCSCCollH->get(hscp.muonRef().key());

      fwlite::Handle<MuonTimeExtraMap> TOFCombCollH;
      TOFCombCollH.getByLabel(ev, "muontiming","combined");
      if(!TOFCombCollH.isValid()){printf("Invalid TOF Combined collection\n");return;}
      MuonTimeExtra tofComb      = TOFCombCollH->get(hscp.muonRef().key());

      fprintf(pFile,"MassTOF = %7.2fGeV\n",GetTOFMass(track->p(),tofComb.inverseBeta()));

      fprintf(pFile,"Quality=%i type=%i P=%7.2f  Pt=%7.2f Eta=%+6.2f Phi=%+6.2f #Chambers=%i\n" ,muon->isQualityValid(),muon->type(),muon->p(),muon->pt(),muon->eta(),muon->phi(),muon->numberOfChambers());
      fprintf(pFile,"muonTimeDT      : NDOF=%2i InvBeta=%6.2f+-%6.2f (Cut=%6.2f) --> beta=%6.2f FreeInvBeta=%6.2f+-%6.2f\n",tofDT  .nDof(),tofDT  .inverseBeta(), tofDT  .inverseBetaErr(), CutTOF[0], (1.0/tofDT  .inverseBeta()), tofDT  .freeInverseBeta(),tofDT  .freeInverseBetaErr());
      fprintf(pFile,"muonTimeCSC     : NDOF=%2i InvBeta=%6.2f+-%6.2f (Cut=%6.2f) --> beta=%6.2f FreeInvBeta=%6.2f+-%6.2f\n",tofCSC .nDof(),tofCSC .inverseBeta(), tofCSC .inverseBetaErr(), CutTOF[0], (1.0/tofCSC .inverseBeta()), tofCSC .freeInverseBeta(),tofCSC .freeInverseBetaErr());
      fprintf(pFile,"muonTimeCombined: NDOF=%2i InvBeta=%6.2f+-%6.2f (Cut=%6.2f) --> beta=%6.2f FreeInvBeta=%6.2f+-%6.2f\n",tofComb.nDof(),tofComb.inverseBeta(), tofComb.inverseBetaErr(), CutTOF[0], (1.0/tofComb.inverseBeta()), tofComb.freeInverseBeta(),tofComb.freeInverseBetaErr());
   }
   if(hscp.hasRpcInfo()){
      fprintf(pFile,"------------------------------------------ RPC INFO -----------------------------------------------\n");
      fprintf(pFile,"isCandidate %i Beta=%6.2f\n",hscp.rpc().isCandidate,hscp.rpc().beta);
   }
   if(hscp.hasCaloInfo() && hscp.caloInfoRef()->ecalTime!=-9999){
      fprintf(pFile,"------------------------------------------ CALO INFO ----------------------------------------------\n");
      fprintf(pFile,"HCAL: E=%6.2f E3x3=%6.2f E5x5=%6.2f HO E=%6.2f\n",hscp.caloInfoRef()->hcalCrossedEnergy,hscp.caloInfoRef()->hcal3by3dir, hscp.caloInfoRef()->hcal5by5dir, hscp.caloInfoRef()->hoCrossedEnergy);
      fprintf(pFile,"ECAL: E=%6.2f E3x3=%6.2f E5x5=%6.2f\n"           ,hscp.caloInfoRef()->ecalCrossedEnergy,hscp.caloInfoRef()->ecal3by3dir, hscp.caloInfoRef()->ecal5by5dir);
      fprintf(pFile,"ECAL: time=%6.2f beta=%6.2f trkisodr=%6.2f\n"    ,hscp.caloInfoRef()->ecalTime  ,hscp.caloInfoRef()->ecalBeta   , hscp.caloInfoRef()->trkIsoDr);
   }
   fprintf(pFile,"------------------------------------------ ISOL INFO ----------------------------------------------\n");
   fwlite::Handle<HSCPIsolationValueMap> IsolationH05;
   IsolationH05.getByLabel(ev, "HSCPIsolation05");
   if(!IsolationH05.isValid()){printf("Invalid IsolationH\n");return;}
   const ValueMap<HSCPIsolation>& IsolationMap05 = *IsolationH05.product();

   fwlite::Handle<HSCPIsolationValueMap> IsolationH03;
   IsolationH03.getByLabel(ev, "HSCPIsolation03");
   if(!IsolationH03.isValid()){printf("Invalid IsolationH\n");return;}
   const ValueMap<HSCPIsolation>& IsolationMap03 = *IsolationH03.product();

   fwlite::Handle<HSCPIsolationValueMap> IsolationH01;
   IsolationH01.getByLabel(ev, "HSCPIsolation01");
   if(!IsolationH01.isValid()){printf("Invalid IsolationH\n");return;}
   const ValueMap<HSCPIsolation>& IsolationMap01 = *IsolationH01.product();

   HSCPIsolation hscpIso05 = IsolationMap05.get((size_t)track.key());
   HSCPIsolation hscpIso03 = IsolationMap03.get((size_t)track.key());
   HSCPIsolation hscpIso01 = IsolationMap01.get((size_t)track.key());
   fprintf(pFile,"Isolation05 --> TkCount=%6.2f TkSumEt=%6.2f EcalE/P=%6.2f HcalE/P=%6.2f --> E/P=%6.2f\n",hscpIso05.Get_TK_Count(), hscpIso05.Get_TK_SumEt(), hscpIso05.Get_ECAL_Energy()/track->p(), hscpIso05.Get_HCAL_Energy()/track->p(), (hscpIso05.Get_ECAL_Energy()+hscpIso05.Get_HCAL_Energy())/track->p());
   fprintf(pFile,"Isolation03 --> TkCount=%6.2f TkSumEt=%6.2f EcalE/P=%6.2f HcalE/P=%6.2f --> E/P=%6.2f\n",hscpIso03.Get_TK_Count(), hscpIso03.Get_TK_SumEt(), hscpIso03.Get_ECAL_Energy()/track->p(), hscpIso03.Get_HCAL_Energy()/track->p(), (hscpIso03.Get_ECAL_Energy()+hscpIso03.Get_HCAL_Energy())/track->p());
   fprintf(pFile,"Isolation01 --> TkCount=%6.2f TkSumEt=%6.2f EcalE/P=%6.2f HcalE/P=%6.2f --> E/P=%6.2f\n",hscpIso01.Get_TK_Count(), hscpIso01.Get_TK_SumEt(), hscpIso01.Get_ECAL_Energy()/track->p(), hscpIso01.Get_HCAL_Energy()/track->p(), (hscpIso01.Get_ECAL_Energy()+hscpIso01.Get_HCAL_Energy())/track->p());
   fprintf(pFile,"\n");
}

bool PassTrigger(const fwlite::ChainEvent& ev)
{
      edm::TriggerResultsByName tr = ev.triggerResultsByName("Merge");
      if(!tr.isValid())return false;
      if(tr.accept(tr.triggerIndex("HscpPathMu")))return true;
      if(tr.accept(tr.triggerIndex("HscpPathMet")))return true;
      return false;
}



void Analysis_Step3(char* SavePath)
{
   printf("Step3: Building Mass Spectrum for B and S\n");

   int TreeStep;
   //////////////////////////////////////////////////     BUILD BACKGROUND MASS SPECTRUM

   stPlots_Init(HistoFile, DataPlots,"Data", CutPt.size());
   HistoFile->cd();

   fwlite::ChainEvent treeD(DataFileName);
   SetWeight(-1);
   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Building Mass Spectrum for D :");
   TreeStep = treeD.size()/50;if(TreeStep==0)TreeStep=1;

   for(Long64_t ientry=0;ientry<treeD.size();ientry++){
      treeD.to(ientry);
      if(MaxEntry>0 && ientry>MaxEntry)break;
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}

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
      
      bool* HSCPTk = new bool[CutPt.size()]; for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk[CutIndex] = false;   }
      for(unsigned int c=0;c<hscpColl.size();c++){
         susybsm::HSCParticle hscp  = hscpColl[c];
         reco::MuonRef  muon  = hscp.muonRef();
         reco::TrackRef track = hscp.trackRef();
         if(track.isNull())continue;

         const DeDxData& dedxSObj  = dEdxSCollH->get(track.key());
         const DeDxData& dedxMObj  = dEdxMCollH->get(track.key());
         const reco::MuonTimeExtra* tof = NULL;
        if(TypeMode==2 && !hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); }


         double MuonTOF = GlobalMinTOF;
         if(tof){MuonTOF = tof->inverseBeta(); }
 
         ///////////////////////////////  PREDICTION BEGINS ////////////////////////////////

         if(!PassPreselection(hscp, dedxSObj, dedxMObj, tof, treeD, &DataPlots))continue;

	 Data_Pt->Fill(track->pt(),Event_Weight);
         Data_I->Fill(dedxSObj.dEdx(),Event_Weight);
         Data_TOF->Fill(MuonTOF,Event_Weight);


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

            if(track->pt()<35){
               CtrlPt_BckgIs->Fill(dedxSObj.dEdx(), Event_Weight);
               CtrlPt_BckgIm->Fill(dedxMObj.dEdx(), Event_Weight);
               if(MuonTOF!=GlobalMinTOF)CtrlPt_BckgTOF->Fill(MuonTOF, Event_Weight);
            }else{
               CtrlPt_SignIs->Fill(dedxSObj.dEdx(), Event_Weight);
               CtrlPt_SignIm->Fill(dedxMObj.dEdx(), Event_Weight);
               if(MuonTOF!=GlobalMinTOF)CtrlPt_SignTOF->Fill(MuonTOF, Event_Weight);
            }

            if(dedxSObj.dEdx()<0.2){
               CtrlIs_BckgPt->Fill(track->pt(), Event_Weight);
               if(MuonTOF!=GlobalMinTOF)CtrlIs_BckgTOF->Fill(MuonTOF, Event_Weight);
            }else{
               CtrlIs_SignPt->Fill(track->pt(), Event_Weight);
               if(MuonTOF!=GlobalMinTOF)CtrlIs_SignTOF->Fill(MuonTOF, Event_Weight);
            }

            if(MuonTOF!=GlobalMinTOF){
            if(MuonTOF<1.1){
               CtrlTOF_BckgPt->Fill(track->pt()    , Event_Weight);
               CtrlTOF_BckgIs->Fill(dedxSObj.dEdx(), Event_Weight);
            }else{
               CtrlTOF_SignPt->Fill(track->pt()    , Event_Weight);
               CtrlTOF_SignIs->Fill(dedxSObj.dEdx(), Event_Weight);
            }
            }

         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
//            bool PassPtCut  = track->pt()>=CutPt[CutIndex];
            bool PassPtCut  = track->pt()- track->ptError()>=CutPt[CutIndex];
            bool PassICut   = (dedxSObj.dEdx()>=CutI[CutIndex]);
            bool PassTOFCut = MuonTOF>=CutTOF[CutIndex];

            if(       PassTOFCut &&  PassPtCut &&  PassICut){   //Region D
               H_D      ->Fill(CutIndex,                Event_Weight);
               DataD_P  ->Fill(CutIndex,track->p(),     Event_Weight);
               DataD_I  ->Fill(CutIndex,dedxMObj.dEdx(),Event_Weight);
               DataD_TOF->Fill(CutIndex,MuonTOF,        Event_Weight);
            }else if( PassTOFCut &&  PassPtCut && !PassICut){   //Region C
               H_C     ->Fill(CutIndex,                 Event_Weight);
               Pred_P  ->Fill(CutIndex,track->p(),      Event_Weight);
               Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);
            }else if( PassTOFCut && !PassPtCut &&  PassICut){   //Region B
               H_B     ->Fill(CutIndex,                 Event_Weight);
               Pred_I  ->Fill(CutIndex,dedxMObj.dEdx(), Event_Weight);
               Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);
            }else if( PassTOFCut && !PassPtCut && !PassICut){   //Region A
               H_A     ->Fill(CutIndex,                 Event_Weight);
               Pred_TOF->Fill(CutIndex,MuonTOF,         Event_Weight);
            }else if(!PassTOFCut &&  PassPtCut &&  PassICut){   //Region H
               H_H   ->Fill(CutIndex,          Event_Weight);
               Pred_P->Fill(CutIndex,track->p(),        Event_Weight);
               Pred_I->Fill(CutIndex,dedxMObj.dEdx(),   Event_Weight);
            }else if(!PassTOFCut &&  PassPtCut && !PassICut){   //Region G
               H_G     ->Fill(CutIndex,                 Event_Weight);
               Pred_P  ->Fill(CutIndex,track->p(),      Event_Weight);
            }else if(!PassTOFCut && !PassPtCut &&  PassICut){   //Region F
               H_F     ->Fill(CutIndex,                 Event_Weight);
               Pred_I  ->Fill(CutIndex,dedxMObj.dEdx(), Event_Weight);
            }else if(!PassTOFCut && !PassPtCut && !PassICut){   //Region E
               H_E     ->Fill(CutIndex,                 Event_Weight);
            }
         ///////////////////////////////  PREDICTION ENDS   ////////////////////////////////

            //DEBUG
         }



/*         double PBinned = Pred_P->GetYaxis()->GetBinCenter(Pred_P->GetYaxis()->FindBin(track->p()));
         double IBinned = Pred_I->GetYaxis()->GetBinCenter(Pred_I->GetYaxis()->FindBin(dedxMObj.dEdx()));
         double TBinned = -1; if(tof)TBinned = Pred_TOF->GetYaxis()->GetBinCenter(Pred_TOF->GetYaxis()->FindBin(tof->inverseBeta()));

         double Mass     = GetMass(PBinned,IBinned);
         double MassTOF  = -1; if(tof)GetTOFMass(PBinned,TBinned);
         double MassComb = (Mass+MassTOF)*0.5;
*/

         double Mass     = GetMass(track->p(),dedxMObj.dEdx());
         double MassTOF  = -1; if(tof)GetTOFMass(track->p(),tof->inverseBeta());
         double MassComb = (Mass+MassTOF)*0.5;


         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
         //Full Selection
         if(!PassSelection   (hscp, dedxSObj, dedxMObj, tof, treeD, CutIndex, &DataPlots))continue;
         HSCPTk[CutIndex] = true;

	 DataPlots.Mass->Fill(CutIndex, Mass,Event_Weight);
         if(tof){
            DataPlots.MassTOF ->Fill(CutIndex, MassTOF , Event_Weight);
            DataPlots.MassComb->Fill(CutIndex, MassComb, Event_Weight);
         }

         } //end of Cut loop
          //DEBUG

         if(track->pt()>40 && Mass>75)stPlots_FillTree(DataPlots, treeD.eventAuxiliary().run(),treeD.eventAuxiliary().event(), c, track->pt(), dedxSObj.dEdx(), tof ? tof->inverseBeta() : -1);
      } // end of Track Loop
          TH1D* tmp = DataPlots.Mass->ProjectionY("TMP",44,44);
          printf("D=%7.0f M=%7.0f\n",H_D->GetBinContent(44), tmp->GetEntries());
          delete tmp;

      for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  if(HSCPTk[CutIndex]){DataPlots.HSCPE->Fill(CutIndex,Event_Weight); }  }
   }// end of Event Loop
   //stPlots_CloseTree(DataPlots);
   printf("\n");

   stPlots_Clear(DataPlots, true);


//   for(int CutIndex=0;CutIndex<CutPt.size();CutIndex++) printf("CutIndex=%3i Pt>%6.2f, I>%6.2F TOF>%6.2F --> A=%6.2E B=%6.E C=%6.2E D=%6.2E E=%6.2E F=%6.2E G=%6.2E H=%6.2E\n",CutIndex,CutPt[CutIndex], CutI[CutIndex], CutTOF[CutIndex], H_A->GetBinContent(CutIndex+1), H_B->GetBinContent(CutIndex+1), H_C->GetBinContent(CutIndex+1), H_D->GetBinContent(CutIndex+1), H_E->GetBinContent(CutIndex+1), H_F->GetBinContent(CutIndex+1), H_G->GetBinContent(CutIndex+1), H_H->GetBinContent(CutIndex+1) );




   //////////////////////////////////////////////////     BUILD MCTRUTH MASS SPECTRUM
   stPlots_Init(HistoFile, MCTrPlots,"MCTr", CutPt.size());
   for(unsigned int m=0;m<MCsample.size();m++){
      stPlots_Init(HistoFile,MCPlots[m],MCsample[m].Name, CutPt.size());

      std::vector<string> FileName;
      GetInputFiles(FileName, MCsample[m].Name);

      fwlite::ChainEvent treeM(FileName);
      SetWeightMC(IntegratedLuminosity,MCsample[m].ILumi, treeM.size(), MCsample[m].MaxEvent);
      printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
      printf("Building Mass for %10s :",MCsample[m].Name.c_str());
      TreeStep = treeM.size()/50;if(TreeStep==0)TreeStep=1;
      for(Long64_t ientry=0;ientry<treeM.size();ientry++){       
          treeM.to(ientry);
         if(MaxEntry>0 && ientry>MaxEntry)break;
         if(MCsample[m].MaxEvent>0 && ientry>MCsample[m].MaxEvent)break;
         if(ientry%TreeStep==0){printf(".");fflush(stdout);}

         if(!hasGoodPtHat(treeM, MCsample[m].MaxPtHat)){continue;}


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
         
         bool* HSCPTk = new bool[CutPt.size()]; for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk[CutIndex] = false;   }
         for(unsigned int c=0;c<hscpColl.size();c++){
            susybsm::HSCParticle hscp  = hscpColl[c];
            reco::MuonRef  muon  = hscp.muonRef();
            reco::TrackRef track = hscp.trackRef();
            if(track.isNull())continue;

            const DeDxData& dedxSObj  = dEdxSCollH->get(track.key());
            const DeDxData& dedxMObj  = dEdxMCollH->get(track.key());
            const reco::MuonTimeExtra* tof = NULL;
            if(TypeMode==2 && !hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); }

                PassPreselection(hscp, dedxSObj, dedxMObj, tof, treeM,           &MCPlots[m]);
            if(!PassPreselection(hscp, dedxSObj, dedxMObj, tof, treeM,           &MCTrPlots))continue;


/*            double PBinned = Pred_P->GetYaxis()->GetBinCenter(Pred_P->GetYaxis()->FindBin(track->p()));
            double IBinned = Pred_I->GetYaxis()->GetBinCenter(Pred_I->GetYaxis()->FindBin(dedxMObj.dEdx()));
            double TBinned = -1; if(tof)TBinned = Pred_TOF->GetYaxis()->GetBinCenter(Pred_TOF->GetYaxis()->FindBin(tof->inverseBeta()));


            double Mass     = GetMass(PBinned,IBinned, true);
            double MassTOF  = -1; if(tof)GetTOFMass(PBinned,TBinned);
            double MassComb = (Mass+MassTOF)*0.5;
*/

         double Mass     = GetMass(track->p(),dedxMObj.dEdx());
         double MassTOF  = -1; if(tof)GetTOFMass(track->p(),tof->inverseBeta());
         double MassComb = (Mass+MassTOF)*0.5;


            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){

                PassSelection   (hscp, dedxSObj, dedxMObj, tof, treeM, CutIndex, &MCPlots[m]);
            if(!PassSelection   (hscp, dedxSObj, dedxMObj, tof, treeM, CutIndex, &MCTrPlots))continue;
            HSCPTk[CutIndex] = true;

            MCTrPlots .Mass->Fill(CutIndex , Mass,Event_Weight);
            MCPlots[m].Mass->Fill(CutIndex, Mass,Event_Weight);

            if(tof){
               MCPlots[m].MassTOF ->Fill(CutIndex, MassTOF , Event_Weight);
               MCPlots[m].MassComb->Fill(CutIndex, MassComb, Event_Weight);

               MCPlots[m].MassTOF ->Fill(CutIndex, MassTOF , Event_Weight);
               MCPlots[m].MassComb->Fill(CutIndex, MassComb, Event_Weight);
            }
         } //end of Cut loop
         if(track->pt()>40 && Mass>75)stPlots_FillTree(MCTrPlots , treeM.eventAuxiliary().run(),treeM.eventAuxiliary().event(), c, track->pt(), dedxSObj.dEdx(), tof ? tof->inverseBeta() : -1);
         if(track->pt()>40 && Mass>75)stPlots_FillTree(MCPlots[m], treeM.eventAuxiliary().run(),treeM.eventAuxiliary().event(), c, track->pt(), dedxSObj.dEdx(), tof ? tof->inverseBeta() : -1);

         } // end of Track Loop 
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  if(HSCPTk[CutIndex]){MCTrPlots .HSCPE->Fill(CutIndex,Event_Weight);MCPlots[m].HSCPE->Fill(CutIndex,Event_Weight); } }
      }// end of Event Loop
      //stPlots_CloseTree(MCPlots[m]);
      stPlots_Clear(MCPlots[m], true);
      printf("\n");
   }
   //stPlots_CloseTree(MCTrPlots);
   stPlots_Clear(MCTrPlots, true);


   //////////////////////////////////////////////////     BUILD SIGNAL MASS SPECTRUM

   for(unsigned int s=0;s<signals.size();s++){
      stPlots_Init(HistoFile,SignPlots[4*s+0],signals[s].Name       , CutPt.size());
      stPlots_Init(HistoFile,SignPlots[4*s+1],signals[s].Name+"_NC0", CutPt.size());
      stPlots_Init(HistoFile,SignPlots[4*s+2],signals[s].Name+"_NC1", CutPt.size());
      stPlots_Init(HistoFile,SignPlots[4*s+3],signals[s].Name+"_NC2", CutPt.size());


      std::vector<string> SignFileName;
      GetInputFiles(SignFileName, signals[s].Name);

      fwlite::ChainEvent treeS(SignFileName);
      SetWeight(IntegratedLuminosity,signals[s].XSec,(double)treeS.size());
      printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
      printf("Building Mass for %10s :",signals[s].Name.c_str());
      TreeStep = treeS.size()/50;if(TreeStep==0)TreeStep=1;


      for(Long64_t ientry=0;ientry<treeS.size();ientry++){
         treeS.to(ientry);
         if(MaxEntry>0 && ientry>MaxEntry)break;
         if(ientry%TreeStep==0){printf(".");fflush(stdout);}

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

         bool* HSCPTk = new bool[CutPt.size()]; for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  HSCPTk[CutIndex] = false;   }
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
            if(TypeMode==2 && !hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); }

               PassPreselection(hscp,  dedxSObj, dedxMObj, tof, treeS,           &SignPlots[4*s+NChargedHSCP+1], genColl[ClosestGen].p()/genColl[ClosestGen].energy());
            if(!PassPreselection(hscp,  dedxSObj, dedxMObj, tof, treeS,           &SignPlots[4*s               ], genColl[ClosestGen].p()/genColl[ClosestGen].energy()))continue;         


/*            double PBinned = Pred_P->GetYaxis()->GetBinCenter(Pred_P->GetYaxis()->FindBin(track->p()));
            double IBinned = Pred_I->GetYaxis()->GetBinCenter(Pred_I->GetYaxis()->FindBin(dedxMObj.dEdx()));
            double TBinned = -1; if(tof)TBinned = Pred_TOF->GetYaxis()->GetBinCenter(Pred_TOF->GetYaxis()->FindBin(tof->inverseBeta()));

            double Mass     = GetMass(PBinned,IBinned, true);
            double MassTOF  = -1; if(tof)GetTOFMass(PBinned,TBinned);
            double MassComb = (Mass+MassTOF)*0.5;
*/

            double Mass     = GetMass(track->p(),dedxMObj.dEdx());
            double MassTOF  = -1; if(tof)GetTOFMass(track->p(),tof->inverseBeta());
            double MassComb = (Mass+MassTOF)*0.5;


            for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){
                PassSelection   (hscp,  dedxSObj, dedxMObj, tof, treeS, CutIndex, &SignPlots[4*s+NChargedHSCP+1], genColl[ClosestGen].p()/genColl[ClosestGen].energy());
            if(!PassSelection   (hscp,  dedxSObj, dedxMObj, tof, treeS, CutIndex, &SignPlots[4*s               ], genColl[ClosestGen].p()/genColl[ClosestGen].energy()))continue;    

            HSCPTk[CutIndex] = true;

            SignPlots[4*s               ].Mass->Fill(CutIndex, Mass,Event_Weight);
            SignPlots[4*s+NChargedHSCP+1].Mass->Fill(CutIndex, Mass,Event_Weight);
            if(tof){
               SignPlots[4*s               ].MassTOF ->Fill(CutIndex, MassTOF , Event_Weight);
               SignPlots[4*s               ].MassComb->Fill(CutIndex, MassComb, Event_Weight);
               SignPlots[4*s+NChargedHSCP+1].MassTOF ->Fill(CutIndex, MassTOF , Event_Weight);
               SignPlots[4*s+NChargedHSCP+1].MassComb->Fill(CutIndex, MassComb, Event_Weight);
            }
         } //end of Cut loop
            if(track->pt()>40 && Mass>75)stPlots_FillTree(SignPlots[4*s               ] , treeS.eventAuxiliary().run(),treeS.eventAuxiliary().event(), c, track->pt(), dedxSObj.dEdx(), tof ? tof->inverseBeta() : -1);
         } // end of Track Loop 
         for(unsigned int CutIndex=0;CutIndex<CutPt.size();CutIndex++){  if(HSCPTk[CutIndex]){SignPlots[4*s               ].HSCPE      ->Fill(CutIndex,Event_Weight); SignPlots[4*s+NChargedHSCP+1].HSCPE      ->Fill(CutIndex,Event_Weight); } }
       }// end of Event Loop
      printf("\n");
      //stPlots_CloseTree(SignPlots[4*s  ]);

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
   for(int i=0;i<=PDF->GetNbinsX();i++){
      if(PDF->GetBinContent(i)>uniform){
         return PDF->GetXaxis()->GetBinLowEdge(i);
      }
   }
   return PDF->GetXaxis()->GetBinLowEdge(PDF->GetNbinsX());
}



void Analysis_Step4(char* SavePath)
{
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

      TH1D* Pred_P_Proj = Pred_P->ProjectionY("ProjP",CutIndex+1,CutIndex+1);
      TH1D* Pred_I_Proj = Pred_I->ProjectionY("ProjI",CutIndex+1,CutIndex+1);
      TH1D* Pred_T_Proj = Pred_TOF->ProjectionY("ProjT",CutIndex+1,CutIndex+1);

      TH1D* Pred_P_PDF = GetPDF(Pred_P_Proj);
      TH1D* Pred_I_PDF = GetPDF(Pred_I_Proj);
      TH1D* Pred_T_PDF = GetPDF(Pred_T_Proj);

      TProfile* Pred_Prof_Mass     =  new TProfile("Pred_Prof_Mass"    ,"Pred_Prof_Mass"    ,100,0,MassHistoUpperBound); 
      TProfile* Pred_Prof_MassTOF  =  new TProfile("Pred_Prof_MassTOF" ,"Pred_Prof_MassTOF" ,100,0,MassHistoUpperBound);  
      TProfile* Pred_Prof_MassComb =  new TProfile("Pred_Prof_MassComb","Pred_Prof_MassComb",100,0,MassHistoUpperBound);


      TRandom3* RNG = new TRandom3();

      printf("Predicting (%4i / %4i)     :",CutIndex+1,CutPt.size());
      int TreeStep = 100/50;if(TreeStep==0)TreeStep=1;
      for(unsigned int pe=0;pe<100;pe++){    
      if(pe%TreeStep==0){printf(".");fflush(stdout);}

      TH1D* tmpH_Mass     =  new TH1D("tmpH_Mass"    ,"tmpH_Mass"    ,100,0,MassHistoUpperBound);
      TH1D* tmpH_MassTOF  =  new TH1D("tmpH_MassTOF" ,"tmpH_MassTOF" ,100,0,MassHistoUpperBound);
      TH1D* tmpH_MassComb =  new TH1D("tmpH_MassComb","tmpH_MassComb",100,0,MassHistoUpperBound);
      int III = 0;

      unsigned int NSimulation = 100000;
      double Wheight = RNG->Gaus(P,Perr) / NSimulation;
      for(unsigned int r=0;r<NSimulation;r++){
         double p = GetRandValue(Pred_P_PDF)*RNG->Gaus(1.0,0.10);
         double i = -1; while(i<3.2){ i=GetRandValue(Pred_I_PDF)*RNG->Gaus(1.0,0.10);}
/*
         double PBinned = Pred_P->GetYaxis()->GetBinCenter(Pred_P->GetYaxis()->FindBin(p));
         double IBinned = Pred_I->GetYaxis()->GetBinCenter(Pred_I->GetYaxis()->FindBin(i));

         double MI = GetMass(PBinned,IBinned);
         tmpH_Mass->Fill(MI,Wheight);
 
         if(TypeMode==2){
            double t  = GetRandValue(Pred_T_PDF)*RNG->Gaus(1.0,0.10);
            double TBinned = Pred_TOF->GetYaxis()->GetBinCenter(Pred_TOF->GetYaxis()->FindBin(t));
            double MT = GetTOFMass(PBinned,TBinned);
            tmpH_MassTOF->Fill(MT,Wheight);
            tmpH_MassComb->Fill((MI+MT)*0.5,Wheight);
         }
*/


         double MI = GetMass(p,i);
         if(CutIndex==43){
            if(MI<=0){
               printf("%2i M = %6.2fE  <-- p=%6.2f i=%6.2f\n",III,MI,p,i);
               III++;
      
            }
         }


         tmpH_Mass->Fill(MI,Wheight);
         if(TypeMode==2){
            double t  = GetRandValue(Pred_T_PDF)*RNG->Gaus(1.0,0.10);
            double MT = GetTOFMass(p,t);
            tmpH_MassTOF->Fill(MT,Wheight);
            tmpH_MassComb->Fill((MI+MT)*0.5,Wheight);
         }


      }

      for(int x=0;x<tmpH_Mass->GetNbinsX()+1;x++){
         double M = tmpH_Mass->GetXaxis()->GetBinCenter(x);
         Pred_Prof_Mass    ->Fill(M,tmpH_Mass    ->GetBinContent(x));
         Pred_Prof_MassTOF ->Fill(M,tmpH_MassTOF ->GetBinContent(x));
         Pred_Prof_MassComb->Fill(M,tmpH_MassComb->GetBinContent(x));
      }

      delete tmpH_Mass;
      delete tmpH_MassTOF;
      delete tmpH_MassComb;
     }printf("\n");

    for(int x=0;x<Pred_Mass->GetNbinsY()+1;x++){
       if(CutIndex==43 && x==0)printf("RescaleFactor=%6.2E\n",Perr/P);
       if(CutIndex==43)printf("MassBin=%6.2f  NEntries=%6.2E  StatError=%6.2E  RescaleError=%6.2E   Total=%7.2f\n",Pred_Prof_Mass->GetXaxis()->GetBinCenter(x),Pred_Prof_Mass->GetBinContent(x),Pred_Prof_Mass->GetBinError(x),Pred_Prof_Mass    ->GetBinContent(x)*(Perr/P),sqrt(pow(Pred_Prof_Mass->GetBinError(x),2) + Pred_Prof_Mass->GetBinContent(x)*(Perr/P)));
       Pred_Mass    ->SetBinContent(CutIndex+1,x,Pred_Prof_Mass    ->GetBinContent(x)); Pred_Mass      ->SetBinError(CutIndex+1,x,sqrt(pow(Pred_Prof_Mass    ->GetBinError(x),2) + Pred_Prof_Mass    ->GetBinContent(x)*(Perr/P)));
       Pred_MassTOF ->SetBinContent(CutIndex+1,x,Pred_Prof_MassTOF ->GetBinContent(x)); Pred_MassTOF   ->SetBinError(CutIndex+1,x,sqrt(pow(Pred_Prof_MassTOF ->GetBinError(x),2) + Pred_Prof_MassTOF ->GetBinContent(x)*(Perr/P)));
       Pred_MassComb->SetBinContent(CutIndex+1,x,Pred_Prof_MassComb->GetBinContent(x)); Pred_MassComb  ->SetBinError(CutIndex+1,x,sqrt(pow(Pred_Prof_MassComb->GetBinError(x),2) + Pred_Prof_MassComb->GetBinContent(x)*(Perr/P)));
    }

    delete Pred_Prof_Mass;
    delete Pred_Prof_MassTOF;
    delete Pred_Prof_MassComb;
    delete Pred_P_PDF;
    delete Pred_I_PDF;
    delete Pred_T_PDF;
    delete Pred_P_Proj;
    delete Pred_I_Proj;
    delete Pred_T_Proj;
   }


   //////////////////////////////////////////////////     DUMP USEFUL INFORMATION

   char Buffer[2048];
   sprintf(Buffer,"%s/Info.txt",SavePath);
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

      fprintf(pFile  ,"CutIndex=%4i --> (Pt>%6.2f I>%6.2f TOF>%6.2f) Ndata=%+6.2E  NPred=%6.3E+-%6.3E <--> A=%6.2E B=%6.E C=%6.2E D=%6.2E E=%6.2E F=%6.2E G=%6.2E H=%6.2E\n",CutIndex,HCuts_Pt ->GetBinContent(CutIndex+1), HCuts_I  ->GetBinContent(CutIndex+1), HCuts_TOF->GetBinContent(CutIndex+1), D,H_P->GetBinContent(CutIndex+1),H_P->GetBinError(CutIndex+1) ,A, B, C, D, E, F, G, H);
   }


/*
  fprintf(pFile,"Cut PT         = %6.2f --> Eff=%4.3E\n",CutPt [0], Efficiency(Data_Pt,CutPt[0]));
   fprintf(pFile,"Cut I          = %6.2f --> Eff=%4.3E\n",CutI  [0], Efficiency(Data_I,CutI[0]));
   fprintf(pFile,"Cut TOF        = %6.2f --> Eff=%4.3E\n",CutTOF[0], Efficiency(Data_TOF,CutTOF[0]));


   fprintf(pFile,"--------------------\n");

   fprintf(pFile,"\n\n--------------------\n");
   fprintf(pFile,"DATA SELECTION DETAILS\n");
   fprintf(pFile,"--------------------\n");
   stPlots_Dump(DataPlots, pFile, 0);

   fprintf(pFile,"\n\n--------------------\n");
   fprintf(pFile,"MC TRUTH SELECTION DETAILS\n");
   fprintf(pFile,"--------------------\n");
   stPlots_Dump(MCTrPlots, pFile, 0);

   for(unsigned int m=0;m<MCsample.size();m++){
      fprintf(pFile,"##### ##### %10s ##### #####\n",MCsample[m].Name.c_str());
      stPlots_Dump(MCPlots[m], pFile, 0);
   }

   fprintf(pFile,"\n\n--------------------\n");
   fprintf(pFile,"SIGNAL SELECTION DETAILS\n");
   fprintf(pFile,"--------------------\n");
   for(unsigned int s=0;s<SignPlots.size();s++){   
      fprintf(pFile,"##### ##### %10s ##### #####\n",SignPlots[s].Name.c_str());
      stPlots_Dump(SignPlots[s], pFile, 0);
   }

   fprintf(pFile,"\n\n--------------------\n");
   fprintf(pFile,"PREDICTION OF THE MASS DISTRIBUTION\n");
   fprintf(pFile,"--------------------\n");
   fprintf(pFile,"H_A->GetBinContent(CutIndex+1)=%E, H_B->GetBinContent(CutIndex+1)=%E, H_C->GetBinContent(CutIndex+1)=%E, H_D->GetBinContent(CutIndex+1)=%E  H_E->GetBinContent(CutIndex+1)=%E, H_F->GetBinContent(CutIndex+1)=%E, H_G->GetBinContent(CutIndex+1)=%E, H_H->GetBinContent(CutIndex+1)=%E<--> %E+-%E\n",H_A->GetBinContent(CutIndex+1),H_B->GetBinContent(CutIndex+1),H_C->GetBinContent(CutIndex+1),H_D->GetBinContent(CutIndex+1),H_E->GetBinContent(CutIndex+1),H_F->GetBinContent(CutIndex+1),H_G->GetBinContent(CutIndex+1),H_H->GetBinContent(CutIndex+1), H_P->GetBinContent(CutIndex+1),H_P->GetBinError(CutIndex+1));      
   fprintf(pFile,"--------------------\n");

   fprintf(pFile,"\nIntegral in range [0,2000]GeV:\n");
   double error = 0;
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","D ",GetEventInRange(0,2000,DataPlots.Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","P ",GetEventInRange(0,2000,Pred_Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","M ",GetEventInRange(0,2000,MCTrPlots.Mass,error),error);
   for(unsigned int s=0;s<signals.size();s++){
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n",signals[s].Name.c_str(),GetEventInRange(0,2000,SignPlots[4*s].Mass,error),error);
   }
   fprintf(pFile,"\nIntegral in range [100,2000]GeV:\n");
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","D ",GetEventInRange(100,2000,DataPlots.Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","P ",GetEventInRange(100,2000,Pred_Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","M ",GetEventInRange(100,2000,MCTrPlots.Mass,error),error);
   for(unsigned int s=0;s<signals.size();s++){
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n",signals[s].Name.c_str(),GetEventInRange(100,2000,SignPlots[4*s].Mass,error),error);
   }
   fprintf(pFile,"\nIntegral in range [200,2000]GeV:\n");
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","D ",GetEventInRange(125,2000,DataPlots.Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","P ",GetEventInRange(125,2000,Pred_Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","M ",GetEventInRange(125,2000,MCTrPlots.Mass,error),error);
   for(unsigned int s=0;s<signals.size();s++){
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n",signals[s].Name.c_str(),GetEventInRange(125,2000,SignPlots[4*s].Mass,error),error);
   }
   fprintf(pFile,"\nIntegral in range [300,2000]GeV:\n");
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","D ",GetEventInRange(300,2000,DataPlots.Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","P ",GetEventInRange(300,2000,Pred_Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","M ",GetEventInRange(300,2000,MCTrPlots.Mass,error),error);
   for(unsigned int s=0;s<signals.size();s++){
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n",signals[s].Name.c_str(),GetEventInRange(300,2000,SignPlots[4*s].Mass,error),error);
   }
   fprintf(pFile,"\nIntegral in range [400,2000]GeV:\n");
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","D ",GetEventInRange(400,2000,DataPlots.Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","P ",GetEventInRange(400,2000,Pred_Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","M ",GetEventInRange(400,2000,MCTrPlots.Mass,error),error);
   for(unsigned int s=0;s<signals.size();s++){
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n",signals[s].Name.c_str(),GetEventInRange(400,2000,SignPlots[4*s].Mass,error),error);
   }
   fprintf(pFile,"\nIntegral in range [500,2000]GeV:\n");
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","D ",GetEventInRange(500,2000,DataPlots.Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","P ",GetEventInRange(500,2000,Pred_Mass,error),error);
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n","M ",GetEventInRange(500,2000,MCTrPlots.Mass,error),error);
   for(unsigned int s=0;s<signals.size();s++){
   fprintf(pFile,"%15s = %5.3E+-%5.3E\n",signals[s].Name.c_str(),GetEventInRange(500,2000,SignPlots[4*s].Mass,error),error);
   }
*/
   fprintf(pFile,"--------------------\n");
   fclose(pFile);

   //////////////////////////////////////////////////     CREATE EFFICIENCY FILE

   fflush(stdout);
}




void InitHistos(){
   //stPlots_Init(HistoFile, DataPlots,"Data", CutPt.size());
   //stPlots_Init(HistoFile, MCTrPlots,"MCTr", CutPt.size());
   for(unsigned int m=0;m<MCsample.size();m++){
      stPlots tmp;
      //stPlots_Init(HistoFile, tmp,MCsample[m].Name, CutPt.size());
      MCPlots.push_back(tmp);
   }

   for(unsigned int s=0;s<signals.size();s++){
   for(int NC=0;NC<4;NC++){
      stPlots tmp;
      if(NC==0){
         //stPlots_Init(HistoFile, tmp,signals[s].Name, CutPt.size());
      }else{
         char buffer[256];sprintf(buffer,"_NC%i",NC-1);
         //stPlots_Init(HistoFile, tmp,signals[s].Name + buffer,CutPt.size(), true);
      }
      SignPlots.push_back(tmp);
   }}
   HistoFile->cd();

   HCuts_Pt  = new TH1D("HCuts_Pt" ,"HCuts_Pt" ,CutPt.size(),0,CutPt.size());
   HCuts_I   = new TH1D("HCuts_I"  ,"HCuts_I"  ,CutPt.size(),0,CutPt.size());
   HCuts_TOF = new TH1D("HCuts_TOF","HCuts_TOF",CutPt.size(),0,CutPt.size());
   for(unsigned int i=0;i<CutPt.size();i++){  HCuts_Pt->Fill(i,CutPt[i]);     HCuts_I->Fill(i,CutI[i]);    HCuts_TOF->Fill(i,CutTOF[i]);   }

   H_A = new TH1D("H_A" ,"H_A" ,CutPt.size(),0,CutPt.size());
   H_B = new TH1D("H_B" ,"H_B" ,CutPt.size(),0,CutPt.size());
   H_C = new TH1D("H_C" ,"H_C" ,CutPt.size(),0,CutPt.size());
   H_D = new TH1D("H_D" ,"H_D" ,CutPt.size(),0,CutPt.size());
   H_E = new TH1D("H_E" ,"H_E" ,CutPt.size(),0,CutPt.size());
   H_F = new TH1D("H_F" ,"H_F" ,CutPt.size(),0,CutPt.size());
   H_G = new TH1D("H_G" ,"H_G" ,CutPt.size(),0,CutPt.size());
   H_H = new TH1D("H_H" ,"H_H" ,CutPt.size(),0,CutPt.size());
   H_P = new TH1D("H_P" ,"H_P" ,CutPt.size(),0,CutPt.size());

   CtrlPt_BckgIs   = new TH1D("CtrlPt_BckgIs" ,"CtrlPt_BckgIs" ,200,0,dEdxS_UpLim);  CtrlPt_BckgIs ->Sumw2();
   CtrlPt_BckgIm   = new TH1D("CtrlPt_BckgIm" ,"CtrlPt_BckgIm" ,200,0,dEdxM_UpLim);  CtrlPt_BckgIm ->Sumw2();
   CtrlPt_BckgTOF  = new TH1D("CtrlPt_BckgTOF","CtrlPt_BckgTOF",200,0,20);           CtrlPt_BckgTOF->Sumw2();
   CtrlPt_SignIs   = new TH1D("CtrlPt_SignIs" ,"CtrlPt_SignIs" ,200,0,dEdxS_UpLim);  CtrlPt_SignIs ->Sumw2();
   CtrlPt_SignIm   = new TH1D("CtrlPt_SignIm" ,"CtrlPt_SignIm" ,200,0,dEdxM_UpLim);  CtrlPt_SignIm ->Sumw2();
   CtrlPt_SignTOF  = new TH1D("CtrlPt_SignTOF","CtrlPt_SignTOF",200,0,20);           CtrlPt_SignTOF->Sumw2();

   CtrlIs_BckgPt   = new TH1D("CtrlIs_BckgPt" ,"CtrlIs_BckgPt" ,200,0,1500);         CtrlIs_BckgPt ->Sumw2();
   CtrlIs_BckgTOF  = new TH1D("CtrlIs_BckgTOF","CtrlIs_BckgTOF",200,0,20);           CtrlIs_BckgTOF->Sumw2();
   CtrlIs_SignPt   = new TH1D("CtrlIs_SignPt" ,"CtrlIs_SignPt" ,200,0,1500);         CtrlIs_SignPt ->Sumw2();
   CtrlIs_SignTOF  = new TH1D("CtrlIs_SignTOF","CtrlIs_SignTOF",200,0,20);           CtrlIs_SignTOF->Sumw2();

   CtrlTOF_BckgPt  = new TH1D("CtrlTOF_BckgPt","CtrlTOF_BckgPt",200,0,1500);         CtrlTOF_BckgPt->Sumw2();
   CtrlTOF_BckgIs  = new TH1D("CtrlTOF_BckgIs","CtrlTOF_BckgIs",200,0,dEdxS_UpLim);  CtrlTOF_BckgIs->Sumw2();
   CtrlTOF_SignPt  = new TH1D("CtrlTOF_SignPt","CtrlTOF_SignPt",200,0,1500);         CtrlTOF_SignPt->Sumw2();
   CtrlTOF_SignIs  = new TH1D("CtrlTOF_SignIs","CtrlTOF_SignIs",200,0,dEdxS_UpLim);  CtrlTOF_SignIs->Sumw2();


   char Name   [1024];


   sprintf(Name,"CutFinder_I");
   Data_I         = new TH1D(Name,Name, 200,0,dEdxS_UpLim);
   Data_I->Sumw2(); 

   sprintf(Name,"CutFinder_Pt");
   Data_Pt       = new TH1D(Name,Name,200,0,PtHistoUpperBound);
   Data_Pt->Sumw2();

   sprintf(Name,"CutFinder_TOF");
   Data_TOF       = new TH1D(Name,Name,200,-10,20);
   Data_TOF->Sumw2();



   sprintf(Name,"Pred_Mass");
   Pred_Mass = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),100,0,MassHistoUpperBound);
   Pred_Mass->Sumw2();

   sprintf(Name,"Pred_MassTOF");
   Pred_MassTOF = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(), 100,0,MassHistoUpperBound);
   Pred_MassTOF->Sumw2();

   sprintf(Name,"Pred_MassComb");
   Pred_MassComb = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(),100,0,MassHistoUpperBound);
   Pred_MassComb->Sumw2();

   sprintf(Name,"Pred_I");
   Pred_I  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(), 200,0,dEdxM_UpLim);
   Pred_I->Sumw2();

   sprintf(Name,"Pred_P");
   Pred_P  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(), 200,0,PtHistoUpperBound);
   Pred_P->Sumw2();

   sprintf(Name,"Pred_TOF");
   Pred_TOF  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(), 200,0,5);
   Pred_TOF->Sumw2();


   sprintf(Name,"DataD_I");
   DataD_I  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(), 200,0,dEdxM_UpLim);
   DataD_I->Sumw2();

   sprintf(Name,"DataD_P");
   DataD_P  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(), 200,0,PtHistoUpperBound);
   DataD_P->Sumw2();

   sprintf(Name,"DataD_TOF");
   DataD_TOF  = new TH2D(Name,Name,CutPt.size(),0,CutPt.size(), 200,0,5);
   DataD_TOF->Sumw2();
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

void SetWeight(const double& IntegratedLuminosityInPb, const double& CrossSection, const double& MCEvents){
   if(IntegratedLuminosityInPb>0){
      double NMCEvents = MCEvents;
      if(MaxEntry>0)NMCEvents=std::min(MCEvents,(double)MaxEntry);
      Event_Weight = (CrossSection * IntegratedLuminosityInPb) / NMCEvents;
   }else{
      Event_Weight=1;
   }
}

void SetWeightMC(const double& IntegratedLuminosityInPb, const double& SampleEquivalentLumi, const double& SampleSize, double MaxEvent){
   if(MaxEvent<0)MaxEvent=SampleSize;
   printf("SetWeight MC: IntLumi = %6.2E  SampleLumi = %6.2E --> EventWeight = %6.2E\n",IntegratedLuminosityInPb,SampleEquivalentLumi, IntegratedLuminosityInPb/SampleEquivalentLumi);
   printf("Sample NEvent = %6.2E   SampleEventUsed = %6.2E --> Weight Rescale = %6.2E\n",SampleSize, MaxEvent, SampleSize/MaxEvent);
   Event_Weight = (IntegratedLuminosityInPb/SampleEquivalentLumi) * (SampleSize/MaxEvent);
   printf("FinalWeight = %6.2f\n",Event_Weight);
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
