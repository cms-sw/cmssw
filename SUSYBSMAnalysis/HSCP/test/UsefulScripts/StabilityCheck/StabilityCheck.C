
#include <exception>
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
#include "TProfile.h"
#include "TPaveText.h"
#include "tdrstyle.C"


namespace reco    { class Vertex; class Track; class GenParticle; class DeDxData; class MuonTimeExtra;}
namespace susybsm { class HSCParticle;}
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
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"

#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPIsolation.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"

using namespace fwlite;
using namespace reco;
using namespace susybsm;
using namespace std;
using namespace edm;
using namespace trigger;


#include "../../ICHEP_Analysis/Analysis_PlotFunction.h"
#include "../../ICHEP_Analysis/Analysis_Samples.h"
#include "../../ICHEP_Analysis/Analysis_Global.h"


#endif


bool PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const reco::MuonTimeExtra* dttof, const reco::MuonTimeExtra* csctof, const fwlite::ChainEvent& ev);
bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, int NObjectAboveThreshold, bool averageThreshold=false);


bool PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const reco::MuonTimeExtra* dttof, const reco::MuonTimeExtra* csctof, const fwlite::ChainEvent& ev)
{
   if(TypeMode==1 && !(hscp.type() == HSCParticleType::trackerMuon || hscp.type() == HSCParticleType::globalMuon))return false;
   if(TypeMode==2 && hscp.type() != HSCParticleType::globalMuon)return false;
   reco::TrackRef   track = hscp.trackRef(); if(track.isNull())return false;

   if(fabs(track->eta())>GlobalMaxEta) return false;
   if(track->found()<GlobalMinNOH)return false;
   if(track->hitPattern().numberOfValidPixelHits()<2)return false; 
   if(dedxSObj.numberOfMeasurements()<GlobalMinNOM)return false;
//   if(tof && tof->nDof()<GlobalMinNDOF && (dttof->nDof()<GlobalMinNDOFDT || csctof->nDof()<GlobalMinNDOFCSC) )return false;

   if(track->qualityMask()<GlobalMinQual )return false;
   if(track->chi2()/track->ndof()>GlobalMaxChi2 )return false;
   if(track->pt()<GlobalMinPt)return false;
   if(dedxSObj.dEdx()<GlobalMinIs)return false;
   if(dedxMObj.dEdx()<GlobalMinIm)return false;
//   if(tof && tof->inverseBeta()<GlobalMinTOF)return false;
//   if(tof && tof->inverseBetaErr()>GlobalMaxTOFErr)return false;

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
   if(v3d>GlobalMaxV3D )return false;

   fwlite::Handle<HSCPIsolationValueMap> IsolationH;
   IsolationH.getByLabel(ev, "HSCPIsolation03");
   if(!IsolationH.isValid()){printf("Invalid IsolationH\n");return false;}
   const ValueMap<HSCPIsolation>& IsolationMap = *IsolationH.product();

   HSCPIsolation hscpIso = IsolationMap.get((size_t)track.key());
    if(hscpIso.Get_TK_SumEt()>GlobalMaxTIsol)return false;

   double EoP = (hscpIso.Get_ECAL_Energy() + hscpIso.Get_HCAL_Energy())/track->p();
   if(EoP>GlobalMaxEIsol)return false;

   if((track->ptError()/track->pt())>GlobalMaxPterr)return false;
   if(std::max(0.0,track->pt() - track->ptError())<GlobalMinPt)return false;
   return true;
}


bool PassingTrigger(const fwlite::ChainEvent& ev, const std::string& TriggerName){
      edm::TriggerResultsByName tr = ev.triggerResultsByName("Merge");
      if(!tr.isValid())return false;

      bool Accept = false;
      if(TriggerName=="Any"){
         Accept = true;
      }else{
         Accept = tr.accept(tr.triggerIndex(TriggerName.c_str()));
      }

   return Accept;
}




void StabilityCheck()
{
   Event_Weight = 1;
   MaxEntry = -1;


   system("mkdir pictures");

   setTDRStyle();
   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.15);
   gStyle->SetPadRightMargin (0.03);
   gStyle->SetPadLeftMargin  (0.07);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetNdivisions(505,"X");
   TH1::AddDirectory(kTRUE);

   std::map<unsigned int, unsigned int> RunBinIndex;
   unsigned int NextIndex=0;

   vector<string> DataFileName;
   GetInputFiles(DataFileName, "Data");
//   DataFileName.push_back(" /storage/data/cms/users/quertenmont/HSCP/CMSSW_3_8_6/10_01_11/Data_135821_141887.root");
//   DataFileName.push_back(" /storage/data/cms/users/quertenmont/HSCP/CMSSW_3_8_6/10_01_11/Data_141888_144114.root");
//   DataFileName.push_back(" /storage/data/cms/users/quertenmont/HSCP/CMSSW_3_8_6/10_01_11/Data_146240_148000.root");
//   DataFileName.push_back(" /storage/data/cms/users/quertenmont/HSCP/CMSSW_3_8_6/10_01_11/Data_148001_149711.root");


   std::vector<string> triggers;
   triggers.push_back("Any");
//   triggers.push_back("HscpPathMu");
//   triggers.push_back("HscpPathMet");

   triggers.push_back("HscpPathSingleMu");
//   triggers.push_back("HscpPathDoubleMu");
   triggers.push_back("HscpPathPFMet");
//   triggers.push_back("HscpPathCaloMet");


/*   triggers.push_back("HLT_MET100");
   triggers.push_back("HLT_Jet140U");
   triggers.push_back("HLT_DiJetAve140U");
   triggers.push_back("HLT_QuadJet25U");
   triggers.push_back("HLT_QuadJet30U");
   triggers.push_back("HLT_QuadJet35U");
   triggers.push_back("HLT_Mu15");
   triggers.push_back("HLT_DoubleMu3");
*/
   TProfile** dEdxProf = new TProfile*[triggers.size()];
   TProfile** dEdxMProf = new TProfile*[triggers.size()];
   TProfile** PtProf   = new TProfile*[triggers.size()];
   TProfile** TOFProf   = new TProfile*[triggers.size()];
   TProfile** TOFDTProf   = new TProfile*[triggers.size()];
   TProfile** TOFCSCProf   = new TProfile*[triggers.size()];
   TH1D**     Count    = new TH1D*    [triggers.size()];
   TH1D**     CountMu  = new TH1D*    [triggers.size()];
   TH1D**     HdEdx    = new TH1D*    [triggers.size()];
   TH1D**     HPt      = new TH1D*    [triggers.size()];
   TH1D**     HTOF      = new TH1D*    [triggers.size()];



   system("mkdir pictures/");
   TFile* OutputHisto = new TFile((string("pictures/") + "/Histos.root").c_str(),"RECREATE");
   for(unsigned int i=0;i<triggers.size();i++){
      dEdxProf[i] = new TProfile((triggers[i] + "dEdxProf").c_str(), "dEdxProf", 1000 ,0, 1000);
      dEdxMProf[i] = new TProfile((triggers[i] + "dEdxMProf").c_str(), "dEdxMProf", 1000 ,0, 1000);
      PtProf  [i] = new TProfile((triggers[i] + "PtProf"  ).c_str(), "PtProf"  , 1000 ,0, 1000);
      TOFProf  [i] = new TProfile((triggers[i] + "TOFProf"  ).c_str(), "TOFProf"  , 1000 ,0, 1000);
      TOFDTProf  [i] = new TProfile((triggers[i] + "TOFDTProf"  ).c_str(), "TOFDTProf"  , 1000 ,0, 1000);
      TOFCSCProf  [i] = new TProfile((triggers[i] + "TOFCSCProf"  ).c_str(), "TOFCSCProf"  , 1000 ,0, 1000);

      Count   [i] = new TH1D(    (triggers[i] + "Count"   ).c_str(), "Count"   , 1000 ,0, 1000);  Count  [i]->Sumw2();
      CountMu [i] = new TH1D(    (triggers[i] + "CountMu" ).c_str(), "CountMu" , 1000 ,0, 1000);  CountMu[i]->Sumw2();
      HdEdx   [i] = new TH1D(    (triggers[i] + "HdEdx"   ).c_str(), "HdEdx"   , 1000 ,0, 1000);  HdEdx  [i]->Sumw2();
      HPt     [i] = new TH1D(    (triggers[i] + "HPt"     ).c_str(), "HPt"     , 1000 ,0, 1000);  HPt    [i]->Sumw2();
      HTOF     [i] = new TH1D(    (triggers[i] + "HTOF"     ).c_str(), "HTOF"     , 1000 ,0, 1000);  HTOF    [i]->Sumw2();
   }

   TypeMode      = 0;

   fwlite::ChainEvent tree(DataFileName);
   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on Tree              :");
   int TreeStep = tree.size()/50;if(TreeStep==0)TreeStep=1;
   for(Long64_t e=0;e<tree.size();e++){
//      if(e>100000)break;
      tree.to(e); 
      if(e%TreeStep==0){printf(".");fflush(stdout);}
//      if(!PassTrigger(tree))continue;

      if(RunBinIndex.find(tree.eventAuxiliary().run()/10) == RunBinIndex.end()){
         RunBinIndex[tree.eventAuxiliary().run()/10] = NextIndex;
         for(unsigned int i=0;i<triggers.size();i++){
            int Bin = HdEdx[i]->GetXaxis()->FindBin(NextIndex);
            char Label[2048]; sprintf(Label,"%5iX",tree.eventAuxiliary().run()/10);
            HdEdx[i]->GetXaxis()->SetBinLabel(Bin, Label);
            HPt[i]->GetXaxis()->SetBinLabel(Bin, Label);
            HTOF[i]->GetXaxis()->SetBinLabel(Bin, Label);
            Count[i]->GetXaxis()->SetBinLabel(Bin, Label);
            dEdxProf[i]->GetXaxis()->SetBinLabel(Bin, Label);      
            dEdxMProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            PtProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            TOFProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            TOFDTProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            TOFCSCProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
         }
         NextIndex++;
      }
      unsigned int CurrentRunIndex = RunBinIndex[tree.eventAuxiliary().run()/10];
        
      fwlite::Handle<susybsm::HSCParticleCollection> hscpCollHandle;
      hscpCollHandle.getByLabel(tree,"HSCParticleProducer");
      if(!hscpCollHandle.isValid()){printf("HSCP Collection NotFound\n");continue;}
      susybsm::HSCParticleCollection hscpColl = *hscpCollHandle;

      fwlite::Handle<DeDxDataValueMap> dEdxSCollH;
      dEdxSCollH.getByLabel(tree, dEdxS_Label.c_str());
      if(!dEdxSCollH.isValid()){printf("Invalid dEdx Selection collection\n");continue;}

      fwlite::Handle<DeDxDataValueMap> dEdxMCollH;
      dEdxMCollH.getByLabel(tree, dEdxM_Label.c_str());
      if(!dEdxMCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}

      fwlite::Handle<MuonTimeExtraMap> TOFCollH;
      TOFCollH.getByLabel(tree, "muontiming",TOF_Label.c_str());
      if(!TOFCollH.isValid()){printf("Invalid TOF collection\n");return;}


      fwlite::Handle<MuonTimeExtraMap> TOFDTCollH;
      TOFDTCollH.getByLabel(tree, "muontiming",TOFdt_Label.c_str());
      if(!TOFDTCollH.isValid()){printf("Invalid DT TOF collection\n");continue;}

      fwlite::Handle<MuonTimeExtraMap> TOFCSCCollH;
      TOFCSCCollH.getByLabel(tree, "muontiming",TOFcsc_Label.c_str());
      if(!TOFCSCCollH.isValid()){printf("Invalid CSCTOF collection\n");continue;}

      for(unsigned int c=0;c<hscpColl.size();c++){
         susybsm::HSCParticle hscp  = hscpColl[c];
         reco::TrackRef track = hscp.trackRef();
         if(track.isNull())continue;

         const DeDxData& dedxSObj  = dEdxSCollH->get(track.key());
         const DeDxData& dedxMObj  = dEdxMCollH->get(track.key());
         const reco::MuonTimeExtra* tof = NULL;
         const reco::MuonTimeExtra* dttof = NULL;
         const reco::MuonTimeExtra* csctof = NULL;
         if(!hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); dttof  = &TOFDTCollH->get(hscp.muonRef().key()); csctof  = &TOFCSCCollH->get(hscp.muonRef().key());}


         if(!PassPreselection(hscp, dedxSObj, dedxMObj, tof, dttof, csctof, tree))continue;

         for(unsigned int i=0;i<triggers.size();i++){
            if(!PassingTrigger(tree,triggers[i]))continue;

            if(tof && tof->nDof()>=GlobalMinNDOF && (dttof->nDof()>=GlobalMinNDOFDT || csctof->nDof()>=GlobalMinNDOFCSC) && tof->inverseBetaErr()<=GlobalMaxTOFErr){
               if(tof->inverseBeta()>=GlobalMinTOF)CountMu[i]->Fill(CurrentRunIndex);
               if(tof->inverseBeta()>=GlobalMinTOF)TOFProf[i]->Fill(CurrentRunIndex, tof->inverseBeta());
               if(dttof->inverseBeta()>=GlobalMinTOF)TOFDTProf[i]->Fill(CurrentRunIndex, dttof->inverseBeta());
               if(csctof->inverseBeta()>=GlobalMinTOF)TOFCSCProf[i]->Fill(CurrentRunIndex, csctof->inverseBeta());
               if(tof->inverseBeta() > 1.1 ) HTOF[i]->Fill(CurrentRunIndex);            
            }

            if(hscp.trackRef()->pt() > 30 ) HPt[i]->Fill(CurrentRunIndex);
            if(dedxSObj.dEdx() > 0.10 ) HdEdx[i]->Fill(CurrentRunIndex);
            Count[i]->Fill(CurrentRunIndex);

            dEdxProf[i]->Fill(CurrentRunIndex, dedxSObj.dEdx());
            dEdxMProf[i]->Fill(CurrentRunIndex, dedxMObj.dEdx());
            PtProf[i]->Fill(CurrentRunIndex, hscp.trackRef()->pt());
         }

      }
   }printf("\n");

   TCanvas* c1;
   TLegend* leg;

   for(unsigned int i=0;i<triggers.size();i++){
   c1 = new TCanvas("c1","c1",600,600);
   HdEdx[i]->Divide(Count[i]);
   HdEdx[i]->LabelsDeflate("X");
   HdEdx[i]->LabelsOption("av","X");
   HdEdx[i]->GetXaxis()->SetNdivisions(505);
   HdEdx[i]->SetTitle("");
   HdEdx[i]->SetStats(kFALSE);
   HdEdx[i]->GetXaxis()->SetTitle("");
   HdEdx[i]->GetYaxis()->SetTitle("Ratio over Threshold");
   HdEdx[i]->GetYaxis()->SetTitleOffset(0.9);
   HdEdx[i]->GetXaxis()->SetLabelSize(0.04);
   HdEdx[i]->SetLineColor(Color[0]);
   HdEdx[i]->SetFillColor(Color[0]);
   HdEdx[i]->SetMarkerSize(0.4);
   HdEdx[i]->SetMarkerStyle(Marker[0]);
   HdEdx[i]->SetMarkerColor(Color[0]);
   HdEdx[i]->Draw("E1");

   leg = new TLegend(0.55,0.86,0.79,0.93,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetFillColor(0);
   leg->AddEntry(HdEdx[i],"I_{as} > 0.10","P");
   leg->Draw();

   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"ROT_Is");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   HPt[i]->Divide(Count[i]);
   HPt[i]->LabelsDeflate("X");
   HPt[i]->LabelsOption("av","X");
   HPt[i]->GetXaxis()->SetNdivisions(505);
   HPt[i]->SetTitle("");
   HPt[i]->SetStats(kFALSE);
   HPt[i]->GetXaxis()->SetTitle("");
   HPt[i]->GetYaxis()->SetTitle("Ratio over Threshold");
   HPt[i]->GetYaxis()->SetTitleOffset(0.9);
   HPt[i]->GetXaxis()->SetLabelSize(0.04);
   HPt[i]->SetLineColor(Color[0]);
   HPt[i]->SetFillColor(Color[0]);
   HPt[i]->SetMarkerSize(0.4);
   HPt[i]->SetMarkerStyle(Marker[0]);
   HPt[i]->SetMarkerColor(Color[0]);
   HPt[i]->Draw("E1");

   leg = new TLegend(0.55,0.86,0.79,0.93,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetFillColor(0);
   leg->AddEntry(HPt[i],"p_{T} > 30 GeV/c","P");
   leg->Draw();
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"ROT_Pt");
   delete c1;



   c1 = new TCanvas("c1","c1",600,600);
   HTOF[i]->Divide(CountMu[i]);
   HTOF[i]->LabelsDeflate("X");
   HTOF[i]->LabelsOption("av","X");
   HTOF[i]->GetXaxis()->SetNdivisions(505);
   HTOF[i]->SetTitle("");
   HTOF[i]->SetStats(kFALSE);
   HTOF[i]->GetXaxis()->SetTitle("");
   HTOF[i]->GetYaxis()->SetTitle("Ratio over Threshold");
   HTOF[i]->GetYaxis()->SetTitleOffset(0.9);
   HTOF[i]->GetXaxis()->SetLabelSize(0.04);
   HTOF[i]->SetLineColor(Color[0]);
   HTOF[i]->SetFillColor(Color[0]);
   HTOF[i]->SetMarkerSize(0.4);
   HTOF[i]->SetMarkerStyle(Marker[0]);
   HTOF[i]->SetMarkerColor(Color[0]);
   HTOF[i]->Draw("E1");

   leg = new TLegend(0.55,0.86,0.79,0.93,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetFillColor(0);
   leg->AddEntry(HTOF[i],"1/#beta > 1.1","P");
   leg->Draw();
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"ROT_TOF");
   delete c1;


   c1 = new TCanvas("c1","c1",600,600);
   c1->SetLogy(true);
   Count[i]->LabelsDeflate("X");
   Count[i]->LabelsOption("av","X");
   Count[i]->GetXaxis()->SetNdivisions(505);
   Count[i]->SetTitle("");
   Count[i]->SetStats(kFALSE);
   Count[i]->GetXaxis()->SetTitle("");
   Count[i]->GetYaxis()->SetTitle("#Tracks");
   Count[i]->GetYaxis()->SetTitleOffset(0.9);
   Count[i]->GetXaxis()->SetLabelSize(0.04);
   Count[i]->SetLineColor(Color[0]);
   Count[i]->SetFillColor(Color[0]);
   Count[i]->SetMarkerSize(0.4);
   Count[i]->SetMarkerStyle(Marker[0]);
   Count[i]->SetMarkerColor(Color[0]);
   Count[i]->Draw("E1");

   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Count");
   delete c1;



   c1 = new TCanvas("c1","c1",600,600);
   dEdxProf[i]->LabelsDeflate("X");
   dEdxProf[i]->LabelsOption("av","X");
   dEdxProf[i]->GetXaxis()->SetNdivisions(505);
   dEdxProf[i]->SetTitle("");
   dEdxProf[i]->SetStats(kFALSE);
   dEdxProf[i]->GetXaxis()->SetTitle("");
   dEdxProf[i]->GetYaxis()->SetTitle("dE/dx discriminator");
   dEdxProf[i]->GetYaxis()->SetTitleOffset(0.9);
   dEdxProf[i]->GetXaxis()->SetLabelSize(0.04);
   dEdxProf[i]->SetLineColor(Color[0]);
   dEdxProf[i]->SetFillColor(Color[0]);
   dEdxProf[i]->SetMarkerSize(0.4);
   dEdxProf[i]->SetMarkerStyle(Marker[0]);
   dEdxProf[i]->SetMarkerColor(Color[0]);
   dEdxProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_Is");
   delete c1;



   c1 = new TCanvas("c1","c1",600,600);
   dEdxMProf[i]->LabelsDeflate("X");
   dEdxMProf[i]->LabelsOption("av","X");
   dEdxMProf[i]->GetXaxis()->SetNdivisions(505);
   dEdxMProf[i]->SetTitle("");
   dEdxMProf[i]->SetStats(kFALSE);
   dEdxMProf[i]->GetXaxis()->SetTitle("");
   dEdxMProf[i]->GetYaxis()->SetTitle("dE/dx discriminator");
   dEdxMProf[i]->GetYaxis()->SetTitleOffset(0.9);
   dEdxMProf[i]->GetXaxis()->SetLabelSize(0.04);
   dEdxMProf[i]->SetLineColor(Color[0]);
   dEdxMProf[i]->SetFillColor(Color[0]);
   dEdxMProf[i]->SetMarkerSize(0.4);
   dEdxMProf[i]->SetMarkerStyle(Marker[0]);
   dEdxMProf[i]->SetMarkerColor(Color[0]);
   dEdxMProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_Im");
   delete c1;



   c1 = new TCanvas("c1","c1",600,600);
   PtProf[i]->LabelsDeflate("X");
   PtProf[i]->LabelsOption("av","X");
   PtProf[i]->GetXaxis()->SetNdivisions(505);
   PtProf[i]->SetTitle("");
   PtProf[i]->SetStats(kFALSE);
   PtProf[i]->GetXaxis()->SetTitle("");
   PtProf[i]->GetYaxis()->SetTitle("p_{T} (GeV/c)");
   PtProf[i]->GetYaxis()->SetTitleOffset(0.9);
   PtProf[i]->GetXaxis()->SetLabelSize(0.04);
   PtProf[i]->SetLineColor(Color[0]);
   PtProf[i]->SetFillColor(Color[0]);
   PtProf[i]->SetMarkerSize(0.4);
   PtProf[i]->SetMarkerStyle(Marker[0]);
   PtProf[i]->SetMarkerColor(Color[0]);
   PtProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_Pt");
   delete c1;


   c1 = new TCanvas("c1","c1",600,600);
   TOFProf[i]->LabelsDeflate("X");
   TOFProf[i]->LabelsOption("av","X");
   TOFProf[i]->GetXaxis()->SetNdivisions(505);
   TOFProf[i]->SetTitle("");
   TOFProf[i]->SetStats(kFALSE);
   TOFProf[i]->GetXaxis()->SetTitle("");
   TOFProf[i]->GetYaxis()->SetTitle("1/#beta");
   TOFProf[i]->GetYaxis()->SetTitleOffset(0.9);
   TOFProf[i]->GetXaxis()->SetLabelSize(0.04);
   TOFProf[i]->SetLineColor(Color[0]);
   TOFProf[i]->SetFillColor(Color[0]);
   TOFProf[i]->SetMarkerSize(0.4);
   TOFProf[i]->SetMarkerStyle(Marker[0]);
   TOFProf[i]->SetMarkerColor(Color[0]);
   TOFProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_TOF");
   delete c1;


   c1 = new TCanvas("c1","c1",600,600);
   TOFDTProf[i]->LabelsDeflate("X");
   TOFDTProf[i]->LabelsOption("av","X");
   TOFDTProf[i]->GetXaxis()->SetNdivisions(505);
   TOFDTProf[i]->SetTitle("");
   TOFDTProf[i]->SetStats(kFALSE);
   TOFDTProf[i]->GetXaxis()->SetTitle("");
   TOFDTProf[i]->GetYaxis()->SetTitle("1/#beta");
   TOFDTProf[i]->GetYaxis()->SetTitleOffset(0.9);
   TOFDTProf[i]->GetXaxis()->SetLabelSize(0.04);
   TOFDTProf[i]->SetLineColor(Color[0]);
   TOFDTProf[i]->SetFillColor(Color[0]);
   TOFDTProf[i]->SetMarkerSize(0.4);
   TOFDTProf[i]->SetMarkerStyle(Marker[0]);
   TOFDTProf[i]->SetMarkerColor(Color[0]);
   TOFDTProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_TOFDT");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   TOFCSCProf[i]->LabelsDeflate("X");
   TOFCSCProf[i]->LabelsOption("av","X");
   TOFCSCProf[i]->GetXaxis()->SetNdivisions(505);
   TOFCSCProf[i]->SetTitle("");
   TOFCSCProf[i]->SetStats(kFALSE);
   TOFCSCProf[i]->GetXaxis()->SetTitle("");
   TOFCSCProf[i]->GetYaxis()->SetTitle("1/#beta");
   TOFCSCProf[i]->GetYaxis()->SetTitleOffset(0.9);
   TOFCSCProf[i]->GetXaxis()->SetLabelSize(0.04);
   TOFCSCProf[i]->SetLineColor(Color[0]);
   TOFCSCProf[i]->SetFillColor(Color[0]);
   TOFCSCProf[i]->SetMarkerSize(0.4);
   TOFCSCProf[i]->SetMarkerStyle(Marker[0]);
   TOFCSCProf[i]->SetMarkerColor(Color[0]);
   TOFCSCProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_TOFCSC");
   delete c1;


   }

   OutputHisto->Write();
   OutputHisto->Close();  
}



bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, int NObjectAboveThreshold, bool averageThreshold)
{
   unsigned int filterIndex = trEv.filterIndex(InputPath);
   //if(filterIndex<trEv.sizeFilters())printf("SELECTED INDEX =%i --> %s    XXX   %s\n",filterIndex,trEv.filterTag(filterIndex).label().c_str(), trEv.filterTag(filterIndex).process().c_str());
         
   if (filterIndex<trEv.sizeFilters()){
      const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
      const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
      const size_type nI(VIDS.size());
      const size_type nK(KEYS.size());
      assert(nI==nK);
      const size_type n(max(nI,nK));
      const trigger::TriggerObjectCollection& TOC(trEv.getObjects());


      if(!averageThreshold){
         int NObjectAboveThresholdObserved = 0;
         for (size_type i=0; i!=n; ++i) {
            const TriggerObject& TO(TOC[KEYS[i]]);
            if(TO.pt()> NewThreshold) NObjectAboveThresholdObserved++;
   	    //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "<< TO.id() << " " << TO.pt() << " " << TO.eta() << " " << TO.phi() << " " << TO.mass()<< endl;
         }          
         if(NObjectAboveThresholdObserved>=NObjectAboveThreshold)return true;

      }else{
         std::vector<double> ObjPt;

         for (size_type i=0; i!=n; ++i) {
            const TriggerObject& TO(TOC[KEYS[i]]);
            ObjPt.push_back(TO.pt());
            //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "<< TO.id() << " " << TO.pt() << " " << TO.eta() << " " << TO.phi() << " " << TO.mass()<< endl;
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


