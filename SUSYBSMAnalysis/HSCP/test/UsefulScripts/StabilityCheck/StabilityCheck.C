#include <exception>
#include <vector>

#include "TROOT.h"
#include "TFile.h"
#include "TDCacheFile.h"
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
#include "TCanvas.h"
#include "TProfile.h"
#include "TPaveText.h"


namespace reco { class Vertex; class Track; class GenParticle; class DeDxData; class MuonTimeExtra;}
namespace susybsm { class HSCParticle; class HSCPIsolation;}
namespace fwlite { class ChainEvent;}
namespace trigger { class TriggerEvent;}
namespace edm {class TriggerResults; class TriggerResultsByName; class InputTag; class LumiReWeighting;}
namespace reweight{class PoissonMeanShifter;}

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

#include "../../ICHEP_Analysis/Analysis_Global.h"
#include "../../ICHEP_Analysis/Analysis_PlotFunction.h"
#include "../../ICHEP_Analysis/Analysis_Samples.h"

#endif


bool PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const reco::MuonTimeExtra* dttof, const reco::MuonTimeExtra* csctof, const fwlite::ChainEvent& ev);
bool IncreasedTreshold(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double NewThreshold, int NObjectAboveThreshold, bool averageThreshold=false);


bool PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const reco::MuonTimeExtra* dttof, const reco::MuonTimeExtra* csctof, const fwlite::ChainEvent& ev)
{
   if(TypeMode==1 && !(hscp.type() == HSCParticleType::trackerMuon || hscp.type() == HSCParticleType::globalMuon))return false;
   if(TypeMode==2 && hscp.type() != HSCParticleType::globalMuon)return false;
   reco::TrackRef   track = hscp.trackRef(); if(track.isNull())return false;
   GlobalMaxEta=2.1;
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
      edm::TriggerResultsByName tr = ev.triggerResultsByName("MergeHLT");
      if(!tr.isValid())return false;

      //   for(unsigned int i=0;i<tr.size();i++){
      //printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
      //}fflush(stdout);


      bool Accept = false;
      if(TriggerName=="Any"){
         Accept = true;
      }else{
         Accept = tr.accept(tr.triggerIndex(TriggerName.c_str()));
      }

   return Accept;
}




void StabilityCheck(string MODE="COMPILE")
{
  if(MODE=="COMPILE") return;

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

   std::vector<string> triggers;
   triggers.push_back("Any");
//   triggers.push_back("HscpPathMu");
//   triggers.push_back("HscpPathMet");

   triggers.push_back("HSCPHLTTriggerMuFilter");
   triggers.push_back("HSCPHLTTriggerMetDeDxFilter");
   triggers.push_back("HSCPHLTTriggerL2MuFilter");
   triggers.push_back("HSCPHLTTriggerHtDeDxFilter");
   triggers.push_back("HSCPHLTTriggerMuDeDxFilter");
//   triggers.push_back("HscpPathDoubleMu");
//   triggers.push_back("HscpPathCaloMet");



   TProfile** NVertProf = new TProfile*[triggers.size()];
   TProfile** dEdxProf = new TProfile*[triggers.size()];
   TProfile** dEdxMProf = new TProfile*[triggers.size()];
   TProfile** dEdxMSProf = new TProfile*[triggers.size()];
   TProfile** dEdxMPProf = new TProfile*[triggers.size()];
   TProfile** dEdxMSCProf = new TProfile*[triggers.size()];
   TProfile** dEdxMPCProf = new TProfile*[triggers.size()];
   TProfile** dEdxMSFProf = new TProfile*[triggers.size()];
   TProfile** dEdxMPFProf = new TProfile*[triggers.size()];
   TProfile** PtProf   = new TProfile*[triggers.size()];
   TProfile** TOFProf   = new TProfile*[triggers.size()];
   TProfile** TOFDTProf   = new TProfile*[triggers.size()];
   TProfile** TOFCSCProf   = new TProfile*[triggers.size()];
   TProfile** TOFOverMinProf   = new TProfile*[triggers.size()];
   TProfile** TOFDTOverMinProf   = new TProfile*[triggers.size()];
   TProfile** TOFCSCOverMinProf   = new TProfile*[triggers.size()];
   TProfile** VertexProf   = new TProfile*[triggers.size()];
   TProfile** VertexDTProf   = new TProfile*[triggers.size()];
   TProfile** VertexCSCProf   = new TProfile*[triggers.size()];
   TH1D**     Count    = new TH1D*    [triggers.size()];
   TH1D**     CountMu  = new TH1D*    [triggers.size()];
   TH1D**     HdEdx    = new TH1D*    [triggers.size()];
   TH1D**     HPt      = new TH1D*    [triggers.size()];
   TH1D**     HTOF      = new TH1D*    [triggers.size()];



   system("mkdir pictures/");
   TFile* OutputHisto = new TFile((string("pictures/") + "/Histos.root").c_str(),"RECREATE");
   for(unsigned int i=0;i<triggers.size();i++){
      NVertProf[i] = new TProfile((triggers[i] + "NVertProf").c_str(), "NVertProf", 10000 ,0, 10000);
      dEdxProf[i] = new TProfile((triggers[i] + "dEdxProf").c_str(), "dEdxProf", 10000 ,0, 10000);
      dEdxMProf[i] = new TProfile((triggers[i] + "dEdxMProf").c_str(), "dEdxMProf", 10000 ,0, 10000);
      dEdxMSProf[i] = new TProfile((triggers[i] + "dEdxMSProf").c_str(), "dEdxMSProf", 10000 ,0, 10000);
      dEdxMPProf[i] = new TProfile((triggers[i] + "dEdxMPProf").c_str(), "dEdxMPProf", 10000 ,0, 10000);
      dEdxMSCProf[i] = new TProfile((triggers[i] + "dEdxMSCProf").c_str(), "dEdxMSCProf", 10000 ,0, 10000);
      dEdxMPCProf[i] = new TProfile((triggers[i] + "dEdxMPCProf").c_str(), "dEdxMPCProf", 10000 ,0, 10000);
      dEdxMSFProf[i] = new TProfile((triggers[i] + "dEdxMSFProf").c_str(), "dEdxMSFProf", 10000 ,0, 10000);
      dEdxMPFProf[i] = new TProfile((triggers[i] + "dEdxMPFProf").c_str(), "dEdxMPFProf", 10000 ,0, 10000);

      PtProf  [i] = new TProfile((triggers[i] + "PtProf"  ).c_str(), "PtProf"  , 10000 ,0, 10000);
      TOFProf  [i] = new TProfile((triggers[i] + "TOFProf"  ).c_str(), "TOFProf"  , 10000 ,0, 10000);
      TOFDTProf  [i] = new TProfile((triggers[i] + "TOFDTProf"  ).c_str(), "TOFDTProf"  , 10000 ,0, 10000);
      TOFCSCProf  [i] = new TProfile((triggers[i] + "TOFCSCProf"  ).c_str(), "TOFCSCProf"  , 10000 ,0, 10000);

      TOFOverMinProf  [i] = new TProfile((triggers[i] + "TOFOverMinProf"  ).c_str(), "TOFOverMinProf"  , 10000 ,0, 10000);
      TOFDTOverMinProf  [i] = new TProfile((triggers[i] + "TOFDTOverMinProf"  ).c_str(), "TOFDTOverMinProf"  , 10000 ,0, 10000);
      TOFCSCOverMinProf  [i] = new TProfile((triggers[i] + "TOFCSCOverMinProf"  ).c_str(), "TOFCSCOverMinProf"  , 10000 ,0, 10000);

      VertexProf  [i] = new TProfile((triggers[i] + "VertexProf"  ).c_str(), "VertexProf"  , 10000 ,0, 10000);
      VertexDTProf  [i] = new TProfile((triggers[i] + "VertexDTProf"  ).c_str(), "VertexDTProf"  , 10000 ,0, 10000);
      VertexCSCProf  [i] = new TProfile((triggers[i] + "VertexCSCProf"  ).c_str(), "VertexCSCProf"  , 10000 ,0, 10000);

      Count   [i] = new TH1D(    (triggers[i] + "Count"   ).c_str(), "Count"   , 10000 ,0, 10000);  Count  [i]->Sumw2();
      CountMu [i] = new TH1D(    (triggers[i] + "CountMu" ).c_str(), "CountMu" , 10000 ,0, 10000);  CountMu[i]->Sumw2();
      HdEdx   [i] = new TH1D(    (triggers[i] + "HdEdx"   ).c_str(), "HdEdx"   , 10000 ,0, 10000);  HdEdx  [i]->Sumw2();
      HPt     [i] = new TH1D(    (triggers[i] + "HPt"     ).c_str(), "HPt"     , 10000 ,0, 10000);  HPt    [i]->Sumw2();
      HTOF     [i] = new TH1D(    (triggers[i] + "HTOF"     ).c_str(), "HTOF"     , 10000 ,0, 10000);  HTOF    [i]->Sumw2();
   }

   TypeMode      = 0;

   std::vector<stSample> samples;
   // get all the samples and clean the list to keep only the one we want to run on... Also initialize the BaseDirectory
   InitBaseDirectory();
   GetSampleDefinition(samples, "../../ICHEP_Analysis/Analysis_Samples.txt");

#ifdef ANALYSIS2011
   keepOnlySamplesOfNameX(samples,"Data7TeV");
#else
   keepOnlySamplesOfNameX(samples,"Data8TeV");
#endif

   printf("----------------------------------------------------------------------------------------------------------------------------------------------------\n");
   printf("Run on the following samples:\n");
   for(unsigned int s=0;s<samples.size();s++){samples[s].print();}
   printf("----------------------------------------------------------------------------------------------------------------------------------------------------\n\n");

   for(unsigned int s=0;s<samples.size();s++){
     std::vector<string> FileName;
     GetInputFiles(samples[s], BaseDirectory, FileName);
     fwlite::ChainEvent tree(FileName);

   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on Tree              :");
   int TreeStep = tree.size()/50;if(TreeStep==0)TreeStep=1;
   for(Long64_t e=0;e<tree.size();e++){
//      if(e>10)break;
      tree.to(e); 
      if(e%TreeStep==0){printf(".");fflush(stdout);}
//      if(!PassTrigger(tree))continue;

      if(RunBinIndex.find(tree.eventAuxiliary().run()) == RunBinIndex.end()){
         RunBinIndex[tree.eventAuxiliary().run()] = NextIndex;
         for(unsigned int i=0;i<triggers.size();i++){
            int Bin = HdEdx[i]->GetXaxis()->FindBin(NextIndex);
            char Label[2048]; sprintf(Label,"%6i",tree.eventAuxiliary().run());
            HdEdx[i]->GetXaxis()->SetBinLabel(Bin, Label);
            HPt[i]->GetXaxis()->SetBinLabel(Bin, Label);
            HTOF[i]->GetXaxis()->SetBinLabel(Bin, Label);
            Count[i]->GetXaxis()->SetBinLabel(Bin, Label);
            NVertProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            dEdxProf[i]->GetXaxis()->SetBinLabel(Bin, Label);      
            dEdxMProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            dEdxMSProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            dEdxMPProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            dEdxMSCProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            dEdxMPCProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            dEdxMSFProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            dEdxMPFProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            PtProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            TOFProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            TOFDTProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            TOFCSCProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            TOFOverMinProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            TOFDTOverMinProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            TOFCSCOverMinProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            VertexProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            VertexDTProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
            VertexCSCProf[i]->GetXaxis()->SetBinLabel(Bin, Label);
         }
         NextIndex++;
      }

      unsigned int CurrentRunIndex = RunBinIndex[tree.eventAuxiliary().run()];

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

      fwlite::Handle<DeDxDataValueMap> dEdxMSCollH;
      dEdxMSCollH.getByLabel(tree, "dedxNPHarm2");
      if(!dEdxMSCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}

      fwlite::Handle<DeDxDataValueMap> dEdxMPCollH;
      dEdxMPCollH.getByLabel(tree, "dedxNSHarm2");
      if(!dEdxMPCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}


      fwlite::Handle<MuonTimeExtraMap> TOFCollH;
      TOFCollH.getByLabel(tree, "muontiming",TOF_Label.c_str());
      if(!TOFCollH.isValid()){printf("Invalid TOF collection\n");return;}


      fwlite::Handle<MuonTimeExtraMap> TOFDTCollH;
      TOFDTCollH.getByLabel(tree, "muontiming",TOFdt_Label.c_str());
      if(!TOFDTCollH.isValid()){printf("Invalid DT TOF collection\n");continue;}

      fwlite::Handle<MuonTimeExtraMap> TOFCSCCollH;
      TOFCSCCollH.getByLabel(tree, "muontiming",TOFcsc_Label.c_str());
      if(!TOFCSCCollH.isValid()){printf("Invalid CSCTOF collection\n");continue;}

      fwlite::Handle< std::vector<reco::Vertex> > vertexCollHandle;
      vertexCollHandle.getByLabel(tree,"offlinePrimaryVertices");
      if(!vertexCollHandle.isValid()){printf("Vertex Collection NotFound\n");continue;}
      const std::vector<reco::Vertex>& vertexColl = *vertexCollHandle;


      for(unsigned int c=0;c<hscpColl.size();c++){
         susybsm::HSCParticle hscp  = hscpColl[c];
         reco::TrackRef track = hscp.trackRef();
         if(track.isNull())continue;

         const DeDxData& dedxSObj  = dEdxSCollH->get(track.key());
         const DeDxData& dedxMObj  = dEdxMCollH->get(track.key());
         const DeDxData& dedxMSObj  = dEdxMSCollH->get(track.key());
         const DeDxData& dedxMPObj  = dEdxMPCollH->get(track.key());
         const reco::MuonTimeExtra* tof = NULL;
         const reco::MuonTimeExtra* dttof = NULL;
         const reco::MuonTimeExtra* csctof = NULL;
         if(!hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); dttof  = &TOFDTCollH->get(hscp.muonRef().key()); csctof  = &TOFCSCCollH->get(hscp.muonRef().key());}


         if(!PassPreselection(hscp, dedxSObj, dedxMObj, tof, dttof, csctof, tree))continue;

         for(unsigned int i=0;i<triggers.size();i++){
            if(!PassingTrigger(tree,triggers[i]))continue;

            NVertProf[i]->Fill(CurrentRunIndex, vertexColl.size()); 

            if(tof && tof->nDof()>=GlobalMinNDOF && (dttof->nDof()>=GlobalMinNDOFDT || csctof->nDof()>=GlobalMinNDOFCSC) && tof->inverseBetaErr()<=GlobalMaxTOFErr){
               if(tof->inverseBeta()>=GlobalMinTOF)CountMu[i]->Fill(CurrentRunIndex);
               if(tof->inverseBeta()>=GlobalMinTOF)TOFOverMinProf[i]->Fill(CurrentRunIndex, tof->inverseBeta());
               if(dttof->inverseBeta()>=GlobalMinTOF)TOFDTOverMinProf[i]->Fill(CurrentRunIndex, dttof->inverseBeta());
               if(csctof->inverseBeta()>=GlobalMinTOF)TOFCSCOverMinProf[i]->Fill(CurrentRunIndex, csctof->inverseBeta());
               TOFProf[i]->Fill(CurrentRunIndex, tof->inverseBeta());
               if(dttof->nDof()>=GlobalMinNDOFDT) TOFDTProf[i]->Fill(CurrentRunIndex, dttof->inverseBeta());
               if(csctof->nDof()>=GlobalMinNDOFCSC) TOFCSCProf[i]->Fill(CurrentRunIndex, csctof->inverseBeta());
               if(tof->inverseBeta() > 1.1 ) HTOF[i]->Fill(CurrentRunIndex);            
               VertexProf[i]->Fill(CurrentRunIndex, tof->timeAtIpInOut());
               if(dttof->nDof()>=GlobalMinNDOFDT) VertexDTProf[i]->Fill(CurrentRunIndex, dttof->timeAtIpInOut());
               if(csctof->nDof()>=GlobalMinNDOFCSC) VertexCSCProf[i]->Fill(CurrentRunIndex, csctof->timeAtIpInOut());
            }

            if(hscp.trackRef()->pt() >60 ) HPt[i]->Fill(CurrentRunIndex);
            if(dedxSObj.dEdx() > 0.15 ) HdEdx[i]->Fill(CurrentRunIndex);
            Count[i]->Fill(CurrentRunIndex);

            dEdxProf[i]->Fill(CurrentRunIndex, dedxSObj.dEdx());
            dEdxMProf[i]->Fill(CurrentRunIndex, dedxMObj.dEdx());
            dEdxMSProf[i]->Fill(CurrentRunIndex, dedxMSObj.dEdx());
            dEdxMPProf[i]->Fill(CurrentRunIndex, dedxMPObj.dEdx());
            if(fabs(track->eta())<0.5){
            dEdxMSCProf[i]->Fill(CurrentRunIndex, dedxMSObj.dEdx());
            dEdxMPCProf[i]->Fill(CurrentRunIndex, dedxMPObj.dEdx());
            }
            if(fabs(track->eta())>1.5){
            dEdxMSFProf[i]->Fill(CurrentRunIndex, dedxMSObj.dEdx());
            dEdxMPFProf[i]->Fill(CurrentRunIndex, dedxMPObj.dEdx());
            }
            PtProf[i]->Fill(CurrentRunIndex, hscp.trackRef()->pt());
         }

      }
   }printf("\n");
   }

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
   leg->AddEntry(HdEdx[i],"I_{as} > 0.15","P");
   leg->Draw();

   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
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
   leg->AddEntry(HPt[i],"p_{T} > 60 GeV/c","P");
   leg->Draw();
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
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
   DrawPreliminary(SQRTS, IntegratedLuminosity);
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
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Count");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   NVertProf[i]->LabelsDeflate("X");
   NVertProf[i]->LabelsOption("av","X");
   NVertProf[i]->GetXaxis()->SetNdivisions(505);
   NVertProf[i]->SetTitle("");
   NVertProf[i]->SetStats(kFALSE);
   NVertProf[i]->GetXaxis()->SetTitle("");
   NVertProf[i]->GetYaxis()->SetTitle("#RecoVertex");
   NVertProf[i]->GetYaxis()->SetTitleOffset(0.9);
   NVertProf[i]->GetXaxis()->SetLabelSize(0.04);
   NVertProf[i]->SetLineColor(Color[0]);
   NVertProf[i]->SetFillColor(Color[0]);
   NVertProf[i]->SetMarkerSize(0.4);
   NVertProf[i]->SetMarkerStyle(Marker[0]);
   NVertProf[i]->SetMarkerColor(Color[0]);
   NVertProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_NVert");
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
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_Is");
   delete c1;



   c1 = new TCanvas("c1","c1",600,600);
   dEdxMProf[i]->LabelsDeflate("X");
   dEdxMProf[i]->LabelsOption("av","X");
   dEdxMProf[i]->GetXaxis()->SetNdivisions(505);
   dEdxMProf[i]->SetTitle("");
   dEdxMProf[i]->SetStats(kFALSE);
   dEdxMProf[i]->GetXaxis()->SetTitle("");
   dEdxMProf[i]->GetYaxis()->SetTitle("dE/dx estimator");
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
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_Im");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   dEdxMSProf[i]->LabelsDeflate("X");
   dEdxMSProf[i]->LabelsOption("av","X");
   dEdxMSProf[i]->GetXaxis()->SetNdivisions(505);
   dEdxMSProf[i]->SetTitle("");
   dEdxMSProf[i]->SetStats(kFALSE);
   dEdxMSProf[i]->GetXaxis()->SetTitle("");
   dEdxMSProf[i]->GetYaxis()->SetTitle("dE/dx estimator");
   dEdxMSProf[i]->GetYaxis()->SetTitleOffset(0.9);
   dEdxMSProf[i]->GetXaxis()->SetLabelSize(0.04);
   dEdxMSProf[i]->SetLineColor(Color[0]);
   dEdxMSProf[i]->SetFillColor(Color[0]);
   dEdxMSProf[i]->SetMarkerSize(0.4);
   dEdxMSProf[i]->SetMarkerStyle(Marker[0]);
   dEdxMSProf[i]->SetMarkerColor(Color[0]);
   dEdxMSProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_ImS");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   dEdxMPProf[i]->LabelsDeflate("X");
   dEdxMPProf[i]->LabelsOption("av","X");
   dEdxMPProf[i]->GetXaxis()->SetNdivisions(505);
   dEdxMPProf[i]->SetTitle("");
   dEdxMPProf[i]->SetStats(kFALSE);
   dEdxMPProf[i]->GetXaxis()->SetTitle("");
   dEdxMPProf[i]->GetYaxis()->SetTitle("dE/dx estimator");
   dEdxMPProf[i]->GetYaxis()->SetTitleOffset(0.9);
   dEdxMPProf[i]->GetXaxis()->SetLabelSize(0.04);
   dEdxMPProf[i]->SetLineColor(Color[0]);
   dEdxMPProf[i]->SetFillColor(Color[0]);
   dEdxMPProf[i]->SetMarkerSize(0.4);
   dEdxMPProf[i]->SetMarkerStyle(Marker[0]);
   dEdxMPProf[i]->SetMarkerColor(Color[0]);
   dEdxMPProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_ImP");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   dEdxMSCProf[i]->LabelsDeflate("X");
   dEdxMSCProf[i]->LabelsOption("av","X");
   dEdxMSCProf[i]->GetXaxis()->SetNdivisions(505);
   dEdxMSCProf[i]->SetTitle("");
   dEdxMSCProf[i]->SetStats(kFALSE);
   dEdxMSCProf[i]->GetXaxis()->SetTitle("");
   dEdxMSCProf[i]->GetYaxis()->SetTitle("dE/dx estimator");
   dEdxMSCProf[i]->GetYaxis()->SetTitleOffset(0.9);
   dEdxMSCProf[i]->GetXaxis()->SetLabelSize(0.04);
   dEdxMSCProf[i]->SetLineColor(Color[0]);
   dEdxMSCProf[i]->SetFillColor(Color[0]);
   dEdxMSCProf[i]->SetMarkerSize(0.4);
   dEdxMSCProf[i]->SetMarkerStyle(Marker[0]);
   dEdxMSCProf[i]->SetMarkerColor(Color[0]);
   dEdxMSCProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_ImSC");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   dEdxMPCProf[i]->LabelsDeflate("X");
   dEdxMPCProf[i]->LabelsOption("av","X");
   dEdxMPCProf[i]->GetXaxis()->SetNdivisions(505);
   dEdxMPCProf[i]->SetTitle("");
   dEdxMPCProf[i]->SetStats(kFALSE);
   dEdxMPCProf[i]->GetXaxis()->SetTitle("");
   dEdxMPCProf[i]->GetYaxis()->SetTitle("dE/dx estimator");
   dEdxMPCProf[i]->GetYaxis()->SetTitleOffset(0.9);
   dEdxMPCProf[i]->GetXaxis()->SetLabelSize(0.04);
   dEdxMPCProf[i]->SetLineColor(Color[0]);
   dEdxMPCProf[i]->SetFillColor(Color[0]);
   dEdxMPCProf[i]->SetMarkerSize(0.4);
   dEdxMPCProf[i]->SetMarkerStyle(Marker[0]);
   dEdxMPCProf[i]->SetMarkerColor(Color[0]);
   dEdxMPCProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_ImPC");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   dEdxMSFProf[i]->LabelsDeflate("X");
   dEdxMSFProf[i]->LabelsOption("av","X");
   dEdxMSFProf[i]->GetXaxis()->SetNdivisions(505);
   dEdxMSFProf[i]->SetTitle("");
   dEdxMSFProf[i]->SetStats(kFALSE);
   dEdxMSFProf[i]->GetXaxis()->SetTitle("");
   dEdxMSFProf[i]->GetYaxis()->SetTitle("dE/dx estimator");
   dEdxMSFProf[i]->GetYaxis()->SetTitleOffset(0.9);
   dEdxMSFProf[i]->GetXaxis()->SetLabelSize(0.04);
   dEdxMSFProf[i]->SetLineColor(Color[0]);
   dEdxMSFProf[i]->SetFillColor(Color[0]);
   dEdxMSFProf[i]->SetMarkerSize(0.4);
   dEdxMSFProf[i]->SetMarkerStyle(Marker[0]);
   dEdxMSFProf[i]->SetMarkerColor(Color[0]);
   dEdxMSFProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_ImSF");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   dEdxMPFProf[i]->LabelsDeflate("X");
   dEdxMPFProf[i]->LabelsOption("av","X");
   dEdxMPFProf[i]->GetXaxis()->SetNdivisions(505);
   dEdxMPFProf[i]->SetTitle("");
   dEdxMPFProf[i]->SetStats(kFALSE);
   dEdxMPFProf[i]->GetXaxis()->SetTitle("");
   dEdxMPFProf[i]->GetYaxis()->SetTitle("dE/dx estimator");
   dEdxMPFProf[i]->GetYaxis()->SetTitleOffset(0.9);
   dEdxMPFProf[i]->GetXaxis()->SetLabelSize(0.04);
   dEdxMPFProf[i]->SetLineColor(Color[0]);
   dEdxMPFProf[i]->SetFillColor(Color[0]);
   dEdxMPFProf[i]->SetMarkerSize(0.4);
   dEdxMPFProf[i]->SetMarkerStyle(Marker[0]);
   dEdxMPFProf[i]->SetMarkerColor(Color[0]);
   dEdxMPFProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_ImPF");
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
   DrawPreliminary(SQRTS, IntegratedLuminosity);
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
   DrawPreliminary(SQRTS, IntegratedLuminosity);
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
   DrawPreliminary(SQRTS, IntegratedLuminosity);
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
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_TOFCSC");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   TOFOverMinProf[i]->LabelsDeflate("X");
   TOFOverMinProf[i]->LabelsOption("av","X");
   TOFOverMinProf[i]->GetXaxis()->SetNdivisions(505);
   TOFOverMinProf[i]->SetTitle("");
   TOFOverMinProf[i]->SetStats(kFALSE);
   TOFOverMinProf[i]->GetXaxis()->SetTitle("");
   TOFOverMinProf[i]->GetYaxis()->SetTitle("1/#beta");
   TOFOverMinProf[i]->GetYaxis()->SetTitleOffset(0.9);
   TOFOverMinProf[i]->GetXaxis()->SetLabelSize(0.04);
   TOFOverMinProf[i]->SetLineColor(Color[0]);
   TOFOverMinProf[i]->SetFillColor(Color[0]);
   TOFOverMinProf[i]->SetMarkerSize(0.4);
   TOFOverMinProf[i]->SetMarkerStyle(Marker[0]);
   TOFOverMinProf[i]->SetMarkerColor(Color[0]);
   TOFOverMinProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_TOFOverMin");
   delete c1;


   c1 = new TCanvas("c1","c1",600,600);
   TOFDTOverMinProf[i]->LabelsDeflate("X");
   TOFDTOverMinProf[i]->LabelsOption("av","X");
   TOFDTOverMinProf[i]->GetXaxis()->SetNdivisions(505);
   TOFDTOverMinProf[i]->SetTitle("");
   TOFDTOverMinProf[i]->SetStats(kFALSE);
   TOFDTOverMinProf[i]->GetXaxis()->SetTitle("");
   TOFDTOverMinProf[i]->GetYaxis()->SetTitle("1/#beta");
   TOFDTOverMinProf[i]->GetYaxis()->SetTitleOffset(0.9);
   TOFDTOverMinProf[i]->GetXaxis()->SetLabelSize(0.04);
   TOFDTOverMinProf[i]->SetLineColor(Color[0]);
   TOFDTOverMinProf[i]->SetFillColor(Color[0]);
   TOFDTOverMinProf[i]->SetMarkerSize(0.4);
   TOFDTOverMinProf[i]->SetMarkerStyle(Marker[0]);
   TOFDTOverMinProf[i]->SetMarkerColor(Color[0]);
   TOFDTOverMinProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_TOFDTOverMin");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   TOFCSCOverMinProf[i]->LabelsDeflate("X");
   TOFCSCOverMinProf[i]->LabelsOption("av","X");
   TOFCSCOverMinProf[i]->GetXaxis()->SetNdivisions(505);
   TOFCSCOverMinProf[i]->SetTitle("");
   TOFCSCOverMinProf[i]->SetStats(kFALSE);
   TOFCSCOverMinProf[i]->GetXaxis()->SetTitle("");
   TOFCSCOverMinProf[i]->GetYaxis()->SetTitle("1/#beta");
   TOFCSCOverMinProf[i]->GetYaxis()->SetTitleOffset(0.9);
   TOFCSCOverMinProf[i]->GetXaxis()->SetLabelSize(0.04);
   TOFCSCOverMinProf[i]->SetLineColor(Color[0]);
   TOFCSCOverMinProf[i]->SetFillColor(Color[0]);
   TOFCSCOverMinProf[i]->SetMarkerSize(0.4);
   TOFCSCOverMinProf[i]->SetMarkerStyle(Marker[0]);
   TOFCSCOverMinProf[i]->SetMarkerColor(Color[0]);
   TOFCSCOverMinProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_TOFCSCOverMin");
   delete c1;


   c1 = new TCanvas("c1","c1",600,600);
   VertexProf[i]->LabelsDeflate("X");
   VertexProf[i]->LabelsOption("av","X");
   VertexProf[i]->GetXaxis()->SetNdivisions(505);
   VertexProf[i]->SetTitle("");
   VertexProf[i]->SetStats(kFALSE);
   VertexProf[i]->GetXaxis()->SetTitle("");
   VertexProf[i]->GetYaxis()->SetTitle("1/#beta");
   VertexProf[i]->GetYaxis()->SetTitleOffset(0.9);
   VertexProf[i]->GetXaxis()->SetLabelSize(0.04);
   VertexProf[i]->SetLineColor(Color[0]);
   VertexProf[i]->SetFillColor(Color[0]);
   VertexProf[i]->SetMarkerSize(0.4);
   VertexProf[i]->SetMarkerStyle(Marker[0]);
   VertexProf[i]->SetMarkerColor(Color[0]);
   VertexProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_Vertex");
   delete c1;


   c1 = new TCanvas("c1","c1",600,600);
   VertexDTProf[i]->LabelsDeflate("X");
   VertexDTProf[i]->LabelsOption("av","X");
   VertexDTProf[i]->GetXaxis()->SetNdivisions(505);
   VertexDTProf[i]->SetTitle("");
   VertexDTProf[i]->SetStats(kFALSE);
   VertexDTProf[i]->GetXaxis()->SetTitle("");
   VertexDTProf[i]->GetYaxis()->SetTitle("1/#beta");
   VertexDTProf[i]->GetYaxis()->SetTitleOffset(0.9);
   VertexDTProf[i]->GetXaxis()->SetLabelSize(0.04);
   VertexDTProf[i]->SetLineColor(Color[0]);
   VertexDTProf[i]->SetFillColor(Color[0]);
   VertexDTProf[i]->SetMarkerSize(0.4);
   VertexDTProf[i]->SetMarkerStyle(Marker[0]);
   VertexDTProf[i]->SetMarkerColor(Color[0]);
   VertexDTProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_VertexDT");
   delete c1;

   c1 = new TCanvas("c1","c1",600,600);
   VertexCSCProf[i]->LabelsDeflate("X");
   VertexCSCProf[i]->LabelsOption("av","X");
   VertexCSCProf[i]->GetXaxis()->SetNdivisions(505);
   VertexCSCProf[i]->SetTitle("");
   VertexCSCProf[i]->SetStats(kFALSE);
   VertexCSCProf[i]->GetXaxis()->SetTitle("");
   VertexCSCProf[i]->GetYaxis()->SetTitle("1/#beta");
   VertexCSCProf[i]->GetYaxis()->SetTitleOffset(0.9);
   VertexCSCProf[i]->GetXaxis()->SetLabelSize(0.04);
   VertexCSCProf[i]->SetLineColor(Color[0]);
   VertexCSCProf[i]->SetFillColor(Color[0]);
   VertexCSCProf[i]->SetMarkerSize(0.4);
   VertexCSCProf[i]->SetMarkerStyle(Marker[0]);
   VertexCSCProf[i]->SetMarkerColor(Color[0]);
   VertexCSCProf[i]->Draw("E1");
   c1->Modified();
   c1->SetGridx(true);
   DrawPreliminary(SQRTS, IntegratedLuminosity);
   SaveCanvas(c1,string("pictures/") + triggers[i],"Profile_VertexCSC");
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


