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


bool PassingTrigger(const fwlite::ChainEvent& ev){
      edm::TriggerResultsByName tr = ev.triggerResultsByName("MergeHLT");
      if(!tr.isValid())return false;
       return tr.accept(tr.triggerIndex("HSCPHLTTriggerMuFilter"));
}




void TOFCheck(string MODE="COMPILE")
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

   // get all the samples and clean the list to keep only the one we want to run on... Also initialize the BaseDirectory
   std::vector<stSample> samples;

   InitBaseDirectory();
   GetSampleDefinition(samples, "../../ICHEP_Analysis/Analysis_Samples.txt");
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
   system("mkdir pictures/");
   TFile* OutputHisto = new TFile(("pictures/Histos_"+samples[0].Name+"_"+samples[0].FileName+".root").c_str(),"RECREATE");

   TH1D* TOF = new TH1D("TOF", "TOF", 100, 0, 2); TOF->Sumw2();
   TH1D* TOFDT  = new TH1D("TOFDT", "TOFDT", 100, 0, 2); TOFDT->Sumw2();
   TH1D* TOFCSC  = new TH1D("TOFCSC", "TOFCSC", 100, 0, 2); TOFCSC->Sumw2();
   TH1D* Vertex  = new TH1D("Vertex", "Vertex", 100, -10, 10); Vertex->Sumw2();
   TH1D* VertexDT = new TH1D("VertexDT", "VertexDT", 100, -10, 10); VertexDT->Sumw2();
   TH1D* VertexCSC = new TH1D("VertexCSC", "VertexCSC", 100, -10, 10); VertexCSC->Sumw2();
   TProfile* TOFVsEta = new TProfile("TOFVsEta", "TOFVsEta", 50, -2.1, 2.1); TOFVsEta->Sumw2();
   TProfile* TOFVsPhi = new TProfile("TOFVsPhi", "TOFVsPhi", 50, -3.14, 3.14); TOFVsPhi->Sumw2();
   TProfile* TOFVsPt  = new TProfile("TOFVsPt" , "TOFVsPt" , 50, 40, 140); TOFVsPt->Sumw2();
   TProfile* CSCTOFVsEta = new TProfile("CSCTOFVsEta", "CSCTOFVsEta", 50, -2.1, 2.1); CSCTOFVsEta->Sumw2();
   TProfile* CSCTOFVsPhi = new TProfile("CSCTOFVsPhi", "CSCTOFVsPhi", 50, -3.14, 3.14); CSCTOFVsPhi->Sumw2();
   TProfile* CSCTOFVsPt  = new TProfile("CSCTOFVsPt" , "CSCTOFVsPt" , 50, 40, 140); CSCTOFVsPt->Sumw2();
   TProfile* DTTOFVsEta = new TProfile("DTTOFVsEta", "DTTOFVsEta", 50, -2.1, 2.1); DTTOFVsEta->Sumw2();
   TProfile* DTTOFVsPhi = new TProfile("DTTOFVsPhi", "DTTOFVsPhi", 50, -3.14, 3.14); DTTOFVsPhi->Sumw2();
   TProfile* DTTOFVsPt  = new TProfile("DTTOFVsPt" , "DTTOFVsPt" , 50, 40, 140); DTTOFVsPt->Sumw2();

   TProfile* VertexVsEta = new TProfile("VertexVsEta", "VertexVsEta", 50, -2.1, 2.1); VertexVsEta->Sumw2();
   TProfile* VertexVsPhi = new TProfile("VertexVsPhi", "VertexVsPhi", 50, -3.14, 3.14); VertexVsPhi->Sumw2();
   TProfile* VertexVsPt  = new TProfile("VertexVsPt" , "VertexVsPt" , 50, 40, 140); VertexVsPt->Sumw2();
   TProfile* CSCVertexVsEta = new TProfile("CSCVertexVsEta", "CSCVertexVsEta", 50, -2.1, 2.1); CSCVertexVsEta->Sumw2();
   TProfile* CSCVertexVsPhi = new TProfile("CSCVertexVsPhi", "CSCVertexVsPhi", 50, -3.14, 3.14); CSCVertexVsPhi->Sumw2();
   TProfile* CSCVertexVsPt  = new TProfile("CSCVertexVsPt" , "CSCVertexVsPt" , 50, 40, 140); CSCVertexVsPt->Sumw2();
   TProfile* DTVertexVsEta = new TProfile("DTVertexVsEta", "DTVertexVsEta", 50, -2.1, 2.1); DTVertexVsEta->Sumw2();
   TProfile* DTVertexVsPhi = new TProfile("DTVertexVsPhi", "DTVertexVsPhi", 50, -3.14, 3.14); DTVertexVsPhi->Sumw2();
   TProfile* DTVertexVsPt  = new TProfile("DTVertexVsPt" , "DTVertexVsPt" , 50, 40, 140); DTVertexVsPt->Sumw2();

   TypeMode      = 2;


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

      if(!PassingTrigger(tree))continue;

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

      double Mass=0;
      int ind1=-1, ind2=-1;

      for(unsigned int c=0;c<hscpColl.size();c++){
         susybsm::HSCParticle hscp  = hscpColl[c];
         reco::TrackRef track = hscp.trackRef();
         if(track.isNull())continue;
	 if(hscp.muonRef().isNull()) continue;

         const DeDxData& dedxSObj  = dEdxSCollH->get(track.key());
         const DeDxData& dedxMObj  = dEdxMCollH->get(track.key());

         const reco::MuonTimeExtra* tof = NULL;
         const reco::MuonTimeExtra* dttof = NULL;
         const reco::MuonTimeExtra* csctof = NULL;
         if(!hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); dttof  = &TOFDTCollH->get(hscp.muonRef().key()); csctof  = &TOFCSCCollH->get(hscp.muonRef().key());}
         if(!PassPreselection(hscp, dedxSObj, dedxMObj, tof, dttof, csctof, tree))continue;

	 for(unsigned int d=c+1;d<hscpColl.size();d++){
	   susybsm::HSCParticle hscp2  = hscpColl[d];
	   reco::TrackRef track2 = hscp2.trackRef();
	   if(track2.isNull())continue;
	   if(hscp2.muonRef().isNull()) continue;

	   const DeDxData& dedxSObj2  = dEdxSCollH->get(track.key());
	   const DeDxData& dedxMObj2  = dEdxMCollH->get(track.key());

	   const reco::MuonTimeExtra* tof2 = NULL;
	   const reco::MuonTimeExtra* dttof2 = NULL;
	   const reco::MuonTimeExtra* csctof2 = NULL;
	   if(!hscp2.muonRef().isNull()){ tof2  = &TOFCollH->get(hscp.muonRef().key()); dttof2  = &TOFDTCollH->get(hscp.muonRef().key()); csctof2  = &TOFCSCCollH->get(hscp.muonRef().key());}
	   if(!PassPreselection(hscp2, dedxSObj2, dedxMObj2, tof2, dttof2, csctof2, tree))continue;

	   if(track->charge()==track2->charge()) continue;
	   double E = track->p() + track2->p();
	   double px = track->px() + track2->px();
	   double py = track->py() + track2->py();
	   double pz = track->pz() + track2->pz();
	   double p = px + py +pz;
	   double M = sqrt(E*E - p*p);
	   if(fabs(M-91.1876)>fabs(Mass-91.1876)) continue;
	   Mass=M;
	   ind1=c;
	   ind2=d;
	 }
      }

      for(int c=0;c<(int)hscpColl.size();c++){
	if(c!=ind1 && c!=ind2) continue;
	susybsm::HSCParticle hscp  = hscpColl[c];
	reco::TrackRef track = hscp.trackRef();
	if(track.isNull())continue;
	if(hscp.muonRef().isNull()) continue;

	const reco::MuonTimeExtra* tof = NULL;
	const reco::MuonTimeExtra* dttof = NULL;
	const reco::MuonTimeExtra* csctof = NULL;
	if(!hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); dttof  = &TOFDTCollH->get(hscp.muonRef().key()); csctof  = &TOFCSCCollH->get(hscp.muonRef().key());}

	if(tof && tof->nDof()>=GlobalMinNDOF && (dttof->nDof()>=GlobalMinNDOFDT || csctof->nDof()>=GlobalMinNDOFCSC) && tof->inverseBetaErr()<=GlobalMaxTOFErr){
	  TOF->Fill(tof->inverseBeta());
          TOFVsEta->Fill(track->eta(), tof->inverseBeta());
          TOFVsPhi->Fill(track->phi(), tof->inverseBeta());
          TOFVsPt->Fill(track->pt(), tof->inverseBeta());
	  if(dttof->nDof()>=GlobalMinNDOFDT) {
	    TOFDT->Fill(dttof->inverseBeta());
	    DTTOFVsEta->Fill(track->eta(), dttof->inverseBeta());
	    DTTOFVsPhi->Fill(track->phi(), dttof->inverseBeta());
	    DTTOFVsPt->Fill(track->pt(), dttof->inverseBeta());
	  }
	  if(csctof->nDof()>=GlobalMinNDOFCSC) {
	    TOFCSC->Fill(csctof->inverseBeta());
	    CSCTOFVsEta->Fill(track->eta(), csctof->inverseBeta());
	    CSCTOFVsPhi->Fill(track->phi(), csctof->inverseBeta());
	    CSCTOFVsPt->Fill(track->pt(), csctof->inverseBeta());
	  }


	  Vertex->Fill(tof->timeAtIpInOut());
          VertexVsEta->Fill(track->eta(), tof->timeAtIpInOut());
          VertexVsPhi->Fill(track->phi(), tof->timeAtIpInOut());
          VertexVsPt->Fill(track->pt(), tof->timeAtIpInOut());
	  if(dttof->nDof()>=GlobalMinNDOFDT) {
	    VertexDT->Fill(dttof->timeAtIpInOut());
	    DTVertexVsEta->Fill(track->eta(), dttof->timeAtIpInOut());
	    DTVertexVsPhi->Fill(track->phi(), dttof->timeAtIpInOut());
	    DTVertexVsPt->Fill(track->pt(), dttof->timeAtIpInOut());
	  }
	  if(csctof->nDof()>=GlobalMinNDOFCSC) {
	    VertexCSC->Fill(csctof->timeAtIpInOut());
	    CSCVertexVsEta->Fill(track->eta(), csctof->timeAtIpInOut());
	    CSCVertexVsPhi->Fill(track->phi(), csctof->timeAtIpInOut());
	    CSCVertexVsPt->Fill(track->pt(), csctof->timeAtIpInOut());
	  }
	}
      }
     }printf("\n");
   }

   OutputHisto->Write();
   OutputHisto->Close();  
}

