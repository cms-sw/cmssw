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
#include "TProfile.h"

#include <exception>
#include <vector>
#include <string>


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

#include "../../ICHEP_Analysis/Analysis_Global.h"
#include "../../ICHEP_Analysis/Analysis_Samples.h"
#include "../../ICHEP_Analysis/Analysis_PlotFunction.h"

#endif

bool PassPreselection(const susybsm::HSCParticle& hscp,  const reco::DeDxData& dedxSObj, const reco::DeDxData& dedxMObj, const reco::MuonTimeExtra* tof, const reco::MuonTimeExtra* dttof, const reco::MuonTimeExtra* csctof, const fwlite::ChainEvent& ev);

double deltaR(double eta1, double phi1, double eta2, double phi2) {
   double deta = eta1 - eta2;
   double dphi = phi1 - phi2;
   while (dphi >   M_PI) dphi -= 2*M_PI;
   while (dphi <= -M_PI) dphi += 2*M_PI;
   return sqrt(deta*deta + dphi*dphi);
}


bool PassingTrigger(const fwlite::ChainEvent& ev, const std::string& TriggerName){
      edm::TriggerResultsByName tr = ev.triggerResultsByName("HLT");
      if(!tr.isValid())return false;

      unsigned int TrIndex_Unknown     = tr.size();

      if(TriggerName=="MET" || TriggerName=="ANY"){
        if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMET150_v1")) {
          if(tr.accept(tr.triggerIndex("HLT_PFMET150_v1"))){return true;}
        }else if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMET150_v2")) {
          if(tr.accept(tr.triggerIndex("HLT_PFMET150_v2"))){return true;}
        }else if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMET150_v3")) {
          if(tr.accept(tr.triggerIndex("HLT_PFMET150_v3"))){return true;}
        }else if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMET150_v4")) {
          if(tr.accept(tr.triggerIndex("HLT_PFMET150_v4"))){return true;}
	}else if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMET150_v5")) {
          if(tr.accept(tr.triggerIndex("HLT_PFMET150_v5"))){return true;}
        }else if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMET150_v6")) {
          if(tr.accept(tr.triggerIndex("HLT_PFMET150_v6"))){return true;}
	}else if(TrIndex_Unknown != tr.triggerIndex("HLT_PFMET150_v7")) {
          if(tr.accept(tr.triggerIndex("HLT_PFMET150_v7"))){return true;}
	}
      }

      if(TriggerName=="HT" || TriggerName=="ANY"){
      if(TrIndex_Unknown != tr.triggerIndex("HLT_HT650_v1")) {
        if(tr.accept(tr.triggerIndex("HLT_HT650_v1"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_HT650_v2")) {
        if(tr.accept(tr.triggerIndex("HLT_HT650_v2"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_HT650_v3")) {
        if(tr.accept(tr.triggerIndex("HLT_HT650_v3"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_HT650_v4")) {
        if(tr.accept(tr.triggerIndex("HLT_HT650_v4"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_HT650_v5")) {
        if(tr.accept(tr.triggerIndex("HLT_HT650_v5"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_HT650_v6")) {
        if(tr.accept(tr.triggerIndex("HLT_HT650_v6"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_HT650_v7")) {
        if(tr.accept(tr.triggerIndex("HLT_HT650_v7"))){return true;}
      }
      }

      if(TriggerName=="Mu" || TriggerName=="ANY"){
      if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v1")) {
        if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v1"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v2")) {
	if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v2"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v3")) {
        if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v3"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v4")) {
        if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v4"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v5")) {
        if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v5"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v6")) {
        if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v6"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v7")) {
	if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v7"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v8")) {
	if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v8"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v9")) {
	if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v9"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v10")) {
        if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v10"))){return true;}
      }else if(TrIndex_Unknown != tr.triggerIndex("HLT_Mu40_eta2p1_v11")) {
        if(tr.accept(tr.triggerIndex("HLT_Mu40_eta2p1_v11"))){return true;}
      }
      }

   return false;
}

double HLTObjectDr(const trigger::TriggerEvent& trEv, std::string trigger, const susybsm::HSCParticle& hscp)
{
    reco::TrackRef   track = hscp.trackRef(); if(track.isNull())return false;

    unsigned int filterIndex=trEv.sizeFilters();
    if(trigger=="Mu") filterIndex = trEv.filterIndex(InputTag("hltDeDxFilter50DEDX3p6Mu","","HLT"));
    if(trigger=="MET" || trigger=="HT") filterIndex = trEv.filterIndex(InputTag("hltDeDxFilter50DEDX3p6","","HLT"));

    double mindR = 999;                                                                                                                                                          
   if (filterIndex<trEv.sizeFilters()){      
      const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
      const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
      const size_type nI(VIDS.size());
      const size_type nK(KEYS.size());
      assert(nI==nK);
      const size_type n(max(nI,nK));
      const trigger::TriggerObjectCollection& TOC(trEv.getObjects());

        for (size_type i=0; i!=n; ++i) {
         const TriggerObject& TO(TOC[KEYS[i]]);
	 double dR = deltaR(track->eta(), track->phi(), TO.eta(), TO.phi());
	 if(dR<mindR) mindR=dR;
      }
   }
   return mindR;
} 


void TriggerEfficiency(string MODE="COMPILE")
{
   if(MODE=="COMPILE") return;

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

   system("mkdir pictures/");
   TFile* OutputHisto = new TFile(("pictures/Efficiency_Histos_"+samples[0].Name+"_"+samples[0].FileName+".root").c_str(),"RECREATE");
   //TFile* OutputHisto = new TFile("out.root","RECREATE");

   std::vector<string> triggers;
   triggers.push_back("Mu");
   triggers.push_back("MET");
   triggers.push_back("HT");

   TH1D* HDr[triggers.size()];
   TH1D* MDeDxBot[triggers.size()];
   TH1D* SDeDxBot[triggers.size()];
   TH1D* MSDeDxBot[triggers.size()];
   TH1D* MDeDxTop[triggers.size()];
   TH1D* SDeDxTop[triggers.size()];
   TH1D* MSDeDxTop[triggers.size()];
   TProfile* MDeDxEff[triggers.size()];
   TProfile* SDeDxEff[triggers.size()];
   TProfile* MSDeDxEff[triggers.size()];

   for(unsigned int i=0;i<triggers.size();i++){
     HDr[i]  = new TH1D((triggers[i] + "Dr").c_str()  ,"Dr",100, 0, 2);
     MDeDxBot[i]  = new TH1D((triggers[i] + "MDeDxBot").c_str()  ,"MDeDx",240, 0, 6);
     SDeDxBot[i]  = new TH1D((triggers[i] + "SDeDxBot").c_str()  ,"SDeDx",240, 0, 1);
     MSDeDxBot[i]  = new TH1D((triggers[i] + "MSDeDxBot").c_str()  ,"MSDeDx",240, 0, 6);
     MDeDxTop[i]  = new TH1D((triggers[i] + "MDeDxTop").c_str()  ,"MDeDx",240, 0, 6);
     SDeDxTop[i]  = new TH1D((triggers[i] + "SDeDxTop").c_str()  ,"SDeDx",240, 0, 1);
     MSDeDxTop[i]  = new TH1D((triggers[i] + "MSDeDxTop").c_str()  ,"MSDeDx",240, 0, 6);
     MDeDxEff[i]  = new TProfile((triggers[i] + "MDeDxEff").c_str()  ,"MDeDxEff",240, 0, 6);
     SDeDxEff[i]  = new TProfile((triggers[i] + "SDeDxEff").c_str() ,"SDeDxEff",240, 0,1);
     MSDeDxEff[i]  = new TProfile((triggers[i] + "MSDeDxEff").c_str()  ,"MSDeDxEff",240, 0, 6);
   }

   for(unsigned int s=0;s<samples.size();s++){
     std::vector<string> FileName;
     GetInputFiles(samples[s], BaseDirectory, FileName);
     fwlite::ChainEvent ev(FileName);

   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on %10s        :","");
   int TreeStep = ev.size()/50;if(TreeStep==0)TreeStep=1;
   for(Long64_t e=0;e<ev.size();e++){
      if(e%TreeStep==0){printf(".");fflush(stdout);}
      ev.to(e);       

      if(!PassingTrigger(ev, "ANY")) continue;

//      printf("e=%i Run=%i Lumi=%i Event=%i BX=%i  Orbit=%i Store=%i\n",e,ev.eventAuxiliary().run(),ev.eventAuxiliary().luminosityBlock(),ev.eventAuxiliary().event(),ev.eventAuxiliary().luminosityBlock(),ev.eventAuxiliary().orbitNumber(),ev.eventAuxiliary().storeNumber());

      fwlite::Handle<susybsm::HSCParticleCollection> hscpCollHandle;
      hscpCollHandle.getByLabel(ev,"HSCParticleProducer");
      if(!hscpCollHandle.isValid()){printf("HSCP Collection NotFound\n");continue;}
      susybsm::HSCParticleCollection hscpColl = *hscpCollHandle;

      fwlite::Handle<DeDxDataValueMap> dEdxSCollH;
      dEdxSCollH.getByLabel(ev, "dedxASmi");
      if(!dEdxSCollH.isValid()){printf("Invalid dEdx Selection collection\n");continue;}

      fwlite::Handle<DeDxDataValueMap> dEdxMCollH;
      dEdxMCollH.getByLabel(ev, "dedxHarm2");
      if(!dEdxMCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}

      fwlite::Handle<DeDxDataValueMap> dEdxMSCollH;
      dEdxMSCollH.getByLabel(ev, "dedxNPHarm2");
      if(!dEdxMSCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}

      //fwlite::Handle<DeDxDataValueMap> dEdxMNSTCollH;
      //dEdxMNSTCollH.getByLabel(ev, "dedxNSTHarm2");
      //if(!dEdxMNSTCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}

      fwlite::Handle<MuonTimeExtraMap> TOFCollH;
      TOFCollH.getByLabel(ev, "muontiming",TOF_Label.c_str());
      if(!TOFCollH.isValid()){printf("Invalid TOF collection\n");continue;}

      fwlite::Handle<MuonTimeExtraMap> TOFDTCollH;
      TOFDTCollH.getByLabel(ev, "muontiming",TOFdt_Label.c_str());
      if(!TOFDTCollH.isValid()){printf("Invalid DT TOF collection\n");continue;}

      fwlite::Handle<MuonTimeExtraMap> TOFCSCCollH;
      TOFCSCCollH.getByLabel(ev, "muontiming",TOFcsc_Label.c_str());
      if(!TOFCSCCollH.isValid()){printf("Invalid CSCTOF collection\n");continue;}

      fwlite::Handle< trigger::TriggerEvent > trEvHandle;
      trEvHandle.getByLabel(ev,"hltTriggerSummaryAOD");
      if(!trEvHandle.isValid())continue;
      trigger::TriggerEvent trEv = *trEvHandle;

      for(unsigned int c=0;c<hscpColl.size();c++){
	susybsm::HSCParticle hscp  = hscpColl[c];
	reco::TrackRef track = hscp.trackRef();
	if(track.isNull())continue;

	const DeDxData& dedxSObj  = dEdxSCollH->get(track.key());
	const DeDxData& dedxMObj  = dEdxMCollH->get(track.key());
	const DeDxData& dedxMSObj  = dEdxMSCollH->get(track.key());
	//const DeDxData& dedxMNSTObj  = dEdxMNSTCollH->get(track.key());

        const reco::MuonTimeExtra* tof = NULL;
         const reco::MuonTimeExtra* dttof = NULL;
         const reco::MuonTimeExtra* csctof = NULL;
         if(!hscp.muonRef().isNull()){ tof  = &TOFCollH->get(hscp.muonRef().key()); dttof  = &TOFDTCollH->get(hscp.muonRef().key()); csctof  = &TOFCSCCollH->get(hscp.muonRef().key());}

         if(!PassPreselection(hscp, dedxSObj, dedxMObj, tof, dttof, csctof, ev))continue;

	 for(unsigned int i=0; i<triggers.size(); i++) {
	   if(!PassingTrigger(ev, triggers[i])) continue;
           MDeDxBot[i]->Fill(dedxMObj.dEdx());
           SDeDxBot[i]->Fill(dedxSObj.dEdx());
           MSDeDxBot[i]->Fill(dedxMSObj.dEdx());

	   int match=0;
	   double dR = HLTObjectDr(trEv, triggers[i], hscp);
	   HDr[i]->Fill(dR);
	   if(dR<0.3) match=1;
	   if(match==1) {
	     MDeDxTop[i]->Fill(dedxMObj.dEdx());
	     SDeDxTop[i]->Fill(dedxSObj.dEdx());
             MSDeDxTop[i]->Fill(dedxMSObj.dEdx());
	   }
	   MDeDxEff[i]->Fill(dedxMObj.dEdx(), match);
	   SDeDxEff[i]->Fill(dedxSObj.dEdx(), match);
           MSDeDxEff[i]->Fill(dedxMSObj.dEdx(), match);
	 }
      }
   }printf("\n");
   }

   OutputHisto->Write();
   OutputHisto->Close();
}

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
   if(track->pt()<55)return false;

   //if(dedxSObj.dEdx()<GlobalMinIs)return false;
   //if(dedxMObj.dEdx()<GlobalMinIm)return false;


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
