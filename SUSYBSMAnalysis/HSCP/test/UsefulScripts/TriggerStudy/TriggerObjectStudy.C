
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

#endif

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


double deltaR(double eta1, double phi1, double eta2, double phi2) {
   double deta = eta1 - eta2;
   double dphi = phi1 - phi2;
   while (dphi >   M_PI) dphi -= 2*M_PI;
   while (dphi <= -M_PI) dphi += 2*M_PI;
   return sqrt(deta*deta + dphi*dphi);
}


int GetHLTObject(const trigger::TriggerEvent& trEv, const edm::InputTag& InputPath, double& pt, double& eta, double& phi, double& dEdx, int& NOH, int& NOM, int& NOS)
{
   unsigned int filterIndex = trEv.filterIndex(InputPath);
//   if(filterIndex<trEv.sizeFilters())printf("SELECTED INDEX =%i --> %s    XXX   %s\n",filterIndex,trEv.filterTag(filterIndex).label().c_str(), trEv.filterTag(filterIndex).process().c_str());

   if (filterIndex<trEv.sizeFilters()){      
      const trigger::Vids& VIDS(trEv.filterIds(filterIndex));
      const trigger::Keys& KEYS(trEv.filterKeys(filterIndex));
      const size_type nI(VIDS.size());
      const size_type nK(KEYS.size());
      assert(nI==nK);
      const size_type n(max(nI,nK));
      const trigger::TriggerObjectCollection& TOC(trEv.getObjects());

//      printf("CollectionSize = %i vs %i\n",(int)n, (int)TOC.size());
   

      pt   = 0.0;
      eta  = 0.0;
      phi  = 0.0;
      dEdx = 0.0;
      NOH  = 0;
      NOM  = 0;
      NOS  = 0;
      for (size_type i=0; i!=n; ++i) {
         const TriggerObject& TO(TOC[KEYS[i]]);
//         printf("%i  %3i/%3i pt=%6.2f eta=%+6.2f phi=%+6.2f  M=%6.2f NOH=%2i  NOM=%2i  NOS=%2i\n",i,VIDS[i],KEYS[i],TO.pt(),TO.eta(),TO.phi(),TO.mass(), TO.id()&0x0000FF, (TO.id()&0x00FF00)>>8, (TO.id()&0xFF0000)>>16);
	 pt   = TO.pt();
         eta  = TO.eta();
         phi  = TO.phi();
         dEdx = TO.mass();
         NOH  = TO.id()&0x0000FF;
         NOM  = (TO.id()&0x00FF00)>>8;
         NOS  = (TO.id()&0xFF0000)>>16;
         return n;
      }
   }
   return 0;
} 


void TriggerObjectStudy()
{
   TFile* OutputHisto = new TFile("out.root","RECREATE");
   TH2D* HEta  = new TH2D("Eta"  ,";#eta_{hlt};#eta_{offline}",50,-2.2,2.2, 50, -2.2,2.2);
   TH2D* HPhi  = new TH2D("Phi"  ,";#phi_{hlt};#phi_{offline}",50,-3.2,3.2, 50, -3.2,3.2);
   TH2D* HPt   = new TH2D("Pt"   ,";pt_{hlt};pt_{offline}"    ,50,40,200  , 50, 40,200);
   TH2D* HIh   = new TH2D("Ih"   ,";Ih_{hlt};Ih_{offline}",50,3,8, 50, 2,8);
   TH2D* HIhST = new TH2D("Ih ST",";Ih_{hlt};Ih_{offline-cluster cleaning}",50,3,8, 50, 2,8);
   TH2D* HIsST = new TH2D("Is ST",";Ih_{hlt};Is_{offline-cluster cleaning}",50,3,8, 50, 0,1);
   TH2D* HNOH  = new TH2D("NOH",";NOH_{hlt};NOH_{offline}",30,0,30, 30, 0,30);
   TH2D* HNOM  = new TH2D("NOM",";NOM_{hlt};NOM_{offline}",30,0,30, 30, 0,30);
   TH2D* HNOS  = new TH2D("NOS",";NOS_{hlt};NOH_{offline}",10,0,10, 10, 0,10);

   vector<string> fileNames;
   fileNames.push_back("../../BuildHSCParticles/Data20122/HSCP.root");
   fwlite::ChainEvent ev(fileNames);
   int N   = 0;
   int Nmu = 0;
   int Nht = 0;
   int Nmet = 0;
   int Ntriggered = 0;
   int Nhlt = 0;
   int Nhscp   = 0;

   //DuplicatesClass Duplicates;
   //Duplicates.Clear();

   for(Long64_t e=0;e<ev.size();e++){
      ev.to(e);       
      edm::TriggerResultsByName tr = ev.triggerResultsByName("HLT");      if(!tr.isValid())continue;
//     for(unsigned int i=0;i<tr.size();i++){
//         printf("Path %3i %50s --> %1i\n",i, tr.triggerName(i).c_str(),tr.accept(i));
//      }fflush(stdout);

      //if(Duplicates.isDuplicate(ev.eventAuxiliary().run(),ev.eventAuxiliary().event())){continue;}

      bool accept=false;
      if(tr.accept("HLT_Mu40_eta2p1_Track50_dEdx3p6_v3")){Nmu++; accept=true;}
      if(tr.accept("HLT_HT650_Track50_dEdx3p6_v4")){Nht++; accept=true;}
      if(tr.accept("HLT_MET80_Track50_dEdx3p6_v3")){Nmet++; accept=true;}
      N++;
 
      if(!accept)continue;     
      Ntriggered++;


      fwlite::Handle< trigger::TriggerEvent > trEvHandle;
      trEvHandle.getByLabel(ev,"hltTriggerSummaryAOD");
      trigger::TriggerEvent trEv = *trEvHandle;

//      for(unsigned int i=0;i<trEvHandle->sizeFilters();i++){
//         if(strncmp(trEvHandle->filterTag(i).label().c_str(),"hltL1",5)==0)continue;
//         printf("%i - %s\n",i,trEvHandle->filterTag(i).label().c_str());
//      }

      double hlt_pt; double hlt_eta; double hlt_phi; double hlt_dEdx; int hlt_NOH; int hlt_NOM; int hlt_NOS;
      bool isMuon=false;
      if(tr.accept("HLT_Mu40_eta2p1_Track50_dEdx3p6_v3")){
         Nhlt += GetHLTObject(trEv, InputTag("hltDeDxFilter50DEDX3p6Mu","","HLT"), hlt_pt, hlt_eta, hlt_phi, hlt_dEdx, hlt_NOH, hlt_NOM, hlt_NOS );
         isMuon=true;
      }else{
         Nhlt += GetHLTObject(trEv, InputTag("hltDeDxFilter50DEDX3p6","","HLT"), hlt_pt, hlt_eta, hlt_phi, hlt_dEdx, hlt_NOH, hlt_NOM, hlt_NOS );
      }

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

      fwlite::Handle<DeDxDataValueMap> dEdxMNSTCollH;
      dEdxMNSTCollH.getByLabel(ev, "dedxNSTHarm2");
      if(!dEdxMNSTCollH.isValid()){printf("Invalid dEdx Mass collection\n");continue;}



      double mindR = 999;
      int index=-1;
      for(unsigned int c=0;c<hscpColl.size();c++){
         susybsm::HSCParticle hscp  = hscpColl[c];
         reco::TrackRef track = hscp.trackRef();
         if(track.isNull())continue;
//         if(track->pt()<45)continue;

         double dR = deltaR(track->eta(), track->phi(), hlt_eta, hlt_phi);
         if(dR<mindR){mindR=dR; index=c;}
      }

      if(mindR>0.3 || index<0){
         printf("unmatched: isMuon=%1i pt=%7.2f eta=%+6.2f phi=%+6.2f  dEdx=%6.2f NOH=%2i  NOM=%2i  NOS=%2i\n",(int)isMuon, hlt_pt,hlt_eta,hlt_phi,hlt_dEdx, hlt_NOH, hlt_NOM, hlt_NOS);
         continue;
      }
      Nhscp++;
      susybsm::HSCParticle hscp  = hscpColl[index];
      reco::TrackRef track = hscp.trackRef();

      const DeDxData& dedxSObj  = dEdxSCollH->get(track.key());
      const DeDxData& dedxMObj  = dEdxMCollH->get(track.key());
      const DeDxData& dedxMSObj  = dEdxMSCollH->get(track.key());
      const DeDxData& dedxMNSTObj  = dEdxMNSTCollH->get(track.key());

      printf("  matched: isMuon=%1i pt=%7.2f(%7.2f) eta=%+6.2f(%+6.2f) phi=%+6.2f(%+6.2f)  dEdx=%6.2f(%6.2f Ih=%6.2f Ias=%6.2f) NOH=%2i(%2i)  NOM=%2i(%2i)  NOS=%2i(%2i)\n",(int)isMuon, hlt_pt, track->pt(),hlt_eta, track->eta(),hlt_phi,track->phi(),hlt_dEdx, dedxMNSTObj.dEdx(), dedxMObj.dEdx(), dedxSObj.dEdx(), hlt_NOH, track->found(), hlt_NOM, dedxMNSTObj.numberOfMeasurements(), hlt_NOS, dedxMNSTObj.numberOfSaturatedMeasurements());


      HEta ->Fill(hlt_eta, track->eta());
      HPhi ->Fill(hlt_phi,track->phi());
      HPt  ->Fill(hlt_pt, track->pt());
      HIh  ->Fill(hlt_dEdx, dedxMNSTObj.dEdx());
      HIhST->Fill(hlt_dEdx, dedxMObj.dEdx());
      HIsST->Fill(hlt_dEdx, dedxSObj.dEdx());
      HNOH ->Fill(hlt_NOH, track->found());
      HNOM ->Fill(hlt_NOM, dedxMNSTObj.numberOfMeasurements());
      HNOS ->Fill(hlt_NOS, dedxMNSTObj.numberOfSaturatedMeasurements());
 


   }
   printf("Nmu /N = %8i/%8i = %6.2f%%\n",Nmu ,N,(100.0*Nmu )/N);
   printf("Nmt /N = %8i/%8i = %6.2f%%\n",Nht ,N,(100.0*Nht )/N);
   printf("Nmet/N = %8i/%8i = %6.2f%%\n",Nmet,N,(100.0*Nmet)/N);
   printf("Nhscp=%8i Nhlt=%8i NTriggeredEvents=%8i\n",Nhscp ,Nhlt, Ntriggered);


   OutputHisto->Write();
   OutputHisto->Close();

}

