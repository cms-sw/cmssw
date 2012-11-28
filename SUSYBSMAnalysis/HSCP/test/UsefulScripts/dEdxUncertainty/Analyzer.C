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
namespace susybsm { class HSCParticle; class HSCPIsolation; class HSCPDeDxInfo;}
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
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPDeDxInfo.h"
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

#define FWLITE
#include "../../ICHEP_Analysis/Analysis_Global.h"
#include "../../ICHEP_Analysis/Analysis_PlotFunction.h"
#include "../../ICHEP_Analysis/Analysis_Samples.h"
#include "../../ICHEP_Analysis/Analysis_CommonFunction.h"

#endif

void Analyzer(string MODE="COMPILE")
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

   system("mkdir pictures/");
   string saveName = "dedx";
   TFile* OutputHisto = new TFile((string("pictures/") + "/Histos.root").c_str(),"RECREATE");
   TH1D* HdedxMIP          = new TH1D(    (saveName + "_MIP"    ).c_str(), "MIP"    ,  1000, 0, 10);
   TH2D* HdedxVsP          = new TH2D(    (saveName + "_dedxVsP").c_str(), "dedxVsP", 3000, 0, 30,1500,0,15);
   TH2D* HdedxVsPM         = new TH2D(    (saveName + "_dedxVsPM").c_str(), "dedxVsPM", 3000, 0, 30,1500,0,15);
   TH2D* HdedxVsQP         = new TH2D(    (saveName + "_dedxVsQP").c_str(), "dedxVsQP", 6000, -30, 30,1500,0,25);
   TProfile* HdedxVsPProfile   = new TProfile((saveName + "_Profile").c_str(), "Profile",  100, 0,100);
   TProfile* HdedxVsEtaProfile = new TProfile((saveName + "_Eta"    ).c_str(), "Eta"    ,  100,-3,  3);
   TH2D* HdedxVsEta        = new TH2D    ((saveName + "_Eta2D"  ).c_str(), "Eta"    ,  100,-3,  3, 1000,0,5);
   TProfile* HNOSVsEtaProfile  = new TProfile((saveName + "_NOS"    ).c_str(), "NOS"    ,  100,-3,  3);
   TProfile* HNOMVsEtaProfile  = new TProfile((saveName + "_NOM"    ).c_str(), "NOM"    ,  100,-3,  3);
   TProfile* HNOMSVsEtaProfile = new TProfile((saveName + "_NOMS"    ).c_str(), "NOMS"    ,  100,-3,  3);
   TH1D* HMass             = new TH1D(    (saveName + "_Mass"   ).c_str(), "Mass"   ,  500, 0, 10);
   TH1D* HP                = new TH1D(    (saveName + "_P"      ).c_str(), "P"      ,  500, 0, 100);

   TH1D* HHit          = new TH1D(    (saveName + "_Hit"      ).c_str(), "P"      ,  600, 0, 6);

   TH2D* HIasVsP          = new TH2D(    (saveName + "_IasVsP").c_str(), "IasVsP", 3000, 0, 30,1500,0,1);
   TH2D* HIasVsPM         = new TH2D(    (saveName + "_IasVsPM").c_str(), "IasVsPM", 3000, 0, 30,1500,0,1);
   TH1D* HIasMIP          = new TH1D(    (saveName + "_IasMIP"    ).c_str(), "IasMIP"    ,  1000, 0, 1);

   reco::DeDxData* emptyDeDx = new reco::DeDxData(0,0,0);
//   TH3F* dEdxTemplates = loadDeDxTemplate("../../../data/Data7TeV_Deco_SiStripDeDxMip_3D_Rcd.root");
//   TH3F* dEdxTemplates = loadDeDxTemplate("../../../data/MC7TeV_Deco_SiStripDeDxMip_3D_Rcd.root");
//   TH3F* dEdxTemplates = loadDeDxTemplate("../../../data/Discrim_Templates_MC_2012b.root");
   TH3F* dEdxTemplates = loadDeDxTemplate("../../../data/Discrim_Templates_MC_2012_new2.root");
   double SF = 1.05; //0.9;

   std::vector<string> FileName;
//   FileName.push_back("root://eoscms//eos/cms/store/cmst3/user/querten/12_11_19_MC_dEdx/dedx_data_2012.root");
   FileName.push_back("root://eoscms//eos/cms/store/cmst3/user/querten/12_11_19_MC_dEdx/dedx_mc_2012.root");
   fwlite::ChainEvent ev(FileName);

   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on Tree              :");
   int TreeStep = ev.size()/50;if(TreeStep==0)TreeStep=1;
   for(Long64_t e=0;e<ev.size();e++){
//      if(e>5000)break;
      ev.to(e); 
      if(e%TreeStep==0){printf(".");fflush(stdout);}

      fwlite::Handle< std::vector<reco::Track> > trackCollHandle;
      trackCollHandle.getByLabel(ev,"TrackRefitter");
      if(!trackCollHandle.isValid()){printf("Invalid trackCollHandle\n");continue;}

      for(unsigned int c=0;c<trackCollHandle->size();c++){
          reco::TrackRef track = reco::TrackRef( trackCollHandle.product(), c );       

          if(track->chi2()/track->ndof()>5 )continue;  //WAS >1
          if(track->found()<8)continue;
///////////////

     fwlite::Handle<HSCPDeDxInfoValueMap> dEdxHitsH;
     dEdxHitsH.getByLabel(ev, "dedxHitInfo");
     if(!dEdxHitsH.isValid()){printf("Invalid dEdxHitInfo\n");return;}
     const ValueMap<HSCPDeDxInfo>& dEdxHitMap = *dEdxHitsH.product();
     const HSCPDeDxInfo& hscpHitsInfo = dEdxHitMap.get((size_t)track.key());

     for(unsigned int h=0;h<hscpHitsInfo.charge.size();h++){
        DetId detid(hscpHitsInfo.detIds[h]);
        if(detid.subdetId()<3)continue; // skip pixels
        if(!hscpHitsInfo.shapetest[h])continue;

        double Norm = (detid.subdetId()<3)?3.61e-06:3.61e-06*265;
        Norm*=10.0; //mm --> cm
        Norm*=SF;
        HHit->Fill(Norm * hscpHitsInfo.charge[h]/hscpHitsInfo.pathlength[h]);

//        vect_charge.push_back(Norm*hscpHitsInfo.charge[h]/hscpHitsInfo.pathlength[h]);
     }
//////////////

          reco::DeDxData* dedxObj = dEdxEstimOnTheFly(ev, track, emptyDeDx, SF);
          reco::DeDxData* dedxIasObj = dEdxOnTheFly(ev, track, emptyDeDx, SF, dEdxTemplates, false);


          if(track->pt()>20 && track->pt()<40 && dedxObj->numberOfMeasurements()>6 ){
            HdedxVsEtaProfile->Fill(track->eta(), dedxObj->dEdx() );
            HdedxVsEta->Fill(track->eta(), dedxObj->dEdx() );
            HNOMVsEtaProfile->Fill(track->eta(),dedxObj->numberOfMeasurements() );
            HNOSVsEtaProfile->Fill(track->eta(),dedxObj->numberOfSaturatedMeasurements() );
            HNOMSVsEtaProfile->Fill(track->eta(),dedxObj->numberOfMeasurements() - dedxObj->numberOfSaturatedMeasurements() );
          }
          if(fabs(track->eta())>2.1)continue;
          if((int)dedxObj->numberOfMeasurements()<10)continue;
          if(track->p()>5 && track->p()<40){
             HdedxMIP->Fill(dedxObj->dEdx());
             HP->Fill(track->p());
             HIasMIP->Fill(dedxIasObj->dEdx());
          }
          HdedxVsP ->Fill(track->p(), dedxObj->dEdx() );
          HdedxVsQP->Fill(track->p()*track->charge(), dedxObj->dEdx() );
          HIasVsP ->Fill(track->p(), dedxIasObj->dEdx() );

          if(fabs(track->eta())<0.4)HdedxVsPProfile->Fill(track->p(), dedxObj->dEdx() );
          double Mass = GetMass(track->p(),dedxObj->dEdx(), false);
          if(dedxObj->dEdx()>4.0 && track->p()<3.0){
             HMass->Fill(Mass);
             if(isnan((float)Mass) || Mass<0.94-0.3 || Mass>0.94+0.3)continue;
             HdedxVsPM ->Fill(track->p(), dedxObj->dEdx() );
             HIasVsPM ->Fill(track->p(), dedxIasObj->dEdx() );
          }


      }


   }


   OutputHisto->Write();
   OutputHisto->Close();  
}
