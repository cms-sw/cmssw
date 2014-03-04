// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TTree.h"
#include "TFile.h"
#include "TGraphAsymmErrors.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

///Data Format
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

///Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DQMServices/Core/interface/DQMStore.h"

///Log messages
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Validation/MuonGEMHits/interface/SimTrackMatchManager.h"
#include "Validation/MuonGEMDigis/plugins/MuonGEMDigis_Harvesting.h"


MuonGEMDigis_Harvesting::MuonGEMDigis_Harvesting(const edm::ParameterSet& ps)
{
  dbe_ = edm::Service<DQMStore>().operator->();
}


MuonGEMDigis_Harvesting::~MuonGEMDigis_Harvesting()
{
}


void
MuonGEMDigis_Harvesting::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


void 
MuonGEMDigis_Harvesting::beginJob()
{
}


void 
MuonGEMDigis_Harvesting::endJob() 
{
}


void 
MuonGEMDigis_Harvesting::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
}


void 
MuonGEMDigis_Harvesting::endRun(edm::Run const&, edm::EventSetup const&)
{
  dbe_->cd();
  dbe_->setCurrentFolder("MuonGEMDigisV/GEMDigiTask");
  
  std::vector< std::string > postfix;
  postfix.push_back("_l1");
  postfix.push_back("_l2");
  postfix.push_back("_l1or2");
  postfix.push_back("_l1and2");
  TH1F* gem_trk_phi = nullptr;  
  TH1F* gem_trk_eta = nullptr;
  TH1F* sh_eta[4]={nullptr};
  TH1F* sh_phi[4]={nullptr};
  
  if ( dbe_->get("MuonGEMDigisV/GEMDigiTask/track_phi") != nullptr && dbe_->get("MuonGEMDigisV/GEMDigiTask/track_eta") !=nullptr ) {
    gem_trk_phi = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/track_phi")->getTH1F()->Clone();
    gem_trk_eta = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/track_eta")->getTH1F()->Clone();
    gem_trk_phi->Sumw2();
    gem_trk_eta->Sumw2();
  }
  
  if( dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_sh_eta"+postfix[0]) !=nullptr && dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_sh_phi"+postfix[0]) !=nullptr ) {
    for ( int i =0 ; i< 4 ; i++) {
      sh_eta[i] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_sh_eta"+postfix[i])->getTH1F()->Clone();
      sh_eta[i]->Sumw2();
      sh_phi[i] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_sh_phi"+postfix[i])->getTH1F()->Clone();
      sh_phi[i]->Sumw2();
    }

  }
  if( dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_eta"+postfix[0]) !=nullptr && sh_eta[0] !=nullptr && gem_trk_eta !=nullptr ) {
    for ( int i =0 ; i< 4 ; i++) {
      TH1F* dg_eta =    (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_eta"+postfix[i])->getTH1F()->Clone();
      TH1F* dg_eta_2 =  (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_eta"+postfix[i])->getTH1F()->Clone();
      dg_eta->Sumw2();
      dg_eta_2->Sumw2();
      dg_eta->Divide(gem_trk_eta);
      dg_eta_2->Divide(sh_eta[i]);
      dbe_->book1D( TString::Format("eff_%s",dg_eta->GetName()),dg_eta); 
      dbe_->book1D( TString::Format("eff_%s_with_sh",dg_eta_2->GetName()),dg_eta_2); 
    }
  }
  if( dbe_->get("MuonGEMDigisV/GEMDigiTask/pad_eta"+postfix[0]) !=nullptr && sh_eta[0] !=nullptr && gem_trk_eta !=nullptr )  {
    for ( int i =0 ; i< 4 ; i++) {
      TH1F* pad_eta;
      TH1F* pad_eta_2;
      if ( i <3) { 
        pad_eta = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/pad_eta"+postfix[i])->getTH1F()->Clone();
        pad_eta_2 = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/pad_eta"+postfix[i])->getTH1F()->Clone();
      }
      else       {
        pad_eta = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/copad_eta")->getTH1F()->Clone();
        pad_eta_2 = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/copad_eta")->getTH1F()->Clone();
      }
      pad_eta->Sumw2();
      pad_eta_2->Sumw2();
      gem_trk_eta = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/track_eta")->getTH1F()->Clone();
      
      pad_eta->Divide(gem_trk_eta);
      pad_eta_2->Divide(sh_eta[i]);
      dbe_->book1D( TString::Format("eff_%s",pad_eta->GetName()),pad_eta); 
      dbe_->book1D( TString::Format("eff_%s_with_sh",pad_eta_2->GetName()),pad_eta_2); 
    }
  }

  if( dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_phi"+postfix[0]) !=nullptr && sh_eta[0] !=nullptr && gem_trk_phi !=nullptr) {
    for ( int i =0 ; i< 4 ; i++) {
      TH1F* dg_phi =    (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_phi"+postfix[i])->getTH1F()->Clone();
      TH1F* dg_phi_2 =  (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_phi"+postfix[i])->getTH1F()->Clone();
      dg_phi->Sumw2();
      dg_phi_2->Sumw2();
      dg_phi->Divide(gem_trk_phi);
      dg_phi_2->Divide(sh_phi[i]);
      dbe_->book1D( TString::Format("eff_%s",dg_phi->GetName()),dg_phi); 
      dbe_->book1D( TString::Format("eff_%s_with_sh",dg_phi_2->GetName()),dg_phi_2); 
    }
  }
  if( dbe_->get("MuonGEMDigisV/GEMDigiTask/pad_phi"+postfix[0]) !=nullptr && sh_phi[0] !=nullptr && gem_trk_phi !=nullptr)  {
    for ( int i =0 ; i< 4 ; i++) {
      TH1F* pad_phi;
      TH1F* pad_phi_2;
      if ( i <3) { 
        pad_phi = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/pad_phi"+postfix[i])->getTH1F()->Clone();
        pad_phi_2 = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/pad_phi"+postfix[i])->getTH1F()->Clone();
      }
      else       {
        pad_phi = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/copad_phi")->getTH1F()->Clone();
        pad_phi_2 = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/copad_phi")->getTH1F()->Clone();
      }
      pad_phi->Sumw2();
      pad_phi_2->Sumw2();
      pad_phi->Divide(gem_trk_phi);
      pad_phi_2->Divide(sh_phi[i]);
      dbe_->book1D( TString::Format("eff_%s",pad_phi->GetName()),pad_phi); 
      dbe_->book1D( TString::Format("eff_%s_with_sh",pad_phi_2->GetName()),pad_phi_2); 
    }
  }

  if( dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_even") != nullptr) { 

    TH1F* gem_dg_lx_even[5];
    gem_dg_lx_even[0] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_even")->getTH1F()->Clone(); 
    gem_dg_lx_even[1] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_even_l1")->getTH1F()->Clone();
    gem_dg_lx_even[2] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_even_l2")->getTH1F()->Clone();
    gem_dg_lx_even[3] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_even_l1or2")->getTH1F()->Clone();
    gem_dg_lx_even[4] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_even_l1and2")->getTH1F()->Clone();
 
    TH1F* gem_dg_ly_even[5];
    gem_dg_ly_even[0] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_ly_even")->getTH1F()->Clone(); 
    gem_dg_ly_even[1] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_ly_even_l1")->getTH1F()->Clone();
    gem_dg_ly_even[2] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_ly_even_l2")->getTH1F()->Clone();
    gem_dg_ly_even[3] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_ly_even_l1or2")->getTH1F()->Clone();
    gem_dg_ly_even[4] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_ly_even_l1and2")->getTH1F()->Clone();
 
    TH1F* gem_dg_lx_odd[5];
    gem_dg_lx_odd[0] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_odd")->getTH1F()->Clone(); 
    gem_dg_lx_odd[1] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_odd_l1")->getTH1F()->Clone();
    gem_dg_lx_odd[2] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_odd_l2")->getTH1F()->Clone();
    gem_dg_lx_odd[3] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_odd_l1or2")->getTH1F()->Clone();
    gem_dg_lx_odd[4] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_lx_odd_l1and2")->getTH1F()->Clone();
 
    TH1F* gem_dg_ly_odd[5];
    gem_dg_ly_odd[0] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_ly_odd")->getTH1F()->Clone(); 
    gem_dg_ly_odd[1] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_ly_odd_l1")->getTH1F()->Clone();
    gem_dg_ly_odd[2] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_ly_odd_l2")->getTH1F()->Clone();
    gem_dg_ly_odd[3] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_ly_odd_l1or2")->getTH1F()->Clone();
    gem_dg_ly_odd[4] = (TH1F*)dbe_->get("MuonGEMDigisV/GEMDigiTask/dg_ly_odd_l1and2")->getTH1F()->Clone();
 
    for ( int i= 0; i<5 ; i++) {
      gem_dg_lx_even[i]->Sumw2(); 
      gem_dg_ly_even[i]->Sumw2(); 
      gem_dg_lx_odd[i]->Sumw2(); 
      gem_dg_ly_odd[i]->Sumw2(); 
 
    }
    for( int i=1 ; i<5 ; i++) {
      gem_dg_lx_even[i]->Divide( gem_dg_lx_even[0]);
      gem_dg_ly_even[i]->Divide( gem_dg_ly_even[0]);
      gem_dg_lx_odd[i]->Divide( gem_dg_lx_odd[0]);
      gem_dg_ly_odd[i]->Divide( gem_dg_ly_odd[0]);
      
      dbe_->book1D( TString::Format("%s%s","eff_",gem_dg_lx_even[i]->GetName()),gem_dg_lx_even[i] ); 
      dbe_->book1D( TString::Format("%s%s","eff_",gem_dg_ly_even[i]->GetName()),gem_dg_ly_even[i] ); 
      dbe_->book1D( TString::Format("%s%s","eff_",gem_dg_lx_odd[i]->GetName()),gem_dg_lx_odd[i] ); 
      dbe_->book1D( TString::Format("%s%s","eff_",gem_dg_ly_odd[i]->GetName()),gem_dg_ly_odd[i] ); 
    }
  }
}


void
MuonGEMDigis_Harvesting::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGEMDigis_Harvesting);
