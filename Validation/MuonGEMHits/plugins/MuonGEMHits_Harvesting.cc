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
#include "Validation/MuonGEMHits/plugins/MuonGEMHits_Harvesting.h"


MuonGEMHits_Harvesting::MuonGEMHits_Harvesting(const edm::ParameterSet& ps)
{
  dbe_ = edm::Service<DQMStore>().operator->();
}


MuonGEMHits_Harvesting::~MuonGEMHits_Harvesting()
{
}


void
MuonGEMHits_Harvesting::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


void 
MuonGEMHits_Harvesting::beginJob()
{
}


void 
MuonGEMHits_Harvesting::endJob() 
{
}


void 
MuonGEMHits_Harvesting::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
}


void 
MuonGEMHits_Harvesting::endRun(edm::Run const&, edm::EventSetup const&)
{
  dbe_->cd();
  dbe_->setCurrentFolder("MuonGEMHitsV/GEMHitsTask");
  TH1F* track_eta[5]={nullptr};
  TH1F* track_phi[5]={nullptr};
  TH1F* gem_lx_even[5]={nullptr};
  TH1F* gem_ly_even[5]={nullptr};
  TH1F* gem_lx_odd[5]={nullptr};
  TH1F* gem_ly_odd[5]={nullptr};
  if ( dbe_->get("MuonGEMHitsV/GEMHitsTask/track_eta")!=nullptr ) {
    track_eta[0] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/track_eta")->getTH1F()->Clone(); 
    track_eta[1] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/track_eta_l1")->getTH1F()->Clone(); 
    track_eta[2] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/track_eta_l2")->getTH1F()->Clone(); 
    track_eta[3] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/track_eta_l1or2")->getTH1F()->Clone(); 
    track_eta[4] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/track_eta_l1and2")->getTH1F()->Clone(); 
  }
  if ( dbe_->get("MuonGEMHitsV/GEMHitsTask/track_phi")!=nullptr ) {
    track_phi[0] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/track_phi")->getTH1F()->Clone(); 
    track_phi[1] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/track_phi_l1")->getTH1F()->Clone(); 
    track_phi[2] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/track_phi_l2")->getTH1F()->Clone(); 
    track_phi[3] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/track_phi_l1or2")->getTH1F()->Clone(); 
    track_phi[4] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/track_phi_l1and2")->getTH1F()->Clone(); 
  }

  if ( dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_even")!=nullptr ) {
    gem_lx_even[0] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_even")->getTH1F()->Clone(); 
    gem_lx_even[1] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_even_l1")->getTH1F()->Clone(); 
    gem_lx_even[2] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_even_l2")->getTH1F()->Clone(); 
    gem_lx_even[3] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_even_l1or2")->getTH1F()->Clone(); 
    gem_lx_even[4] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_even_l1and2")->getTH1F()->Clone(); 
  }
  if ( dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_even")!=nullptr ) {
    gem_ly_even[0] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_even")->getTH1F()->Clone(); 
    gem_ly_even[1] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_even_l1")->getTH1F()->Clone(); 
    gem_ly_even[2] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_even_l2")->getTH1F()->Clone(); 
    gem_ly_even[3] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_even_l1or2")->getTH1F()->Clone(); 
    gem_ly_even[4] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_even_l1and2")->getTH1F()->Clone(); 
  }
  if ( dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_odd")!=nullptr ) {
    gem_lx_odd[0] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_odd")->getTH1F()->Clone(); 
    gem_lx_odd[1] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_odd_l1")->getTH1F()->Clone(); 
    gem_lx_odd[2] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_odd_l2")->getTH1F()->Clone(); 
    gem_lx_odd[3] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_odd_l1or2")->getTH1F()->Clone(); 
    gem_lx_odd[4] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_lx_odd_l1and2")->getTH1F()->Clone(); 
  }
  if ( dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_odd")!=nullptr ) {
    gem_ly_odd[0] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_odd")->getTH1F()->Clone(); 
    gem_ly_odd[1] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_odd_l1")->getTH1F()->Clone(); 
    gem_ly_odd[2] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_odd_l2")->getTH1F()->Clone(); 
    gem_ly_odd[3] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_odd_l1or2")->getTH1F()->Clone(); 
    gem_ly_odd[4] = (TH1F*)dbe_->get("MuonGEMHitsV/GEMHitsTask/gem_ly_odd_l1and2")->getTH1F()->Clone(); 
  }
  if( track_eta[0] !=nullptr && track_phi[0] != nullptr && gem_lx_even[0] != nullptr && gem_ly_even[0] != nullptr && gem_lx_odd[0] != nullptr && gem_ly_odd[0] != nullptr  ) {
    for ( int i= 0; i<5 ; i++) {
      track_eta[i]->Sumw2();
      track_phi[i]->Sumw2();
      gem_lx_even[i]->Sumw2();
      gem_ly_even[i]->Sumw2();
      gem_lx_odd[i]->Sumw2();
      gem_ly_odd[i]->Sumw2();
    }
    for ( int i= 1; i<5 ; i++) {
      track_eta[i]->Divide( track_eta[0]);
      track_phi[i]->Divide( track_phi[0]);
      gem_lx_even[i]->Divide( gem_lx_even[0]);
      gem_ly_even[i]->Divide( gem_ly_even[0]);
      gem_lx_odd[i]->Divide( gem_lx_odd[0]); 
      gem_ly_odd[i]->Divide( gem_ly_odd[0]);
 
      dbe_->book1D( TString::Format("%s%s","eff_",track_eta[i]->GetName()),track_eta[i]);    
      dbe_->book1D( TString::Format("%s%s","eff_",track_phi[i]->GetName()),track_phi[i]);    
      dbe_->book1D( TString::Format("%s%s","eff_",gem_lx_even[i]->GetName()),gem_lx_even[i]);    
      dbe_->book1D( TString::Format("%s%s","eff_",gem_ly_even[i]->GetName()),gem_ly_even[i]);    
      dbe_->book1D( TString::Format("%s%s","eff_",gem_lx_odd[i]->GetName()),gem_lx_odd[i]);    
      dbe_->book1D( TString::Format("%s%s","eff_",gem_ly_odd[i]->GetName()),gem_ly_odd[i]);    
    }
  }
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void
MuonGEMHits_Harvesting::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGEMHits_Harvesting);
