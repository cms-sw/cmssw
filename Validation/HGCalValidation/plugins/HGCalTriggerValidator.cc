// system include files
#include <memory>
#include <iostream>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/Provenance.h"

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"

//
// class declaration
//

class HGCalTriggerValidator : public DQMEDAnalyzer {

  public:
    explicit HGCalTriggerValidator(const edm::ParameterSet&);
    ~HGCalTriggerValidator() override;

    void analyze(const edm::Event&, const edm::EventSetup&) override;

  protected:
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  private:
// ----------member data ---------------------------
    //histogram tc related
    MonitorElement *h_tc_n_;
    MonitorElement *h_tc_mipPt_;
    MonitorElement *h_tc_pt_;
    MonitorElement *h_tc_energy_; 
    MonitorElement *h_tc_eta_;
    MonitorElement *h_tc_phi_;
    MonitorElement *h_tc_x_;
    MonitorElement *h_tc_y_; 
    MonitorElement *h_tc_z_;
    MonitorElement *h_tc_layer_;

    //histogram cl related
    MonitorElement *h_cl_n_;
    MonitorElement *h_cl_mipPt_;
    MonitorElement *h_cl_pt_; 
    MonitorElement *h_cl_energy_; 
    MonitorElement *h_cl_eta_;
    MonitorElement *h_cl_phi_;
    MonitorElement *h_cl_layer_;
    MonitorElement *h_cl_cells_n_;

    //histogram multicl related
    MonitorElement *h_cl3d_n_;
    MonitorElement *h_cl3d_pt_;
    MonitorElement *h_cl3d_energy_;
    MonitorElement *h_cl3d_eta_;
    MonitorElement *h_cl3d_phi_;
    MonitorElement *h_cl3d_clusters_n_;
    // cluster shower shapes
    MonitorElement *h_cl3d_showerlength_;
    MonitorElement *h_cl3d_coreshowerlength_;
    MonitorElement *h_cl3d_firstlayer_;
    MonitorElement *h_cl3d_maxlayer_;
    MonitorElement *h_cl3d_seetot_;
    MonitorElement *h_cl3d_seemax_;
    MonitorElement *h_cl3d_spptot_;
    MonitorElement *h_cl3d_sppmax_;
    MonitorElement *h_cl3d_szz_;
    MonitorElement *h_cl3d_srrtot_;
    MonitorElement *h_cl3d_srrmax_;
    MonitorElement *h_cl3d_srrmean_;
    MonitorElement *h_cl3d_emaxe_;
    MonitorElement *h_cl3d_bdteg_;
    MonitorElement *h_cl3d_quality_;

    //histogram tower related
    MonitorElement *h_tower_n_;
    MonitorElement *h_tower_pt_;
    MonitorElement *h_tower_energy_;
    MonitorElement *h_tower_eta_;
    MonitorElement *h_tower_phi_;
    MonitorElement *h_tower_etEm_;
    MonitorElement *h_tower_etHad_;
    MonitorElement *h_tower_iEta_;
    MonitorElement *h_tower_iPhi_;


    edm::EDGetToken trigger_cells_token_;
    edm::EDGetToken clusters_token_;
    edm::EDGetToken multiclusters_token_; 
    edm::EDGetToken towers_token_;
   
    std::unique_ptr<HGCalTriggerClusterIdentificationBase> id_;
   
    HGCalTriggerTools triggerTools_;
};

HGCalTriggerValidator::HGCalTriggerValidator(const edm::ParameterSet& iConfig){

  trigger_cells_token_ = consumes<l1t::HGCalTriggerCellBxCollection>(iConfig.getParameter<edm::InputTag>("TriggerCells"));
  clusters_token_ = consumes<l1t::HGCalClusterBxCollection>(iConfig.getParameter<edm::InputTag>("Clusters"));
  multiclusters_token_ = consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("Multiclusters"));
  towers_token_ = consumes<l1t::HGCalTowerBxCollection>(iConfig.getParameter<edm::InputTag>("Towers"));
  
  id_.reset( HGCalTriggerClusterIdentificationFactory::get()->create("HGCalTriggerClusterIdentificationBDT") );
  id_->initialize(iConfig.getParameter<edm::ParameterSet>("EGIdentification"));

}

HGCalTriggerValidator::~HGCalTriggerValidator(){
}

void HGCalTriggerValidator::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, edm::EventSetup const &) {
    
   iBooker.setCurrentFolder("HGCALTPG");
   
  //initiating histograms
  // trigger cells
  h_tc_n_ = iBooker.book1D("tc_n","trigger cell number; number", 400, 0, 400);
  h_tc_mipPt_ = iBooker.book1D("tc_mipPt","trigger cell mipPt; mipPt", 400, 0, 400);
  h_tc_pt_ = iBooker.book1D("tc_pt","trigger cell pt; pt [GeV]", 15, 0, 15);
  h_tc_energy_ = iBooker.book1D("tc_energy","trigger cell energy; energy [GeV]", 70, 0, 70);
  h_tc_eta_ = iBooker.book1D("tc_eta","trigger cell eta; eta", 60, -3.14, 3.14);
  h_tc_phi_ = iBooker.book1D("tc_phi","trigger cell phi; phi", 60, -3.14, 3.14);
  h_tc_x_ = iBooker.book1D("tc_x","trigger cell x; x [cm]", 500, -250, 250);
  h_tc_y_ = iBooker.book1D("tc_y","trigger cell y; y [cm]", 500, -250, 250); 
  h_tc_z_ = iBooker.book1D("tc_z","trigger cell z; z [cm]", 1100, -550, 550);
  h_tc_layer_ = iBooker.book1D("tc_layer","trigger cell layer; layer", 50, 0, 50);

  // cluster 2D histograms
  h_cl_n_ = iBooker.book1D("cl_n","cluster2D number; number",80, 0, 80);
  h_cl_mipPt_ = iBooker.book1D("cl_mipPt","cluster2D mipPt; mipPt",600, 0, 600);
  h_cl_pt_ = iBooker.book1D("cl_pt","cluster2D pt; pt [GeV]",20, 0, 20);
  h_cl_energy_= iBooker.book1D("cl_energy","cluster2D energy; energy [GeV]", 80, 0, 80);
  h_cl_eta_ = iBooker.book1D("cl_eta","cluster2D eta; eta", 60, -3.14, 3.14);
  h_cl_phi_ = iBooker.book1D("cl_phi","cluster2D phi; phi", 60, -3.14, 3.14);
  h_cl_cells_n_ = iBooker.book1D("cl_cells_n","cluster2D cells_n; cells_n", 16, 0, 16);
  h_cl_layer_ = iBooker.book1D("cl_layer","cluster2D layer; layer", 50, 0, 50);
    
  // multiclusters
  h_cl3d_n_ = iBooker.book1D("cl3d_n","cl3duster3D number; number", 12, 0, 12);
  h_cl3d_pt_ = iBooker.book1D("cl3d_pt","cl3duster3D pt; pt [GeV]", 50, 0, 50);
  h_cl3d_energy_ = iBooker.book1D("cl3d_energy","cl3duster3D energy; energy [GeV]", 80, 0, 80);
  h_cl3d_eta_ = iBooker.book1D("cl3d_eta","cl3duster3D eta; eta", 60, -3.14, 3.14);
  h_cl3d_phi_ = iBooker.book1D("cl3d_phi","cl3duster3D phi; phi", 60, -3.14, 3.14);
  h_cl3d_clusters_n_ = iBooker.book1D("cl3d_clusters_n","cl3duster3D clusters_n; clusters_n", 30, 0, 30);
  // cluster shower shapes
  h_cl3d_showerlength_ = iBooker.book1D("cl3d_showerlength","cl3duster3D showerlength; showerlength", 50, 0, 50);
  h_cl3d_coreshowerlength_ = iBooker.book1D("cl3d_coreshowerlength","cl3duster3D coreshowerlength; coreshowerlength", 16, 0, 16);
  h_cl3d_firstlayer_ = iBooker.book1D("cl3d_firstlayer","cl3duster3D firstlayer; firstlayer", 50, 0, 50);
  h_cl3d_maxlayer_ = iBooker.book1D("cl3d_maxlayer","cl3duster3D maxlayer; maxlayer", 50, 0, 50);
  h_cl3d_seetot_ = iBooker.book1D("cl3d_seetot","cl3duster3D seetot; seetot", 50, 0, 0.05);
  h_cl3d_seemax_ = iBooker.book1D("cl3d_seemax","cl3duster3D seemax; seemax", 40, 0, 0.04);
  h_cl3d_spptot_ = iBooker.book1D("cl3d_spptot","cl3duster3D spptot; spptot", 800, 0, 0.08);
  h_cl3d_sppmax_ = iBooker.book1D("cl3d_sppmax","cl3duster3D sppmax; sppmax", 800, 0, 0.08);
  h_cl3d_szz_ = iBooker.book1D("cl3d_szz","cl3duster3D szz; szz", 50, 0, 50);
  h_cl3d_srrtot_ = iBooker.book1D("cl3d_srrtot","cl3duster3D srrtot; srrtot", 800, 0, 0.008);
  h_cl3d_srrmax_ = iBooker.book1D("cl3d_srrmax","cl3duster3D srrmax; srrmax", 900, 0, 0.009);
  h_cl3d_srrmean_ = iBooker.book1D("cl3d_srrmean","cl3duster3D srrmean; srrmean", 800, 0, 0.008);
  h_cl3d_emaxe_ = iBooker.book1D("cl3d_emaxe","cl3duster3D emaxe; emaxe", 15, 0, 1.5);
  h_cl3d_bdteg_ = iBooker.book1D("cl3d_bdteg","cl3duster3D bdteg; bdteg", 30, -0.7, 0.4);
  h_cl3d_quality_ = iBooker.book1D("cl3d_quality","cl3duster3D quality; quality", 20, 0, 2);

  // towers
  h_tower_n_ = iBooker.book1D("tower_n","tower n; number", 400, 1200, 1600);
  h_tower_pt_ = iBooker.book1D("tower_pt","tower pt; pt [GeV]", 50, 0, 50);
  h_tower_energy_ = iBooker.book1D("tower_energy","tower energy; energy [GeV]", 200, 0, 200);
  h_tower_eta_ = iBooker.book1D("tower_eta","tower eta; eta", 60, -3.14, 3.14); 
  h_tower_phi_ = iBooker.book1D("tower_phi","tower phi; phi", 60, -3.14, 3.14);
  h_tower_etEm_ = iBooker.book1D("tower_etEm","tower etEm; etEm",  50, 0, 50);
  h_tower_etHad_ = iBooker.book1D("tower_etHad","tower etHad; etHad", 30, 0, 0.3);
  h_tower_iEta_ = iBooker.book1D("tower_iEta","tower iEta; iEta", 20, 0, 20);
  h_tower_iPhi_ = iBooker.book1D("tower_iPhi","tower iPhi; iPhi", 80, 0, 80);
}

void HGCalTriggerValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  int tc_n = 0;
  int cl_n = 0;
  int cl3d_n = 0;
  int tower_n = 0;
  
  triggerTools_.eventSetup(iSetup);

  // retrieve trigger cells
  edm::Handle<l1t::HGCalTriggerCellBxCollection> trigger_cells_h;
  iEvent.getByToken(trigger_cells_token_, trigger_cells_h);
  const l1t::HGCalTriggerCellBxCollection& trigger_cells = *trigger_cells_h;

  if (trigger_cells_h.isValid()){
    for(auto tc_itr=trigger_cells.begin(0); tc_itr!=trigger_cells.end(0); tc_itr++)
    {
      tc_n++;
      HGCalDetId id(tc_itr->detId());
      h_tc_pt_->Fill(tc_itr->pt());
      h_tc_mipPt_->Fill(tc_itr->mipPt());
      h_tc_energy_->Fill(tc_itr->energy());
      h_tc_eta_->Fill(tc_itr->eta());
      h_tc_phi_->Fill(tc_itr->phi());
      h_tc_x_->Fill(tc_itr->position().x());
      h_tc_y_->Fill(tc_itr->position().y()); 
      h_tc_z_->Fill(tc_itr->position().z());
      h_tc_layer_->Fill(triggerTools_.layerWithOffset(id));
    }
  }
  h_tc_n_->Fill(tc_n);

  // retrieve clusters
  edm::Handle<l1t::HGCalClusterBxCollection> clusters_h;
  iEvent.getByToken(clusters_token_, clusters_h);
  const l1t::HGCalClusterBxCollection& clusters = *clusters_h;

  if (clusters_h.isValid()){
    for(auto cl_itr=clusters.begin(0); cl_itr!=clusters.end(0); cl_itr++){
      cl_n++;
      h_cl_mipPt_->Fill(cl_itr->mipPt());
      h_cl_pt_->Fill(cl_itr->pt());
      h_cl_energy_->Fill(cl_itr->energy());
      h_cl_eta_->Fill(cl_itr->eta());
      h_cl_phi_->Fill(cl_itr->phi());
      h_cl_layer_->Fill(triggerTools_.layerWithOffset(cl_itr->detId()));
      h_cl_cells_n_->Fill(cl_itr->constituents().size());
    }
  }
  h_cl_n_->Fill(cl_n);

  // retrieve clusters 3D
  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters_h;
  iEvent.getByToken(multiclusters_token_, multiclusters_h);
  const l1t::HGCalMulticlusterBxCollection& multiclusters = *multiclusters_h;

  if (multiclusters_h.isValid()){
       
    for(auto cl3d_itr=multiclusters.begin(0); cl3d_itr!=multiclusters.end(0); cl3d_itr++){
      
      cl3d_n++;
      h_cl3d_pt_->Fill(cl3d_itr->pt());
      h_cl3d_energy_->Fill(cl3d_itr->energy());
      h_cl3d_eta_->Fill(cl3d_itr->eta());
      h_cl3d_phi_->Fill(cl3d_itr->phi());
      h_cl3d_clusters_n_->Fill(cl3d_itr->constituents().size());
      // cluster shower shapes
      h_cl3d_showerlength_->Fill(cl3d_itr->showerLength());
      h_cl3d_coreshowerlength_->Fill(cl3d_itr->coreShowerLength());
      h_cl3d_firstlayer_->Fill(cl3d_itr->firstLayer());
      h_cl3d_maxlayer_->Fill(cl3d_itr->maxLayer());
      h_cl3d_seetot_->Fill(cl3d_itr->sigmaEtaEtaTot());
      h_cl3d_seemax_->Fill(cl3d_itr->sigmaEtaEtaMax());
      h_cl3d_spptot_->Fill(cl3d_itr->sigmaPhiPhiTot());
      h_cl3d_sppmax_->Fill(cl3d_itr->sigmaPhiPhiMax());
      h_cl3d_szz_->Fill(cl3d_itr->sigmaZZ());
      h_cl3d_srrtot_->Fill(cl3d_itr->sigmaRRTot());
      h_cl3d_srrmax_->Fill(cl3d_itr->sigmaRRMax());
      h_cl3d_srrmean_->Fill(cl3d_itr->sigmaRRMean());
      h_cl3d_emaxe_->Fill(cl3d_itr->eMax()/cl3d_itr->energy());
      h_cl3d_bdteg_->Fill(id_->value(*cl3d_itr));
      h_cl3d_quality_->Fill(cl3d_itr->hwQual());
    }
  }
  h_cl3d_n_->Fill(cl3d_n);

  // retrieve towers
  edm::Handle<l1t::HGCalTowerBxCollection> towers_h;
  iEvent.getByToken(towers_token_, towers_h);
  const l1t::HGCalTowerBxCollection& towers = *towers_h;

  if (towers_h.isValid()){
    for(auto tower_itr=towers.begin(0); tower_itr!=towers.end(0); tower_itr++){
      tower_n++;
      h_tower_pt_->Fill(tower_itr->pt());
      h_tower_energy_->Fill(tower_itr->energy());
      h_tower_eta_->Fill(tower_itr->eta());
      h_tower_phi_->Fill(tower_itr->phi());
      h_tower_etEm_->Fill(tower_itr->etEm());
      h_tower_etHad_->Fill(tower_itr->etHad());
      h_tower_iEta_->Fill(tower_itr->id().iEta());
      h_tower_iPhi_->Fill(tower_itr->id().iPhi());
    }
  }
  h_tower_n_->Fill(tower_n);

}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTriggerValidator);
