/*
 * MuSimHitOccupancy: analyzer module for SimHit level occupancy studies in muon systems
 *
 * The MuSimHitOccupancy that was used for CSC studies was taken as a base.
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"


#include "TH1.h"
#include "TH2.h"
#include "TGraphErrors.h"
#include "TTree.h"

#include "GEMCode/SimMuL1/interface/PSimHitMapCSC.h"
#include "GEMCode/SimMuL1/interface/MuGeometryHelpers.h"
#include "GEMCode/SimMuL1/interface/MuNtupleClasses.h"

#include <iomanip>

using std::cout;
using std::endl;
using namespace mugeo;

namespace
{
// constants for PDG IDs to consider
enum ENumPDG {N_PDGIDS=7};
const int pdg_ids[N_PDGIDS] = {0, 11, 13, 211, 321, 2212, 1000000000};
const std::string pdg_ids_names[N_PDGIDS]  = {"unknown", "e", "#mu", "#pi", "K", "p", "nuclei"};
const std::string pdg_ids_names_[N_PDGIDS] = {"", "e", "mu", "pi", "K", "p", "nucl"};
const int pdg_colors[N_PDGIDS] = {0, kBlack, kBlue, kGreen+1, kOrange-3, kMagenta, kRed};
const int pdg_markers[N_PDGIDS] = {0, 1, 24, 3, 5, 26, 2};
} // local namespace


// ================================================================================================
// the analyzer

class MuSimHitOccupancy : public edm::EDAnalyzer
{
public:
  explicit MuSimHitOccupancy(const edm::ParameterSet&);
  ~MuSimHitOccupancy();

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  void analyzeCSC();
  void analyzeGEM();
  void analyzeDT();
  void analyzeRPC();

  std::vector<std::vector<MyCSCSimHit> > clusterCSCHitsInLayer(std::vector<MyCSCSimHit> &hits);
  std::vector<std::vector<MyGEMSimHit> > clusterGEMHitsInPart(std::vector<MyGEMSimHit> &hits);
  std::vector<std::vector<MyRPCSimHit> > clusterRPCHitsInRoll(std::vector<MyRPCSimHit> &hits);
  std::vector<std::vector<MyDTSimHit> > clusterDTHitsInLayer(std::vector<MyDTSimHit> &hits);

private:

  // configuration parameters:

  edm::InputTag input_tag_csc_;
  edm::InputTag input_tag_gem_;
  edm::InputTag input_tag_dt_;
  edm::InputTag input_tag_rpc_;

  bool input_is_neutrons_;

  bool do_csc_;
  bool do_gem_;
  bool do_rpc_;
  bool do_dt_;

  bool fill_csc_sh_tree_;
  bool fill_gem_sh_tree_;
  bool fill_rpc_sh_tree_;
  bool fill_dt_sh_tree_;

  // misc. utilities

  const CSCGeometry* csc_geometry;
  const GEMGeometry* gem_geometry;
  const DTGeometry*  dt_geometry;
  const RPCGeometry* rpc_geometry;

  const ParticleDataTable * pdt_;

  SimHitAnalysis::PSimHitMapCSC simhit_map_csc;
  SimHitAnalysis::PSimHitMap simhit_map_gem;
  SimHitAnalysis::PSimHitMap simhit_map_rpc;
  SimHitAnalysis::PSimHitMap simhit_map_dt;

  // sensitive areas
  mugeo::MuGeometryAreas areas_;

  // some counters:

  UInt_t evtn;
  UInt_t csc_shn;
  UInt_t gem_shn;
  UInt_t rpc_shn;
  UInt_t dt_shn;

  int nevt_with_cscsh, nevt_with_cscsh_in_rpc;
  int n_cscsh, n_cscsh_in_rpc;

  int nevt_with_gemsh;
  int n_gemsh;

  int nevt_with_rpcsh, nevt_with_rpcsh_e, nevt_with_rpcsh_b;
  int n_rpcsh, n_rpcsh_e, n_rpcsh_b;

  int nevt_with_dtsh;
  int n_dtsh;

  // some histos:

  TH2D * h_csc_rz_sh_xray;
  TH2D * h_csc_rz_sh_heatmap;
  TH2D * h_csc_rz_clu_heatmap;
  TH1D * h_csc_nlayers_in_ch[CSC_TYPES+1];
  TH1D * h_csc_nevt_fraction_with_sh;
  TH1D * h_csc_hit_flux_per_layer;
  TH1D * h_csc_hit_rate_per_ch;
  TH1D * h_csc_clu_flux_per_layer;
  TH1D * h_csc_clu_rate_per_ch;

  std::map<int,int> pdg2idx;
  TH2D * h_csc_tof_vs_ekin[CSC_TYPES+1][N_PDGIDS];


  TH2D * h_gem_rz_sh_heatmap;
  TH1D * h_gem_nevt_fraction_with_sh;
  TH1D * h_gem_hit_flux_per_layer;
  TH1D * h_gem_hit_rate_per_ch;
  TH1D * h_gem_clu_flux_per_layer;
  TH1D * h_gem_clu_rate_per_ch;
  TH2D * h_gem_tof_vs_ekin[N_PDGIDS];


  TH2D * h_rpc_rz_sh_heatmap;
  TH1D * h_rpcf_nevt_fraction_with_sh;
  TH1D * h_rpcf_hit_flux_per_layer;
  TH1D * h_rpcf_hit_rate_per_ch;
  TH1D * h_rpcf_clu_flux_per_layer;
  TH1D * h_rpcf_clu_rate_per_ch;

  TH1D * h_rpcb_nevt_fraction_with_sh;
  TH1D * h_rpcb_hit_flux_per_layer;
  TH1D * h_rpcb_hit_rate_per_ch;
  TH1D * h_rpcb_clu_flux_per_layer;
  TH1D * h_rpcb_clu_rate_per_ch;

  TH2D * h_rpcb_tof_vs_ekin[N_PDGIDS];
  TH2D * h_rpcf_tof_vs_ekin[N_PDGIDS];


  TH2D * h_dt_xy_sh_heatmap;
  TH1D * h_dt_nevt_fraction_with_sh;
  TH1D * h_dt_hit_flux_per_layer;
  TH1D * h_dt_hit_rate_per_ch;
  TH2D * h_dt_tof_vs_ekin[N_PDGIDS];


  TH2D * h_mu_rz_sh_heatmap;
  TH2D * h_mu_xy_sh_heatmap;


  TGraphErrors *gr_csc_hit_flux_me1, *gr_csc_hit_flux_me2, *gr_csc_hit_flux_me3, *gr_csc_hit_flux_me4;
  TGraphErrors *gr_dt_hit_flux_mb1, *gr_dt_hit_flux_mb2, *gr_dt_hit_flux_mb3, *gr_dt_hit_flux_mb4;

  TGraphErrors *gr_gem_hit_flux_ge1;

  // some ntuples:

  void bookCSCSimHitsTrees();
  TTree* csc_ev_tree;
  TTree* csc_ch_tree;
  TTree* csc_la_tree;
  TTree* csc_cl_tree;
  TTree* csc_sh_tree;
  MyCSCDetId c_id;
  MyCSCDetId c_cid;
  MyCSCSimHit c_h;
  MyCSCCluster c_cl;
  MyCSCLayer c_la;
  MyCSCChamber c_ch;
  MyCSCEvent c_ev;


  void bookGEMSimHitsTrees();
  TTree* gem_ev_tree;
  TTree* gem_ch_tree;
  TTree* gem_part_tree;
  TTree* gem_cl_tree;
  TTree* gem_sh_tree;
  MyGEMDetId g_id;
  MyGEMSimHit g_h;
  MyGEMCluster g_cl;
  MyGEMPart g_part;
  MyGEMChamber g_ch;
  MyGEMEvent g_ev;


  void bookRPCSimHitsTrees();
  TTree* rpc_ev_tree;
  TTree* rpc_ch_tree;
  TTree* rpc_rl_tree;
  TTree* rpc_cl_tree;
  TTree* rpc_sh_tree;
  MyRPCDetId r_id;
  MyRPCSimHit r_h;
  MyRPCCluster r_cl;
  MyRPCRoll r_rl;
  MyRPCChamber r_ch;
  MyRPCEvent r_ev;


  void bookDTSimHitsTrees();
  TTree* dt_sh_tree;
  MyDTDetId d_id;
  MyDTSimHit d_h;

};


// ================================================================================================
MuSimHitOccupancy::MuSimHitOccupancy(const edm::ParameterSet& iConfig)
{
  // should be set to false if running over a regular MB sample
  input_is_neutrons_ = iConfig.getUntrackedParameter< bool >("inputIsNeutrons", true);

  edm::InputTag default_tag_csc("g4SimHits","MuonCSCHits");
  edm::InputTag default_tag_gem("g4SimHits","MuonGEMHits");
  edm::InputTag default_tag_dt("g4SimHits","MuonDTHits");
  edm::InputTag default_tag_rpc("g4SimHits","MuonRPCHits");
  if (input_is_neutrons_)
  {
    default_tag_csc = edm::InputTag("cscNeutronWriter","");
    default_tag_gem = edm::InputTag("gemNeutronWriter","");
    default_tag_dt  = edm::InputTag("dtNeutronWriter","");
    default_tag_rpc = edm::InputTag("rpcNeutronWriter","");
  }

  input_tag_csc_ = iConfig.getUntrackedParameter<edm::InputTag>("inputTagCSC", default_tag_csc);
  input_tag_gem_ = iConfig.getUntrackedParameter<edm::InputTag>("inputTagGEM", default_tag_gem);
  input_tag_dt_  = iConfig.getUntrackedParameter<edm::InputTag>("inputTagDT",  default_tag_dt);
  input_tag_rpc_ = iConfig.getUntrackedParameter<edm::InputTag>("inputTagRPC", default_tag_rpc);

  simhit_map_csc.setInputTag(input_tag_csc_);
  simhit_map_gem.setInputTag(input_tag_gem_);
  simhit_map_rpc.setInputTag(input_tag_rpc_);
  simhit_map_dt.setInputTag(input_tag_dt_);


  do_csc_ = iConfig.getUntrackedParameter< bool >("doCSC", true);
  do_gem_ = iConfig.getUntrackedParameter< bool >("doGEM", true);
  do_rpc_ = iConfig.getUntrackedParameter< bool >("doRPC", true);
  do_dt_  = iConfig.getUntrackedParameter< bool >("doDT",  true);


  fill_csc_sh_tree_ = do_csc_ && iConfig.getUntrackedParameter< bool >("fillCSCSimHitsTrees",true);
  if (fill_csc_sh_tree_) bookCSCSimHitsTrees();

  fill_gem_sh_tree_ = do_gem_ && iConfig.getUntrackedParameter< bool >("fillGEMSimHitsTrees",true);
  if (fill_gem_sh_tree_) bookGEMSimHitsTrees();

  fill_rpc_sh_tree_ = do_rpc_ && iConfig.getUntrackedParameter< bool >("fillRPCSimHitsTree",true);
  if (fill_rpc_sh_tree_) bookRPCSimHitsTrees();

  fill_dt_sh_tree_  = do_dt_ && iConfig.getUntrackedParameter< bool >("fillDTSimHitsTree",true);
  if (fill_dt_sh_tree_) bookDTSimHitsTrees();


  evtn = 0;
  nevt_with_cscsh = nevt_with_cscsh_in_rpc = n_cscsh = n_cscsh_in_rpc = 0;
  nevt_with_gemsh = n_gemsh = 0;
  nevt_with_rpcsh = nevt_with_rpcsh_e = nevt_with_rpcsh_b = n_rpcsh = n_rpcsh_e = n_rpcsh_b = 0;
  nevt_with_dtsh = n_dtsh = 0;


  edm::Service<TFileService> fs;

  std::string n_simhits = "n-SimHits";
  if (!input_is_neutrons_) n_simhits = "SimHits";
  std::string n_clusters = "n-Clusters";
  if (!input_is_neutrons_) n_clusters = "Clusters";

  h_csc_rz_sh_xray = fs->make<TH2D>("h_csc_rz_sh_xray",("CSC "+n_simhits+" #rho-z X-ray;z, cm;#rho, cm").c_str(), 2060, 550, 1080, 755, 0, 755);
  h_csc_rz_sh_heatmap = fs->make<TH2D>("h_csc_rz_sh_heatmap",("CSC "+n_simhits+" #rho-z;z, cm;#rho, cm").c_str(), 220, 541.46, 1101.46, 150, 0, 755);
  h_csc_rz_clu_heatmap = fs->make<TH2D>("h_csc_rz_clu_heatmap",("CSC "+n_clusters+" #rho-z;z, cm;#rho, cm").c_str(), 220, 541.46, 1101.46, 150, 0, 755);


  char label[200], nlabel[200];
  for (int me=0; me<=CSC_TYPES; me++)
  {
    sprintf(label,"h_csc_nlayers_in_ch_%s",csc_type_[me].c_str());
    sprintf(nlabel,"# of layers with %s: %s CSC;# layers", n_simhits.c_str(), csc_type[me].c_str());
    h_csc_nlayers_in_ch[me]  = fs->make<TH1D>(label, nlabel, 6, 0.5, 6.5);
  }

  h_csc_nevt_fraction_with_sh = fs->make<TH1D>("h_csc_nevt_fraction_with_sh",
      ("Fraction of events with "+n_simhits+" by CSC type;ME station/ring").c_str(), CSC_TYPES+1, -0.5, CSC_TYPES+0.5);
  for (int i=1; i<=h_csc_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_csc_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());

  h_csc_hit_flux_per_layer = fs->make<TH1D>("h_csc_hit_flux_per_layer",
      (n_simhits+" Flux per CSC layer at L=10^{34};ME station/ring;Hz/cm^{2}").c_str(), CSC_TYPES, 0.5,  CSC_TYPES+0.5);
  for (int i=1; i<=h_csc_hit_flux_per_layer->GetXaxis()->GetNbins();i++)
    h_csc_hit_flux_per_layer->GetXaxis()->SetBinLabel(i,csc_type[i].c_str());

  h_csc_hit_rate_per_ch = fs->make<TH1D>("h_csc_hit_rate_per_ch",
      (n_simhits+" rate per chamber in CSC at L=10^{34};ME station/ring;kHz").c_str(), CSC_TYPES, 0.5, CSC_TYPES+0.5);
  for (int i=1; i<=h_csc_hit_rate_per_ch->GetXaxis()->GetNbins();i++)
    h_csc_hit_rate_per_ch->GetXaxis()->SetBinLabel(i,csc_type[i].c_str());

  h_csc_clu_flux_per_layer = fs->make<TH1D>("h_csc_clu_flux_per_layer",
      (n_clusters+" Flux per CSC layer at L=10^{34};ME station/ring;Hz/cm^{2}").c_str(), CSC_TYPES, 0.5,  CSC_TYPES+0.5);
  for (int i=1; i<=h_csc_clu_flux_per_layer->GetXaxis()->GetNbins();i++)
    h_csc_clu_flux_per_layer->GetXaxis()->SetBinLabel(i,csc_type[i].c_str());

  h_csc_clu_rate_per_ch = fs->make<TH1D>("h_csc_clu_rate_per_ch",
      (n_clusters+" rate per chamber in CSC at L=10^{34};ME station/ring;Hz/cm^{2}").c_str(), CSC_TYPES, 0.5, CSC_TYPES+0.5);
  for (int i=1; i<=h_csc_clu_rate_per_ch->GetXaxis()->GetNbins();i++)
    h_csc_clu_rate_per_ch->GetXaxis()->SetBinLabel(i,csc_type[i].c_str());

  for (int pdg=1; pdg<N_PDGIDS; pdg++) pdg2idx[pdg_ids[pdg]] = pdg;
  //for (int me=0; me<=CSC_TYPES; me++) for (int pdg=1; pdg<N_PDGIDS; pdg++)
  for (int me=0; me<=0; me++) for (int pdg=1; pdg<N_PDGIDS; pdg++)
  {
    sprintf(label,"h_csc_tof_vs_ekin_%s_%s",csc_type_[me].c_str(), pdg_ids_names_[pdg].c_str());
    //sprintf(nlabel,"SimHit time vs. E_{kin}: %s in %s CSC;log_{10}(p, MeV);log_{10}(t, nsec)", pdg_ids_names[pdg].c_str(), csc_type[me].c_str());
    sprintf(nlabel,"SimHit time vs. E_{kin}: %s;log_{10}E_{kin}(MeV);log_{10}TOF(ns)", csc_type[me].c_str());
    h_csc_tof_vs_ekin[me][pdg]  = fs->make<TH2D>(label, nlabel, 450, -4, 5, 350, 1, 8);
    h_csc_tof_vs_ekin[me][pdg]->SetMarkerColor(pdg_colors[pdg]);
    h_csc_tof_vs_ekin[me][pdg]->SetMarkerStyle(pdg_markers[pdg]);
  }


  h_gem_rz_sh_heatmap = fs->make<TH2D>("h_gem_rz_sh_heatmap",("GEM "+n_simhits+" #rho-z;z, cm;#rho, cm").c_str(),572,0,1120,160,0,800);

  h_gem_nevt_fraction_with_sh = fs->make<TH1D>("h_gem_nevt_fraction_with_sh",
      ("Fraction of events with "+n_simhits+" by GEM type;GE station/ring").c_str(), GEM_TYPES+1, -0.5, GEM_TYPES+0.5);
  for (int i=1; i<=h_gem_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_gem_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,gem_type[i-1].c_str());

  h_gem_hit_flux_per_layer = fs->make<TH1D>("h_gem_hit_flux_per_layer",
      (n_simhits+" Flux in GEM at L=10^{34};RE station/ring;Hz/cm^{2}").c_str(), GEM_TYPES, 0.5,  GEM_TYPES+0.5);
  for (int i=1; i<=h_gem_hit_flux_per_layer->GetXaxis()->GetNbins();i++)
    h_gem_hit_flux_per_layer->GetXaxis()->SetBinLabel(i,gem_type[i].c_str());

  h_gem_hit_rate_per_ch = fs->make<TH1D>("h_gem_hit_rate_per_ch",
      (n_simhits+" rate per chamber in GEM at L=10^{34};GE station/ring;kHz").c_str(), GEM_TYPES, 0.5, GEM_TYPES+0.5);
  for (int i=1; i<=h_gem_hit_rate_per_ch->GetXaxis()->GetNbins();i++)
    h_gem_hit_rate_per_ch->GetXaxis()->SetBinLabel(i,gem_type[i].c_str());

  h_gem_clu_flux_per_layer = fs->make<TH1D>("h_gem_clu_flux_per_layer",
      (n_clusters+" Flux in GEM at L=10^{34};GE station/ring;Hz/cm^{2}").c_str(), GEM_TYPES, 0.5,  GEM_TYPES+0.5);
  for (int i=1; i<=h_gem_clu_flux_per_layer->GetXaxis()->GetNbins();i++)
    h_gem_clu_flux_per_layer->GetXaxis()->SetBinLabel(i,gem_type[i].c_str());

  h_gem_clu_rate_per_ch = fs->make<TH1D>("h_gem_clu_rate_per_ch",
      (n_clusters+" rate per chamber in GEM L=10^{34};GE station/ring;kHz").c_str(), GEM_TYPES, 0.5, GEM_TYPES+0.5);
  for (int i=1; i<=h_gem_clu_rate_per_ch->GetXaxis()->GetNbins();i++)
    h_gem_clu_rate_per_ch->GetXaxis()->SetBinLabel(i,gem_type[i].c_str());

  for (int pdg=1; pdg<N_PDGIDS; pdg++)
  {
    sprintf(label,"h_gem_tof_vs_ekin_%s", pdg_ids_names_[pdg].c_str());
    sprintf(nlabel,"SimHit time vs. E_{kin}: GEM;log_{10}E_{kin}(MeV);log_{10}TOF(ns)");
    h_gem_tof_vs_ekin[pdg]  = fs->make<TH2D>(label, nlabel, 450, -4, 5, 350, 1, 8);
    h_gem_tof_vs_ekin[pdg]->SetMarkerColor(pdg_colors[pdg]);
    h_gem_tof_vs_ekin[pdg]->SetMarkerStyle(pdg_markers[pdg]);
  }


  h_rpc_rz_sh_heatmap = fs->make<TH2D>("h_rpc_rz_sh_heatmap",("RPC "+n_simhits+" #rho-z;z, cm;#rho, cm").c_str(),572,0,1120,160,0,800);

  h_rpcf_nevt_fraction_with_sh = fs->make<TH1D>("h_rpcf_nevt_fraction_with_sh",
      ("Fraction of events with "+n_simhits+" by RPFf type;RE station/ring").c_str(), RPCF_TYPES+1, -0.5, RPCF_TYPES+0.5);
  for (int i=1; i<=h_rpcf_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_rpcf_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,rpcf_type[i-1].c_str());

  h_rpcf_hit_flux_per_layer = fs->make<TH1D>("h_rpcf_hit_flux_per_layer",
      (n_simhits+" Flux in RPCf at L=10^{34};RE station/ring;Hz/cm^{2}").c_str(), RPCF_TYPES, 0.5,  RPCF_TYPES+0.5);
  for (int i=1; i<=h_rpcf_hit_flux_per_layer->GetXaxis()->GetNbins();i++)
    h_rpcf_hit_flux_per_layer->GetXaxis()->SetBinLabel(i,rpcf_type[i].c_str());

  h_rpcf_hit_rate_per_ch = fs->make<TH1D>("h_rpcf_hit_rate_per_ch",
      (n_simhits+" rate per chamber in RPCf at L=10^{34};RE station/ring;kHz").c_str(), RPCF_TYPES, 0.5, RPCF_TYPES+0.5);
  for (int i=1; i<=h_rpcf_hit_rate_per_ch->GetXaxis()->GetNbins();i++)
    h_rpcf_hit_rate_per_ch->GetXaxis()->SetBinLabel(i,rpcf_type[i].c_str());

  h_rpcf_clu_flux_per_layer = fs->make<TH1D>("h_rpcf_clu_flux_per_layer",
      (n_clusters+" Flux in RPCf at L=10^{34};RE station/ring;Hz/cm^{2}").c_str(), RPCF_TYPES, 0.5,  RPCF_TYPES+0.5);
  for (int i=1; i<=h_rpcf_clu_flux_per_layer->GetXaxis()->GetNbins();i++)
    h_rpcf_clu_flux_per_layer->GetXaxis()->SetBinLabel(i,rpcf_type[i].c_str());

  h_rpcf_clu_rate_per_ch = fs->make<TH1D>("h_rpcf_clu_rate_per_ch",
      (n_clusters+" rate per chamber in RPCf at L=10^{34};RE station/ring;kHz").c_str(), RPCF_TYPES, 0.5, RPCF_TYPES+0.5);
  for (int i=1; i<=h_rpcf_clu_rate_per_ch->GetXaxis()->GetNbins();i++)
    h_rpcf_clu_rate_per_ch->GetXaxis()->SetBinLabel(i,rpcf_type[i].c_str());


  h_rpcb_nevt_fraction_with_sh = fs->make<TH1D>("h_rpcb_nevt_fraction_with_sh",
      ("Fraction of events with "+n_simhits+" by RPFb type;RB station/|wheel|").c_str(), RPCB_TYPES+1, -0.5, RPCB_TYPES+0.5);
  for (int i=1; i<=h_rpcb_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_rpcb_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,rpcb_type[i-1].c_str());

  h_rpcb_hit_flux_per_layer = fs->make<TH1D>("h_rpcb_hit_flux_per_layer",
      (n_simhits+" Flux in RPCb at L=10^{34};RB station/|wheel|;Hz/cm^{2}").c_str(), RPCB_TYPES, 0.5, RPCB_TYPES+0.5);
  for (int i=1; i<=h_rpcb_hit_flux_per_layer->GetXaxis()->GetNbins();i++)
    h_rpcb_hit_flux_per_layer->GetXaxis()->SetBinLabel(i,rpcb_type[i].c_str());

  h_rpcb_hit_rate_per_ch = fs->make<TH1D>("h_rpcb_hit_rate_per_ch",
      (n_simhits+" rate per chamber in RPCb at L=10^{34};RB station/|wheel|;kHz").c_str(), RPCB_TYPES, 0.5, RPCB_TYPES+0.5);
  for (int i=1; i<=h_rpcb_hit_rate_per_ch->GetXaxis()->GetNbins();i++)
    h_rpcb_hit_rate_per_ch->GetXaxis()->SetBinLabel(i,rpcb_type[i].c_str());

  h_rpcb_clu_flux_per_layer = fs->make<TH1D>("h_rpcb_clu_flux_per_layer",
      (n_clusters+" Flux in RPCb at L=10^{34};RB station/|wheel|;Hz/cm^{2}").c_str(), RPCB_TYPES, 0.5, RPCB_TYPES+0.5);
  for (int i=1; i<=h_rpcb_clu_flux_per_layer->GetXaxis()->GetNbins();i++)
    h_rpcb_clu_flux_per_layer->GetXaxis()->SetBinLabel(i,rpcb_type[i].c_str());

  h_rpcb_clu_rate_per_ch = fs->make<TH1D>("h_rpcb_clu_rate_per_ch",
      (n_clusters+" rate per chamber in RPCb at L=10^{34};RB station/|wheel|;kHz").c_str(), RPCB_TYPES, 0.5, RPCB_TYPES+0.5);
  for (int i=1; i<=h_rpcb_clu_rate_per_ch->GetXaxis()->GetNbins();i++)
    h_rpcb_clu_rate_per_ch->GetXaxis()->SetBinLabel(i,rpcb_type[i].c_str());

  for (int pdg=1; pdg<N_PDGIDS; pdg++)
  {
    sprintf(label,"h_rpcb_tof_vs_ekin_%s", pdg_ids_names_[pdg].c_str());
    sprintf(nlabel,"SimHit time vs. E_{kin}: RPCb;log_{10}E_{kin}(MeV);log_{10}TOF(ns)");
    h_rpcb_tof_vs_ekin[pdg]  = fs->make<TH2D>(label, nlabel, 450, -4, 5, 350, 1, 8);
    h_rpcb_tof_vs_ekin[pdg]->SetMarkerColor(pdg_colors[pdg]);
    h_rpcb_tof_vs_ekin[pdg]->SetMarkerStyle(pdg_markers[pdg]);

    sprintf(label,"h_rpcf_tof_vs_ekin_%s", pdg_ids_names_[pdg].c_str());
    sprintf(nlabel,"SimHit time vs. E_{kin}: RPCf;log_{10}E_{kin}(MeV);log_{10}TOF(ns)");
    h_rpcf_tof_vs_ekin[pdg]  = fs->make<TH2D>(label, nlabel, 450, -4, 5, 350, 1, 8);
    h_rpcf_tof_vs_ekin[pdg]->SetMarkerColor(pdg_colors[pdg]);
    h_rpcf_tof_vs_ekin[pdg]->SetMarkerStyle(pdg_markers[pdg]);
  }



  h_dt_xy_sh_heatmap = fs->make<TH2D>("h_dt_xy_sh_heatmap",("DT "+n_simhits+" xy;x, cm;y, cm").c_str(),320,-800.,800.,320,-800.,800.);

  h_dt_nevt_fraction_with_sh = fs->make<TH1D>("h_dt_nevt_fraction_with_sh",
      ("Fraction of events with "+n_simhits+" by DT type;DT station/|wheel|").c_str(), DT_TYPES+1, -0.5, DT_TYPES+0.5);
  for (int i=1; i<=h_dt_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_dt_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,dt_type[i-1].c_str());

  h_dt_hit_flux_per_layer = fs->make<TH1D>("h_dt_hit_flux_per_layer",
      (n_simhits+" Flux per DT layer at L=10^{34};DT station/|wheel|;Hz/cm^{2}").c_str(), DT_TYPES, 0.5,  DT_TYPES+0.5);
  for (int i=1; i<=h_dt_hit_flux_per_layer->GetXaxis()->GetNbins();i++)
    h_dt_hit_flux_per_layer->GetXaxis()->SetBinLabel(i,dt_type[i].c_str());

  h_dt_hit_rate_per_ch = fs->make<TH1D>("h_dt_hit_rate_per_ch",
      (n_simhits+" rate per chamber in DT at L=10^{34};DT station/|wheel|;kHz").c_str(), DT_TYPES, 0.5, DT_TYPES+0.5);
  for (int i=1; i<=h_dt_hit_rate_per_ch->GetXaxis()->GetNbins();i++)
    h_dt_hit_rate_per_ch->GetXaxis()->SetBinLabel(i,dt_type[i].c_str());

  for (int pdg=1; pdg<N_PDGIDS; pdg++)
  {
    sprintf(label,"h_dt_tof_vs_ekin_%s",pdg_ids_names_[pdg].c_str());
    sprintf(nlabel,"SimHit time vs. E_{kin}: DT;log_{10}E_{kin}(MeV);log_{10}TOF(ns)");
    h_dt_tof_vs_ekin[pdg]  = fs->make<TH2D>(label, nlabel, 450, -4, 5, 350, 1, 8);
    h_dt_tof_vs_ekin[pdg]->SetMarkerColor(pdg_colors[pdg]);
    h_dt_tof_vs_ekin[pdg]->SetMarkerStyle(pdg_markers[pdg]);
  }


  h_mu_xy_sh_heatmap = fs->make<TH2D>("h_mu_xy_sh_heatmap",("Barrel "+n_simhits+" xy;x, cm;y, cm").c_str(),320,-800.,800.,320,-800.,800.);

  h_mu_rz_sh_heatmap = fs->make<TH2D>("h_mu_rz_sh_heatmap", (n_simhits+" #rho-z;z, cm;#rho, cm").c_str(),572,0,1120,160,0,800);


  gr_csc_hit_flux_me1 = fs->make<TGraphErrors>(4);
  gr_csc_hit_flux_me1->SetName("gr_csc_hit_flux_me1");
  gr_csc_hit_flux_me1->SetTitle("SimHit Flux in ME1;r, cm;Hz/cm^{2}");
  gr_csc_hit_flux_me2 = fs->make<TGraphErrors>(2);
  gr_csc_hit_flux_me2->SetName("gr_csc_hit_flux_me2");
  gr_csc_hit_flux_me2->SetTitle("SimHit Flux in ME2;r, cm;Hz/cm^{2}");
  gr_csc_hit_flux_me3 = fs->make<TGraphErrors>(2);
  gr_csc_hit_flux_me3->SetName("gr_csc_hit_flux_me3");
  gr_csc_hit_flux_me3->SetTitle("SimHit Flux in ME3;r, cm;Hz/cm^{2}");
  gr_csc_hit_flux_me4 = fs->make<TGraphErrors>(2);
  gr_csc_hit_flux_me4->SetName("gr_csc_hit_flux_me4");
  gr_csc_hit_flux_me4->SetTitle("SimHit Flux in ME3;r, cm;Hz/cm^{2}");

  gr_dt_hit_flux_mb1 = fs->make<TGraphErrors>(3);
  gr_dt_hit_flux_mb1->SetName("gr_dt_hit_flux_mb1");
  gr_dt_hit_flux_mb1->SetTitle("SimHit Flux in MB1;z, cm;Hz/cm^{2}");
  gr_dt_hit_flux_mb2 = fs->make<TGraphErrors>(3);
  gr_dt_hit_flux_mb2->SetName("gr_dt_hit_flux_mb2");
  gr_dt_hit_flux_mb2->SetTitle("SimHit Flux in MB2;z, cm;Hz/cm^{2}");
  gr_dt_hit_flux_mb3 = fs->make<TGraphErrors>(3);
  gr_dt_hit_flux_mb3->SetName("gr_dt_hit_flux_mb3");
  gr_dt_hit_flux_mb3->SetTitle("SimHit Flux in MB3;z, cm;Hz/cm^{2}");
  gr_dt_hit_flux_mb4 = fs->make<TGraphErrors>(3);
  gr_dt_hit_flux_mb4->SetName("gr_dt_hit_flux_mb4");
  gr_dt_hit_flux_mb4->SetTitle("SimHit Flux in MB4;z, cm;Hz/cm^{2}");

  gr_gem_hit_flux_ge1 = fs->make<TGraphErrors>(10);
  gr_gem_hit_flux_ge1->SetName("gr_gem_hit_flux_ge1");
  gr_gem_hit_flux_ge1->SetTitle("SimHit Flux in GE1;r, cm;Hz/cm^{2}");
}

// ================================================================================================
MuSimHitOccupancy::~MuSimHitOccupancy()
{}


// ================================================================================================
void
MuSimHitOccupancy::bookCSCSimHitsTrees()
{
  edm::Service<TFileService> fs;

  csc_sh_tree = fs->make<TTree>("CSCSimHitsTree", "CSCSimHitsTree");
  csc_sh_tree->Branch("evtn", &evtn,"evtn/i");
  csc_sh_tree->Branch("shn", &csc_shn,"shn/i");
  c_id.book(csc_sh_tree);
  c_h.book(csc_sh_tree);

  csc_cl_tree = fs->make<TTree>("CSCClustersTree", "CSCClustersTree");
  csc_cl_tree->Branch("evtn", &evtn,"evtn/i");
  c_id.book(csc_cl_tree);
  c_cl.book(csc_cl_tree);

  csc_la_tree = fs->make<TTree>("CSCLayersTree", "CSCLayersTree");
  csc_la_tree->Branch("evtn", &evtn,"evtn/i");
  c_id.book(csc_la_tree);
  c_la.book(csc_la_tree);

  csc_ch_tree = fs->make<TTree>("CSCChambersTree", "CSCChambersTree");
  csc_ch_tree->Branch("evtn", &evtn,"evtn/i");
  c_id.book(csc_ch_tree);
  c_ch.book(csc_ch_tree);

  csc_ev_tree = fs->make<TTree>("CSCEventsTree", "CSCEventsTree");
  csc_ev_tree->Branch("evtn", &evtn,"evtn/i");
  c_ev.book(csc_ev_tree);
}


// ================================================================================================
void
MuSimHitOccupancy::bookGEMSimHitsTrees()
{
  edm::Service<TFileService> fs;
  gem_sh_tree = fs->make<TTree>("GEMSimHitsTree", "GEMSimHitsTree");
  gem_sh_tree->Branch("evtn", &evtn,"evtn/i");
  gem_sh_tree->Branch("shn", &gem_shn,"shn/i");
  g_id.book(gem_sh_tree);
  g_h.book(gem_sh_tree);

  gem_cl_tree = fs->make<TTree>("GEMClustersTree", "GEMClustersTree");
  gem_cl_tree->Branch("evtn", &evtn,"evtn/i");
  g_id.book(gem_cl_tree);
  g_cl.book(gem_cl_tree);

  gem_part_tree = fs->make<TTree>("GEMPartTree", "GEMPartTree");
  gem_part_tree->Branch("evtn", &evtn,"evtn/i");
  g_id.book(gem_part_tree);
  g_part.book(gem_part_tree);

  gem_ch_tree = fs->make<TTree>("GEMChambersTree", "GEMChambersTree");
  gem_ch_tree->Branch("evtn", &evtn,"evtn/i");
  g_id.book(gem_ch_tree);
  g_ch.book(gem_ch_tree);

  gem_ev_tree = fs->make<TTree>("GEMEventsTree", "GEMEventsTree");
  gem_ev_tree->Branch("evtn", &evtn,"evtn/i");
  g_ev.book(gem_ev_tree);
}


// ================================================================================================
void
MuSimHitOccupancy::bookRPCSimHitsTrees()
{
  edm::Service<TFileService> fs;
  rpc_sh_tree = fs->make<TTree>("RPCSimHitsTree", "RPCSimHitsTree");
  rpc_sh_tree->Branch("evtn", &evtn,"evtn/i");
  rpc_sh_tree->Branch("shn", &rpc_shn,"shn/i");
  r_id.book(rpc_sh_tree);
  r_h.book(rpc_sh_tree);

  rpc_cl_tree = fs->make<TTree>("RPCClustersTree", "RPCClustersTree");
  rpc_cl_tree->Branch("evtn", &evtn,"evtn/i");
  r_id.book(rpc_cl_tree);
  r_cl.book(rpc_cl_tree);

  rpc_rl_tree = fs->make<TTree>("RPCRollsTree", "RPCRollsTree");
  rpc_rl_tree->Branch("evtn", &evtn,"evtn/i");
  r_id.book(rpc_rl_tree);
  r_rl.book(rpc_rl_tree);

  rpc_ch_tree = fs->make<TTree>("RPCChambersTree", "RPCChambersTree");
  rpc_ch_tree->Branch("evtn", &evtn,"evtn/i");
  r_id.book(rpc_ch_tree);
  r_ch.book(rpc_ch_tree);

  rpc_ev_tree = fs->make<TTree>("RPCEventsTree", "RPCEventsTree");
  rpc_ev_tree->Branch("evtn", &evtn,"evtn/i");
  r_ev.book(rpc_ev_tree);
}


// ================================================================================================
void
MuSimHitOccupancy::bookDTSimHitsTrees()
{
  edm::Service<TFileService> fs;
  dt_sh_tree = fs->make<TTree>("DTSimHitsTree", "DTSimHitsTree");
  dt_sh_tree->Branch("evtn", &evtn,"evtn/i");
  dt_sh_tree->Branch("shn", &dt_shn,"shn/i");
  d_id.book(dt_sh_tree);
  d_h.book(dt_sh_tree);
}



// ================================================================================================
void
MuSimHitOccupancy::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  evtn += 1;
  csc_shn = 0;
  gem_shn = 0;
  rpc_shn = 0;
  dt_shn = 0;

  edm::ESHandle < ParticleDataTable > pdt_h;
  iSetup.getData(pdt_h);
  pdt_ = &*pdt_h;

  if (do_csc_)
  {
    ESHandle< CSCGeometry > csc_geom;
    iSetup.get< MuonGeometryRecord >().get(csc_geom);
    csc_geometry = &*csc_geom;

    if (evtn==1) areas_.calculateCSCDetectorAreas(csc_geometry);

    // get SimHits
    simhit_map_csc.fill(iEvent);
   
    analyzeCSC();
  }

  if (do_gem_)
  {
    ESHandle< GEMGeometry > gem_geom;
    iSetup.get< MuonGeometryRecord >().get(gem_geom);
    gem_geometry = &*gem_geom;

    if (evtn==1) areas_.calculateGEMDetectorAreas(gem_geometry);

    // get SimHits
    simhit_map_gem.fill(iEvent);
   
    analyzeGEM();
  }

  if (do_rpc_)
  {
    ESHandle< RPCGeometry > rpc_geom;
    iSetup.get< MuonGeometryRecord >().get(rpc_geom);
    rpc_geometry = &*rpc_geom;

    if (evtn==1) areas_.calculateRPCDetectorAreas(rpc_geometry);

    // get SimHits
    simhit_map_rpc.fill(iEvent);

    analyzeRPC();
  }

  if (do_dt_)
  {
    ESHandle< DTGeometry > dt_geom;
    iSetup.get< MuonGeometryRecord >().get(dt_geom);
    dt_geometry = &*dt_geom;

    if (evtn==1) areas_.calculateDTDetectorAreas(dt_geometry);

    // get SimHits
    simhit_map_dt.fill(iEvent);

    analyzeDT();
  }

}

// ================================================================================================
void
MuSimHitOccupancy::analyzeCSC()
{
  using namespace edm;
  using namespace std;

  bool ev_has_csc_type[CSC_TYPES+1]={0,0,0,0,0,0,0,0,0,0,0};
  bool has_cscsh_in_rpc = false;

  vector<int> chIds = simhit_map_csc.chambersWithHits();
  if (chIds.size()) {
    //cout<<"--- CSC chambers with hits: "<<chIds.size()<<endl;
    nevt_with_cscsh++;
  }

  vector<MyCSCChamber> evt_mychambers;
  for (size_t ch = 0; ch < chIds.size(); ch++)
  {
    CSCDetId chId(chIds[ch]);
    c_cid.init(chId);

    std::vector<int> layer_ids = simhit_map_csc.chamberLayersWithHits(chIds[ch]);
    //if (layer_ids.size()) cout<<"------ layers with hits: "<<layer_ids.size()<<endl;

    vector<MyCSCLayer> chamber_mylayers;
    for  (size_t la = 0; la < layer_ids.size(); la++)
    {
      CSCDetId layerId(layer_ids[la]);
      c_id.init(layerId);

      PSimHitContainer hits = simhit_map_csc.hits(layer_ids[la]);
      vector<MyCSCSimHit> layer_mysimhits;
      for (unsigned j = 0; j < hits.size(); j++)
      {
        csc_shn += 1;
        c_h.init(hits[j], csc_geometry, pdt_);
        layer_mysimhits.push_back(c_h);
        if (fill_csc_sh_tree_) csc_sh_tree->Fill();

        // count hits
        n_cscsh++;
        if (fabs(c_h.eta) < 1.6 && c_id.s < 4) // RPC region in 1-3 stations
        {
          n_cscsh_in_rpc++;
          has_cscsh_in_rpc = true;
        }

        // Fill some histos
        h_csc_rz_sh_xray->Fill(fabs(c_h.gz), c_h.r);
        h_csc_rz_sh_heatmap->Fill(fabs(c_h.gz), c_h.r);
        h_mu_rz_sh_heatmap->Fill(fabs(c_h.gz), c_h.r);

        h_csc_hit_flux_per_layer->Fill(c_cid.t);
        h_csc_hit_rate_per_ch->Fill(c_cid.t);

        int sh_pdg = abs(c_h.pdg);
        if (sh_pdg > 100000) sh_pdg = 1000000000;
        int idx = pdg2idx[sh_pdg];
        if (idx > 0)
        {
          double ekin = log10(c_h.eKin() * 1000.);
          //h_csc_tof_vs_ekin[c_cid.t][idx]->Fill(ekin, log10(c_h.t));
          h_csc_tof_vs_ekin[0][idx]->Fill(ekin, log10(c_h.t));
        }
        else cout << " *** non-registered pdgid: " << sh_pdg << endl;
      }

      cout<<" calling recursive clustering:"<<endl;
      vector<vector<MyCSCSimHit> > clusters = clusterCSCHitsInLayer(layer_mysimhits);
      //cout<<"      "<< layer_ids[la]<<" "<<layerId<<"   # hits = "<<hits.size()<<"  # clusters = "<<clusters.size()<<endl;

      vector<MyCSCCluster> layer_myclusters;
      for (unsigned cl = 0; cl < clusters.size(); cl++)
      {
        vector<MyCSCSimHit> cluster_mysimhits = clusters[cl];

        c_cl.init(cluster_mysimhits);
        layer_myclusters.push_back(c_cl);
        if (fill_csc_sh_tree_) csc_cl_tree->Fill();

        // fill some histograms
        h_csc_rz_clu_heatmap->Fill(fabs(c_cl.gz), c_cl.r);

        h_csc_clu_flux_per_layer->Fill(c_cid.t);
        h_csc_clu_rate_per_ch->Fill(c_cid.t);
      }

      c_la.init(layerId.layer(),layer_myclusters);
      chamber_mylayers.push_back(c_la);
      if (fill_csc_sh_tree_) csc_la_tree->Fill();
    }
    c_ch.init(chamber_mylayers);
    evt_mychambers.push_back(c_ch);
    if (fill_csc_sh_tree_) csc_ch_tree->Fill();

    // this chamber type has hits:
    if (c_ch.nl)
    {
      ev_has_csc_type[c_cid.t] = true;
      ev_has_csc_type[0] = true;
    }

    // Fill some histos
    h_csc_nlayers_in_ch[c_cid.t]->Fill(c_ch.nl);
    h_csc_nlayers_in_ch[0]->Fill(c_ch.nl);
  }
  c_ev.init(evt_mychambers);
  if (fill_csc_sh_tree_) csc_ev_tree->Fill();

  // fill events with CSC hits by CSC type:
  for (int t=0; t<=CSC_TYPES; t++)
    if (ev_has_csc_type[t]) h_csc_nevt_fraction_with_sh->Fill(t);

  // count events with CSC hits in PRC region
  if (has_cscsh_in_rpc) nevt_with_cscsh_in_rpc++;
}


// ================================================================================================
void
MuSimHitOccupancy::analyzeGEM()
{
  using namespace edm;
  using namespace std;

  bool ev_has_gem_type[GEM_TYPES+1]={0,0};
  bool has_gemsh = false;

  vector<int> gem_ids = simhit_map_gem.detsWithHits();
  if (gem_ids.size()) nevt_with_gemsh++;

  map<int, vector<MyGEMPart> > mapChamberParts;

  // loop over ipartitions
  for (size_t id = 0; id < gem_ids.size(); id++)
  {
    //cout<<" gemid: "<<gem_ids[id]<<endl;
    //DetId d(gem_ids[id]);
    //if (d.det()!=DetId::Muon || d.subdetId()!=MuonSubdetId::GEM) { cout<<"weird non-gem hit:"<<d.det()<<" "<<d.subdetId()<<endl; continue;}
    GEMDetId shid(gem_ids[id]);
    g_id.init(shid);

    PSimHitContainer hits = simhit_map_gem.hits(gem_ids[id]);

    // hits in a partition
    vector<MyGEMSimHit> part_mysimhits;
    for (size_t ih=0; ih<hits.size(); ih++)
    {
      gem_shn += 1;
      PSimHit &sh = hits[ih];

      g_h.init(sh, gem_geometry, pdt_);
      if (fill_gem_sh_tree_) gem_sh_tree->Fill();
      part_mysimhits.push_back(g_h);

      n_gemsh++;
      has_gemsh = true;

      // Fill some histos
      h_gem_rz_sh_heatmap->Fill(fabs(g_h.gz), g_h.r);
      h_mu_rz_sh_heatmap->Fill(fabs(g_h.gz), g_h.r);

      int sh_pdg = abs(g_h.pdg);
      if (sh_pdg > 100000) sh_pdg = 1000000000;
      int idx = pdg2idx[sh_pdg];
      if (idx<=0) cout << " *** non-registered pdgid: " << sh_pdg << endl;

      ev_has_gem_type[g_id.t] = true;
      ev_has_gem_type[0] = true;

      h_gem_hit_flux_per_layer->Fill(g_id.t);
      h_gem_hit_rate_per_ch->Fill(g_id.t);

      if (idx > 0) h_gem_tof_vs_ekin[idx]->Fill( log10(g_h.eKin() * 1000.), log10(g_h.t) );
    }

    cout<<" calling GEM recursive clustering:"<<endl;
    vector<vector<MyGEMSimHit> > clusters = clusterGEMHitsInPart(part_mysimhits);

    vector<MyGEMCluster> part_myclusters;
    for (unsigned cl = 0; cl < clusters.size(); cl++)
    {
      vector<MyGEMSimHit> cluster_mysimhits = clusters[cl];

      g_cl.init(cluster_mysimhits);
      part_myclusters.push_back(g_cl);
      if (fill_gem_sh_tree_) gem_cl_tree->Fill();

      // fill some histograms
      h_gem_clu_flux_per_layer->Fill(g_id.t);
      h_gem_clu_rate_per_ch->Fill(g_id.t);
    }

    g_part.init(g_id.part, g_id.layer, part_myclusters);
    if (fill_gem_sh_tree_) gem_part_tree->Fill();

    GEMDetId chid(shid.region(), shid.station(), shid.ring(), 1, shid.chamber(), 0);
    int ch_id = chid.rawId();
    map<int, vector<MyGEMPart> >::const_iterator is_there = mapChamberParts.find(ch_id);
    if (is_there != mapChamberParts.end()) mapChamberParts[ch_id].push_back(g_part);
    else
    {
      vector<MyGEMPart> vp;
      vp.push_back(g_part);
      mapChamberParts[ch_id] = vp;
    }
  }

  // loop over chambers
  vector<MyGEMChamber> evt_mychambers;
  map<int, vector<MyGEMPart> >::iterator itr = mapChamberParts.begin();
  for (; itr!= mapChamberParts.end(); itr++)
  {
    GEMDetId id(itr->first);
    g_id.init(id);

    g_ch.init(itr->second);
    evt_mychambers.push_back(g_ch);
    if (fill_gem_sh_tree_) gem_ch_tree->Fill();
  }
  g_ev.init(evt_mychambers);
  if (fill_gem_sh_tree_) gem_ev_tree->Fill();


  // fill events with GEM hits by GEM type:
  for (int t=0; t<=GEM_TYPES; t++)
    if (ev_has_gem_type[t]) h_gem_nevt_fraction_with_sh->Fill(t);

  if (has_gemsh) nevt_with_gemsh++;
}


// ================================================================================================
void
MuSimHitOccupancy::analyzeRPC()
{
  using namespace edm;
  using namespace std;

  bool ev_has_rpcf_type[RPCF_TYPES+1]={0,0,0,0,0,0,0,0,0,0,0,0,0};
  bool ev_has_rpcb_type[RPCB_TYPES+1]={0,0,0,0,0,0,0,0,0,0,0,0,0};
  bool has_rpcsh_e = false, has_rpcsh_b = false;

  vector<int> rpc_ids = simhit_map_rpc.detsWithHits();
  if (rpc_ids.size()) nevt_with_rpcsh++;

  map<int, vector<MyRPCRoll> > mapChamberRolls;

  // loop over rolls
  for (size_t id = 0; id < rpc_ids.size(); id++)
  {
    //cout<<" rpcid: "<<rpc_ids[id]<<endl;
    //DetId d(rpc_ids[id]);
    //if (d.det()!=DetId::Muon || d.subdetId()!=MuonSubdetId::RPC) { cout<<"weird non-rpc hit:"<<d.det()<<" "<<d.subdetId()<<endl; continue;}
    RPCDetId shid(rpc_ids[id]);
    r_id.init(shid);
    //RPCGeomServ rpcsrv(shid);
    //cout<<"   "<<rpcsrv.name()<<"  "<<rpcsrv.shortname()<<endl;

    PSimHitContainer hits = simhit_map_rpc.hits(rpc_ids[id]);

    // hits in a roll
    vector<MyRPCSimHit> roll_mysimhits;
    for (size_t ih=0; ih<hits.size(); ih++)
    {
      rpc_shn += 1;
      PSimHit &sh = hits[ih];

      r_h.init(sh, rpc_geometry, pdt_);
      if (fill_rpc_sh_tree_) rpc_sh_tree->Fill();
      roll_mysimhits.push_back(r_h);

      n_rpcsh++;
      if (shid.region() == 0) { n_rpcsh_b++; has_rpcsh_b = true; }
      else                    { n_rpcsh_e++; has_rpcsh_e = true; }

      // Fill some histos
      h_rpc_rz_sh_heatmap->Fill(fabs(r_h.gz), r_h.r);
      h_mu_rz_sh_heatmap->Fill(fabs(r_h.gz), r_h.r);

      int sh_pdg = abs(r_h.pdg);
      if (sh_pdg > 100000) sh_pdg = 1000000000;
      int idx = pdg2idx[sh_pdg];
      if (idx<=0) cout << " *** non-registered pdgid: " << sh_pdg << endl;

      if (shid.region() == 0)
      {
        h_mu_xy_sh_heatmap->Fill(r_h.gx, r_h.gy);
	
        ev_has_rpcb_type[r_id.t] = true;
        ev_has_rpcb_type[0] = true;

        h_rpcb_hit_flux_per_layer->Fill(r_id.t);
        h_rpcb_hit_rate_per_ch->Fill(r_id.t);

        if (idx > 0) h_rpcb_tof_vs_ekin[idx]->Fill( log10(r_h.eKin() * 1000.), log10(r_h.t) );
      }
      else
      {
        ev_has_rpcf_type[r_id.t] = true;
        ev_has_rpcf_type[0] = true;

        h_rpcf_hit_flux_per_layer->Fill(r_id.t);
        h_rpcf_hit_rate_per_ch->Fill(r_id.t);

        if (idx > 0) h_rpcf_tof_vs_ekin[idx]->Fill( log10(r_h.eKin() * 1000.), log10(r_h.t) );
      }
    }

    cout<<" calling recursive clustering:"<<endl;
    vector<vector<MyRPCSimHit> > clusters = clusterRPCHitsInRoll(roll_mysimhits);

    vector<MyRPCCluster> roll_myclusters;
    for (unsigned cl = 0; cl < clusters.size(); cl++)
    {
      vector<MyRPCSimHit> cluster_mysimhits = clusters[cl];

      r_cl.init(cluster_mysimhits);
      roll_myclusters.push_back(r_cl);
      if (fill_rpc_sh_tree_) rpc_cl_tree->Fill();

      // fill some histograms
      if (shid.region() == 0)
      {
        //h_csc_rz_clu_heatmap->Fill(fabs(c_cl.gz), c_cl.r);

        h_rpcb_clu_flux_per_layer->Fill(r_id.t);
        h_rpcb_clu_rate_per_ch->Fill(r_id.t);
      }
      else
      {
        h_rpcf_clu_flux_per_layer->Fill(r_id.t);
        h_rpcf_clu_rate_per_ch->Fill(r_id.t);
      }
    }

    r_rl.init(r_id.roll, r_id.layer, roll_myclusters);
    if (fill_rpc_sh_tree_) rpc_rl_tree->Fill();

    int ch_id = shid.chamberId().rawId();
    map<int, vector<MyRPCRoll> >::const_iterator is_there = mapChamberRolls.find(ch_id);
    if (is_there != mapChamberRolls.end()) mapChamberRolls[ch_id].push_back(r_rl);
    else
    {
      vector<MyRPCRoll> vr;
      vr.push_back(r_rl);
      mapChamberRolls[ch_id] = vr;
    }
  }

  // loop over chambers
  vector<MyRPCChamber> evt_mychambers;
  map<int, vector<MyRPCRoll> >::iterator itr = mapChamberRolls.begin();
  for (; itr!= mapChamberRolls.end(); itr++)
  {
    RPCDetId id(itr->first);
    r_id.init(id);

    r_ch.init(itr->second);
    evt_mychambers.push_back(r_ch);
    if (fill_rpc_sh_tree_) rpc_ch_tree->Fill();
  }
  r_ev.init(evt_mychambers);
  if (fill_rpc_sh_tree_) rpc_ev_tree->Fill();


  // fill events with RPCf hits by RPCf type:
  for (int t=0; t<=RPCF_TYPES; t++)
    if (ev_has_rpcf_type[t]) h_rpcf_nevt_fraction_with_sh->Fill(t);

  // fill events with RPCb hits by RPCb type:
  for (int t=0; t<=RPCB_TYPES; t++)
    if (ev_has_rpcb_type[t]) h_rpcb_nevt_fraction_with_sh->Fill(t);

  if (has_rpcsh_e) nevt_with_rpcsh_e++;
  if (has_rpcsh_b) nevt_with_rpcsh_b++;
}


// ================================================================================================
void
MuSimHitOccupancy::analyzeDT()
{
  using namespace edm;
  using namespace std;

  bool ev_has_dt_type[DT_TYPES+1]={0,0,0,0,0,0,0,0,0,0,0,0,0};

  vector<int> dt_ids = simhit_map_dt.detsWithHits();
  if (dt_ids.size()) nevt_with_dtsh++;

  map<int, vector<MyRPCRoll> > mapChamberRolls;

  // loop over wires?
  for (size_t id = 0; id < dt_ids.size(); id++)
  {
    DTWireId shid(dt_ids[id]);
    d_id.init(shid);

    PSimHitContainer hits = simhit_map_dt.hits(dt_ids[id]);

    for (size_t ih=0; ih<hits.size(); ih++)
    {
      dt_shn += 1;
      PSimHit &sh = hits[ih];

      d_h.init(sh, dt_geometry, pdt_);
      if (fill_dt_sh_tree_) dt_sh_tree->Fill();

      n_dtsh++;

      // Fill some histos
      h_dt_xy_sh_heatmap->Fill(d_h.gx, d_h.gy);
      h_mu_rz_sh_heatmap->Fill(fabs(d_h.gz), d_h.r);
      h_mu_xy_sh_heatmap->Fill(d_h.gx, d_h.gy);

      int sh_pdg = abs(d_h.pdg);
      if (sh_pdg > 100000) sh_pdg = 1000000000;
      int idx = pdg2idx[sh_pdg];
      if (idx<=0) cout << " *** non-registered pdgid: " << sh_pdg << endl;

      ev_has_dt_type[d_id.t] = true;
      ev_has_dt_type[0] = true;

      h_dt_hit_flux_per_layer->Fill(d_id.t);
      h_dt_hit_rate_per_ch->Fill(d_id.t);

      if (idx > 0) h_dt_tof_vs_ekin[idx]->Fill( log10(d_h.eKin() * 1000.), log10(d_h.t) );
    }

  }

  // fill events with DT hits by DT type:
  for (int t=0; t<=DT_TYPES; t++)
    if (ev_has_dt_type[t]) h_dt_nevt_fraction_with_sh->Fill(t);
}



// ================================================================================================
void MuSimHitOccupancy::beginJob() {}


// ================================================================================================
void MuSimHitOccupancy::endJob()
{
  using namespace std;
  cout<<"******************* COUNTERS *******************"<<endl;
  cout<<"* #events: "<< evtn <<endl;
  cout<<"* #events with SimHits in:"<<endl;
  cout<<"*   CSC:                     "<<nevt_with_cscsh<<" ("<<(double)nevt_with_cscsh/evtn*100<<"%)"<<endl;
  cout<<"*   CSC (|eta|<1.6, st 1-3): "<<nevt_with_cscsh_in_rpc<<" ("<<(double)nevt_with_cscsh_in_rpc/evtn*100<<"%)"<<endl;
  cout<<"*   GEM:                     "<<nevt_with_gemsh<<" ("<<(double)nevt_with_gemsh/evtn*100<<"%)"<<endl;
  cout<<"*   RPC:                     "<<nevt_with_rpcsh<<" ("<<(double)nevt_with_rpcsh/evtn*100<<"%)"<<endl;
  cout<<"*   RPC endcaps:             "<<nevt_with_rpcsh_e<<" ("<<(double)nevt_with_rpcsh_e/evtn*100<<"%)"<<endl;
  cout<<"*   RPC barrel:              "<<nevt_with_rpcsh_b<<" ("<<(double)nevt_with_rpcsh_b/evtn*100<<"%)"<<endl;
  cout<<"*   DT:                      "<<nevt_with_dtsh<<" ("<<(double)nevt_with_dtsh/evtn*100<<"%)"<<endl;
  cout<<"* total SimHit numbers in: "<<endl;
  cout<<"*   CSC:                     "<<n_cscsh<<" (~"<<(double)n_cscsh/nevt_with_cscsh<<" sh/evt with hits)"<<endl;
  cout<<"*   CSC (|eta|<1.6, st 1-3): "<<n_cscsh_in_rpc<<" (~"<<(double)n_cscsh_in_rpc/nevt_with_cscsh_in_rpc<<" sh/evt with hits)"<<endl;
  cout<<"*   GEM:                     "<<n_gemsh<<" (~"<<(double)n_gemsh/nevt_with_gemsh<<" sh/evt with hits)"<<endl;
  cout<<"*   RPC:                     "<<n_rpcsh<<" (~"<<(double)n_rpcsh/nevt_with_rpcsh<<" sh/evt with hits)"<<endl;
  cout<<"*   RPC endcaps:             "<<n_rpcsh_e<<" (~"<<(double)n_rpcsh_e/nevt_with_rpcsh_e<<" sh/evt with hits)"<<endl;
  cout<<"*   RPC barrel:              "<<n_rpcsh_b<<" (~"<< ( (nevt_with_rpcsh_b>0) ? (double)n_rpcsh_b/nevt_with_rpcsh_b : 0 )<<" sh/evt with hits)"<<endl;
  cout<<"*   DT:                      "<<n_dtsh<<" (~"<<(double)n_dtsh/nevt_with_dtsh<<" sh/evt with hits)"<<endl;
  cout<<"************************************************"<<endl;

  edm::Service<TFileService> fs;

  // convert to fraction:
  h_csc_nevt_fraction_with_sh->Sumw2();
  h_csc_nevt_fraction_with_sh->Scale(1./evtn);
  h_gem_nevt_fraction_with_sh->Sumw2();
  h_gem_nevt_fraction_with_sh->Scale(1./evtn);
  h_rpcf_nevt_fraction_with_sh->Sumw2();
  h_rpcf_nevt_fraction_with_sh->Scale(1./evtn);
  h_rpcb_nevt_fraction_with_sh->Sumw2();
  h_rpcb_nevt_fraction_with_sh->Scale(1./evtn);
  h_dt_nevt_fraction_with_sh->Sumw2();
  h_dt_nevt_fraction_with_sh->Scale(1./evtn);

  // ---- calculate fluxes per cm2 and rates per chamber at L=10^34 ----

  // pileup at nominal lumi
  double n_pu = 25.;
  // fraction of full BXs : 2808 filled bickets out of 3564
  double f_full_bx = 0.7879;
  if (!input_is_neutrons_) f_full_bx = 1.;
  // bx rate 40 MHz
  double bxrate = 40000000.;

  h_csc_hit_flux_per_layer->Sumw2();
  h_csc_hit_rate_per_ch->Sumw2();
  h_csc_clu_flux_per_layer->Sumw2();
  h_csc_clu_rate_per_ch->Sumw2();
  if (do_csc_) for (int t=1; t <= CSC_TYPES; t++)
  {
    // 2 endcaps , 6 layers
    double scale = bxrate * n_pu * f_full_bx /areas_.csc_total_areas_cm2[t]/evtn;
    double rt = scale * h_csc_hit_flux_per_layer->GetBinContent(t);
    double er = scale * h_csc_hit_flux_per_layer->GetBinError(t);
    h_csc_hit_flux_per_layer->SetBinContent(t, rt);
    h_csc_hit_flux_per_layer->SetBinError(t, er);

    rt = scale * h_csc_clu_flux_per_layer->GetBinContent(t);
    er = scale * h_csc_clu_flux_per_layer->GetBinError(t);
    h_csc_clu_flux_per_layer->SetBinContent(t, rt);
    h_csc_clu_flux_per_layer->SetBinError(t, er);

    scale = bxrate * n_pu * f_full_bx /csc_radial_segm[t]/2./evtn/1000.;
    rt = scale * h_csc_hit_rate_per_ch->GetBinContent(t);
    er = scale * h_csc_hit_rate_per_ch->GetBinError(t);
    h_csc_hit_rate_per_ch->SetBinContent(t, rt);
    h_csc_hit_rate_per_ch->SetBinError(t, er);

    rt = scale * h_csc_clu_rate_per_ch->GetBinContent(t);
    er = scale * h_csc_clu_rate_per_ch->GetBinError(t);
    h_csc_clu_rate_per_ch->SetBinContent(t, rt);
    h_csc_clu_rate_per_ch->SetBinError(t, er);

    for (int i=0; i<4; i++)
    {
      gr_csc_hit_flux_me1->SetPoint(i, areas_.csc_ch_radius[i+1], h_csc_hit_flux_per_layer->GetBinContent(i+1) );
      gr_csc_hit_flux_me1->SetPointError(i, areas_.csc_ch_halfheight[i+1], h_csc_hit_flux_per_layer->GetBinError(i+1) );
      if (i>1) continue;
      gr_csc_hit_flux_me2->SetPoint(i, areas_.csc_ch_radius[i+5], h_csc_hit_flux_per_layer->GetBinContent(i+5) );
      gr_csc_hit_flux_me3->SetPoint(i, areas_.csc_ch_radius[i+7], h_csc_hit_flux_per_layer->GetBinContent(i+7) );
      gr_csc_hit_flux_me4->SetPoint(i, areas_.csc_ch_radius[i+9], h_csc_hit_flux_per_layer->GetBinContent(i+9) );
      gr_csc_hit_flux_me2->SetPointError(i, areas_.csc_ch_halfheight[i+5], h_csc_hit_flux_per_layer->GetBinError(i+5) );
      gr_csc_hit_flux_me3->SetPointError(i, areas_.csc_ch_halfheight[i+7], h_csc_hit_flux_per_layer->GetBinError(i+7) );
      gr_csc_hit_flux_me4->SetPointError(i, areas_.csc_ch_halfheight[i+9], h_csc_hit_flux_per_layer->GetBinError(i+9) );
    }
  }


  h_gem_hit_flux_per_layer->Sumw2();
  h_gem_hit_rate_per_ch->Sumw2();
  if (do_gem_) for (int t=1; t<=GEM_TYPES; t++)
  {
    double scale = bxrate * n_pu * f_full_bx /areas_.gem_total_areas_cm2[t]/evtn;

    double rt = scale * h_gem_hit_flux_per_layer->GetBinContent(t);
    double er = scale * h_gem_hit_flux_per_layer->GetBinError(t);
    h_gem_hit_flux_per_layer->SetBinContent(t,rt);
    h_gem_hit_flux_per_layer->SetBinError(t,er);

    rt = scale * h_gem_clu_flux_per_layer->GetBinContent(t);
    er = scale * h_gem_clu_flux_per_layer->GetBinError(t);
    h_gem_clu_flux_per_layer->SetBinContent(t,rt);
    h_gem_clu_flux_per_layer->SetBinError(t,er);

    scale = bxrate * n_pu * f_full_bx /gem_radial_segm[t]/2/evtn/1000;

    rt = scale * h_gem_hit_rate_per_ch->GetBinContent(t);
    er = scale * h_gem_hit_rate_per_ch->GetBinError(t);
    h_gem_hit_rate_per_ch->SetBinContent(t,rt);
    h_gem_hit_rate_per_ch->SetBinError(t,er);

    rt = scale * h_gem_clu_rate_per_ch->GetBinContent(t);
    er = scale * h_gem_clu_rate_per_ch->GetBinError(t);
    h_gem_clu_rate_per_ch->SetBinContent(t,rt);
    h_gem_clu_rate_per_ch->SetBinError(t,er);
  }
  

  h_rpcf_hit_flux_per_layer->Sumw2();
  h_rpcf_hit_rate_per_ch->Sumw2();
  if (do_rpc_) for (int t=1; t<=RPCF_TYPES; t++)
  {
    double scale = bxrate * n_pu * f_full_bx /areas_.rpcf_total_areas_cm2[t]/evtn;
    double rt = scale * h_rpcf_hit_flux_per_layer->GetBinContent(t);
    double er = scale * h_rpcf_hit_flux_per_layer->GetBinError(t);
    h_rpcf_hit_flux_per_layer->SetBinContent(t,rt);
    h_rpcf_hit_flux_per_layer->SetBinError(t,er);

    rt = scale * h_rpcf_clu_flux_per_layer->GetBinContent(t);
    er = scale * h_rpcf_clu_flux_per_layer->GetBinError(t);
    h_rpcf_clu_flux_per_layer->SetBinContent(t,rt);
    h_rpcf_clu_flux_per_layer->SetBinError(t,er);

    scale = bxrate * n_pu * f_full_bx /rpcf_radial_segm[t]/2/evtn/1000;
    rt = scale * h_rpcf_hit_rate_per_ch->GetBinContent(t);
    er = scale * h_rpcf_hit_rate_per_ch->GetBinError(t);
    h_rpcf_hit_rate_per_ch->SetBinContent(t,rt);
    h_rpcf_hit_rate_per_ch->SetBinError(t,er);

    rt = scale * h_rpcf_clu_rate_per_ch->GetBinContent(t);
    er = scale * h_rpcf_clu_rate_per_ch->GetBinError(t);
    h_rpcf_clu_rate_per_ch->SetBinContent(t,rt);
    h_rpcf_clu_rate_per_ch->SetBinError(t,er);
  }

  
  h_rpcb_hit_flux_per_layer->Sumw2();
  if (do_rpc_) for (int t=1; t<=RPCB_TYPES; t++)
  {
    double scale = bxrate * n_pu * f_full_bx /areas_.rpcb_total_areas_cm2[t]/evtn;
    double rt = scale * h_rpcb_hit_flux_per_layer->GetBinContent(t);
    double er = scale * h_rpcb_hit_flux_per_layer->GetBinError(t);
    h_rpcb_hit_flux_per_layer->SetBinContent(t,rt);
    h_rpcb_hit_flux_per_layer->SetBinError(t,er);

    rt = scale * h_rpcb_clu_flux_per_layer->GetBinContent(t);
    er = scale * h_rpcb_clu_flux_per_layer->GetBinError(t);
    h_rpcb_clu_flux_per_layer->SetBinContent(t,rt);
    h_rpcb_clu_flux_per_layer->SetBinError(t,er);

    scale = bxrate * n_pu * f_full_bx /rpcb_radial_segm[t]/evtn/1000;
    rt = scale * h_rpcb_hit_rate_per_ch->GetBinContent(t);
    er = scale * h_rpcb_hit_rate_per_ch->GetBinError(t);
    h_rpcb_hit_rate_per_ch->SetBinContent(t,rt);
    h_rpcb_hit_rate_per_ch->SetBinError(t,er);

    rt = scale * h_rpcb_clu_rate_per_ch->GetBinContent(t);
    er = scale * h_rpcb_clu_rate_per_ch->GetBinError(t);
    h_rpcb_clu_rate_per_ch->SetBinContent(t,rt);
    h_rpcb_clu_rate_per_ch->SetBinError(t,er);
  }


  h_dt_hit_flux_per_layer->Sumw2();
  h_dt_hit_rate_per_ch->Sumw2();
  if (do_dt_) for (int t=1; t<=DT_TYPES; t++)
  {
    // 2 endcaps , 6 layers
    double scale = bxrate * n_pu * f_full_bx /areas_.dt_total_areas_cm2[t] /evtn;
    double rt = scale * h_dt_hit_flux_per_layer->GetBinContent(t);
    double er = scale * h_dt_hit_flux_per_layer->GetBinError(t);
    h_dt_hit_flux_per_layer->SetBinContent(t,rt);
    h_dt_hit_flux_per_layer->SetBinError(t,er);

    scale = bxrate * n_pu * f_full_bx /dt_radial_segm[t]/evtn/1000.;
    rt = scale * h_dt_hit_rate_per_ch->GetBinContent(t);
    er = scale * h_dt_hit_rate_per_ch->GetBinError(t);
    h_dt_hit_rate_per_ch->SetBinContent(t,rt);
    h_dt_hit_rate_per_ch->SetBinError(t,er);

    for (int i=0; i<3; i++)
    {
      gr_dt_hit_flux_mb1->SetPoint(i, areas_.dt_ch_z[i+1], h_dt_hit_flux_per_layer->GetBinContent(i+1) );
      gr_dt_hit_flux_mb2->SetPoint(i, areas_.dt_ch_z[i+4], h_dt_hit_flux_per_layer->GetBinContent(i+4) );
      gr_dt_hit_flux_mb3->SetPoint(i, areas_.dt_ch_z[i+7], h_dt_hit_flux_per_layer->GetBinContent(i+7) );
      gr_dt_hit_flux_mb4->SetPoint(i, areas_.dt_ch_z[i+10], h_dt_hit_flux_per_layer->GetBinContent(i+10) );
      gr_dt_hit_flux_mb1->SetPointError(i, areas_.dt_ch_halfspanz[i+1], h_dt_hit_flux_per_layer->GetBinError(i+1) );
      gr_dt_hit_flux_mb2->SetPointError(i, areas_.dt_ch_halfspanz[i+4], h_dt_hit_flux_per_layer->GetBinError(i+4) );
      gr_dt_hit_flux_mb3->SetPointError(i, areas_.dt_ch_halfspanz[i+7], h_dt_hit_flux_per_layer->GetBinError(i+7) );
      gr_dt_hit_flux_mb4->SetPointError(i, areas_.dt_ch_halfspanz[i+10], h_dt_hit_flux_per_layer->GetBinError(i+10) );
    }
  }

}


// ================================================================================================
std::vector<std::vector<MyCSCSimHit> > MuSimHitOccupancy::clusterCSCHitsInLayer(std::vector<MyCSCSimHit> &hits)
{
  /* Does recursive clustering:
      - sort by WG,Strip,TOF
      - take first hit
      - cluster the other hits around it by requiring dWG<2 && dS<4 && dTOF<2ns
      - recursively call itself over the hits that didn't get into the cluster
  */
  using namespace std;

  vector< vector<MyCSCSimHit> > result;
  vector<MyCSCSimHit> cluster;
  size_t N = hits.size();
  cout<<" clusterCSC: #hits="<<N<<endl;
  if (N==0) {cout<<"  DONE 0???"<<endl;  return result;}

  sort(hits.begin(), hits.end());
  MyCSCSimHit sh1 = hits[0];
  cout<<"    hit  1: t="<<sh1.t<<" w="<<sh1.w<<" s="<<sh1.s<<endl;
  cluster.push_back(sh1);
  if (N==1)
  {
    result.push_back(cluster);
    cout<<"  DONE"<<endl;
    return result;
  }

  vector<MyCSCSimHit> not_clustered;
  for (size_t i=1; i<N; i++)
  {
    MyCSCSimHit shi = hits[i];
    cout<<"    hit  "<<i<<": t="<<shi.t<<" w="<<shi.w<<" s="<<shi.s<<"   ";
    if ( fabs(sh1.t - shi.t) > 2 || abs(sh1.w - shi.w) > 1 || abs(sh1.s - shi.s) > 3 ) {
      not_clustered.push_back(shi);
      cout<<"NO"<<endl;
    }
    else {cluster.push_back(shi); cout<<"ok"<<endl;}
  }
  result.push_back(cluster);
  cout<<"     made cluster of size "<<cluster.size()<<endl;

  if (not_clustered.size())
  {
    cout<<"   recursing..."<<endl;
    vector< vector<MyCSCSimHit> > result_recursive = clusterCSCHitsInLayer(not_clustered);
    result.insert(result.end(), result_recursive.begin(), result_recursive.end());
  }
  cout<<"   returning "<<result.size()<<" clusters"<<endl;
  return result;
}



// ================================================================================================
std::vector<std::vector<MyGEMSimHit> > MuSimHitOccupancy::clusterGEMHitsInPart(std::vector<MyGEMSimHit> &hits)
{
  /* Does recursive clustering:
      - sort by Strip,TOF
      - take first hit
      - cluster the other hits around it by requiring dS<4 && dTOF<4ns
      - recursively call itself over the hits that didn't get into the cluster
  */
  using namespace std;

  vector< vector<MyGEMSimHit> > result;
  vector<MyGEMSimHit> cluster;
  size_t N = hits.size();
  cout<<" clusterGEM: #hits="<<N<<endl;
  if (N==0) {cout<<"  DONE 0???"<<endl;  return result;}

  sort(hits.begin(), hits.end());
  MyGEMSimHit sh1 = hits[0];
  cout<<"    hit  1: t="<<sh1.t<<" s="<<sh1.s<<endl;
  cluster.push_back(sh1);
  if (N==1)
  {
    result.push_back(cluster);
    cout<<"  DONE"<<endl;
    return result;
  }

  vector<MyGEMSimHit> not_clustered;
  for (size_t i=1; i<N; i++)
  {
    MyGEMSimHit &shi = hits[i];
    cout<<"    hit  "<<i<<": t="<<shi.t<<" s="<<shi.s<<"   ";
    if ( fabs(sh1.t - shi.t) > 4. || abs(sh1.s - shi.s) > 3 ) {
      not_clustered.push_back(shi);
      cout<<"NO"<<endl;
    }
    else {cluster.push_back(shi); cout<<"ok"<<endl;}
  }
  result.push_back(cluster);
  cout<<"     made cluster of size "<<cluster.size()<<endl;

  if (not_clustered.size())
  {
    cout<<"   recursing..."<<endl;
    vector< vector<MyGEMSimHit> > result_recursive = clusterGEMHitsInPart(not_clustered);
    result.insert(result.end(), result_recursive.begin(), result_recursive.end());
  }
  cout<<"   returning "<<result.size()<<" clusters"<<endl;
  return result;
}

// ================================================================================================
std::vector<std::vector<MyRPCSimHit> > MuSimHitOccupancy::clusterRPCHitsInRoll(std::vector<MyRPCSimHit> &hits)
{
  /* Does recursive clustering:
      - sort by Strip,TOF
      - take first hit
      - cluster the other hits around it by requiring dS<3 && dTOF<2ns
      - recursively call itself over the hits that didn't get into the cluster
  */
  using namespace std;

  vector< vector<MyRPCSimHit> > result;
  vector<MyRPCSimHit> cluster;
  size_t N = hits.size();
  cout<<" clusterRPC: #hits="<<N<<endl;
  if (N==0) {cout<<"  DONE 0???"<<endl;  return result;}

  sort(hits.begin(), hits.end());
  MyRPCSimHit sh1 = hits[0];
  cout<<"    hit  1: t="<<sh1.t<<" s="<<sh1.s<<endl;
  cluster.push_back(sh1);
  if (N==1)
  {
    result.push_back(cluster);
    cout<<"  DONE"<<endl;
    return result;
  }

  vector<MyRPCSimHit> not_clustered;
  for (size_t i=1; i<N; i++)
  {
    MyRPCSimHit shi = hits[i];
    cout<<"    hit  "<<i<<": t="<<shi.t<<" s="<<shi.s<<"   ";
    if ( fabs(sh1.t - shi.t) > 2 || abs(sh1.s - shi.s) > 2 ) {
      not_clustered.push_back(shi);
      cout<<"NO"<<endl;
    }
    else {cluster.push_back(shi); cout<<"ok"<<endl;}
  }
  result.push_back(cluster);
  cout<<"     made cluster of size "<<cluster.size()<<endl;

  if (not_clustered.size())
  {
    cout<<"   recursing..."<<endl;
    vector< vector<MyRPCSimHit> > result_recursive = clusterRPCHitsInRoll(not_clustered);
    result.insert(result.end(), result_recursive.begin(), result_recursive.end());
  }
  cout<<"   returning "<<result.size()<<" clusters"<<endl;
  return result;
}


// ================================================================================================
//define this as a plug-in
DEFINE_FWK_MODULE(MuSimHitOccupancy);
