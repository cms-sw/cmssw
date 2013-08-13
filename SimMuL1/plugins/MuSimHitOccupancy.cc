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
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

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

#include <iomanip>

using std::cout;
using std::endl;

namespace
{
enum ETrigCSC {MAX_CSC_STATIONS = 4, CSC_TYPES = 10};
enum ETrigGEM {MAX_GEM_STATIONS = 1, GEM_TYPES = 1};
enum ETrigDT {MAX_DT_STATIONS = 4, DT_TYPES = 12};
enum ETrigRPCF {MAX_RPCF_STATIONS = 4, RPCF_TYPES = 12};
enum ETrigRPCB {MAX_RPCB_STATIONS = 4, RPCB_TYPES = 12};

int typeRPCb(RPCDetId &d) {return  3*d.station() + abs(d.ring()) - 2;}
int typeRPCf(RPCDetId &d) {return  3*d.station() + d.ring() - 3;}
int typeGEM(GEMDetId &d)  {return  3*d.station() + d.ring() - 3;}
int typeDT(DTWireId &d)   {return  3*d.station() + abs(d.wheel()) - 2;}

const std::string csc_type[CSC_TYPES+1] =
  { "all", "ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2"};
const std::string csc_type_[CSC_TYPES+1] =
  { "all", "ME1a", "ME1b", "ME12", "ME13", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42"};

const std::string gem_type[GEM_TYPES+1] =
  { "all", "GE1/1"};
const std::string gem_type_[GEM_TYPES+1] =
  { "all", "GE11"};

const std::string dt_type[DT_TYPES+1] =
  { "all", "MB1/0", "MB1/1", "MB1/2", "MB2/0", "MB2/1", "MB2/2", "MB3/0", "MB3/1", "MB3/2", "MB4/0", "MB4/1", "MB4/2",};
const std::string dt_type_[DT_TYPES+1] =
  { "all", "MB10", "MB11", "MB12", "MB20", "MB21", "MB22", "MB30", "MB31", "MB32", "MB40", "MB41", "MB42",};

const std::string rpcf_type[RPCF_TYPES+1] =
  { "all", "RE1/1", "RE1/2", "RE1/3", "RE2/1", "RE2/2", "RE2/3", "RE3/1", "RE3/2", "RE3/3", "RE4/1", "RE4/2", "RE4/3"};
const std::string rpcf_type_[RPCF_TYPES+1] =
  { "all", "RE11", "RE12", "RE13", "RE21", "RE22", "RE23", "RE31", "RE32", "RE33", "RE41", "RE42", "RE43"};

const std::string rpcb_type[RPCB_TYPES+1] =
  { "all", "RB1/0", "RB1/1", "RB1/2", "RB2/0", "RB2/1", "RB2/2", "RB3/0", "RB3/1", "RB3/2", "RB4/0", "RB4/1", "RB4/2",};
const std::string rpcb_type_[RPCB_TYPES+1] =
  { "all", "RB10", "RB11", "RB12", "RB20", "RB21", "RB22", "RB30", "RB31", "RB32", "RB40", "RB41", "RB42",};

// chamber radial segmentations (including factor of 2 for non-zero wheels in barrel):
const double csc_radial_segm[CSC_TYPES+1] = {1, 36, 36, 36, 36, 18, 36, 18, 36, 18, 36};
const double gem_radial_segm[GEM_TYPES+1] = {1, 36};
const double dt_radial_segm[DT_TYPES+1]   = {1, 12, 12*2, 12*2, 12, 12*2, 12*2, 12, 12*2, 12*2, 14, 14*2, 14*2};
const double rpcb_radial_segm[RPCF_TYPES+1] = {1, 12, 12*2, 12*2, 12, 12*2, 12*2, 24, 24*2, 24*2, 12, 24*2, 24*2};
const double rpcf_radial_segm[RPCF_TYPES+1] = {1, 36, 36, 36, 18, 36, 36, 18, 36, 36, 18, 36, 36};

// DT # of superlayers in chamber
const double dt_n_superlayers[DT_TYPES+1]   = {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2};

enum ENumPDG {N_PDGIDS=7};
const int pdg_ids[N_PDGIDS] = {0,11,13,211,321,2212,1000000000};
const std::string pdg_ids_names[N_PDGIDS]  = {"unknown","e","#mu","#pi","K","p","nuclei"};
const std::string pdg_ids_names_[N_PDGIDS] = {"","e","mu","pi","K","p","nucl"};
const int pdg_colors[N_PDGIDS] = {0, kBlack, kBlue, kGreen+1, kOrange-3, kMagenta, kRed};
const int pdg_markers[N_PDGIDS] = {0, 1, 24, 3, 5, 26, 2};
} // local namespace

//
// class declarations
//
// ================================================================================================

struct MyCSCDetId
{
  void init(CSCDetId &id);
  Short_t e, s, r, c, l;
  Short_t t; // type 1-10: ME1/a,1/b,1/2,1/3,2/1...4/2
};

struct MyCSCSimHit
{
  void init(PSimHit &sh, const CSCGeometry* csc_g, const ParticleDataTable * pdt);
  float eKin() {return sqrt(p*p + m*m) - m;}
  bool operator < (const MyCSCSimHit &rhs) const;
  Float_t x, y, z;    // local
  Float_t r, eta, phi, gx, gy, gz; // global
  Float_t e;          // energy deposit
  Float_t p;          // particle momentum
  Float_t m;          // particle mass
  Float_t t;          // TOF
  Int_t trid;         // trackId
  Int_t pdg;          // PDG
  Int_t w, s;         // WG & Strip
};


struct MyCSCCluster
{
  void init(std::vector<MyCSCSimHit> &shits);
  float eKin() {return sqrt(p*p + m*m) - m;}
  std::vector<MyCSCSimHit> hits;
  Int_t nh;             // # of hits
  Float_t r, eta, phi, gx, gy, gz; // globals fot 1st hit
  Float_t e;            // total energy deposit
  Float_t p, m;         // particle mass and initial momentum
  Float_t mint, maxt;   // min/max TOF
  Float_t meant, sigmat;// mean&stdev of TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t pdg;            // PDG
  Int_t minw, maxw, mins, maxs; // min/max WG & Strip
};

struct MyCSCLayer
{
  void init(int l, std::vector<MyCSCCluster> &sclusters);
  std::vector<MyCSCCluster> clusters;
  Int_t ln;             // layer #, not stored
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Float_t mint, maxt;   // min/max TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t minw, maxw, mins, maxs; // min/max WG & Strip
};

struct MyCSCChamber
{
  void init(std::vector<MyCSCLayer> &slayers);
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Int_t nl, l1, ln;     // # of layers, 1st and last layer #
  Float_t mint, maxt;   // min/max TOF
  Int_t minw, maxw, mins, maxs; // min/max WG & Strip
};

struct MyCSCEvent
{
  void init(std::vector<MyCSCChamber> &schambers);
  Int_t nh;        // #hits
  Int_t nclu;      // #clusters
  Int_t nch;       // #chambers w/ hits
  Int_t nch2, nch3, nch4, nch5, nch6; // #chambers w/ at least 2,3... hits
};


// ================================================================================================

struct MyGEMDetId
{
  void init(GEMDetId &id);
  Short_t reg, ring, st, layer, ch, part;
  Short_t t; // type 1: GE1/1
};


struct MyGEMSimHit
{
  void init(PSimHit &sh, const GEMGeometry* gem_g, const ParticleDataTable * pdt);
  float eKin() {return sqrt(p*p + m*m) - m;}
  bool operator < (const MyGEMSimHit &rhs) const;
  Float_t x, y, z;    // local
  Float_t r, eta, phi, gx, gy, gz; // global
  Float_t e;          // energy deposit
  Float_t p;          // particle momentum
  Float_t m;          // particle mass
  Float_t t;          // TOF
  Int_t trid;         // trackId
  Int_t pdg;          // PDG
  Int_t s;            // Strip
};


struct MyGEMCluster
{
  void init(std::vector<MyGEMSimHit> &shits);
  float eKin() {return sqrt(p*p + m*m) - m;}
  std::vector<MyGEMSimHit> hits;
  Int_t nh;             // # of hits
  Float_t r, eta, phi, gx, gy, gz; // globals fot 1st hit
  Float_t e;            // total energy deposit
  Float_t p, m;         // 1st particle mass and initial momentum
  Float_t mint, maxt;   // min/max TOF
  Float_t meant, sigmat;// mean&stdev of TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t pdg;            // PDG
  Int_t mins, maxs; // min/max strip
};


struct MyGEMPart
{
  void init(int r, int l, std::vector<MyGEMCluster> &sclusters);
  std::vector<MyGEMCluster> clusters;
  Int_t pn;             // partition #, not stored
  Int_t ln;             // layer #, not stored
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Float_t mint, maxt;   // min/max TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t mins, maxs;      // min/max strip
};


struct MyGEMChamber
{
  void init(std::vector<MyGEMPart> &sparts);
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Int_t np;             // # of partitions
  Int_t nl;             // # of layers
  Float_t mint, maxt;   // min/max TOF
};


struct MyGEMEvent
{
  void init(std::vector<MyGEMChamber> &schambers);
  Int_t nh;        // #hits
  Int_t nclu;      // #clusters
  Int_t np;        // #partitions
  Int_t nch;       // #chambers w/ hits
  //Short_t nch2, nch3, nch4, nch5, nch6; // #chambers w/ at least 2,3... hits
};


// ================================================================================================

struct MyRPCDetId
{
  void init(RPCDetId &id);
  Short_t reg, ring, st, sec, layer, subsec, roll;
  Short_t t; // type 1-8: RE1/2,1/3,2/2,2/3,3/2,3/3,4/2,4/3
};


struct MyRPCSimHit
{
  void init(PSimHit &sh, const RPCGeometry* rpc_g, const ParticleDataTable * pdt);
  float eKin() {return sqrt(p*p + m*m) - m;}
  bool operator < (const MyRPCSimHit &rhs) const;
  Float_t x, y, z;    // local
  Float_t r, eta, phi, gx, gy, gz; // global
  Float_t e;          // energy deposit
  Float_t p;          // particle momentum
  Float_t m;          // particle mass
  Float_t t;          // TOF
  Int_t trid;         // trackId
  Int_t pdg;          // PDG
  Int_t s;            // Strip
};


struct MyRPCCluster
{
  void init(std::vector<MyRPCSimHit> &shits);
  float eKin() {return sqrt(p*p + m*m) - m;}
  std::vector<MyRPCSimHit> hits;
  Int_t nh;             // # of hits
  Float_t r, eta, phi, gx, gy, gz; // globals fot 1st hit
  Float_t e;            // total energy deposit
  Float_t p, m;         // 1st particle mass and initial momentum
  Float_t mint, maxt;   // min/max TOF
  Float_t meant, sigmat;// mean&stdev of TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t pdg;            // PDG
  Int_t mins, maxs; // min/max strip
};


struct MyRPCRoll
{
  void init(int r, int l, std::vector<MyRPCCluster> &sclusters);
  std::vector<MyRPCCluster> clusters;
  Int_t rn;             // roll #, not stored
  Int_t ln;             // layer #, not stored
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Float_t mint, maxt;   // min/max TOF
  Int_t mintrid, maxtrid;// trackId
  Int_t mins, maxs;      // min/max strip
};


struct MyRPCChamber
{
  void init(std::vector<MyRPCRoll> &srolls);
  Int_t nh;             // # of hits
  Int_t nclu;           // # of clusters
  Int_t nr;             // # of rolls
  Int_t nl;             // # of layers
  Float_t mint, maxt;   // min/max TOF
};


struct MyRPCEvent
{
  void init(std::vector<MyRPCChamber> &schambers);
  Int_t nh;        // #hits
  Int_t nclu;      // #clusters
  Int_t nr;        // #rolls
  Int_t nch;       // #chambers w/ hits
  //Short_t nch2, nch3, nch4, nch5, nch6; // #chambers w/ at least 2,3... hits
};


// ================================================================================================

struct MyDTDetId
{
  void init(DTWireId &id);
  Short_t st, wh, sec, sl, l, wire;
  Short_t t; //
};


struct MyDTSimHit
{
  void init(PSimHit &sh, const DTGeometry* dt_g, const ParticleDataTable * pdt);
  float eKin() {return sqrt(p*p + m*m) - m;}
  Float_t x, y, z;    // local
  Float_t r, eta, phi, gx, gy, gz; // global
  Float_t e;          // energy deposit
  Float_t p;          // particle momentum
  Float_t m;          // particle mass
  Float_t t;          // TOF
  Int_t trid;         // trackId
  Int_t pdg;          // PDG
  //Int_t w, s;         // WG & Strip
};


// ================================================================================================

class MuSimHitOccupancy : public edm::EDAnalyzer {
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

  void calculateCSCDetectorAreas();
  void calculateGEMDetectorAreas();
  void calculateDTDetectorAreas();
  void calculateRPCDetectorAreas();

  float csc_total_areas_cm2[CSC_TYPES+1];
  float gem_total_areas_cm2[GEM_TYPES+1];
  float dt_total_areas_cm2[DT_TYPES+1];
  float rpcb_total_areas_cm2[RPCB_TYPES+1];
  float rpcf_total_areas_cm2[RPCF_TYPES+1];

  // vector index is over partitions
  std::vector<float> gem_total_part_areas_cm2[GEM_TYPES+1];
  std::vector<float> gem_part_radii[GEM_TYPES+1];

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

  //TGraphErrors *gr_gem_hit_flux_me1;

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

  h_csc_rz_sh_xray = fs->make<TH2D>("h_csc_rz_sh_xray",("CSC "+n_simhits+" #rho-z X-ray;z, cm;#rho, cm").c_str(),2060,550,1080,755,0,755);
  h_csc_rz_sh_heatmap = fs->make<TH2D>("h_csc_rz_sh_heatmap",("CSC "+n_simhits+" #rho-z;z, cm;#rho, cm").c_str(),220,541.46,1101.46,150,0,755);
  h_csc_rz_clu_heatmap = fs->make<TH2D>("h_csc_rz_clu_heatmap",("CSC "+n_clusters+" #rho-z;z, cm;#rho, cm").c_str(),220,541.46,1101.46,150,0,755);


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

  //gr_gem_hit_flux_me1 = fs->make<TGraphErrors>(2);
  //gr_gem_hit_flux_me1->SetName("gr_gem_hit_flux_me1");
  //gr_gem_hit_flux_me1->SetTitle("SimHit Flux in GE1;r, cm;Hz/cm^{2}");
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
  csc_sh_tree->Branch("id", &c_id.e,"e/S:s:r:c:l:t");
  csc_sh_tree->Branch("sh", &c_h.x,"x/F:y:z:r:eta:phi:gx:gy:gz:e:p:m:t:trid/I:pdg:w:s");
  //csc_sh_tree->Branch("", &., "/I");
  //csc_sh_tree->Branch("" , "vector<double>" , & );

  csc_cl_tree = fs->make<TTree>("CSCClustersTree", "CSCClustersTree");
  csc_cl_tree->Branch("evtn", &evtn,"evtn/i");
  csc_cl_tree->Branch("id", &c_id.e,"e/S:s:r:c:l:t");
  csc_cl_tree->Branch("cl", &c_cl.nh,"nh/I:r/F:eta:phi:gx:gy:gz:e:p:m:mint:maxt:meant:sigmat:mintrid/I:maxtrid:pdg/I:minw:maxw:mins:maxs");

  csc_la_tree = fs->make<TTree>("CSCLayersTree", "CSCLayersTree");
  csc_la_tree->Branch("evtn", &evtn,"evtn/i");
  csc_la_tree->Branch("id", &c_id.e,"e/S:s:r:c:l:t");
  csc_la_tree->Branch("la", &c_la.nh,"nh/I:nclu:mint/F:maxt:mintrid/I:maxtrid:minw:maxw:mins:maxs");

  csc_ch_tree = fs->make<TTree>("CSCChambersTree", "CSCChambersTree");
  csc_ch_tree->Branch("evtn", &evtn,"evtn/i");
  csc_ch_tree->Branch("id", &c_cid.e,"e/S:s:r:c:l:t");
  csc_ch_tree->Branch("ch", &c_ch.nh,"nh/I:nclu:nl:l1:ln:mint/F:maxt:minw/I:maxw:mins:maxs");

  csc_ev_tree = fs->make<TTree>("CSCEventsTree", "CSCEventsTree");
  csc_ev_tree->Branch("evtn", &evtn,"evtn/i");
  csc_ev_tree->Branch("ev", &c_ev.nh,"nh/I:nclu:nch:nch2:nch3:nch4:nch5:nch6");
}


// ================================================================================================
void
MuSimHitOccupancy::bookGEMSimHitsTrees()
{
  edm::Service<TFileService> fs;
  gem_sh_tree = fs->make<TTree>("GEMSimHitsTree", "GEMSimHitsTree");
  gem_sh_tree->Branch("evtn", &evtn,"evtn/i");
  gem_sh_tree->Branch("shn", &gem_shn,"shn/i");
  gem_sh_tree->Branch("id", &g_id.reg,"reg/S:ring:st:layer:ch:part:t");
  gem_sh_tree->Branch("sh", &g_h.x,"x/F:y:z:r:eta:phi:gx:gy:gz:e:p:m:t:trid/I:pdg:s");

  gem_cl_tree = fs->make<TTree>("GEMClustersTree", "GEMClustersTree");
  gem_cl_tree->Branch("evtn", &evtn,"evtn/i");
  gem_cl_tree->Branch("id", &g_id.reg,"reg/S:ring:st:layer:ch:part:t");
  gem_cl_tree->Branch("cl", &g_cl.nh,"nh/I:r/F:eta:phi:gx:gy:gz:e:p:m:mint:maxt:meant:sigmat:mintrid/I:maxtrid:pdg/I:mins:maxs");

  gem_part_tree = fs->make<TTree>("GEMPartTree", "GEMPartTree");
  gem_part_tree->Branch("evtn", &evtn,"evtn/i");
  gem_part_tree->Branch("id", &g_id.reg,"reg/S:ring:st:layer:ch:part:t");
  gem_part_tree->Branch("part", &g_part.nh,"nh/I:nclu:mint/F:maxt:mintrid/I:maxtrid:mins:maxs");

  gem_ch_tree = fs->make<TTree>("GEMChambersTree", "GEMChambersTree");
  gem_ch_tree->Branch("evtn", &evtn,"evtn/i");
  gem_ch_tree->Branch("id", &g_id.reg,"reg/S:ring:st:layer:ch:part:t");
  gem_ch_tree->Branch("ch", &g_ch.nh,"nh/I:nclu:np:nl:mint/F:maxt");

  gem_ev_tree = fs->make<TTree>("GEMEventsTree", "GEMEventsTree");
  gem_ev_tree->Branch("evtn", &evtn,"evtn/i");
  gem_ev_tree->Branch("ev", &g_ev.nh,"nh/I:nclu:np:nch");

  //gem_sh_tree->Branch("", &., "/I");
  //gem_sh_tree->Branch("" , "vector<double>" , & );
}


// ================================================================================================
void
MuSimHitOccupancy::bookRPCSimHitsTrees()
{
  edm::Service<TFileService> fs;
  rpc_sh_tree = fs->make<TTree>("RPCSimHitsTree", "RPCSimHitsTree");
  rpc_sh_tree->Branch("evtn", &evtn,"evtn/i");
  rpc_sh_tree->Branch("shn", &rpc_shn,"shn/i");
  rpc_sh_tree->Branch("id", &r_id.reg,"reg/S:ring:st:sec:layer:subsec:roll:t");
  rpc_sh_tree->Branch("sh", &r_h.x,"x/F:y:z:r:eta:phi:gx:gy:gz:e:p:m:t:trid/I:pdg:s");

  rpc_cl_tree = fs->make<TTree>("RPCClustersTree", "RPCClustersTree");
  rpc_cl_tree->Branch("evtn", &evtn,"evtn/i");
  rpc_cl_tree->Branch("id", &r_id.reg,"reg/S:ring:st:sec:layer:subsec:roll:t");
  rpc_cl_tree->Branch("cl", &r_cl.nh,"nh/I:r/F:eta:phi:gx:gy:gz:e:p:m:mint:maxt:meant:sigmat:mintrid/I:maxtrid:pdg/I:mins:maxs");

  rpc_rl_tree = fs->make<TTree>("RPCRollsTree", "RPCRollsTree");
  rpc_rl_tree->Branch("evtn", &evtn,"evtn/i");
  rpc_rl_tree->Branch("id", &r_id.reg,"reg/S:ring:st:sec:layer:subsec:roll:t");
  rpc_rl_tree->Branch("rl", &r_rl.nh,"nh/I:nclu:mint/F:maxt:mintrid/I:maxtrid:mins:maxs");

  rpc_ch_tree = fs->make<TTree>("RPCChambersTree", "RPCChambersTree");
  rpc_ch_tree->Branch("evtn", &evtn,"evtn/i");
  rpc_ch_tree->Branch("id", &r_id.reg,"reg/S:ring:st:sec:layer:subsec:roll:t");
  rpc_ch_tree->Branch("ch", &r_ch.nh,"nh/I:nclu:nr:nl:mint/F:maxt");

  rpc_ev_tree = fs->make<TTree>("RPCEventsTree", "RPCEventsTree");
  rpc_ev_tree->Branch("evtn", &evtn,"evtn/i");
  rpc_ev_tree->Branch("ev", &r_ev.nh,"nh/I:nclu:nr:nch");

  //rpc_sh_tree->Branch("", &., "/I");
  //rpc_sh_tree->Branch("" , "vector<double>" , & );
}


// ================================================================================================
void
MuSimHitOccupancy::bookDTSimHitsTrees()
{
  edm::Service<TFileService> fs;
  dt_sh_tree = fs->make<TTree>("DTSimHitsTree", "DTSimHitsTree");
  dt_sh_tree->Branch("evtn", &evtn,"evtn/i");
  dt_sh_tree->Branch("shn", &dt_shn,"shn/i");
  dt_sh_tree->Branch("id", &d_id.st,"st/I:wh:sec:sl:l:wire:t");
  dt_sh_tree->Branch("sh", &r_h.x,"x/F:y:z:r:eta:phi:gx:gy:gz:e:p:m:t:trid/I:pdg");
  //dt_sh_tree->Branch("", &., "/I");
  //dt_sh_tree->Branch("" , "vector<double>" , & );
}


// ================================================================================================
void
MyCSCDetId::init(CSCDetId &id)
{
  e = id.endcap();
  s = id.station();
  r = id.ring();
  c = id.chamber();
  l = id.layer();
  t = id.iChamberType();
}


// ================================================================================================
void
MyCSCSimHit::init(PSimHit &sh, const CSCGeometry* csc_g, const ParticleDataTable * pdt)
{
  LocalPoint hitLP = sh.localPosition();
  pdg = sh.particleType();
  m = 0.00051;

  ParticleData const *pdata = 0;
  HepPDT::ParticleID particleType(pdg);
  if (particleType.isValid()) pdata = pdt->particle(particleType);
  if (pdata)  m = pdata->mass();
  //  cout<<"  "<<pdg<<" "<<pdata->name()<<" "<<m;
  else cout<<" pdg not in PDT: "<< pdg<<endl;

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  p = sh.pabs();
  t = sh.tof();
  trid = sh.trackId();

  CSCDetId layerId(sh.detUnitId());
  const CSCLayer* csclayer = csc_g->layer(layerId);
  GlobalPoint hitGP = csclayer->toGlobal(hitLP);

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gx = hitGP.x();
  gy = hitGP.y();
  gz = hitGP.z();

  w = csclayer->geometry()->wireGroup(csclayer->geometry()->nearestWire(hitLP));
  s = csclayer->geometry()->nearestStrip(hitLP);
}


bool MyCSCSimHit::operator<(const MyCSCSimHit & rhs) const
{
  // first sort by wire group, then by strip, then by TOF
  if (w==rhs.w)
  {
    if (s==rhs.s) return t<rhs.t;
    else return s<rhs.s;
  }
  else return w<rhs.w;
}


// ================================================================================================
void
MyCSCCluster::init(std::vector<MyCSCSimHit> &shits)
{
  hits = shits;
  nh = hits.size();
  mint = 1000000000.;
  maxt = -1.;
  minw = mins = 1000;
  maxw = maxs = -1;
  meant = 0;
  sigmat = 0;
  if (nh==0) return;
  r = hits[0].r;
  eta = hits[0].eta;
  phi = hits[0].phi;
  gx = hits[0].gx;
  gy = hits[0].gy;
  gz = hits[0].gz;
  p = hits[0].p;
  m = hits[0].m;
  pdg = hits[0].pdg;
  e = 0;
  mintrid = 1000000000;
  maxtrid = -1;
  for (std::vector<MyCSCSimHit>::const_iterator itr = hits.begin(); itr != hits.end(); itr++)
  {
    MyCSCSimHit sh = *itr;
    e += sh.e;
    if (sh.t < mint) mint = sh.t;
    if (sh.t > maxt) maxt = sh.t;
    if (sh.w < minw) minw = sh.w;
    if (sh.w > maxw) maxw = sh.w;
    if (sh.s < mins) mins = sh.s;
    if (sh.s > maxs) maxs = sh.s;
    if (sh.trid < mintrid) mintrid = sh.trid;
    if (sh.trid > maxtrid) maxtrid = sh.trid;
    meant += sh.t;
    sigmat += sh.t*sh.t;
  }
  meant = meant/nh;
  sigmat = sqrt( sigmat/nh - meant*meant);
cout<<" clu: "<<nh<<" "<<mint<<" "<<minw<<" "<<meant<<" "<<r<<" "<<gz<<" "<<m<<" "<<endl;
}


// ================================================================================================
void
MyCSCLayer::init(int l, std::vector<MyCSCCluster> &sclusters)
{
  clusters = sclusters;
  ln = l;
  nclu = clusters.size();
  nh = 0;
  mint = 1000000000.;
  maxt = -1.;
  minw = mins = 1000;
  maxw = maxs = -1;
  mintrid = 1000000000;
  maxtrid = -1;
  if (nclu==0) return;
  for (std::vector<MyCSCCluster>::const_iterator itr = clusters.begin(); itr != clusters.end(); itr++)
  {
    MyCSCCluster cl = *itr;
    nh += cl.nh;
    if (cl.mint < mint) mint = cl.mint;
    if (cl.maxt > maxt) maxt = cl.maxt;
    if (cl.minw < minw) minw = cl.minw;
    if (cl.maxw > maxw) maxw = cl.maxw;
    if (cl.mins < mins) mins = cl.mins;
    if (cl.maxs > maxs) maxs = cl.maxs;
    if (cl.mintrid < mintrid) mintrid = cl.mintrid;
    if (cl.maxtrid > maxtrid) maxtrid = cl.maxtrid;
  }
}


// ================================================================================================
void
MyCSCChamber::init(std::vector<MyCSCLayer> &slayers)
{
  nh = nclu = 0;
  nl = slayers.size();
  mint = 1000000000.;
  maxt = -1.;
  minw = mins = 1000;
  maxw = maxs = -1;
  if (nl==0) return;
  l1 = 7;
  ln = -1;
  for (std::vector<MyCSCLayer>::const_iterator itr = slayers.begin(); itr != slayers.end(); itr++)
  {
    MyCSCLayer la = *itr;
    nh += la.nh;
    nclu += la.nclu;
    if (la.ln < l1) l1 = la.ln;
    if (la.ln > ln) ln = la.ln;
    if (la.mint < mint) mint = la.mint;
    if (la.maxt > maxt) maxt = la.maxt;
    if (la.minw < minw) minw = la.minw;
    if (la.maxw > maxw) maxw = la.maxw;
    if (la.mins < mins) mins = la.mins;
    if (la.maxs > maxs) maxs = la.maxs;

  }
}


// ================================================================================================
void
MyCSCEvent::init(std::vector<MyCSCChamber> &schambers)
{
  nch = schambers.size();
  nh = nclu = 0;
  nch2 = nch3 = nch4 = nch5 = nch6 = 0;
  for (std::vector<MyCSCChamber>::const_iterator itr = schambers.begin(); itr != schambers.end(); itr++)
  {
    MyCSCChamber ch = *itr;
    nh += ch.nh;
    nclu += ch.nclu;
    if (ch.nl>1) nch2++;
    if (ch.nl>2) nch3++;
    if (ch.nl>3) nch4++;
    if (ch.nl>4) nch5++;
    if (ch.nl>5) nch6++;
  }
}



// ================================================================================================
void
MyGEMDetId::init(GEMDetId &id)
{
  reg    = id.region();
  ring   = id.ring();
  st     = id.station();
  layer  = id.layer();
  ch     = id.chamber();
  part   = id.roll();
  t      = typeGEM(id);
}


// ================================================================================================
void
MyGEMSimHit::init(PSimHit &sh, const GEMGeometry* gem_g, const ParticleDataTable * pdt)
{
  LocalPoint hitLP = sh.localPosition();
  pdg = sh.particleType();
  m = 0.00051;

  ParticleData const *pdata = 0;
  HepPDT::ParticleID particleType(pdg);
  if (particleType.isValid()) pdata = pdt->particle(particleType);
  if (pdata)  m = pdata->mass();
  //  cout<<"  "<<pdg<<" "<<pdata->name()<<" "<<m;
  else cout<<" pdg not in PDT: "<< pdg<<endl;

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  p = sh.pabs();
  t = sh.tof();
  trid = sh.trackId();

  GlobalPoint hitGP = gem_g->idToDet(DetId(sh.detUnitId()))->surface().toGlobal(hitLP);

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gx = hitGP.x();
  gy = hitGP.y();
  gz = hitGP.z();

  GEMDetId rollId(sh.detUnitId());
  s = gem_g->etaPartition(rollId)->strip(hitLP);
}


bool MyGEMSimHit::operator<(const MyGEMSimHit & rhs) const
{
  // first sort by strip, then by TOF
  if (s==rhs.s) return t<rhs.t;
  else return s<rhs.s;
}


// ================================================================================================
void
MyGEMCluster::init(std::vector<MyGEMSimHit> &shits)
{
  hits = shits;
  nh = hits.size();
  mint = 1000000000.;
  maxt = -1.;
  mins = 1000;
  maxs = -1;
  meant = 0;
  sigmat = 0;
  if (nh==0) return;
  r = hits[0].r;
  eta = hits[0].eta;
  phi = hits[0].phi;
  gx = hits[0].gx;
  gy = hits[0].gy;
  gz = hits[0].gz;
  p = hits[0].p;
  m = hits[0].m;
  pdg = hits[0].pdg;
  e = 0;
  mintrid = 1000000000;
  maxtrid = -1;
  for (std::vector<MyGEMSimHit>::const_iterator itr = hits.begin(); itr != hits.end(); itr++)
  {
    MyGEMSimHit sh = *itr;
    e += sh.e;
    if (sh.t < mint) mint = sh.t;
    if (sh.t > maxt) maxt = sh.t;
    if (sh.s < mins) mins = sh.s;
    if (sh.s > maxs) maxs = sh.s;
    if (sh.trid < mintrid) mintrid = sh.trid;
    if (sh.trid > maxtrid) maxtrid = sh.trid;
    meant += sh.t;
    sigmat += sh.t*sh.t;
  }
  meant = meant/nh;
  sigmat = sqrt( sigmat/nh - meant*meant);
  cout<<" gem clu: "<<nh<<" "<<mint<<" "<<mins<<" "<<meant<<" "<<r<<" "<<gz<<" "<<m<<" "<<endl;
}


// ================================================================================================
void
MyGEMPart::init(int p, int l, std::vector<MyGEMCluster> &sclusters)
{
  clusters = sclusters;
  pn = p;
  ln = l;
  nclu = clusters.size();
  nh = 0;
  mint = 1000000000.;
  maxt = -1.;
  mins = 1000;
  maxs = -1;
  mintrid = 1000000000;
  maxtrid = -1;
  if (nclu==0) return;
  for (std::vector<MyGEMCluster>::const_iterator itr = clusters.begin(); itr != clusters.end(); itr++)
  {
    MyGEMCluster cl = *itr;
    nh += cl.nh;
    if (cl.mint < mint) mint = cl.mint;
    if (cl.maxt > maxt) maxt = cl.maxt;
    if (cl.mins < mins) mins = cl.mins;
    if (cl.maxs > maxs) maxs = cl.maxs;
    if (cl.mintrid < mintrid) mintrid = cl.mintrid;
    if (cl.maxtrid > maxtrid) maxtrid = cl.maxtrid;
  }
}


// ================================================================================================
void
MyGEMChamber::init(std::vector<MyGEMPart> &sparts)
{
  nh = nclu = nl = 0;
  np = sparts.size();
  mint = 1000000000.;
  maxt = -1.;
  if (np==0) return;
  std::set<int> layers;
  for (std::vector<MyGEMPart>::const_iterator itr = sparts.begin(); itr != sparts.end(); itr++)
  {
    MyGEMPart rl = *itr;
    nh += rl.nh;
    nclu += rl.nclu;
    layers.insert(rl.ln);
    if (rl.mint < mint) mint = rl.mint;
    if (rl.maxt > maxt) maxt = rl.maxt;
  }
  nl = layers.size();
}


// ================================================================================================
void
MyGEMEvent::init(std::vector<MyGEMChamber> &schambers)
{
  nch = schambers.size();
  nh = nclu = np = 0;
  for (std::vector<MyGEMChamber>::const_iterator itr = schambers.begin(); itr != schambers.end(); itr++)
  {
    MyGEMChamber ch = *itr;
    nh += ch.nh;
    nclu += ch.nclu;
    np += ch.np;
  }
}


// ================================================================================================
void
MyRPCDetId::init(RPCDetId &id)
{
  reg    = id.region();
  ring   = id.ring();
  st     = id.station();
  sec    = id.sector();
  layer  = id.layer();
  subsec = id.subsector();
  roll   = id.roll();
  if (reg!=0) t = typeRPCf(id);
  else t = typeRPCb(id);
}


// ================================================================================================
void
MyRPCSimHit::init(PSimHit &sh, const RPCGeometry* rpc_g, const ParticleDataTable * pdt)
{
  LocalPoint hitLP = sh.localPosition();
  pdg = sh.particleType();
  m = 0.00051;

  ParticleData const *pdata = 0;
  HepPDT::ParticleID particleType(pdg);
  if (particleType.isValid()) pdata = pdt->particle(particleType);
  if (pdata)  m = pdata->mass();
  //  cout<<"  "<<pdg<<" "<<pdata->name()<<" "<<m;
  else cout<<" pdg not in PDT: "<< pdg<<endl;

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  p = sh.pabs();
  t = sh.tof();
  trid = sh.trackId();

  GlobalPoint hitGP = rpc_g->idToDet(DetId(sh.detUnitId()))->surface().toGlobal(hitLP);

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gx = hitGP.x();
  gy = hitGP.y();
  gz = hitGP.z();

  RPCDetId rollId(sh.detUnitId());
  s = rpc_g->roll(rollId)->strip(hitLP);
}


bool MyRPCSimHit::operator<(const MyRPCSimHit & rhs) const
{
  // first sort by strip, then by TOF
  if (s==rhs.s) return t<rhs.t;
  else return s<rhs.s;
}


// ================================================================================================
void
MyRPCCluster::init(std::vector<MyRPCSimHit> &shits)
{
  hits = shits;
  nh = hits.size();
  mint = 1000000000.;
  maxt = -1.;
  mins = 1000;
  maxs = -1;
  meant = 0;
  sigmat = 0;
  if (nh==0) return;
  r = hits[0].r;
  eta = hits[0].eta;
  phi = hits[0].phi;
  gx = hits[0].gx;
  gy = hits[0].gy;
  gz = hits[0].gz;
  p = hits[0].p;
  m = hits[0].m;
  pdg = hits[0].pdg;
  e = 0;
  mintrid = 1000000000;
  maxtrid = -1;
  for (std::vector<MyRPCSimHit>::const_iterator itr = hits.begin(); itr != hits.end(); itr++)
  {
    MyRPCSimHit sh = *itr;
    e += sh.e;
    if (sh.t < mint) mint = sh.t;
    if (sh.t > maxt) maxt = sh.t;
    if (sh.s < mins) mins = sh.s;
    if (sh.s > maxs) maxs = sh.s;
    if (sh.trid < mintrid) mintrid = sh.trid;
    if (sh.trid > maxtrid) maxtrid = sh.trid;
    meant += sh.t;
    sigmat += sh.t*sh.t;
  }
  meant = meant/nh;
  sigmat = sqrt( sigmat/nh - meant*meant);
  cout<<" rpc clu: "<<nh<<" "<<mint<<" "<<mins<<" "<<meant<<" "<<r<<" "<<gz<<" "<<m<<" "<<endl;
}


// ================================================================================================
void
MyRPCRoll::init(int r, int l, std::vector<MyRPCCluster> &sclusters)
{
  clusters = sclusters;
  rn = r;
  ln = l;
  nclu = clusters.size();
  nh = 0;
  mint = 1000000000.;
  maxt = -1.;
  mins = 1000;
  maxs = -1;
  mintrid = 1000000000;
  maxtrid = -1;
  if (nclu==0) return;
  for (std::vector<MyRPCCluster>::const_iterator itr = clusters.begin(); itr != clusters.end(); itr++)
  {
    MyRPCCluster cl = *itr;
    nh += cl.nh;
    if (cl.mint < mint) mint = cl.mint;
    if (cl.maxt > maxt) maxt = cl.maxt;
    if (cl.mins < mins) mins = cl.mins;
    if (cl.maxs > maxs) maxs = cl.maxs;
    if (cl.mintrid < mintrid) mintrid = cl.mintrid;
    if (cl.maxtrid > maxtrid) maxtrid = cl.maxtrid;
  }
}


// ================================================================================================
void
MyRPCChamber::init(std::vector<MyRPCRoll> &srolls)
{
  nh = nclu = nl = 0;
  nr = srolls.size();
  mint = 1000000000.;
  maxt = -1.;
  if (nr==0) return;
  std::set<int> layers;
  for (std::vector<MyRPCRoll>::const_iterator itr = srolls.begin(); itr != srolls.end(); itr++)
  {
    MyRPCRoll rl = *itr;
    nh += rl.nh;
    nclu += rl.nclu;
    layers.insert(rl.ln);
    if (rl.mint < mint) mint = rl.mint;
    if (rl.maxt > maxt) maxt = rl.maxt;
  }
  nl = layers.size();
}


// ================================================================================================
void
MyRPCEvent::init(std::vector<MyRPCChamber> &schambers)
{
  nch = schambers.size();
  nh = nclu = nr = 0;
  for (std::vector<MyRPCChamber>::const_iterator itr = schambers.begin(); itr != schambers.end(); itr++)
  {
    MyRPCChamber ch = *itr;
    nh += ch.nh;
    nclu += ch.nclu;
    nr += ch.nr;
  }
}


// ================================================================================================
void
MyDTDetId::init(DTWireId &id)
{
  st     = id.station();
  wh     = id.wheel();
  sec    = id.sector();
  sl     = id.superLayer();
  l      = id.layer();
  wire   = id.wire();
  t = typeDT(id);
}


// ================================================================================================
void
MyDTSimHit::init(PSimHit &sh, const DTGeometry* dt_g, const ParticleDataTable * pdt)
{
  LocalPoint hitLP = sh.localPosition();
  pdg = sh.particleType();
  m = 0.00051;

  ParticleData const *pdata = 0;
  HepPDT::ParticleID particleType(pdg);
  if (particleType.isValid()) pdata = pdt->particle(particleType);
  if (pdata)  m = pdata->mass();
  //  cout<<"  "<<pdg<<" "<<pdata->name()<<" "<<m;
  else cout<<" pdg not in PDT: "<< pdg<<endl;

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  p = sh.pabs();
  t = sh.tof();
  trid = sh.trackId();

  GlobalPoint hitGP = dt_g->idToDet(DetId(sh.detUnitId()))->surface().toGlobal(hitLP);

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gx = hitGP.x();
  gy = hitGP.y();
  gz = hitGP.z();

  //w = csclayer->geometry()->wireGroup(csclayer->geometry()->nearestWire(hitLP));
  //s = csclayer->geometry()->nearestStrip(hitLP);
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

    if (evtn==1) calculateCSCDetectorAreas();

    // get SimHits
    simhit_map_csc.fill(iEvent);
   
    analyzeCSC();
  }

  if (do_gem_)
  {
    ESHandle< GEMGeometry > gem_geom;
    iSetup.get< MuonGeometryRecord >().get(gem_geom);
    gem_geometry = &*gem_geom;

    if (evtn==1) calculateGEMDetectorAreas();

    // get SimHits
    simhit_map_gem.fill(iEvent);
   
    analyzeGEM();
  }

  if (do_rpc_)
  {
    ESHandle< RPCGeometry > rpc_geom;
    iSetup.get< MuonGeometryRecord >().get(rpc_geom);
    rpc_geometry = &*rpc_geom;

    if (evtn==1) calculateRPCDetectorAreas();

    // get SimHits
    simhit_map_rpc.fill(iEvent);

    analyzeRPC();
  }

  if (do_dt_)
  {
    ESHandle< DTGeometry > dt_geom;
    iSetup.get< MuonGeometryRecord >().get(dt_geom);
    dt_geometry = &*dt_geom;

    if (evtn==1) calculateDTDetectorAreas();

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
void MuSimHitOccupancy::calculateCSCDetectorAreas()
{
  for (int i=0; i<=CSC_TYPES; i++) csc_total_areas_cm2[i]=0.;

  for(std::vector<CSCLayer*>::const_iterator it = csc_geometry->layers().begin(); it != csc_geometry->layers().end(); it++)
  {
    if( dynamic_cast<CSCLayer*>( *it ) == 0 ) continue;

    CSCLayer* layer = dynamic_cast<CSCLayer*>( *it );
    CSCDetId id = layer->id();
    int ctype = id.iChamberType();

    const CSCWireTopology*  wire_topo  = layer->geometry()->wireTopology();
    const CSCStripTopology* strip_topo = layer->geometry()->topology();
    float b = wire_topo->narrowWidthOfPlane();
    float t = wire_topo->wideWidthOfPlane();
    float w_h   = wire_topo->lengthOfPlane();
    float s_h = fabs(strip_topo->yLimitsOfStripPlane().first - strip_topo->yLimitsOfStripPlane().second);
    float h = (w_h < s_h)? w_h : s_h;

    // special cases:
    if (ctype==1) // ME1/1a
    {
      h += -0.5; // adjustment in order to agree with the official me1a height number
      t = ( b*(w_h - h) + t*h )/w_h;
    }
    if (ctype==2) // ME1/1a
    {
      h += -1.;
      b = ( b*h + t*(w_h - h) )/w_h;
    }

    float layer_area = h*(t + b)*0.5;
    csc_total_areas_cm2[0] += layer_area;
    csc_total_areas_cm2[ctype] += layer_area;

    if (id.layer()==1) cout<<"CSC type "<<ctype<<"  "<<id<<"  layer area: "<<layer_area<<" cm2   "
        <<"  b="<<b<<" t="<<t<<" h="<<h<<"  w_h="<<w_h<<" s_h="<<s_h<<endl;
  }
  cout<<"========================"<<endl;
  cout<<"= CSC chamber sensitive areas per layer (cm2):"<<endl;
  for (int i=1; i<=CSC_TYPES; i++) cout<<"= "<<csc_type[i]<<" "<<csc_total_areas_cm2[i]/6./2./csc_radial_segm[i]<<endl;
  cout<<"========================"<<endl;
}


// ================================================================================================
void MuSimHitOccupancy::calculateGEMDetectorAreas()
{
  std::vector<float> emptyv(12, 0.);
  float minr[GEM_TYPES+1], maxr[GEM_TYPES+1];
  for (int i=0; i<=GEM_TYPES; i++) 
  {
    gem_total_areas_cm2[i]=0.;
    gem_total_part_areas_cm2[i] = emptyv;
    gem_part_radii[i] = emptyv;
    minr[i] = 9999.;
    maxr[i] = 0.;
  }


  auto etaPartitions = gem_geometry->etaPartitions();
  for(auto p: etaPartitions)
  {
    GEMDetId id = p->id();
    int t = typeGEM(id);
    int part = id.roll();

    const TrapezoidalStripTopology* top = dynamic_cast<const TrapezoidalStripTopology*>(&(p->topology()));
    float xmin = top->localPosition(0.).x();
    float xmax = top->localPosition((float)p->nstrips()).x();
    float rollarea = top->stripLength() * (xmax - xmin);
    gem_total_areas_cm2[0] += rollarea;
    gem_total_areas_cm2[typeGEM(id)] += rollarea;
    gem_total_part_areas_cm2[0][0] += rollarea;
    gem_total_part_areas_cm2[t][part] += rollarea;
    cout<<"Partition: "<<id.rawId()<<" "<<id<<" area: "<<rollarea<<" cm2"<<endl;

    GlobalPoint gp = gem_geometry->idToDet(id)->surface().toGlobal(LocalPoint(0.,0.,0.));
    gem_part_radii[t][part] = gp.perp();

    if (maxr[t] < gp.perp() + top->stripLength()/2.) maxr[t] = gp.perp() + top->stripLength()/2.;
    if (minr[t] > gp.perp() - top->stripLength()/2.) minr[t] = gp.perp() - top->stripLength()/2.;
  }

  for (int t=1; t<=GEM_TYPES; t++)
  {
    gem_part_radii[t][0] = (minr[t] + maxr[t])/2.;
  }

  cout<<"========================"<<endl;
  cout<<"= GEM chamber sensitive areas per layer (cm2):"<<endl;
  for (int i=0; i<=GEM_TYPES; i++) cout<<"= "<<gem_type[i]<<" "<<rpcf_total_areas_cm2[i]/2./2./rpcf_radial_segm[i]<<endl;
  cout<<"========================"<<endl;
}



// ================================================================================================
void MuSimHitOccupancy::calculateDTDetectorAreas()
{
  for (int i=0; i<=DT_TYPES; i++) dt_total_areas_cm2[i]=0.;

  for(std::vector<DTLayer*>::const_iterator it = dt_geometry->layers().begin(); it != dt_geometry->layers().end(); it++)
  {
    if( dynamic_cast<DTLayer*>( *it ) == 0 ) continue;

    DTLayer* layer = dynamic_cast<DTLayer*>( *it );
    DTWireId id = (DTWireId) layer->id();
    int ctype = typeDT(id);

    const DTTopology& topo = layer->specificTopology();
    // cell's sensible width * # cells
    float w = topo.sensibleWidth() * topo.channels();
    float l = topo.cellLenght();

    float layer_area = w*l;
    dt_total_areas_cm2[0] += layer_area;
    dt_total_areas_cm2[ctype] += layer_area;

    if (id.layer()==1) cout<<"DT type "<<ctype<<"  "<<id<<"  layer area: "<<layer_area<<" cm2   "
        <<"  w="<<w<<" l="<<l<<" ncells="<<topo.channels()<<endl;
  }

  cout<<"========================"<<endl;
  cout<<"= DT *total* sensitive areas (cm2):"<<endl;
  for (int i=0; i<=DT_TYPES; i++) cout<<"= "<<dt_type[i]<<" "<<dt_total_areas_cm2[i]<<endl;
  cout<<"========================"<<endl;
}


// ================================================================================================
void MuSimHitOccupancy::calculateRPCDetectorAreas()
{
  for (int i=0; i<=RPCB_TYPES; i++) rpcb_total_areas_cm2[i]=0.;
  for (int i=0; i<=RPCF_TYPES; i++) rpcf_total_areas_cm2[i]=0.;

  // adapted from Piet's RPCGeomAnalyzer

  for(std::vector<RPCRoll*>::const_iterator it = rpc_geometry->rolls().begin(); it != rpc_geometry->rolls().end(); it++)
  {
    if( dynamic_cast<RPCRoll*>( *it ) != 0 ) { // check if dynamic cast is ok: cast ok => 1
      RPCRoll* roll = dynamic_cast<RPCRoll*>( *it );
      RPCDetId id = roll->id();
      //RPCGeomServ rpcsrv(detId);
      //std::string name = rpcsrv.name();
      if (id.region() == 0) {
        const RectangularStripTopology* top_ = dynamic_cast<const RectangularStripTopology*>(&(roll->topology()));
        float xmin = (top_->localPosition(0.)).x();
        float xmax = (top_->localPosition((float)roll->nstrips())).x();
        float rollarea = top_->stripLength() * (xmax - xmin);
        rpcb_total_areas_cm2[0] += rollarea;
        rpcb_total_areas_cm2[typeRPCb(id)] += rollarea;
        // cout<<"Roll: RawId: "<<id.rawId()<<" Name: "<<name<<" RPCDetId: "<<id<<" rollarea: "<<rollarea<<" cm2"<<endl;
      }
      else
      {
        const TrapezoidalStripTopology* top_=dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology()));
        float xmin = (top_->localPosition(0.)).x();
        float xmax = (top_->localPosition((float)roll->nstrips())).x();
        float rollarea = top_->stripLength() * (xmax - xmin);
        rpcf_total_areas_cm2[0] += rollarea;
        rpcf_total_areas_cm2[typeRPCf(id)] += rollarea;
        // cout<<"Roll: RawId: "<<id.rawId()<<" Name: "<<name<<" RPCDetId: "<<id<<" rollarea: "<<rollarea<<" cm2"<<endl;
      }
    }
  }
  cout<<"========================"<<endl;
  cout<<"= RPCb *total* sensitive areas (cm2):"<<endl;
  for (int i=0; i<=RPCB_TYPES; i++) cout<<"= "<<rpcb_type[i]<<" "<<rpcb_total_areas_cm2[i]<<endl;
  cout<<"= RPCf chamber sensitive areas (cm2):"<<endl;
  for (int i=0; i<=RPCF_TYPES; i++) cout<<"= "<<rpcf_type[i]<<" "<<rpcf_total_areas_cm2[i]/2./rpcf_radial_segm[i]<<endl;
  cout<<"========================"<<endl;
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

  //  chamber areas
  //const double csc_areas_cm2[CSC_TYPES+1] =
  //  {10000000, 1068.32, 4108.77, 10872.75, 13559.025, 16986, 31121.2, 15716.4, 31121.2, 14542.5, 31121.2};
  h_csc_hit_flux_per_layer->Sumw2();
  h_csc_hit_rate_per_ch->Sumw2();
  h_csc_clu_flux_per_layer->Sumw2();
  h_csc_clu_rate_per_ch->Sumw2();
  if (do_csc_) for (int t=1; t<=CSC_TYPES; t++)
  {
    // 2 endcaps , 6 layers
    double scale = bxrate * n_pu * f_full_bx /csc_total_areas_cm2[t]/evtn;
    double rt = scale * h_csc_hit_flux_per_layer->GetBinContent(t);
    double er = scale * h_csc_hit_flux_per_layer->GetBinError(t);
    h_csc_hit_flux_per_layer->SetBinContent(t,rt);
    h_csc_hit_flux_per_layer->SetBinError(t,er);

    rt = scale * h_csc_clu_flux_per_layer->GetBinContent(t);
    er = scale * h_csc_clu_flux_per_layer->GetBinError(t);
    h_csc_clu_flux_per_layer->SetBinContent(t,rt);
    h_csc_clu_flux_per_layer->SetBinError(t,er);

    scale = bxrate * n_pu * f_full_bx /csc_radial_segm[t]/2/evtn/1000;
    rt = scale * h_csc_hit_rate_per_ch->GetBinContent(t);
    er = scale * h_csc_hit_rate_per_ch->GetBinError(t);
    h_csc_hit_rate_per_ch->SetBinContent(t,rt);
    h_csc_hit_rate_per_ch->SetBinError(t,er);

    rt = scale * h_csc_clu_rate_per_ch->GetBinContent(t);
    er = scale * h_csc_clu_rate_per_ch->GetBinError(t);
    h_csc_clu_rate_per_ch->SetBinContent(t,rt);
    h_csc_clu_rate_per_ch->SetBinError(t,er);


    // centers of ME stations in r
    const Double_t xx[4][4] = {
        {128., 203.25, 369.75, 594.1}, {239.05, 525.55, 0, 0}, {251.75, 525.55, 0, 0}, {261.7, 525.55, 0, 0}};
    // half-spans of ME stations in r
    const Double_t xe[4][4] = {
        {22., 53.25, 87.25, 82.1}, {94.85, 161.55, 0, 0}, {84.85, 161.55, 0, 0}, {74.7, 161.55, 0, 0}};
    // fluxes and errors
    const Double_t yy[4][4] = {
        { h_csc_hit_flux_per_layer->GetBinContent(1), h_csc_hit_flux_per_layer->GetBinContent(2),
          h_csc_hit_flux_per_layer->GetBinContent(3), h_csc_hit_flux_per_layer->GetBinContent(4)},
        { h_csc_hit_flux_per_layer->GetBinContent(5), h_csc_hit_flux_per_layer->GetBinContent(6), 0, 0},
        { h_csc_hit_flux_per_layer->GetBinContent(7), h_csc_hit_flux_per_layer->GetBinContent(8), 0, 0},
        { h_csc_hit_flux_per_layer->GetBinContent(9), h_csc_hit_flux_per_layer->GetBinContent(10), 0, 0} };
    const Double_t ye[4][4] = {
        { h_csc_hit_flux_per_layer->GetBinError(1), h_csc_hit_flux_per_layer->GetBinError(2),
          h_csc_hit_flux_per_layer->GetBinError(3), h_csc_hit_flux_per_layer->GetBinError(4)},
        { h_csc_hit_flux_per_layer->GetBinError(5), h_csc_hit_flux_per_layer->GetBinError(6), 0, 0},
        { h_csc_hit_flux_per_layer->GetBinError(7), h_csc_hit_flux_per_layer->GetBinError(8), 0, 0},
        { h_csc_hit_flux_per_layer->GetBinError(9), h_csc_hit_flux_per_layer->GetBinError(10), 0, 0} };
    for (int i=0; i<4; i++)
    {
      gr_csc_hit_flux_me1->SetPoint(i, xx[0][i], yy[0][i]);
      gr_csc_hit_flux_me1->SetPointError(i, xe[0][i], ye[0][i]);
      if (i>1) continue;
      gr_csc_hit_flux_me2->SetPoint(i, xx[1][i], yy[1][i]);
      gr_csc_hit_flux_me3->SetPoint(i, xx[2][i], yy[2][i]);
      gr_csc_hit_flux_me4->SetPoint(i, xx[3][i], yy[3][i]);
      gr_csc_hit_flux_me2->SetPointError(i, xe[1][i], ye[1][i]);
      gr_csc_hit_flux_me3->SetPointError(i, xe[2][i], ye[2][i]);
      gr_csc_hit_flux_me4->SetPointError(i, xe[3][i], ye[3][i]);
    }
  }


  h_gem_hit_flux_per_layer->Sumw2();
  h_gem_hit_rate_per_ch->Sumw2();
  if (do_gem_) for (int t=1; t<=GEM_TYPES; t++)
  {
    double scale = bxrate * n_pu * f_full_bx /gem_total_areas_cm2[t]/evtn;

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
  
  //const double rpcf_areas_cm2[RPCF_TYPES+1] = {100000, 3150,11700,17360,11070,11690,19660,7330,11690,19660,5330,11690,19660};
  //const double rpcf_total_areas_cm2[RPCF_TYPES+1] =
  //  { 10000000, 244093, 738256, 1289290, 397166, 854259, 1418920, 276969, 854259, 1418920, 204059, 854259, 1418920};

  h_rpcf_hit_flux_per_layer->Sumw2();
  h_rpcf_hit_rate_per_ch->Sumw2();
  if (do_rpc_) for (int t=1; t<=RPCF_TYPES; t++)
  {
    double scale = bxrate * n_pu * f_full_bx /rpcf_total_areas_cm2[t]/evtn;
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
  
  // chambers with two layers have their area doubled:
  //const double rpcb_total_areas_cm2[RPCF_TYPES+1] =
  //  { 10000000, 1196940, 2361750, 2393880, 1439190, 2839700, 2878390, 859939, 1696790, 1719880, 1102040, 2163920, 2204080};
  h_rpcb_hit_flux_per_layer->Sumw2();
  if (do_rpc_) for (int t=1; t<=RPCB_TYPES; t++)
  {
    double scale = bxrate * n_pu * f_full_bx /rpcb_total_areas_cm2[t]/evtn;
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


    //  chamber areas
  //const double dt_total_areas_cm2[DT_TYPES+1] =
  //  {1.65696e+08,6.73021e+06,1.32739e+07,1.34604e+07,8.20906e+06,1.61906e+07,1.64182e+07,9.88901e+06,1.95039e+07,1.9778e+07,8.50898e+06,1.67151e+07,1.7018e+07};
  h_dt_hit_flux_per_layer->Sumw2();
  h_dt_hit_rate_per_ch->Sumw2();
  if (do_dt_) for (int t=1; t<=DT_TYPES; t++)
  {
    // 2 endcaps , 6 layers
    double scale = bxrate * n_pu * f_full_bx /dt_total_areas_cm2[t] /evtn;
    double rt = scale * h_dt_hit_flux_per_layer->GetBinContent(t);
    double er = scale * h_dt_hit_flux_per_layer->GetBinError(t);
    h_dt_hit_flux_per_layer->SetBinContent(t,rt);
    h_dt_hit_flux_per_layer->SetBinError(t,er);

    scale = bxrate * n_pu * f_full_bx /dt_radial_segm[t]/evtn/1000;
    rt = scale * h_dt_hit_rate_per_ch->GetBinContent(t);
    er = scale * h_dt_hit_rate_per_ch->GetBinError(t);
    h_dt_hit_rate_per_ch->SetBinContent(t,rt);
    h_dt_hit_rate_per_ch->SetBinError(t,er);

    // centers of MB stations in |z|
    const Double_t xx[4][3] = {{58.7, 273, 528}, {58.7, 273, 528}, {58.7, 273, 528}, {58.7, 273, 528}};
    // half-spans of MB stations in |z|
    const Double_t xe[4][3] = {{58.7, 117.4, 117.4}, {58.7, 117.4, 117.4}, {58.7, 117.4, 117.4}, {58.7, 117.4, 117.4}};
    // fluxes and errors
    const Double_t yy[4][3] = {
        { h_dt_hit_flux_per_layer->GetBinContent(1), h_dt_hit_flux_per_layer->GetBinContent(2), h_dt_hit_flux_per_layer->GetBinContent(3)},
        { h_dt_hit_flux_per_layer->GetBinContent(4), h_dt_hit_flux_per_layer->GetBinContent(5), h_dt_hit_flux_per_layer->GetBinContent(6)},
        { h_dt_hit_flux_per_layer->GetBinContent(7), h_dt_hit_flux_per_layer->GetBinContent(8), h_dt_hit_flux_per_layer->GetBinContent(9)},
        { h_dt_hit_flux_per_layer->GetBinContent(10), h_dt_hit_flux_per_layer->GetBinContent(11), h_dt_hit_flux_per_layer->GetBinContent(12)}
    };
    const Double_t ye[4][3] = {
        { h_dt_hit_flux_per_layer->GetBinError(1), h_dt_hit_flux_per_layer->GetBinError(2), h_dt_hit_flux_per_layer->GetBinError(3)},
        { h_dt_hit_flux_per_layer->GetBinError(4), h_dt_hit_flux_per_layer->GetBinError(5), h_dt_hit_flux_per_layer->GetBinError(6)},
        { h_dt_hit_flux_per_layer->GetBinError(7), h_dt_hit_flux_per_layer->GetBinError(8), h_dt_hit_flux_per_layer->GetBinError(9)},
        { h_dt_hit_flux_per_layer->GetBinError(10), h_dt_hit_flux_per_layer->GetBinError(11), h_dt_hit_flux_per_layer->GetBinError(12)}
    };
    for (int i=0; i<3; i++)
    {
      gr_dt_hit_flux_mb1->SetPoint(i, xx[0][i], yy[0][i]);
      gr_dt_hit_flux_mb2->SetPoint(i, xx[1][i], yy[1][i]);
      gr_dt_hit_flux_mb3->SetPoint(i, xx[2][i], yy[2][i]);
      gr_dt_hit_flux_mb4->SetPoint(i, xx[3][i], yy[3][i]);
      gr_dt_hit_flux_mb1->SetPointError(i, xe[0][i], ye[0][i]);
      gr_dt_hit_flux_mb2->SetPointError(i, xe[1][i], ye[1][i]);
      gr_dt_hit_flux_mb3->SetPointError(i, xe[2][i], ye[2][i]);
      gr_dt_hit_flux_mb4->SetPointError(i, xe[3][i], ye[3][i]);
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
