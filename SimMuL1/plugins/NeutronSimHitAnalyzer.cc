// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TString.h"

#include "GEMCode/SimMuL1/interface/PSimHitMapCSC.h"

//
// class decleration
//

struct MyCSCDetId
{
  void init(CSCDetId &id);
  Int_t e, s, r, c, l;
  Int_t t; // type 1-10: ME1/a,1/b,1/2,1/3,2/1...4/2
};

struct MyCSCSimHit
{
  void init(PSimHit &sh, const CSCGeometry* csc_g);
  Float_t x, y, z;    // local
  Float_t r, eta, phi, gz; // global
  Float_t e;          // energy deposit
  Float_t t;          // TOF
  Int_t pdg;          // PDG
  Int_t w, s;         // WG & Strip
};

struct MyCSCLayer
{
  void init(int l, std::vector<MyCSCSimHit> &shits);
  std::vector<MyCSCSimHit> hits;
  Int_t ln;             // layer #
  Int_t nh;             // # of hits
  Float_t mint, maxt;   // min/max TOF
  Float_t meant, sigmat;// mean&stdev of TOF
  Int_t minw, maxw, mins, maxs; // min/max WG & Strip
};

struct MyCSCChamber
{
  void init(std::vector<MyCSCLayer> &slayers);
  Int_t nh;             // nhits
  Int_t nl, l1, ln;     // nlayers, 1st and last layer #
  Float_t mint, maxt;   // min/max TOF
  Float_t meant, sigmat;// mean&stdev of TOF
  Int_t minw, maxw, mins, maxs; // min/max WG & Strip
};

struct MyCSCEvent
{
  void init(std::vector<MyCSCChamber> &schambers);
  Int_t nh;        // nhits
  Int_t nch;       // nchambers
};


struct MyRPCDetId
{
  void init(RPCDetId &id);
  Int_t reg, ring, st, sec, layer, subsec, roll;
  Int_t t; // type 1-8: RE1/2,1/3,2/2,2/3,3/2,3/3,4/2,4/3
};

struct MyRPCSimHit
{
  void init(const PSimHit &sh, const RPCGeometry* rpc_g);
  Float_t x, y, z;    // local
  Float_t r, eta, phi, gz; // global
  Float_t e;          // energy deposit
  Float_t t;          // TOF
  Int_t pdg;          // PDG
  //Int_t w, s;         // WG & Strip
};

struct MyGEMDetId
{
  void init(GEMDetId &id);
  Int_t region, ring, station, layer, chamber, roll;
  Int_t t; // GE1/1
};

struct MyGEMSimHit
{
  void init(const PSimHit &sh, const GEMGeometry* gem_g);
  Float_t l_x, l_y, l_z;    // local
  Float_t g_r, g_eta, g_phi, g_z; // global
  Float_t energy;          // energy deposit
  Float_t tof;          // TOF
  Int_t pdg;          // PDG
  Int_t strip;
};


class NeutronSimHitAnalyzer : public edm::EDAnalyzer 
{
public:

  explicit NeutronSimHitAnalyzer(const edm::ParameterSet&);
  ~NeutronSimHitAnalyzer();

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

private:

  // configuration parameters:
  edm::InputTag defaultInputTagCSC_;
  edm::InputTag defaultInputTagGEM_;
  edm::InputTag defaultInputTagRPC_;
  edm::InputTag defaultInputTagDT_;
  edm::InputTag inputTagCSC_;
  edm::InputTag inputTagGEM_;
  edm::InputTag inputTagRPC_;
  edm::InputTag inputTagDT_;

  // misc utilities
  const CSCGeometry* csc_geometry;
  const DTGeometry*  dt_geometry;
  const RPCGeometry* rpc_geometry;
  const GEMGeometry* gem_geometry;


  // some counters:
  int evtn;
  int csc_shn;
  int rpc_shn;
  int gem_shn;
  int nevt_with_cscsh, nevt_with_cscsh_in_rpc;
  int n_cscsh, n_cscsh_in_rpc;
  int nevt_with_rpcsh, nevt_with_rpcsh_e, nevt_with_rpcsh_b;
  int n_rpcsh, n_rpcsh_e, n_rpcsh_b;
  int nevt_with_gemsh;
  int n_gemsh;

  // some histos:
  TH2D* h_csc_rz_sh_xray;
  TH1D* h_csc_nlayers_in_ch[11];
  TH1D* h_csc_nevt_fraction_with_sh;
  TH1D* h_csc_shflux_per_layer;
  TH1D* h_rpcf_nevt_fraction_with_sh;
  TH1D* h_rpcf_shflux_per_layer;
  TH1D* h_rpcb_nevt_fraction_with_sh;
  TH1D* h_gem_nevt_fraction_with_sh;
  TH1D* h_gem_shflux_per_layer;
  TH2D* h_csc_rz_sh_heatmap;
  TH2D* h_rpc_rz_sh_heatmap;
  TH2D* h_gem_rz_sh_heatmap;
  TH2D* h_mu_rz_sh_heatmap;

  // quantities for rate calculation
  int pu_;
  double fractionEmptyBX_;
  int bxRate_;

  // Muon detector types (long and short)
  int maxCSCStations_;
  int nCSCTypes_;
  int maxDTStations_;
  int nDTTypes_;
  int maxRPCfStations_;
  int nRPCfTypes_;
  int maxRPCbStations_;
  int nRPCbTypes_;
  int maxGEMStations_;
  int nGEMTypes_;

  std::vector<std::string> cscTypesLong_;
  std::vector<std::string> cscTypesShort_;
  std::vector<std::string> dtTypesLong_;
  std::vector<std::string> dtTypesShort_;
  std::vector<std::string> rpcfTypesLong_;
  std::vector<std::string> rpcfTypesShort_;
  std::vector<std::string> rpcbTypesLong_;
  std::vector<std::string> rpcbTypesShort_;
  std::vector<std::string> gemTypesLong_;
  std::vector<std::string> gemTypesShort_;

  SimHitAnalysis::PSimHitMapCSC simhit_map_csc;
  bool inputIsNeutrons_;

  // chamber sizes and segmentation
  std::vector<double> cscAreascm2_;
  std::vector<double> rpcfAreascm2_;
  std::vector<double> gemAreascm2_;
  std::vector<int> cscRadialSegmentation_;
  std::vector<int> rpcfRadialSegmentation_;
  std::vector<int> gemRadialSegmentation_;

  // layers
  int nRPCLayers_;
  int nGEMLayers_;
  int nCSCLayers_;

  // ntuples:
  void bookCSCChambersTree();
  void bookCSCLayersTree();
  void bookCSCSimHitsTree();
  void bookRPCSimHitsTree();
  void bookGEMSimHitsTree();

  TTree* csc_ch_tree;
  TTree* csc_la_tree;
  TTree* csc_sh_tree;
  TTree* rpc_sh_tree;
  TTree* gem_sh_tree;

  MyCSCDetId c_id;
  MyCSCDetId c_cid;
  MyCSCSimHit c_h;
  MyCSCLayer c_la;
  MyCSCChamber c_ch;
  MyRPCDetId r_id;
  MyRPCSimHit r_h;
  MyGEMDetId g_id;
  MyGEMSimHit g_h;
};

// ================================================================================================
NeutronSimHitAnalyzer::NeutronSimHitAnalyzer(const edm::ParameterSet& iConfig):
  defaultInputTagCSC_(iConfig.getParameter<edm::InputTag>("defaultInputTagCSC")),
  defaultInputTagGEM_(iConfig.getParameter<edm::InputTag>("defaultInputTagCSC")),
  defaultInputTagRPC_(iConfig.getParameter<edm::InputTag>("defaultInputTagCSC")),
  defaultInputTagDT_(iConfig.getParameter<edm::InputTag>("defaultInputTagCSC")),
  inputTagCSC_(iConfig.getParameter<edm::InputTag>("inputTagCSC")),
  inputTagGEM_(iConfig.getParameter<edm::InputTag>("inputTagGEM")),
  inputTagRPC_(iConfig.getParameter<edm::InputTag>("inputTagRPC")),
  inputTagDT_(iConfig.getParameter<edm::InputTag>("inputTagDT")),
  pu_(iConfig.getParameter<int>("pu")),
  fractionEmptyBX_(iConfig.getParameter<double>("fractionEmptyBX")),
  bxRate_(iConfig.getParameter<int>( "bxRate")),                            
  maxCSCStations_(iConfig.getParameter<int>("maxCSCStations")),
  nCSCTypes_(iConfig.getParameter<int>("nCSCTypes")),
  maxDTStations_(iConfig.getParameter<int>("maxDTStations")),
  nDTTypes_(iConfig.getParameter<int>("nDTTypes")),
  maxRPCfStations_(iConfig.getParameter<int>("maxRPCfStations")), 
  nRPCfTypes_(iConfig.getParameter<int>("nRPCfTypes")), 
  maxRPCbStations_(iConfig.getParameter<int>("maxRPCbStations")),
  nRPCbTypes_(iConfig.getParameter<int>("nRPCbTypes")),
  maxGEMStations_(iConfig.getParameter<int>("maxGEMStations")),
  nGEMTypes_(iConfig.getParameter<int>("nGEMTypes")),
  cscTypesLong_(iConfig.getParameter<std::vector<std::string> >("cscTypesLong")),
  cscTypesShort_(iConfig.getParameter<std::vector<std::string> >("cscTypesShort")),
  dtTypesLong_(iConfig.getParameter<std::vector<std::string> >("dtTypesLong")),
  dtTypesShort_(iConfig.getParameter<std::vector<std::string> >("dtTypesShort")),
  rpcfTypesLong_(iConfig.getParameter<std::vector<std::string> >("rpcfTypesLong")),
  rpcfTypesShort_(iConfig.getParameter<std::vector<std::string> >("rpcfTypesShort")),
  rpcbTypesLong_(iConfig.getParameter<std::vector<std::string> >("rpcbTypesLong")),
  rpcbTypesShort_(iConfig.getParameter<std::vector<std::string> >("rpcbTypesShort")),
  gemTypesLong_(iConfig.getParameter<std::vector<std::string> >("gemTypesLong")),
  gemTypesShort_(iConfig.getParameter<std::vector<std::string> >("gemTypesLong")),
  simhit_map_csc(),
  // should be set to false if running over a regular MB sample
  inputIsNeutrons_(iConfig.getParameter<bool>("inputIsNeutrons")),
  //chambers sizes
  cscAreascm2_(iConfig.getParameter<std::vector<double> >("cscAreascm2")),
  rpcfAreascm2_(iConfig.getParameter<std::vector<double> >("rpfcAreascm2")),
  gemAreascm2_(iConfig.getParameter<std::vector<double> >("gemAreascm2")),
  cscRadialSegmentation_(iConfig.getParameter<std::vector<int> >("cscRadialSegmentation")),
  rpcfRadialSegmentation_(iConfig.getParameter<std::vector<int> >("rpcfRadialSegmentation")),
  gemRadialSegmentation_(iConfig.getParameter<std::vector<int> >("gemRadialSegmentation")),
  nRPCLayers_(iConfig.getParameter<int>("nRPCLayers")),
  nGEMLayers_(iConfig.getParameter<int>("nGEMLayers")),
  nCSCLayers_(iConfig.getParameter<int>("nCSCLayers"))
{
  bookCSCSimHitsTree();
  bookCSCLayersTree();
  bookCSCChambersTree();
  bookRPCSimHitsTree();
  bookGEMSimHitsTree();
  
  if (inputIsNeutrons_)
  {
    inputTagCSC_ = defaultInputTagCSC_;
    inputTagGEM_ = defaultInputTagGEM_;
    inputTagRPC_ = defaultInputTagRPC_;
    inputTagDT_ = defaultInputTagDT_;
  }
  
  simhit_map_csc.setInputTag(inputTagCSC_);
  nevt_with_cscsh = nevt_with_cscsh_in_rpc = n_cscsh = n_cscsh_in_rpc = 0;
  nevt_with_rpcsh = nevt_with_rpcsh_e = nevt_with_rpcsh_b = n_rpcsh = n_rpcsh_e = n_rpcsh_b = 0;

  edm::Service<TFileService> fs;

  const TString n_simhits(inputIsNeutrons_? "n-SimHits" : "SimHits");

  // Heat map
  h_csc_rz_sh_xray = fs->make<TH2D>("h_csc_rz_sh_xray","CSC "+n_simhits+" #rho-z X-ray;z, cm;#rho, cm",2060,550,1080,750,0,750);
  h_csc_rz_sh_heatmap = fs->make<TH2D>("h_csc_rz_sh_heatmap","CSC "+n_simhits+" #rho-z;z, cm;#rho, cm",230,550,1080,150,0,750);
  h_rpc_rz_sh_heatmap = fs->make<TH2D>("h_rpc_rz_sh_heatmap","RPC "+n_simhits+" #rho-z;z, cm;#rho, cm",450,0,1080,160,0,800);
  h_gem_rz_sh_heatmap = fs->make<TH2D>("h_gem_rz_sh_heatmap","GEM "+n_simhits+" #rho-z;z, cm;#rho, cm",450,0,1080,160,0,800);
  h_mu_rz_sh_heatmap = fs->make<TH2D>("h_mu_rz_sh_heatmap", n_simhits+" #rho-z;z, cm;#rho, cm",480,0,1100,160,0,800);


  // CSC 
  char label[200], nlabel[200];
  for (int me=0; me<=nCSCTypes_; me++)
  {
    sprintf(label,"h_csc_nlayers_in_ch_%s",cscTypesShort_.at(me).c_str());
    sprintf(nlabel,"# of layers with %s: %s CSC;# layers", n_simhits.Data(), cscTypesLong_.at(me).c_str());
    h_csc_nlayers_in_ch[me]  = fs->make<TH1D>(label, nlabel, 6, 0.5, 6.5);
  }
  h_csc_nevt_fraction_with_sh =
      fs->make<TH1D>("h_csc_nevt_fraction_with_sh", "Fraction of events with "+n_simhits+" by CSC type", nCSCTypes_+1, -0.5, nCSCTypes_+0.5);
  for (int i=1; i<=h_csc_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_csc_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,cscTypesLong_.at(i-1).c_str());

  h_csc_shflux_per_layer = fs->make<TH1D>("h_csc_shflux_per_layer", n_simhits+" Flux per CSC layer at L=10^{34};;Hz/cm^{2}", nCSCTypes_, 0.5,  nCSCTypes_+0.5);
  for (int i=1; i<=h_csc_shflux_per_layer->GetXaxis()->GetNbins();i++)
    h_csc_shflux_per_layer->GetXaxis()->SetBinLabel(i,cscTypesLong_.at(i).c_str());

  // RPC
  h_rpcf_nevt_fraction_with_sh =
      fs->make<TH1D>("h_rpcf_nevt_fraction_with_sh", "Fraction of events with "+n_simhits+" by RPCf type", nRPCfTypes_+1, -0.5, nRPCfTypes_+0.5);
  for (int i=1; i<=h_rpcf_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_rpcf_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,rpcfTypesLong_.at(i-1).c_str());

  h_rpcf_shflux_per_layer = fs->make<TH1D>("h_rpcf_shflux_per_layer", n_simhits+" Flux in RPCf at L=10^{34};;Hz/cm^{2}", nRPCfTypes_, 0.5,  nRPCfTypes_+0.5);
  for (int i=1; i<=h_rpcf_shflux_per_layer->GetXaxis()->GetNbins();i++)
    h_rpcf_shflux_per_layer->GetXaxis()->SetBinLabel(i,rpcfTypesLong_.at(i).c_str());

  h_rpcb_nevt_fraction_with_sh =
      fs->make<TH1D>("h_rpcb_nevt_fraction_with_sh", "Fraction of events with "+n_simhits+" by RPCb type", nRPCbTypes_+1, -0.5, nRPCbTypes_+0.5);
  for (int i=1; i<=h_rpcb_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_rpcb_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,rpcbTypesLong_.at(i-1).c_str());

  // GEM
  h_gem_nevt_fraction_with_sh =
      fs->make<TH1D>("h_gem_nevt_fraction_with_sh", "Fraction of events with "+n_simhits+" by GEM type", nGEMTypes_+1, -0.5, nGEMTypes_+0.5);
  for (int i=1; i<=h_gem_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_gem_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,gemTypesLong_.at(i-1).c_str());

  h_gem_shflux_per_layer = fs->make<TH1D>("h_gem_shflux_per_layer", n_simhits+" Flux per GEM layer at L=10^{34};;Hz/cm^{2}", nGEMTypes_, 0.5,  nGEMTypes_+0.5);
  for (int i=1; i<=h_gem_shflux_per_layer->GetXaxis()->GetNbins();i++)
    h_gem_shflux_per_layer->GetXaxis()->SetBinLabel(i,gemTypesLong_.at(i).c_str());
}

// ================================================================================================
NeutronSimHitAnalyzer::~NeutronSimHitAnalyzer()
{}


// ================================================================================================
void
NeutronSimHitAnalyzer::bookCSCSimHitsTree()
{
  edm::Service<TFileService> fs;
  csc_sh_tree = fs->make<TTree>("CSCSimHitsTree", "CSCSimHitsTree");
  csc_sh_tree->Branch("evtn", &evtn);
  csc_sh_tree->Branch("shn", &csc_shn);
  csc_sh_tree->Branch("id", &c_id.e,"e/I:s:r:c:l:t");
  csc_sh_tree->Branch("sh", &c_h.x,"x/F:y:z:r:eta:phi:gz:e:t:pdg/I:w:s");
  //csc_sh_tree->Branch("", &., "/I");
  //csc_sh_tree->Branch("" , "vector<double>" , & );
}


// ================================================================================================
void
NeutronSimHitAnalyzer::bookCSCLayersTree()
{
  edm::Service<TFileService> fs;
  csc_la_tree = fs->make<TTree>("CSCLayersTree", "CSCLayersTree");
  csc_la_tree->Branch("evtn", &evtn,"evtn/i");
  csc_la_tree->Branch("id", &c_id.e,"e/I:s:r:c:l:t");
  csc_la_tree->Branch("la", &c_la.ln,"ln/I:nh:mint/F:maxt:meant:sigmat:minw/I:maxw:mins:maxs");
}


// ================================================================================================
void
NeutronSimHitAnalyzer::bookCSCChambersTree()
{
  edm::Service<TFileService> fs;
  csc_ch_tree = fs->make<TTree>("CSCChambersTree", "CSCChambersTree");
  csc_ch_tree->Branch("evtn", &evtn,"evtn/i");
  csc_ch_tree->Branch("id", &c_cid.e,"e/I:s:r:c:l:t");
  csc_ch_tree->Branch("ch", &c_ch.nh,"nh/I:nl:l1:ln:mint/F:maxt:meant:sigmat:minw/I:maxw:mins:maxs");
}


// ================================================================================================
void
NeutronSimHitAnalyzer::bookRPCSimHitsTree()
{
  edm::Service<TFileService> fs;
  rpc_sh_tree = fs->make<TTree>("RPCSimHitsTree", "RPCSimHitsTree");
  rpc_sh_tree->Branch("evtn", &evtn,"evtn/i");
  rpc_sh_tree->Branch("shn", &rpc_shn,"shn/i");
  rpc_sh_tree->Branch("id", &r_id.reg,"reg/I:ring:st:sec:layer:subsec:roll:t");
  rpc_sh_tree->Branch("sh", &r_h.x,"x/F:y:z:r:eta:phi:gz:e:t:pdg/I");
}


// ================================================================================================
void
NeutronSimHitAnalyzer::bookGEMSimHitsTree()
{
  edm::Service<TFileService> fs;
  gem_sh_tree = fs->make<TTree>("GEMSimHitsTree", "GEMSimHitsTree");
  gem_sh_tree->Branch("evtn", &evtn,"evtn/i");
  gem_sh_tree->Branch("shn", &gem_shn,"shn/i");
  gem_sh_tree->Branch("region", &g_id.region);
  gem_sh_tree->Branch("ring", &g_id.ring);
  gem_sh_tree->Branch("station", &g_id.station);
  gem_sh_tree->Branch("layer", &g_id.layer);
  gem_sh_tree->Branch("chamber", &g_id.chamber);
  gem_sh_tree->Branch("roll", &g_id.roll);
  gem_sh_tree->Branch("l_x", &g_h.l_x);
  gem_sh_tree->Branch("l_y", &g_h.l_y);
  gem_sh_tree->Branch("l_z", &g_h.g_z);
  gem_sh_tree->Branch("g_r", &g_h.g_r);
  gem_sh_tree->Branch("g_eta", &g_h.g_eta);
  gem_sh_tree->Branch("g_phi", &g_h.g_phi);
  gem_sh_tree->Branch("g_z", &g_h.g_z);
  gem_sh_tree->Branch("energyLoss", &g_h.energy);
  gem_sh_tree->Branch("timeOfFlight", &g_h.tof);
  gem_sh_tree->Branch("particleType", &g_h.pdg);
  gem_sh_tree->Branch("strip", &g_h.strip);
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
MyCSCSimHit::init(PSimHit &sh, const CSCGeometry* csc_g)
{
  const LocalPoint hitLP(sh.localPosition());

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  t = sh.tof();
  pdg = sh.particleType();

  const CSCDetId layerId(sh.detUnitId());
  const CSCLayer* csclayer(csc_g->layer(layerId));
  const GlobalPoint hitGP(csclayer->toGlobal(hitLP));

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gz = hitGP.z();

  w = csclayer->geometry()->wireGroup(csclayer->geometry()->nearestWire(hitLP));
  s = csclayer->geometry()->nearestStrip(hitLP);
}


// ================================================================================================
void
MyCSCLayer::init(int l, std::vector<MyCSCSimHit> &shits)
{
  hits = shits;
  ln = l;
  nh = hits.size();
  mint = 1000.;
  maxt = -1.;
  meant = 0;
  sigmat = 0;
  minw = 1000.;
  maxw = -1;
  mins = 1000;
  maxs = -1;
  if (nh==0) return;
  for (std::vector<MyCSCSimHit>::const_iterator itr = hits.begin(); itr != hits.end(); itr++)
  {
    const MyCSCSimHit sh(*itr);
    if (sh.t < mint) mint = sh.t;
    if (sh.t > maxt) maxt = sh.t;
    if (sh.w < minw) minw = sh.w;
    if (sh.w > maxw) maxw = sh.w;
    if (sh.s < mins) mins = sh.s;
    if (sh.s > maxs) maxs = sh.s;
    meant += sh.t;
    sigmat += sh.t*sh.t;
  }
  meant = meant/nh;
  sigmat = sqrt( sigmat/nh - meant*meant);
}


// ================================================================================================
void
MyCSCChamber::init(std::vector<MyCSCLayer> &slayers)
{
  nh = 0;
  nl = slayers.size();
  mint = 1000.;
  maxt = -1.;
  meant = 0;
  sigmat = 0;
  minw = 1000.;
  maxw = -1;
  mins = 1000;
  maxs = -1;
  if (nl==0) return;
  l1 = slayers[0].ln;
  ln = slayers[nl-1].ln;
  for (std::vector<MyCSCLayer>::const_iterator sl = slayers.begin(); sl != slayers.end(); sl++)
  {
    nh += sl->hits.size();
    for (std::vector<MyCSCSimHit>::const_iterator itr = sl->hits.begin(); itr != sl->hits.end(); itr++)
    {
      const MyCSCSimHit sh(*itr);
      if (sh.t < mint) mint = sh.t;
      if (sh.t > maxt) maxt = sh.t;
      if (sh.w < minw) minw = sh.w;
      if (sh.w > maxw) maxw = sh.w;
      if (sh.s < mins) mins = sh.s;
      if (sh.s > maxs) maxs = sh.s;
      meant += sh.t;
      sigmat += sh.t*sh.t;
    }
  }
  meant = meant/nh;
  sigmat = sqrt( sigmat/nh - meant*meant);
}


// ================================================================================================
void
MyCSCEvent::init(std::vector<MyCSCChamber> &schambers)
{
  nh = 0;
  nch = 0;
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
  if (reg!=0) t = st*2 + ring - 3;  // endcap
  else t = st*3 + abs(ring) - 2;
}


// ================================================================================================
void
MyRPCSimHit::init(const PSimHit &sh, const RPCGeometry* rpc_g)
{
  const LocalPoint hitLP(sh.localPosition());

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  t = sh.tof();
  pdg = sh.particleType();

  const GlobalPoint hitGP(rpc_g->idToDet(DetId(sh.detUnitId()))->surface().toGlobal(hitLP));

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gz = hitGP.z();
}

// ================================================================================================
void
MyGEMDetId::init(GEMDetId &id)
{
  region = id.region();
  ring = id.ring();
  station = id.station();
  layer = id.layer();
  chamber = id.chamber();
  roll = id.roll();
  t = 1;
}

// ================================================================================================
void
MyGEMSimHit::init(const PSimHit &sh, const GEMGeometry* gem_g)
{
  const LocalPoint hitLP(sh.localPosition());

  l_x = hitLP.x();
  l_y = hitLP.y();
  l_z = hitLP.z();
  energy = sh.energyLoss();
  tof = sh.tof();
  pdg = sh.particleType();

  const GlobalPoint hitGP(gem_g->idToDet(DetId(sh.detUnitId()))->surface().toGlobal(hitLP));

  g_r = hitGP.perp();
  g_eta = hitGP.eta();
  g_phi = hitGP.phi();
  g_z = hitGP.z();

//   const LocalPoint hitEP(sh->entryPoint());
//   strip = gem_geometry_->etaPartition(sh->detUnitId())->strip(hitEP);

}

// ================================================================================================
void
NeutronSimHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  evtn += 1;
  csc_shn = 0;
  rpc_shn = 0;

  // Get the geometry
  edm::ESHandle<CSCGeometry> csc_geom;
  iSetup.get< MuonGeometryRecord >().get(csc_geom);
  csc_geometry = &*csc_geom;

  edm::ESHandle<DTGeometry>  dt_geom;
  iSetup.get< MuonGeometryRecord >().get(dt_geom);
  dt_geometry  = &*dt_geom;

  edm::ESHandle<RPCGeometry> rpc_geom;
  iSetup.get< MuonGeometryRecord >().get(rpc_geom);
  rpc_geometry = &*rpc_geom;

  edm::ESHandle<GEMGeometry> gem_geom;
  iSetup.get< MuonGeometryRecord >().get(gem_geom);
  gem_geometry = &*gem_geom;

  // get SimHits
  simhit_map_csc.fill(iEvent);

  std::vector<int> chIds = simhit_map_csc.chambersWithHits();
  if (chIds.size()) {
    //std::cout<<"--- chambers with hits: "<<chIds.size()<<std::endl;
    nevt_with_cscsh++;
  }
  bool ev_has_csc_type[11]={0,0,0,0,0,0,0,0,0,0,0};
  bool has_cscsh_in_rpc = false;
  for (size_t ch = 0; ch < chIds.size(); ch++)
  {
    CSCDetId chId(chIds[ch]);
    c_cid.init(chId);

    std::vector<int> layer_ids = simhit_map_csc.chamberLayersWithHits(chIds[ch]);
    //if (layer_ids.size()) std::cout<<"------ layers with hits: "<<layer_ids.size()<<std::endl;

    std::vector<MyCSCLayer> chamber_layers;
    for  (size_t la = 0; la < layer_ids.size(); la++)
    {
      CSCDetId layerId(layer_ids[la]);
      c_id.init(layerId);

      edm::PSimHitContainer hits(simhit_map_csc.hits(layer_ids[la]));
      //std::couts<<"      "<< layer_ids[la]<<" "<<layerId<<"   no. of hits = "<<hits.size()<<std::endl;

      std::vector<MyCSCSimHit> layer_simhits;
      for (unsigned j=0; j<hits.size(); j++)
      {
        csc_shn += 1;
        c_h.init(hits[j], csc_geometry);
        layer_simhits.push_back(c_h);
        csc_sh_tree->Fill();

        // count hits
        n_cscsh++;
        if ( fabs(c_h.eta) < 1.6 && c_id.s < 4 ) // RPC region in 1-3 stations
        {
          n_cscsh_in_rpc++;
          has_cscsh_in_rpc = true;
        }

        // Fill some histos
        h_csc_rz_sh_xray->Fill(fabs(c_h.gz), c_h.r);
        h_csc_rz_sh_heatmap->Fill(fabs(c_h.gz), c_h.r);
        h_mu_rz_sh_heatmap->Fill(fabs(c_h.gz), c_h.r);

        h_csc_shflux_per_layer->Fill(c_cid.t);
      }

      c_la.init(layerId.layer(),layer_simhits);
      chamber_layers.push_back(c_la);
      csc_la_tree->Fill();
    }
    c_ch.init(chamber_layers);
    csc_ch_tree->Fill();

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
  // fill events with CSC hits by CSC type:
  for (int t=0; t<=nCSCTypes_; t++)
    if (ev_has_csc_type[t]) h_csc_nevt_fraction_with_sh->Fill(t);

  // count events with CSC hits in PRC region
  if (has_cscsh_in_rpc) nevt_with_cscsh_in_rpc++;

  


  // ************ RPC *************

  edm::Handle< edm::PSimHitContainer > h_rpc_simhits;
  iEvent.getByLabel(inputTagRPC_, h_rpc_simhits);
  const edm::PSimHitContainer* rpc_simhits = h_rpc_simhits.product();

  if (rpc_simhits->size()) nevt_with_rpcsh++;

  bool ev_has_rpcf_type[9]={0,0,0,0,0,0,0,0,0};
  bool ev_has_rpcb_type[13]={0,0,0,0,0,0,0,0,0,0,0,0,0};
  bool has_rpcsh_e = false, has_rpcsh_b = false;
  for (edm::PSimHitContainer::const_iterator rsh = rpc_simhits->begin(); rsh!=rpc_simhits->end(); rsh++)
  {
    rpc_shn += 1;
    const PSimHit &sh(*rsh);
    RPCDetId shid(sh.detUnitId());

    r_id.init(shid);
    r_h.init(sh, rpc_geometry);
    rpc_sh_tree->Fill();

    n_rpcsh++;
    if (shid.region()==0) {n_rpcsh_b++; has_rpcsh_b=true;}
    else {n_rpcsh_e++; has_rpcsh_e=true;}

    // Fill some histos
    h_rpc_rz_sh_heatmap->Fill(fabs(r_h.gz), r_h.r);
    h_mu_rz_sh_heatmap->Fill(fabs(r_h.gz), r_h.r);

    if (shid.region() == 0)
    {
      ev_has_rpcb_type[r_id.t]=true;
      ev_has_rpcb_type[0]=true;
    }
    else
    {
      ev_has_rpcf_type[r_id.t]=true;
      ev_has_rpcf_type[0]=true;

      h_rpcf_shflux_per_layer->Fill(r_id.t);
    }
  }
  // fill events with RPCf hits by RPCf type:
  for (int t=0; t<=nRPCfTypes_; t++)
    if (ev_has_rpcf_type[t]) h_rpcf_nevt_fraction_with_sh->Fill(t);

  // fill events with RPCb hits by RPCb type:
  for (int t=0; t<=nRPCbTypes_; t++)
    if (ev_has_rpcb_type[t]) h_rpcb_nevt_fraction_with_sh->Fill(t);

  if (has_rpcsh_e) nevt_with_rpcsh_e++;
  if (has_rpcsh_b) nevt_with_rpcsh_b++;



 
  // ************ GEM *************

  edm::Handle< edm::PSimHitContainer > h_gem_simhits;
  iEvent.getByLabel(inputTagGEM_, h_gem_simhits);
  const edm::PSimHitContainer* gem_simhits = h_gem_simhits.product();

  if (gem_simhits->size()) nevt_with_gemsh++;

  bool ev_has_gem_type[2]={0,0};
  bool has_gemsh = false;
  for (edm::PSimHitContainer::const_iterator rsh = gem_simhits->begin(); rsh!=gem_simhits->end(); rsh++)
  {
    gem_shn += 1;
    const PSimHit &sh(*rsh);
    GEMDetId shid(sh.detUnitId());

    g_id.init(shid);
    g_h.init(sh, gem_geometry);
    gem_sh_tree->Fill();

    n_gemsh++;
    has_gemsh = true;

    // Fill some histos
    h_gem_rz_sh_heatmap->Fill(fabs(g_h.g_z), g_h.g_r);
    h_mu_rz_sh_heatmap->Fill(fabs(g_h.g_z), g_h.g_r);

    if (abs(shid.region()) == 1)
    {
      ev_has_gem_type[g_id.t] = true;
      ev_has_gem_type[0] = true;
    }

  }
  // fill events with GEM hits by GEM type:
  for (int t=0; t<=nGEMTypes_; t++)
    if (ev_has_gem_type[t]) h_gem_nevt_fraction_with_sh->Fill(t);

  if (has_gemsh) nevt_with_gemsh++;
}


// ================================================================================================
void NeutronSimHitAnalyzer::beginJob() {}


// ================================================================================================
void NeutronSimHitAnalyzer::endJob()
{
  using namespace std;
  std::cout<<"******************* COUNTERS *******************"<<std::endl;
  std::cout<<"* #events: "<< evtn <<std::endl;
  std::cout<<"* #events with SimHits in:"<<std::endl;
  std::cout<<"*   GEM:                     "<<nevt_with_gemsh<<" ("<<(double)nevt_with_gemsh/evtn*100<<"%)"<<std::endl;
  std::cout<<"*   CSC:                     "<<nevt_with_cscsh<<" ("<<(double)nevt_with_cscsh/evtn*100<<"%)"<<std::endl;
  std::cout<<"*   CSC (|eta|<1.6, st 1-3): "<<nevt_with_cscsh_in_rpc<<" ("<<(double)nevt_with_cscsh_in_rpc/evtn*100<<"%)"<<std::endl;
  std::cout<<"*   RPC:                     "<<nevt_with_rpcsh<<" ("<<(double)nevt_with_rpcsh/evtn*100<<"%)"<<std::endl;
  std::cout<<"*   RPC endcaps:             "<<nevt_with_rpcsh_e<<" ("<<(double)nevt_with_rpcsh_e/evtn*100<<"%)"<<std::endl;
  std::cout<<"*   RPC barrel:              "<<nevt_with_rpcsh_b<<" ("<<(double)nevt_with_rpcsh_b/evtn*100<<"%)"<<std::endl;
  std::cout<<"* total SimHit numbers in: "<<std::endl;
  std::cout<<"*   GEM:                     "<<n_gemsh<<" (~"<<(double)n_gemsh/nevt_with_gemsh<<" sh/evt with hits)"<<std::endl;
  std::cout<<"*   CSC:                     "<<n_cscsh<<" (~"<<(double)n_cscsh/nevt_with_cscsh<<" sh/evt with hits)"<<std::endl;
  std::cout<<"*   CSC (|eta|<1.6, st 1-3): "<<n_cscsh_in_rpc<<" (~"<<(double)n_cscsh_in_rpc/nevt_with_cscsh_in_rpc<<" sh/evt with hits)"<<std::endl;
  std::cout<<"*   RPC:                     "<<n_rpcsh<<" (~"<<(double)n_rpcsh/nevt_with_rpcsh<<" sh/evt with hits)"<<std::endl;
  std::cout<<"*   RPC endcaps:             "<<n_rpcsh_e<<" (~"<<(double)n_rpcsh_e/nevt_with_rpcsh_e<<" sh/evt with hits)"<<std::endl;
  std::cout<<"*   RPC barrel:              "<<n_rpcsh_b<<" (~"<< ( (nevt_with_rpcsh_b>0) ? (double)n_rpcsh_b/nevt_with_rpcsh_b : 0 )<<" sh/evt with hits)"<<std::endl;
  std::cout<<"************************************************"<<std::endl;

  // convert to fraction:
  h_gem_nevt_fraction_with_sh->Sumw2();
  h_gem_nevt_fraction_with_sh->Scale(1./evtn);
  h_csc_nevt_fraction_with_sh->Sumw2();
  h_csc_nevt_fraction_with_sh->Scale(1./evtn);
  h_rpcf_nevt_fraction_with_sh->Sumw2();
  h_rpcf_nevt_fraction_with_sh->Scale(1./evtn);
  h_rpcb_nevt_fraction_with_sh->Sumw2();
  h_rpcb_nevt_fraction_with_sh->Scale(1./evtn);

  // ---- calculate fluxes at L=10^34

  h_csc_shflux_per_layer->Sumw2();
  for (int t=1; t<=nCSCTypes_; t++)
  {
    // 2 endcaps , 6 layers
    const double scale(bxRate_ * pu_ * fractionEmptyBX_ /cscAreascm2_.at(t)/cscRadialSegmentation_.at(t)/2/6 /evtn);
    const double rt(scale * h_csc_shflux_per_layer->GetBinContent(t));
    const double er(scale * h_csc_shflux_per_layer->GetBinError(t));
    h_csc_shflux_per_layer->SetBinContent(t,rt);
    h_csc_shflux_per_layer->SetBinError(t,er);
  }

  h_rpcf_shflux_per_layer->Sumw2();
  for (int t=1; t<=nRPCfTypes_; t++)
  {
    const double scale(bxRate_ * pu_ * fractionEmptyBX_ /rpcfAreascm2_.at(t)/rpcfRadialSegmentation_.at(t)/2 /evtn);
    const double rt(scale * h_rpcf_shflux_per_layer->GetBinContent(t));
    const double er(scale * h_rpcf_shflux_per_layer->GetBinError(t));
    h_rpcf_shflux_per_layer->SetBinContent(t,rt);
    h_rpcf_shflux_per_layer->SetBinError(t,er);
  }

  h_gem_shflux_per_layer->Sumw2();
  for (int t=1; t<=nGEMTypes_; t++)
  {
    const double scale(bxRate_ * pu_ * fractionEmptyBX_ /gemAreascm2_.at(t)/gemRadialSegmentation_.at(t)/2 /evtn);
    const double rt(scale * h_gem_shflux_per_layer->GetBinContent(t));
    const double er(scale * h_gem_shflux_per_layer->GetBinError(t));
    h_gem_shflux_per_layer->SetBinContent(t,rt);
    h_gem_shflux_per_layer->SetBinError(t,er);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(NeutronSimHitAnalyzer);
