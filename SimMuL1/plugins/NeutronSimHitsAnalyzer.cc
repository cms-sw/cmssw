/*
 *
 *
 *
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

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "TH1.h"
#include "TH2.h"
#include "TTree.h"

#include "PSimHitMapCSC.h"


namespace
{
enum trig_cscs {MAX_CSC_STATIONS = 4, CSC_TYPES = 10};
enum trig_dts {MAX_DT_STATIONS = 4, DT_TYPES = 12};
enum trig_rpcf {MAX_RPCF_STATIONS = 4, RPCF_TYPES = 8};
enum trig_rpcb {MAX_RPCB_STATIONS = 4, RPCB_TYPES = 12};
int getCSCType(CSCDetId &id);
}

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


class NeutronSimHitsAnalyzer : public edm::EDAnalyzer {
public:
  explicit NeutronSimHitsAnalyzer(const edm::ParameterSet&);
  ~NeutronSimHitsAnalyzer();

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  static const std::string csc_type[CSC_TYPES+1];
  static const std::string csc_type_[CSC_TYPES+1];
  static const std::string dt_type[DT_TYPES+1];
  static const std::string dt_type_[DT_TYPES+1];
  static const std::string rpcf_type[RPCF_TYPES+1];
  static const std::string rpcf_type_[RPCF_TYPES+1];
  static const std::string rpcb_type[RPCB_TYPES+1];
  static const std::string rpcb_type_[RPCB_TYPES+1];

private:

  // configuration parameters:

  edm::InputTag input_tag_csc_;
  edm::InputTag input_tag_dt_;
  edm::InputTag input_tag_rpc_;

  bool input_is_neutrons_;

  bool fill_csc_ch_tree_;
  bool fill_csc_la_tree_;
  bool fill_csc_sh_tree_;
  bool fill_rpc_sh_tree_;

  // misc utilities

  const CSCGeometry* csc_geometry;
  const DTGeometry* dt_geometry;
  const RPCGeometry* rpc_geometry;

  SimHitAnalysis::PSimHitMapCSC simhit_map_csc;


  // some counters:

  UInt_t evtn;
  UInt_t csc_shn;
  UInt_t rpc_shn;

  int nevt_with_cscsh, nevt_with_cscsh_in_rpc;
  int n_cscsh, n_cscsh_in_rpc;

  int nevt_with_rpcsh, nevt_with_rpcsh_e, nevt_with_rpcsh_b;
  int n_rpcsh, n_rpcsh_e, n_rpcsh_b;

  // some histos:

  TH2D * h_csc_rz_sh_xray;
  TH2D * h_csc_rz_sh_heatmap;
  TH1D * h_csc_nlayers_in_ch[CSC_TYPES+1];
  TH1D * h_csc_nevt_fraction_with_sh;
  TH1D * h_csc_shflux_per_layer;

  TH2D * h_rpc_rz_sh_heatmap;
  TH1D * h_rpcf_nevt_fraction_with_sh;
  TH1D * h_rpcf_shflux_per_layer;

  TH1D * h_rpcb_nevt_fraction_with_sh;

  TH2D * h_mu_rz_sh_heatmap;

  // some ntuples:

  void bookCSCChambersTree();
  void bookCSCLayersTree();
  void bookCSCSimHitsTree();

  void bookRPCSimHitsTree();

  TTree* csc_ch_tree;
  TTree* csc_la_tree;
  TTree* csc_sh_tree;

  TTree* rpc_sh_tree;

  MyCSCDetId c_id;
  MyCSCDetId c_cid;
  MyCSCSimHit c_h;
  MyCSCLayer c_la;
  MyCSCChamber c_ch;

  MyRPCDetId r_id;
  MyRPCSimHit r_h;
};

//
// constants, enums and typedefs
//

const std::string NeutronSimHitsAnalyzer::csc_type[CSC_TYPES+1] =
  { "Any", "ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2"};
const std::string NeutronSimHitsAnalyzer::csc_type_[CSC_TYPES+1] =
  { "Any", "ME1a", "ME1b", "ME12", "ME13", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42"};

const std::string NeutronSimHitsAnalyzer::dt_type[DT_TYPES+1] =
  { "Any", "MB1/0", "MB1/1", "MB1/2", "MB2/0", "MB2/1", "MB2/2", "MB3/0", "MB3/1", "MB3/2", "MB4/0", "MB4/1", "MB4/2",};
const std::string NeutronSimHitsAnalyzer::dt_type_[DT_TYPES+1] =
  { "Any", "MB10", "MB11", "MB12", "MB20", "MB21", "MB22", "MB30", "MB31", "MB32", "MB40", "MB41", "MB42",};

const std::string NeutronSimHitsAnalyzer::rpcf_type[RPCF_TYPES+1] =
  { "Any", "RE1/2", "RE1/3", "RE2/2", "RE2/3", "RE3/2", "RE3/3", "RE4/2", "RE4/3"};
const std::string NeutronSimHitsAnalyzer::rpcf_type_[RPCF_TYPES+1] =
  { "Any", "RE12", "RE13", "RE22", "RE23", "RE32", "RE33", "RE42", "RE43"};

const std::string NeutronSimHitsAnalyzer::rpcb_type[RPCB_TYPES+1] =
  { "Any", "RB1/0", "RB1/1", "RB1/2", "RB2/0", "RB2/1", "RB2/2", "RB3/0", "RB3/1", "RB3/2", "RB4/0", "RB4/1", "RB4/2",};
const std::string NeutronSimHitsAnalyzer::rpcb_type_[RPCB_TYPES+1] =
  { "Any", "RB10", "RB11", "RB12", "RB20", "RB21", "RB22", "RB30", "RB31", "RB32", "RB40", "RB41", "RB42",};


// ================================================================================================
NeutronSimHitsAnalyzer::NeutronSimHitsAnalyzer(const edm::ParameterSet& iConfig):
    simhit_map_csc()
{
  // should be set to false if running over a regular MB sample
  input_is_neutrons_ = iConfig.getUntrackedParameter< bool >("inputIsNeutrons",true);

  edm::InputTag default_tag_csc("g4SimHits","MuonCSCHits");
  edm::InputTag default_tag_dt("g4SimHits","MuonDTHits");
  edm::InputTag default_tag_rpc("g4SimHits","MuonRPCHits");
  if (input_is_neutrons_)
  {
    default_tag_csc = edm::InputTag("cscNeutronWriter","");
    default_tag_dt  = edm::InputTag("dtNeutronWriter","");
    default_tag_rpc = edm::InputTag("rpcNeutronWriter","");
  }

  input_tag_csc_ = iConfig.getUntrackedParameter<edm::InputTag>("inputTagCSC", default_tag_csc);
  input_tag_dt_  = iConfig.getUntrackedParameter<edm::InputTag>("inputTagDT",  default_tag_dt);
  input_tag_rpc_ = iConfig.getUntrackedParameter<edm::InputTag>("inputTagCSC", default_tag_rpc);

  simhit_map_csc.setInputTag(input_tag_csc_);


  nevt_with_cscsh = nevt_with_cscsh_in_rpc = n_cscsh = n_cscsh_in_rpc = 0;
  nevt_with_rpcsh = nevt_with_rpcsh_e = nevt_with_rpcsh_b = n_rpcsh = n_rpcsh_e = n_rpcsh_b = 0;

  fill_csc_sh_tree_ = iConfig.getUntrackedParameter< bool >("fillCSCSimHitsTree",true);
  if (fill_csc_sh_tree_) bookCSCSimHitsTree();
  fill_csc_la_tree_ = iConfig.getUntrackedParameter< bool >("fillCSCLayersTree",true);
  if (fill_csc_la_tree_) bookCSCLayersTree();
  fill_csc_ch_tree_ = iConfig.getUntrackedParameter< bool >("fillCSCChambersTree",true);
  if (fill_csc_ch_tree_) bookCSCChambersTree();

  fill_rpc_sh_tree_ = iConfig.getUntrackedParameter< bool >("fillRPCSimHitsTree",true);
  if (fill_rpc_sh_tree_) bookRPCSimHitsTree();

  edm::Service<TFileService> fs;

  std::string n_simhits = "n-SimHits";
  if (!input_is_neutrons_) n_simhits = "SimHits";

  h_csc_rz_sh_xray = fs->make<TH2D>("h_csc_rz_sh_xray",("CSC "+n_simhits+" #rho-z X-ray;z, cm;#rho, cm").c_str(),2060,550,1080,750,0,750);
  h_csc_rz_sh_heatmap = fs->make<TH2D>("h_csc_rz_sh_heatmap",("CSC "+n_simhits+" #rho-z;z, cm;#rho, cm").c_str(),230,550,1080,150,0,750);


  char label[200], nlabel[200];
  for (int me=0; me<=CSC_TYPES; me++)
  {
    sprintf(label,"h_csc_nlayers_in_ch_%s",csc_type_[me].c_str());
    sprintf(nlabel,"# of layers with %s: %s CSC;# layers", n_simhits.c_str(), csc_type[me].c_str());
    h_csc_nlayers_in_ch[me]  = fs->make<TH1D>(label, nlabel, 6, 0.5, 6.5);
  }

  h_csc_nevt_fraction_with_sh =
      fs->make<TH1D>("h_csc_nevt_fraction_with_sh", ("Fraction of events with "+n_simhits+" by CSC type").c_str(), CSC_TYPES+1, -0.5, CSC_TYPES+0.5);
  for (int i=1; i<=h_csc_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_csc_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());

  h_csc_shflux_per_layer = fs->make<TH1D>("h_csc_shflux_per_layer", (n_simhits+" Flux per CSC layer at L=10^{34}").c_str(), CSC_TYPES, 0.5,  CSC_TYPES+0.5);
  for (int i=1; i<=h_csc_shflux_per_layer->GetXaxis()->GetNbins();i++)
    h_csc_shflux_per_layer->GetXaxis()->SetBinLabel(i,csc_type[i].c_str());
  h_csc_shflux_per_layer->GetYaxis()->SetTitle("Hz/cm^{2}");

  h_rpc_rz_sh_heatmap = fs->make<TH2D>("h_rpc_rz_sh_heatmap",("RPC "+n_simhits+" #rho-z;z, cm;#rho, cm").c_str(),450,0,1080,160,0,800);

  h_rpcf_nevt_fraction_with_sh =
      fs->make<TH1D>("h_rpcf_nevt_fraction_with_sh", ("Fraction of events with "+n_simhits+" by RPFf type").c_str(), RPCF_TYPES+1, -0.5, RPCF_TYPES+0.5);
  for (int i=1; i<=h_rpcf_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_rpcf_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,rpcf_type[i-1].c_str());

  h_rpcf_shflux_per_layer = fs->make<TH1D>("h_rpcf_shflux_per_layer", (n_simhits+" Flux in RPCf at L=10^{34}").c_str(), RPCF_TYPES, 0.5,  RPCF_TYPES+0.5);
  for (int i=1; i<=h_rpcf_shflux_per_layer->GetXaxis()->GetNbins();i++)
    h_rpcf_shflux_per_layer->GetXaxis()->SetBinLabel(i,rpcf_type[i].c_str());
  h_rpcf_shflux_per_layer->GetYaxis()->SetTitle("Hz/cm^{2}");

  h_rpcb_nevt_fraction_with_sh =
      fs->make<TH1D>("h_rpcb_nevt_fraction_with_sh", ("Fraction of events with "+n_simhits+" by RPFb type").c_str(), RPCB_TYPES+1, -0.5, RPCB_TYPES+0.5);
  for (int i=1; i<=h_rpcb_nevt_fraction_with_sh->GetXaxis()->GetNbins();i++)
    h_rpcb_nevt_fraction_with_sh->GetXaxis()->SetBinLabel(i,rpcb_type[i-1].c_str());

  h_mu_rz_sh_heatmap = fs->make<TH2D>("h_mu_rz_sh_heatmap", (n_simhits+" #rho-z;z, cm;#rho, cm").c_str(),480,0,1100,160,0,800);
}

// ================================================================================================
NeutronSimHitsAnalyzer::~NeutronSimHitsAnalyzer()
{}


// ================================================================================================
void
NeutronSimHitsAnalyzer::bookCSCSimHitsTree()
{
  edm::Service<TFileService> fs;
  csc_sh_tree = fs->make<TTree>("CSCSimHitsTree", "CSCSimHitsTree");
  csc_sh_tree->Branch("evtn", &evtn,"evtn/i");
  csc_sh_tree->Branch("shn", &csc_shn,"shn/i");
  csc_sh_tree->Branch("id", &c_id.e,"e/I:s:r:c:l:t");
  csc_sh_tree->Branch("sh", &c_h.x,"x/F:y:z:r:eta:phi:gz:e:t:pdg/I:w:s");
  //csc_sh_tree->Branch("", &., "/I");
  //csc_sh_tree->Branch("" , "vector<double>" , & );
}


// ================================================================================================
void
NeutronSimHitsAnalyzer::bookCSCLayersTree()
{
  edm::Service<TFileService> fs;
  csc_la_tree = fs->make<TTree>("CSCLayersTree", "CSCLayersTree");
  csc_la_tree->Branch("evtn", &evtn,"evtn/i");
  csc_la_tree->Branch("id", &c_id.e,"e/I:s:r:c:l:t");
  csc_la_tree->Branch("la", &c_la.ln,"ln/I:nh:mint/F:maxt:meant:sigmat:minw/I:maxw:mins:maxs");
}


// ================================================================================================
void
NeutronSimHitsAnalyzer::bookCSCChambersTree()
{
  edm::Service<TFileService> fs;
  csc_ch_tree = fs->make<TTree>("CSCChambersTree", "CSCChambersTree");
  csc_ch_tree->Branch("evtn", &evtn,"evtn/i");
  csc_ch_tree->Branch("id", &c_cid.e,"e/I:s:r:c:l:t");
  csc_ch_tree->Branch("ch", &c_ch.nh,"nh/I:nl:l1:ln:mint/F:maxt:meant:sigmat:minw/I:maxw:mins:maxs");
}


// ================================================================================================
void
NeutronSimHitsAnalyzer::bookRPCSimHitsTree()
{
  edm::Service<TFileService> fs;
  rpc_sh_tree = fs->make<TTree>("RPCSimHitsTree", "RPCSimHitsTree");
  rpc_sh_tree->Branch("evtn", &evtn,"evtn/i");
  rpc_sh_tree->Branch("shn", &rpc_shn,"shn/i");
  rpc_sh_tree->Branch("id", &r_id.reg,"reg/I:ring:st:sec:layer:subsec:roll:t");
  rpc_sh_tree->Branch("sh", &r_h.x,"x/F:y:z:r:eta:phi:gz:e:t:pdg/I");
  //rpc_sh_tree->Branch("", &., "/I");
  //rpc_sh_tree->Branch("" , "vector<double>" , & );
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
  LocalPoint hitLP = sh.localPosition();

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  t = sh.tof();
  pdg = sh.particleType();

  CSCDetId layerId(sh.detUnitId());
  const CSCLayer* csclayer = csc_g->layer(layerId);
  GlobalPoint hitGP = csclayer->toGlobal(hitLP);

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
    MyCSCSimHit sh = *itr;
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
      MyCSCSimHit sh = *itr;
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
  LocalPoint hitLP = sh.localPosition();

  x = hitLP.x();
  y = hitLP.y();
  z = hitLP.z();
  e = sh.energyLoss();
  t = sh.tof();
  pdg = sh.particleType();

  GlobalPoint hitGP = rpc_g->idToDet(DetId(sh.detUnitId()))->surface().toGlobal(hitLP);

  r = hitGP.perp();
  eta = hitGP.eta();
  phi = hitGP.phi();
  gz = hitGP.z();

  //w = csclayer->geometry()->wireGroup(csclayer->geometry()->nearestWire(hitLP));
  //s = csclayer->geometry()->nearestStrip(hitLP);
}


// ================================================================================================
void
NeutronSimHitsAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  evtn += 1;
  csc_shn = 0;
  rpc_shn = 0;

  ESHandle< CSCGeometry > csc_geom;
  ESHandle< DTGeometry >  dt_geom;
  ESHandle< RPCGeometry > rpc_geom;

  iSetup.get< MuonGeometryRecord >().get(csc_geom);
  iSetup.get< MuonGeometryRecord >().get(dt_geom);
  iSetup.get< MuonGeometryRecord >().get(rpc_geom);

  csc_geometry = &*csc_geom;
  dt_geometry  = &*dt_geom;
  rpc_geometry = &*rpc_geom;

  // get SimHits
  simhit_map_csc.fill(iEvent);

  vector<int> chIds = simhit_map_csc.chambersWithHits();
  if (chIds.size()) {
    //cout<<"--- chambers with hits: "<<chIds.size()<<endl;
    nevt_with_cscsh++;
  }
  bool ev_has_csc_type[CSC_TYPES+1]={0,0,0,0,0,0,0,0,0,0,0};
  bool has_cscsh_in_rpc = false;
  for (size_t ch = 0; ch < chIds.size(); ch++)
  {
    CSCDetId chId(chIds[ch]);
    c_cid.init(chId);

    std::vector<int> layer_ids = simhit_map_csc.chamberLayersWithHits(chIds[ch]);
    //if (layer_ids.size()) cout<<"------ layers with hits: "<<layer_ids.size()<<endl;

    vector<MyCSCLayer> chamber_layers;
    for  (size_t la = 0; la < layer_ids.size(); la++)
    {
      CSCDetId layerId(layer_ids[la]);
      c_id.init(layerId);

      PSimHitContainer hits = simhit_map_csc.hits(layer_ids[la]);
      //cout<<"      "<< layer_ids[la]<<" "<<layerId<<"   no. of hits = "<<hits.size()<<endl;

      vector<MyCSCSimHit> layer_simhits;
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
  for (int t=0; t<=CSC_TYPES; t++)
    if (ev_has_csc_type[t]) h_csc_nevt_fraction_with_sh->Fill(t);

  // count events with CSC hits in PRC region
  if (has_cscsh_in_rpc) nevt_with_cscsh_in_rpc++;

  //************ RPC *************

  edm::Handle< PSimHitContainer > h_rpc_simhits;
  iEvent.getByLabel(input_tag_rpc_, h_rpc_simhits);
  const PSimHitContainer* rpc_simhits = h_rpc_simhits.product();

  if (rpc_simhits->size()) nevt_with_rpcsh++;

  bool ev_has_rpcf_type[RPCF_TYPES+1]={0,0,0,0,0,0,0,0,0};
  bool ev_has_rpcb_type[RPCB_TYPES+1]={0,0,0,0,0,0,0,0,0,0,0,0,0};
  bool has_rpcsh_e = false, has_rpcsh_b = false;
  for (PSimHitContainer::const_iterator rsh = rpc_simhits->begin(); rsh!=rpc_simhits->end(); rsh++)
  {
    rpc_shn += 1;
    const PSimHit &sh = *rsh;
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
  for (int t=0; t<=RPCF_TYPES; t++)
    if (ev_has_rpcf_type[t]) h_rpcf_nevt_fraction_with_sh->Fill(t);

  // fill events with RPCb hits by RPCb type:
  for (int t=0; t<=RPCB_TYPES; t++)
    if (ev_has_rpcb_type[t]) h_rpcb_nevt_fraction_with_sh->Fill(t);

  if (has_rpcsh_e) nevt_with_rpcsh_e++;
  if (has_rpcsh_b) nevt_with_rpcsh_b++;
}


// ================================================================================================
void NeutronSimHitsAnalyzer::beginJob() {}


// ================================================================================================
void NeutronSimHitsAnalyzer::endJob()
{
  using namespace std;
  cout<<"******************* COUNTERS *******************"<<endl;
  cout<<"* #events: "<< evtn <<endl;
  cout<<"* #events with SimHits in:"<<endl;
  cout<<"*   CSC:                     "<<nevt_with_cscsh<<" ("<<(double)nevt_with_cscsh/evtn*100<<"%)"<<endl;
  cout<<"*   CSC (|eta|<1.6, st 1-3): "<<nevt_with_cscsh_in_rpc<<" ("<<(double)nevt_with_cscsh_in_rpc/evtn*100<<"%)"<<endl;
  cout<<"*   RPC:                     "<<nevt_with_rpcsh<<" ("<<(double)nevt_with_rpcsh/evtn*100<<"%)"<<endl;
  cout<<"*   RPC endcaps:             "<<nevt_with_rpcsh_e<<" ("<<(double)nevt_with_rpcsh_e/evtn*100<<"%)"<<endl;
  cout<<"*   RPC barrel:              "<<nevt_with_rpcsh_b<<" ("<<(double)nevt_with_rpcsh_b/evtn*100<<"%)"<<endl;
  cout<<"* total SimHit numbers in: "<<endl;
  cout<<"*   CSC:                     "<<n_cscsh<<" (~"<<(double)n_cscsh/nevt_with_cscsh<<" sh/evt with hits)"<<endl;
  cout<<"*   CSC (|eta|<1.6, st 1-3): "<<n_cscsh_in_rpc<<" (~"<<(double)n_cscsh_in_rpc/nevt_with_cscsh_in_rpc<<" sh/evt with hits)"<<endl;
  cout<<"*   RPC:                     "<<n_rpcsh<<" (~"<<(double)n_rpcsh/nevt_with_rpcsh<<" sh/evt with hits)"<<endl;
  cout<<"*   RPC endcaps:             "<<n_rpcsh_e<<" (~"<<(double)n_rpcsh_e/nevt_with_rpcsh_e<<" sh/evt with hits)"<<endl;
  cout<<"*   RPC barrel:              "<<n_rpcsh_b<<" (~"<< ( (nevt_with_rpcsh_b>0) ? (double)n_rpcsh_b/nevt_with_rpcsh_b : 0 )<<" sh/evt with hits)"<<endl;
  cout<<"************************************************"<<endl;

  // convert to fraction:
  h_csc_nevt_fraction_with_sh->Sumw2();
  h_csc_nevt_fraction_with_sh->Scale(1./evtn);
  h_rpcf_nevt_fraction_with_sh->Sumw2();
  h_rpcf_nevt_fraction_with_sh->Scale(1./evtn);
  h_rpcb_nevt_fraction_with_sh->Sumw2();
  h_rpcb_nevt_fraction_with_sh->Scale(1./evtn);

  // ---- calculate fluxes at L=10^34

  //  chamber areas
  const double csc_areas_cm2[CSC_TYPES+1] = {100000, 1068.32, 4108.77, 10872.75, 13559.025, 16986, 31121.2, 15716.4, 31121.2, 14542.5, 31121.2};
  const double rpcf_areas_cm2[RPCF_TYPES+1] = {100000, 11700, 17360, 11690, 19660, 11690, 19660, 11690, 19660};
  // chamber radial segmentations
  const double csc_radial_segm[CSC_TYPES+1] = {100, 36, 36, 36, 36, 18, 36, 18, 36, 18, 36};
  const double rpcf_radial_segm = 36;
  // pileup at nominal lumi
  double n_pu = 25.;
  // fraction of empty BXs
  double f_empty_bx = 0.77;
  if (!input_is_neutrons_) f_empty_bx = 1.;
  // bx rate 40 MHz
  double bxrate = 40000000.;

  h_csc_shflux_per_layer->Sumw2();
  for (int t=1; t<=CSC_TYPES; t++)
  {
    // 2 endcaps , 6 layers
    double scale = bxrate * n_pu * f_empty_bx /csc_areas_cm2[t]/csc_radial_segm[t]/2/6 /evtn;
    double rt = scale * h_csc_shflux_per_layer->GetBinContent(t);
    double er = scale * h_csc_shflux_per_layer->GetBinError(t);
    h_csc_shflux_per_layer->SetBinContent(t,rt);
    h_csc_shflux_per_layer->SetBinError(t,er);
  }

  h_rpcf_shflux_per_layer->Sumw2();
  for (int t=1; t<=RPCF_TYPES; t++)
  {
    double scale = bxrate * n_pu * f_empty_bx /rpcf_areas_cm2[t]/rpcf_radial_segm/2 /evtn;
    double rt = scale * h_rpcf_shflux_per_layer->GetBinContent(t);
    double er = scale * h_rpcf_shflux_per_layer->GetBinError(t);
    h_rpcf_shflux_per_layer->SetBinContent(t,rt);
    h_rpcf_shflux_per_layer->SetBinError(t,er);
  }


}

// ================================================================================================
//define this as a plug-in
//DEFINE_FWK_MODULE(NeutronSimHitsAnalyzer);
