#ifndef Ntuple_h
#define Ntuple_h

#include "TChain.h" 
#include "TBranch.h" 

typedef std::vector<std::vector<Float_t> > vvfloat;
typedef std::vector<std::vector<Int_t> > vvint;
typedef std::vector<Float_t> vfloat;
typedef std::vector<Int_t> vint;

struct MyNtuple
{
  void init(); // initialize to default values
  TTree* book(TTree *t, const std::string & name = "trk_eff");
  void initialize();

  // event
  Float_t eventNumber;

  // generator level information
  vfloat mc_pt;
  vfloat mc_eta;
  vfloat mc_phi;
  vint mc_vertex_id;

  // simtrack information
  vint has_mc_match;
  vfloat st_pt;
  vfloat st_eta;
  vfloat st_phi;
  vfloat st_min_dr;

  vint st_n_csc_simhits;
  vint st_n_gem_simhits;
  vint st_n_alcts;
  vint st_n_alcts_readout;
  vint st_n_clcts;
  vint st_n_clcts_readout;
  vint st_n_tmblcts;
  vint st_n_tmblcts_readout;
  vint st_n_mpclcts;
  vint st_n_mpclcts_readout;
  vint st_n_tfTracks;
  vint st_n_tfTracksAll;
  vint st_n_tfCands;
  vint st_n_tfCandsAll;
  vint st_n_gmtRegCands;
  vint st_n_gmtRegCandsAll;
  vint st_n_gmtRegBest;
  vint st_n_gmtCands;
  vint st_n_gmtCandsAll;
  vint st_n_gmtBest;
  vint st_n_l1Extra;
  vint st_n_l1ExtraAll;
  vint st_n_l1ExtraBest;

  // gem simhits
  vvint gem_sh_detUnitId;
  vvint gem_sh_particleType;
  vvfloat gem_sh_lx;
  vvfloat gem_sh_ly;
  vvfloat gem_sh_energyLoss;
  vvfloat gem_sh_pabs;
  vvfloat gem_sh_timeOfFlight;
  vvint gem_sh_detId;
  vvfloat gem_sh_gx;
  vvfloat gem_sh_gz;
  vvfloat gem_sh_gr;
  vvfloat gem_sh_geta;
  vvfloat gem_sh_gphi;
  vvint gem_sh_strip;
  vvint gem_sh_part;
  vvint has_gem_sh_l1;
  vvint has_gem_sh_l2;
  
  // gem digi 
  vvint gem_dg_detId;
  vvint gem_dg_strip;
  vvint gem_dg_bx;
  vvfloat gem_dg_lx;
  vvfloat gem_dg_ly;
  vvfloat gem_dg_gr;
  vvfloat gem_dg_geta;
  vvfloat gem_dg_gphi;
  vvfloat gem_dg_gx;
  vvfloat gem_dg_gy; 
  vvfloat gem_dg_gz;

  // gem pad
  vvint gem_pad_detId;
  vvint gem_pad_strip;
  vvint gem_pad_bx;
  vvfloat gem_pad_lx;
  vvfloat gem_pad_ly;
  vvfloat gem_pad_gr;
  vvfloat gem_pad_geta;
  vvfloat gem_pad_gphi;
  vvfloat gem_pad_gx;
  vvfloat gem_pad_gy; 
  vvfloat gem_pad_gz;
  vvfloat gem_pad_is_l1;
  vvfloat gem_pad_is_copad;

  // csc simhit
  vvint csc_sh_detUnitId;
  vvint csc_sh_particleType;
  vvfloat csc_sh_lx;
  vvfloat csc_sh_ly;
  vvfloat csc_sh_energyLoss;
  vvfloat csc_sh_pabs;
  vvfloat csc_sh_timeOfFlight;
  vvfloat csc_sh_gx;
  vvfloat csc_sh_gy;
  vvfloat csc_sh_gz;
  vvfloat csc_sh_gr;
  vvfloat csc_sh_geta;
  vvfloat csc_sh_gphi;
  vvint csc_sh_strip;
  vvint csc_sh_wire;

  // csc ALCT
  vvint csc_alct_valid;
  vvint csc_alct_quality; 
  vvint csc_alct_accel; 
  vvint csc_alct_keywire;
  vvint csc_alct_bx;
  vvint csc_alct_trknmb;
  vvint csc_alct_fullbx;
  vvint csc_alct_isGood;
  vvint csc_alct_detId;
  vvint csc_alct_deltaOk;

  // csc CLCT
  vvint csc_clct_valid;
  vvint csc_clct_pattern;
  vvint csc_clct_quality; 
  vvint csc_clct_bend;       
  vvint csc_clct_strip;      
  vvint csc_clct_bx;
  vvint csc_clct_trknmb;
  vvint csc_clct_fullbx;
  vvint csc_clct_isGood;
  vvint csc_clct_detId;
  vvint csc_clct_deltaOk;

  // csc TMB LCT
  vvint csc_tmblct_trknmb;
  vvint csc_tmblct_valid;
  vvint csc_tmblct_quality;
  vvint csc_tmblct_keywire;
  vvint csc_tmblct_strip;
  vvint csc_tmblct_pattern;
  vvint csc_tmblct_bend;
  vvint csc_tmblct_bx;
  vvfloat csc_tmblct_gemDPhi;
  vvint csc_tmblct_hasGEM;
  vvint csc_tmblct_isAlctGood;
  vvint csc_tmblct_isClctGood;
  vvint csc_tmblct_detId;
  vvint csc_tmblct_mpclink;
  

  // CSC TF
  Int_t has_TF_match;
  // number of TF 
  
  // CSC 
};

TTree* MyNtuple::book(TTree *t, const std::string & name)
{
  edm::Service< TFileService > fs;
  t = fs->make<TTree>(name.c_str(), name.c_str());

  t->Branch("eventNumber", &eventNumber);
  t->Branch("mc_pt", &mc_pt);
  t->Branch("mc_eta", &mc_eta);
  t->Branch("mc_phi", &mc_phi);
  t->Branch("mc_vertex_id", &mc_vertex_id);
  t->Branch("has_mc_match", &has_mc_match);
  t->Branch("st_pt", &st_pt);
  t->Branch("st_eta", &st_eta);
  t->Branch("st_phi", &st_phi);
  t->Branch("st_min_dr", &st_min_dr);

  t->Branch("st_n_csc_simhits",&st_n_csc_simhits);
  t->Branch("st_n_gem_simhits",&st_n_gem_simhits);
  t->Branch("st_n_alcts",&st_n_alcts);
  t->Branch("st_n_alcts_readout",&st_n_alcts_readout);
  t->Branch("st_n_clcts",&st_n_clcts);
  t->Branch("st_n_clcts_readout",&st_n_clcts_readout);
  t->Branch("st_n_tmblcts",&st_n_tmblcts);
  t->Branch("st_n_tmblcts_readout",&st_n_tmblcts_readout);
  t->Branch("st_n_mpclcts",&st_n_mpclcts);
  t->Branch("st_n_mpclcts_readout",&st_n_mpclcts_readout);
  t->Branch("st_n_tfTracks",&st_n_tfTracks);
  t->Branch("st_n_tfTracksAll",&st_n_tfTracksAll);
  t->Branch("st_n_tfCands",&st_n_tfCands);
  t->Branch("st_n_tfCandsAll",&st_n_tfCandsAll);
  t->Branch("st_n_gmtRegCands",&st_n_gmtRegCands);
  t->Branch("st_n_gmtRegCandsAll",&st_n_gmtRegCandsAll);
  t->Branch("st_n_gmtRegBest",&st_n_gmtRegBest);
  t->Branch("st_n_gmtCands",&st_n_gmtCands);
  t->Branch("st_n_gmtCandsAll",&st_n_gmtCandsAll);
  t->Branch("st_n_gmtBest",&st_n_gmtBest);
  t->Branch("st_n_l1Extra",&st_n_l1Extra);
  t->Branch("st_n_l1ExtraAll",&st_n_l1ExtraAll);
  t->Branch("st_n_l1ExtraBest",&st_n_l1ExtraBest);
  
  t->Branch("gem_sh_detUnitId", &gem_sh_detUnitId);
  t->Branch("gem_sh_particleType",&gem_sh_particleType);
  t->Branch("gem_sh_lx",&gem_sh_lx);
  t->Branch("gem_sh_ly",&gem_sh_ly);
  t->Branch("gem_sh_energyLoss",&gem_sh_energyLoss);
  t->Branch("gem_sh_pabs",&gem_sh_pabs);
  t->Branch("gem_sh_timeOfFlight",&gem_sh_timeOfFlight);
  t->Branch("gem_sh_detId",&gem_sh_detId);
  t->Branch("gem_sh_gx",&gem_sh_gx);
  t->Branch("gem_sh_gz",&gem_sh_gz);
  t->Branch("gem_sh_gr",&gem_sh_gr);
  t->Branch("gem_sh_geta",&gem_sh_geta);
  t->Branch("gem_sh_gphi",&gem_sh_gphi);
  t->Branch("gem_sh_strip",&gem_sh_strip);

  t->Branch("gem_dg_detId",&gem_dg_detId);
  t->Branch("gem_dg_strip",&gem_dg_strip);
  t->Branch("gem_dg_bx",&gem_dg_bx);
  t->Branch("gem_dg_lx",&gem_dg_lx);
  t->Branch("gem_dg_ly",&gem_dg_ly);
  t->Branch("gem_dg_gr",&gem_dg_gr);
  t->Branch("gem_dg_geta",&gem_dg_geta);
  t->Branch("gem_dg_gphi",&gem_dg_gphi);
  t->Branch("gem_dg_gx",&gem_dg_gx);
  t->Branch("gem_dg_gy",&gem_dg_gy);
  t->Branch("gem_dg_gz",&gem_dg_gz);

  t->Branch("gem_pad_detId",&gem_pad_detId);
  t->Branch("gem_pad_strip",&gem_pad_strip);
  t->Branch("gem_pad_bx",&gem_pad_bx);
  t->Branch("gem_pad_lx",&gem_pad_lx);
  t->Branch("gem_pad_ly",&gem_pad_ly);
  t->Branch("gem_pad_gr",&gem_pad_gr);
  t->Branch("gem_pad_geta",&gem_pad_geta);
  t->Branch("gem_pad_gphi",&gem_pad_gphi);
  t->Branch("gem_pad_gx",&gem_pad_gx);
  t->Branch("gem_pad_gy",&gem_pad_gy);
  t->Branch("gem_pad_gz",&gem_pad_gz);
  t->Branch("gem_pad_is_l1",&gem_pad_is_l1);
  t->Branch("gem_pad_is_copad",&gem_pad_is_copad);

  t->Branch("csc_sh_detUnitId",&csc_sh_detUnitId);
  t->Branch("csc_sh_particleType",&csc_sh_particleType);
  t->Branch("csc_sh_lx",&csc_sh_lx);
  t->Branch("csc_sh_ly",&csc_sh_ly);
  t->Branch("csc_sh_energyLoss",&csc_sh_energyLoss);
  t->Branch("csc_sh_pabs",&csc_sh_pabs);
  t->Branch("csc_sh_timeOfFlight",&csc_sh_timeOfFlight);
  t->Branch("csc_sh_gx",&csc_sh_gx);
  t->Branch("csc_sh_gy",&csc_sh_gy);
  t->Branch("csc_sh_gz",&csc_sh_gz);
  t->Branch("csc_sh_gr",&csc_sh_gr);
  t->Branch("csc_sh_geta",&csc_sh_geta);
  t->Branch("csc_sh_gphi",&csc_sh_gphi);
  t->Branch("csc_sh_strip",&csc_sh_strip);
  t->Branch("csc_sh_wire",&csc_sh_wire);

  t->Branch("csc_alct_valid",&csc_alct_valid);
  t->Branch("csc_alct_quality",&csc_alct_quality);
  t->Branch("csc_alct_accel",&csc_alct_accel);
  t->Branch("csc_alct_keywire",&csc_alct_keywire);
  t->Branch("csc_alct_bx",&csc_alct_bx);
  t->Branch("csc_alct_trknmb",&csc_alct_trknmb);
  t->Branch("csc_alct_fullbx",&csc_alct_fullbx);
  t->Branch("csc_alct_detId",&csc_alct_detId);
  t->Branch("csc_alct_isGood",&csc_alct_isGood);
  t->Branch("csc_alct_deltaOk",&csc_alct_deltaOk);

  t->Branch("csc_clct_valid",&csc_clct_valid);
  t->Branch("csc_clct_quality",&csc_clct_quality);
  t->Branch("csc_clct_pattern",&csc_clct_pattern);
  t->Branch("csc_clct_bend",&csc_clct_bend);
  t->Branch("csc_clct_strip",&csc_clct_strip);
  t->Branch("csc_clct_bx",&csc_clct_bx);
  t->Branch("csc_clct_trknmb",&csc_clct_trknmb);
  t->Branch("csc_clct_fullbx",&csc_clct_fullbx);
  t->Branch("csc_clct_isGood",&csc_clct_isGood);
  t->Branch("csc_clct_detId",&csc_clct_detId);
  t->Branch("csc_clct_deltaOk",&csc_clct_deltaOk);

  t->Branch("csc_tmblct_trknmb",&csc_tmblct_trknmb);
  t->Branch("csc_tmblct_valid",&csc_tmblct_valid);
  t->Branch("csc_tmblct_quality",&csc_tmblct_quality);
  t->Branch("csc_tmblct_keywire",&csc_tmblct_keywire);
  t->Branch("csc_tmblct_strip",&csc_tmblct_strip);
  t->Branch("csc_tmblct_pattern",&csc_tmblct_pattern);
  t->Branch("csc_tmblct_bend",&csc_tmblct_bend);
  t->Branch("csc_tmblct_bx",&csc_tmblct_bx);
  t->Branch("csc_tmblct_mpclink",&csc_tmblct_mpclink);
  t->Branch("csc_tmblct_gemDPhi",&csc_tmblct_gemDPhi);
  t->Branch("csc_tmblct_isAlctGood",&csc_tmblct_isAlctGood);
  t->Branch("csc_tmblct_isClctGood",&csc_tmblct_isClctGood);
  t->Branch("csc_tmblct_detId",&csc_tmblct_detId);
  t->Branch("csc_tmblct_gemDPhi",&csc_tmblct_gemDPhi);
  t->Branch("csc_tmblct_hasGEM",&csc_tmblct_hasGEM);

  return t;
}

void MyNtuple::initialize()
{
  eventNumber = 0;

  st_n_csc_simhits.clear();
  st_n_gem_simhits.clear();
  st_n_alcts.clear();
  st_n_alcts_readout.clear();
  st_n_clcts.clear();
  st_n_tmblcts.clear();
  st_n_mpclcts.clear();
  st_n_tfTracks.clear();
  st_n_tfTracksAll.clear();
  st_n_tfCands.clear();
  st_n_tfCandsAll.clear();
  st_n_gmtRegCands.clear();
  st_n_gmtRegCandsAll.clear();
  st_n_gmtRegBest.clear();
  st_n_gmtCands.clear();
  st_n_gmtCandsAll.clear();
  st_n_gmtBest.clear();
  st_n_l1Extra.clear();
  st_n_l1ExtraAll.clear();
  st_n_l1ExtraBest.clear();
  gem_dg_detId.clear();
  gem_dg_strip.clear();
  gem_dg_bx.clear();
  gem_dg_lx.clear();
  gem_dg_ly.clear();
  gem_dg_gr.clear();
  gem_dg_geta.clear();
  gem_dg_gphi.clear();
  gem_dg_gx.clear();
  gem_dg_gy.clear();
  gem_dg_gz.clear();
  gem_pad_detId.clear();
  gem_pad_strip.clear();
  gem_pad_bx.clear();
  gem_pad_lx.clear();
  gem_pad_ly.clear();
  gem_pad_gr.clear();
  gem_pad_geta.clear();
  gem_pad_gphi.clear();
  gem_pad_gx.clear();
  gem_pad_gy.clear();
  gem_pad_gz.clear();
  gem_pad_is_l1.clear();
  gem_pad_is_copad.clear();

  csc_sh_detUnitId.clear();
  csc_sh_particleType.clear();
  csc_sh_lx.clear();
  csc_sh_ly.clear();
  csc_sh_energyLoss.clear();
  csc_sh_pabs.clear();
  csc_sh_timeOfFlight.clear();
  csc_sh_gx.clear();
  csc_sh_gy.clear();
  csc_sh_gz.clear();
  csc_sh_gr.clear();
  csc_sh_geta.clear();
  csc_sh_gphi.clear();
  csc_sh_strip.clear();
  csc_sh_wire.clear();

  csc_alct_valid.clear();
  csc_alct_quality.clear();
  csc_alct_accel.clear();
  csc_alct_keywire.clear();
  csc_alct_bx.clear();
  csc_alct_trknmb.clear();
  csc_alct_fullbx.clear();
  csc_alct_isGood.clear();
  csc_alct_deltaOk.clear();
  csc_alct_detId.clear();

  csc_clct_valid.clear();
  csc_clct_pattern.clear();
  csc_clct_quality.clear(); 
  csc_clct_bend.clear();       
  csc_clct_strip.clear();      
  csc_clct_bx.clear();
  csc_clct_trknmb.clear();
  csc_clct_fullbx.clear();
  csc_clct_isGood.clear();
  csc_clct_detId.clear();
  csc_clct_deltaOk.clear();

/*   csc_lct_valid.clear(); */
/*   csc_lct_pattern.clear(); */
/*   csc_lct_quality.clear();  */
/*   csc_lct_bend.clear();        */
/*   csc_lct_strip.clear();       */
/*   csc_lct_bx.clear(); */
/*   csc_lct_trknmb.clear(); */
/*   csc_lct_isAlctGood.clear(); */
/*   csc_lct_isClctGood.clear(); */
/*   csc_lct_isAlctClctGood.clear(); */
/*   csc_lct_detId.clear(); */
/*   csc_lct_gemDPhi.clear(); */
/*   csc_lct_hasGEM.clear(); */
/*   csc_lct_mpclink.clear(); */

  csc_tmblct_valid.clear();
  csc_tmblct_pattern.clear();
  csc_tmblct_quality.clear(); 
  csc_tmblct_bend.clear();       
  csc_tmblct_strip.clear();      
  csc_tmblct_bx.clear();
  csc_tmblct_trknmb.clear();
  csc_tmblct_isAlctGood.clear();
  csc_tmblct_isClctGood.clear();
  csc_tmblct_detId.clear();
  csc_tmblct_gemDPhi.clear();
  csc_tmblct_hasGEM.clear();
  csc_tmblct_mpclink.clear();

}

#endif
