#ifndef Ntuple_h
#define Ntuple_h

#include "TChain.h" 

#define MAXMU 1000
#define MAXGEMSIMHIT 1000
#define MAXGEMDIGI 1000
#define MAXGEMPAD 1000
#define MAXCSCSIMHIT 100
#define MAXCSCWIREDIGI 1000
#define MAXCSCSTRIPDIGI 1000
#define MAXCSCALCT 1000
#define MAXCSCCLCT 1000
#define MAXCSCLCTTMB 1000
#define MAXCSCLCTMPC 1000
#define MAXCSCTF 1000
#define MAXCSCTFCAND 1000
#define MAXL1GMTCAND 1000
#define MAXL1GMTFCAND 1000
#define MAXL1GMTCSCCAND 1000
#define MAXL1GMTRPCFCAND 1000
#define MAXL1GMTRPCBCAND 1000
#define MAXL1GMTDTCAND 1000

struct MyNtuple
{
  void init(); // initialize to default values
  TTree* book(TTree *t, const std::string & name = "trk_eff");

  // event
  float eventNumber;

  // generator level information
  float mc_pt[MAXMU];
  float mc_eta[MAXMU];
  float mc_phi[MAXMU];
  int mc_vertex_id[MAXMU];
  int has_mc_match[MAXMU];

  // simtrack information
  float st_pt[MAXMU];
  float st_eta[MAXMU];
  float st_phi[MAXMU];
  bool st_is_matched[MAXMU];
  float st_min_dr[MAXMU];
  int st_n_csc_simhits[MAXGEMSIMHIT];
  int st_n_gem_simhits[MAXGEMSIMHIT];

  /*   Char_t gem_sh_layer1, gem_sh_layer2; // bit1: in odd  bit2: even */
  /*   Float_t gem_sh_eta, gem_sh_phi; */
  /*   Float_t gem_trk_eta, gem_trk_phi, gem_trk_rho; */
  /*   Float_t gem_lx_even, gem_ly_even; */
  /*   Float_t gem_lx_odd, gem_ly_odd; */
  /*   Char_t  has_gem_sh_l1, has_gem_sh_l2; */
  
  // gem simhits
  std::vector<std::vector<Int_t> > gem_sh_detUnitId;
  //Int_t gem_sh_detUnitId[MAXMU][MAXGEMSIMHIT];
  Int_t gem_sh_particleType[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_lx[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_ly[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_energyLoss[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_pabs[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_timeOfFlight[MAXMU][MAXGEMSIMHIT];
  Int_t gem_sh_detId[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_gx[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_gz[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_gr[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_geta[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_gphi[MAXMU][MAXGEMSIMHIT];
  Int_t gem_sh_strip[MAXMU][MAXGEMSIMHIT];
  Int_t gem_sh_part[MAXMU][MAXGEMSIMHIT];

  // gem digi 
  Int_t gem_dg_detId[MAXMU][MAXGEMDIGI];
  Short_t gem_dg_strip[MAXMU][MAXGEMDIGI];
  Short_t gem_dg_bx[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_lx[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_ly[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_gr[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_geta[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_gphi[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_gx[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_gy[MAXMU][MAXGEMDIGI]; 
  Float_t gem_dg_gz[MAXMU][MAXGEMDIGI];

  // gem pad
  Int_t   gem_l1_pad_detId[MAXMU];
  Short_t gem_l1_pad_strip[MAXMU];
  Short_t gem_l1_pad_bx[MAXMU];
  Float_t gem_l1_pad_lx[MAXMU];
  Float_t gem_l1_pad_ly[MAXMU];
  Float_t gem_l1_pad_gr[MAXMU];
  Float_t gem_l1_pad_geta[MAXMU];
  Float_t gem_l1_pad_gphi[MAXMU];
  Float_t gem_l1_pad_gx[MAXMU];
  Float_t gem_l1_pad_gy[MAXMU]; 
  Float_t gem_l1_pad_gz[MAXMU];

  Int_t   gem_l2_pad_detId[MAXMU];
  Short_t gem_l2_pad_strip[MAXMU];
  Short_t gem_l2_pad_bx[MAXMU];
  Float_t gem_l2_pad_lx[MAXMU];
  Float_t gem_l2_pad_ly[MAXMU];
  Float_t gem_l2_pad_gr[MAXMU];
  Float_t gem_l2_pad_geta[MAXMU];
  Float_t gem_l2_pad_gphi[MAXMU];
  Float_t gem_l2_pad_gx[MAXMU];
  Float_t gem_l2_pad_gy[MAXMU]; 
  Float_t gem_l2_pad_gz[MAXMU];

  // csc simhit
  Int_t csc_sh_detUnitId[MAXMU][MAXCSCSIMHIT];
  Int_t csc_sh_particleType[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_lx[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_ly[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_energyLoss[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_pabs[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_timeOfFlight[MAXMU][MAXCSCSIMHIT];
  Int_t csc_sh_detId[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_gx[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_gz[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_gr[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_geta[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_gphi[MAXMU][MAXCSCSIMHIT];
  Int_t csc_sh_strip[MAXMU][MAXCSCSIMHIT];
  Int_t csc_sh_wire[MAXMU][MAXCSCSIMHIT];

  // csc ALCT
  // max 5 stubs per muon
 
  uint16_t csc_st1_alct_valid[MAXMU];
  uint16_t csc_st1_alct_quality[MAXMU]; 
  uint16_t csc_st1_alct_accel[MAXMU]; 
  uint16_t csc_st1_alct_keywire[MAXMU];
  uint16_t csc_st1_alct_bx[MAXMU];
  uint16_t csc_st1_alct_trknmb[MAXMU];
  uint16_t csc_st1_alct_fullbx[MAXMU];
  uint16_t csc_st1_alct_isGood[MAXMU];

  uint16_t csc_st2_alct_valid[MAXMU];
  uint16_t csc_st2_alct_quality[MAXMU]; 
  uint16_t csc_st2_alct_accel[MAXMU]; 
  uint16_t csc_st2_alct_keywire[MAXMU];
  uint16_t csc_st2_alct_bx[MAXMU];
  uint16_t csc_st2_alct_trknmb[MAXMU];
  uint16_t csc_st2_alct_fullbx[MAXMU];
  uint16_t csc_st2_alct_isGood[MAXMU];

  uint16_t csc_st3_alct_valid[MAXMU];
  uint16_t csc_st3_alct_quality[MAXMU]; 
  uint16_t csc_st3_alct_accel[MAXMU]; 
  uint16_t csc_st3_alct_keywire[MAXMU];
  uint16_t csc_st3_alct_bx[MAXMU];
  uint16_t csc_st3_alct_trknmb[MAXMU];
  uint16_t csc_st3_alct_fullbx[MAXMU];
  uint16_t csc_st3_alct_isGood[MAXMU];

  uint16_t csc_st4_alct_valid[MAXMU];
  uint16_t csc_st4_alct_quality[MAXMU]; 
  uint16_t csc_st4_alct_accel[MAXMU]; 
  uint16_t csc_st4_alct_keywire[MAXMU];
  uint16_t csc_st4_alct_bx[MAXMU];
  uint16_t csc_st4_alct_trknmb[MAXMU];
  uint16_t csc_st4_alct_fullbx[MAXMU];
  uint16_t csc_st4_alct_isGood[MAXMU];

  uint16_t csc_st5_alct_valid[MAXMU];
  uint16_t csc_st5_alct_quality[MAXMU]; 
  uint16_t csc_st5_alct_accel[MAXMU]; 
  uint16_t csc_st5_alct_keywire[MAXMU];
  uint16_t csc_st5_alct_bx[MAXMU];
  uint16_t csc_st5_alct_trknmb[MAXMU];
  uint16_t csc_st5_alct_fullbx[MAXMU];
  uint16_t csc_st5_alct_isGood[MAXMU];

  // csc CLCT
  uint16_t csc_st1_clct_valid[MAXMU];      
  uint16_t csc_st1_clct_quality[MAXMU];    
  uint16_t csc_st1_clct_pattern[MAXMU];    
  uint16_t csc_st1_clct_bend[MAXMU];       
  uint16_t csc_st1_clct_strip[MAXMU];      
  uint16_t csc_st1_clct_cfeb[MAXMU];      
  uint16_t csc_st1_clct_bx[MAXMU];        
  uint16_t csc_st1_clct_trknmb[MAXMU];    
  uint16_t csc_st1_clct_fullbx[MAXMU];
  uint16_t csc_st1_clct_isGood[MAXMU];

  // csc LCT
  uint16_t csc_lct_trknmb[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_valid[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_quality[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_keywire[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_strip[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_pattern[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_bend[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_bx[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_mpclink[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_bx0[MAXMU][MAXCSCSIMHIT]; 
  uint16_t csc_lct_syncErr[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_cscID[MAXMU][MAXCSCSIMHIT];
  float csc_lct_gemDPhi[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_isGood[MAXMU][MAXCSCSIMHIT];     

  // CSC TF
  

};

// ------------------------------------------------------------------------
/* //! branch addresses settings */
/* void setBranchAddresses(TTree* chain, Ntuple& treeVars); */

/* //! create branches for a tree */
/* void setBranches(TTree* chain, Ntuple& treeVars); */

/* //! initialize branches */
/* void initializeBranches(TTree* chain, Ntuple& treeVars); */


TTree* MyNtuple::book(TTree *t, const std::string & name)
{
  edm::Service< TFileService > fs;
  t = fs->make<TTree>(name.c_str(), name.c_str());

  t->Branch("eventNumber", &eventNumber);
  //  t->Branch("mc_pt",         &mc_pt,                "mc_pt[MAXMU]/F");
  t->Branch("gem_sh_detUnitId","std::vector<std::vector<Int_t> >", &gem_sh_detUnitId);
/*   t->Branch("st_pt", st_pt, "st_pt[MAXMU]/F"); */
/*   t->Branch("st_eta", st_eta, "st_eta[MAXMU]/F"); */
/*   t->Branch("st_phi", st_phi, "st_phi[MAXMU]/F"); */
/*   t->Branch("csc_sh_pabs", csc_sh_pabs, "csc_sh_pabs[MAXMU][MAXGEMSIMHIT]/F"); */

/*   t->SetBranchAddress("st_pt", st_pt); */
/*   t->SetBranchAddress("st_eta", st_eta); */
/*   t->SetBranchAddress("st_phi", st_phi); */
/*   t->SetBranchAddress("csc_sh_pabs", csc_sh_pabs); */


  //  t -> SetBranchAddress("st_pt",       &st_pt);
  return t;


  /*
  // generator level information
  float mc_pt[MAXMU];
  float mc_eta[MAXMU];
  float mc_phi[MAXMU];
  int mc_vertex_id[MAXMU];

  // simtrack information
  float st_pt[MAXMU];
  float sim_eta[MAXMU];
  float sim_phi[MAXMU];
  bool st_is_matched[MAXMU];
  float st_min_dr[MAXMU];
  int st_n_csc_simhits[MAXGEMSIMHIT];
  int st_n_gem_simhits[MAXGEMSIMHIT];

  // gem simhits
  Int_t gem_sh_detUnitId[MAXMU][MAXGEMSIMHIT];
  Int_t gem_sh_particleType[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_lx[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_ly[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_energyLoss[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_pabs[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_timeOfFlight[MAXMU][MAXGEMSIMHIT];
  Int_t gem_sh_detId[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_gx[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_gz[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_gz[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_gr[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_geta[MAXMU][MAXGEMSIMHIT];
  Float_t gem_sh_gphi[MAXMU][MAXGEMSIMHIT];
  Int_t gem_sh_strip[MAXMU][MAXGEMSIMHIT];
  Int_t gem_sh_part[MAXMU][MAXGEMSIMHIT];

  // gem digi 
  Int_t gem_dg_detId[MAXMU][MAXGEMDIGI];
  Short_t gem_dg_strip[MAXMU][MAXGEMDIGI];
  Short_t gem_dg_bx[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_lx[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_ly[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_gr[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_geta[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_gphi[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_gx[MAXMU][MAXGEMDIGI];
  Float_t gem_dg_gy[MAXMU][MAXGEMDIGI]; 
  Float_t gem_dg_gz[MAXMU][MAXGEMDIGI];

  // gem pad
  Int_t gem_pad_detId[MAXMU][MAXGEMPAD];
  Short_t gem_pad_strip[MAXMU][MAXGEMPAD];
  Short_t gem_pad_bx[MAXMU][MAXGEMPAD];
  Float_t gem_pad_lx[MAXMU][MAXGEMPAD];
  Float_t gem_pad_ly[MAXMU][MAXGEMPAD];
  Float_t gem_pad_gr[MAXMU][MAXGEMPAD];
  Float_t gem_pad_geta[MAXMU][MAXGEMPAD];
  Float_t gem_pad_gphi[MAXMU][MAXGEMPAD];
  Float_t gem_pad_gx[MAXMU][MAXGEMPAD];
  Float_t gem_pad_gy[MAXMU][MAXGEMPAD]; 
  Float_t gem_pad_gz[MAXMU][MAXGEMPAD];

  // gem copad
  Int_t gem_copad_detId[MAXMU][MAXGEMPAD];
  Short_t gem_copad_strip[MAXMU][MAXGEMPAD];
  Short_t gem_copad_bx[MAXMU][MAXGEMPAD];
  Float_t gem_copad_lx[MAXMU][MAXGEMPAD];
  Float_t gem_copad_ly[MAXMU][MAXGEMPAD];
  Float_t gem_copad_gr[MAXMU][MAXGEMPAD];
  Float_t gem_copad_geta[MAXMU][MAXGEMPAD];
  Float_t gem_copad_gphi[MAXMU][MAXGEMPAD];
  Float_t gem_copad_gx[MAXMU][MAXGEMPAD];
  Float_t gem_copad_gy[MAXMU][MAXGEMPAD]; 
  Float_t gem_copad_gz[MAXMU][MAXGEMPAD];

  // csc simhit
  Int_t csc_sh_detUnitId[MAXMU][MAXCSCSIMHIT];
  Int_t csc_sh_particleType[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_lx[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_ly[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_energyLoss[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_pabs[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_timeOfFlight[MAXMU][MAXCSCSIMHIT];
  Int_t csc_sh_detId[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_gx[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_gz[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_gz[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_gr[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_geta[MAXMU][MAXCSCSIMHIT];
  Float_t csc_sh_gphi[MAXMU][MAXCSCSIMHIT];
  Int_t csc_sh_strip[MAXMU][MAXCSCSIMHIT];
  Int_t csc_sh_wire[MAXMU][MAXCSCSIMHIT];

  // csc ALCT
  uint16_t csc_alct_valid[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_alct_quality[MAXMU][MAXCSCSIMHIT];    
  uint16_t csc_alct_accel[MAXMU][MAXCSCSIMHIT];     
  uint16_t csc_alct_keywire[MAXMU][MAXCSCSIMHIT];    
  uint16_t csc_alct_bx[MAXMU][MAXCSCSIMHIT];         
  uint16_t csc_alct_trknmb[MAXMU][MAXCSCSIMHIT];     
  uint16_t csc_alct_fullbx[MAXMU][MAXCSCSIMHIT];     
  uint16_t csc_alct_isGood[MAXMU][MAXCSCSIMHIT];     

  // csc CLCT
  uint16_t csc_clct_valid[MAXMU][MAXCSCSIMHIT];      
  uint16_t csc_clct_quality[MAXMU][MAXCSCSIMHIT];    
  uint16_t csc_clct_pattern[MAXMU][MAXCSCSIMHIT];    
  uint16_t csc_clct_bend[MAXMU][MAXCSCSIMHIT];       
  uint16_t csc_clct_strip[MAXMU][MAXCSCSIMHIT];      
  uint16_t csc_clct_cfeb[MAXMU][MAXCSCSIMHIT];      
  uint16_t csc_clct_bx[MAXMU][MAXCSCSIMHIT];        
  uint16_t csc_clct_trknmb[MAXMU][MAXCSCSIMHIT];     
  uint16_t csc_clct_fullbx[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_clct_isGood[MAXMU][MAXCSCSIMHIT];     

  // csc LCT
  uint16_t csc_lct_trknmb[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_valid[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_quality[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_keywire[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_strip[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_pattern[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_bend[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_bx[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_mpclink[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_bx0[MAXMU][MAXCSCSIMHIT]; 
  uint16_t csc_lct_syncErr[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_cscID[MAXMU][MAXCSCSIMHIT];
  float csc_lct_gemDPhi[MAXMU][MAXCSCSIMHIT];
  uint16_t csc_lct_isGood[MAXMU][MAXCSCSIMHIT];     


  chain -> SetBranchAddress("runId",       &treeVars.runId);
  chain -> SetBranchAddress("lumiSection", &treeVars.lumiSection);
  chain -> SetBranchAddress("orbit",       &treeVars.orbit);
  chain -> SetBranchAddress("bx",          &treeVars.bx);
  chain -> SetBranchAddress("eventId",     &treeVars.eventId);
  chain -> SetBranchAddress("triggered",   &treeVars.triggered);
  chain -> SetBranchAddress("L1a",         &treeVars.L1a );
  
  //Selection variables
  chain -> SetBranchAddress("Triggersel",      &treeVars.Triggersel     );
  chain -> SetBranchAddress("Photonsel",      &treeVars.Photonsel     );
  chain -> SetBranchAddress("Vertexsel",      &treeVars.Vertexsel     );
  chain -> SetBranchAddress("Beamhalosel",      &treeVars.Beamhalosel     );
  chain -> SetBranchAddress("Jetsel",      &treeVars.Jetsel     );
  chain -> SetBranchAddress("METsel",      &treeVars.METsel     );

  // PAT VARIABLES
  chain -> SetBranchAddress("nMuons",      &treeVars.nMuons     );
  chain -> SetBranchAddress("nElectrons",  &treeVars.nElectrons );
  chain -> SetBranchAddress("nJets",       &treeVars.nJets      );
  chain -> SetBranchAddress("nPhotons",    &treeVars.nPhotons   );
  chain -> SetBranchAddress("nVertices",   &treeVars.nVertices  );
  chain -> SetBranchAddress("totalNVtx",   &treeVars.totalNVtx  );
  chain -> SetBranchAddress("nGen",        &treeVars.nGen       );
  chain -> SetBranchAddress("nOutTimeHits",  &treeVars.nOutTimeHits );
  chain -> SetBranchAddress("nHaloTrack",    &treeVars.nHaloTrack );
  chain -> SetBranchAddress("haloPhi",       &treeVars.haloPhi    );
  chain -> SetBranchAddress("haloRho",       &treeVars.haloRho    );

  chain -> SetBranchAddress("met0Px",      &treeVars.met0Px     );
  chain -> SetBranchAddress("met0Py",      &treeVars.met0Py     );
  chain -> SetBranchAddress("met0",        &treeVars.met0       );
  chain -> SetBranchAddress("metPx",       &treeVars.metPx     );
  chain -> SetBranchAddress("metPy",       &treeVars.metPy     );
  chain -> SetBranchAddress("met",         &treeVars.met       );
  chain -> SetBranchAddress("met_dx1",     &treeVars.met_dx1    );
  chain -> SetBranchAddress("met_dy1",     &treeVars.met_dy1    );
  chain -> SetBranchAddress("met_dx2",     &treeVars.met_dx2    );
  chain -> SetBranchAddress("met_dy2",     &treeVars.met_dy2    );
  chain -> SetBranchAddress("met_dx3",     &treeVars.met_dx3    );
  chain -> SetBranchAddress("met_dy3",     &treeVars.met_dy3    );

  // trigger matched objects 
  chain -> SetBranchAddress("t_phoPx",      &treeVars.t_phoPx    );
  chain -> SetBranchAddress("t_phoPy",      &treeVars.t_phoPy    );
  chain -> SetBranchAddress("t_phoPz",      &treeVars.t_phoPz    );
  chain -> SetBranchAddress("t_phoE",       &treeVars.t_phoE     );
  chain -> SetBranchAddress("t_phoPt",      &treeVars.t_phoPt     );
  chain -> SetBranchAddress("t_phodR",      &treeVars.t_phodR    );
  chain -> SetBranchAddress("t_phoDxy",      &treeVars.t_phoDxy    );
  chain -> SetBranchAddress("t_phoDz",      &treeVars.t_phoDz    );
  chain -> SetBranchAddress("t_metPx",      &treeVars.t_metPx    );
  chain -> SetBranchAddress("t_metPy",      &treeVars.t_metPy    );
  chain -> SetBranchAddress("t_met",        &treeVars.t_met      );
  chain -> SetBranchAddress("t_metdR",      &treeVars.t_metdR    );

  // vertex variables
  chain -> SetBranchAddress("vtxNTracks", treeVars.vtxNTracks);
  chain -> SetBranchAddress("vtxChi2",    treeVars.vtxChi2   );
  chain -> SetBranchAddress("vtxNdof",    treeVars.vtxNdof   );
  chain -> SetBranchAddress("vtxX",       treeVars.vtxX      );
  chain -> SetBranchAddress("vtxY",       treeVars.vtxY      );
  chain -> SetBranchAddress("vtxZ",       treeVars.vtxZ      );
  chain -> SetBranchAddress("vtxDx",      treeVars.vtxDx     );
  chain -> SetBranchAddress("vtxDy",      treeVars.vtxDy     );
  chain -> SetBranchAddress("vtxDz",      treeVars.vtxDz     );

  chain -> SetBranchAddress("muPx",        treeVars.muPx       );
  chain -> SetBranchAddress("muPy",        treeVars.muPy       );
  chain -> SetBranchAddress("muPz",        treeVars.muPz       );
  chain -> SetBranchAddress("muE",         treeVars.muE        );
  
  chain -> SetBranchAddress("elePx",        treeVars.elePx     );
  chain -> SetBranchAddress("elePy",        treeVars.elePy     );
  chain -> SetBranchAddress("elePz",        treeVars.elePz     );
  chain -> SetBranchAddress("eleE",         treeVars.eleE      );
  chain -> SetBranchAddress("eleNLostHits", treeVars.eleNLostHits );
  chain -> SetBranchAddress("eleEcalIso",   treeVars.eleEcalIso ) ;
  chain -> SetBranchAddress("eleHcalIso",   treeVars.eleHcalIso ) ;
  chain -> SetBranchAddress("eleTrkIso",    treeVars.eleTrkIso ) ;
  
  chain -> SetBranchAddress("jetPx",        treeVars.jetPx     );
  chain -> SetBranchAddress("jetPy",        treeVars.jetPy     );
  chain -> SetBranchAddress("jetPz",        treeVars.jetPz     );
  chain -> SetBranchAddress("jetE",         treeVars.jetE      );
  chain -> SetBranchAddress("jetNDau",      treeVars.jetNDau   );
  chain -> SetBranchAddress("jetCM",        treeVars.jetCM     );
  chain -> SetBranchAddress("jetCEF",       treeVars.jetCEF    );
  chain -> SetBranchAddress("jetCHF",       treeVars.jetCHF    );
  chain -> SetBranchAddress("jetNHF",       treeVars.jetNHF    );
  chain -> SetBranchAddress("jetNEF",       treeVars.jetNEF    );
  chain -> SetBranchAddress("jerUnc",        treeVars.jerUnc     );
  chain -> SetBranchAddress("jecUnc",       treeVars.jecUnc    );
  //chain -> SetBranchAddress("jecUncU",      treeVars.jecUncU  );
  //chain -> SetBranchAddress("jecUncD",      treeVars.jecUncD  );
  
  chain -> SetBranchAddress("phoPx",        treeVars.phoPx     );
  chain -> SetBranchAddress("phoPy",        treeVars.phoPy     );
  chain -> SetBranchAddress("phoPz",        treeVars.phoPz     );
  chain -> SetBranchAddress("phoE",         treeVars.phoE      );
  chain -> SetBranchAddress("phoPt",         treeVars.phoPt      );
  chain -> SetBranchAddress("phoEcalIso",   treeVars.phoEcalIso ) ;
  chain -> SetBranchAddress("phoDxy",         treeVars.phoDxy      );
  chain -> SetBranchAddress("phoDz",         treeVars.phoDz      );
  chain -> SetBranchAddress("phoHcalIso",   treeVars.phoHcalIso ) ;
  chain -> SetBranchAddress("phoTrkIso",    treeVars.phoTrkIso ) ;
  chain -> SetBranchAddress("cHadIso",      treeVars.cHadIso ) ;
  chain -> SetBranchAddress("nHadIso",      treeVars.nHadIso ) ;
  chain -> SetBranchAddress("photIso",      treeVars.photIso ) ;
  chain -> SetBranchAddress("dR_TrkPho",    treeVars.dR_TrkPho ) ;
  chain -> SetBranchAddress("pt_TrkPho",    treeVars.pt_TrkPho ) ;
  chain -> SetBranchAddress("phoHoverE",    treeVars.phoHoverE ) ;
  chain -> SetBranchAddress("sMinPho",      treeVars.sMinPho ) ;
  chain -> SetBranchAddress("sMajPho",      treeVars.sMajPho ) ;
  chain -> SetBranchAddress("seedTime",     treeVars.seedTime ) ;
  chain -> SetBranchAddress("seedTimeErr",  treeVars.seedTimeErr ) ;
  chain -> SetBranchAddress("aveTime",      treeVars.aveTime ) ;
  chain -> SetBranchAddress("aveTimeErr",   treeVars.aveTimeErr ) ;
  chain -> SetBranchAddress("aveTime1",     treeVars.aveTime1 ) ;
  chain -> SetBranchAddress("aveTimeErr1",  treeVars.aveTimeErr1 ) ;
  chain -> SetBranchAddress("timeChi2",     treeVars.timeChi2 ) ;
  chain -> SetBranchAddress("nXtals",       treeVars.nXtals ) ;
  chain -> SetBranchAddress("fSpike",       treeVars.fSpike ) ;
  chain -> SetBranchAddress("maxSwissX",    treeVars.maxSwissX ) ;
  chain -> SetBranchAddress("seedSwissX",   treeVars.seedSwissX ) ;
  chain -> SetBranchAddress("nBC",          treeVars.nBC ) ;
  chain -> SetBranchAddress("sigmaEta",     treeVars.sigmaEta ) ;
  chain -> SetBranchAddress("sigmaIeta",    treeVars.sigmaIeta ) ;
  chain -> SetBranchAddress("cscRho",       treeVars.cscRho ) ;
  chain -> SetBranchAddress("cscdPhi",      treeVars.cscdPhi ) ;
  chain -> SetBranchAddress("dtdPhi",       treeVars.dtdPhi ) ;
  chain -> SetBranchAddress("dtdEta",       treeVars.dtdEta ) ;

  chain -> SetBranchAddress("genPx",        treeVars.genPx       );
  chain -> SetBranchAddress("genPy",        treeVars.genPy       );
  chain -> SetBranchAddress("genPz",        treeVars.genPz       );
  chain -> SetBranchAddress("genE",         treeVars.genE        );
  chain -> SetBranchAddress("genM",         treeVars.genM        );
  chain -> SetBranchAddress("genVx",        treeVars.genVx       );
  chain -> SetBranchAddress("genVy",        treeVars.genVy       );
  chain -> SetBranchAddress("genVz",        treeVars.genVz       );
  chain -> SetBranchAddress("genT",         treeVars.genT        );
  chain -> SetBranchAddress("pdgId",        treeVars.pdgId       );
  chain -> SetBranchAddress("momId",        treeVars.momId       );
*/
  
}

/* /\* */
/* void setBranches(TTree* chain, Ntuple& treeVars) */
/* { */
/*   chain -> Branch("runId",         &treeVars.runId,                "runId/i"); */
/*   chain -> Branch("lumiSection",   &treeVars.lumiSection,    "lumiSection/i"); */
/*   chain -> Branch("orbit",         &treeVars.orbit,                "orbit/i"); */
/*   chain -> Branch("bx",            &treeVars.bx,                      "bx/i"); */
/*   chain -> Branch("eventId",       &treeVars.eventId,            "eventId/i"); */
/*   chain -> Branch("triggered",     &treeVars.triggered,        "triggered/I"); */
/*   chain -> Branch("L1a",           &treeVars.L1a,              "L1a/I"); */

  
/*   //Selection variables */
/*   chain -> Branch("Triggersel",       &treeVars.Triggersel,                "Triggersel/I"); */
/*   chain -> Branch("Photonsel",       &treeVars.Photonsel,                "Photonsel/I"); */
/*   chain -> Branch("Vertexsel",       &treeVars.Vertexsel,                "Vertexsel/I"); */
/*   chain -> Branch("Beamhalosel",       &treeVars.Beamhalosel,                "Beamhalosel/I"); */
/*   chain -> Branch("Jetsel",       &treeVars.Jetsel,                "Jetsel/I"); */
/*   chain -> Branch("METsel",       &treeVars.METsel,                "METsel/I"); */

/*   // RECO VARIABLES */
/*   chain -> Branch("nMuons",      &treeVars.nMuons,               "nMuons/I"); */
/*   chain -> Branch("nElectrons",  &treeVars.nElectrons,           "nElectrons/I"); */
/*   chain -> Branch("nJets",       &treeVars.nJets,                "nJets/I"); */
/*   chain -> Branch("nPhotons",    &treeVars.nPhotons,             "nPhotons/I"); */
/*   chain -> Branch("nVertices",   &treeVars.nVertices,            "nVertices/I"); */
/*   chain -> Branch("totalNVtx",   &treeVars.totalNVtx,            "totalNVtx/I"); */
/*   chain -> Branch("nGen",        &treeVars.nGen,                 "nGen/I"); */

/*   chain -> Branch("nOutTimeHits",  &treeVars.nOutTimeHits, "nOutTimeHits/I" ); */
/*   chain -> Branch("nHaloTrack",    &treeVars.nHaloTrack,   "nHaloTrack/I" ); */
/*   chain -> Branch("haloPhi",       &treeVars.haloPhi,      "haloPhi/F" ); */
/*   chain -> Branch("haloRho",       &treeVars.haloRho,      "haloRho/F" ); */

/*   chain -> Branch("met0Px",      &treeVars.met0Px,               "met0Px/F"); */
/*   chain -> Branch("met0Py",      &treeVars.met0Py,               "met0Py/F"); */
/*   chain -> Branch("met0",        &treeVars.met0,                 "met0/F"); */
/*   chain -> Branch("metPx",       &treeVars.metPx,                "metPx/F"); */
/*   chain -> Branch("metPy",       &treeVars.metPy,                "metPy/F"); */
/*   chain -> Branch("met",         &treeVars.met,                  "met/F"); */
/*   chain -> Branch("met_dx1",     &treeVars.met_dx1,              "met_dx1/F"); */
/*   chain -> Branch("met_dy1",     &treeVars.met_dy1,              "met_dy1/F"); */
/*   chain -> Branch("met_dx2",     &treeVars.met_dx2,              "met_dx2/F"); */
/*   chain -> Branch("met_dy2",     &treeVars.met_dy2,              "met_dy2/F"); */
/*   chain -> Branch("met_dx3",     &treeVars.met_dx3,              "met_dx3/F"); */
/*   chain -> Branch("met_dy3",     &treeVars.met_dy3,              "met_dy3/F"); */


/*   chain -> Branch("t_metPx",     &treeVars.t_metPx,              "t_metPx/F"); */
/*   chain -> Branch("t_metPy",     &treeVars.t_metPy,              "t_metPy/F"); */
/*   chain -> Branch("t_met",       &treeVars.t_met,                "t_met/F"); */
/*   chain -> Branch("t_metdR",     &treeVars.t_metdR,              "t_metdR/F"); */
/*   chain -> Branch("t_phoPx",     &treeVars.t_phoPx,              "t_phoPx/F"); */
/*   chain -> Branch("t_phoPy",     &treeVars.t_phoPy,              "t_phoPy/F"); */
/*   chain -> Branch("t_phoPz",     &treeVars.t_phoPz,              "t_phoPz/F"); */
/*   chain -> Branch("t_phoE",      &treeVars.t_phoE,               "t_phoE/F"); */
/*   chain -> Branch("t_phoPt",     &treeVars.t_phoPt,              "t_phoPt/F"); */
/*   chain -> Branch("t_phodR",     &treeVars.t_phodR,              "t_phodR/F"); */
/*   chain -> Branch("t_phoDxy",     &treeVars.t_phoDxy,              "t_phoDxy/F"); */
/*   chain -> Branch("t_phoDz",     &treeVars.t_phoDz,              "t_phoDz/F"); */

/*   chain -> Branch("muPx",        treeVars.muPx,                 "muPx[nMuons]/F"); */
/*   chain -> Branch("muPy",        treeVars.muPy,                 "muPy[nMuons]/F"); */
/*   chain -> Branch("muPz",        treeVars.muPz,                 "muPz[nMuons]/F"); */
/*   chain -> Branch("muE",         treeVars.muE,                  "muE[nMuons]/F"); */
  
/*   chain -> Branch("elePx",        treeVars.elePx,                 "elePx[nElectrons]/F"); */
/*   chain -> Branch("elePy",        treeVars.elePy,                 "elePy[nElectrons]/F"); */
/*   chain -> Branch("elePz",        treeVars.elePz,                 "elePz[nElectrons]/F"); */

/*   chain -> Branch("eleE",         treeVars.eleE,                  "eleE[nElectrons]/F"); */

/*   chain -> Branch("eleNLostHits", treeVars.eleNLostHits,          "eleNLostHits[nElectrons]/I" ); */

/*   chain -> Branch("eleEcalIso",   treeVars.eleEcalIso,            "eleEcalIso[nElectrons]/F") ; */

/*   chain -> Branch("eleHcalIso",   treeVars.eleHcalIso,            "eleHcalIso[nElectrons]/F") ; */
/*   chain -> Branch("eleTrkIso",    treeVars.eleTrkIso,             "eleTrkIso[nElectrons]/F" ) ; */
  
/*   chain -> Branch("jetPx",        treeVars.jetPx,                 "jetPx[nJets]/F"); */
/*   chain -> Branch("jetPy",        treeVars.jetPy,                 "jetPy[nJets]/F"); */
/*   chain -> Branch("jetPz",        treeVars.jetPz,                 "jetPz[nJets]/F"); */
/*   chain -> Branch("jetE",         treeVars.jetE,                  "jetE[nJets]/F"); */
/*   chain -> Branch("jetNDau",      treeVars.jetNDau,               "jetNDau[nJets]/I"); */
/*   chain -> Branch("jetCM",        treeVars.jetCM,                 "jetCM[nJets]/I"  ); */
/*   chain -> Branch("jetCEF",       treeVars.jetCEF,                "jetCEF[nJets]/F" ); */
/*   chain -> Branch("jetCHF",       treeVars.jetCHF,                "jetCHF[nJets]/F" ); */
/*   chain -> Branch("jetNHF",       treeVars.jetNHF,                "jetNHF[nJets]/F" ); */
/*   chain -> Branch("jetNEF",       treeVars.jetNEF,                "jetNEF[nJets]/F" ); */
/*   chain -> Branch("jerUnc",        treeVars.jerUnc,                 "jerUnc[nJets]/F"  ); */
/*   chain -> Branch("jecUnc",       treeVars.jecUnc,                "jecUnc[nJets]/F" ); */
/*   //chain -> Branch("jecUncU",      treeVars.jecUncU,               "jecUncU[nJets]/F" ); */
/*   //chain -> Branch("jecUncD",      treeVars.jecUncD,               "jecUncD[nJets]/F" ); */
  
/*   chain -> Branch("phoPx",        treeVars.phoPx,                 "phoPx[nPhotons]/F"); */
/*   chain -> Branch("phoPy",        treeVars.phoPy,                 "phoPy[nPhotons]/F"); */
/*   chain -> Branch("phoPz",        treeVars.phoPz,                 "phoPz[nPhotons]/F"); */
/*   chain -> Branch("phoE",         treeVars.phoE,                  "phoE[nPhotons]/F"); */
/*   chain -> Branch("phoPt",         treeVars.phoPt,                  "phoPt[nPhotons]/F"); */
/*   chain -> Branch("phoDxy",         treeVars.phoDxy,                  "phoDxy[nPhotons]/F"); */
/*   chain -> Branch("phoDz",         treeVars.phoDz,                  "phoDz[nPhotons]/F"); */
/*   chain -> Branch("phoEcalIso",   treeVars.phoEcalIso,            "phoEcalIso[nPhotons]/F") ; */
/*   chain -> Branch("phoHcalIso",   treeVars.phoHcalIso,            "phoHcalIso[nPhotons]/F") ; */
/*   chain -> Branch("phoTrkIso",    treeVars.phoTrkIso,             "phoTrkIso[nPhotons]/F") ; */
/*   chain -> Branch("cHadIso",      treeVars.cHadIso,               "cHadIso[nPhotons]/F") ; */
/*   chain -> Branch("nHadIso",      treeVars.nHadIso,               "nHadIso[nPhotons]/F") ; */
/*   chain -> Branch("photIso",      treeVars.photIso,               "photIso[nPhotons]/F") ; */
/*   chain -> Branch("dR_TrkPho",    treeVars.dR_TrkPho,             "dR_TrkPho[nPhotons]/F") ; */
/*   chain -> Branch("pt_TrkPho",    treeVars.pt_TrkPho,             "pt_TrkPho[nPhotons]/F") ; */
/*   chain -> Branch("phoHoverE",    treeVars.phoHoverE,             "phoHoverE[nPhotons]/F") ; */
/*   chain -> Branch("sMinPho",      treeVars.sMinPho,               "sMinPho[nPhotons]/F") ; */
/*   chain -> Branch("sMajPho",      treeVars.sMajPho,               "sMajPho[nPhotons]/F") ; */
/*   chain -> Branch("seedTime",     treeVars.seedTime,              "seedTime[nPhotons]/F") ; */
/*   chain -> Branch("seedTimeErr",  treeVars.seedTimeErr,           "seedTimeErr[nPhotons]/F") ; */
/*   chain -> Branch("aveTime",      treeVars.aveTime,               "aveTime[nPhotons]/F") ; */
/*   chain -> Branch("aveTimeErr",   treeVars.aveTimeErr,            "aveTimeErr[nPhotons]/F") ; */
/*   chain -> Branch("aveTime1",     treeVars.aveTime1,              "aveTime1[nPhotons]/F") ; */
/*   chain -> Branch("aveTimeErr1",  treeVars.aveTimeErr1,           "aveTimeErr1[nPhotons]/F") ; */
/*   chain -> Branch("timeChi2",     treeVars.timeChi2,              "timeChi2[nPhotons]/F") ; */
/*   chain -> Branch("fSpike",       treeVars.fSpike,                "fSpike[nPhotons]/F"  ) ; */
/*   chain -> Branch("maxSwissX",    treeVars.maxSwissX,             "maxSwissX[nPhotons]/F"  ) ; */
/*   chain -> Branch("seedSwissX",   treeVars.seedSwissX,            "seedSwissX[nPhotons]/F"  ) ; */
/*   chain -> Branch("nXtals",       treeVars.nXtals,                "nXtals[nPhotons]/I"  ) ; */
/*   chain -> Branch("nBC",          treeVars.nBC,                   "nBC[nPhotons]/I"  ) ; */
/*   chain -> Branch("sigmaEta",     treeVars.sigmaEta,              "sigmaEta[nPhotons]/I"  ) ; */
/*   chain -> Branch("sigmaIeta",    treeVars.sigmaIeta,             "sigmaIeta[nPhotons]/I"  ) ; */
/*   chain -> Branch("cscRho",       treeVars.cscRho,                "cscRho[nPhotons]/F"  ) ; */
/*   chain -> Branch("cscdPhi",      treeVars.cscdPhi,               "cscdPhi[nPhotons]/F"  ) ; */
/*   chain -> Branch("dtdPhi",       treeVars.dtdPhi,                "dtdPhi[nPhotons]/F"  ) ; */

/*   chain -> Branch("dtdEta",       treeVars.dtdEta,                "dtdEta[nPhotons]/F"  ) ; */
 
/*   chain -> Branch("vtxNTracks",       treeVars.vtxNTracks,   "vtxNTracks[nVertices]/I"); */
/*   chain -> Branch("vtxChi2",          treeVars.vtxChi2,      "vtxChi2[nVertices]/F"); */
/*   chain -> Branch("vtxNdof",          treeVars.vtxNdof,      "vtxNdof[nVertices]/F"); */
/*   chain -> Branch("vtxX",             treeVars.vtxX,         "vtxX[nVertices]/F"); */
/*   chain -> Branch("vtxY",             treeVars.vtxY,         "vtxY[nVertices]/F"); */
/*   chain -> Branch("vtxZ",             treeVars.vtxZ,         "vtxZ[nVertices]/F"); */
/*   chain -> Branch("vtxDx",            treeVars.vtxDx,        "vtxDx[nVertices]/F"); */
/*   chain -> Branch("vtxDy",            treeVars.vtxDy,        "vtxDy[nVertices]/F"); */
/*   chain -> Branch("vtxDz",            treeVars.vtxDz,        "vtxDz[nVertices]/F"); */
  
/*   chain -> Branch("pdgId",        treeVars.pdgId,                 "pdgId[nGen]/I"); */
/*   chain -> Branch("momId",        treeVars.momId,                 "momId[nGen]/I"); */
/*   chain -> Branch("genPx",        treeVars.genPx,                 "genPx[nGen]/F"); */
/*   chain -> Branch("genPy",        treeVars.genPy,                 "genPy[nGen]/F"); */
/*   chain -> Branch("genPz",        treeVars.genPz,                 "genPz[nGen]/F"); */
/*   chain -> Branch("genE",         treeVars.genE,                  "genE[nGen]/F"); */
/*   chain -> Branch("genM",         treeVars.genM,                  "genM[nGen]/F"); */
/*   chain -> Branch("genVx",        treeVars.genVx,                 "genVx[nGen]/F"); */
/*   chain -> Branch("genVy",        treeVars.genVy,                 "genVy[nGen]/F"); */
/*   chain -> Branch("genVz",        treeVars.genVz,                 "genVz[nGen]/F"); */
/*   chain -> Branch("genT",         treeVars.genT,                  "genT[nGen]/F"); */
  
/* } */


/* void initializeBranches(TTree* chain, Ntuple& treeVars) */
/* { */
/* } */

/* *\/ */

#endif
