#include "GEMCode/SimMuL1/plugins/GEMCSCTriggerRate.h"

// ================================================================================================
const std::string GEMCSCTriggerRate::csc_type[CSC_TYPES+1] = 
  { "ME1/1", "ME1/2", "ME1/3", "ME1/a", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2", "ME1/T"};
const std::string GEMCSCTriggerRate::csc_type_[CSC_TYPES+1] = 
  { "ME11", "ME12", "ME13", "ME1A", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42", "ME1T"};
const std::string GEMCSCTriggerRate::csc_type_a[CSC_TYPES+2] =
  { "N/A", "ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2", "ME1/T"};
const std::string GEMCSCTriggerRate::csc_type_a_[CSC_TYPES+2] =
  { "NA", "ME1A", "ME1B", "ME12", "ME13", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42", "ME1T"};
const int GEMCSCTriggerRate::pbend[CSCConstants::NUM_CLCT_PATTERNS]= 
  { -999,  -5,  4, -4,  3, -3,  2, -2,  1, -1,  0}; // "signed" pattern (== phiBend)
const double GEMCSCTriggerRate::PT_THRESHOLDS[N_PT_THRESHOLDS] = {0,10,20,30,40,50};
const double GEMCSCTriggerRate::PT_THRESHOLDS_FOR_ETA[N_PT_THRESHOLDS] = {10,15,30,40,55,70};


// ================================================================================================
GEMCSCTriggerRate::GEMCSCTriggerRate(const edm::ParameterSet& iConfig):
  CSCTFSPset(iConfig.getParameter<edm::ParameterSet>("SectorProcessor")),
  ptLUTset(CSCTFSPset.getParameter<edm::ParameterSet>("PTLUT")),
  ptLUT(0),
  debugRATE(iConfig.getUntrackedParameter<int>("debugRATE", 0)),
  minBX_(iConfig.getUntrackedParameter<int>("minBX",-6)),
  maxBX_(iConfig.getUntrackedParameter<int>("maxBX",6)),
  minTMBBX_(iConfig.getUntrackedParameter<int>("minTMBBX",-6)),
  maxTMBBX_(iConfig.getUntrackedParameter<int>("maxTMBBX",6)),
  minRateBX_(iConfig.getUntrackedParameter<int>("minRateBX",-1)),
  maxRateBX_(iConfig.getUntrackedParameter<int>("maxRateBX",1)),
  minBxALCT_(iConfig.getUntrackedParameter<int>("minBxALCT",5)),
  maxBxALCT_(iConfig.getUntrackedParameter<int>("maxBxALCT",7)),
  minBxCLCT_(iConfig.getUntrackedParameter<int>("minBxCLCT",5)),
  maxBxCLCT_(iConfig.getUntrackedParameter<int>("maxBxCLCT",7)),
  minBxLCT_(iConfig.getUntrackedParameter<int>("minBxLCT",5)),
  maxBxLCT_(iConfig.getUntrackedParameter<int>("maxBxLCT",7)),
  minBxMPLCT_(iConfig.getUntrackedParameter<int>("minBxMPLCT",5)),
  maxBxMPLCT_(iConfig.getUntrackedParameter<int>("maxBxMPLCT",7)),
  minBxGMT_(iConfig.getUntrackedParameter<int>("minBxGMT",-1)),
  maxBxGMT_(iConfig.getUntrackedParameter<int>("maxBxGMT",1)),
  centralBxOnlyGMT_(iConfig.getUntrackedParameter< bool >("centralBxOnlyGMT",false)),
  doSelectEtaForGMTRates_(iConfig.getUntrackedParameter< bool >("doSelectEtaForGMTRates",false)),
  doME1a_(iConfig.getUntrackedParameter< bool >("doME1a",false)),
  // special treatment of matching in ME1a for the case of the default emulator
  defaultME1a(iConfig.getUntrackedParameter<bool>("defaultME1a", false))
{
  edm::ParameterSet srLUTset = CSCTFSPset.getParameter<edm::ParameterSet>("SRLUT");

  for(int e=0; e<2; e++) 
    for (int s=0; s<6; s++) 
      my_SPs[e][s] = NULL;
  
  bool TMB07 = true;
  for(int endcap = 1; endcap<=2; endcap++)
  {
    for(int sector=1; sector<=6; sector++)
    {
      for(int station=1,fpga=0; station<=4 && fpga<5; station++)
      {
	if(station==1) for(int subSector=0; subSector<2; subSector++)
	  srLUTs_[fpga++][sector-1][endcap-1] = new CSCSectorReceiverLUT(endcap, sector, subSector+1, station, srLUTset, TMB07);
	else
	  srLUTs_[fpga++][sector-1][endcap-1] = new CSCSectorReceiverLUT(endcap, sector, 0, station, srLUTset, TMB07);
      }
    }
  }

  my_dtrc = new CSCTFDTReceiver();

  // cache flags for event setup records
  muScalesCacheID_ = 0ULL ;
  muPtScaleCacheID_ = 0ULL ;

  bookALCTTree();
  bookCLCTTree();
  bookLCTTree();
  bookMPCLCTTree();
  bookTFTrackTree();
  bookTFCandTree();
  bookGMTRegionalTree();
  bookGMTCandTree();
}

// ================================================================================================
GEMCSCTriggerRate::~GEMCSCTriggerRate()
{
  if(ptLUT) delete ptLUT;
  ptLUT = NULL;

  for(int e=0; e<2; e++) for (int s=0; s<6; s++){
      if  (my_SPs[e][s]) delete my_SPs[e][s];
      my_SPs[e][s] = NULL;

      for(int fpga=0; fpga<5; fpga++)
	{
	  if (srLUTs_[fpga][s][e]) delete srLUTs_[fpga][s][e];
	  srLUTs_[fpga][s][e] = NULL;
	}
    }
  
  if(my_dtrc) delete my_dtrc;
  my_dtrc = NULL;
}

// ================================================================================================
void
GEMCSCTriggerRate::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup)
{
  edm::ESHandle< CSCGeometry > cscGeom;
  iSetup.get<MuonGeometryRecord>().get(cscGeom);
  cscGeometry = &*cscGeom;
  CSCTriggerGeometry::setGeometry(cscGeometry);
}

// ================================================================================================
void 
GEMCSCTriggerRate::beginJob()
{
  edm::Service<TFileService> fs;

  Double_t ETA_BIN = 0.0125 *2;
  //Double_t PHI_BIN = 62.*M_PI/180./4096.; // 0.26 mrad
  int N_ETA_BINS=200;
  double ETA_START=-2.4999;
  double ETA_END = ETA_START + ETA_BIN*N_ETA_BINS;
  
  int   N_ETA_BINS_CSC = 32;
  double ETA_START_CSC = 0.9;
  double ETA_END_CSC   = 2.5;

  int   N_ETA_BINS_DT = 32;
  double ETA_START_DT = 0.;
  double ETA_END_DT   = 1.2;

  const int N_ETA_BINS_RPC = 17;
  double ETA_BINS_RPC[N_ETA_BINS_RPC+1] =
  {0, 0.07, 0.27, 0.44, 0.58, 0.72, 0.83, 0.93, 1.04, 1.14,
   1.24, 1.36, 1.48, 1.61, 1.73, 1.85, 1.97, 2.1};

  const int N_ETA_BINS_GMT = 32;
  double ETA_BINS_GMT[N_ETA_BINS_GMT+1] =
  {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
   1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.75, 1.8,
   1.85, 1.9, 1.95, 2, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3,
   2.35, 2.4, 2.45};
  
  h_rt_nalct = fs->make<TH1D>("h_rt_nalct","h_rt_nalct",101,-0.5, 100.5);
  h_rt_nclct = fs->make<TH1D>("h_rt_nclct","h_rt_nclct",101,-0.5, 100.5);
  h_rt_nlct = fs->make<TH1D>("h_rt_nlct","h_rt_nlct",101,-0.5, 100.5);
  h_rt_nmplct = fs->make<TH1D>("h_rt_nmplct","h_rt_nmplct",101,-0.5, 100.5);
  h_rt_ntftrack = fs->make<TH1D>("h_rt_ntftrack","h_rt_ntftrack",31,-0.5, 30.5);
  h_rt_ntfcand = fs->make<TH1D>("h_rt_ntfcand","h_rt_ntfcand",31,-0.5, 30.5);
  h_rt_ntfcand_pt10 = fs->make<TH1D>("h_rt_ntfcand_pt10","h_rt_ntfcand_pt10",31,-0.5, 30.5);
  h_rt_ngmt_csc = fs->make<TH1D>("h_rt_ngmt_csc","h_rt_ngmt_csc",11,-0.5, 10.5);
  h_rt_ngmt_csc_pt10 = fs->make<TH1D>("h_rt_ngmt_csc_pt10","h_rt_ngmt_csc_pt10",11,-0.5, 10.5);
  h_rt_ngmt_csc_per_bx = fs->make<TH1D>("h_rt_ngmt_csc_per_bx","h_rt_ngmt_csc_per_bx",11,-0.5, 10.5);
  h_rt_ngmt_rpcf = fs->make<TH1D>("h_rt_ngmt_rpcf","h_rt_ngmt_rpcf",11,-0.5, 10.5);
  h_rt_ngmt_rpcf_pt10 = fs->make<TH1D>("h_rt_ngmt_rpcf_pt10","h_rt_ngmt_rpcf_pt10",11,-0.5, 10.5);
  h_rt_ngmt_rpcf_per_bx = fs->make<TH1D>("h_rt_ngmt_rpcf_per_bx","h_rt_ngmt_rpcf_per_bx",11,-0.5, 10.5);
  h_rt_ngmt_rpcb = fs->make<TH1D>("h_rt_ngmt_rpcb","h_rt_ngmt_rpcb",11,-0.5, 10.5);
  h_rt_ngmt_rpcb_pt10 = fs->make<TH1D>("h_rt_ngmt_rpcb_pt10","h_rt_ngmt_rpcb_pt10",11,-0.5, 10.5);
  h_rt_ngmt_rpcb_per_bx = fs->make<TH1D>("h_rt_ngmt_rpcb_per_bx","h_rt_ngmt_rpcb_per_bx",11,-0.5, 10.5);
  h_rt_ngmt_dt = fs->make<TH1D>("h_rt_ngmt_dt","h_rt_ngmt_dt",11,-0.5, 10.5);
  h_rt_ngmt_dt_pt10 = fs->make<TH1D>("h_rt_ngmt_dt_pt10","h_rt_ngmt_dt_pt10",11,-0.5, 10.5);
  h_rt_ngmt_dt_per_bx = fs->make<TH1D>("h_rt_ngmt_dt_per_bx","h_rt_ngmt_dt_per_bx",11,-0.5, 10.5);
  h_rt_ngmt = fs->make<TH1D>("h_rt_ngmt","h_rt_ngmt",11,-0.5, 10.5);
  h_rt_nxtra = fs->make<TH1D>("h_rt_nxtra","h_rt_nxtra",11,-0.5, 10.5);

  h_rt_nalct_per_bx = fs->make<TH1D>("h_rt_nalct_per_bx", "h_rt_nalct_per_bx", 51,-0.5, 50.5);
  h_rt_nclct_per_bx = fs->make<TH1D>("h_rt_nclct_per_bx", "h_rt_nclct_per_bx", 51,-0.5, 50.5);
  h_rt_nlct_per_bx = fs->make<TH1D>("h_rt_nlct_per_bx", "h_rt_nlct_per_bx", 51,-0.5, 50.5);

  h_rt_alct_bx = fs->make<TH1D>("h_rt_alct_bx","h_rt_alct_bx",13,-6.5, 6.5);
  h_rt_clct_bx = fs->make<TH1D>("h_rt_clct_bx","h_rt_clct_bx",13,-6.5, 6.5);
  h_rt_lct_bx = fs->make<TH1D>("h_rt_lct_bx","h_rt_lct_bx",13,-6.5, 6.5);
  h_rt_mplct_bx = fs->make<TH1D>("h_rt_mplct_bx","h_rt_mplct_bx",13,-6.5, 6.5);

  h_rt_csctype_alct_bx567 = fs->make<TH1D>("h_rt_csctype_alct_bx567", "CSC type vs ALCT rate", 10, 0.5,  10.5);
  h_rt_csctype_clct_bx567 = fs->make<TH1D>("h_rt_csctype_clct_bx567", "CSC type vs CLCT rate", 10, 0.5,  10.5);
  h_rt_csctype_lct_bx567 = fs->make<TH1D>("h_rt_csctype_lct_bx567", "CSC type vs LCT rate", 10, 0.5,  10.5);
  h_rt_csctype_mplct_bx567 = fs->make<TH1D>("h_rt_csctype_mplct_bx567", "CSC type vs MPC LCT rate", 10, 0.5,  10.5);
  for (int i=1; i<=CSC_TYPES;i++) {
    h_rt_csctype_alct_bx567->GetXaxis()->SetBinLabel(i,csc_type_a[i].c_str());
    h_rt_csctype_clct_bx567->GetXaxis()->SetBinLabel(i,csc_type_a[i].c_str());
    h_rt_csctype_lct_bx567->GetXaxis()->SetBinLabel(i,csc_type_a[i].c_str());
    h_rt_csctype_mplct_bx567->GetXaxis()->SetBinLabel(i,csc_type_a[i].c_str());
  }
  
  h_rt_lct_qu_vs_bx = fs->make<TH2D>("h_rt_lct_qu_vs_bx","h_rt_lct_qu_vs_bx",20,0., 20.,13,-6.5, 6.5);
  h_rt_mplct_qu_vs_bx = fs->make<TH2D>("h_rt_mplct_qu_vs_bx","h_rt_mplct_qu_vs_bx",20,0., 20.,13,-6.5, 6.5);

  h_rt_nalct_vs_bx = fs->make<TH2D>("h_rt_nalct_vs_bx","h_rt_nalct_vs_bx",20,0., 20.,16,-.5, 15.5);
  h_rt_nclct_vs_bx = fs->make<TH2D>("h_rt_nclct_vs_bx","h_rt_nclct_vs_bx",20,0., 20.,16,-.5, 15.5);
  h_rt_nlct_vs_bx = fs->make<TH2D>("h_rt_nlct_vs_bx","h_rt_nlct_vs_bx",20,0., 20.,16,-.5, 15.5);
  h_rt_nmplct_vs_bx = fs->make<TH2D>("h_rt_nmplct_vs_bx","h_rt_nmplct_vs_bx",20,0., 20.,16,-.5, 15.5);

  h_rt_lct_qu = fs->make<TH1D>("h_rt_lct_qu","h_rt_lct_qu",20,0., 20.);
  h_rt_mplct_qu = fs->make<TH1D>("h_rt_mplct_qu","h_rt_mplct_qu",20,0., 20.);

  h_rt_qu_vs_bxclct__lct = fs->make<TH2D>("h_rt_qu_vs_bxclct__lct","h_rt_qu_vs_bxclct__lct",17,-0.5, 16.5, 15,-7.5, 7.5);

  h_rt_tftrack_pt = fs->make<TH1D>("h_rt_tftrack_pt","h_rt_tftrack_pt",600, 0.,150.);
  h_rt_tfcand_pt = fs->make<TH1D>("h_rt_tfcand_pt","h_rt_tfcand_pt",600, 0.,150.);


  h_rt_gmt_csc_pt = fs->make<TH1D>("h_rt_gmt_csc_pt","h_rt_gmt_csc_pt",600, 0.,150.);
  h_rt_gmt_csc_pt_2st = fs->make<TH1D>("h_rt_gmt_csc_pt_2st","h_rt_gmt_csc_pt_2st",600, 0.,150.);
  h_rt_gmt_csc_pt_3st = fs->make<TH1D>("h_rt_gmt_csc_pt_3st","h_rt_gmt_csc_pt_3st",600, 0.,150.);
  h_rt_gmt_csc_pt_2q = fs->make<TH1D>("h_rt_gmt_csc_pt_2q","h_rt_gmt_csc_pt_2q",600, 0.,150.);
  h_rt_gmt_csc_pt_3q = fs->make<TH1D>("h_rt_gmt_csc_pt_3q","h_rt_gmt_csc_pt_3q",600, 0.,150.);
  h_rt_gmt_csc_ptmax_2s = fs->make<TH1D>("h_rt_gmt_csc_ptmax_2s","h_rt_gmt_csc_ptmax_2s",600, 0.,150.);
  h_rt_gmt_csc_ptmax_2s_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax_2s_1b","h_rt_gmt_csc_ptmax_2s_1b",600, 0.,150.);
  h_rt_gmt_csc_ptmax_2s_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax_2s_no1a","h_rt_gmt_csc_ptmax_2s_no1a",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s","h_rt_gmt_csc_ptmax_3s",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_1b","h_rt_gmt_csc_ptmax_3s_1b",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_no1a","h_rt_gmt_csc_ptmax_3s_no1a",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_2s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_2s1b","h_rt_gmt_csc_ptmax_3s_2s1b",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_2s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_2s1b_1b","h_rt_gmt_csc_ptmax_3s_2s1b_1b",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_2s123_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_2s123_1b","h_rt_gmt_csc_ptmax_3s_2s123_1b",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_2s13_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_2s13_1b","h_rt_gmt_csc_ptmax_3s_2s13_1b",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_2s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_2s1b_no1a","h_rt_gmt_csc_ptmax_3s_2s1b_no1a",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_2s123_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_2s123_no1a","h_rt_gmt_csc_ptmax_3s_2s123_no1a",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_2s13_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_2s13_no1a","h_rt_gmt_csc_ptmax_3s_2s13_no1a",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_3s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_3s1b","h_rt_gmt_csc_ptmax_3s_3s1b",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_3s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_3s1b_1b","h_rt_gmt_csc_ptmax_3s_3s1b_1b",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s_3s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s_3s1b_no1a","h_rt_gmt_csc_ptmax_3s_3s1b_no1a",600, 0.,150.);
  h_rt_gmt_csc_ptmax_2q = fs->make<TH1D>("h_rt_gmt_csc_ptmax_2q","h_rt_gmt_csc_ptmax_2q",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3q = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3q","h_rt_gmt_csc_ptmax_3q",600, 0.,150.);
  h_rt_gmt_csc_pt_2s42 = fs->make<TH1D>("h_rt_gmt_csc_pt_2s42","h_rt_gmt_csc_pt_2s42",600, 0.,150.);
  h_rt_gmt_csc_pt_3s42 = fs->make<TH1D>("h_rt_gmt_csc_pt_3s42","h_rt_gmt_csc_pt_3s42",600, 0.,150.);
  h_rt_gmt_csc_ptmax_2s42 = fs->make<TH1D>("h_rt_gmt_csc_ptmax_2s42","h_rt_gmt_csc_ptmax_2s42",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s42 = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s42","h_rt_gmt_csc_ptmax_3s42",600, 0.,150.);
  h_rt_gmt_csc_pt_2q42 = fs->make<TH1D>("h_rt_gmt_csc_pt_2q42","h_rt_gmt_csc_pt_2q42",600, 0.,150.);
  h_rt_gmt_csc_pt_3q42 = fs->make<TH1D>("h_rt_gmt_csc_pt_3q42","h_rt_gmt_csc_pt_3q42",600, 0.,150.);
  h_rt_gmt_csc_ptmax_2q42 = fs->make<TH1D>("h_rt_gmt_csc_ptmax_2q42","h_rt_gmt_csc_ptmax_2q42",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3q42 = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3q42","h_rt_gmt_csc_ptmax_3q42",600, 0.,150.);
  h_rt_gmt_csc_pt_2s42r = fs->make<TH1D>("h_rt_gmt_csc_pt_2s42r","h_rt_gmt_csc_pt_2s42r",600, 0.,150.);
  h_rt_gmt_csc_pt_3s42r = fs->make<TH1D>("h_rt_gmt_csc_pt_3s42r","h_rt_gmt_csc_pt_3s42r",600, 0.,150.);
  h_rt_gmt_csc_ptmax_2s42r = fs->make<TH1D>("h_rt_gmt_csc_ptmax_2s42r","h_rt_gmt_csc_ptmax_2s42r",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3s42r = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3s42r","h_rt_gmt_csc_ptmax_3s42r",600, 0.,150.);
  h_rt_gmt_csc_pt_2q42r = fs->make<TH1D>("h_rt_gmt_csc_pt_2q42r","h_rt_gmt_csc_pt_2q42r",600, 0.,150.);
  h_rt_gmt_csc_pt_3q42r = fs->make<TH1D>("h_rt_gmt_csc_pt_3q42r","h_rt_gmt_csc_pt_3q42r",600, 0.,150.);
  h_rt_gmt_csc_ptmax_2q42r = fs->make<TH1D>("h_rt_gmt_csc_ptmax_2q42r","h_rt_gmt_csc_ptmax_2q42r",600, 0.,150.);
  h_rt_gmt_csc_ptmax_3q42r = fs->make<TH1D>("h_rt_gmt_csc_ptmax_3q42r","h_rt_gmt_csc_ptmax_3q42r",600, 0.,150.);
 

  h_rt_gmt_rpcf_pt = fs->make<TH1D>("h_rt_gmt_rpcf_pt","h_rt_gmt_rpcf_pt",600, 0.,150.);
  h_rt_gmt_rpcf_pt_42 = fs->make<TH1D>("h_rt_gmt_rpcf_pt_42","h_rt_gmt_rpcf_pt_42",600, 0.,150.);
  h_rt_gmt_rpcf_ptmax = fs->make<TH1D>("h_rt_gmt_rpcf_ptmax","h_rt_gmt_rpcf_ptmax",600, 0.,150.);
  h_rt_gmt_rpcf_ptmax_42 = fs->make<TH1D>("h_rt_gmt_rpcf_ptmax_42","h_rt_gmt_rpcf_ptmax_42",600, 0.,150.);

  h_rt_gmt_rpcb_pt = fs->make<TH1D>("h_rt_gmt_rpcb_pt","h_rt_gmt_rpcb_pt",600, 0.,150.);
  h_rt_gmt_rpcb_ptmax = fs->make<TH1D>("h_rt_gmt_rpcb_ptmax","h_rt_gmt_rpcb_ptmax",600, 0.,150.);

  h_rt_gmt_dt_pt = fs->make<TH1D>("h_rt_gmt_dt_pt","h_rt_gmt_dt_pt",600, 0.,150.);
  h_rt_gmt_dt_ptmax = fs->make<TH1D>("h_rt_gmt_dt_ptmax","h_rt_gmt_dt_ptmax",600, 0.,150.);

  h_rt_gmt_pt = fs->make<TH1D>("h_rt_gmt_pt","h_rt_gmt_pt",600, 0.,150.);
  h_rt_gmt_pt_2st = fs->make<TH1D>("h_rt_gmt_pt_2st","h_rt_gmt_pt_2st",600, 0.,150.);
  h_rt_gmt_pt_3st = fs->make<TH1D>("h_rt_gmt_pt_3st","h_rt_gmt_pt_3st",600, 0.,150.);
  h_rt_gmt_pt_2q = fs->make<TH1D>("h_rt_gmt_pt_2q","h_rt_gmt_pt_2q",600, 0.,150.);
  h_rt_gmt_pt_3q = fs->make<TH1D>("h_rt_gmt_pt_3q","h_rt_gmt_pt_3q",600, 0.,150.);
  h_rt_gmt_ptmax = fs->make<TH1D>("h_rt_gmt_ptmax","h_rt_gmt_ptmax",600, 0.,150.);
  h_rt_gmt_ptmax_sing = fs->make<TH1D>("h_rt_gmt_ptmax_sing","h_rt_gmt_ptmax_sing",600, 0.,150.);
  h_rt_gmt_ptmax_sing_3s = fs->make<TH1D>("h_rt_gmt_ptmax_sing_3s","h_rt_gmt_ptmax_sing_3s",600, 0.,150.);
  h_rt_gmt_ptmax_sing_csc = fs->make<TH1D>("h_rt_gmt_ptmax_sing_csc","h_rt_gmt_ptmax_sing_csc",600, 0.,150.);
  h_rt_gmt_ptmax_sing_1b = fs->make<TH1D>("h_rt_gmt_ptmax_sing_1b","h_rt_gmt_ptmax_sing_no1a",600, 0.,150.);
  h_rt_gmt_ptmax_sing_no1a = fs->make<TH1D>("h_rt_gmt_ptmax_sing_no1a","h_rt_gmt_ptmax_sing_no1a",600, 0.,150.);
  h_rt_gmt_ptmax_sing6 = fs->make<TH1D>("h_rt_gmt_ptmax_sing6","h_rt_gmt_ptmax_sing6",600, 0.,150.);
  h_rt_gmt_ptmax_sing6_3s = fs->make<TH1D>("h_rt_gmt_ptmax_sing6_3s","h_rt_gmt_ptmax_sing6_3s",600, 0.,150.);
  h_rt_gmt_ptmax_sing6_csc = fs->make<TH1D>("h_rt_gmt_ptmax_sing6_csc","h_rt_gmt_ptmax_sing6_csc",600, 0.,150.);
  h_rt_gmt_ptmax_sing6_1b = fs->make<TH1D>("h_rt_gmt_ptmax_sing6_1b","h_rt_gmt_ptmax_sing6_1b",600, 0.,150.);
  h_rt_gmt_ptmax_sing6_no1a = fs->make<TH1D>("h_rt_gmt_ptmax_sing6_no1a","h_rt_gmt_ptmax_sing6_no1a",600, 0.,150.);
  h_rt_gmt_ptmax_sing6_3s1b_no1a = fs->make<TH1D>("h_rt_gmt_ptmax_sing6_3s1b_no1a","h_rt_gmt_ptmax_sing6_3s1b_no1a",600, 0.,150.);
  h_rt_gmt_ptmax_dbl = fs->make<TH1D>("h_rt_gmt_ptmax_dbl","h_rt_gmt_ptmax_dbl",600, 0.,150.);
  h_rt_gmt_pt_2s42 = fs->make<TH1D>("h_rt_gmt_pt_2s42","h_rt_gmt_pt_2s42",600, 0.,150.);
  h_rt_gmt_pt_3s42 = fs->make<TH1D>("h_rt_gmt_pt_3s42","h_rt_gmt_pt_3s42",600, 0.,150.);
  h_rt_gmt_ptmax_2s42 = fs->make<TH1D>("h_rt_gmt_ptmax_2s42","h_rt_gmt_ptmax_2s42",600, 0.,150.);
  h_rt_gmt_ptmax_3s42 = fs->make<TH1D>("h_rt_gmt_ptmax_3s42","h_rt_gmt_ptmax_3s42",600, 0.,150.);
  h_rt_gmt_ptmax_2s42_sing = fs->make<TH1D>("h_rt_gmt_ptmax_2s42_sing","h_rt_gmt_ptmax_2s42_sing",600, 0.,150.);
  h_rt_gmt_ptmax_3s42_sing = fs->make<TH1D>("h_rt_gmt_ptmax_3s42_sing","h_rt_gmt_ptmax_3s42_sing",600, 0.,150.);
  h_rt_gmt_pt_2q42 = fs->make<TH1D>("h_rt_gmt_pt_2q42","h_rt_gmt_pt_2q42",600, 0.,150.);
  h_rt_gmt_pt_3q42 = fs->make<TH1D>("h_rt_gmt_pt_3q42","h_rt_gmt_pt_3q42",600, 0.,150.);
  h_rt_gmt_ptmax_2q42 = fs->make<TH1D>("h_rt_gmt_ptmax_2q42","h_rt_gmt_ptmax_2q42",600, 0.,150.);
  h_rt_gmt_ptmax_3q42 = fs->make<TH1D>("h_rt_gmt_ptmax_3q42","h_rt_gmt_ptmax_3q42",600, 0.,150.);
  h_rt_gmt_ptmax_2q42_sing = fs->make<TH1D>("h_rt_gmt_ptmax_2q42_sing","h_rt_gmt_ptmax_2q42_sing",600, 0.,150.);
  h_rt_gmt_ptmax_3q42_sing = fs->make<TH1D>("h_rt_gmt_ptmax_3q42_sing","h_rt_gmt_ptmax_3q42_sing",600, 0.,150.);
  h_rt_gmt_pt_2s42r = fs->make<TH1D>("h_rt_gmt_pt_2s42r","h_rt_gmt_pt_2s42r",600, 0.,150.);
  h_rt_gmt_pt_3s42r = fs->make<TH1D>("h_rt_gmt_pt_3s42r","h_rt_gmt_pt_3s42r",600, 0.,150.);
  h_rt_gmt_ptmax_2s42r = fs->make<TH1D>("h_rt_gmt_ptmax_2s42r","h_rt_gmt_ptmax_2s42r",600, 0.,150.);
  h_rt_gmt_ptmax_3s42r = fs->make<TH1D>("h_rt_gmt_ptmax_3s42r","h_rt_gmt_ptmax_3s42r",600, 0.,150.);
  h_rt_gmt_ptmax_2s42r_sing = fs->make<TH1D>("h_rt_gmt_ptmax_2s42r_sing","h_rt_gmt_ptmax_2s42r_sing",600, 0.,150.);
  h_rt_gmt_ptmax_3s42r_sing = fs->make<TH1D>("h_rt_gmt_ptmax_3s42r_sing","h_rt_gmt_ptmax_3s42r_sing",600, 0.,150.);
  h_rt_gmt_pt_2q42r = fs->make<TH1D>("h_rt_gmt_pt_2q42r","h_rt_gmt_pt_2q42r",600, 0.,150.);
  h_rt_gmt_pt_3q42r = fs->make<TH1D>("h_rt_gmt_pt_3q42r","h_rt_gmt_pt_3q42r",600, 0.,150.);
  h_rt_gmt_ptmax_2q42r = fs->make<TH1D>("h_rt_gmt_ptmax_2q42r","h_rt_gmt_ptmax_2q42r",600, 0.,150.);
  h_rt_gmt_ptmax_3q42r = fs->make<TH1D>("h_rt_gmt_ptmax_3q42r","h_rt_gmt_ptmax_3q42r",600, 0.,150.);
  h_rt_gmt_ptmax_2q42r_sing = fs->make<TH1D>("h_rt_gmt_ptmax_2q42r_sing","h_rt_gmt_ptmax_2q42r_sing",600, 0.,150.);
  h_rt_gmt_ptmax_3q42r_sing = fs->make<TH1D>("h_rt_gmt_ptmax_3q42r_sing","h_rt_gmt_ptmax_3q42r_sing",600, 0.,150.);


  h_rt_gmt_csc_eta = fs->make<TH1D>("h_rt_gmt_csc_eta","h_rt_gmt_csc_eta",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_2s = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_2s","h_rt_gmt_csc_ptmax10_eta_2s",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_2s_2s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_2s_2s1b","h_rt_gmt_csc_ptmax10_eta_2s_2s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s","h_rt_gmt_csc_ptmax10_eta_3s",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_1b","h_rt_gmt_csc_ptmax10_eta_3s_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_no1a","h_rt_gmt_csc_ptmax10_eta_3s_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_2s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_2s1b","h_rt_gmt_csc_ptmax10_eta_3s_2s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_2s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_2s1b_1b","h_rt_gmt_csc_ptmax10_eta_3s_2s1b_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_2s123_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_2s123_1b","h_rt_gmt_csc_ptmax10_eta_3s_2s123_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_2s13_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_2s13_1b","h_rt_gmt_csc_ptmax10_eta_3s_2s13_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_2s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_2s1b_no1a","h_rt_gmt_csc_ptmax10_eta_3s_2s1b_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_2s123_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_2s123_no1a","h_rt_gmt_csc_ptmax10_eta_3s_2s123_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_2s13_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_2s13_no1a","h_rt_gmt_csc_ptmax10_eta_3s_2s13_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_3s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_3s1b","h_rt_gmt_csc_ptmax10_eta_3s_3s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_3s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_3s1b_1b","h_rt_gmt_csc_ptmax10_eta_3s_3s1b_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3s_3s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3s_3s1b_no1a","h_rt_gmt_csc_ptmax10_eta_3s_3s1b_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_2q = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_2q","h_rt_gmt_csc_ptmax10_eta_2q",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax10_eta_3q = fs->make<TH1D>("h_rt_gmt_csc_ptmax10_eta_3q","h_rt_gmt_csc_ptmax10_eta_3q",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_csc_ptmax20_eta_2s = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_2s","h_rt_gmt_csc_ptmax20_eta_2s",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_2s_2s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_2s_2s1b","h_rt_gmt_csc_ptmax20_eta_2s_2s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s","h_rt_gmt_csc_ptmax20_eta_3s",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_csc_ptmax20_eta_3s_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_1b","h_rt_gmt_csc_ptmax20_eta_3s_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_1ab = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_1ab","h_rt_gmt_csc_ptmax20_eta_3s_1ab",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_csc_ptmax20_eta_3s_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_no1a","h_rt_gmt_csc_ptmax20_eta_3s_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s1b","h_rt_gmt_csc_ptmax20_eta_3s_2s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s1b_1b","h_rt_gmt_csc_ptmax20_eta_3s_2s1b_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s123_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s123_1b","h_rt_gmt_csc_ptmax20_eta_3s_2s123_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s13_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s13_1b","h_rt_gmt_csc_ptmax20_eta_3s_2s13_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s1b_no1a","h_rt_gmt_csc_ptmax20_eta_3s_2s1b_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s123_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s123_no1a","h_rt_gmt_csc_ptmax20_eta_3s_2s123_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s13_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s13_no1a","h_rt_gmt_csc_ptmax20_eta_3s_2s13_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_csc_ptmax20_eta_3s_3s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_3s1b","h_rt_gmt_csc_ptmax20_eta_3s_3s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_3s1ab = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_3s1ab","h_rt_gmt_csc_ptmax20_eta_3s_3s1ab",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_csc_ptmax20_eta_3s_3s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_3s1b_1b","h_rt_gmt_csc_ptmax20_eta_3s_3s1b_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_3s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_3s1b_no1a","h_rt_gmt_csc_ptmax20_eta_3s_3s1b_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_2q = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_2q","h_rt_gmt_csc_ptmax20_eta_2q",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3q = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3q","h_rt_gmt_csc_ptmax20_eta_3q",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_csc_ptmax30_eta_2s = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_2s","h_rt_gmt_csc_ptmax30_eta_2s",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_2s_2s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_2s_2s1b","h_rt_gmt_csc_ptmax30_eta_2s_2s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s","h_rt_gmt_csc_ptmax30_eta_3s",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_csc_ptmax30_eta_3s_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_1b","h_rt_gmt_csc_ptmax30_eta_3s_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_1ab = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_1ab","h_rt_gmt_csc_ptmax30_eta_3s_1ab",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_csc_ptmax30_eta_3s_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_no1a","h_rt_gmt_csc_ptmax30_eta_3s_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s1b","h_rt_gmt_csc_ptmax30_eta_3s_2s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s1b_1b","h_rt_gmt_csc_ptmax30_eta_3s_2s1b_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s123_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s123_1b","h_rt_gmt_csc_ptmax30_eta_3s_2s123_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s13_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s13_1b","h_rt_gmt_csc_ptmax30_eta_3s_2s13_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s1b_no1a","h_rt_gmt_csc_ptmax30_eta_3s_2s1b_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s123_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s123_no1a","h_rt_gmt_csc_ptmax30_eta_3s_2s123_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s13_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s13_no1a","h_rt_gmt_csc_ptmax30_eta_3s_2s13_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_csc_ptmax30_eta_3s_3s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_3s1b","h_rt_gmt_csc_ptmax30_eta_3s_3s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_3s1ab = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_3s1ab","h_rt_gmt_csc_ptmax30_eta_3s_3s1ab",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);


  h_rt_gmt_csc_ptmax30_eta_3s_3s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_3s1b_1b","h_rt_gmt_csc_ptmax30_eta_3s_3s1b_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_3s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_3s1b_no1a","h_rt_gmt_csc_ptmax30_eta_3s_3s1b_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_2q = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_2q","h_rt_gmt_csc_ptmax30_eta_2q",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3q = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3q","h_rt_gmt_csc_ptmax30_eta_3q",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_rpcf_eta = fs->make<TH1D>("h_rt_gmt_rpcf_eta","h_rt_gmt_rpcf_eta",N_ETA_BINS_RPC, ETA_BINS_RPC);
  h_rt_gmt_rpcf_ptmax10_eta = fs->make<TH1D>("h_rt_gmt_rpcf_ptmax10_eta","h_rt_gmt_rpcf_ptmax10_eta",N_ETA_BINS_RPC, ETA_BINS_RPC);
  h_rt_gmt_rpcf_ptmax20_eta = fs->make<TH1D>("h_rt_gmt_rpcf_ptmax20_eta","h_rt_gmt_rpcf_ptmax20_eta",N_ETA_BINS_RPC, ETA_BINS_RPC);
  h_rt_gmt_rpcb_eta = fs->make<TH1D>("h_rt_gmt_rpcb_eta","h_rt_gmt_rpcb_eta",N_ETA_BINS_RPC, ETA_BINS_RPC);
  h_rt_gmt_rpcb_ptmax10_eta = fs->make<TH1D>("h_rt_gmt_rpcb_ptmax10_eta","h_rt_gmt_rpcb_ptmax10_eta",N_ETA_BINS_RPC, ETA_BINS_RPC);
  h_rt_gmt_rpcb_ptmax20_eta = fs->make<TH1D>("h_rt_gmt_rpcb_ptmax20_eta","h_rt_gmt_rpcb_ptmax20_eta",N_ETA_BINS_RPC, ETA_BINS_RPC);
  h_rt_gmt_dt_eta = fs->make<TH1D>("h_rt_gmt_dt_eta","h_rt_gmt_dt_eta",N_ETA_BINS_DT, ETA_START_DT, ETA_END_DT);
  h_rt_gmt_dt_ptmax10_eta = fs->make<TH1D>("h_rt_gmt_dt_ptmax10_eta","h_rt_gmt_dt_ptmax10_eta",N_ETA_BINS_DT, ETA_START_DT, ETA_END_DT);
  h_rt_gmt_dt_ptmax20_eta = fs->make<TH1D>("h_rt_gmt_dt_ptmax20_eta","h_rt_gmt_dt_ptmax20_eta",N_ETA_BINS_DT, ETA_START_DT, ETA_END_DT);
  h_rt_gmt_eta = fs->make<TH1D>("h_rt_gmt_eta","h_rt_gmt_eta",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax10_eta = fs->make<TH1D>("h_rt_gmt_ptmax10_eta","h_rt_gmt_ptmax10_eta",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax10_eta_sing = fs->make<TH1D>("h_rt_gmt_ptmax10_eta_sing","h_rt_gmt_ptmax10_eta_sing",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax10_eta_sing_3s = fs->make<TH1D>("h_rt_gmt_ptmax10_eta_sing_3s","h_rt_gmt_ptmax10_eta_sing_3s",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax10_eta_sing6 = fs->make<TH1D>("h_rt_gmt_ptmax10_eta_sing6","h_rt_gmt_ptmax10_eta_sing6",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax10_eta_sing6_3s = fs->make<TH1D>("h_rt_gmt_ptmax10_eta_sing6_3s","h_rt_gmt_ptmax10_eta_sing6_3s",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax20_eta = fs->make<TH1D>("h_rt_gmt_ptmax20_eta","h_rt_gmt_ptmax20_eta",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax20_eta_sing = fs->make<TH1D>("h_rt_gmt_ptmax20_eta_sing","h_rt_gmt_ptmax20_eta_sing",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax20_eta_sing_csc = fs->make<TH1D>("h_rt_gmt_ptmax20_eta_sing_csc","h_rt_gmt_ptmax20_eta_sing_csc",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax20_eta_sing_dtcsc = fs->make<TH1D>("h_rt_gmt_ptmax20_eta_sing_dtcsc","h_rt_gmt_ptmax20_eta_sing_dtcsc",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax20_eta_sing_3s = fs->make<TH1D>("h_rt_gmt_ptmax20_eta_sing_3s","h_rt_gmt_ptmax20_eta_sing_3s",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax20_eta_sing6 = fs->make<TH1D>("h_rt_gmt_ptmax20_eta_sing6","h_rt_gmt_ptmax20_eta_sing6",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax20_eta_sing6_csc = fs->make<TH1D>("h_rt_gmt_ptmax20_eta_sing6_csc","h_rt_gmt_ptmax20_eta_sing6_csc",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax20_eta_sing6_3s = fs->make<TH1D>("h_rt_gmt_ptmax20_eta_sing6_3s","h_rt_gmt_ptmax20_eta_sing6_3s",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax30_eta_sing = fs->make<TH1D>("h_rt_gmt_ptmax30_eta_sing","h_rt_gmt_ptmax30_eta_sing",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax30_eta_sing_csc = fs->make<TH1D>("h_rt_gmt_ptmax30_eta_sing_csc","h_rt_gmt_ptmax30_eta_sing_csc",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax30_eta_sing_dtcsc = fs->make<TH1D>("h_rt_gmt_ptmax30_eta_sing_dtcsc","h_rt_gmt_ptmax30_eta_sing_dtcsc",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax30_eta_sing_3s = fs->make<TH1D>("h_rt_gmt_ptmax30_eta_sing_3s","h_rt_gmt_ptmax30_eta_sing_3s",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax30_eta_sing6 = fs->make<TH1D>("h_rt_gmt_ptmax30_eta_sing6","h_rt_gmt_ptmax30_eta_sing6",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax30_eta_sing6_csc = fs->make<TH1D>("h_rt_gmt_ptmax30_eta_sing6_csc","h_rt_gmt_ptmax30_eta_sing6_csc",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax30_eta_sing6_3s = fs->make<TH1D>("h_rt_gmt_ptmax30_eta_sing6_3s","h_rt_gmt_ptmax30_eta_sing6_3s",N_ETA_BINS_GMT, ETA_BINS_GMT);

  h_rt_gmt_ptmax10_eta_dbl = fs->make<TH1D>("h_rt_gmt_ptmax10_eta_dbl","h_rt_gmt_ptmax10_eta_dbl",N_ETA_BINS_GMT, ETA_BINS_GMT);
  h_rt_gmt_ptmax20_eta_dbl = fs->make<TH1D>("h_rt_gmt_ptmax20_eta_dbl","h_rt_gmt_ptmax20_eta_dbl",N_ETA_BINS_GMT, ETA_BINS_GMT);

  const int Nthr = 7;
  std::string str_pts[Nthr] = {"", "_pt10", "_pt15", "_pt20", "_pt25", "_pt30","_pt40"};

  for (int i = 1; i < Nthr; ++i) {
    std::string prefix = "h_rt_gmt_csc_mode_2s1b_1b_";
    h_rt_gmt_csc_mode_2s1b_1b[i-1] = fs->make<TH1D>((prefix + str_pts[i]).c_str(), (prefix + str_pts[i]).c_str(), 16, -0.5, 15.5);
    setupTFModeHisto(h_rt_gmt_csc_mode_2s1b_1b[i-1]);
  }

  h_rt_tfcand_pt_2st = fs->make<TH1D>("h_rt_tfcand_pt_2st","h_rt_tfcand_pt_2st",600, 0.,150.);
  h_rt_tfcand_pt_3st = fs->make<TH1D>("h_rt_tfcand_pt_3st","h_rt_tfcand_pt_3st",600, 0.,150.);

  h_rt_tfcand_pt_h42_2st = fs->make<TH1D>("h_rt_tfcand_pt_h42_2st","h_rt_tfcand_pt_h42_2st",600, 0.,150.);
  h_rt_tfcand_pt_h42_3st = fs->make<TH1D>("h_rt_tfcand_pt_h42_3st","h_rt_tfcand_pt_h42_3st",600, 0.,150.);

  h_rt_tftrack_bx = fs->make<TH1D>("h_rt_tftrack_bx","h_rt_tftrack_bx",13,-6.5, 6.5);
  h_rt_tfcand_bx = fs->make<TH1D>("h_rt_tfcand_bx","h_rt_tfcand_bx",13,-6.5, 6.5);
  h_rt_gmt_csc_bx = fs->make<TH1D>("h_rt_gmt_csc_bx","h_rt_gmt_csc_bx",13,-6.5, 6.5);
  h_rt_gmt_rpcf_bx = fs->make<TH1D>("h_rt_gmt_rpcf_bx","h_rt_gmt_rpcf_bx",13,-6.5, 6.5);
  h_rt_gmt_rpcb_bx = fs->make<TH1D>("h_rt_gmt_rpcb_bx","h_rt_gmt_rpcb_bx",13,-6.5, 6.5);
  h_rt_gmt_dt_bx = fs->make<TH1D>("h_rt_gmt_dt_bx","h_rt_gmt_dt_bx",13,-6.5, 6.5);
  h_rt_gmt_bx = fs->make<TH1D>("h_rt_gmt_bx","h_rt_gmt_bx",13,-6.5, 6.5);

  h_rt_gmt_csc_q = fs->make<TH1D>("h_rt_gmt_csc_q","h_rt_gmt_csc_q",8,-.5, 7.5);
  h_rt_gmt_csc_q_42 = fs->make<TH1D>("h_rt_gmt_csc_q_42","h_rt_gmt_csc_q_42",8,-.5, 7.5);
  h_rt_gmt_csc_q_42r = fs->make<TH1D>("h_rt_gmt_csc_q_42r","h_rt_gmt_csc_q_42r",8,-.5, 7.5);
  h_rt_gmt_rpcf_q = fs->make<TH1D>("h_rt_gmt_rpcf_q","h_rt_gmt_rpcf_q",8,-.5, 7.5);
  h_rt_gmt_rpcf_q_42 = fs->make<TH1D>("h_rt_gmt_rpcf_q_42","h_rt_gmt_rpcf_q_42",8,-.5, 7.5);
  h_rt_gmt_rpcb_q = fs->make<TH1D>("h_rt_gmt_rpcb_q","h_rt_gmt_rpcb_q",8,-.5, 7.5);
  h_rt_gmt_dt_q = fs->make<TH1D>("h_rt_gmt_dt_q","h_rt_gmt_dt_q",8,-.5, 7.5);
  h_rt_gmt_gq = fs->make<TH1D>("h_rt_gmt_gq","h_rt_gmt_gq",8,-.5, 7.5);
  h_rt_gmt_gq_42 = fs->make<TH1D>("h_rt_gmt_gq_42","h_rt_gmt_gq_42",8,-.5, 7.5);
  h_rt_gmt_gq_42r = fs->make<TH1D>("h_rt_gmt_gq_42","h_rt_gmt_gq_42r",8,-.5, 7.5);
  h_rt_gmt_gq_vs_pt_42r = fs->make<TH2D>("h_rt_gmt_gq_vs_pt_42r","h_rt_gmt_gq_vs_pt_42r",8,-.5, 7.5, 600, 0.,150.);
  h_rt_gmt_gq_vs_type_42r = fs->make<TH2D>("h_rt_gmt_gq_vs_type_42r","h_rt_gmt_gq_vs_type_42r",8,-.5, 7.5, 7,-0.5,6.5);
  h_rt_gmt_gq_vs_type_42r->GetYaxis()->SetBinLabel(1,"?");
  h_rt_gmt_gq_vs_type_42r->GetYaxis()->SetBinLabel(2,"RPC q=0");
  h_rt_gmt_gq_vs_type_42r->GetYaxis()->SetBinLabel(3,"RPC q=1");
  h_rt_gmt_gq_vs_type_42r->GetYaxis()->SetBinLabel(4,"CSC q=1");
  h_rt_gmt_gq_vs_type_42r->GetYaxis()->SetBinLabel(5,"CSC q=2");
  h_rt_gmt_gq_vs_type_42r->GetYaxis()->SetBinLabel(6,"CSC q=3");
  h_rt_gmt_gq_vs_type_42r->GetYaxis()->SetBinLabel(7,"matched");

  h_rt_tftrack_mode = fs->make<TH1D>("h_rt_tftrack_mode","TF Track Mode", 16, -0.5, 15.5);
  setupTFModeHisto(h_rt_tftrack_mode);
  h_rt_tftrack_mode->SetTitle("TF Track Mode (all TF tracks)");
  
  h_rt_n_ch_alct_per_bx = fs->make<TH1D>("h_rt_n_ch_alct_per_bx", "h_rt_n_ch_alct_per_bx", 51,-0.5, 50.5);
  h_rt_n_ch_clct_per_bx = fs->make<TH1D>("h_rt_n_ch_clct_per_bx", "h_rt_n_ch_clct_per_bx", 51,-0.5, 50.5);
  h_rt_n_ch_lct_per_bx = fs->make<TH1D>("h_rt_n_ch_lct_per_bx", "h_rt_n_ch_lct_per_bx", 51,-0.5, 50.5);


  h_rt_tfcand_eta = fs->make<TH1D>("h_rt_tfcand_eta","h_rt_tfcand_eta",N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_eta_pt5 = fs->make<TH1D>("h_rt_tfcand_eta_pt5","h_rt_tfcand_eta_pt5",N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_eta_pt10 = fs->make<TH1D>("h_rt_tfcand_eta_pt10","h_rt_tfcand_eta_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_eta_pt15 = fs->make<TH1D>("h_rt_tfcand_eta_pt15","h_rt_tfcand_eta_pt15",N_ETA_BINS, ETA_START, ETA_END);

  h_rt_tfcand_eta_3st = fs->make<TH1D>("h_rt_tfcand_eta_3st","h_rt_tfcand_eta_3st",N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_eta_pt5_3st = fs->make<TH1D>("h_rt_tfcand_eta_pt5_3st","h_rt_tfcand_eta_pt5_3st",N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_eta_pt10_3st = fs->make<TH1D>("h_rt_tfcand_eta_pt10_3st","h_rt_tfcand_eta_pt10_3st",N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_eta_pt15_3st = fs->make<TH1D>("h_rt_tfcand_eta_pt15_3st","h_rt_tfcand_eta_pt15_3st",N_ETA_BINS, ETA_START, ETA_END);

  h_rt_tfcand_eta_3st1a = fs->make<TH1D>("h_rt_tfcand_eta_3st1a","h_rt_tfcand_eta_3st1a",N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_eta_pt5_3st1a = fs->make<TH1D>("h_rt_tfcand_eta_pt5_3st1a","h_rt_tfcand_eta_pt5_3st1a",N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_eta_pt10_3st1a = fs->make<TH1D>("h_rt_tfcand_eta_pt10_3st1a","h_rt_tfcand_eta_pt10_3st1a",N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_eta_pt15_3st1a = fs->make<TH1D>("h_rt_tfcand_eta_pt15_3st1a","h_rt_tfcand_eta_pt15_3st1a",N_ETA_BINS, ETA_START, ETA_END);

  h_rt_tfcand_pt_vs_eta = fs->make<TH2D>("h_rt_tfcand_pt_vs_eta","h_rt_tfcand_pt_vs_eta",600, 0.,150.,N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_pt_vs_eta_3st = fs->make<TH2D>("h_rt_tfcand_pt_vs_eta_3st","h_rt_tfcand_pt_vs_eta_3st",600, 0.,150.,N_ETA_BINS, ETA_START, ETA_END);
  h_rt_tfcand_pt_vs_eta_3st1a = fs->make<TH2D>("h_rt_tfcand_pt_vs_eta_3st1a","h_rt_tfcand_pt_vs_eta_3st1a",600, 0.,150.,N_ETA_BINS, ETA_START, ETA_END);

  char label[200];
  for (int me=0; me<=CSC_TYPES; me++) 
  {
    if (me==3 && !doME1a_) continue; // ME1/a

    sprintf(label,"h_rt_n_per_ch_alct_vs_bx_cscdet_%s",csc_type_[me].c_str());
    h_rt_n_per_ch_alct_vs_bx_cscdet[me] = fs->make<TH2D>(label, label, 5,0,5, 16,-.5, 15.5);

    sprintf(label,"h_rt_n_per_ch_clct_vs_bx_cscdet_%s",csc_type_[me].c_str());
    h_rt_n_per_ch_clct_vs_bx_cscdet[me] = fs->make<TH2D>(label, label, 5,0,5, 16,-.5, 15.5);

    sprintf(label,"h_rt_n_per_ch_lct_vs_bx_cscdet_%s",csc_type_[me].c_str());
    h_rt_n_per_ch_lct_vs_bx_cscdet[me] = fs->make<TH2D>(label, label, 5,0,5, 16,-.5, 15.5);


    sprintf(label,"h_rt_n_ch_alct_per_bx_cscdet_%s",csc_type_[me].c_str());
    h_rt_n_ch_alct_per_bx_cscdet[me] = fs->make<TH1D>(label, label, 51,-0.5, 50.5);

    sprintf(label,"h_rt_n_ch_clct_per_bx_cscdet_%s",csc_type_[me].c_str());
    h_rt_n_ch_clct_per_bx_cscdet[me] = fs->make<TH1D>(label, label, 51,-0.5, 50.5);

    sprintf(label,"h_rt_n_ch_lct_per_bx_cscdet_%s",csc_type_[me].c_str());
    h_rt_n_ch_lct_per_bx_cscdet[me] = fs->make<TH1D>(label, label, 51,-0.5, 50.5);


    sprintf(label,"h_rt_mplct_pattern_cscdet_%s",csc_type_[me].c_str());
    h_rt_mplct_pattern_cscdet[me] = fs->make<TH1D>(label, label, 13,-0.5, 12.5);


    sprintf(label,"h_rt_alct_bx_cscdet_%s",csc_type_[me].c_str());
    h_rt_alct_bx_cscdet[me] = fs->make<TH1D>(label, label,13,-6.5, 6.5);
    sprintf(label,"h_rt_clct_bx_cscdet_%s",csc_type_[me].c_str());
    h_rt_clct_bx_cscdet[me] = fs->make<TH1D>(label, label,13,-6.5, 6.5);
    sprintf(label,"h_rt_lct_bx_cscdet_%s",csc_type_[me].c_str());
    h_rt_lct_bx_cscdet[me] = fs->make<TH1D>(label, label,13,-6.5, 6.5);
    sprintf(label,"h_rt_mplct_bx_cscdet_%s",csc_type_[me].c_str());
    h_rt_mplct_bx_cscdet[me] = fs->make<TH1D>(label, label,13,-6.5, 6.5);

  }//for (int me=0; me<CSC_TYPES; me++) 

  h_rt_lct_per_sector = fs->make<TH1D>("h_rt_lct_per_sector","h_rt_lct_per_sector",20,0., 20.);
  h_rt_lct_per_sector_vs_bx = fs->make<TH2D>("h_rt_lct_per_sector_vs_bx","h_rt_lct_per_sector_vs_bx",20,0., 20.,16,0,16);
  h_rt_mplct_per_sector = fs->make<TH1D>("h_rt_mplct_per_sector","h_rt_mplct_per_sector",20,0., 20.);
  h_rt_mplct_per_sector_vs_bx = fs->make<TH2D>("h_rt_mplct_per_sector_vs_bx","h_rt_mplct_per_sector_vs_bx",20,0., 20.,16,0,16);
  h_rt_lct_per_sector_vs_bx_st1t = fs->make<TH2D>("h_rt_lct_per_sector_vs_bx_st1t","h_rt_lct_per_sector_vs_bx_st1t",20,0., 20.,16,0,16);
  h_rt_mplct_per_sector_vs_bx_st1t = fs->make<TH2D>("h_rt_mplct_per_sector_vs_bx_st1t","h_rt_mplct_per_sector_vs_bx_st1t",20,0., 20.,16,0,16);

  for (int i=0; i<MAX_STATIONS;i++) 
  {
    sprintf(label,"h_rt_n_ch_alct_per_bx_st%d",i+1);
    h_rt_n_ch_alct_per_bx_st[i] = fs->make<TH1D>(label, label, 51,-0.5, 50.5);

    sprintf(label,"h_rt_n_ch_clct_per_bx_st%d",i+1);
    h_rt_n_ch_clct_per_bx_st[i] = fs->make<TH1D>(label, label, 51,-0.5, 50.5);

    sprintf(label,"h_rt_n_ch_lct_per_bx_st%d",i+1);
    h_rt_n_ch_lct_per_bx_st[i] = fs->make<TH1D>(label, label, 51,-0.5, 50.5);


    sprintf(label,"h_rt_lct_per_sector_st%d",i+1);
    h_rt_lct_per_sector_st[i]  = fs->make<TH1D>(label, label, 20,0., 20.);

    sprintf(label,"h_rt_lct_per_sector_vs_bx_st%d",i+1);
    h_rt_lct_per_sector_vs_bx_st[i]  = fs->make<TH2D>(label, label, 20,0., 20.,16,0,16);

    sprintf(label,"h_rt_mplct_per_sector_st%d",i+1);
    h_rt_mplct_per_sector_st[i]  = fs->make<TH1D>(label, label, 20,0., 20.);

    sprintf(label,"h_rt_mplct_per_sector_vs_bx_st%d",i+1);
    h_rt_mplct_per_sector_vs_bx_st[i]  = fs->make<TH2D>(label, label, 20,0., 20.,16,0,16);

  }

  h_rt_mplct_pattern = fs->make<TH1D>("h_rt_mplct_pattern","h_rt_mplct_pattern",13,-0.5, 12.5);


  h_gmt_mindr = fs->make<TH1D>("h_gmt_mindr","h_gmt_mindr",500, 0, 2*M_PI);
  h_gmt_dr_maxrank = fs->make<TH1D>("h_gmt_dr_maxrank","h_gmt_dr_maxrank",500, 0, 2*M_PI);
}


// ================================================================================================
void 
GEMCSCTriggerRate::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // ALCTs and CLCTs
  edm::Handle< CSCCLCTDigiCollection > hclcts;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  hclcts);
  const CSCCLCTDigiCollection* clcts = hclcts.product();

  // strip&wire matching output  after TMB  and after MPC sorting
  edm::Handle< CSCCorrelatedLCTDigiCollection > lcts_tmb;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  lcts_tmb);
  const CSCCorrelatedLCTDigiCollection* lcts = lcts_tmb.product();

  edm::Handle< CSCCorrelatedLCTDigiCollection > lcts_mpc;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis", "MPCSORTED", lcts_mpc);
  const CSCCorrelatedLCTDigiCollection* mplcts = lcts_mpc.product();
  
  // DT primitives for input to TF
  edm::Handle<L1MuDTChambPhContainer> dttrig;
  iEvent.getByLabel("simDtTriggerPrimitiveDigis", dttrig);
  const L1MuDTChambPhContainer* dttrigs = dttrig.product();

  // tracks produced by TF
  edm::Handle< L1CSCTrackCollection > hl1Tracks;
  iEvent.getByLabel("simCsctfTrackDigis",hl1Tracks);
  const L1CSCTrackCollection* l1Tracks = hl1Tracks.product();

  // L1 muon candidates after CSC sorter
  edm::Handle< std::vector< L1MuRegionalCand > > hl1TfCands;
  iEvent.getByLabel("simCsctfDigis", "CSC", hl1TfCands);
  const std::vector< L1MuRegionalCand > *l1TfCands = hl1TfCands.product();

  // GMT readout collection
  edm::Handle< L1MuGMTReadoutCollection > hl1GmtCands;
  iEvent.getByLabel("simGmtDigis", hl1GmtCands ) ;// InputTag("simCsctfDigis","CSC")

  //const L1MuGMTReadoutCollection* l1GmtCands = hl1GmtCands.product();
  std::vector<L1MuGMTExtendedCand> l1GmtCands;
  std::vector<L1MuGMTExtendedCand> l1GmtfCands;
  std::vector<L1MuRegionalCand>    l1GmtCSCCands;
  std::vector<L1MuRegionalCand>    l1GmtRPCfCands;
  std::vector<L1MuRegionalCand>    l1GmtRPCbCands;
  std::vector<L1MuRegionalCand>    l1GmtDTCands;

  // key = BX
  std::map<int, std::vector<L1MuRegionalCand> >  l1GmtCSCCandsInBXs;

  // TOCHECK
  if ( centralBxOnlyGMT_ )
  {
    // Get GMT candidates from central bunch crossing only
    l1GmtCands = hl1GmtCands->getRecord().getGMTCands() ;
    l1GmtfCands = hl1GmtCands->getRecord().getGMTFwdCands() ;
    l1GmtCSCCands = hl1GmtCands->getRecord().getCSCCands() ;
    l1GmtRPCfCands = hl1GmtCands->getRecord().getFwdRPCCands() ;
    l1GmtRPCbCands = hl1GmtCands->getRecord().getBrlRPCCands() ;
    l1GmtDTCands = hl1GmtCands->getRecord().getDTBXCands() ;
    l1GmtCSCCandsInBXs[hl1GmtCands->getRecord().getBxInEvent()] = l1GmtCSCCands;
  }
  else
  {
    // Get GMT candidates from all bunch crossings
    std::vector<L1MuGMTReadoutRecord> gmt_records = hl1GmtCands->getRecords();
    for ( std::vector< L1MuGMTReadoutRecord >::const_iterator rItr=gmt_records.begin(); rItr!=gmt_records.end() ; ++rItr )
      {
	if (rItr->getBxInEvent() < minBxGMT_ || rItr->getBxInEvent() > maxBxGMT_) continue;
	
	std::vector<L1MuGMTExtendedCand> GMTCands = rItr->getGMTCands();
	for ( std::vector<L1MuGMTExtendedCand>::const_iterator  cItr = GMTCands.begin() ; cItr != GMTCands.end() ; ++cItr )
	  if (!cItr->empty()) l1GmtCands.push_back(*cItr);
	
	std::vector<L1MuGMTExtendedCand> GMTfCands = rItr->getGMTFwdCands();
	for ( std::vector<L1MuGMTExtendedCand>::const_iterator  cItr = GMTfCands.begin() ; cItr != GMTfCands.end() ; ++cItr )
	  if (!cItr->empty()) l1GmtfCands.push_back(*cItr);
	
	//std::cout<<" ggg: "<<GMTCands.size()<<" "<<GMTfCands.size()<<std::endl;
	
	std::vector<L1MuRegionalCand> CSCCands = rItr->getCSCCands();
	l1GmtCSCCandsInBXs[rItr->getBxInEvent()] = CSCCands;
	for ( std::vector<L1MuRegionalCand>::const_iterator  cItr = CSCCands.begin() ; cItr != CSCCands.end() ; ++cItr )
	  if (!cItr->empty()) l1GmtCSCCands.push_back(*cItr);
	
	std::vector<L1MuRegionalCand> RPCfCands = rItr->getFwdRPCCands();
	for ( std::vector<L1MuRegionalCand>::const_iterator  cItr = RPCfCands.begin() ; cItr != RPCfCands.end() ; ++cItr )
	  if (!cItr->empty()) l1GmtRPCfCands.push_back(*cItr);
	
	std::vector<L1MuRegionalCand> RPCbCands = rItr->getBrlRPCCands();
	for ( std::vector<L1MuRegionalCand>::const_iterator  cItr = RPCbCands.begin() ; cItr != RPCbCands.end() ; ++cItr )
	  if (!cItr->empty()) l1GmtRPCbCands.push_back(*cItr);
	
	std::vector<L1MuRegionalCand> DTCands = rItr->getDTBXCands();
	for ( std::vector<L1MuRegionalCand>::const_iterator  cItr = DTCands.begin() ; cItr != DTCands.end() ; ++cItr )
	  if (!cItr->empty()) l1GmtDTCands.push_back(*cItr);
      }
    //std::cout<<" sizes: "<<l1GmtCands.size()<<" "<<l1GmtfCands.size()<<" "<<l1GmtCSCCands.size()<<" "<<l1GmtRPCfCands.size()<<std::endl;
  }
  
  // does the trigger sccale need to be defined in the beginrun or analyze method?
  if (iSetup.get< L1MuTriggerScalesRcd >().cacheIdentifier() != muScalesCacheID_ ||
      iSetup.get< L1MuTriggerPtScaleRcd >().cacheIdentifier() != muPtScaleCacheID_ )
    {
      iSetup.get< L1MuTriggerScalesRcd >().get( muScales );

      iSetup.get< L1MuTriggerPtScaleRcd >().get( muPtScale );

      if (ptLUT) delete ptLUT;  
      ptLUT = new CSCTFPtLUT(ptLUTset, muScales.product(), muPtScale.product());
  
      for(int e=0; e<2; e++) for (int s=0; s<6; s++){
  	  if  (my_SPs[e][s]) delete my_SPs[e][s];
  	  my_SPs[e][s] = new CSCTFSectorProcessor(e+1, s+1, CSCTFSPset, true, muScales.product(), muPtScale.product());
  	  my_SPs[e][s]->initialize(iSetup);
  	}
      muScalesCacheID_  = iSetup.get< L1MuTriggerScalesRcd >().cacheIdentifier();
      muPtScaleCacheID_ = iSetup.get< L1MuTriggerPtScaleRcd >().cacheIdentifier();
    }

  // //=======================================================================
  // //============================= RATES ===================================
  analyzeALCTRate(iEvent);
  analyzeCLCTRate(iEvent);
  analyzeLCTRate(iEvent);
  analyzeMPCLCTRate(iEvent);
  analyzeTFTrackRate(iEvent);
  analyzeTFCandRate(iEvent);
  analyzeGMTCandRate(iEvent);


  //============ RATE ALCT ==================


  //============ RATE CLCT ==================

  std::map<int, std::vector<CSCCLCTDigi> > detCLCT;
  detCLCT.clear();
  int nclct=0;
  int nclct_per_bx[16];
  int n_ch_clct_per_bx[16];
  int n_ch_clct_per_bx_st[MAX_STATIONS][16];
  int n_ch_clct_per_bx_cscdet[CSC_TYPES+1][16];
  for (int b=0;b<16;b++)
    {
      nclct_per_bx[b] = n_ch_clct_per_bx[b] = 0;
      for (int s=0; s<MAX_STATIONS; s++) n_ch_clct_per_bx_st[s][b]=0;
      for (int me=0; me<=CSC_TYPES; me++) n_ch_clct_per_bx_cscdet[me][b]=0;
    }
  if (debugRATE) std::cout<< "----- statring nclct"<<std::endl;
  for (CSCCLCTDigiCollection::DigiRangeIterator  cdetUnitIt = clcts->begin(); cdetUnitIt != clcts->end(); cdetUnitIt++)
    {
      const CSCDetId& id = (*cdetUnitIt).first;
      //if (id.endcap() != 1) continue;
      CSCDetId idd(id.rawId());
      int csct = getCSCType( idd );
      int cscst = getCSCSpecsType( idd );
      int nclct_per_ch_bx[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};    
      const CSCCLCTDigiCollection::Range& range = (*cdetUnitIt).second;
      for (CSCCLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
  	{
  	  if ((*digiIt).isValid()) 
  	    {
  	      //        detCLCT[id.rawId()].push_back(*digiIt);
  	      int bx = (*digiIt).getBX();
  	      //if ( bx-5 < minBX_ || bx-7 > maxBX_ )
  	      if ( bx < minBxCLCT_ || bx > maxBxCLCT_ )
  		{
  		  if (debugRATE) std::cout<<"discarding BX = "<< bx-6 <<std::endl;
  		  continue;
  		}
  	      //if (debugCLCT) std::cout<<"raw ID "<<id.rawId()<<" "<<id<<"    NTrackHitsInChamber  nmhits  clctInfo.size  diff  " 
  	      //                   <<trackHitsInChamber.size()<<" "<<nmhits<<" "<<clctInfo.size()<<"  "
  	      //                   << nmhits-clctInfo.size() <<std::endl 
  	      //                   << "  "<<(*digiIt)<<std::endl;
  	      nclct++;
  	      ++nclct_per_bx[bx];
  	      ++nclct_per_ch_bx[bx];
  	      h_rt_clct_bx->Fill( bx - 6 );
  	      h_rt_clct_bx_cscdet[csct]->Fill( bx - 6 );
  	      if (bx>=5 && bx<=7) h_rt_csctype_clct_bx567->Fill(cscst);
  	    } //if (clct_valid) 
  	}
      for (int b=0;b<16;b++) 
  	{
  	  if ( b < minBxALCT_ || b > maxBxALCT_ ) continue;
  	  h_rt_n_per_ch_clct_vs_bx_cscdet[csct]->Fill(nclct_per_ch_bx[b],b);
  	  if (nclct_per_ch_bx[b]>0) {
  	    ++n_ch_clct_per_bx[b];
  	    ++n_ch_clct_per_bx_st[id.station()-1][b];
  	    ++n_ch_clct_per_bx_cscdet[csct][b];
  	  }
  	}
    } // loop CSCCLCTDigiCollection
  h_rt_nclct->Fill(nclct);
  for (int b=0;b<16;b++) {
    if (b < minBxALCT_ || b > maxBxALCT_) continue;
    h_rt_nclct_vs_bx->Fill(nclct_per_bx[b],b);
    h_rt_nclct_per_bx->Fill(nclct_per_bx[b]);
    h_rt_n_ch_clct_per_bx->Fill(n_ch_clct_per_bx[b]);
    for (int s=0; s<MAX_STATIONS; s++) 
      h_rt_n_ch_clct_per_bx_st[s]->Fill(n_ch_clct_per_bx_st[s][b]);
    for (int me=0; me<=CSC_TYPES; me++) 
      h_rt_n_ch_clct_per_bx_cscdet[me]->Fill(n_ch_clct_per_bx_cscdet[me][b]);
  }
  if (debugRATE) std::cout<< "----- end nclct="<<nclct<<std::endl;


  //============ RATE LCT ==================

  int nlct=0;
  int nlct_per_bx[16];
  int n_ch_lct_per_bx[16];
  int n_ch_lct_per_bx_st[MAX_STATIONS][16];
  int n_ch_lct_per_bx_cscdet[CSC_TYPES+1][16];
  for (int b=0;b<16;b++)
    {
      nlct_per_bx[b] = n_ch_lct_per_bx[b] = 0;
      for (int s=0; s<MAX_STATIONS; s++) n_ch_lct_per_bx_st[s][b]=0;
      for (int me=0; me<=CSC_TYPES; me++) n_ch_lct_per_bx_cscdet[me][b]=0;
    }
  int nlct_sector_st[MAX_STATIONS][13], nlct_sector_bx_st[MAX_STATIONS][13][16], nlct_trigsector_bx_st1[13][16];
  for (int s=0; s<MAX_STATIONS; s++) for (int i=0; i<13; i++) {
      nlct_sector_st[s][i]=0;
      for (int j=0; j<16; j++) { nlct_sector_bx_st[s][i][j]=0; nlct_trigsector_bx_st1[i][j]=0; }
    }
  std::map< int , std::vector<const CSCCorrelatedLCTDigi*> > me11lcts;
  if (debugRATE) std::cout<< "----- statring nlct"<<std::endl;
  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt = lcts->begin(); detUnitIt != lcts->end(); detUnitIt++) 
    {
      const CSCDetId& id = (*detUnitIt).first;
      //if (id.endcap() != 1) continue;
      CSCDetId idd(id.rawId());
      int csct = getCSCType( idd );
      int cscst = getCSCSpecsType( idd );
      CSCDetId id11(id.rawId());
      if (csct==3) id11=CSCDetId(id11.endcap(),1,1,id11.chamber());
      int nlct_per_ch_bx[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
      const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
      for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
  	{
  	  if ((*digiIt).isValid()) 
  	    {
  	      int bx = (*digiIt).getBX();
  	      //if (debugLCT) std::cout<< "----- LCT in raw ID "<<id.rawId()<<" "<<id<< " (trig id. " << id.triggerCscId() << ")"<<std::endl;
  	      //if (debugLCT) std::cout<< " "<< (*digiIt);
  	      //if ( bx-6 < minBX_ || bx-6 > maxBX_ )
  	      if ( bx < minBxLCT_ || bx > maxBxLCT_ )
  		{
  		  if (debugRATE) std::cout<<"discarding BX = "<< bx-6 <<std::endl;
  		  continue;
  		}

  	      // store all ME11 lcts together so we can look at them later
  	      if (csct==0 || csct==3) me11lcts[id11.rawId()].push_back(&(*digiIt));

  	      int sect = id.triggerSector();
  	      if (id.station()==1) sect = (sect-1)*2 + cscTriggerSubsector(idd);
  	      nlct_sector_st[id.station()-1][sect] += 1;
  	      nlct_sector_bx_st[id.station()-1][sect][bx] += 1;
  	      if (id.station()==1) nlct_trigsector_bx_st1[id.triggerSector()][bx] += 1;

  	      int quality = (*digiIt).getQuality();

  	      //bool alct_valid = (quality != 4 && quality != 5);
  	      //bool clct_valid = (quality != 1 && quality != 3);
  	      //bool alct_valid = (quality != 2);
  	      //bool clct_valid = (quality != 1);

  	      //if (alct_valid || clct_valid) 
  	      nlct++;
  	      ++nlct_per_bx[bx];
  	      ++nlct_per_ch_bx[bx];
  	      h_rt_lct_bx->Fill( bx - 6 );
  	      h_rt_lct_bx_cscdet[csct]->Fill( bx - 6 );
  	      if (bx>=5 && bx<=7) h_rt_csctype_lct_bx567->Fill(cscst);

  	      h_rt_lct_qu_vs_bx->Fill( quality, bx - 6);
  	      h_rt_lct_qu->Fill( quality );

  	      std::map<int, std::vector<CSCCLCTDigi> >::const_iterator mapItr = detCLCT.find(id.rawId());
  	      if(mapItr != detCLCT.end())
  		for ( unsigned i=0; i<mapItr->second.size(); i++ )
  		  if( (*digiIt).getStrip() == ((mapItr->second)[i]).getKeyStrip() &&
  		      (*digiIt).getPattern() == ((mapItr->second)[i]).getPattern()  )
  		    {
  		      h_rt_qu_vs_bxclct__lct->Fill(quality, ((mapItr->second)[i]).getBX() - 6 );
  		    }
  	    }
  	}
      for (int b=0;b<16;b++) 
  	{
  	  if ( b < minBxALCT_ || b > maxBxALCT_ ) continue;
  	  h_rt_n_per_ch_lct_vs_bx_cscdet[csct]->Fill(nlct_per_ch_bx[b],b);
  	  if (nlct_per_ch_bx[b]>0) {
  	    ++n_ch_lct_per_bx[b];
  	    ++n_ch_lct_per_bx_st[id.station()-1][b];
  	    ++n_ch_lct_per_bx_cscdet[csct][b];
  	  }
  	}
    }
  std::map< int , std::vector<const CSCCorrelatedLCTDigi*> >::const_iterator mapIt = me11lcts.begin();
  for (;mapIt != me11lcts.end(); mapIt++)
    {
      CSCDetId id(mapIt->first);
      int nlct_per_ch_bx[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
      for (size_t i=0; i<(mapIt->second).size(); i++)
  	{
  	  int bx = (mapIt->second)[i]->getBX();
  	  ++nlct_per_ch_bx[bx];
  	}
      for (int b=0;b<16;b++)
  	{
  	  if ( b < minBxALCT_ || b > maxBxALCT_ ) continue;
  	  h_rt_n_per_ch_lct_vs_bx_cscdet[10]->Fill(nlct_per_ch_bx[b],b);
  	  if (nlct_per_ch_bx[b]>0) ++n_ch_lct_per_bx_cscdet[10][b];
  	}
    }
  h_rt_nlct->Fill(nlct);
  for (int b=0;b<16;b++) {
    if (b < minBxALCT_ || b > maxBxALCT_) continue;
    h_rt_nlct_vs_bx->Fill(nlct_per_bx[b],b);
    h_rt_nlct_per_bx->Fill(nlct_per_bx[b]);
    h_rt_n_ch_lct_per_bx->Fill(n_ch_lct_per_bx[b]);
    for (int s=0; s<MAX_STATIONS; s++) 
      h_rt_n_ch_lct_per_bx_st[s]->Fill(n_ch_lct_per_bx_st[s][b]);
    for (int me=0; me<=CSC_TYPES; me++) 
      h_rt_n_ch_lct_per_bx_cscdet[me]->Fill(n_ch_lct_per_bx_cscdet[me][b]);
  }
  for (int s=0; s<MAX_STATIONS; s++)  
    for (int i=1; i<=12; i++)
      {
	if (s!=0 && i>6) continue; // only ME1 has 12 subsectors
	h_rt_lct_per_sector->Fill(nlct_sector_st[s][i]);
	h_rt_lct_per_sector_st[s]->Fill(nlct_sector_st[s][i]);
	for (int j=0; j<16; j++) {
	  if ( j < minBxALCT_ || j > maxBxALCT_ ) continue;
	  h_rt_lct_per_sector_vs_bx->Fill(nlct_sector_bx_st[s][i][j],j+0.5);
	  h_rt_lct_per_sector_vs_bx_st[s]->Fill(nlct_sector_bx_st[s][i][j],j+0.5);
	  if (s==0 && i<7) h_rt_lct_per_sector_vs_bx_st1t->Fill(nlct_trigsector_bx_st1[i][j],j+0.5);
	}
      }
  if (debugRATE) std::cout<< "----- end nlct="<<nlct<<std::endl;


  //============ RATE MPC LCT ==================

  int nmplct=0;
  int nmplct_per_bx[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int nmplct_sector_st[MAX_STATIONS][13], nmplct_sector_bx_st[MAX_STATIONS][13][16], nmplct_trigsector_bx_st1[13][16];
  for (int s=0; s<MAX_STATIONS; s++) for (int i=0; i<13; i++) {
      nmplct_sector_st[s][i]=0;
      for (int j=0; j<16; j++) { nmplct_sector_bx_st[s][i][j]=0; nmplct_trigsector_bx_st1[i][j]=0; }
    }

  if (debugRATE) std::cout<< "----- statring nmplct"<<std::endl;
  std::vector<MatchCSCMuL1::MPLCT> rtMPLCTs;
  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt = mplcts->begin();  detUnitIt != mplcts->end(); detUnitIt++) 
    {
      const CSCDetId& id = (*detUnitIt).first;
      //if ( id.endcap() != 1) continue;
      CSCDetId idd(id.rawId());
      int csct = getCSCType( idd );
      int cscst = getCSCSpecsType( idd );
      const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
      for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
  	{
  	  if ((*digiIt).isValid()) 
  	    {
  	      //if (debugRATE) std::cout<< "----- MPLCT in raw ID "<<id.rawId()<<" "<<id<< " (trig id. " << id.triggerCscId() << ")"<<std::endl;
  	      //if (debugRATE) std::cout<<" "<< (*digiIt);
  	      int bx = (*digiIt).getBX();
  	      //if ( bx-6 < minBX_ || bx-6 > maxBX_ )
  	      if ( bx < minBxMPLCT_ || bx > maxBxMPLCT_ )
  		{
  		  if (debugRATE) std::cout<<"discarding BX = "<< (*digiIt).getBX()-6 <<std::endl;
  		  continue;
  		}

  	      int sect = id.triggerSector();
  	      if (id.station()==1) sect = (sect-1)*2 + cscTriggerSubsector(idd);
  	      nmplct_sector_st[id.station()-1][sect] += 1;
  	      nmplct_sector_bx_st[id.station()-1][sect][bx] += 1;
  	      if (id.station()==1) nmplct_trigsector_bx_st1[id.triggerSector()][bx] += 1;

  	      int quality = (*digiIt).getQuality();

  	      //bool alct_valid = (quality != 4 && quality != 5);
  	      //bool clct_valid = (quality != 1 && quality != 3);
  	      //bool alct_valid = (quality != 2);
  	      //bool clct_valid = (quality != 1);

  	      // Truly correlated LCTs; for DAQ
  	      //if (alct_valid && clct_valid)
  	      nmplct++;
  	      ++nmplct_per_bx[bx];
  	      h_rt_mplct_bx->Fill( bx - 6 );
  	      h_rt_mplct_bx_cscdet[csct]->Fill( bx - 6 );
  	      if (bx>=5 && bx<=7) h_rt_csctype_mplct_bx567->Fill(cscst);

  	      h_rt_mplct_qu_vs_bx->Fill( quality, bx - 6);
  	      h_rt_mplct_qu->Fill( quality );


  	      h_rt_mplct_pattern->Fill( (*digiIt).getPattern() );
  	      h_rt_mplct_pattern_cscdet[csct]->Fill( (*digiIt).getPattern() );

  	      const bool dbg_lut = false;
  	      if (dbg_lut )
  		{
  		  auto etaphi = intersectionEtaPhi(id, (*digiIt).getKeyWG(), (*digiIt).getStrip());
          
  		  //float eta_lut = muScales->getRegionalEtaScale(2)->getCenter(gblEta.global_eta);
  		  //float phi_lut = normalizedPhi( muScales->getPhiScale()->getLowEdge(gblPhi.global_phi));
  		  csctf::TrackStub stub = buildTrackStub((*digiIt), id);
  		  float eta_lut = stub.etaValue();
  		  float phi_lut = stub.phiValue();

  		  std::cout<<"DBGSRLUT "<<id.endcap()<<" "<<id.station()<<" "<<id.ring()<<" "<<id.chamber()<<"  "<<(*digiIt).getKeyWG()<<" "<<(*digiIt).getStrip()<<"  "<<etaphi.first<<" "<<etaphi.second<<"  "<<eta_lut<<" "<<phi_lut<<"  "<<etaphi.first - eta_lut<<" "<<deltaPhi(etaphi.second, phi_lut)<<std::endl;
  		}
  	    }
  	}
    }
  h_rt_nmplct->Fill(nmplct);
  for (int b=0;b<16;b++) {
    if ( b < minBxALCT_ || b > maxBxALCT_ ) continue;
    h_rt_nmplct_vs_bx->Fill(nmplct_per_bx[b],b);
  }
  for (int s=0; s<MAX_STATIONS; s++)  
    for (int i=1; i<=12; i++)
      {
	if (s!=0 && i>6) continue; // only ME1 has 12 subsectors
	h_rt_mplct_per_sector->Fill(nmplct_sector_st[s][i]);
	h_rt_mplct_per_sector_st[s]->Fill(nmplct_sector_st[s][i]);
	for (int j=0; j<16; j++) {
	  if ( j < minBxALCT_ || j > maxBxALCT_ ) continue;
	  h_rt_mplct_per_sector_vs_bx->Fill(nmplct_sector_bx_st[s][i][j],j+0.5);
	  h_rt_mplct_per_sector_vs_bx_st[s]->Fill(nmplct_sector_bx_st[s][i][j],j+0.5);
	  if (s==0 && i<7) h_rt_mplct_per_sector_vs_bx_st1t->Fill(nmplct_trigsector_bx_st1[i][j],j+0.5);
	}
      }
  if (debugRATE) std::cout<< "----- end nmplct="<<nmplct<<std::endl;


  //============ RATE TF TRACK ==================

  int ntftrack=0;
  if (debugRATE) std::cout<< "----- statring ntftrack"<<std::endl;
  std::vector<MatchCSCMuL1::TFTRACK> rtTFTracks;
  //  if (debugTFInef && inefTF) std::cout<<"#################### TF INEFFICIENCY ALL TFTRACKs:"<<std::endl;
  for ( L1CSCTrackCollection::const_iterator trk = l1Tracks->begin(); trk != l1Tracks->end(); trk++)
    {
      if ( trk->first.bx() < minRateBX_ || trk->first.bx() > maxRateBX_ )
  	{
  	  if (debugRATE) std::cout<<"discarding BX = "<< trk->first.bx() <<std::endl;
  	  continue;
  	}
      //if (trk->first.endcap()!=1) continue;
    
      MatchCSCMuL1::TFTRACK myTFTrk;
      myTFTrk.init( &(trk->first) , ptLUT, muScales, muPtScale);
      myTFTrk.dr = 999.;

      for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt = trk->second.begin();
  	   detUnitIt != trk->second.end(); detUnitIt++)
  	{
  	  const CSCDetId& id = (*detUnitIt).first;
  	  CSCDetId cid = id;
  	  const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
  	  for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++)
  	    {
  	      if (!((*digiIt).isValid())) std::cout<<"ALARM!!! match TFCAND to TFTRACK in rates: not valid id="<<id.rawId()<<" "<<id<<std::endl;
  	      bool me1a_case = (defaultME1a && id.station()==1 && id.ring()==1 && (*digiIt).getStrip() > 127);
  	      if (me1a_case){
  		CSCDetId id1a(id.endcap(),id.station(),4,id.chamber(),0);
  		cid = id1a;
  	      }
  	      //if (id.station()==1 && id.ring()==4) std::cout<<"me1adigi check: "<<(*digiIt)<<" "<<std::endl;
  	      myTFTrk.trgdigis.push_back( &*digiIt );
  	      myTFTrk.trgids.push_back( cid );
  	      myTFTrk.trgetaphis.push_back( intersectionEtaPhi(cid, (*digiIt).getKeyWG(), (*digiIt).getStrip()) );
  	      myTFTrk.trgstubs.push_back( buildTrackStub((*digiIt), cid) );
  	    }
  	}

      ntftrack++;
      rtTFTracks.push_back(myTFTrk);

      //    if (debugTFInef && inefTF) myTFTrk.print("(for inef checks)");
    
      if (myTFTrk.pt >= 20. && myTFTrk.hasStub(1) && myTFTrk.hasStub(2)){
  	int i1=-1, i2=-1, k=0;
  	for (auto id: myTFTrk.trgids)
  	  {
  	    if (id.station()==1) i1 = k;
  	    if (id.station()==2) i2 = k;
  	    ++k;
  	  }
  	if (i1>=0 && i2 >=0 ) {
  	  auto etaphi1 = myTFTrk.trgetaphis[i1];
  	  auto etaphi2 = myTFTrk.trgetaphis[i2];
  	  auto d = myTFTrk.trgids[i1];
  	  auto &stub = *(myTFTrk.trgdigis[i1]);
  	  std::cout<<"DBGdeta12 "<<d.endcap()<<" "<<d.ring()<<" "<<d.chamber()<<"  "<<stub.getKeyWG()<<" "<<stub.getStrip()<<"  "<<myTFTrk.nStubs(1,1,1,1,1)<<" "<<myTFTrk.pt<<" "<<myTFTrk.eta<<"  "<<etaphi1.first<<" "<<etaphi2.first<<" "<<etaphi1.first-etaphi2.first<<"  "<<etaphi1.second<<" "<<etaphi2.second<<" "<<deltaPhi(etaphi1.second,etaphi2.second)<<std::endl;

  	  if ( (etaphi1.first-etaphi2.first) > 0.1) {
  	    myTFTrk.print("");
  	    std::cout<<"############### CSCTFSPCoreLogic printout for large deta12 = "<<etaphi1.first-etaphi2.first<< " at "<<d.endcap()<<" "<<d.ring()<<" "<<d.chamber()<<std::endl;
  	    runCSCTFSP(mplcts, dttrigs);
  	    std::cout<<"############### end printout"<<std::endl;
  	  }
  	}
  	else {
  	  std::cout<<"myTFTrk.trgids corrupt"<<std::endl;
  	  myTFTrk.print("");
  	}
      }

      h_rt_tftrack_pt->Fill(myTFTrk.pt);
      h_rt_tftrack_bx->Fill(trk->first.bx());
      h_rt_tftrack_mode->Fill(myTFTrk.mode());
    }
  h_rt_ntftrack->Fill(ntftrack);
  if (debugRATE) std::cout<< "----- end ntftrack="<<ntftrack<<std::endl;


  //============ RATE TFCAND ==================

  int ntfcand=0, ntfcandpt10=0;
  if (debugRATE) std::cout<< "----- statring ntfcand"<<std::endl;
  std::vector<MatchCSCMuL1::TFCAND> rtTFCands;
  for ( std::vector< L1MuRegionalCand >::const_iterator trk = l1TfCands->begin(); trk != l1TfCands->end(); trk++)
    {
      if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
  	{
  	  if (debugRATE) std::cout<<"discarding BX = "<< trk->bx() <<std::endl;
  	  continue;
  	}
      double sign_eta = ( (trk->eta_packed() & 0x20) == 0) ? 1.:-1;
      //if ( sign_eta<0) continue;
      if (doSelectEtaForGMTRates_ && sign_eta<0) continue;

      MatchCSCMuL1::TFCAND myTFCand;
      myTFCand.init( &*trk , ptLUT, muScales, muPtScale);
      myTFCand.dr = 999.;
      //double tfpt = myTFCand.pt;

      //if (debugRATE) std::cout<< "----- eta/phi/pt "<<sign_eta<<"*"<<tfeta<<"/"<<tfphi<<"/"<<tfpt<<" "<< int(trk->eta_packed() & 0x1F) <<std::endl;

      ntfcand++;
      //if (tfpt>=10.) ntfcandpt10++;
      //h_rt_tfcand_pt->Fill(tfpt);
      h_rt_tfcand_bx->Fill(trk->bx());

      // find TF Candidate's  TF Track and its stubs:
      myTFCand.tftrack = 0;
      for (size_t tt = 0; tt<rtTFTracks.size(); tt++)
  	{
  	  if (trk->bx()          != rtTFTracks[tt].l1trk->bx()||
  	      trk->phi_packed()  != rtTFTracks[tt].phi_packed ||
  	      trk->pt_packed()   != rtTFTracks[tt].pt_packed  ||
  	      trk->eta_packed()  != rtTFTracks[tt].eta_packed   ) continue;
  	  myTFCand.tftrack = &(rtTFTracks[tt]);
  	  // ids now hold *trigger segments IDs*
  	  myTFCand.ids = rtTFTracks[tt].trgids;
  	  myTFCand.nTFStubs = rtTFTracks[tt].nStubs(1,1,1,1,1);
  	}
      rtTFCands.push_back(myTFCand);
      if(myTFCand.tftrack == NULL){
  	std::cout<<"myTFCand.tftrack == NULL:"<<std::endl;
  	std::cout<<" cand: "<<trk->pt_packed()<<" "<<trk->eta_packed()<<" "<<trk->phi_packed()<<" "<<trk->bx()<<std::endl;
  	std::cout<<" trk: "<<std::endl;
  	for (size_t tt = 0; tt<rtTFTracks.size(); tt++)
  	  std::cout<<"       "<<rtTFTracks[tt].pt_packed<<" "<<rtTFTracks[tt].eta_packed<<" "<<rtTFTracks[tt].phi_packed<<" "<<rtTFTracks[tt].l1trk->bx()<<std::endl;
      }

      if (myTFCand.tftrack != NULL) {
  	double tfpt = myTFCand.tftrack->pt;
  	double tfeta = myTFCand.tftrack->eta;

  	if (tfpt>=10.) ntfcandpt10++;
      
  	h_rt_tfcand_pt->Fill(tfpt);
    
  	unsigned int ntrg_stubs = myTFCand.tftrack->trgdigis.size();
  	if (ntrg_stubs!=myTFCand.ids.size())
  	  std::cout<<"OBA!!! trgdigis.size()!=ids.size(): "<<ntrg_stubs<<"!="<<myTFCand.ids.size()<<std::endl;
  	if (ntrg_stubs>=2) h_rt_tfcand_pt_2st->Fill(tfpt);
  	if (ntrg_stubs>=3) h_rt_tfcand_pt_3st->Fill(tfpt);
  	//std::cout<<"\n nnntf: "<<ntrg_stubs<<" "<<myTFCand.tftrack->nStubs(0,1,1,1,1)<<std::endl;
  	//if (ntrg_stubs != myTFCand.tftrack->nStubs()) myTFCand.tftrack->print("non-equal nstubs!");
  	//if (fabs(myTFCand.eta)>1.25 && fabs(myTFCand.eta)<1.9) {
  	if (etaRangeHelpers::isME42EtaRegion(myTFCand.eta)) {
  	  if (ntrg_stubs>=2) h_rt_tfcand_pt_h42_2st->Fill(tfpt);
  	  if (ntrg_stubs>=3) h_rt_tfcand_pt_h42_3st->Fill(tfpt);
  	}

  	h_rt_tfcand_eta->Fill(tfeta);
  	if (tfpt>=5.) h_rt_tfcand_eta_pt5->Fill(tfeta);
  	if (tfpt>=10.) h_rt_tfcand_eta_pt10->Fill(tfeta);
  	if (tfpt>=15.) h_rt_tfcand_eta_pt15->Fill(tfeta);
  	h_rt_tfcand_pt_vs_eta->Fill(tfpt,tfeta);

  	unsigned ntf_stubs = myTFCand.tftrack->nStubs();
  	if (ntf_stubs>=3) {
  	  h_rt_tfcand_eta_3st->Fill(tfeta);
  	  if (tfpt>=5.) h_rt_tfcand_eta_pt5_3st->Fill(tfeta);
  	  if (tfpt>=10.) h_rt_tfcand_eta_pt10_3st->Fill(tfeta);
  	  if (tfpt>=15.) h_rt_tfcand_eta_pt15_3st->Fill(tfeta);
  	  h_rt_tfcand_pt_vs_eta_3st->Fill(tfpt,tfeta);
  	}
  	if (tfeta<2.0999 || ntf_stubs>=3) {
  	  h_rt_tfcand_eta_3st1a->Fill(tfeta);
  	  if (tfpt>=5.) h_rt_tfcand_eta_pt5_3st1a->Fill(tfeta);
  	  if (tfpt>=10.) h_rt_tfcand_eta_pt10_3st1a->Fill(tfeta);
  	  if (tfpt>=15.) h_rt_tfcand_eta_pt15_3st1a->Fill(tfeta);
  	  h_rt_tfcand_pt_vs_eta_3st1a->Fill(tfpt,tfeta);
  	}
      }
      //else std::cout<<"Strange: myTFCand.tftrack != NULL"<<std::endl;
    }
  h_rt_ntfcand->Fill(ntfcand);
  h_rt_ntfcand_pt10->Fill(ntfcandpt10);
  if (debugRATE) std::cout<< "----- end ntfcand/ntfcandpt10="<<ntfcand<<"/"<<ntfcandpt10<<std::endl;


  //============ RATE GMT REGIONAL ==================

  int ngmtcsc=0, ngmtcscpt10=0;
  if (debugRATE) std::cout<< "----- statring ngmt csc"<<std::endl;
  std::vector<MatchCSCMuL1::GMTREGCAND> rtGMTREGCands;
  float max_pt_2s = -1, max_pt_3s = -1, max_pt_2q = -1, max_pt_3q = -1;
  float max_pt_2s_eta = -111, max_pt_3s_eta = -111, max_pt_2q_eta = -111, max_pt_3q_eta = -111;
  float max_pt_me42_2s = -1, max_pt_me42_3s = -1, max_pt_me42_2q = -1, max_pt_me42_3q = -1;
  float max_pt_me42r_2s = -1, max_pt_me42r_3s = -1, max_pt_me42r_2q = -1, max_pt_me42r_3q = -1;

  float max_pt_2s_2s1b = -1, max_pt_2s_2s1b_eta = -111; 
  float max_pt_2s_no1a = -1;//, max_pt_2s_eta_no1a = -111;
  float max_pt_2s_1b = -1;//,   max_pt_2s_eta_1b = -111;
  float max_pt_3s_no1a = -1, max_pt_3s_eta_no1a = -111;
  float max_pt_3s_1b = -1,   max_pt_3s_eta_1b = -111;
  float max_pt_3s_1ab = -1,   max_pt_3s_eta_1ab = -111;

  float max_pt_3s_2s1b = -1,      max_pt_3s_2s1b_eta = -111;
  float max_pt_3s_2s1b_no1a = -1, max_pt_3s_2s1b_eta_no1a = -111;
  float max_pt_3s_2s123_no1a = -1, max_pt_3s_2s123_eta_no1a = -111;
  float max_pt_3s_2s13_no1a = -1, max_pt_3s_2s13_eta_no1a = -111;
  float max_pt_3s_2s1b_1b = -1,   max_pt_3s_2s1b_eta_1b = -111;
  float max_pt_3s_2s123_1b = -1, max_pt_3s_2s123_eta_1b = -111;
  float max_pt_3s_2s13_1b = -1, max_pt_3s_2s13_eta_1b = -111;

  float max_pt_3s_3s1b = -1,      max_pt_3s_3s1b_eta = -111;
  float max_pt_3s_3s1b_no1a = -1, max_pt_3s_3s1b_eta_no1a = -111;
  float max_pt_3s_3s1b_1b = -1,   max_pt_3s_3s1b_eta_1b = -111;


  float max_pt_3s_3s1ab = -1,      max_pt_3s_3s1ab_eta = -111;
  float max_pt_3s_3s1ab_no1a = -1;//, max_pt_3s_3s1ab_eta_no1a = -111;
  float max_pt_3s_3s1ab_1b = -1;//,   max_pt_3s_3s1ab_eta_1b = -111;

  MatchCSCMuL1::TFTRACK *trk__max_pt_3s_3s1b_eta = nullptr;
  //  MatchCSCMuL1::TFTRACK *trk__max_pt_3s_3s1ab_eta = nullptr;
  MatchCSCMuL1::TFTRACK *trk__max_pt_2s1b_1b = nullptr;
  const CSCCorrelatedLCTDigi * the_me1_stub = nullptr;
  CSCDetId the_me1_id;
  std::map<int,int> bx2n;
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
  for ( std::vector<L1MuRegionalCand>::const_iterator trk = l1GmtCSCCands.begin(); trk != l1GmtCSCCands.end(); trk++)
    {
      if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
  	{
  	  if (debugRATE) std::cout<<"discarding BX = "<< trk->bx() <<std::endl;
  	  continue;
  	}
      double sign_eta = ( (trk->eta_packed() & 0x20) == 0) ? 1.:-1;
      if (doSelectEtaForGMTRates_ && sign_eta<0) continue;

      bx2n[trk->bx()] += 1;

      MatchCSCMuL1::GMTREGCAND myGMTREGCand;
      myGMTREGCand.init( &*trk , muScales, muPtScale);
      myGMTREGCand.dr = 999.;

      myGMTREGCand.tfcand = NULL;
      for (unsigned i=0; i< rtTFCands.size(); i++)
  	{
  	  if ( trk->bx()          != rtTFCands[i].l1cand->bx()         ||
  	       trk->phi_packed()  != rtTFCands[i].l1cand->phi_packed() ||
  	       trk->eta_packed()  != rtTFCands[i].l1cand->eta_packed()   ) continue;
  	  myGMTREGCand.tfcand = &(rtTFCands[i]);
  	  myGMTREGCand.ids = rtTFCands[i].ids;
  	  myGMTREGCand.nTFStubs = rtTFCands[i].nTFStubs;
  	  break;
  	}
      rtGMTREGCands.push_back(myGMTREGCand);

      float geta = fabs(myGMTREGCand.eta);
      float gpt = myGMTREGCand.pt;

      bool eta_me42 = etaRangeHelpers::isME42EtaRegion(myGMTREGCand.eta);
      bool eta_me42r = etaRangeHelpers::isME42RPCEtaRegion(myGMTREGCand.eta);
      //if (geta>=1.2 && geta<=1.8) eta_me42 = 1;
      bool eta_q = (geta > 1.2);

      bool has_me1_stub = false;
      size_t n_stubs = 0;

      if (myGMTREGCand.tfcand != NULL)
  	{
  	  //rtGMTREGCands.push_back(myGMTREGCand);

  	  if (myGMTREGCand.tfcand->tftrack != NULL)
  	    {
  	      has_me1_stub = myGMTREGCand.tfcand->tftrack->hasStub(1);
  	    }

  	  bool has_1b_stub = false;
  	  for (auto& id: myGMTREGCand.ids) if (id.iChamberType() == 2) {
  	      has_1b_stub = true;
  	      continue;
  	    }

  	  bool has_1a_stub = false;
  	  for (auto& id: myGMTREGCand.ids) if (id.iChamberType() == 1) {
  	      has_1a_stub = true;
  	      continue;
  	    }

  	  bool eta_me1b = etaRangeHelpers::isME1bEtaRegion(myGMTREGCand.eta);
  	  bool eta_me1ab = etaRangeHelpers::isME1abEtaRegion(myGMTREGCand.eta);
  	  bool eta_me1a = etaRangeHelpers::isME1aEtaRegion(myGMTREGCand.eta);
  	  bool eta_me1b_whole = etaRangeHelpers::isME1bEtaRegion(myGMTREGCand.eta, 1.6, 2.14);
  	  bool eta_no1a = (geta >= 1.2 && geta < 2.14);
	  
  	  n_stubs = myGMTREGCand.nTFStubs;
  	  size_t n_stubs_id = myGMTREGCand.ids.size();
  	  //if (n_stubs == n_stubs_id) std::cout<<"n_stubs good"<<std::endl;
  	  if (n_stubs != n_stubs_id) std::cout<<"n_stubs bad: "<<eta_q<<" "<<n_stubs<<" != "<<n_stubs_id<<" "<< geta  <<std::endl;
	  
  	  auto stub_ids = myGMTREGCand.tfcand->tftrack->trgids;
  	  for (size_t i=0; i<stub_ids.size(); ++i)
  	    {
  	      // pick up the ME11 stub of this track
  	      if ( !(stub_ids[i].station() == 1 && (stub_ids[i].ring() == 1 || stub_ids[i].ring() == 4) ) ) continue;
  	      the_me1_stub = (myGMTREGCand.tfcand->tftrack->trgdigis)[i];
  	      the_me1_id = stub_ids[i];
  	    }

  	  int tf_mode = myGMTREGCand.tfcand->tftrack->mode();
  	  bool ok_2s123 = (tf_mode != 0xd); // excludes ME1-ME4 stub tf tracks
  	  bool ok_2s13 = (ok_2s123 && (tf_mode != 0x6)); // excludes ME1-ME2 and ME1-ME4 stub tf tracks

  	  if (n_stubs >= 2)
  	    {
  	      h_rt_gmt_csc_pt_2st->Fill(gpt);
  	      if (eta_me42) h_rt_gmt_csc_pt_2s42->Fill(gpt);
  	      if (eta_me42r) h_rt_gmt_csc_pt_2s42r->Fill(gpt);
  	      if (            gpt > max_pt_2s     ) { max_pt_2s = gpt; max_pt_2s_eta = geta; }
  	      if (eta_me1b && gpt > max_pt_2s_1b  ) { max_pt_2s_1b = gpt; /*max_pt_2s_eta_1b = geta;*/ }
  	      if (eta_no1a && gpt > max_pt_2s_no1a) { max_pt_2s_no1a = gpt; /*max_pt_2s_eta_no1a = geta;*/ }
  	      if (eta_me42 && gpt > max_pt_me42_2s) max_pt_me42_2s = gpt;
  	      if (eta_me42r && gpt>max_pt_me42r_2s) max_pt_me42r_2s = gpt;
  	    }
  	  if ( (has_1b_stub && n_stubs >=2) || ( !has_1b_stub && !eta_me1b_whole && n_stubs >=2 ) )
  	    {
  	      if (            gpt > max_pt_2s_2s1b) { max_pt_2s_2s1b = gpt; max_pt_2s_2s1b_eta = geta; }
  	    }

  	  if (n_stubs >= 3)
  	    {
  	      h_rt_gmt_csc_pt_3st->Fill(gpt);
  	      if (eta_me42) h_rt_gmt_csc_pt_3s42->Fill(gpt);
  	      if (eta_me42r) h_rt_gmt_csc_pt_3s42r->Fill(gpt);
  	      if (            gpt > max_pt_3s     ) { max_pt_3s = gpt; max_pt_3s_eta = geta; }


  	      if (eta_me1b && gpt > max_pt_3s_1b  ) { max_pt_3s_1b = gpt; max_pt_3s_eta_1b = geta; }
  	      if (eta_me1ab && gpt > max_pt_3s_1ab  ) { max_pt_3s_1ab = gpt; max_pt_3s_eta_1ab = geta; }

  	      if (eta_no1a && gpt > max_pt_3s_no1a) { max_pt_3s_no1a = gpt; max_pt_3s_eta_no1a = geta; }
  	      if (eta_me42 && gpt > max_pt_me42_3s) max_pt_me42_3s = gpt;
  	      if (eta_me42r && gpt>max_pt_me42r_3s) max_pt_me42r_3s = gpt;
  	    }

  	  if ( (has_1b_stub && n_stubs >=2) || ( !has_1b_stub && !eta_me1b_whole && n_stubs >=3 ) )
  	    {
  	      if (            gpt > max_pt_3s_2s1b     ) { max_pt_3s_2s1b = gpt; max_pt_3s_2s1b_eta = geta; }

  	      if (eta_me1b && gpt > max_pt_3s_2s1b_1b  ) { max_pt_3s_2s1b_1b = gpt; max_pt_3s_2s1b_eta_1b = geta; 
  		trk__max_pt_2s1b_1b = myGMTREGCand.tfcand->tftrack; }
  	      if (eta_me1b && gpt > max_pt_3s_2s123_1b && ok_2s123 ) 
  		{ max_pt_3s_2s123_1b = gpt; max_pt_3s_2s123_eta_1b = geta; }
  	      if (eta_me1b && gpt > max_pt_3s_2s13_1b && ok_2s13 ) 
  		{ max_pt_3s_2s13_1b = gpt; max_pt_3s_2s13_eta_1b = geta; }

  	      if (eta_no1a && gpt > max_pt_3s_2s1b_no1a) { max_pt_3s_2s1b_no1a = gpt; max_pt_3s_2s1b_eta_no1a = geta; }
  	      if (eta_no1a && gpt > max_pt_3s_2s123_no1a && (!eta_me1b || (eta_me1b && ok_2s123) ) )
  		{ max_pt_3s_2s123_no1a = gpt; max_pt_3s_2s123_eta_no1a = geta; }
  	      if (eta_no1a && gpt > max_pt_3s_2s13_no1a && (!eta_me1b || (eta_me1b && ok_2s13) ) )
  		{ max_pt_3s_2s13_no1a = gpt; max_pt_3s_2s13_eta_no1a = geta; }
  	    }

  	  if ( (has_1b_stub && n_stubs >=3) || ( !has_1b_stub && !eta_me1b_whole && n_stubs >=3 ) )
  	    {
  	      if (            gpt > max_pt_3s_3s1b      ) { max_pt_3s_3s1b = gpt; max_pt_3s_3s1b_eta = geta;
  		trk__max_pt_3s_3s1b_eta = myGMTREGCand.tfcand->tftrack; }
  	      if (eta_me1b && gpt > max_pt_3s_3s1b_1b   ) { max_pt_3s_3s1b_1b = gpt; max_pt_3s_3s1b_eta_1b = geta; }
  	      if (eta_no1a && gpt > max_pt_3s_3s1b_no1a ) { max_pt_3s_3s1b_no1a = gpt; max_pt_3s_3s1b_eta_no1a = geta; }
  	    }

  	  if (n_stubs >=3 && ( (eta_me1a && has_1a_stub) || (eta_me1b && has_1b_stub) || (!has_1a_stub && !has_1b_stub && !eta_me1ab) ) )
  	    {
  	      if (            gpt > max_pt_3s_3s1ab      ) { max_pt_3s_3s1ab = gpt; max_pt_3s_3s1ab_eta = geta;
  		//trk__max_pt_3s_3s1ab_eta = myGMTREGCand.tfcand->tftrack; 
	      }
  	      if (eta_me1b && gpt > max_pt_3s_3s1ab_1b   ) { max_pt_3s_3s1ab_1b = gpt; 
		//max_pt_3s_3s1ab_eta_1b = geta; 
	      }
  	      if (eta_no1a && gpt > max_pt_3s_3s1ab_no1a ) { max_pt_3s_3s1ab_no1a = gpt; 
		//max_pt_3s_3s1ab_eta_no1a = geta; 
	      }
  	    }


  	} else { 
  	std::cout<<"GMTCSC match not found pt="<<gpt<<" eta="<<myGMTREGCand.eta<<"  packed: "<<trk->phi_packed()<<" "<<trk->eta_packed()<<std::endl;
  	for (unsigned i=0; i< rtTFCands.size(); i++) std::cout<<"    "<<rtTFCands[i].l1cand->phi_packed()<<" "<<rtTFCands[i].l1cand->eta_packed();
  	std::cout<<std::endl;
  	std::cout<<"  all tfcands:";
  	for ( std::vector< L1MuRegionalCand >::const_iterator ctrk = l1TfCands->begin(); ctrk != l1TfCands->end(); ctrk++)
  	  if (!( ctrk->bx() < minRateBX_ || ctrk->bx() > maxRateBX_ )) std::cout<<"    "<<ctrk->phi_packed()<<" "<<ctrk->eta_packed();
  	std::cout<<std::endl;
      }
    
      if (trk->quality()>=2) {
  	h_rt_gmt_csc_pt_2q->Fill(gpt);
  	if (eta_me42) h_rt_gmt_csc_pt_2q42->Fill(gpt);
  	if (eta_me42r) h_rt_gmt_csc_pt_2q42r->Fill(gpt);
  	if (gpt > max_pt_2q) {max_pt_2q = gpt; max_pt_2q_eta = geta;}
  	if (eta_me42 && gpt > max_pt_me42_2q) max_pt_me42_2q = gpt;
  	if (eta_me42r && gpt > max_pt_me42r_2q) max_pt_me42r_2q = gpt;
      }
      if ((!eta_q && trk->quality()>=2) || ( eta_q && trk->quality()>=3) ) {
  	h_rt_gmt_csc_pt_3q->Fill(gpt);
  	if (eta_me42) h_rt_gmt_csc_pt_3q42->Fill(gpt);
  	if (eta_me42r) h_rt_gmt_csc_pt_3q42r->Fill(gpt);
  	if (gpt > max_pt_3q) {max_pt_3q = gpt; max_pt_3q_eta = geta;}
  	if (eta_me42 && gpt > max_pt_me42_3q) max_pt_me42_3q = gpt;
  	if (eta_me42r && gpt > max_pt_me42r_3q) max_pt_me42r_3q = gpt;
      }
    
      //if (trk->quality()>=3 && !(myGMTREGCand.ids.size()>=3) ) {
      //  std::cout<<"weird stubs number "<<myGMTREGCand.ids.size()<<" for q="<<trk->quality()<<std::endl;
      //  if (myGMTREGCand.tfcand->tftrack != NULL) myGMTREGCand.tfcand->tftrack->print("");
      //  else std::cout<<"null tftrack!"<<std::endl;
      //}

      //    if (trk->quality()>=3 && gpt >=40. && etaRangeHelpers::isME1bEtaRegion(myGMTREGCand.eta) ) {
      //      std::cout<<"highpt csctf in ME1b "<<std::endl;
      //      myGMTREGCand.tfcand->tftrack->print("");
      //    }
      if (has_me1_stub && n_stubs > 2 && gpt >= 30. && geta> 1.6 && geta < 2.15 ) {
  	std::cout<<"highpt csctf in ME1b "<<std::endl;
  	myGMTREGCand.tfcand->tftrack->print("");
      }


      ngmtcsc++;
      if (gpt>=10.) ngmtcscpt10++;
      h_rt_gmt_csc_pt->Fill(gpt);
      h_rt_gmt_csc_eta->Fill(geta);
      h_rt_gmt_csc_bx->Fill(trk->bx());
  
      h_rt_gmt_csc_q->Fill(trk->quality());
      if (eta_me42) h_rt_gmt_csc_q_42->Fill(trk->quality());
      if (eta_me42r) h_rt_gmt_csc_q_42r->Fill(trk->quality());
    }

  h_rt_ngmt_csc->Fill(ngmtcsc);
  h_rt_ngmt_csc_pt10->Fill(ngmtcscpt10);
  if (max_pt_2s>0) h_rt_gmt_csc_ptmax_2s->Fill(max_pt_2s);
  if (max_pt_3s>0) h_rt_gmt_csc_ptmax_3s->Fill(max_pt_3s);

  if (max_pt_2s_1b>0) h_rt_gmt_csc_ptmax_2s_1b->Fill(max_pt_2s_1b);
  if (max_pt_2s_no1a>0) h_rt_gmt_csc_ptmax_2s_no1a->Fill(max_pt_2s_no1a);
  if (max_pt_3s_1b>0) h_rt_gmt_csc_ptmax_3s_1b->Fill(max_pt_3s_1b);
  if (max_pt_3s_no1a>0) h_rt_gmt_csc_ptmax_3s_no1a->Fill(max_pt_3s_no1a);
  if (max_pt_3s_2s1b>0) h_rt_gmt_csc_ptmax_3s_2s1b->Fill(max_pt_3s_2s1b);
  if (max_pt_3s_2s1b_1b>0) h_rt_gmt_csc_ptmax_3s_2s1b_1b->Fill(max_pt_3s_2s1b_1b);
  if (max_pt_3s_2s123_1b>0) h_rt_gmt_csc_ptmax_3s_2s123_1b->Fill(max_pt_3s_2s123_1b);
  if (max_pt_3s_2s13_1b>0) h_rt_gmt_csc_ptmax_3s_2s13_1b->Fill(max_pt_3s_2s13_1b);
  if (max_pt_3s_2s1b_no1a>0) h_rt_gmt_csc_ptmax_3s_2s1b_no1a->Fill(max_pt_3s_2s1b_no1a);
  if (max_pt_3s_2s123_no1a>0) h_rt_gmt_csc_ptmax_3s_2s123_no1a->Fill(max_pt_3s_2s123_no1a);
  if (max_pt_3s_2s13_no1a>0) h_rt_gmt_csc_ptmax_3s_2s13_no1a->Fill(max_pt_3s_2s13_no1a);
  if (max_pt_3s_3s1b>0) h_rt_gmt_csc_ptmax_3s_3s1b->Fill(max_pt_3s_3s1b);
  if (max_pt_3s_3s1b_1b>0) h_rt_gmt_csc_ptmax_3s_3s1b_1b->Fill(max_pt_3s_3s1b_1b);
  if (max_pt_3s_3s1b_no1a>0) h_rt_gmt_csc_ptmax_3s_3s1b_no1a->Fill(max_pt_3s_3s1b_no1a);

  if (max_pt_2q>0) h_rt_gmt_csc_ptmax_2q->Fill(max_pt_2q);
  if (max_pt_3q>0) h_rt_gmt_csc_ptmax_3q->Fill(max_pt_3q);

  if (max_pt_2s>=10.) h_rt_gmt_csc_ptmax10_eta_2s->Fill(max_pt_2s_eta);
  if (max_pt_2s_2s1b>=10.) h_rt_gmt_csc_ptmax10_eta_2s_2s1b->Fill(max_pt_2s_2s1b_eta);
  if (max_pt_3s>=10.) h_rt_gmt_csc_ptmax10_eta_3s->Fill(max_pt_3s_eta);
  if (max_pt_3s_1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_1b->Fill(max_pt_3s_eta_1b);
  if (max_pt_3s_no1a>=10.) h_rt_gmt_csc_ptmax10_eta_3s_no1a->Fill(max_pt_3s_eta_no1a);
  if (max_pt_3s_2s1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s1b->Fill(max_pt_3s_2s1b_eta);
  if (max_pt_3s_2s1b_1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s1b_1b->Fill(max_pt_3s_2s1b_eta_1b);
  if (max_pt_3s_2s123_1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s123_1b->Fill(max_pt_3s_2s123_eta_1b);
  if (max_pt_3s_2s13_1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s13_1b->Fill(max_pt_3s_2s13_eta_1b);
  if (max_pt_3s_2s1b_no1a>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s1b_no1a->Fill(max_pt_3s_2s1b_eta_no1a);
  if (max_pt_3s_2s123_no1a>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s123_no1a->Fill(max_pt_3s_2s123_eta_no1a);
  if (max_pt_3s_2s13_no1a>=10.) h_rt_gmt_csc_ptmax10_eta_3s_2s13_no1a->Fill(max_pt_3s_2s13_eta_no1a);
  if (max_pt_3s_3s1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_3s1b->Fill(max_pt_3s_3s1b_eta);
  if (max_pt_3s_3s1b_1b>=10.) h_rt_gmt_csc_ptmax10_eta_3s_3s1b_1b->Fill(max_pt_3s_3s1b_eta_1b);
  if (max_pt_3s_3s1b_no1a>=10.) h_rt_gmt_csc_ptmax10_eta_3s_3s1b_no1a->Fill(max_pt_3s_3s1b_eta_no1a);
  if (max_pt_2q>=10.) h_rt_gmt_csc_ptmax10_eta_2q->Fill(max_pt_2q_eta);
  if (max_pt_3q>=10.) h_rt_gmt_csc_ptmax10_eta_3q->Fill(max_pt_3q_eta);

  if (max_pt_2s>=20.) h_rt_gmt_csc_ptmax20_eta_2s->Fill(max_pt_2s_eta);
  if (max_pt_2s_2s1b>=20.) h_rt_gmt_csc_ptmax20_eta_2s_2s1b->Fill(max_pt_2s_2s1b_eta);
  if (max_pt_3s>=20.) h_rt_gmt_csc_ptmax20_eta_3s->Fill(max_pt_3s_eta);

  if (max_pt_3s_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_1b->Fill(max_pt_3s_eta_1b);
  if (max_pt_3s_1ab>=20.) h_rt_gmt_csc_ptmax20_eta_3s_1ab->Fill(max_pt_3s_eta_1ab);

  if (max_pt_3s_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_no1a->Fill(max_pt_3s_eta_no1a);
  if (max_pt_3s_2s1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s1b->Fill(max_pt_3s_2s1b_eta);
  if (max_pt_3s_2s1b_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s1b_1b->Fill(max_pt_3s_2s1b_eta_1b);
  if (max_pt_3s_2s123_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s123_1b->Fill(max_pt_3s_2s123_eta_1b);
  if (max_pt_3s_2s13_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s13_1b->Fill(max_pt_3s_2s13_eta_1b);
  if (max_pt_3s_2s1b_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s1b_no1a->Fill(max_pt_3s_2s1b_eta_no1a);
  if (max_pt_3s_2s123_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s123_no1a->Fill(max_pt_3s_2s123_eta_no1a);
  if (max_pt_3s_2s13_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s13_no1a->Fill(max_pt_3s_2s13_eta_no1a);

  if (max_pt_3s_3s1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1b->Fill(max_pt_3s_3s1b_eta);
  if (max_pt_3s_3s1ab>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1ab->Fill(max_pt_3s_3s1b_eta);

  if (max_pt_3s_3s1b_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1b_1b->Fill(max_pt_3s_3s1b_eta_1b);
  if (max_pt_3s_3s1b_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1b_no1a->Fill(max_pt_3s_3s1b_eta_no1a);
  if (max_pt_2q>=20.) h_rt_gmt_csc_ptmax20_eta_2q->Fill(max_pt_2q_eta);
  if (max_pt_3q>=20.) h_rt_gmt_csc_ptmax20_eta_3q->Fill(max_pt_3q_eta);

  if (max_pt_2s>=30.) h_rt_gmt_csc_ptmax30_eta_2s->Fill(max_pt_2s_eta);
  if (max_pt_2s_2s1b>=30.) h_rt_gmt_csc_ptmax30_eta_2s_2s1b->Fill(max_pt_2s_2s1b_eta);
  if (max_pt_3s>=30.) h_rt_gmt_csc_ptmax30_eta_3s->Fill(max_pt_3s_eta);

  if (max_pt_3s_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_1b->Fill(max_pt_3s_eta_1b);
  if (max_pt_3s_1ab>=30.) h_rt_gmt_csc_ptmax30_eta_3s_1ab->Fill(max_pt_3s_eta_1ab);


  if (max_pt_3s_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_no1a->Fill(max_pt_3s_eta_no1a);
  if (max_pt_3s_2s1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s1b->Fill(max_pt_3s_2s1b_eta);
  if (max_pt_3s_2s1b_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s1b_1b->Fill(max_pt_3s_2s1b_eta_1b);
  if (max_pt_3s_2s123_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s123_1b->Fill(max_pt_3s_2s123_eta_1b);
  if (max_pt_3s_2s13_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s13_1b->Fill(max_pt_3s_2s13_eta_1b);
  if (max_pt_3s_2s1b_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s1b_no1a->Fill(max_pt_3s_2s1b_eta_no1a);
  if (max_pt_3s_2s123_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s123_no1a->Fill(max_pt_3s_2s123_eta_no1a);
  if (max_pt_3s_2s13_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s13_no1a->Fill(max_pt_3s_2s13_eta_no1a);


  if (max_pt_3s_3s1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_3s1b->Fill(max_pt_3s_3s1b_eta);
  if (max_pt_3s_3s1ab>=30.) h_rt_gmt_csc_ptmax30_eta_3s_3s1ab->Fill(max_pt_3s_3s1ab_eta);



  if (max_pt_3s_3s1b_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_3s1b_1b->Fill(max_pt_3s_3s1b_eta_1b);
  if (max_pt_3s_3s1b_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_3s1b_no1a->Fill(max_pt_3s_3s1b_eta_no1a);

  if (max_pt_2q>=30.) h_rt_gmt_csc_ptmax30_eta_2q->Fill(max_pt_2q_eta);
  if (max_pt_3q>=30.) h_rt_gmt_csc_ptmax30_eta_3q->Fill(max_pt_3q_eta);

  if (max_pt_me42_2s>0) h_rt_gmt_csc_ptmax_2s42->Fill(max_pt_me42_2s);
  if (max_pt_me42_3s>0) h_rt_gmt_csc_ptmax_3s42->Fill(max_pt_me42_3s);
  if (max_pt_me42_2q>0) h_rt_gmt_csc_ptmax_2q42->Fill(max_pt_me42_2q);
  if (max_pt_me42_3q>0) h_rt_gmt_csc_ptmax_3q42->Fill(max_pt_me42_3q);
  if (max_pt_me42r_2s>0) h_rt_gmt_csc_ptmax_2s42r->Fill(max_pt_me42r_2s);
  if (max_pt_me42r_3s>0) h_rt_gmt_csc_ptmax_3s42r->Fill(max_pt_me42r_3s);
  if (max_pt_me42r_2q>0) h_rt_gmt_csc_ptmax_2q42r->Fill(max_pt_me42r_2q);
  if (max_pt_me42r_3q>0) h_rt_gmt_csc_ptmax_3q42r->Fill(max_pt_me42r_3q);
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) h_rt_ngmt_csc_per_bx->Fill(bx2n[bx]);
  if (debugRATE) std::cout<< "----- end ngmt csc/ngmtpt10="<<ngmtcsc<<"/"<<ngmtcscpt10<<std::endl;

  if (max_pt_3s_3s1b>=30.) 
    {
      std::cout<<"filled h_rt_gmt_csc_ptmax30_eta_3s_3s1b eta "<<max_pt_3s_3s1b_eta<<std::endl;
      if (trk__max_pt_3s_3s1b_eta) trk__max_pt_3s_3s1b_eta->print("");
    }

  if (max_pt_3s_2s1b_1b >= 10. && trk__max_pt_2s1b_1b)
    {
      const int Nthr = 6;
      float tfc_pt_thr[Nthr] = {10., 15., 20., 25., 30., 40.};
      for (int i=0; i<Nthr; ++i) if (max_pt_3s_2s1b_1b >= tfc_pt_thr[i])
  				   {
  				     h_rt_gmt_csc_mode_2s1b_1b[i]->Fill(trk__max_pt_2s1b_1b->mode());
  				   }
      if (the_me1_stub) std::cout<<"DBGMODE "<<the_me1_id.endcap()<<" "<<the_me1_id.chamber()<<" "<<trk__max_pt_2s1b_1b->pt<<" "<<trk__max_pt_2s1b_1b->mode()<<" "<<pbend[the_me1_stub->getPattern()] <<" "<<the_me1_stub->getGEMDPhi()<<std::endl;
    }

  int ngmtrpcf=0, ngmtrpcfpt10=0;
  if (debugRATE) std::cout<< "----- statring ngmt rpcf"<<std::endl;
  std::vector<MatchCSCMuL1::GMTREGCAND> rtGMTRPCfCands;
  float max_pt_me42 = -1, max_pt = -1, max_pt_eta = -111;
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
  for ( std::vector<L1MuRegionalCand>::const_iterator trk = l1GmtRPCfCands.begin(); trk != l1GmtRPCfCands.end(); trk++)
    {
      if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
  	{
  	  if (debugRATE) std::cout<<"discarding BX = "<< trk->bx() <<std::endl;
  	  continue;
  	}
      double sign_eta = ( (trk->eta_packed() & 0x20) == 0) ? 1.:-1;
      if (doSelectEtaForGMTRates_ && sign_eta<0) continue;

      bx2n[trk->bx()] += 1;
      MatchCSCMuL1::GMTREGCAND myGMTREGCand;

      myGMTREGCand.init( &*trk , muScales, muPtScale);
      myGMTREGCand.dr = 999.;

      myGMTREGCand.tfcand = NULL;
      rtGMTRPCfCands.push_back(myGMTREGCand);

      ngmtrpcf++;
      if (myGMTREGCand.pt>=10.) ngmtrpcfpt10++;
      h_rt_gmt_rpcf_pt->Fill(myGMTREGCand.pt);
      h_rt_gmt_rpcf_eta->Fill(fabs(myGMTREGCand.eta));
      h_rt_gmt_rpcf_bx->Fill(trk->bx());

      bool eta_me42 = etaRangeHelpers::isME42RPCEtaRegion(myGMTREGCand.eta);
      //if (fabs(myGMTREGCand.eta)>=1.2 && fabs(myGMTREGCand.eta)<=1.8) eta_me42 = 1;

      if(eta_me42) h_rt_gmt_rpcf_pt_42->Fill(myGMTREGCand.pt);
      if(eta_me42 && myGMTREGCand.pt > max_pt_me42) max_pt_me42 = myGMTREGCand.pt;
      if(myGMTREGCand.pt > max_pt) { max_pt = myGMTREGCand.pt;  max_pt_eta = fabs(myGMTREGCand.eta);}
    
      h_rt_gmt_rpcf_q->Fill(trk->quality());
      if (eta_me42) h_rt_gmt_rpcf_q_42->Fill(trk->quality());
    }
  h_rt_ngmt_rpcf->Fill(ngmtrpcf);
  h_rt_ngmt_rpcf_pt10->Fill(ngmtrpcfpt10);
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) h_rt_ngmt_rpcf_per_bx->Fill(bx2n[bx]);
  if (max_pt>0) h_rt_gmt_rpcf_ptmax->Fill(max_pt);
  if (max_pt>=10.) h_rt_gmt_rpcf_ptmax10_eta->Fill(max_pt_eta);
  if (max_pt>=20.) h_rt_gmt_rpcf_ptmax20_eta->Fill(max_pt_eta);
  if (max_pt_me42>0) h_rt_gmt_rpcf_ptmax_42->Fill(max_pt_me42);
  if (debugRATE) std::cout<< "----- end ngmt rpcf/ngmtpt10="<<ngmtrpcf<<"/"<<ngmtrpcfpt10<<std::endl;


  int ngmtrpcb=0, ngmtrpcbpt10=0;
  if (debugRATE) std::cout<< "----- statring ngmt rpcb"<<std::endl;
  std::vector<MatchCSCMuL1::GMTREGCAND> rtGMTRPCbCands;
  max_pt = -1, max_pt_eta = -111;
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
  for ( std::vector<L1MuRegionalCand>::const_iterator trk = l1GmtRPCbCands.begin(); trk != l1GmtRPCbCands.end(); trk++)
    {
      if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
  	{
  	  if (debugRATE) std::cout<<"discarding BX = "<< trk->bx() <<std::endl;
  	  continue;
  	}
      double sign_eta = ( (trk->eta_packed() & 0x20) == 0) ? 1.:-1;
      if (doSelectEtaForGMTRates_ && sign_eta<0) continue;

      bx2n[trk->bx()] += 1;
      MatchCSCMuL1::GMTREGCAND myGMTREGCand;

      myGMTREGCand.init( &*trk , muScales, muPtScale);
      myGMTREGCand.dr = 999.;

      myGMTREGCand.tfcand = NULL;
      rtGMTRPCbCands.push_back(myGMTREGCand);

      ngmtrpcb++;
      if (myGMTREGCand.pt>=10.) ngmtrpcbpt10++;
      h_rt_gmt_rpcb_pt->Fill(myGMTREGCand.pt);
      h_rt_gmt_rpcb_eta->Fill(fabs(myGMTREGCand.eta));
      h_rt_gmt_rpcb_bx->Fill(trk->bx());

      if(myGMTREGCand.pt > max_pt) { max_pt = myGMTREGCand.pt;  max_pt_eta = fabs(myGMTREGCand.eta);}

      h_rt_gmt_rpcb_q->Fill(trk->quality());
    }
  h_rt_ngmt_rpcb->Fill(ngmtrpcb);
  h_rt_ngmt_rpcb_pt10->Fill(ngmtrpcbpt10);
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) h_rt_ngmt_rpcb_per_bx->Fill(bx2n[bx]);
  if (max_pt>0) h_rt_gmt_rpcb_ptmax->Fill(max_pt);
  if (max_pt>=10.) h_rt_gmt_rpcb_ptmax10_eta->Fill(max_pt_eta);
  if (max_pt>=20.) h_rt_gmt_rpcb_ptmax20_eta->Fill(max_pt_eta);
  if (debugRATE) std::cout<< "----- end ngmt rpcb/ngmtpt10="<<ngmtrpcb<<"/"<<ngmtrpcbpt10<<std::endl;


  int ngmtdt=0, ngmtdtpt10=0;
  if (debugRATE) std::cout<< "----- statring ngmt dt"<<std::endl;
  std::vector<MatchCSCMuL1::GMTREGCAND> rtGMTDTCands;
  max_pt = -1, max_pt_eta = -111;
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
  for ( std::vector<L1MuRegionalCand>::const_iterator trk = l1GmtDTCands.begin(); trk != l1GmtDTCands.end(); trk++)
    {
      if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
  	{
  	  if (debugRATE) std::cout<<"discarding BX = "<< trk->bx() <<std::endl;
  	  continue;
  	}
      double sign_eta = ( (trk->eta_packed() & 0x20) == 0) ? 1.:-1;
      if (doSelectEtaForGMTRates_ && sign_eta<0) continue;

      bx2n[trk->bx()] += 1;
      MatchCSCMuL1::GMTREGCAND myGMTREGCand;

      myGMTREGCand.init( &*trk , muScales, muPtScale);
      myGMTREGCand.dr = 999.;

      myGMTREGCand.tfcand = NULL;
      rtGMTDTCands.push_back(myGMTREGCand);

      ngmtdt++;
      if (myGMTREGCand.pt>=10.) ngmtdtpt10++;
      h_rt_gmt_dt_pt->Fill(myGMTREGCand.pt);
      h_rt_gmt_dt_eta->Fill(fabs(myGMTREGCand.eta));
      h_rt_gmt_dt_bx->Fill(trk->bx());

      if(myGMTREGCand.pt > max_pt) { max_pt = myGMTREGCand.pt;  max_pt_eta = fabs(myGMTREGCand.eta);}

      h_rt_gmt_dt_q->Fill(trk->quality());
    }
  h_rt_ngmt_dt->Fill(ngmtdt);
  h_rt_ngmt_dt_pt10->Fill(ngmtdtpt10);
  for (int bx=minRateBX_; bx<=maxRateBX_; bx++) h_rt_ngmt_dt_per_bx->Fill(bx2n[bx]);
  if (max_pt>0) h_rt_gmt_dt_ptmax->Fill(max_pt);
  if (max_pt>=10.) h_rt_gmt_dt_ptmax10_eta->Fill(max_pt_eta);
  if (max_pt>=20.) h_rt_gmt_dt_ptmax20_eta->Fill(max_pt_eta);
  if (debugRATE) std::cout<< "----- end ngmt dt/ngmtpt10="<<ngmtdt<<"/"<<ngmtdtpt10<<std::endl;


  //============ RATE GMT ==================

  int ngmt=0;
  if (debugRATE) std::cout<< "----- statring ngmt"<<std::endl;
  std::vector<MatchCSCMuL1::GMTCAND> rtGMTCands;
  max_pt_me42_2s = -1; max_pt_me42_3s = -1;  max_pt_me42_2q = -1; max_pt_me42_3q = -1;
  max_pt_me42r_2s = -1; max_pt_me42r_3s = -1;  max_pt_me42r_2q = -1; max_pt_me42r_3q = -1;
  float max_pt_me42_2s_sing = -1, max_pt_me42_3s_sing = -1, max_pt_me42_2q_sing = -1, max_pt_me42_3q_sing = -1;
  float max_pt_me42r_2s_sing = -1, max_pt_me42r_3s_sing = -1, max_pt_me42r_2q_sing = -1, max_pt_me42r_3q_sing = -1;
  max_pt = -1, max_pt_eta = -999;

  float max_pt_sing = -1, max_pt_eta_sing = -999, max_pt_sing_3s = -1, max_pt_eta_sing_3s = -999;
  float max_pt_sing_csc = -1., max_pt_eta_sing_csc = -999.;
  float max_pt_sing_dtcsc = -1., max_pt_eta_sing_dtcsc = -999.;
  float max_pt_sing_1b = -1.;//, max_pt_eta_sing_1b = -999;
  float max_pt_sing_no1a = -1.;//, max_pt_eta_sing_no1a = -999.;

  float max_pt_sing6 = -1, max_pt_eta_sing6 = -999, max_pt_sing6_3s = -1, max_pt_eta_sing6_3s = -999;
  float max_pt_sing6_csc = -1., max_pt_eta_sing6_csc = -999.;
  float max_pt_sing6_1b = -1.;//, max_pt_eta_sing6_1b = -999;
  float max_pt_sing6_no1a = -1.;//, max_pt_eta_sing6_no1a = -999.;
  float max_pt_sing6_3s1b_no1a = -1.;//, max_pt_eta_sing6_3s1b_no1a = -999.;

  float max_pt_dbl = -1, max_pt_eta_dbl = -999;

  std::vector<L1MuGMTReadoutRecord> gmt_records = hl1GmtCands->getRecords();
  for ( std::vector< L1MuGMTReadoutRecord >::const_iterator rItr=gmt_records.begin(); rItr!=gmt_records.end() ; ++rItr )
  {
    if (rItr->getBxInEvent() < minBxGMT_ || rItr->getBxInEvent() > maxBxGMT_) continue;
    
    std::vector<L1MuRegionalCand> CSCCands = rItr->getCSCCands();
    std::vector<L1MuRegionalCand> DTCands  = rItr->getDTBXCands();
    std::vector<L1MuRegionalCand> RPCfCands = rItr->getFwdRPCCands();
    std::vector<L1MuRegionalCand> RPCbCands = rItr->getBrlRPCCands();
    std::vector<L1MuGMTExtendedCand> GMTCands = rItr->getGMTCands();
    for ( std::vector<L1MuGMTExtendedCand>::const_iterator  muItr = GMTCands.begin() ; muItr != GMTCands.end() ; ++muItr )
    {
      if( muItr->empty() ) continue;
      
      if ( muItr->bx() < minRateBX_ || muItr->bx() > maxRateBX_ )
      {
	if (debugRATE) std::cout<<"discarding BX = "<< muItr->bx() <<std::endl;
	continue;
      }
      
      MatchCSCMuL1::GMTCAND myGMTCand;
      myGMTCand.init( &*muItr , muScales, muPtScale);
      myGMTCand.dr = 999.;
      if (doSelectEtaForGMTRates_ && myGMTCand.eta<0) continue;
      
      myGMTCand.regcand = NULL;
      myGMTCand.regcand_rpc = NULL;
      
      float gpt = myGMTCand.pt;
      float geta = fabs(myGMTCand.eta);
      
      MatchCSCMuL1::GMTREGCAND * gmt_csc = NULL;
      if (muItr->isFwd() && ( muItr->isMatchedCand() || !muItr->isRPC())) 
      {
	L1MuRegionalCand rcsc = CSCCands[muItr->getDTCSCIndex()];
	unsigned my_i = 999;
	for (unsigned i=0; i< rtGMTREGCands.size(); i++)
	{
	  if (rcsc.getDataWord()!=rtGMTREGCands[i].l1reg->getDataWord()) continue;
	  my_i = i;
	  break;
	}
	if (my_i<99) gmt_csc = &rtGMTREGCands[my_i];
	else std::cout<<"DOES NOT EXIST IN rtGMTREGCands! Should not happen!"<<std::endl;
	myGMTCand.regcand = gmt_csc;
	myGMTCand.ids = gmt_csc->ids;
      }
      
      MatchCSCMuL1::GMTREGCAND * gmt_rpcf = NULL;
      if (muItr->isFwd() && (muItr->isMatchedCand() || muItr->isRPC())) 
      {
	L1MuRegionalCand rrpcf = RPCfCands[muItr->getRPCIndex()];
	unsigned my_i = 999;
	for (unsigned i=0; i< rtGMTRPCfCands.size(); i++)
	{
	  if (rrpcf.getDataWord()!=rtGMTRPCfCands[i].l1reg->getDataWord()) continue;
	  my_i = i;
	  break;
	}
	if (my_i<99) gmt_rpcf = &rtGMTRPCfCands[my_i];
	else std::cout<<"DOES NOT EXIST IN rtGMTRPCfCands! Should not happen!"<<std::endl;
	myGMTCand.regcand_rpc = gmt_rpcf;
      }
      
      MatchCSCMuL1::GMTREGCAND * gmt_rpcb = NULL;
      if (!(muItr->isFwd()) && (muItr->isMatchedCand() || muItr->isRPC()))
      {
	L1MuRegionalCand rrpcb = RPCbCands[muItr->getRPCIndex()];
	unsigned my_i = 999;
	for (unsigned i=0; i< rtGMTRPCbCands.size(); i++)
	{
	  if (rrpcb.getDataWord()!=rtGMTRPCbCands[i].l1reg->getDataWord()) continue;
	  my_i = i;
	  break;
	}
	if (my_i<99) gmt_rpcb = &rtGMTRPCbCands[my_i];
	else std::cout<<"DOES NOT EXIST IN rtGMTRPCbCands! Should not happen!"<<std::endl;
	myGMTCand.regcand_rpc = gmt_rpcb;
      }
      
      MatchCSCMuL1::GMTREGCAND * gmt_dt = NULL;
      if (!(muItr->isFwd()) && (muItr->isMatchedCand() || !(muItr->isRPC())))
      {
	L1MuRegionalCand rdt = DTCands[muItr->getDTCSCIndex()];
	unsigned my_i = 999;
	for (unsigned i=0; i< rtGMTDTCands.size(); i++)
	  {
	    if (rdt.getDataWord()!=rtGMTDTCands[i].l1reg->getDataWord()) continue;
	    my_i = i;
	    break;
	  }
	if (my_i<99) gmt_dt = &rtGMTDTCands[my_i];
	else std::cout<<"DOES NOT EXIST IN rtGMTDTCands! Should not happen!"<<std::endl;
	myGMTCand.regcand = gmt_dt;
      }
      
      if ( (gmt_csc != NULL && gmt_rpcf != NULL) && !muItr->isMatchedCand() ) std::cout<<"csc&rpcf but not matched!"<<std::endl;
      
      bool eta_me42 = etaRangeHelpers::isME42EtaRegion(myGMTCand.eta);
      bool eta_me42r = etaRangeHelpers::isME42RPCEtaRegion(myGMTCand.eta);
      //if (geta>=1.2 && geta<=1.8) eta_me42 = 1;
      bool eta_q = (geta > 1.2);
      
      bool eta_me1b = etaRangeHelpers::isME1bEtaRegion(myGMTCand.eta);
      //bool eta_me1b_whole = etaRangeHelpers::isME1bEtaRegion(myGMTCand.eta, 1.6, 2.14);
      bool eta_no1a = (geta >= 1.2 && geta < 2.14);
      //bool eta_csc = (geta > 0.9);
      //
      
      size_t n_stubs = 0;
      if (gmt_csc) n_stubs = gmt_csc->nTFStubs;
      
      bool has_me1_stub = false;
      if (gmt_csc && gmt_csc->tfcand && gmt_csc->tfcand->tftrack)
      {
	has_me1_stub = gmt_csc->tfcand->tftrack->hasStub(1);
      }
      
      
      if (eta_me42) h_rt_gmt_gq_42->Fill(muItr->quality());
      if (eta_me42r) {
	int gtype = 0;
	if (muItr->isMatchedCand()) gtype = 6;
	else if (gmt_csc!=0) gtype = gmt_csc->l1reg->quality()+2;
	else if (gmt_rpcf!=0) gtype = gmt_rpcf->l1reg->quality()+1;
	if (gtype==0) std::cout<<"weird: gtype=0 That shouldn't happen!";
	h_rt_gmt_gq_vs_type_42r->Fill(muItr->quality(), gtype);
	h_rt_gmt_gq_vs_pt_42r->Fill(muItr->quality(), gpt);
	h_rt_gmt_gq_42r->Fill(muItr->quality());
      }
      h_rt_gmt_gq->Fill(muItr->quality());
      
      h_rt_gmt_bx->Fill(muItr->bx());
      
      //if (muItr->quality()<4) continue; // not good for single muon trigger!
      
      bool isSingleTrigOk = muItr->useInSingleMuonTrigger(); // good for single trigger
      bool isDoubleTrigOk = muItr->useInDiMuonTrigger(); // good for single trigger
      
      bool isSingle6TrigOk = (muItr->quality() >= 6); // unmatched or matched CSC or DT
      
      if (muItr->quality()<3) continue; // not good for neither single nor dimuon triggers
      
      bool isCSC = (gmt_csc != NULL);
      bool isDT  = (gmt_dt  != NULL);
      bool isRPCf = (gmt_rpcf != NULL);
      bool isRPCb = (gmt_rpcb != NULL);
      
      if (isCSC && gmt_csc->tfcand != NULL && gmt_csc->tfcand->tftrack == NULL) std::cout<<"warning: gmt_csc->tfcand->tftrack == NULL"<<std::endl;
      if (isCSC && gmt_csc->tfcand != NULL && gmt_csc->tfcand->tftrack != NULL && gmt_csc->tfcand->tftrack->l1trk == NULL)
	std::cout<<"warning: gmt_csc->tfcand->tftrack->l1trk == NULL"<<std::endl;
      //bool isCSC2s = (isCSC && gmt_csc->tfcand != NULL && myGMTCand.ids.size()>=2);
      //bool isCSC3s = (isCSC && gmt_csc->tfcand != NULL && myGMTCand.ids.size()>=3);
      bool isCSC2s = (isCSC && gmt_csc->tfcand != NULL && gmt_csc->tfcand->tftrack != NULL && gmt_csc->tfcand->tftrack->nStubs()>=2);
      bool isCSC3s = (isCSC && gmt_csc->tfcand != NULL && gmt_csc->tfcand->tftrack != NULL
		      && ( (!eta_q && isCSC2s) || (eta_q && gmt_csc->tfcand->tftrack->nStubs()>=3) ) );
      bool isCSC2q = (isCSC && gmt_csc->l1reg != NULL && gmt_csc->l1reg->quality()>=2);
      bool isCSC3q = (isCSC && gmt_csc->l1reg != NULL
		      && ( (!eta_q && isCSC2q) || (eta_q && gmt_csc->l1reg->quality()>=3) ) );
      
      myGMTCand.isCSC = isCSC;
      myGMTCand.isDT = isDT;
      myGMTCand.isRPCf = isRPCf;
      myGMTCand.isRPCb = isRPCb;
      myGMTCand.isCSC2s = isCSC2s;
      myGMTCand.isCSC3s = isCSC3s;
      myGMTCand.isCSC2q = isCSC2q;
      myGMTCand.isCSC3q = isCSC3q;
      
      rtGMTCands.push_back(myGMTCand);
      
      
      if (isCSC2q || isRPCf) {
	h_rt_gmt_pt_2q->Fill(gpt);
	if (eta_me42) {
	  h_rt_gmt_pt_2q42->Fill(gpt);
	  if (gpt > max_pt_me42_2q) max_pt_me42_2q = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42_2q_sing) max_pt_me42_2q_sing = gpt;
	}
	if (eta_me42r) {
	  h_rt_gmt_pt_2q42r->Fill(gpt);
	  if (gpt > max_pt_me42r_2q) max_pt_me42r_2q = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42r_2q_sing) max_pt_me42r_2q_sing = gpt;
	}
      }
      if (isCSC3q || isRPCf) {
	h_rt_gmt_pt_3q->Fill(gpt);
	if (eta_me42) {
	  h_rt_gmt_pt_3q42->Fill(gpt);
	  if (gpt > max_pt_me42_3q) max_pt_me42_3q = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42_3q_sing) max_pt_me42_3q_sing = gpt;
	}
	if (eta_me42r) {
	  h_rt_gmt_pt_3q42r->Fill(gpt);
	  if (gpt > max_pt_me42r_3q) max_pt_me42r_3q = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42r_3q_sing) max_pt_me42r_3q_sing = gpt;
	}
      }

      if (isCSC2s || isRPCf) {
	h_rt_gmt_pt_2st->Fill(gpt);
	if (eta_me42) {
	  h_rt_gmt_pt_2s42->Fill(gpt);
	  if (gpt > max_pt_me42_2s) max_pt_me42_2s = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42_2s_sing) max_pt_me42_2s_sing = gpt;
	}
	if (eta_me42r) {
	  h_rt_gmt_pt_2s42r->Fill(gpt);
	  if (gpt > max_pt_me42r_2s) max_pt_me42r_2s = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42r_2s_sing) max_pt_me42r_2s_sing = gpt;
	}
      }
      if (isCSC3s || isRPCf) {
	h_rt_gmt_pt_3st->Fill(gpt);
	if (eta_me42) {
	  h_rt_gmt_pt_3s42->Fill(gpt);
	  if (gpt > max_pt_me42_3s) max_pt_me42_3s = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42_3s_sing) max_pt_me42_3s_sing = gpt;
	}
	if (eta_me42r) {
	  h_rt_gmt_pt_3s42r->Fill(gpt);
	  if (gpt > max_pt_me42r_3s) max_pt_me42r_3s = gpt;
	  if (isSingleTrigOk && gpt > max_pt_me42r_3s_sing) max_pt_me42r_3s_sing = gpt;
	}
      }

      ngmt++;
      h_rt_gmt_pt->Fill(gpt);
      h_rt_gmt_eta->Fill(geta);
      if (gpt > max_pt) {max_pt = gpt; max_pt_eta = geta;}
      if (isDoubleTrigOk && gpt > max_pt_dbl) {max_pt_dbl = gpt; max_pt_eta_dbl = geta;}
      if (isSingleTrigOk)
	{
	  if (            gpt > max_pt_sing     ) { max_pt_sing = gpt;     max_pt_eta_sing = geta;}
	  if (isCSC    && gpt > max_pt_sing_csc ) { max_pt_sing_csc = gpt; max_pt_eta_sing_csc = geta; }
	  if ((isCSC||isDT) && gpt > max_pt_sing_dtcsc ) { max_pt_sing_dtcsc = gpt; max_pt_eta_sing_dtcsc = geta; }
	  if (gpt > max_pt_sing_3s && ( !isCSC || isCSC3s ) ) {max_pt_sing_3s = gpt; max_pt_eta_sing_3s = geta;}
	  if (eta_me1b && gpt > max_pt_sing_1b  ) { max_pt_sing_1b = gpt; /*max_pt_eta_sing_1b = geta;*/ }
	  if (eta_no1a && gpt > max_pt_sing_no1a) { max_pt_sing_no1a = gpt; /*max_pt_eta_sing_no1a = geta;*/ }
	}
      if (isSingle6TrigOk)
	{
	  if (            gpt > max_pt_sing6     ) { max_pt_sing6 = gpt;     max_pt_eta_sing6 = geta;}
	  if (isCSC    && gpt > max_pt_sing6_csc ) { max_pt_sing6_csc = gpt; max_pt_eta_sing6_csc = geta; }
	  if (gpt > max_pt_sing6_3s && ( !isCSC || isCSC3s ) ) {max_pt_sing6_3s = gpt; max_pt_eta_sing6_3s = geta;}
	  if (eta_me1b && gpt > max_pt_sing6_1b  ) { max_pt_sing6_1b = gpt; /*max_pt_eta_sing6_1b = geta;*/ }
	  if (eta_no1a && gpt > max_pt_sing6_no1a) { max_pt_sing6_no1a = gpt; /*max_pt_eta_sing6_no1a = geta;*/ }
	  if (eta_no1a && gpt > max_pt_sing6_3s1b_no1a && 
	      (!eta_me1b  || (eta_me1b && has_me1_stub && n_stubs >=3) ) ) { max_pt_sing6_3s1b_no1a = gpt; /*max_pt_eta_sing6_no1a = geta;*/ }
	}
    }
  }
  h_rt_ngmt->Fill(ngmt);
  if (max_pt_me42_2s>0) h_rt_gmt_ptmax_2s42->Fill(max_pt_me42_2s);
  if (max_pt_me42_3s>0) h_rt_gmt_ptmax_3s42->Fill(max_pt_me42_3s);
  if (max_pt_me42_2q>0) h_rt_gmt_ptmax_2q42->Fill(max_pt_me42_2q);
  if (max_pt_me42_3q>0) h_rt_gmt_ptmax_3q42->Fill(max_pt_me42_3q);
  if (max_pt_me42_2s_sing>0) h_rt_gmt_ptmax_2s42_sing->Fill(max_pt_me42_2s_sing);
  if (max_pt_me42_3s_sing>0) h_rt_gmt_ptmax_3s42_sing->Fill(max_pt_me42_3s_sing);
  if (max_pt_me42_2q_sing>0) h_rt_gmt_ptmax_2q42_sing->Fill(max_pt_me42_2q_sing);
  if (max_pt_me42_3q_sing>0) h_rt_gmt_ptmax_3q42_sing->Fill(max_pt_me42_3q_sing);
  if (max_pt_me42r_2s>0) h_rt_gmt_ptmax_2s42r->Fill(max_pt_me42r_2s);
  if (max_pt_me42r_3s>0) h_rt_gmt_ptmax_3s42r->Fill(max_pt_me42r_3s);
  if (max_pt_me42r_2q>0) h_rt_gmt_ptmax_2q42r->Fill(max_pt_me42r_2q);
  if (max_pt_me42r_3q>0) h_rt_gmt_ptmax_3q42r->Fill(max_pt_me42r_3q);
  if (max_pt_me42r_2s_sing>0) h_rt_gmt_ptmax_2s42r_sing->Fill(max_pt_me42r_2s_sing);
  if (max_pt_me42r_3s_sing>0) h_rt_gmt_ptmax_3s42r_sing->Fill(max_pt_me42r_3s_sing);
  if (max_pt_me42r_2q_sing>0) h_rt_gmt_ptmax_2q42r_sing->Fill(max_pt_me42r_2q_sing);
  if (max_pt_me42r_3q_sing>0) h_rt_gmt_ptmax_3q42r_sing->Fill(max_pt_me42r_3q_sing);
  if (max_pt>0) h_rt_gmt_ptmax->Fill(max_pt);
  if (max_pt>=10.) h_rt_gmt_ptmax10_eta->Fill(max_pt_eta);
  if (max_pt>=20.) h_rt_gmt_ptmax20_eta->Fill(max_pt_eta);

  if (max_pt_sing>0) h_rt_gmt_ptmax_sing->Fill(max_pt_sing);
  if (max_pt_sing_3s>0) h_rt_gmt_ptmax_sing_3s->Fill(max_pt_sing_3s);
  if (max_pt_sing>=10.) h_rt_gmt_ptmax10_eta_sing->Fill(max_pt_eta_sing);
  if (max_pt_sing_3s>=10.) h_rt_gmt_ptmax10_eta_sing_3s->Fill(max_pt_eta_sing_3s);
  if (max_pt_sing>=20.) h_rt_gmt_ptmax20_eta_sing->Fill(max_pt_eta_sing);
  if (max_pt_sing_csc>=20.) h_rt_gmt_ptmax20_eta_sing_csc->Fill(max_pt_eta_sing_csc);
  if (max_pt_sing_dtcsc>=20.) h_rt_gmt_ptmax20_eta_sing_dtcsc->Fill(max_pt_eta_sing_dtcsc);
  if (max_pt_sing_3s>=20.) h_rt_gmt_ptmax20_eta_sing_3s->Fill(max_pt_eta_sing_3s);
  if (max_pt_sing>=30.) h_rt_gmt_ptmax30_eta_sing->Fill(max_pt_eta_sing);
  if (max_pt_sing_csc>=30.) h_rt_gmt_ptmax30_eta_sing_csc->Fill(max_pt_eta_sing_csc);
  if (max_pt_sing_dtcsc>=30.) h_rt_gmt_ptmax30_eta_sing_dtcsc->Fill(max_pt_eta_sing_dtcsc);
  if (max_pt_sing_3s>=30.) h_rt_gmt_ptmax30_eta_sing_3s->Fill(max_pt_eta_sing_3s);
  if (max_pt_sing_csc > 0.) h_rt_gmt_ptmax_sing_csc->Fill(max_pt_sing_csc);
  if (max_pt_sing_1b > 0. ) h_rt_gmt_ptmax_sing_1b->Fill(max_pt_sing_1b);
  if (max_pt_sing_no1a > 0.) h_rt_gmt_ptmax_sing_no1a->Fill(max_pt_sing_no1a);

  if (max_pt_sing6>0) h_rt_gmt_ptmax_sing6->Fill(max_pt_sing6);
  if (max_pt_sing6_3s>0) h_rt_gmt_ptmax_sing6_3s->Fill(max_pt_sing6_3s);
  if (max_pt_sing6>=10.) h_rt_gmt_ptmax10_eta_sing6->Fill(max_pt_eta_sing6);
  if (max_pt_sing6_3s>=10.) h_rt_gmt_ptmax10_eta_sing6_3s->Fill(max_pt_eta_sing6_3s);
  if (max_pt_sing6>=20.) h_rt_gmt_ptmax20_eta_sing6->Fill(max_pt_eta_sing6);
  if (max_pt_sing6_csc>=20.) h_rt_gmt_ptmax20_eta_sing6_csc->Fill(max_pt_eta_sing6_csc);
  if (max_pt_sing6_3s>=20.) h_rt_gmt_ptmax20_eta_sing6_3s->Fill(max_pt_eta_sing6_3s);
  if (max_pt_sing6>=30.) h_rt_gmt_ptmax30_eta_sing6->Fill(max_pt_eta_sing6);
  if (max_pt_sing6_csc>=30.) h_rt_gmt_ptmax30_eta_sing6_csc->Fill(max_pt_eta_sing6_csc);
  if (max_pt_sing6_3s>=30.) h_rt_gmt_ptmax30_eta_sing6_3s->Fill(max_pt_eta_sing6_3s);
  if (max_pt_sing6_csc > 0.) h_rt_gmt_ptmax_sing6_csc->Fill(max_pt_sing6_csc);
  if (max_pt_sing6_1b > 0. ) h_rt_gmt_ptmax_sing6_1b->Fill(max_pt_sing6_1b);
  if (max_pt_sing6_no1a > 0.) h_rt_gmt_ptmax_sing6_no1a->Fill(max_pt_sing6_no1a);
  if (max_pt_sing6_3s1b_no1a > 0.) h_rt_gmt_ptmax_sing6_3s1b_no1a->Fill(max_pt_sing6_3s1b_no1a);

  if (max_pt_dbl>0) h_rt_gmt_ptmax_dbl->Fill(max_pt_dbl);
  if (max_pt_dbl>=10.) h_rt_gmt_ptmax10_eta_dbl->Fill(max_pt_eta_dbl);
  if (max_pt_dbl>=20.) h_rt_gmt_ptmax20_eta_dbl->Fill(max_pt_eta_dbl);
  if (debugRATE) std::cout<< "----- end ngmt="<<ngmt<<std::endl;
}

// ================================================================================================
void  
GEMCSCTriggerRate::bookALCTTree()
{
  edm::Service< TFileService > fs;
  alct_tree_ = fs->make<TTree>("ALCTs", "ALCTs");
  alct_tree_->Branch("event",&alct_.event);
  alct_tree_->Branch("detId",&alct_.detId);
  alct_tree_->Branch("pt",&alct_.pt);
  alct_tree_->Branch("eta",&alct_.eta);
  alct_tree_->Branch("phi",&alct_.phi);
  alct_tree_->Branch("bx",&alct_.bx);
}

// ================================================================================================
void  
GEMCSCTriggerRate::bookCLCTTree()
{
  edm::Service< TFileService > fs;
  clct_tree_ = fs->make<TTree>("CLCTs", "CLCTs");
  clct_tree_->Branch("event",&clct_.event);
  clct_tree_->Branch("detId",&clct_.detId);
  clct_tree_->Branch("pt",&clct_.pt);
  clct_tree_->Branch("eta",&clct_.eta);
  clct_tree_->Branch("phi",&clct_.phi);
  clct_tree_->Branch("bx",&clct_.bx);
}

// ================================================================================================
void  
GEMCSCTriggerRate::bookLCTTree()
{
  // edm::Service< TFileService > fs;
  // lct_tree_ = fs->make<TTree>("LCTs", "LCTs");
  // lct_tree_->Branch("event",&lct_.event);
  // lct_tree_->Branch("detId",&lct_.detId);
  // lct_tree_->Branch("pt",&lct_.pt);
  // lct_tree_->Branch("eta",&lct_.eta);
  // lct_tree_->Branch("phi",&lct_.phi);
  // lct_tree_->Branch("bx",&lct_.bx);
}

// ================================================================================================
void  
GEMCSCTriggerRate::bookMPCLCTTree()
{
  // edm::Service< TFileService > fs;
  // mpclct_tree_ = fs->make<TTree>("MPCLCTs", "MPCLCTs");
  // mpclct_tree_->Branch("event",&mpclct_.event);
  // mpclct_tree_->Branch("detId",&mpclct_.detId);
  // mpclct_tree_->Branch("pt",&mpclct_.pt);
  // mpclct_tree_->Branch("eta",&mpclct_.eta);
  // mpclct_tree_->Branch("phi",&mpclct_.phi);
  // mpclct_tree_->Branch("bx",&mpclct_.bx);
}

// ================================================================================================
void  
GEMCSCTriggerRate::bookTFTrackTree()
{
}

// ================================================================================================
void  
GEMCSCTriggerRate::bookTFCandTree()
{
}

// ================================================================================================
void  
GEMCSCTriggerRate::bookGMTRegionalTree()
{
}

// ================================================================================================
void  
GEMCSCTriggerRate::bookGMTCandTree()
{
}

// ================================================================================================
void  
GEMCSCTriggerRate::analyzeALCTRate(const edm::Event& iEvent)
{
  edm::Handle< CSCALCTDigiCollection > halcts;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  halcts);
  const CSCALCTDigiCollection* alcts = halcts.product();

  int nalct=0;
  int nalct_per_bx[16];
  int n_ch_alct_per_bx[16];
  int n_ch_alct_per_bx_st[MAX_STATIONS][16];
  int n_ch_alct_per_bx_cscdet[CSC_TYPES+1][16];
  for (int b=0;b<16;b++)
  {
    nalct_per_bx[b] = n_ch_alct_per_bx[b] = 0;
    for (int s=0; s<MAX_STATIONS; s++) n_ch_alct_per_bx_st[s][b]=0;
    for (int me=0; me<=CSC_TYPES; me++) n_ch_alct_per_bx_cscdet[me][b]=0;
  }
  if (debugRATE) std::cout<< "----- statring nalct"<<std::endl;

  std::map< int , std::vector<const CSCALCTDigi*> > me11alcts;
  for (CSCALCTDigiCollection::DigiRangeIterator  adetUnitIt = alcts->begin(); adetUnitIt != alcts->end(); adetUnitIt++)
  {
    const CSCDetId& id = (*adetUnitIt).first;
    //if (id.endcap() != 1) continue;
    CSCDetId idd(id.rawId());
    int csct = getCSCType( idd );
    int cscst = getCSCSpecsType( idd );
    //int is11 = isME11(csct);
    int nalct_per_ch_bx[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    const CSCALCTDigiCollection::Range& range = (*adetUnitIt).second;
    for (CSCALCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
    {
      if ((*digiIt).isValid()) 
      {o
	int bx = (*digiIt).getBX();
	//if ( bx-6 < minBX_ || bx-6 > maxBX_ )
	if ( bx < minBxALCT_ || bx > maxBxALCT_ )
	{
	  if (debugRATE) std::cout<<"discarding BX = "<< bx-6 <<std::endl;
	  continue;
	}
	
	// store all ME11 alcts together so we can look at them later
	// take into acount that 10<=WG<=15 alcts are present in both 1a and 1b
	if (csct==0) me11alcts[idd.rawId()].push_back(&(*digiIt));
	if (csct==3 && (*digiIt).getKeyWG() < 10) 
        {
	  CSCDetId id11(idd.endcap(),1,1,idd.chamber());
	  me11alcts[id11.rawId()].push_back(&(*digiIt));
	}
	
	//        if (debugALCT) std::cout<<"raw ID "<<id.rawId()<<" "<<id<<"    NTrackHitsInChamber  nmhits  alctInfo.size  diff  " 
	//                           <<trackHitsInChamber.size()<<" "<<nmhits<<" "<<alctInfo.size()<<"  "
	//                           << nmhits-alctInfo.size() <<std::endl 
	//                           << "  "<<(*digiIt)<<std::endl;
	nalct++;
	++nalct_per_bx[bx];
	++nalct_per_ch_bx[bx];
	h_rt_alct_bx->Fill( bx - 6 );
	h_rt_alct_bx_cscdet[csct]->Fill( bx - 6 );
	if (bx>=5 && bx<=7) h_rt_csctype_alct_bx567->Fill(cscst);
	
      } //if (alct_valid) 
    }
    for (int b=0;b<16;b++) 
    {
      if ( b < minBxALCT_ || b > maxBxALCT_ ) continue;
      h_rt_n_per_ch_alct_vs_bx_cscdet[csct]->Fill(nalct_per_ch_bx[b],b);
      if (nalct_per_ch_bx[b]>0) 
      {
	++n_ch_alct_per_bx[b];
	++n_ch_alct_per_bx_st[id.station()-1][b];
	++n_ch_alct_per_bx_cscdet[csct][b];
      }
    }
  } // loop CSCALCTDigiCollection
  //std::map< CSCDetId , std::vector<const CSCALCTDigi*> >::const_iterator mapIt = me11alcts.begin();
  //for (;mapIt != me11alcts.end(); mapIt++){}
  std::map< int , std::vector<const CSCALCTDigi*> >::const_iterator aMapIt = me11alcts.begin();
  for (;aMapIt != me11alcts.end(); aMapIt++)
  {
    CSCDetId id(aMapIt->first);
    int nalct_per_ch_bx[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (size_t i=0; i<(aMapIt->second).size(); i++)
    {
      int bx = (aMapIt->second)[i]->getBX();
      ++nalct_per_ch_bx[bx];
    }
    for (int b=0;b<16;b++)
    {
      if ( b < minBxALCT_ || b > maxBxALCT_ ) continue;
      h_rt_n_per_ch_alct_vs_bx_cscdet[10]->Fill(nalct_per_ch_bx[b],b);
      if (nalct_per_ch_bx[b]>0) ++n_ch_alct_per_bx_cscdet[10][b];
    }
  }
  h_rt_nalct->Fill(nalct);
  for (int b=0;b<16;b++) 
  {
    if (b < minBxALCT_ || b > maxBxALCT_) continue;
    h_rt_nalct_vs_bx->Fill(nalct_per_bx[b],b);
    h_rt_nalct_per_bx->Fill(nalct_per_bx[b]);
    h_rt_n_ch_alct_per_bx->Fill(n_ch_alct_per_bx[b]);
    for (int s=0; s<MAX_STATIONS; s++) 
      h_rt_n_ch_alct_per_bx_st[s]->Fill(n_ch_alct_per_bx_st[s][b]);
    for (int me=0; me<=CSC_TYPES; me++) 
      h_rt_n_ch_alct_per_bx_cscdet[me]->Fill(n_ch_alct_per_bx_cscdet[me][b]);
  }

  
  if (debugRATE) std::cout<< "----- end nalct="<<nalct<<std::endl;

  // start of the ntuplization
  /*
  // Loop on all ALCTs
  for (auto adetUnitIt& : alcts);
  {
    CSCDetId detId((*adetUnitIt).first);
    if (detId.endcap() != 1) continue;
    auto range = (*adetUnitIt).second;
    // loop on all ALCTs in that detId
    for (auto digiIt& : range)
    {
      const int bx((*digiIt).getBX());
      if (bx < minBxALCT_ || bx > maxBxALCT_)
      {
	if (debugRATE) std::cout<<"discarding BX = "<< bx-6 <<std::endl;
	continue;
      }
      // central bx for CSC is 6!!!
      alct_.event = iEvent.id().event();
      alct_.detId = id;
      alct_.bx = bx - 6;
      alct_.pt = pt();
      alct_.eta = eta();
      alct_.phi = phi();
      alct_.pattern =
      alct_tree_->Fill();
    }
  */
}

// ================================================================================================
void  
GEMCSCTriggerRate::analyzeCLCTRate(const edm::Event& iEvent)
{
  edm::Handle< CSCCLCTDigiCollection > hclcts;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  hclcts);
  const CSCCLCTDigiCollection* clcts = hclcts.product();

  /*
  // Loop on all CLCTs
  for (auto adetUnitIt& : clcts);
  {
    CSCDetId detId((*adetUnitIt).first);
    if (detId.endcap() != 1) continue;
    auto range = (*adetUnitIt).second;
    // loop on all CLCTs in that detId
    for (auto digiIt& : range)
    {
      const int bx((*digiIt).getBX());
      if (bx < minBxCLCT_ || bx > maxBxCLCT_)
      {
	if (debugRATE) std::cout<<"discarding BX = "<< bx-6 <<std::endl;
	continue;
      }
      // central bx for CSC is 6!!!
      clct_.event = iEvent.id().event();
      clct_.detId = id;
      clct_.bx = bx - 6;
      clct_.pt = pt();
      clct_.eta = eta();
      clct_.phi = phi();
      clct_.pattern =
      clct_tree_->Fill();
    }
  */
}

// ================================================================================================
void  
GEMCSCTriggerRate::analyzeLCTRate(const edm::Event& iEvent)
{
  /*
  edm::Handle< CSCCorrelatedLCTDigiCollection > lcts_tmb;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  lcts_tmb);
  const CSCCorrelatedLCTDigiCollection* lcts = lcts_tmb.product();

  for (auto detUnitIt& : lcts);
  {
    CSCDetId detId((*adetUnitIt).first);
    if (detId.endcap() != 1) continue;
    auto range = (*adetUnitIt).second;
    // loop on all CLCTs in that detId
    for (auto digiIt& : range)
    {
      const int bx((*digiIt).getBX());
      if (bx < minBxLCT_ || bx > maxBxLCT_)
      {
	if (debugRATE) std::cout<<"discarding BX = "<< bx-6 <<std::endl;
	continue;
      }
      // central bx for CSC is 6!!!
      lct_.event = iEvent.id().event();
      lct_.detId = id;
      lct_.bx = bx - 6;
      // lct_.pt = pt();
      // lct_.eta = eta();
      // lct_.phi = phi();
      lct_.quality = (*digiIt).getQuality();
      lct_.strip = (*digiIt).getStrip();
      lct_.triggerSector = id.triggerSector();
      lct_.hasGEM = 
      lct_tree_->Fill();
    }
  */
}

// ================================================================================================
void  
GEMCSCTriggerRate::analyzeMPCLCTRate(const edm::Event& iEvent)
{
  /*
  edm::Handle< CSCCorrelatedLCTDigiCollection > lcts_mpc;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis", "MPCSORTED", lcts_mpc);
  const CSCCorrelatedLCTDigiCollection* mpclcts = lcts_mpc.product();

  for (auto detUnitIt& : mpclcts);
  {
    CSCDetId detId((*adetUnitIt).first);
    if (detId.endcap() != 1) continue;
    auto range = (*adetUnitIt).second;
    // loop on all CMpclcts in that detId
    for (auto digiIt& : range)
    {
      const int bx((*digiIt).getBX());
      if (bx < minBxMPCLCT_ || bx > maxBxMPCLCT_)
      {
	if (debugRATE) std::cout<<"discarding BX = "<< bx-6 <<std::endl;
	continue;
      }
       // central bx for CSC is 6!!!
      mpclct_.event = iEvent.id().event();
      mpclct_.detId = id;
      mpclct_.bx = bx - 6;
      // mpclct_.pt = pt();
      // mpclct_.eta = eta();
      // mpclct_.phi = phi();
      // mpclct_.lut_eta = eta();
      // mpclct_.lut_phi = phi();
      mpclct_.quality = (*digiIt).getQuality();
      mpclct_.strip = (*digiIt).getStrip();
      mpclct_.keyWG = (*digiIt).getKeyWG()
      mpclct_.triggerSector = id.triggerSector();
      mpclct_.hasGEM = ;
      // auto etaphi = intersectionEtaPhi(id, (*digiIt).getKeyWG(), (*digiIt).getStrip());
      // //float eta_lut = muScales->getRegionalEtaScale(2)->getCenter(gblEta.global_eta);
      // //float phi_lut = normalizedPhi( muScales->getPhiScale()->getLowEdge(gblPhi.global_phi));
      // csctf::TrackStub stub = buildTrackStub((*digiIt), id);
      // float eta_lut = stub.etaValue();
      // float phi_lut = stub.phiValue();
  		  // std::cout<<"DBGSRLUT "<<id.endcap()<<" "<<id.station()<<" "<<id.ring()<<" "<<id.chamber()<<"  "<<(*digiIt).getKeyWG()<<" "<<(*digiIt).getStrip()<<"  "<<etaphi.first<<" "<<etaphi.second<<"  "<<eta_lut<<" "<<phi_lut<<"  "<<etaphi.first - eta_lut<<" "<<deltaPhi(etaphi.second, phi_lut)<<std::endl;
      
      mpclct_.
      mpclct_tree_->Fill();
    }
  }
*/
}

// ================================================================================================
void  
GEMCSCTriggerRate::analyzeTFTrackRate(const edm::Event& iEvent)
{
}

// ================================================================================================
void  
GEMCSCTriggerRate::analyzeTFCandRate(const edm::Event& iEvent)
{
}

// ================================================================================================
void  
GEMCSCTriggerRate::analyzeGMTCandRate(const edm::Event& iEvent)
{
}




// ================================================================================================
void 
GEMCSCTriggerRate::runCSCTFSP(const CSCCorrelatedLCTDigiCollection* mplcts, const L1MuDTChambPhContainer* dttrig)
   //, L1CSCTrackCollection*, CSCTriggerContainer<csctf::TrackStub>*)
{
// Just run it for the sake of its debug printout, do not return any results

  // Create csctf::TrackStubs collection from MPC LCTs
  CSCTriggerContainer<csctf::TrackStub> stub_list;
  CSCCorrelatedLCTDigiCollection::DigiRangeIterator Citer;
  for(Citer = mplcts->begin(); Citer != mplcts->end(); Citer++)
  {
    CSCCorrelatedLCTDigiCollection::const_iterator Diter = (*Citer).second.first;
    CSCCorrelatedLCTDigiCollection::const_iterator Dend = (*Citer).second.second;
    for(; Diter != Dend; Diter++)
    {
      csctf::TrackStub theStub((*Diter),(*Citer).first);
      stub_list.push_back(theStub);
    }
  }
  
  // Now we append the track stubs the the DT Sector Collector
  // after processing from the DT Receiver.
  CSCTriggerContainer<csctf::TrackStub> dtstubs = my_dtrc->process(dttrig);
  stub_list.push_many(dtstubs);
  
  //for(int e=0; e<2; e++) for (int s=0; s<6; s++) {
  int e=0;
  for (int s=0; s<6; s++) 
  {
    CSCTriggerContainer<csctf::TrackStub> current_e_s = stub_list.get(e+1, s+1);
    if (current_e_s.get().size()>0) 
    {
      std::cout<<"sector "<<s+1<<":"<<std::endl<<std::endl;
      my_SPs[e][s]->run(current_e_s);
    }
  }
}

// ================================================================================================
// Returns chamber type (0-9) according to the station and ring number
int 
GEMCSCTriggerRate::getCSCType(CSCDetId &id) 
{
  int type = -999;

  if (id.station() == 1) {
    type = (id.triggerCscId()-1)/3;
    if (id.ring() == 4) {
      type = 3;
    }
  }
  else { // stations 2-4
    type = 3 + id.ring() + 2*(id.station()-2);
  }
  assert(type >= 0 && type < CSC_TYPES); // include ME4/2
  return type;
}

// ================================================================================================
int
GEMCSCTriggerRate::isME11(int t)
{
  if (t==0 || t==3) return CSC_TYPES;
  return 0;
}

// ================================================================================================
// Returns chamber type (0-9) according to CSCChamberSpecs type
// 1..10 -> 1/a, 1/b, 1/2, 1/3, 2/1...
int
GEMCSCTriggerRate::getCSCSpecsType(CSCDetId &id)
{
  return cscGeometry->chamber(id)->specs()->chamberType();
}

// ================================================================================================
int
GEMCSCTriggerRate::cscTriggerSubsector(CSCDetId &id)
{
  if(id.station() != 1) return 0; // only station one has subsectors
  int chamber = id.chamber();
  switch(chamber) // first make things easier to deal with
  {
    case 1:
      chamber = 36;
      break;
    case 2:
      chamber = 35;
      break;
    default:
      chamber -= 2;
  }
  chamber = ((chamber-1)%6) + 1; // renumber all chambers to 1-6
  return ((chamber-1) / 3) + 1; // [1,3] -> 1 , [4,6]->2
}


// ================================================================================================
void 
GEMCSCTriggerRate::setupTFModeHisto(TH1D* h)
{
  if (h==0) return;
  if (h->GetXaxis()->GetNbins()<16) {
    std::cout<<"TF mode histogram should have 16 bins, nbins="<<h->GetXaxis()->GetNbins()<<std::endl;
    return;
  }
  h->GetXaxis()->SetTitle("Track Type");
  h->GetXaxis()->SetTitleOffset(1.2);
  h->GetXaxis()->SetBinLabel(1,"No Track");
  h->GetXaxis()->SetBinLabel(2,"Bad Phi Road");
  h->GetXaxis()->SetBinLabel(3,"ME1-2-3(-4)");
  h->GetXaxis()->SetBinLabel(4,"ME1-2-4");
  h->GetXaxis()->SetBinLabel(5,"ME1-3-4");
  h->GetXaxis()->SetBinLabel(6,"ME2-3-4");
  h->GetXaxis()->SetBinLabel(7,"ME1-2");
  h->GetXaxis()->SetBinLabel(8,"ME1-3");
  h->GetXaxis()->SetBinLabel(9,"ME2-3");
  h->GetXaxis()->SetBinLabel(10,"ME2-4");
  h->GetXaxis()->SetBinLabel(11,"ME3-4");
  h->GetXaxis()->SetBinLabel(12,"B1-ME3,B1-ME1-");
  h->GetXaxis()->SetBinLabel(13,"B1-ME2(-3)");
  h->GetXaxis()->SetBinLabel(14,"ME1-4");
  h->GetXaxis()->SetBinLabel(15,"B1-ME1(-2)(-3)");
  h->GetXaxis()->SetBinLabel(16,"Halo Trigger");
}

// ================================================================================================
std::pair<float, float> 
GEMCSCTriggerRate::intersectionEtaPhi(CSCDetId id, int wg, int hs)
{

  CSCDetId layerId(id.endcap(), id.station(), id.ring(), id.chamber(), CSCConstants::KEY_CLCT_LAYER);
  const CSCLayer* csclayer = cscGeometry->layer(layerId);
  const CSCLayerGeometry* layer_geo = csclayer->geometry();
    
  // LCT::getKeyWG() starts from 0
  float wire = layer_geo->middleWireOfGroup(wg + 1);

  // half-strip to strip
  // note that LCT's HS starts from 0, but in geometry strips start from 1
  float fractional_strip = 0.5 * (hs + 1) - 0.25;
  
  LocalPoint csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
  
  GlobalPoint csc_gp = cscGeometry->idToDet(layerId)->surface().toGlobal(csc_intersect);
  
  return std::make_pair(csc_gp.eta(), csc_gp.phi());
}

// ================================================================================================
csctf::TrackStub 
GEMCSCTriggerRate::buildTrackStub(const CSCCorrelatedLCTDigi &d, CSCDetId id)
{
  unsigned fpga = (id.station() == 1) ? CSCTriggerNumbering::triggerSubSectorFromLabels(id) - 1 : id.station();
  CSCSectorReceiverLUT* srLUT = srLUTs_[fpga][id.triggerSector()-1][id.endcap()-1];

  unsigned cscid = CSCTriggerNumbering::triggerCscIdFromLabels(id);
  unsigned cscid_special = cscid;
  if (id.station()==1 && id.ring()==4) cscid_special = cscid + 9;

  lclphidat lclPhi;
  lclPhi = srLUT->localPhi(d.getStrip(), d.getPattern(), d.getQuality(), d.getBend());

  gblphidat gblPhi;
  gblPhi = srLUT->globalPhiME(lclPhi.phi_local, d.getKeyWG(), cscid_special);

  gbletadat gblEta;
  gblEta = srLUT->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, d.getKeyWG(), cscid);

  return csctf::TrackStub(d, id, gblPhi.global_phi, gblEta.global_eta);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMCSCTriggerRate);
