// -*- C++ -*-
//
// Package:    SimMuL1
// Class:      SimMuL1
// 
/**\class SimMuL1 SimMuL1.cc MyCode/SimMuL1/src/SimMuL1.cc

   Description: <one line class summary>

   Implementation:
   <Notes on implementation>
*/
//
// Original Author:  "Vadim Khotilovich"
//         Created:  Mon May  5 20:50:43 CDT 2008
// $Id: SimMuL1.cc,v 1.1 2013/05/30 17:22:23 dildick Exp $
//
//


#include "SimMuL1.h"

// system include files
#include <memory>
#include <cmath>


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"
#include <L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h>
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "GEMCode/GEMValidation/src/SimTrackMatchManager.h"

using namespace std;
using namespace reco;
using namespace edm;

// ================================================================================================
namespace 
{
  void linearRegression(int n, double *xx, double *yy, double &a, double &b)
  {
    //  estimate a and b in   y = a + b*x
    //  http://en.wikipedia.org/wiki/Linear_regression
  
    double sx=0., sx2=0., sy=0., sxy=0.;
    for (int i=0;i<n; i++)
      {
	sx  += xx[i];
	sx2 += xx[i]*xx[i];
	sy  += yy[i];
	sxy += xx[i]*yy[i];
      }
    double delta = n*sx2 - sx*sx;
    if (delta==0.){
      cout<<"delta==0: "<<" n sx sy sx2 sxy: "<<n<<" "<<sx<<" "<<sy<<" "<<sx2<<" "<<sxy<<endl;
      for (int i=0;i<n; i++)   cout<<"    "<<i<<": "<<xx[i]<<" "<<yy[i]<<endl;
      a=9999.; b=0.;
      return;
    }
    
    a = ( sx2*sy - sx*sxy ) / delta;
    b = ( n*sxy -sx*sy ) / delta;
  }

  bool isME1bEtaRegion(float eta, float eta_min = 1.64, float eta_max = 2.14)
  {
    if (fabs(eta) >= eta_min && fabs(eta) <= eta_max) return true;
    else return false;
  }

  //
  // constants, enums and typedefs
  //

  int MYDEBUG = 1;

  const Double_t ETA_BIN = 0.0125 *2;
  const Double_t PHI_BIN = 62.*M_PI/180./4096.; // 0.26 mrad

  //const double pi = TMath::Pi();
}


// ================================================================================================
// class' constants
//
const string SimMuL1::csc_type[CSC_TYPES+1] = 
  { "ME1/1", "ME1/2", "ME1/3", "ME1/a", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2", "ME1/T"};
const string SimMuL1::csc_type_[CSC_TYPES+1] = 
  { "ME11", "ME12", "ME13", "ME1A", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42", "ME1T"};
const string SimMuL1::csc_type_a[CSC_TYPES+2] =
  { "N/A", "ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2", "ME1/T"};
const string SimMuL1::csc_type_a_[CSC_TYPES+2] =
  { "NA", "ME1A", "ME1B", "ME12", "ME13", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42", "ME1T"};

const int SimMuL1::NCHAMBERS[CSC_TYPES] = 
  { 36,  36,  36,  36, 18,  36,  18,  36,  18,  36};

const int SimMuL1::MAX_WG[CSC_TYPES] = 
  { 48,  64,  32,  48, 112, 64,  96,  64,  96,  64};//max. number of wiregroups

const int SimMuL1::MAX_HS[CSC_TYPES] = 
  { 128, 160, 128, 96, 160, 160, 160, 160, 160, 160}; // max. # of halfstrips

//const int SimMuL1::ptype[CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07]= 
//  { -999,  3, -3,  2, -2,  1, -1,  0};  // "signed" pattern (== phiBend)
const int SimMuL1::pbend[CSCConstants::NUM_CLCT_PATTERNS]= 
  { -999,  -5,  4, -4,  3, -3,  2, -2,  1, -1,  0}; // "signed" pattern (== phiBend)


const double SimMuL1::PT_THRESHOLDS[N_PT_THRESHOLDS] = {0,10,20,30,40,50};
const double SimMuL1::PT_THRESHOLDS_FOR_ETA[N_PT_THRESHOLDS] = {10,15,30,40,55,70};


//
// static data member definitions
//


// ================================================================================================
//
// constructors and destructor
//
SimMuL1::SimMuL1(const edm::ParameterSet& iConfig):
  //  theCSCSimHitMap("MuonCSCHits"), theDTSimHitMap("MuonDTHits"), theRPCSimHitMap("MuonRPCHits")
  ptLUT(0),
  theCSCSimHitMap()
{
  simHitsFromCrossingFrame_ = iConfig.getUntrackedParameter<bool>("SimHitsFromCrossingFrame", false);
  simHitsModuleName_        = iConfig.getUntrackedParameter<string>("SimHitsModuleName",    "g4SimHits");
  simHitsCollectionName_    = iConfig.getUntrackedParameter<string>("SimHitsCollectionName","MuonCSCHits");
  theCSCSimHitMap.setUseCrossingFrame(simHitsFromCrossingFrame_);
  theCSCSimHitMap.setModuleName(simHitsModuleName_);
  theCSCSimHitMap.setCollectionName(simHitsCollectionName_);

  doStrictSimHitToTrackMatch_ = iConfig.getUntrackedParameter<bool>("doStrictSimHitToTrackMatch", false);
  matchAllTrigPrimitivesInChamber_ = iConfig.getUntrackedParameter<bool>("matchAllTrigPrimitivesInChamber", false);

  minNHitsShared_ = iConfig.getUntrackedParameter<int>("minNHitsShared_", -1);
  
  minDeltaYAnode_    = iConfig.getUntrackedParameter<double>("minDeltaYAnode", -1.);
  minDeltaYCathode_  = iConfig.getUntrackedParameter<double>("minDeltaYCathode", -1.);

  minDeltaWire_    = iConfig.getUntrackedParameter<int>("minDeltaWire", 0);
  maxDeltaWire_    = iConfig.getUntrackedParameter<int>("maxDeltaWire", 2);
  minDeltaStrip_   = iConfig.getUntrackedParameter<int>("minDeltaStrip", 1);
 
  debugALLEVENT = iConfig.getUntrackedParameter<int>("debugALLEVENT", 0);
  debugINHISTOS = iConfig.getUntrackedParameter<int>("debugINHISTOS", 0);
  debugALCT     = iConfig.getUntrackedParameter<int>("debugALCT", 0);
  debugCLCT     = iConfig.getUntrackedParameter<int>("debugCLCT", 0);
  debugLCT      = iConfig.getUntrackedParameter<int>("debugLCT", 0);
  debugMPLCT    = iConfig.getUntrackedParameter<int>("debugMPLCT", 0);
  debugTFTRACK  = iConfig.getUntrackedParameter<int>("debugTFTRACK", 0);
  debugTFCAND   = iConfig.getUntrackedParameter<int>("debugTFCAND", 0);
  debugGMTCAND  = iConfig.getUntrackedParameter<int>("debugGMTCAND", 0);
  debugL1EXTRA  = iConfig.getUntrackedParameter<int>("debugL1EXTRA", 0);
  debugRATE     = iConfig.getUntrackedParameter<int>("debugRATE", 0);

  minSimTrPt_   = iConfig.getUntrackedParameter<double>("minSimTrPt", 2.);
  minSimTrPhi_  = iConfig.getUntrackedParameter<double>("minSimTrPhi",-3.15);
  maxSimTrPhi_  = iConfig.getUntrackedParameter<double>("maxSimTrPhi", 3.15);
  minSimTrEta_  = iConfig.getUntrackedParameter<double>("minSimTrEta",-5.);
  maxSimTrEta_  = iConfig.getUntrackedParameter<double>("maxSimTrEta", 5.);
  invertSimTrPhiEta_ = iConfig.getUntrackedParameter<bool>("invertSimTrPhiEta", false);
  bestPtMatch_  = iConfig.getUntrackedParameter<bool>("bestPtMatch", true);

  minBX_    = iConfig.getUntrackedParameter< int >("minBX",-6);
  maxBX_    = iConfig.getUntrackedParameter< int >("maxBX",6);
  minTMBBX_ = iConfig.getUntrackedParameter< int >("minTMBBX",-6);
  maxTMBBX_ = iConfig.getUntrackedParameter< int >("maxTMBBX",6);
  minRateBX_    = iConfig.getUntrackedParameter< int >("minRateBX",-1);
  maxRateBX_    = iConfig.getUntrackedParameter< int >("maxRateBX",1);

  minBxALCT_ = iConfig.getUntrackedParameter< int >("minBxALCT",5);
  maxBxALCT_ = iConfig.getUntrackedParameter< int >("maxBxALCT",7);
  minBxCLCT_ = iConfig.getUntrackedParameter< int >("minBxCLCT",5);
  maxBxCLCT_ = iConfig.getUntrackedParameter< int >("maxBxCLCT",7);
  minBxLCT_ = iConfig.getUntrackedParameter< int >("minBxLCT",5);
  maxBxLCT_ = iConfig.getUntrackedParameter< int >("maxBxLCT",7);
  minBxMPLCT_ = iConfig.getUntrackedParameter< int >("minBxMPLCT",5);
  maxBxMPLCT_ = iConfig.getUntrackedParameter< int >("maxBxMPLCT",7);

  minBxGMT_ = iConfig.getUntrackedParameter< int >("minBxGMT",-1);
  maxBxGMT_ = iConfig.getUntrackedParameter< int >("maxBxGMT",1);

  centralBxOnlyGMT_ = iConfig.getUntrackedParameter< bool >("centralBxOnlyGMT",false);

  doSelectEtaForGMTRates_ = iConfig.getUntrackedParameter< bool >("doSelectEtaForGMTRates",false);
  
  goodChambersOnly_ = iConfig.getUntrackedParameter< bool >("goodChambersOnly",false);
  
  lookAtTrackCondition_ = iConfig.getUntrackedParameter<int>("lookAtTrackCondition", 0);
  
  doME1a_ = iConfig.getUntrackedParameter< bool >("doME1a",false);
  naiveME1a_ = iConfig.getUntrackedParameter< bool >("naiveME1a",true);

  // no GMT and L1Extra processing
  lightRun = iConfig.getUntrackedParameter<bool>("lightRun", true);

  // special treatment of matching in ME1a for the case of the default emulator
  defaultME1a = iConfig.getUntrackedParameter<bool>("defaultME1a", false);

  // properly treat ganged ME1a in matching (consider triple ambiguity)
  gangedME1a = iConfig.getUntrackedParameter<bool>("gangedME1a", false);
  //if (defaultME1a) gangedME1a = true;

  addGhostLCTs_ = iConfig.getUntrackedParameter< bool >("addGhostLCTs",true);

  minNStWith4Hits_ = iConfig.getUntrackedParameter< int >("minNStWith4Hits", 0);
  requireME1With4Hits_ = iConfig.getUntrackedParameter< bool >("requireME1With4Hits",false);

  minSimTrackDR_ = iConfig.getUntrackedParameter<double>("minSimTrackDR", 0.);
  
  ParameterSet stripPSet = iConfig.getParameter<edm::ParameterSet>("strips");
  theStripConditions = new CSCDbStripConditions(stripPSet);

  CSCTFSPset = iConfig.getParameter<edm::ParameterSet>("SectorProcessor");
  ptLUTset = CSCTFSPset.getParameter<edm::ParameterSet>("PTLUT");
  edm::ParameterSet srLUTset = CSCTFSPset.getParameter<edm::ParameterSet>("SRLUT");

  for(int e=0; e<2; e++) for (int s=0; s<6; s++) my_SPs[e][s] = NULL;
  
  bool TMB07 = true;
  for(int endcap = 1; endcap<=2; endcap++) for(int sector=1; sector<=6; sector++)
					     {
					       for(int station=1,fpga=0; station<=4 && fpga<5; station++)
						 {
						   if(station==1) for(int subSector=0; subSector<2; subSector++)
								    srLUTs_[fpga++][sector-1][endcap-1] = new CSCSectorReceiverLUT(endcap, sector, subSector+1, station, srLUTset, TMB07);
						   else
						     srLUTs_[fpga++][sector-1][endcap-1] = new CSCSectorReceiverLUT(endcap, sector, 0, station, srLUTset, TMB07);
						 }
					     }

  my_dtrc = new CSCTFDTReceiver();

  // cache flags for event setup records
  muScalesCacheID_ = 0ULL ;
  muPtScaleCacheID_ = 0ULL ;

  fill_debug_tree_ = iConfig.getUntrackedParameter< bool >("fill_debug_tree",false);
  if (fill_debug_tree_) bookDbgTTree();
  
  // processed event counter
  nevt = 0;

  gemMatchCfg_ = iConfig.getParameterSet("simTrackGEMMatching");
  gemPTs_ = iConfig.getParameter<std::vector<double> >("gemPTs");
  gemDPhisOdd_ = iConfig.getParameter<std::vector<double> >("gemDPhisOdd");
  gemDPhisEven_ = iConfig.getParameter<std::vector<double> >("gemDPhisEven");

  assert(std::is_sorted(gemPTs_.begin(), gemPTs_.end()));
  assert(gemPTs_.size() == gemDPhisOdd_.size() && gemPTs_.size() == gemDPhisEven_.size());

 
  // *********************************** HISTOGRAMS ******************************************
  Service<TFileService> fs;

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


  h_N_mctr  = fs->make<TH1D>("h_N_mctr","No of MC muons",16,-0.5,15.5);
  h_N_simtr = fs->make<TH1D>("h_N_simtr","No of SimTrack muons",16,-0.5,15.5);

  h_pt_mctr  = fs->make<TH1D>("h_pt_mctr","p_{T} of MC muons",50, 0.,100.);
  h_eta_mctr  = fs->make<TH1D>("h_eta_mctr","#eta of MC muons",N_ETA_BINS, ETA_START, ETA_END);
  h_phi_mctr  = fs->make<TH1D>("h_phi_mctr","#phi of MC muons",100, -M_PI,M_PI);

  h_DR_mctr_simtr    = fs->make<TH1D>("h_DR_mctr_simtr","#Delta R(MC trk, SimTrack)",300,0.,M_PI); 
  h_MinDR_mctr_simtr = fs->make<TH1D>("h_MinDR_mctr_simtr","min #Delta R(MC trk, SimTrack)",300,0.,M_PI); 

  h_DR_2SimTr        = fs->make<TH1D>("h_DR_2SimTr","#Delta R(SimTr 1, SimTr 2)",270,0.,M_PI*3/2); 
  h_DR_2SimTr_looked = fs->make<TH1D>("h_DR_2SimTr_looked","#Delta R(SimTr 1, SimTr 2)",270,0.,M_PI*3/2); 


  h_eta_vs_nalct = fs->make<TH2D>("h_eta_vs_nalct","h_eta_vs_nalct",N_ETA_BINS, ETA_START, ETA_END,13,-0.5,12.5); 
  h_eta_vs_nclct = fs->make<TH2D>("h_eta_vs_nclct","h_eta_vs_nclct",N_ETA_BINS, ETA_START, ETA_END,13,-0.5,12.5); 
  h_eta_vs_nlct  = fs->make<TH2D>("h_eta_vs_nlct","h_eta_vs_nlct",N_ETA_BINS, ETA_START, ETA_END,13,-0.5,12.5); 
  h_eta_vs_nmplct  = fs->make<TH2D>("h_eta_vs_nmplct","h_eta_vs_nmplct",N_ETA_BINS, ETA_START, ETA_END,13,-0.5,12.5); 
  
  h_pt_vs_nalct = fs->make<TH2D>("h_pt_vs_nalct","h_pt_vs_nalct",50, 0.,100.,13,-0.5,12.5);
  h_pt_vs_nclct = fs->make<TH2D>("h_pt_vs_nclct","h_pt_vs_nclct",50, 0.,100.,13,-0.5,12.5);
  h_pt_vs_nlct = fs->make<TH2D>("h_pt_vs_nlct","h_pt_vs_nlct",50, 0.,100.,13,-0.5,12.5);
  h_pt_vs_nmplct = fs->make<TH2D>("h_pt_vs_nmplct","h_pt_vs_nmplct",50, 0.,100.,13,-0.5,12.5);

  h_csctype_vs_alct_occup = fs->make<TH2D>("h_csctype_vs_alct_occup", "CSC type vs. ALCT chamber occupancy", 10, -0.5,  9.5, 5,-0.5,4.5);
  for (int i=1; i<=h_csctype_vs_alct_occup->GetXaxis()->GetNbins();i++)
    h_csctype_vs_alct_occup->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());

  h_csctype_vs_clct_occup = fs->make<TH2D>("h_csctype_vs_clct_occup", "CSC type vs. CLCT chamber occupancy", 10, -0.5,  9.5, 5,-0.5,4.5);
  for (int i=1; i<=h_csctype_vs_clct_occup->GetXaxis()->GetNbins();i++)
    h_csctype_vs_clct_occup->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());

  h_csctype_vs_nlct = fs->make<TH2D>("h_csctype_vs_nlct", "CSC type vs. No.LCTs per track", 10, -0.5,  9.5, 13,-0.5,12.5);
  for (int i=1; i<=h_csctype_vs_nlct->GetXaxis()->GetNbins();i++)
    h_csctype_vs_nlct->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());

  h_csctype_vs_nmplct = fs->make<TH2D>("h_csctype_vs_nmplct", "CSC type vs. No.MPCLCTs per track", 10, -0.5,  9.5, 13,-0.5,12.5);
  for (int i=1; i<=h_csctype_vs_nmplct->GetXaxis()->GetNbins();i++)
    h_csctype_vs_nmplct->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());

  h_n_bx_per_ch_alct = fs->make<TH1D>("h_n_bx_per_ch_alct", "h_n_bx_per_ch_alct", 12,0, 12);
  h_n_bx_per_ch_clct = fs->make<TH1D>("h_n_bx_per_ch_clct", "h_n_bx_per_ch_clct", 12,0, 12);
  h_n_bx_per_ch_lct  = fs->make<TH1D>("h_n_bx_per_ch_lct",  "h_n_bx_per_ch_lct",  12,0, 12);

  h_n_per_ch_alct = fs->make<TH1D>("h_n_per_ch_alct", "h_n_per_ch_alct", 6,-.5, 5.5);
  h_n_per_ch_clct = fs->make<TH1D>("h_n_per_ch_clct", "h_n_per_ch_clct", 6,-.5, 5.5);
  h_n_per_ch_lct  = fs->make<TH1D>("h_n_per_ch_lct",  "h_n_per_ch_lct",  6,-.5, 5.5);
  h_n_per_ch_mplct= fs->make<TH1D>("h_n_per_ch_mplct", "h_n_per_ch_mplct", 6,-.5, 5.5);


  h_dBx_LctAlct = fs->make<TH1D>("h_dBx_LctAlct","h_dBx_LctAlct",15,-7.5, 7.5);
  h_dBx_LctClct = fs->make<TH1D>("h_dBx_LctClct","h_dBx_LctClct",15,-7.5, 7.5);
  h_dBx_1inCh_LctClct  = fs->make<TH1D>("h_dBx_1inCh_LctClct","h_dBx_1inCh_LctClct",15,-7.5, 7.5);
  h_dBx_2inCh_LctClct  = fs->make<TH1D>("h_dBx_2inCh_LctClct","h_dBx_2inCh_LctClct",15,-7.5, 7.5);
  h_dBx_2inCh_LctClct2 = fs->make<TH2D>("h_dBx_2inCh_LctClct2","h_dBx_2inCh_LctClct2",15,-7.5, 7.5,15,-7.5, 7.5);

  h_type_lct = fs->make<TH1D>("h_type_lct","h_type_lct",11,-0.5, 10.5);

  h_bxdbx_alct_a1_da2 = fs->make<TH2D>("h_bxdbx_alct_a1_da2","h_bxdbx_alct_a1_da2",15,-7.5, 7.5,15,-7.5, 7.5);
  h_bxdbx_clct_c1_dc2 = fs->make<TH2D>("h_bxdbx_clct_c1_dc2","h_bxdbx_clct_c1_dc2",15,-7.5, 7.5,15,-7.5, 7.5);

  h_dbx_lct_a1_a2 = fs->make<TH2D>("h_dbx_lct_a1_a2","h_dbx_lct_a1_a2",15,-7.5, 7.5,15,-7.5, 7.5);
  h_bx_lct_a1_a2 = fs->make<TH2D>("h_bx_lct_a1_a2","h_bx_lct_a1_a2",15,-7.5, 7.5,15,-7.5, 7.5);

  h_tf_stub_bx = fs->make<TH1D>("h_tf_stub_bx","h_tf_stub_bx",15,-7.5, 7.5);
  h_tf_stub_qu = fs->make<TH1D>("h_tf_stub_qu","h_tf_stub_qu",17,-0.5, 16.5);
  h_tf_stub_qu_vs_bx = fs->make<TH2D>("h_tf_stub_qu_vs_bx","h_tf_stub_qu_vs_bx",17,-0.5, 16.5,15,-7.5, 7.5);

  h_tf_stub_bxclct = fs->make<TH1D>("h_tf_stub_bxclct","h_tf_stub_bxclct",15,-7.5, 7.5);

  h_tf_stub_pattern = fs->make<TH1D>("h_tf_stub_pattern","h_tf_stub_pattern",13,-0.5, 12.5);

  h_pattern_mplct = fs->make<TH1D>("h_pattern_mplct","h_pattern_mplct",13,-0.5, 12.5);
 
  h_bx_me11nomatchclct_alct_vs_clct = fs->make<TH2D>("h_bx_me11nomatchclct_alct_vs_clct","h_bx_me11nomatchclct_alct_vs_clct",13,-6.5,6.5,13,-6.5,6.5);
  h_bx_me11nomatchalct_alct_vs_clct = fs->make<TH2D>("h_bx_me11nomatchalct_alct_vs_clct","h_bx_me11nomatchalct_alct_vs_clct",13,-6.5,6.5,13,-6.5,6.5);
  
  h_bx_me1_aclct_ok_lct_no__bx_alct_vs_dbx_ACLCT = fs->make<TH2D>("h_bx_me1_aclct_ok_lct_no__bx_alct_vs_dbx_ACLCT","h_bx_me1_aclct_ok_lct_no__bx_alct_vs_dbx_ACLCT",13,-0.5,12.5, 13, -6.5,6.5);

  h_tf_stub_csctype = fs->make<TH1D>("h_tf_stub_csctype", "CSC type of TF track stubs", 10, -0.5,  9.5);
  for (int i=1; i<=h_tf_stub_csctype->GetXaxis()->GetNbins();i++)
    h_tf_stub_csctype->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());

  h_tf_stub_csctype_org = fs->make<TH1D>("h_tf_stub_csctype_org", "CSC type of TF track original stubs", 10, -0.5,  9.5);
  for (int i=1; i<=h_tf_stub_csctype_org->GetXaxis()->GetNbins();i++)
    h_tf_stub_csctype_org->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());

  h_tf_stub_csctype_org_unmatch = fs->make<TH1D>("h_tf_stub_csctype_org_unmatch", "CSC type of non-matched TF track org. stubs", 10, -0.5,  9.5);
  for (int i=1; i<=h_tf_stub_csctype_org_unmatch->GetXaxis()->GetNbins();i++)
    h_tf_stub_csctype_org_unmatch->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());


  h_strip_v_wireg_me1a = fs->make<TH2D>("h_strip_v_wireg_me1a","h_strip_v_wireg_me1a",97,-0.5, 96.5, 20,-0.5,19.5);
  h_strip_v_wireg_me1b = fs->make<TH2D>("h_strip_v_wireg_me1b","h_strip_v_wireg_me1b",129,-0.5, 128.5, 40, 8.5,48.5);


  h_station_tf_pu_no = fs->make<TH1D>("h_station_tf_pu_no","h_station_tf_pu_no",5, 0.5,5.5);
  h_station_tf_pu_ok = fs->make<TH1D>("h_station_tf_pu_ok","h_station_tf_pu_ok",5, 0.5,5.5);
  h_station_tf_pu_no_once = fs->make<TH1D>("h_station_tf_pu_no_once","h_station_tf_pu_no_once",5, 0.5,5.5);
  h_station_tf_pu_ok_once = fs->make<TH1D>("h_station_tf_pu_ok_once","h_station_tf_pu_ok_once",5, 0.5,5.5);
  h_station_tforg_pu_no = fs->make<TH1D>("h_station_tforg_pu_no","h_station_tforg_pu_no",5, 0.5,5.5);
  h_station_tforg_pu_ok = fs->make<TH1D>("h_station_tforg_pu_ok","h_station_tforg_pu_ok",5, 0.5,5.5);
  h_station_tforg_pu_no_once = fs->make<TH1D>("h_station_tforg_pu_no_once","h_station_tforg_pu_no_once",5, 0.5,5.5);
  h_station_tforg_pu_ok_once = fs->make<TH1D>("h_station_tforg_pu_ok_once","h_station_tforg_pu_ok_once",5, 0.5,5.5);
  h_tfqu_pt10 = fs->make<TH1D>("h_tfqu_pt10","h_tfqu_pt10",5,-0.5, 4.5);
  h_tfqu_pt10_no = fs->make<TH1D>("h_tfqu_pt10_no","h_tfqu_pt10_no",5,-0.5, 4.5);
  

  h_nMplct_vs_nDigiMplct = fs->make<TH2D>("h_nMplct_vs_nDigiMplct","h_nMplct_vs_nDigiMplct",9,-.5, 8.5,9,-.5, 8.5);
  h_qu_vs_nDigiMplct = fs->make<TH2D>("h_qu_vs_nDigiMplct","h_qu_vs_nDigiMplct",5,-0.5, 4.5,9,-.5, 8.5);
  
  h_ntftrackall_vs_ntftrack = fs->make<TH2D>("h_ntftrackall_vs_ntftrack","h_ntftrackall_vs_ntftrack",6,-0.5,5.5,6,-0.5,5.5);
  h_ntfcandall_vs_ntfcand = fs->make<TH2D>("h_ntfcandall_vs_ntfcand","h_ntfcandall_vs_ntfcand",6,-0.5,5.5,6,-0.5,5.5);

  h_pt_vs_ntfcand = fs->make<TH2D>("h_pt_vs_ntfcand","h_pt_vs_ntfcand",50, 0.,100.,6,-0.5,5.5);
  h_eta_vs_ntfcand = fs->make<TH2D>("h_eta_vs_ntfcand","h_eta_vs_ntfcand",N_ETA_BINS, ETA_START, ETA_END,4,-0.5,3.5); 

  h_pt_vs_qu = fs->make<TH2D>("h_pt_vs_qu","h_pt_vs_qu",100, 0.,100.,5,-0.5, 4.5);
  h_eta_vs_qu = fs->make<TH2D>("h_eta_vs_qu","h_eta_vs_qu",N_ETA_BINS, ETA_START, ETA_END,5,-0.5, 4.5);

  h_cscdet_of_chamber = fs->make<TH1D>("h_cscdet_of_chamber","h_cscdet_of_chamber",10, -0.5,  9.5);
  h_cscdet_of_chamber_w_alct = fs->make<TH1D>("h_cscdet_of_chamber_w_alct","h_cscdet_of_chamber_w_alct",10, -0.5,  9.5);
  h_cscdet_of_chamber_w_clct = fs->make<TH1D>("h_cscdet_of_chamber_w_clct","h_cscdet_of_chamber_w_clct",10, -0.5,  9.5);
  h_cscdet_of_chamber_w_mplct = fs->make<TH1D>("h_cscdet_of_chamber_w_mplct","h_cscdet_of_chamber_w_mplct",10, -0.5,  9.5);
  for (int i=1; i<=h_cscdet_of_chamber->GetXaxis()->GetNbins();i++) {
    h_cscdet_of_chamber->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());
    h_cscdet_of_chamber_w_alct->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());
    h_cscdet_of_chamber_w_clct->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());
    h_cscdet_of_chamber_w_mplct->GetXaxis()->SetBinLabel(i,csc_type[i-1].c_str());
  }
  
  int N_PT_BINS=100;
  double PT_START = 0.;
  double PT_END = 100.;

  h_pt_initial0 = fs->make<TH1D>("h_pt_initial0","h_pt_initial0",N_PT_BINS, PT_START, PT_END);
  h_pt_initial = fs->make<TH1D>("h_pt_initial","h_pt_initial",N_PT_BINS, PT_START, PT_END);
  h_pt_initial_1b = fs->make<TH1D>("h_pt_initial_1b","h_pt_initial_1b",N_PT_BINS, PT_START, PT_END);
  h_pt_initial_gem_1b = fs->make<TH1D>("h_pt_initial_gem_1b","h_pt_initial_gem_1b",N_PT_BINS, PT_START, PT_END);

  h_pt_me1_initial = fs->make<TH1D>("h_pt_me1_initial","h_pt_me1_initial",N_PT_BINS, PT_START, PT_END);
  h_pt_me2_initial = fs->make<TH1D>("h_pt_me2_initial","h_pt_me2_initial",N_PT_BINS, PT_START, PT_END);
  h_pt_me3_initial = fs->make<TH1D>("h_pt_me3_initial","h_pt_me3_initial",N_PT_BINS, PT_START, PT_END);
  h_pt_me4_initial = fs->make<TH1D>("h_pt_me4_initial","h_pt_me4_initial",N_PT_BINS, PT_START, PT_END);

  h_pt_initial_1st = fs->make<TH1D>("h_pt_initial_1st","h_pt_initial_1st",N_PT_BINS, PT_START, PT_END);
  h_pt_initial_2st = fs->make<TH1D>("h_pt_initial_2st","h_pt_initial_2st",N_PT_BINS, PT_START, PT_END);
  h_pt_initial_3st = fs->make<TH1D>("h_pt_initial_3st","h_pt_initial_3st",N_PT_BINS, PT_START, PT_END);

  h_pt_me1_initial_2st = fs->make<TH1D>("h_pt_me1_initial_2st","h_pt_me1_initial_2st",N_PT_BINS, PT_START, PT_END);
  h_pt_me1_initial_3st = fs->make<TH1D>("h_pt_me1_initial_3st","h_pt_me1_initial_3st",N_PT_BINS, PT_START, PT_END);


  h_pt_gem_1b = fs->make<TH1D>("h_pt_gem_1b","h_pt_gem_1b",N_PT_BINS, PT_START, PT_END);
  h_pt_lctgem_1b = fs->make<TH1D>("h_pt_lctgem_1b","h_pt_lctgem_1b",N_PT_BINS, PT_START, PT_END);

  h_pt_me1_mpc = fs->make<TH1D>("h_pt_me1_mpc","h_pt_me1_mpc",N_PT_BINS, PT_START, PT_END);
  h_pt_me2_mpc = fs->make<TH1D>("h_pt_me2_mpc","h_pt_me2_mpc",N_PT_BINS, PT_START, PT_END);
  h_pt_me3_mpc = fs->make<TH1D>("h_pt_me3_mpc","h_pt_me3_mpc",N_PT_BINS, PT_START, PT_END);
  h_pt_me4_mpc = fs->make<TH1D>("h_pt_me4_mpc","h_pt_me4_mpc",N_PT_BINS, PT_START, PT_END);

  h_pt_mpc_1st = fs->make<TH1D>("h_pt_mpc_1st","h_pt_mpc_1st",N_PT_BINS, PT_START, PT_END);
  h_pt_mpc_2st = fs->make<TH1D>("h_pt_mpc_2st","h_pt_mpc_2st",N_PT_BINS, PT_START, PT_END);
  h_pt_mpc_3st = fs->make<TH1D>("h_pt_mpc_3st","h_pt_mpc_3st",N_PT_BINS, PT_START, PT_END);

  h_pt_me1_mpc_2st = fs->make<TH1D>("h_pt_me1_mpc_2st","h_pt_me1_mpc_2st",N_PT_BINS, PT_START, PT_END);
  h_pt_me1_mpc_3st = fs->make<TH1D>("h_pt_me1_mpc_3st","h_pt_me1_mpc_3st",N_PT_BINS, PT_START, PT_END);


  for (int i=0; i<N_PT_THRESHOLDS; i++){
    int tfpt = (int) PT_THRESHOLDS[i];
    sprintf(label,"h_pt_tf_initial0_tfpt%d",tfpt);
    h_pt_tf_initial0_tfpt[i]  = fs->make<TH1D>(label, label, N_PT_BINS, PT_START, PT_END);
    sprintf(label,"h_pt_tf_initial_tfpt%d",tfpt);
    h_pt_tf_initial_tfpt[i]  = fs->make<TH1D>(label, label, N_PT_BINS, PT_START, PT_END);
    sprintf(label,"h_pt_tf_stubs222_tfpt%d",tfpt);
    h_pt_tf_stubs222_tfpt[i]  = fs->make<TH1D>(label, label, N_PT_BINS, PT_START, PT_END);
    sprintf(label,"h_pt_tf_stubs223_tfpt%d",tfpt);
    h_pt_tf_stubs223_tfpt[i]  = fs->make<TH1D>(label, label, N_PT_BINS, PT_START, PT_END);
    sprintf(label,"h_pt_tf_stubs233_tfpt%d",tfpt);
    h_pt_tf_stubs233_tfpt[i]  = fs->make<TH1D>(label, label, N_PT_BINS, PT_START, PT_END);
  }


  h_pt_after_alct = fs->make<TH1D>("h_pt_after_alct","h_pt_after_alct",N_PT_BINS, PT_START, PT_END);
  h_pt_after_clct = fs->make<TH1D>("h_pt_after_clct","h_pt_after_clct",N_PT_BINS, PT_START, PT_END);
  h_pt_after_lct = fs->make<TH1D>("h_pt_after_lct","h_pt_after_lct",N_PT_BINS, PT_START, PT_END);
  h_pt_after_mpc = fs->make<TH1D>("h_pt_after_mpc","h_pt_after_mpc",N_PT_BINS, PT_START, PT_END);
  h_pt_after_mpc_ok_plus = fs->make<TH1D>("h_pt_after_mpc_ok_plus","h_pt_after_mpc_ok_plus",N_PT_BINS, PT_START, PT_END);
  h_pt_me1_after_mpc_ok_plus = fs->make<TH1D>("h_pt_after_mpc_me1_plus","h_pt_me1_after_mpc_ok_plus",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tftrack = fs->make<TH1D>("h_pt_after_tftrack","h_pt_after_tftrack",N_PT_BINS, PT_START, PT_END);

  h_pt_after_tfcand = fs->make<TH1D>("h_pt_after_tfcand","h_pt_after_tfcand",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_pt10 = fs->make<TH1D>("h_pt_after_tfcand_pt10","h_pt_after_tfcand_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_pt20 = fs->make<TH1D>("h_pt_after_tfcand_pt20","h_pt_after_tfcand_pt20",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_pt40 = fs->make<TH1D>("h_pt_after_tfcand_pt40","h_pt_after_tfcand_pt40",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_pt60 = fs->make<TH1D>("h_pt_after_tfcand_pt60","h_pt_after_tfcand_pt60",N_PT_BINS, PT_START, PT_END);

  h_pt_after_tfcand_ok = fs->make<TH1D>("h_pt_after_tfcand_ok","h_pt_after_tfcand_ok",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_pt10_ok = fs->make<TH1D>("h_pt_after_tfcand_pt10_ok","h_pt_after_tfcand_pt10_ok",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_pt20_ok = fs->make<TH1D>("h_pt_after_tfcand_pt20_ok","h_pt_after_tfcand_pt20_ok",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_pt40_ok = fs->make<TH1D>("h_pt_after_tfcand_pt40_ok","h_pt_after_tfcand_pt40_ok",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_pt60_ok = fs->make<TH1D>("h_pt_after_tfcand_pt60_ok","h_pt_after_tfcand_pt60_ok",N_PT_BINS, PT_START, PT_END);

  const int Nthr = 7;
  string str_pts[Nthr] = {"", "_pt10", "_pt15", "_pt20", "_pt25", "_pt30","_pt40"};
  for (int i = 0; i < Nthr; ++i) {
    string prefix = "h_pt_after_tfcand_eta1b_";
    h_pt_after_tfcand_eta1b_2s[i] = fs->make<TH1D>((prefix + "2s" + str_pts[i]).c_str(), (prefix + "2s" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_eta1b_2s1b[i] = fs->make<TH1D>((prefix + "2s1b" + str_pts[i]).c_str(), (prefix + "2s1b" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_eta1b_2s123[i] = fs->make<TH1D>((prefix + "2s123" + str_pts[i]).c_str(), (prefix + "2s123" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_eta1b_2s13[i] = fs->make<TH1D>((prefix + "2s13" + str_pts[i]).c_str(), (prefix + "2s13" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_eta1b_3s[i] = fs->make<TH1D>((prefix + "3s" + str_pts[i]).c_str(), (prefix + "3s" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_eta1b_3s1b[i] = fs->make<TH1D>((prefix + "3s1b" + str_pts[i]).c_str(), (prefix + "3s1b" + str_pts[i]).c_str(),N_PT_BINS, PT_START, PT_END);
    prefix = "h_pt_after_tfcand_gem1b_";
    h_pt_after_tfcand_gem1b_2s1b[i] = fs->make<TH1D>((prefix + "2s1b" + str_pts[i]).c_str(), (prefix + "2s1b" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_gem1b_2s123[i] = fs->make<TH1D>((prefix + "2s123" + str_pts[i]).c_str(), (prefix + "2s123" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_gem1b_2s13[i] = fs->make<TH1D>((prefix + "2s13" + str_pts[i]).c_str(), (prefix + "2s13" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_gem1b_3s1b[i] = fs->make<TH1D>((prefix + "3s1b" + str_pts[i]).c_str(), (prefix + "3s1b" + str_pts[i]).c_str(),N_PT_BINS, PT_START, PT_END);
    prefix = "h_pt_after_tfcand_dphigem1b_";
    h_pt_after_tfcand_dphigem1b_2s1b[i] = fs->make<TH1D>((prefix + "2s1b" + str_pts[i]).c_str(), (prefix + "2s1b" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_dphigem1b_2s123[i] = fs->make<TH1D>((prefix + "2s123" + str_pts[i]).c_str(), (prefix + "2s123" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_dphigem1b_2s13[i] = fs->make<TH1D>((prefix + "2s13" + str_pts[i]).c_str(), (prefix + "2s13" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    h_pt_after_tfcand_dphigem1b_3s1b[i] = fs->make<TH1D>((prefix + "3s1b" + str_pts[i]).c_str(), (prefix + "3s1b" + str_pts[i]).c_str(),N_PT_BINS, PT_START, PT_END);

    prefix = "h_mode_tfcand_gem1b_2s1b_1b_";
    h_mode_tfcand_gem1b_2s1b_1b[i] = fs->make<TH1D>((prefix + str_pts[i]).c_str(), (prefix + str_pts[i]).c_str(), 16, -0.5, 15.5);
    setupTFModeHisto(h_mode_tfcand_gem1b_2s1b_1b[i]);
  }

  h_pt_after_tfcand_ok_plus = fs->make<TH1D>("h_pt_after_tfcand_ok_plus","h_pt_after_tfcand_ok_plus",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_ok_plus_pt10 = fs->make<TH1D>("h_pt_after_tfcand_ok_plus_pt10","h_pt_after_tfcand_ok_plus_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_ok_plus_q[0] = fs->make<TH1D>("h_pt_after_tfcand_ok_plus_q1","h_pt_after_tfcand_ok_plus_q1",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_ok_plus_q[1] = fs->make<TH1D>("h_pt_after_tfcand_ok_plus_q2","h_pt_after_tfcand_ok_plus_q2",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_ok_plus_q[2] = fs->make<TH1D>("h_pt_after_tfcand_ok_plus_q3","h_pt_after_tfcand_ok_plus_q3",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_ok_plus_pt10_q[0] = fs->make<TH1D>("h_pt_after_tfcand_ok_plus_pt10_q1","h_pt_after_tfcand_ok_plus_pt10_q1",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_ok_plus_pt10_q[1] = fs->make<TH1D>("h_pt_after_tfcand_ok_plus_pt10_q2","h_pt_after_tfcand_ok_plus_pt10_q2",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_ok_plus_pt10_q[2] = fs->make<TH1D>("h_pt_after_tfcand_ok_plus_pt10_q3","h_pt_after_tfcand_ok_plus_pt10_q3",N_PT_BINS, PT_START, PT_END);

  h_pt_after_tfcand_all = fs->make<TH1D>("h_pt_after_tfcand_all","h_pt_after_tfcand_all",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_all_ok = fs->make<TH1D>("h_pt_after_tfcand_all_ok","h_pt_after_tfcand_all_ok",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_all_pt10_ok = fs->make<TH1D>("h_pt_after_tfcand_all_pt10_ok","h_pt_after_tfcand_all_pt10_ok",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_all_pt20_ok = fs->make<TH1D>("h_pt_after_tfcand_all_pt20_ok","h_pt_after_tfcand_all_pt20_ok",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_all_pt40_ok = fs->make<TH1D>("h_pt_after_tfcand_all_pt40_ok","h_pt_after_tfcand_all_pt40_ok",N_PT_BINS, PT_START, PT_END);
  h_pt_after_tfcand_all_pt60_ok = fs->make<TH1D>("h_pt_after_tfcand_all_pt60_ok","h_pt_after_tfcand_all_pt60_ok",N_PT_BINS, PT_START, PT_END);

  h_pt_after_gmtreg = fs->make<TH1D>("h_pt_after_gmtreg","h_pt_after_gmtreg",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmtreg_all = fs->make<TH1D>("h_pt_after_gmtreg_all","h_pt_after_gmtreg_all",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmtreg_dr = fs->make<TH1D>("h_pt_after_gmtreg_dr","h_pt_after_gmtreg_dr",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmt = fs->make<TH1D>("h_pt_after_gmt","h_pt_after_gmt",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmt_all = fs->make<TH1D>("h_pt_after_gmt_all","h_pt_after_gmt_all",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmt_dr = fs->make<TH1D>("h_pt_after_gmt_dr","h_pt_after_gmt_dr",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmt_dr_nocsc = fs->make<TH1D>("h_pt_after_gmt_dr_nocsc","h_pt_after_gmt_dr_nocsc",N_PT_BINS, PT_START, PT_END);

  h_pt_after_gmtreg_pt10 = fs->make<TH1D>("h_pt_after_gmtreg_pt10","h_pt_after_gmtreg_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmtreg_all_pt10 = fs->make<TH1D>("h_pt_after_gmtreg_all_pt10","h_pt_after_gmtreg_all_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmtreg_dr_pt10 = fs->make<TH1D>("h_pt_after_gmtreg_dr_pt10","h_pt_after_gmtreg_dr_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmt_pt10 = fs->make<TH1D>("h_pt_after_gmt_pt10","h_pt_after_gmt_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmt_all_pt10 = fs->make<TH1D>("h_pt_after_gmt_all_pt10","h_pt_after_gmt_all_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmt_dr_pt10 = fs->make<TH1D>("h_pt_after_gmt_dr_pt10","h_pt_after_gmt_dr_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_after_gmt_dr_nocsc_pt10 = fs->make<TH1D>("h_pt_after_gmt_dr_nocsc_pt10","h_pt_after_gmt_dr_nocsc_pt10",N_PT_BINS, PT_START, PT_END);

  for (int i = 0; i < Nthr; ++i) {
    string prefix = "h_pt_after_gmt_eta1b_";
    h_pt_after_gmt_eta1b_1mu[i] = fs->make<TH1D>((prefix + "1mu" + str_pts[i]).c_str(), (prefix + "1mu" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    prefix = "h_pt_after_gmt_gem1b_";
    h_pt_after_gmt_gem1b_1mu[i] = fs->make<TH1D>((prefix + "1mu" + str_pts[i]).c_str(), (prefix + "1mu" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
    prefix = "h_pt_after_gmt_dphigem1b_";
    h_pt_after_gmt_dphigem1b_1mu[i] = fs->make<TH1D>((prefix + "1mu" + str_pts[i]).c_str(), (prefix + "1mu" + str_pts[i]).c_str(), N_PT_BINS, PT_START, PT_END);
  }


  h_pt_after_xtra = fs->make<TH1D>("h_pt_after_xtra","h_pt_after_xtra",N_PT_BINS, PT_START, PT_END);
  h_pt_after_xtra_all = fs->make<TH1D>("h_pt_after_xtra_all","h_pt_after_xtra_all",N_PT_BINS, PT_START, PT_END);
  h_pt_after_xtra_dr = fs->make<TH1D>("h_pt_after_xtra_dr","h_pt_after_xtra_dr",N_PT_BINS, PT_START, PT_END);

  h_pt_after_xtra_pt10 = fs->make<TH1D>("h_pt_after_xtra_pt10","h_pt_after_xtra_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_after_xtra_all_pt10 = fs->make<TH1D>("h_pt_after_xtra_all_pt10","h_pt_after_xtra_all_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_after_xtra_dr_pt10 = fs->make<TH1D>("h_pt_after_xtra_dr_pt10","h_pt_after_xtra_dr_pt10",N_PT_BINS, PT_START, PT_END);

  h_pt_me1_after_tf_ok_plus = fs->make<TH1D>("h_pt_me1_after_tf_ok_plus","h_pt_me1_after_tf_ok_plus",N_PT_BINS, PT_START, PT_END);
  h_pt_me1_after_tf_ok_plus_pt10 = fs->make<TH1D>("h_pt_me1_after_tf_ok_plus_pt10","h_pt_me1_after_tf_ok_plus_pt10",N_PT_BINS, PT_START, PT_END);
  h_pt_me1_after_tf_ok_plus_q[0] = fs->make<TH1D>("h_pt_me1_after_tf_ok_plus_q1","h_pt_me1_after_tf_ok_plus_q1",N_PT_BINS, PT_START, PT_END);
  h_pt_me1_after_tf_ok_plus_q[1] = fs->make<TH1D>("h_pt_me1_after_tf_ok_plus_q2","h_pt_me1_after_tf_ok_plus_q2",N_PT_BINS, PT_START, PT_END);
  h_pt_me1_after_tf_ok_plus_q[2] = fs->make<TH1D>("h_pt_me1_after_tf_ok_plus_q3","h_pt_me1_after_tf_ok_plus_q3",N_PT_BINS, PT_START, PT_END);
  h_pt_me1_after_tf_ok_plus_pt10_q[0] = fs->make<TH1D>("h_pt_me1_after_tf_ok_plus_pt10_q1","h_pt_me1_after_tf_ok_plus_pt10_q1",N_PT_BINS, PT_START, PT_END);
  h_pt_me1_after_tf_ok_plus_pt10_q[1] = fs->make<TH1D>("h_pt_me1_after_tf_ok_plus_pt10_q2","h_pt_me1_after_tf_ok_plus_pt10_q2",N_PT_BINS, PT_START, PT_END);
  h_pt_me1_after_tf_ok_plus_pt10_q[2] = fs->make<TH1D>("h_pt_me1_after_tf_ok_plus_pt10_q3","h_pt_me1_after_tf_ok_plus_pt10_q3",N_PT_BINS, PT_START, PT_END);


  // high eta
  h_pth_initial = fs->make<TH1D>("h_pth_initial","h_pth_initial",N_PT_BINS, PT_START, PT_END);
  h_pth_after_mpc = fs->make<TH1D>("h_pth_after_mpc","h_pth_after_mpc",N_PT_BINS, PT_START, PT_END);
  h_pth_after_mpc_ok_plus = fs->make<TH1D>("h_pth_after_mpc_ok_plus","h_pth_after_mpc_ok_plus",N_PT_BINS, PT_START, PT_END);

  h_pth_after_tfcand = fs->make<TH1D>("h_pth_after_tfcand","h_pth_after_tfcand",N_PT_BINS, PT_START, PT_END);
  h_pth_after_tfcand_pt10 = fs->make<TH1D>("h_pth_after_tfcand_pt10","h_pth_after_tfcand_pt10",N_PT_BINS, PT_START, PT_END);

  h_pth_after_tfcand_ok = fs->make<TH1D>("h_pth_after_tfcand_ok","h_pth_after_tfcand_ok",N_PT_BINS, PT_START, PT_END);
  h_pth_after_tfcand_pt10_ok = fs->make<TH1D>("h_pth_after_tfcand_pt10_ok","h_pth_after_tfcand_pt10_ok",N_PT_BINS, PT_START, PT_END);

  h_pth_after_tfcand_ok_plus = fs->make<TH1D>("h_pth_after_tfcand_ok_plus","h_pth_after_tfcand_ok_plus",N_PT_BINS, PT_START, PT_END);
  h_pth_after_tfcand_ok_plus_pt10 = fs->make<TH1D>("h_pth_after_tfcand_ok_plus_pt10","h_pth_after_tfcand_ok_plus_pt10",N_PT_BINS, PT_START, PT_END);
  h_pth_after_tfcand_ok_plus_q[0] = fs->make<TH1D>("h_pth_after_tfcand_ok_plus_q1","h_pth_after_tfcand_ok_plus_q1",N_PT_BINS, PT_START, PT_END);
  h_pth_after_tfcand_ok_plus_q[1] = fs->make<TH1D>("h_pth_after_tfcand_ok_plus_q2","h_pth_after_tfcand_ok_plus_q2",N_PT_BINS, PT_START, PT_END);
  h_pth_after_tfcand_ok_plus_q[2] = fs->make<TH1D>("h_pth_after_tfcand_ok_plus_q3","h_pth_after_tfcand_ok_plus_q3",N_PT_BINS, PT_START, PT_END);
  h_pth_after_tfcand_ok_plus_pt10_q[0] = fs->make<TH1D>("h_pth_after_tfcand_ok_plus_pt10_q1","h_pth_after_tfcand_ok_plus_pt10_q1",N_PT_BINS, PT_START, PT_END);
  h_pth_after_tfcand_ok_plus_pt10_q[1] = fs->make<TH1D>("h_pth_after_tfcand_ok_plus_pt10_q2","h_pth_after_tfcand_ok_plus_pt10_q2",N_PT_BINS, PT_START, PT_END);
  h_pth_after_tfcand_ok_plus_pt10_q[2] = fs->make<TH1D>("h_pth_after_tfcand_ok_plus_pt10_q3","h_pth_after_tfcand_ok_plus_pt10_q3",N_PT_BINS, PT_START, PT_END);

  h_pth_me1_after_mpc_ok_plus = fs->make<TH1D>("h_pth_after_mpc_me1_plus","h_pth_me1_after_mpc_ok_plus",N_PT_BINS, PT_START, PT_END);
  h_pth_me1_after_tf_ok_plus = fs->make<TH1D>("h_pth_me1_after_tf_ok_plus","h_pth_me1_after_tf_ok_plus",N_PT_BINS, PT_START, PT_END);
  h_pth_me1_after_tf_ok_plus_pt10 = fs->make<TH1D>("h_pth_me1_after_tf_ok_plus_pt10","h_pth_me1_after_tf_ok_plus_pt10",N_PT_BINS, PT_START, PT_END);
  h_pth_me1_after_tf_ok_plus_q[0] = fs->make<TH1D>("h_pth_me1_after_tf_ok_plus_q1","h_pth_me1_after_tf_ok_plus_q1",N_PT_BINS, PT_START, PT_END);
  h_pth_me1_after_tf_ok_plus_q[1] = fs->make<TH1D>("h_pth_me1_after_tf_ok_plus_q2","h_pth_me1_after_tf_ok_plus_q2",N_PT_BINS, PT_START, PT_END);
  h_pth_me1_after_tf_ok_plus_q[2] = fs->make<TH1D>("h_pth_me1_after_tf_ok_plus_q3","h_pth_me1_after_tf_ok_plus_q3",N_PT_BINS, PT_START, PT_END);
  h_pth_me1_after_tf_ok_plus_pt10_q[0] = fs->make<TH1D>("h_pth_me1_after_tf_ok_plus_pt10_q1","h_pth_me1_after_tf_ok_plus_pt10_q1",N_PT_BINS, PT_START, PT_END);
  h_pth_me1_after_tf_ok_plus_pt10_q[1] = fs->make<TH1D>("h_pth_me1_after_tf_ok_plus_pt10_q2","h_pth_me1_after_tf_ok_plus_pt10_q2",N_PT_BINS, PT_START, PT_END);
  h_pth_me1_after_tf_ok_plus_pt10_q[2] = fs->make<TH1D>("h_pth_me1_after_tf_ok_plus_pt10_q3","h_pth_me1_after_tf_ok_plus_pt10_q3",N_PT_BINS, PT_START, PT_END);

  h_pth_over_tfpt_resol = fs->make<TH1D>("h_pth_over_pttf_resol","h_pth_over_pttf_resol",300, -1.5,1.5);
  h_pth_over_tfpt_resol_vs_pt = fs->make<TH2D>("h_pth_over_pttf_resol_vs_pt","h_pth_over_pttf_resol_vs_pt",150, -1.5,1.5,N_PT_BINS, PT_START, PT_END);


  h_pth_after_tfcand_ok_plus_3st1a = fs->make<TH1D>("h_pth_after_tfcand_ok_plus_3st1a","h_pth_after_tfcand_ok_plus_3st1a",N_PT_BINS, PT_START, PT_END);
  h_pth_after_tfcand_ok_plus_pt10_3st1a = fs->make<TH1D>("h_pth_after_tfcand_ok_plus_pt10_3st1a","h_pth_after_tfcand_ok_plus_pt10_3st1a",N_PT_BINS, PT_START, PT_END);
  h_pth_me1_after_tf_ok_plus_3st1a = fs->make<TH1D>("h_pth_me1_after_tf_ok_plus_3st1a","h_pth_me1_after_tf_ok_plus_3st1a",N_PT_BINS, PT_START, PT_END);
  h_pth_me1_after_tf_ok_plus_pt10_3st1a = fs->make<TH1D>("h_pth_me1_after_tf_ok_plus_pt10_3st1a","h_pth_me1_after_tf_ok_plus_pt10_3st1a",N_PT_BINS, PT_START, PT_END);



  h_eta_initial0 = fs->make<TH1D>("h_eta_initial0","h_eta_initial0",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_initial = fs->make<TH1D>("h_eta_initial","h_eta_initial",N_ETA_BINS, ETA_START, ETA_END);

  h_eta_me1_initial = fs->make<TH1D>("h_eta_me1_initial","h_eta_me1_initial",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me2_initial = fs->make<TH1D>("h_eta_me2_initial","h_eta_me2_initial",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me3_initial = fs->make<TH1D>("h_eta_me3_initial","h_eta_me3_initial",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me4_initial = fs->make<TH1D>("h_eta_me4_initial","h_eta_me4_initial",N_ETA_BINS, ETA_START, ETA_END);

  h_eta_initial_1st = fs->make<TH1D>("h_eta_initial_1st","h_eta_initial_1st",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_initial_2st = fs->make<TH1D>("h_eta_initial_2st","h_eta_initial_2st",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_initial_3st = fs->make<TH1D>("h_eta_initial_3st","h_eta_initial_3st",N_ETA_BINS, ETA_START, ETA_END);

  h_eta_me1_initial_2st = fs->make<TH1D>("h_eta_me1_initial_2st","h_eta_me1_initial_2st",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_initial_3st = fs->make<TH1D>("h_eta_me1_initial_3st","h_eta_me1_initial_3st",N_ETA_BINS, ETA_START, ETA_END);


  h_eta_me1_mpc = fs->make<TH1D>("h_eta_me1_mpc","h_eta_me1_mpc",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me2_mpc = fs->make<TH1D>("h_eta_me2_mpc","h_eta_me2_mpc",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me3_mpc = fs->make<TH1D>("h_eta_me3_mpc","h_eta_me3_mpc",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me4_mpc = fs->make<TH1D>("h_eta_me4_mpc","h_eta_me4_mpc",N_ETA_BINS, ETA_START, ETA_END);

  h_eta_mpc_1st = fs->make<TH1D>("h_eta_mpc_1st","h_eta_mpc_1st",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_mpc_2st = fs->make<TH1D>("h_eta_mpc_2st","h_eta_mpc_2st",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_mpc_3st = fs->make<TH1D>("h_eta_mpc_3st","h_eta_mpc_3st",N_ETA_BINS, ETA_START, ETA_END);

  h_eta_me1_mpc_2st = fs->make<TH1D>("h_eta_me1_mpc_2st","h_eta_me1_mpc_2st",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_mpc_3st = fs->make<TH1D>("h_eta_me1_mpc_3st","h_eta_me1_mpc_3st",N_ETA_BINS, ETA_START, ETA_END);



  h_eta_after_alct = fs->make<TH1D>("h_eta_after_alct","h_eta_after_alct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_clct = fs->make<TH1D>("h_eta_after_clct","h_eta_after_clct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_lct = fs->make<TH1D>("h_eta_after_lct","h_eta_after_lct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_mpc = fs->make<TH1D>("h_eta_after_mpc","h_eta_after_mpc",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_mpc_ok = fs->make<TH1D>("h_eta_after_mpc_ok","h_eta_after_mpc_ok",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_mpc_ok_plus = fs->make<TH1D>("h_eta_after_mpc_ok_plus","h_eta_after_mpc_ok_plus",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_mpc_ok_plus_3st = fs->make<TH1D>("h_eta_after_mpc_ok_plus_3st","h_eta_after_mpc_ok_plus_3st",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_mpc_st1 = fs->make<TH1D>("h_eta_after_mpc_st1","h_eta_after_mpc_st1",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_mpc_st1_good = fs->make<TH1D>("h_eta_after_mpc_st1_good","h_eta_after_mpc_st1_good",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tftrack = fs->make<TH1D>("h_eta_after_tftrack","h_eta_after_tftrack",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand = fs->make<TH1D>("h_eta_after_tfcand","h_eta_after_tfcand",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_q[0] = fs->make<TH1D>("h_eta_after_tfcand_q1","h_eta_after_tfcand_q1",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_q[1] = fs->make<TH1D>("h_eta_after_tfcand_q2","h_eta_after_tfcand_q2",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_q[2] = fs->make<TH1D>("h_eta_after_tfcand_q3","h_eta_after_tfcand_q3",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok = fs->make<TH1D>("h_eta_after_tfcand_ok","h_eta_after_tfcand_ok",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_plus = fs->make<TH1D>("h_eta_after_tfcand_ok_plus","h_eta_after_tfcand_ok_plus",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_pt10 = fs->make<TH1D>("h_eta_after_tfcand_ok_pt10","h_eta_after_tfcand_ok_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_plus_pt10 = fs->make<TH1D>("h_eta_after_tfcand_ok_plus_pt10","h_eta_after_tfcand_ok_plus_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_plus_q[0] = fs->make<TH1D>("h_eta_after_tfcand_ok_plus_q1","h_eta_after_tfcand_ok_plus_q1",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_plus_q[1] = fs->make<TH1D>("h_eta_after_tfcand_ok_plus_q2","h_eta_after_tfcand_ok_plus_q2",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_plus_q[2] = fs->make<TH1D>("h_eta_after_tfcand_ok_plus_q3","h_eta_after_tfcand_ok_plus_q3",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_plus_pt10_q[0] = fs->make<TH1D>("h_eta_after_tfcand_ok_plus_pt10_q1","h_eta_after_tfcand_ok_plus_pt10_q1",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_plus_pt10_q[1] = fs->make<TH1D>("h_eta_after_tfcand_ok_plus_pt10_q2","h_eta_after_tfcand_ok_plus_pt10_q2",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_plus_pt10_q[2] = fs->make<TH1D>("h_eta_after_tfcand_ok_plus_pt10_q3","h_eta_after_tfcand_ok_plus_pt10_q3",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_all = fs->make<TH1D>("h_eta_after_tfcand_all","h_eta_after_tfcand_all",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_pt10 = fs->make<TH1D>("h_eta_after_tfcand_pt10","h_eta_after_tfcand_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_all_pt10 = fs->make<TH1D>("h_eta_after_tfcand_all_pt10","h_eta_after_tfcand_all_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_my_st1 = fs->make<TH1D>("h_eta_after_tfcand_my_st1","h_eta_after_tfcand_my_st1",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_org_st1 = fs->make<TH1D>("h_eta_after_tfcand_org_st1","h_eta_after_tfcand_org_st1",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_comm_st1 = fs->make<TH1D>("h_eta_after_tfcand_comm_st1","h_eta_after_tfcand_comm_st1",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_my_st1_pt10 = fs->make<TH1D>("h_eta_after_tfcand_my_st1_pt10","h_eta_after_tfcand_my_st1_pt10",N_ETA_BINS, ETA_START, ETA_END);
  
  h_eta_after_gmtreg = fs->make<TH1D>("h_eta_after_gmtreg","h_eta_after_gmtreg",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmtreg_all = fs->make<TH1D>("h_eta_after_gmtreg_all","h_eta_after_gmtreg_all",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmtreg_dr = fs->make<TH1D>("h_eta_after_gmtreg_dr","h_eta_after_gmtreg_dr",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmt = fs->make<TH1D>("h_eta_after_gmt","h_eta_after_gmt",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmt_all = fs->make<TH1D>("h_eta_after_gmt_all","h_eta_after_gmt_all",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmt_dr = fs->make<TH1D>("h_eta_after_gmt_dr","h_eta_after_gmt_dr",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmt_dr_nocsc = fs->make<TH1D>("h_eta_after_gmt_dr_nocsc","h_eta_after_gmt_dr_nocsc",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmtreg_pt10 = fs->make<TH1D>("h_eta_after_gmtreg_pt10","h_eta_after_gmtreg_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmtreg_all_pt10 = fs->make<TH1D>("h_eta_after_gmtreg_all_pt10","h_eta_after_gmtreg_all_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmtreg_dr_pt10 = fs->make<TH1D>("h_eta_after_gmtreg_dr_pt10","h_eta_after_gmtreg_dr_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmt_pt10 = fs->make<TH1D>("h_eta_after_gmt_pt10","h_eta_after_gmt_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmt_all_pt10 = fs->make<TH1D>("h_eta_after_gmt_all_pt10","h_eta_after_gmt_all_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmt_dr_pt10 = fs->make<TH1D>("h_eta_after_gmt_dr_pt10","h_eta_after_gmt_dr_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_gmt_dr_nocsc_pt10 = fs->make<TH1D>("h_eta_after_gmt_dr_nocsc_pt10","h_eta_after_gmt_dr_nocsc_pt10",N_ETA_BINS, ETA_START, ETA_END);

  h_eta_vs_bx_after_alct = fs->make<TH2D>("h_eta_vs_bx_after_alct","h_eta_vs_bx_after_alct",N_ETA_BINS, ETA_START, ETA_END,13,-6.5, 6.5);
  h_eta_vs_bx_after_clct = fs->make<TH2D>("h_eta_vs_bx_after_clct","h_eta_vs_bx_after_clct",N_ETA_BINS, ETA_START, ETA_END,13,-6.5, 6.5);
  h_eta_vs_bx_after_lct = fs->make<TH2D>("h_eta_vs_bx_after_lct","h_eta_vs_bx_after_lct",N_ETA_BINS, ETA_START, ETA_END,13,-6.5, 6.5);
  h_eta_vs_bx_after_mpc = fs->make<TH2D>("h_eta_vs_bx_after_mpc","h_eta_vs_bx_after_mpc",N_ETA_BINS, ETA_START, ETA_END,13,-6.5, 6.5);

  h_eta_me1_after_alct = fs->make<TH1D>("h_eta_me1_after_alct","h_eta_me1_after_alct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_alct_okAlct = fs->make<TH1D>("h_eta_me1_after_alct_okAlct","h_eta_me1_after_alct_okAlct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_clct = fs->make<TH1D>("h_eta_me1_after_clct","h_eta_me1_after_clct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_clct_okClct = fs->make<TH1D>("h_eta_me1_after_clct_okClct","h_eta_me1_after_clct_okClct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_alctclct = fs->make<TH1D>("h_eta_me1_after_alctclct","h_eta_me1_after_alctclct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_alctclct_okAlct = fs->make<TH1D>("h_eta_me1_after_alctclct_okAlct","h_eta_me1_after_alctclct_okAlct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_alctclct_okClct = fs->make<TH1D>("h_eta_me1_after_alctclct_okClct","h_eta_me1_after_alctclct_okClct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_alctclct_okAlctClct = fs->make<TH1D>("h_eta_me1_after_alctclct_okAlctClct","h_eta_me1_after_alctclct_okAlctClct",N_ETA_BINS, ETA_START, ETA_END);

  h_eta_me1_after_lct = fs->make<TH1D>("h_eta_me1_after_lct","h_eta_me1_after_lct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_lct_okAlct = fs->make<TH1D>("h_eta_me1_after_lct_okAlct","h_eta_me1_after_lct_okAlct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_lct_okAlctClct = fs->make<TH1D>("h_eta_me1_after_lct_okAlctClct","h_eta_me1_after_lct_okAlctClct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_lct_okClct = fs->make<TH1D>("h_eta_me1_after_lct_okClct","h_eta_me1_after_lct_okClct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_lct_okClctAlct = fs->make<TH1D>("h_eta_me1_after_lct_okClctAlct","h_eta_me1_after_lct_okClctAlct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_mplct_okAlctClct = fs->make<TH1D>("h_eta_me1_after_mplct_okAlctClct","h_eta_me1_after_mplct_okAlctClct",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_mplct_okAlctClct_plus = fs->make<TH1D>("h_eta_me1_after_mplct_okAlctClct_plus","h_eta_me1_after_mplct_okAlctClct_plus",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok = fs->make<TH1D>("h_eta_me1_after_tf_ok","h_eta_me1_after_tf_ok",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_pt10 = fs->make<TH1D>("h_eta_me1_after_tf_ok_pt10","h_eta_me1_after_tf_ok_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_plus = fs->make<TH1D>("h_eta_me1_after_tf_ok_plus","h_eta_me1_after_tf_ok_plus",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_plus_pt10 = fs->make<TH1D>("h_eta_me1_after_tf_ok_plus_pt10","h_eta_me1_after_tf_ok_plus_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_plus_q[0] = fs->make<TH1D>("h_eta_me1_after_tf_ok_plus_q1","h_eta_me1_after_tf_ok_plus_q1",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_plus_q[1] = fs->make<TH1D>("h_eta_me1_after_tf_ok_plus_q2","h_eta_me1_after_tf_ok_plus_q2",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_plus_q[2] = fs->make<TH1D>("h_eta_me1_after_tf_ok_plus_q3","h_eta_me1_after_tf_ok_plus_q3",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_plus_pt10_q[0] = fs->make<TH1D>("h_eta_me1_after_tf_ok_plus_pt10_q1","h_eta_me1_after_tf_ok_plus_pt10_q1",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_plus_pt10_q[1] = fs->make<TH1D>("h_eta_me1_after_tf_ok_plus_pt10_q2","h_eta_me1_after_tf_ok_plus_pt10_q2",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_plus_pt10_q[2] = fs->make<TH1D>("h_eta_me1_after_tf_ok_plus_pt10_q3","h_eta_me1_after_tf_ok_plus_pt10_q3",N_ETA_BINS, ETA_START, ETA_END);

  h_eta_me1_after_mplct_ok = fs->make<TH1D>("h_eta_me1_after_mplct_ok","h_eta_me1_after_mplct_ok",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me2_after_mplct_ok = fs->make<TH1D>("h_eta_me2_after_mplct_ok","h_eta_me2_after_mplct_ok",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me3_after_mplct_ok = fs->make<TH1D>("h_eta_me3_after_mplct_ok","h_eta_me3_after_mplct_ok",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me4_after_mplct_ok = fs->make<TH1D>("h_eta_me4_after_mplct_ok","h_eta_me4_after_mplct_ok",N_ETA_BINS, ETA_START, ETA_END);


  h_eta_after_mpc_ok_plus_3st1a = fs->make<TH1D>("h_eta_after_mpc_ok_plus_3st1a","h_eta_after_mpc_ok_plus_3st1a",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_plus_3st1a= fs->make<TH1D>("h_eta_after_tfcand_ok_plus_3st1a","h_eta_after_tfcand_ok_plus_3st1a",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_tfcand_ok_plus_pt10_3st1a = fs->make<TH1D>("h_eta_after_tfcand_ok_plus_pt10_3st1a","h_eta_after_tfcand_ok_plus_pt10_3st1a",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_plus_3st1a= fs->make<TH1D>("h_eta_me1_after_tf_ok_plus_3st1a","h_eta_me1_after_tf_ok_plus_3st1a",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_me1_after_tf_ok_plus_pt10_3st1a = fs->make<TH1D>("h_eta_me1_after_tf_ok_plus_pt10_3st1a","h_eta_me1_after_tf_ok_plus_pt10_3st1a",N_ETA_BINS, ETA_START, ETA_END);


  h_bx_after_alct = fs->make<TH1D>("h_bx_after_alct","h_bx_after_alct",13,-6.5, 6.5);
  h_bx_after_clct = fs->make<TH1D>("h_bx_after_clct","h_bx_after_clct",13,-6.5, 6.5);
  h_bx_after_lct = fs->make<TH1D>("h_bx_after_lct","h_bx_after_lct",13,-6.5, 6.5);
  h_bx_after_mpc = fs->make<TH1D>("h_bx_after_mpc","h_bx_after_mpc",13,-6.5, 6.5);

  h_eta_after_xtra = fs->make<TH1D>("h_eta_after_xtra","h_eta_after_xtra",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_xtra_all = fs->make<TH1D>("h_eta_after_xtra_all","h_eta_after_xtra_all",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_xtra_dr = fs->make<TH1D>("h_eta_after_xtra_dr","h_eta_after_xtra_dr",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_xtra_pt10 = fs->make<TH1D>("h_eta_after_xtra_pt10","h_eta_after_xtra_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_xtra_all_pt10 = fs->make<TH1D>("h_eta_after_xtra_all_pt10","h_eta_after_xtra_all_pt10",N_ETA_BINS, ETA_START, ETA_END);
  h_eta_after_xtra_dr_pt10 = fs->make<TH1D>("h_eta_after_xtra_dr_pt10","h_eta_after_xtra_dr_pt10",N_ETA_BINS, ETA_START, ETA_END);

  h_wg_me11_initial = fs->make<TH1D>("h_wg_me11_initial","h_wg_me11_initial",50, -1,49);
  h_wg_me11_after_alct_okAlct = fs->make<TH1D>("h_wg_me11_after_alct_okAlct","h_wg_me11_after_alct_okAlct",50, -1,49);
  h_wg_me11_after_alctclct_okAlctClct = fs->make<TH1D>("h_wg_me11_after_alctclct_okAlctClct","h_wg_me11_after_alctclct_okAlctClct",50, -1,49);
  h_wg_me11_after_lct_okAlctClct = fs->make<TH1D>("h_wg_me11_after_lct_okAlctClct","h_wg_me11_after_lct_okAlctClct",50, -1,49);

  h_phi_initial  = fs->make<TH1D>("h_phi_initial","h_phi_initial",128, -M_PI,M_PI);
  h_phi_after_alct  = fs->make<TH1D>("h_phi_after_alct","h_phi_after_alct",128, -M_PI,M_PI);
  h_phi_after_clct  = fs->make<TH1D>("h_phi_after_clct","h_phi_after_clct",128, -M_PI,M_PI);
  h_phi_after_lct  = fs->make<TH1D>("h_phi_after_lct","h_phi_after_lct",128, -M_PI,M_PI);
  h_phi_after_mpc  = fs->make<TH1D>("h_phi_after_mpc","h_phi_after_mpc",128, -M_PI,M_PI);
  h_phi_after_tftrack  = fs->make<TH1D>("h_phi_after_tftrack","h_phi_after_tftrack",128, -M_PI,M_PI);
  h_phi_after_tfcand  = fs->make<TH1D>("h_phi_after_tfcand","h_phi_after_tfcand",128, -M_PI,M_PI);
  h_phi_after_tfcand_all = fs->make<TH1D>("h_phi_after_tfcand_all","h_phi_after_tfcand_all",128, -M_PI,M_PI);
  h_phi_after_gmtreg = fs->make<TH1D>("h_phi_after_gmtreg","h_phi_after_gmtreg",128, -M_PI,M_PI);
  h_phi_after_gmtreg_all = fs->make<TH1D>("h_phi_after_gmtreg_all","h_phi_after_gmtreg_all",128, -M_PI,M_PI);
  h_phi_after_gmtreg_dr = fs->make<TH1D>("h_phi_after_gmtreg_dr","h_phi_after_gmtreg_dr",128, -M_PI,M_PI);
  h_phi_after_gmt = fs->make<TH1D>("h_phi_after_gmt","h_phi_after_gmt",128, -M_PI,M_PI);
  h_phi_after_gmt_all = fs->make<TH1D>("h_phi_after_gmt_all","h_phi_after_gmt_all",128, -M_PI,M_PI);
  h_phi_after_gmt_dr = fs->make<TH1D>("h_phi_after_gmt_dr","h_phi_after_gmt_dr",128, -M_PI,M_PI);
  h_phi_after_gmt_dr_nocsc = fs->make<TH1D>("h_phi_after_gmt_dr_nocsc","h_phi_after_gmt_dr_nocsc",128, -M_PI,M_PI);
  h_phi_after_xtra = fs->make<TH1D>("h_phi_after_xtra","h_phi_after_xtra",128, -M_PI,M_PI);
  h_phi_after_xtra_all = fs->make<TH1D>("h_phi_after_xtra_all","h_phi_after_xtra_all",128, -M_PI,M_PI);
  h_phi_after_xtra_dr = fs->make<TH1D>("h_phi_after_xtra_dr","h_phi_after_xtra_dr",128, -M_PI,M_PI);

  h_qu_alct = fs->make<TH1D>("h_qu_alct","h_qu_alct",17,-0.5, 16.5);
  h_qu_clct = fs->make<TH1D>("h_qu_clct","h_qu_clct",17,-0.5, 16.5);
  h_qu_lct = fs->make<TH1D>("h_qu_lct","h_qu_lct",17,-0.5, 16.5);
  h_qu_mplct = fs->make<TH1D>("h_qu_mplct","h_qu_mplct",17,-0.5, 16.5);
  h_qu_vs_bx__alct = fs->make<TH2D>("h_qu_vs_bx__alct","h_qu_vs_bx__alct",17,-0.5, 16.5, 15,-7.5, 7.5);
  h_qu_vs_bx__clct = fs->make<TH2D>("h_qu_vs_bx__clct","h_qu_vs_bx__clct",17,-0.5, 16.5, 15,-7.5, 7.5);
  h_qu_vs_bx__lct = fs->make<TH2D>("h_qu_vs_bx__lct","h_qu_vs_bx__lct",17,-0.5, 16.5, 15,-7.5, 7.5);
  h_qu_vs_bx__mplct = fs->make<TH2D>("h_qu_vs_bx__mplct","h_qu_vs_bx__mplct",17,-0.5, 16.5, 15,-7.5, 7.5);

  h_qu_vs_bxclct__lct = fs->make<TH2D>("h_qu_vs_bxclct__lct","h_qu_vs_bxclct__lct",17,-0.5, 16.5, 15,-7.5, 7.5);

  h_tf_n_uncommon_stubs = fs->make<TH1D>("h_tf_n_uncommon_stubs","h_tf_n_uncommon_stubs",16, 0.,16.);
  h_tf_n_stubs = fs->make<TH1D>("h_tf_n_stubs","h_tf_n_stubs",16, 0.,16.);
  h_tf_n_matchstubs = fs->make<TH1D>("h_tf_n_matchstubs","h_tf_n_matchstubs",16, 0.,16.);
  h_tf_n_stubs_vs_matchstubs = fs->make<TH2D>("h_tf_n_stubs_vs_matchstubs","h_tf_n_stubs_vs_matchstubs",16, 0.,16.,16, 0.,16.);

  h_tfpt = fs->make<TH1D>("h_tfpt","h_tfpt",300, 0.,150.);
  h_tfeta = fs->make<TH1D>("h_tfeta","h_tfeta",500,-2.5, 2.5);
  h_tfphi = fs->make<TH1D>("h_tfphi","h_tfphi",128*5, -M_PI,M_PI);
  h_tfbx = fs->make<TH1D>("h_tfbx","h_tfbx",13,-6.5, 6.5);
  h_tfqu = fs->make<TH1D>("h_tfqu","h_tfqu",5,-0.5, 4.5);
  h_tfdr = fs->make<TH1D>("h_tfdr","h_tfdr",120,0., 0.6);
  h_tfpt_vs_qu = fs->make<TH2D>("h_tfpt_vs_qu","h_tfpt_vs_qu",N_PT_BINS*2, 0.,150.,5,-0.5, 4.5);

  h_tf_mode = fs->make<TH1D>("h_tf_mode","TF Track Mode", 16, -0.5, 15.5);
  setupTFModeHisto(h_tf_mode);
  h_tf_mode->SetTitle("TF Track Mode (SimTrack match)");

  h_tf_pt_h42_2st = fs->make<TH1D>("h_tf_pt_h42_2st","h_tf_pt_h42_2st",300, 0.,150.);
  h_tf_pt_h42_3st = fs->make<TH1D>("h_tf_pt_h42_3st","h_tf_pt_h42_3st",300, 0.,150.);
  h_tf_pt_h42_2st_w = fs->make<TH1D>("h_tf_pt_h42_2st_w","h_tf_pt_h42_2st_w",300, 0.,150.);
  h_tf_pt_h42_3st_w = fs->make<TH1D>("h_tf_pt_h42_3st_w","h_tf_pt_h42_3st_w",300, 0.,150.);


  h_tf_check_mode = fs->make<TH1D>("h_tf_check_mode","h_tf_check_mode", 16, -0.5, 15.5);
  setupTFModeHisto(h_tf_check_mode);
  h_tf_check_bx = fs->make<TH1D>("h_tf_check_bx","h_tf_check_bx", 13,-6.5, 6.5);
  h_tf_check_n_stubs_vs_matched = fs->make<TH2D>("h_tf_check_n_stubs_vs_matched","h_tf_check_n_stubs_vs_matched",16, 0.,16.,16, 0.,16.);
  h_tf_check_st1_wg = fs->make<TH1D>("h_tf_check_st1_wg","h_tf_check_st1_wg", 96, 0., 96.);
  h_tf_check_st1_strip = fs->make<TH1D>("h_tf_check_st1_strip","h_tf_check_st1_strip", 100, 0., 100.);
  h_tf_check_st1_mcStrip = fs->make<TH1D>("h_tf_check_st1_mcStrip","h_tf_check_st1_mcStrip", 100, 0., 100.);
  h_tf_check_st1_mcStrip_vs_ptbin = fs->make<TH2D>("h_tf_check_st1_mcStrip_vs_ptbin","h_tf_check_st1_mcStrip_vs_ptbin", 100, 0., 100., 34,0.,34.);
  h_tf_check_st1_mcStrip_vs_ptbin_all = fs->make<TH2D>("h_tf_check_st1_mcStrip_vs_ptbin_all","h_tf_check_st1_mcStrip_vs_ptbin_all", 100, 0., 100., 34,0.,34.);
  h_tf_check_st1_mcWG_vs_ptbin_all = fs->make<TH2D>("h_tf_check_st1_mcWG_vs_ptbin_all","h_tf_check_st1_mcWG_vs_ptbin_all", 50, 0., 50., 34,0.,34.);
  h_tf_check_st1_chamber = fs->make<TH1D>("h_tf_check_st1_chamber","h_tf_check_st1_chamber", 37,0., 37.);
  
  
  h_gmtpt = fs->make<TH1D>("h_gmtpt","h_gmtpt",300, 0.,150.);
  h_gmteta = fs->make<TH1D>("h_gmteta","h_gmteta",500,-2.5, 2.5);
  h_gmtphi = fs->make<TH1D>("h_gmtphi","h_gmtphi",128*5, -M_PI,M_PI);
  h_gmtbx = fs->make<TH1D>("h_gmtbx","h_gmtbx",13,-6.5, 6.5);
  h_gmtrank = fs->make<TH1D>("h_gmtrank","h_gmtrank",250,-0.001, 250-0.001);
  h_gmtqu = fs->make<TH1D>("h_gmtqu","h_gmtqu",11,-0.5, 10.5);
  h_gmtisrpc = fs->make<TH1D>("h_gmtisrpc","h_gmtisrpc",4,-0.5, 3.5);
  h_gmtdr = fs->make<TH1D>("h_gmtdr","h_gmtdr",120,0., 0.6);

  h_gmtxpt = fs->make<TH1D>("h_gmtxpt","h_gmtxpt",300, 0.,150.);
  h_gmtxeta = fs->make<TH1D>("h_gmtxeta","h_gmtxeta",500,-2.5, 2.5);
  h_gmtxphi = fs->make<TH1D>("h_gmtxphi","h_gmtxphi",128*5, -M_PI,M_PI);
  h_gmtxbx = fs->make<TH1D>("h_gmtxbx","h_gmtxbx",13,-6.5, 6.5);
  h_gmtxrank = fs->make<TH1D>("h_gmtxrank","h_gmtxrank",250,-0.001, 250-0.001);
  h_gmtxqu = fs->make<TH1D>("h_gmtxqu","h_gmtxqu",11,-0.5, 10.5);
  h_gmtxisrpc = fs->make<TH1D>("h_gmtxisrpc","h_gmtxisrpc",4,-0.5, 3.5);
  h_gmtxdr = fs->make<TH1D>("h_gmtxdr","h_gmtxdr",120,0., 0.6);

  h_gmtxpt_nocsc = fs->make<TH1D>("h_gmtxpt_nocsc","h_gmtxpt_nocsc",300, 0.,150.);
  h_gmtxeta_nocsc = fs->make<TH1D>("h_gmtxeta_nocsc","h_gmtxeta_nocsc",500,-2.5, 2.5);
  h_gmtxphi_nocsc = fs->make<TH1D>("h_gmtxphi_nocsc","h_gmtxphi_nocsc",128*5, -M_PI,M_PI);
  h_gmtxbx_nocsc = fs->make<TH1D>("h_gmtxbx_nocsc","h_gmtxbx_nocsc",13,-6.5, 6.5);
  h_gmtxrank_nocsc = fs->make<TH1D>("h_gmtxrank_nocsc","h_gmtxrank_nocsc",250,0, 250);
  h_gmtxqu_nocsc = fs->make<TH1D>("h_gmtxqu_nocsc","h_gmtxqu_nocsc",11,-0.5, 10.5);
  h_gmtxisrpc_nocsc = fs->make<TH1D>("h_gmtxisrpc_nocsc","h_gmtxisrpc_nocsc",4,-0.5, 3.5);
  h_gmtxdr_nocsc = fs->make<TH1D>("h_gmtxdr_nocsc","h_gmtxdr_nocsc",120,0., 0.6);

  h_gmtxqu_nogmtreg = fs->make<TH1D>("h_gmtxqu_nogmtreg","h_gmtxqu_nogmtreg",11,-0.5, 10.5);
  h_gmtxisrpc_nogmtreg = fs->make<TH1D>("h_gmtxisrpc_nogmtreg","h_gmtxisrpc_nogmtreg",4,-0.5, 3.5);
  h_gmtxqu_notfcand = fs->make<TH1D>("h_gmtxqu_notfcand","h_gmtxqu_notfcand",11,-0.5, 10.5);
  h_gmtxisrpc_notfcand = fs->make<TH1D>("h_gmtxisrpc_notfcand","h_gmtxisrpc_notfcand",4,-0.5, 3.5);
  h_gmtxqu_nompc = fs->make<TH1D>("h_gmtxqu_nompc","h_gmtxqu_nompc",11,-0.5, 10.5);
  h_gmtxisrpc_nompc = fs->make<TH1D>("h_gmtxisrpc_nompc","h_gmtxisrpc_nompc",4,-0.5, 3.5);

  h_xtrapt = fs->make<TH1D>("h_xtrapt","h_xtrapt",300, 0.,150.);
  h_xtraeta = fs->make<TH1D>("h_xtraeta","h_xtraeta",500,-2.5, 2.5);
  h_xtraphi = fs->make<TH1D>("h_xtraphi","h_xtraphi",128*5, -M_PI,M_PI);
  //  h_xtrabx = fs->make<TH1D>("h_xtrabx","h_xtrabx",13,-6.5, 6.5);
  h_xtradr = fs->make<TH1D>("h_xtradr","h_xtradr",120,0., 0.6);

  h_n_alct = fs->make<TH1D>("h_n_alct", "h_n_alct", 9,-0.5,8.5);
  h_n_clct = fs->make<TH1D>("h_n_clct", "h_n_clct", 9,-0.5,8.5);
  h_n_lct = fs->make<TH1D>("h_n_lct", "h_n_lct", 9,-0.5,8.5);
  h_n_mplct = fs->make<TH1D>("h_n_mplct", "h_n_mplct", 9,-0.5,8.5);
  h_n_tftrack = fs->make<TH1D>("h_n_tftrack", "h_n_tftrack", 6,-0.5,5.5);
  h_n_tftrack_all = fs->make<TH1D>("h_n_tftrack_all", "h_n_tftrack_all", 6,-0.5,5.5);
  h_n_tfcand = fs->make<TH1D>("h_n_tfcand", "h_n_tfcand", 6,-0.5,5.5);
  h_n_tfcand_all = fs->make<TH1D>("h_n_tfcand_all", "h_n_tfcand_all", 6,-0.5,5.5);
  h_n_gmtregcand = fs->make<TH1D>("h_n_gmtregcand", "h_n_gmtregcand", 6,-0.5,5.5);
  h_n_gmtregcand_all = fs->make<TH1D>("h_n_gmtregcand_all", "h_n_gmtregcand_all", 6,-0.5,5.5);
  h_n_gmtcand = fs->make<TH1D>("h_n_gmtcand", "h_n_gmtcand", 6,-0.5,5.5);
  h_n_gmtcand_all = fs->make<TH1D>("h_n_gmtcand_all", "h_n_gmtcand_all", 6,-0.5,5.5);
  h_n_xtra = fs->make<TH1D>("h_n_xtra", "h_n_xtra", 6,-0.5,5.5);
  h_n_xtra_all = fs->make<TH1D>("h_n_xtra_all", "h_n_xtra_all", 6,-0.5,5.5);

  h_n_ch_w_alct = fs->make<TH1D>("h_n_ch_w_alct", "h_n_ch_w_alct", 9,-0.5,8.5);
  h_n_ch_w_clct = fs->make<TH1D>("h_n_ch_w_clct", "h_n_ch_w_clct", 9,-0.5,8.5);
  h_n_ch_w_lct = fs->make<TH1D>("h_n_ch_w_lct", "h_n_ch_w_lct", 9,-0.5,8.5);
  h_n_ch_w_mplct = fs->make<TH1D>("h_n_ch_w_mplct", "h_n_ch_w_mplct", 9,-0.5,8.5);


  h_pt_over_tfpt_resol = fs->make<TH1D>("h_pt_over_pttf_resol","h_pt_over_pttf_resol",300, -1.5,1.5);
  h_pt_over_tfpt_resol_vs_pt = fs->make<TH2D>("h_pt_over_pttf_resol_vs_pt","h_pt_over_pttf_resol_vs_pt",150, -1.5,1.5,N_PT_BINS, PT_START, PT_END);

  h_eta_minus_tfeta_resol = fs->make<TH1D>("h_eta_minus_tfeta_resol","h_eta_minus_tfeta_resol",N_ETA_BINS,-1., 1.);
  h_phi_minus_tfphi_resol = fs->make<TH1D>("h_phi_minus_tfphi_resol","h_phi_minus_tfphi_resol",200,-1., 1.);



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
  h_rt_gmt_csc_ptmax20_eta_3s_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_no1a","h_rt_gmt_csc_ptmax20_eta_3s_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s1b","h_rt_gmt_csc_ptmax20_eta_3s_2s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s1b_1b","h_rt_gmt_csc_ptmax20_eta_3s_2s1b_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s123_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s123_1b","h_rt_gmt_csc_ptmax20_eta_3s_2s123_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s13_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s13_1b","h_rt_gmt_csc_ptmax20_eta_3s_2s13_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s1b_no1a","h_rt_gmt_csc_ptmax20_eta_3s_2s1b_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s123_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s123_no1a","h_rt_gmt_csc_ptmax20_eta_3s_2s123_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_2s13_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_2s13_no1a","h_rt_gmt_csc_ptmax20_eta_3s_2s13_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_3s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_3s1b","h_rt_gmt_csc_ptmax20_eta_3s_3s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_3s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_3s1b_1b","h_rt_gmt_csc_ptmax20_eta_3s_3s1b_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3s_3s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3s_3s1b_no1a","h_rt_gmt_csc_ptmax20_eta_3s_3s1b_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_2q = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_2q","h_rt_gmt_csc_ptmax20_eta_2q",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax20_eta_3q = fs->make<TH1D>("h_rt_gmt_csc_ptmax20_eta_3q","h_rt_gmt_csc_ptmax20_eta_3q",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);

  h_rt_gmt_csc_ptmax30_eta_2s = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_2s","h_rt_gmt_csc_ptmax30_eta_2s",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_2s_2s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_2s_2s1b","h_rt_gmt_csc_ptmax30_eta_2s_2s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s","h_rt_gmt_csc_ptmax30_eta_3s",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_1b","h_rt_gmt_csc_ptmax30_eta_3s_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_no1a","h_rt_gmt_csc_ptmax30_eta_3s_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s1b","h_rt_gmt_csc_ptmax30_eta_3s_2s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s1b_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s1b_1b","h_rt_gmt_csc_ptmax30_eta_3s_2s1b_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s123_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s123_1b","h_rt_gmt_csc_ptmax30_eta_3s_2s123_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s13_1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s13_1b","h_rt_gmt_csc_ptmax30_eta_3s_2s13_1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s1b_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s1b_no1a","h_rt_gmt_csc_ptmax30_eta_3s_2s1b_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s123_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s123_no1a","h_rt_gmt_csc_ptmax30_eta_3s_2s123_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_2s13_no1a = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_2s13_no1a","h_rt_gmt_csc_ptmax30_eta_3s_2s13_no1a",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
  h_rt_gmt_csc_ptmax30_eta_3s_3s1b = fs->make<TH1D>("h_rt_gmt_csc_ptmax30_eta_3s_3s1b","h_rt_gmt_csc_ptmax30_eta_3s_3s1b",N_ETA_BINS_CSC, ETA_START_CSC, ETA_END_CSC);
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

  for (int i = 1; i < Nthr; ++i) {
    string prefix = "h_rt_gmt_csc_mode_2s1b_1b_";
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

      //    sprintf(label,"_cscdet_%s",csc_type_[me].c_str());
      //    _cscdet[me] = fs->make<TH1D>(label, label, 15,-7.5, 7.5);

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

      //sprintf(label,"_st%d",i+1);
      //_st[i]  = fs->make<TH2D>(label, label, 20,0., 20.,16,0,16);
    }

  h_rt_mplct_pattern = fs->make<TH1D>("h_rt_mplct_pattern","h_rt_mplct_pattern",13,-0.5, 12.5);

}


// ================================================================================================
SimMuL1::~SimMuL1()
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

  if (theStripConditions) delete theStripConditions;
  theStripConditions = 0;
}


//
// member functions
//
// ================================================================================================
void 
SimMuL1::bookDbgTTree()
{
  // see http://cmslxr.fnal.gov/lxr/source/Calibration/IsolatedParticles/plugins/IsolatedTracksCone.cc
  Service<TFileService> fs;
  dbg_tree = fs->make<TTree>("dbg_tree", "dbg_tree");
  dbg_tree->Branch("evtn", &dbg_.evtn, "evtn/I");
  dbg_tree->Branch("trkn", &dbg_.trkn, "trkn/I");
  dbg_tree->Branch("pt", &dbg_.pt, "pt/F");
  dbg_tree->Branch("eta", &dbg_.eta, "eta/F");
  dbg_tree->Branch("phi", &dbg_.phi, "phi/F");
  dbg_tree->Branch("tfpt", &dbg_.tfpt, "tfpt/F");
  dbg_tree->Branch("tfeta", &dbg_.tfeta, "tfeta/F");
  dbg_tree->Branch("tfphi", &dbg_.tfphi, "tfphi/F");
  dbg_tree->Branch("tfpt_packed", &dbg_.tfpt_packed, "tfpt_packed/I");
  dbg_tree->Branch("tfeta_packed", &dbg_.tfeta_packed, "tfeta_packed/I");
  dbg_tree->Branch("tfphi_packed", &dbg_.tfphi_packed, "tfphi_packed/I");
  dbg_tree->Branch("dPhi12", &dbg_.dPhi12, "dPhi12/I");
  dbg_tree->Branch("dPhi23", &dbg_.dPhi23, "dPhi23/I");
  dbg_tree->Branch("nseg", &dbg_.nseg, "nseg/I");
  dbg_tree->Branch("nseg_ok", &dbg_.nseg_ok, "nseg_ok/I");
  dbg_tree->Branch("meEtap", &dbg_.meEtap, "meEtap/I");
  dbg_tree->Branch("mePhip", &dbg_.mePhip, "mePhip/I");
  dbg_tree->Branch("mcStrip", &dbg_.mcStrip, "mcStrip/I");
  dbg_tree->Branch("mcWG", &dbg_.mcWG, "mcWG/I");
  dbg_tree->Branch("strip", &dbg_.strip, "strip/I");
  dbg_tree->Branch("wg", &dbg_.wg, "wg/I");
  dbg_tree->Branch("chamber", &dbg_.chamber, "chamber/I");
  //dbg_tree->Branch("", &dbg_., "/I");
  //dbg_tree->Branch("" , "vector<double>" , & );
}

void
SimMuL1::resetDbg(DbgStruct& d)
{
  d.evtn = d.trkn = -1;
  d.pt = d.eta = d.phi = d.tfpt = d.tfeta = d.tfphi = -1.;
  d.tfpt_packed = d.tfeta_packed = d.tfphi_packed = d.dPhi12 = d.dPhi23 = d.nseg = d.nseg_ok = -1;
  d.meEtap = d.mePhip = d.mcStrip = d.mcWG = d.strip = d.wg = d.chamber = -1;
}


// ================================================================================================
// ------------ method called to for each event  ------------
bool SimMuL1::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  nevt++;
  
  if (addGhostLCTs_)
    {
      for (size_t i=0; i<ghostLCTs.size();i++) if (ghostLCTs[i]) delete ghostLCTs[i];
      ghostLCTs.clear();
    }

  ESHandle< CSCGeometry > cscGeom;
  
  iSetup.get< MuonGeometryRecord >().get(cscGeom);
  iSetup.get<MuonRecoGeometryRecord>().get(muonGeometry);

  cscGeometry = &*cscGeom;
  
  CSCTriggerGeometry::setGeometry(cscGeometry);

  // get conditions for bad chambers (don't need random engine)
  theStripConditions->initializeEvent(iSetup);


  //Get the Magnetic field from the setup
  iSetup.get<IdealMagneticFieldRecord>().get(theBField);


  // Get the propagators
  iSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAnyRK", propagatorAlong);
  iSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAnyOpposite", propagatorOpposite);


  // get MC
  Handle< GenParticleCollection > hMCCand;
  iEvent.getByLabel("genParticles", hMCCand);
  const GenParticleCollection & cands  = *(hMCCand.product()); 
  
  // get SimTracks
  Handle< SimTrackContainer > hSimTracks;
  iEvent.getByLabel("g4SimHits", hSimTracks);
  const SimTrackContainer & simTracks = *(hSimTracks.product());

  // get simVertices
  Handle< SimVertexContainer > hSimVertices;
  //iEvent.getByType<SimVertexContainer>(hSimVertices);
  iEvent.getByLabel("g4SimHits", hSimVertices);
  const SimVertexContainer & simVertices = *(hSimVertices.product());

  // get SimHits
  theCSCSimHitMap.fill(iEvent);

  edm::Handle< PSimHitContainer > MuonCSCHits;
  iEvent.getByLabel("g4SimHits", "MuonCSCHits", MuonCSCHits);
  const PSimHitContainer* allCSCSimHits = MuonCSCHits.product();

  // strip and wire digis
  Handle< CSCWireDigiCollection >       wireDigis;
  Handle< CSCComparatorDigiCollection > compDigis;
  iEvent.getByLabel("simMuonCSCDigis","MuonCSCWireDigi",       wireDigis);
  iEvent.getByLabel("simMuonCSCDigis","MuonCSCComparatorDigi", compDigis);
  const CSCWireDigiCollection* wiredc = wireDigis.product();
  const CSCComparatorDigiCollection* compdc = compDigis.product();

  // ALCTs and CLCTs
  Handle< CSCALCTDigiCollection > halcts;
  Handle< CSCCLCTDigiCollection > hclcts;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  halcts);
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  hclcts);
  const CSCALCTDigiCollection* alcts = halcts.product();
  const CSCCLCTDigiCollection* clcts = hclcts.product();

  // strip&wire matching output  after TMB  and after MPC sorting
  Handle< CSCCorrelatedLCTDigiCollection > lcts_tmb;
  Handle< CSCCorrelatedLCTDigiCollection > lcts_mpc;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  lcts_tmb);
  iEvent.getByLabel("simCscTriggerPrimitiveDigis", "MPCSORTED", lcts_mpc);
  const CSCCorrelatedLCTDigiCollection* lcts = lcts_tmb.product();
  const CSCCorrelatedLCTDigiCollection* mplcts = lcts_mpc.product();
  
  // DT primitives for input to TF
  Handle<L1MuDTChambPhContainer> dttrig;
  iEvent.getByLabel("simDtTriggerPrimitiveDigis", dttrig);
  const L1MuDTChambPhContainer* dttrigs = dttrig.product();

  // tracks produced by TF
  Handle< L1CSCTrackCollection > hl1Tracks;
  iEvent.getByLabel("simCsctfTrackDigis",hl1Tracks);
  const L1CSCTrackCollection* l1Tracks = hl1Tracks.product();

  // L1 muon candidates after CSC sorter
  Handle< vector< L1MuRegionalCand > > hl1TfCands;
  iEvent.getByLabel("simCsctfDigis", "CSC", hl1TfCands);
  const vector< L1MuRegionalCand > *l1TfCands = hl1TfCands.product();

  // GMT readout collection
  Handle< L1MuGMTReadoutCollection > hl1GmtCands;
  if (!lightRun) iEvent.getByLabel("simGmtDigis", hl1GmtCands ) ;// InputTag("simCsctfDigis","CSC")
  //const L1MuGMTReadoutCollection* l1GmtCands = hl1GmtCands.product();
  vector<L1MuGMTExtendedCand> l1GmtCands;
  vector<L1MuGMTExtendedCand> l1GmtfCands;
  vector<L1MuRegionalCand>    l1GmtCSCCands;
  vector<L1MuRegionalCand>    l1GmtRPCfCands;
  vector<L1MuRegionalCand>    l1GmtRPCbCands;
  vector<L1MuRegionalCand>    l1GmtDTCands;

  // key = BX
  map<int, vector<L1MuRegionalCand> >  l1GmtCSCCandsInBXs;

  // TOCHECK
  if( !lightRun )
    {
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
	  vector<L1MuGMTReadoutRecord> gmt_records = hl1GmtCands->getRecords();
	  for ( vector< L1MuGMTReadoutRecord >::const_iterator rItr=gmt_records.begin(); rItr!=gmt_records.end() ; ++rItr )
	    {
	      if (rItr->getBxInEvent() < minBxGMT_ || rItr->getBxInEvent() > maxBxGMT_) continue;

	      vector<L1MuGMTExtendedCand> GMTCands = rItr->getGMTCands();
	      for ( vector<L1MuGMTExtendedCand>::const_iterator  cItr = GMTCands.begin() ; cItr != GMTCands.end() ; ++cItr )
		if (!cItr->empty()) l1GmtCands.push_back(*cItr);

	      vector<L1MuGMTExtendedCand> GMTfCands = rItr->getGMTFwdCands();
	      for ( vector<L1MuGMTExtendedCand>::const_iterator  cItr = GMTfCands.begin() ; cItr != GMTfCands.end() ; ++cItr )
		if (!cItr->empty()) l1GmtfCands.push_back(*cItr);

	      //cout<<" ggg: "<<GMTCands.size()<<" "<<GMTfCands.size()<<endl;

	      vector<L1MuRegionalCand> CSCCands = rItr->getCSCCands();
	      l1GmtCSCCandsInBXs[rItr->getBxInEvent()] = CSCCands;
	      for ( vector<L1MuRegionalCand>::const_iterator  cItr = CSCCands.begin() ; cItr != CSCCands.end() ; ++cItr )
		if (!cItr->empty()) l1GmtCSCCands.push_back(*cItr);

	      vector<L1MuRegionalCand> RPCfCands = rItr->getFwdRPCCands();
	      for ( vector<L1MuRegionalCand>::const_iterator  cItr = RPCfCands.begin() ; cItr != RPCfCands.end() ; ++cItr )
		if (!cItr->empty()) l1GmtRPCfCands.push_back(*cItr);

	      vector<L1MuRegionalCand> RPCbCands = rItr->getBrlRPCCands();
	      for ( vector<L1MuRegionalCand>::const_iterator  cItr = RPCbCands.begin() ; cItr != RPCbCands.end() ; ++cItr )
		if (!cItr->empty()) l1GmtRPCbCands.push_back(*cItr);

	      vector<L1MuRegionalCand> DTCands = rItr->getDTBXCands();
	      for ( vector<L1MuRegionalCand>::const_iterator  cItr = DTCands.begin() ; cItr != DTCands.end() ; ++cItr )
		if (!cItr->empty()) l1GmtDTCands.push_back(*cItr);
	    }
	  //cout<<" sizes: "<<l1GmtCands.size()<<" "<<l1GmtfCands.size()<<" "<<l1GmtCSCCands.size()<<" "<<l1GmtRPCfCands.size()<<endl;
	}
    }


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


  // simple trigger muons from L1Extra
  Handle< l1extra::L1MuonParticleCollection > hl1Muons;
  const l1extra::L1MuonParticleCollection* l1Muons = 0;
  if (!lightRun) {
    iEvent.getByLabel("l1extraParticles", hl1Muons);
    l1Muons  = hl1Muons.product();
  }


  double mcpt, mceta, mcphi, stpt, steta, stphi ;
  int numberMCTr=0;

  double sim_eta[2], sim_phi[2];
  int sim_n=0;
  
  // check MC truch loop
  for ( size_t ic = 0; ic < cands.size(); ic++ )
    {
      const GenParticle * cand = &(cands[ic]);
    
      if ( abs(cand->pdgId()) == 13 &&  cand->status() == 1 )
	{
	  //cout<<" MC MUON!"<<endl;
	  numberMCTr++;
	  mceta = cand->eta();
	  mcphi = normalizedPhi( cand->phi() );
	  mcpt = cand->pt();
      
	  h_pt_mctr ->Fill(mcpt);
	  h_eta_mctr->Fill(mceta);
	  h_phi_mctr->Fill(mcphi);

	  if (fabs(mceta)>10) continue;
      
	  // match with SimTrack
	  SimTrackContainer::const_iterator matchSimTr = simTracks.end();
	  double minDeltaRSimTr = 999.;
	  int numberSimTr=0;
	  for (SimTrackContainer::const_iterator istrk = simTracks.begin(); istrk != simTracks.end(); ++istrk)
	    {
	      if ( abs(istrk->type()) != 13 ) continue;

	      numberSimTr++;
	      stpt = sqrt(istrk->momentum().perp2());
	      steta = istrk->momentum().eta();
	      stphi = normalizedPhi( istrk->momentum().phi() );
	
	      if (stpt<1.) continue;
	
	      //charge = static_cast<int> (-itrack->type()/13); //static_cast<int> (itrack->charge());
	      if (sim_n<2){
		sim_eta[sim_n] = steta;
		sim_phi[sim_n] = stphi;
	      }
	      sim_n++;
      
	      double dr = deltaR(mceta,mcphi,steta,stphi);
	      h_DR_mctr_simtr->Fill(dr);
	      if ( dr < minDeltaRSimTr ){
		matchSimTr = istrk;
		minDeltaRSimTr = dr;
	      }
	    }
      
	  h_N_simtr->Fill(numberSimTr);
      
	  if (matchSimTr == simTracks.end()) {
	    cout<<"+++ Warning: no matching sim track for MC track!"<<endl;
	    MYDEBUG = 1;
	  }
	  //      if (MYDEBUG){
	  //         cout<<"MC Cand "<<numberMCTr<<" \t eta phi pt :"<<mceta<<" "<<mcphi<<" "<<mcpt<<endl;
	  //         if (matchSimTr != simTracks.end()) cout<<"SimTrk\t\t eta phi pt dr : "<<matchSimTr->momentum().eta()<<" "<<normalizedPhi( matchSimTr->momentum().phi() )<<" "<<matchSimTr->momentum().Pt()<<" "<<minDeltaRSimTr<<endl;
	  //      }
	  if (matchSimTr == simTracks.end()) continue;
	  h_MinDR_mctr_simtr->Fill(minDeltaRSimTr);

	} // if muon MC ca
    } // MC cands loop

  h_N_mctr->Fill(numberMCTr);
  
  double deltaR2Tr = -1;
  if (sim_n>1)  {
    deltaR2Tr = deltaR(sim_eta[0],sim_phi[0],sim_eta[1],sim_phi[1]);
    h_DR_2SimTr->Fill(deltaR2Tr);
    if (deltaR2Tr>M_PI && debugALLEVENT) cout<<"PI<deltaR2Tr="<<deltaR2Tr<<endl;
    
    // select only well separated or close simtracks
    //if (fabs(minSimTrackDR_)>0.01){
    //  if (minSimTrackDR_>0. && deltaR2Tr < minSimTrackDR_ ) return true;
    //  if (minSimTrackDR_<0. && deltaR2Tr > fabs(minSimTrackDR_) ) return true;
    //}
  }

  // debuggin' 
  if (debugALLEVENT) {
    cout<<"--- detIDs with hits: "<<endl;
    vector<int> detIds = theCSCSimHitMap.detsWithHits();
    for (size_t di = 0; di < detIds.size(); di++) {
      CSCDetId layerId(detIds[di]);
      PSimHitContainer hits = theCSCSimHitMap.hits(detIds[di]);
      cout<<"   "<< detIds[di]<<" "<<layerId<<"   no. of hits = "<<hits.size()<<endl;

      const CSCLayer* csclayer = cscGeometry->layer(layerId);
      const CSCLayerGeometry* layerGeom = csclayer->geometry();

      for (unsigned j=0; j<hits.size(); j++) 
	{
	  LocalPoint hitLP = hits[j].localPosition();
	  GlobalPoint hitGP = csclayer->toGlobal(hitLP);
	  double hitEta = hitGP.eta();
	  double hitPhi = hitGP.phi();
	  cout<<"     "<<hitEta<<" "<<hitPhi<<"  "<<hits[j].entryPoint()<<" "<<hits[j].exitPoint()<<" "<<hits[j].particleType()<<" "<<hits[j].trackId()<<endl;
	}

      cout<<"     wire digis: etas"<<endl;
      const CSCWireDigiCollection::Range rwired = wiredc->get(layerId);
      for (CSCWireDigiCollection::const_iterator digiIt = rwired.first; digiIt != rwired.second; ++digiIt) 
	{
	  //int bx_time = (*digiIt).getTimeBin();
	  int wiregroup = (*digiIt).getWireGroup(); // counted from 1
	  LocalPoint  digiLP = layerGeom->localCenterOfWireGroup(wiregroup); // x==0
	  GlobalPoint digiGP = csclayer->toGlobal(digiLP);
	  double eta = digiGP.eta();
	  cout <<"      " << eta <<"  "<< (*digiIt)<<endl;
	}
    
      cout<<"     strip digis: phis"<<endl;
      const CSCComparatorDigiCollection::Range rcompd = compdc->get(layerId);
      for (CSCComparatorDigiCollection::const_iterator digiIt = rcompd.first; digiIt != rcompd.second; ++digiIt) 
	{
	  //int bx_time = (*digiIt).getTimeBin();
	  int strip = (*digiIt).getStrip();
	  // Position at the center of the strip y==0
	  LocalPoint  digiLP = layerGeom->topology()->localPosition(strip-0.5);
	  GlobalPoint digiGP = csclayer->toGlobal(digiLP);
	  double phi = normalizedPhi ( digiGP.phi() );
	  cout <<"      " << phi <<"  strip/comparator/time ="<< (*digiIt)<<endl;
	}
    }

    cout<<"--- SIMVERTICES: "<<endl;
    int no = 0;
    for (SimVertexContainer::const_iterator isvtx = simVertices.begin(); isvtx != simVertices.end(); ++isvtx) 
      cout<<no++<<":\t"<<*isvtx<<endl;
  }

  if (debugALLEVENT) cout<<"--- ALL ALCTs ---"<<endl;
  for (CSCALCTDigiCollection::DigiRangeIterator  cdetUnitIt = alcts->begin(); 
       cdetUnitIt != alcts->end(); cdetUnitIt++)
    {
      CSCDetId id( (*cdetUnitIt).first.rawId() );
      const CSCALCTDigiCollection::Range& range = (*cdetUnitIt).second;
      int noALCTs=0;
      for (CSCALCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
	{
	  noALCTs++;
	  if (debugALLEVENT) cout<<" * raw ID "<<id.rawId()<<" "<<id<<endl<<"   "<<(*digiIt)<<endl;
	}
      h_csctype_vs_alct_occup->Fill( getCSCType( id ), noALCTs);
    }

  if (debugALLEVENT) cout<<"--- ALL CLCTs ---"<<endl;
  for (CSCCLCTDigiCollection::DigiRangeIterator  cdetUnitIt = clcts->begin(); 
       cdetUnitIt != clcts->end(); cdetUnitIt++)
    {
      CSCDetId id( (*cdetUnitIt).first.rawId() );
      const CSCCLCTDigiCollection::Range& range = (*cdetUnitIt).second;
      int noCLCTs=0;
      for (CSCCLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
	{
	  noCLCTs++;
	  if (debugALLEVENT) cout<<" * raw ID "<<id.rawId()<<" "<<id<<endl<<"   "<<(*digiIt)<<endl;
	}
      h_csctype_vs_clct_occup->Fill( getCSCType( id ), noCLCTs);
    }


  

  // find primary vertex index and fill trkId2Index association:

  if (debugALLEVENT) cout<<"--- SIMTRACKS: "<<endl;
  int no = 0, primaryVert = -1;
  trkId2Index.clear();
  for (SimTrackContainer::const_iterator istrk = simTracks.begin(); istrk != simTracks.end(); ++istrk){
    if (debugALLEVENT) cout<<no<<":\t"<<istrk->trackId()<<" "<<*istrk<<endl;
    if ( primaryVert == -1 && !(istrk->noVertex()) ) primaryVert = istrk->vertIndex();
    trkId2Index[istrk->trackId()] = no;
    no++;
  }
  if ( primaryVert == -1 ) { 
    if (debugALLEVENT) cout<<" Warning: NO PRIMARY SIMVERTEX!"<<endl; 
    if (simTracks.size()>0) return 0;
  }


  // main loop over muon SimTracks:
  
  int numberSimTr=0;
  for (SimTrackContainer::const_iterator istrk = simTracks.begin(); istrk != simTracks.end(); ++istrk)
    {
      if ( !( abs(istrk->type()) == 13 && istrk->vertIndex() == primaryVert ) ) continue;
      stpt = sqrt(istrk->momentum().perp2());
      if (stpt < minSimTrPt_ ) {
	if (debugALLEVENT) cout<<" - rejected mu SimTrack: with low pt = "<<stpt<<endl;
	continue;
      }
      steta = istrk->momentum().eta();
      stphi = normalizedPhi( istrk->momentum().phi() );
      if (fabs (steta) > 2.5 || fabs (steta) < .8 ) {
	if (debugALLEVENT) cout<<" - rejected mu SimTrack: eta not in CSC  eta = "<<steta<<endl;
	continue;
      }
      bool inPhiEta = ( stphi>=minSimTrPhi_ && stphi<=maxSimTrPhi_ && steta>=minSimTrEta_ && steta<=maxSimTrEta_ );
      bool goodPhiEta = invertSimTrPhiEta_ ? !inPhiEta  : inPhiEta;
      if (!goodPhiEta) {
	if (debugALLEVENT) cout<<" - rejected mu SimTrack: phi = "<<stphi<<" eta = "<<steta<<" outside region"<<endl;
	continue;
      }
      if (debugALLEVENT) cout<<" *** Accepting mu SimTrack: pt = "<<stpt<<"  phi = "<<stphi<<" eta = "<<steta<<endl;

      MatchCSCMuL1 *match = new MatchCSCMuL1(&*istrk, &(simVertices[istrk->vertIndex()]), cscGeometry);
      match->muOnly = doStrictSimHitToTrackMatch_;
      match->minBxALCT  = minBxALCT_;
      match->maxBxALCT  = maxBxALCT_;
      match->minBxCLCT  = minBxCLCT_;
      match->maxBxCLCT  = maxBxCLCT_;
      match->minBxLCT   = minBxLCT_;
      match->maxBxLCT   = maxBxLCT_;
      match->minBxMPLCT = minBxMPLCT_;
      match->maxBxMPLCT = maxBxMPLCT_;

      propagateToCSCStations(match);

      // match SimHits and do some checks
      matchSimTrack2SimHits(match, simTracks, simVertices, allCSCSimHits);

      // match ALCT digis and SimHits;
      // if there are common SimHits in SimTrack, match to SimTrack
      matchSimTrack2ALCTs(match, allCSCSimHits, alcts, wiredc );

      // match CLCT digis and SimHits;
      // if there are common SimHits in SimTrack, match to SimTrack
      matchSimTrack2CLCTs(match, allCSCSimHits, clcts, compdc );

      // match CorrelatedLCT digis after TMB
      matchSimTrack2LCTs(match, lcts);

      // match CorrelatedLCT digis after MPC
      matchSimTrack2MPLCTs(match, mplcts);

      // match TrackFinder's tracks after Sector Processor
      matchSimtrack2TFTRACKs(match, muScales, muPtScale, l1Tracks);

      // match TrackFinder's track candidates after CSC Sorter
      matchSimtrack2TFCANDs(match, muScales, muPtScale, l1TfCands);

      if (!lightRun) {
	// match GMT candidates from GMT Readout
	matchSimtrack2GMTCANDs(match, muScales, muPtScale, l1GmtCands, l1GmtCSCCands, l1GmtCSCCandsInBXs);

	// match trigger muons from l1extra
	matchSimtrack2L1EXTRAs(match, l1Muons);
      }


      matches.push_back(match);
      numberSimTr++;

      // checks
      if (debugALLEVENT) {
	//charge = static_cast<int> (-itrack->type()/13); //static_cast<int> (itrack->charge());
	cout<<"SimTrk\t id eta phi pt nSH: "<<istrk->trackId()<<" "<<steta <<" "<< stphi <<" "<<stpt <<" "<<match->simHits.size()<<endl;
	cout<<"      \t nALCT: "<<match->ALCTs.size() <<" nCLCT: "<<match->CLCTs.size() <<" nLCT: "<<match->LCTs.size() <<" nMPLCT: "<<match->MPLCTs.size() <<" TFTRACKs/All: "<<match->TFTRACKs.size()<<"/"<<match->TFTRACKsAll.size()<<" TFCANDs/All: "<<match->TFCANDs.size()<<"/"<<match->TFCANDsAll.size()<<" GMTREGs/All: "<<match->GMTREGCANDs.size()<<"/"<<match->GMTREGCANDsAll.size()<<"  GMTRegBest:"<<(match->GMTREGCANDBest.l1reg != NULL)<<" GMTs/All: "<<match->GMTCANDs.size()<<"/"<<match->GMTCANDsAll.size()<<" GMTBest:"<<(match->GMTCANDBest.l1gmt != NULL)<<" L1EXTRAs/All: "<<match->L1EXTRAs.size()<<"/"<<match->L1EXTRAsAll.size()<<" L1EXTRABest:"<<(match->L1EXTRABest.l1extra != NULL)<<endl;
      }
    }
  
  
  // check overlapping chambers (have hits from two simtracks):
  vector<int> ch_vecs[5];
  set<int> ch_sets[5];
  unsigned int im=0;
  for (; im<matches.size() && im<5; im++) {
    ch_vecs[im] = matches[im]->chambersWithHits();
    ch_sets[im].insert(ch_vecs[im].begin(), ch_vecs[im].end());
  }
  set<int> ch_overlap;
  if (im>1)  set_intersection(ch_sets[0].begin(), ch_sets[0].end(), 
                              ch_sets[1].begin(), ch_sets[1].end(),
                              inserter(ch_overlap, ch_overlap.begin()) );
  if (debugALLEVENT) {
    cout<<"Number of overlapping chambers = "<<ch_overlap.size();
    if (ch_overlap.size()==0)cout<<endl;
    else {
      cout<<"  in stations ";
      for (set<int>::iterator ich = ch_overlap.begin(); ich!= ch_overlap.end(); ich++) {
        CSCDetId chId (*ich);
        cout<<chId.station()<<" ";
      }
      cout<<endl;
    }
  }



  map<int, vector<CSCALCTDigi> > detALCT;
  detALCT.clear();
  
  map<int, vector<CSCCLCTDigi> > detCLCT;
  detCLCT.clear();
  
  map<int, vector<CSCCorrelatedLCTDigi> > detMPLCT;
  detMPLCT.clear();
  
  unsigned inefTF = 0;
  
  for (unsigned int im=0; im<matches.size(); im++) 
    {
      if (debugINHISTOS ) cout<<"HISTOS for trk "<<im<<endl;
    
      MatchCSCMuL1 * match = matches[im];

      if (debugINHISTOS ) match->print("",1,0,1,1,1,0);

      stpt = sqrt(match->strk->momentum().perp2());
      steta = match->strk->momentum().eta();
      stphi = normalizedPhi( match->strk->momentum().phi() );

 
      //bool pt_ok = fabs(stpt)>2.;
      bool pt_ok = fabs(stpt) > 20.;
      //bool pt20_ok = fabs(stpt)>20.;
      bool eta_ok = ( fabs(steta) >= 1.2 &&  fabs(steta) <= 2.14 );
      bool etapt_ok = eta_ok && pt_ok;

      bool eta_1b = isME1bEtaRegion(steta, 1.6, 2.12);
      bool eta_gem_1b = isME1bEtaRegion(steta, 1.64, 2.05);


      unsigned nst_with_hits = match->nStationsWithHits();
      bool has_hits_in_st[5] = {0, match->hasHitsInStation(1), match->hasHitsInStation(2), 
				match->hasHitsInStation(3), match->hasHitsInStation(4)};

      bool okME1mplct = 0, okME2mplct = 0, okME3mplct = 0, okME4mplct = 0;
      int okNmplct = 0;
      int has_mplct_me1b = 0;
      vector<MatchCSCMuL1::MPLCT> rMPLCTs = match->MPLCTsInReadOut();
      if (rMPLCTs.size())
	{
	  // count matched
	  set<int> mpcStations;
	  for (unsigned i=0; i<rMPLCTs.size();i++)
	    {
	      if (!(rMPLCTs[i].deltaOk)) continue;
	      mpcStations.insert(rMPLCTs[i].id.station());
	      if (rMPLCTs[i].id.station() == 1) okME1mplct = 1;
	      if (rMPLCTs[i].id.station() == 2) okME2mplct = 1;
	      if (rMPLCTs[i].id.station() == 3) okME3mplct = 1;
	      if (rMPLCTs[i].id.station() == 4) okME4mplct = 1;
	      if (rMPLCTs[i].id.station() == 1 && rMPLCTs[i].id.ring() == 1) ++has_mplct_me1b;
	    }
	  okNmplct = mpcStations.size();
	}
      bool has_mpcs_in_st[5] = {0, has_hits_in_st[1] && okME1mplct, has_hits_in_st[2] && okME2mplct,
				has_hits_in_st[3] && okME3mplct, has_hits_in_st[4] && okME4mplct};
      unsigned nst_with_mpcs = has_mpcs_in_st[1] + has_mpcs_in_st[2] + has_mpcs_in_st[3] + has_mpcs_in_st[4];

      if (pt_ok) {
	h_eta_initial0->Fill(steta);

	// SimHIts:
	if (has_hits_in_st[1]) h_eta_me1_initial->Fill(steta);
	if (has_hits_in_st[2]) h_eta_me2_initial->Fill(steta);
	if (has_hits_in_st[3]) h_eta_me3_initial->Fill(steta);
	if (has_hits_in_st[4]) h_eta_me4_initial->Fill(steta);
      
	if (nst_with_hits>0) h_eta_initial_1st->Fill(steta);
	if (nst_with_hits>1) h_eta_initial_2st->Fill(steta);
	if (nst_with_hits>2) h_eta_initial_3st->Fill(steta);

	if (has_hits_in_st[1] && nst_with_hits>1) h_eta_me1_initial_2st->Fill(steta);
	if (has_hits_in_st[1] && nst_with_hits>2) h_eta_me1_initial_3st->Fill(steta);

	// MPC:
	if (has_mpcs_in_st[1]) h_eta_me1_mpc->Fill(steta);
	if (has_mpcs_in_st[2]) h_eta_me2_mpc->Fill(steta);
	if (has_mpcs_in_st[3]) h_eta_me3_mpc->Fill(steta);
	if (has_mpcs_in_st[4]) h_eta_me4_mpc->Fill(steta);

	if (nst_with_mpcs>0) h_eta_mpc_1st->Fill(steta);
	if (nst_with_mpcs>1) h_eta_mpc_2st->Fill(steta);
	if (nst_with_mpcs>2) h_eta_mpc_3st->Fill(steta);

	if (has_mpcs_in_st[1] && nst_with_mpcs>1) h_eta_me1_mpc_2st->Fill(steta);
	if (has_mpcs_in_st[1] && nst_with_mpcs>2) h_eta_me1_mpc_3st->Fill(steta);
      }
      if (eta_ok) {
	h_pt_initial0->Fill(stpt);

	// SimHIts:
	if (has_hits_in_st[1]) h_pt_me1_initial->Fill(stpt);
	if (has_hits_in_st[2]) h_pt_me2_initial->Fill(stpt);
	if (has_hits_in_st[3]) h_pt_me3_initial->Fill(stpt);
	if (has_hits_in_st[4]) h_pt_me4_initial->Fill(stpt);
      
	if (nst_with_hits>0) h_pt_initial_1st->Fill(stpt);
	if (nst_with_hits>1) h_pt_initial_2st->Fill(stpt);
	if (nst_with_hits>2) h_pt_initial_3st->Fill(stpt);

	if (has_hits_in_st[1] && nst_with_hits>1) h_pt_me1_initial_2st->Fill(stpt);
	if (has_hits_in_st[1] && nst_with_hits>2) h_pt_me1_initial_3st->Fill(stpt);

	// MPC:
	if (has_mpcs_in_st[1]) h_pt_me1_mpc->Fill(stpt);
	if (has_mpcs_in_st[2]) h_pt_me2_mpc->Fill(stpt);
	if (has_mpcs_in_st[3]) h_pt_me3_mpc->Fill(stpt);
	if (has_mpcs_in_st[4]) h_pt_me4_mpc->Fill(stpt);

	if (nst_with_mpcs>0) h_pt_mpc_1st->Fill(stpt);
	if (nst_with_mpcs>1) h_pt_mpc_2st->Fill(stpt);
	if (nst_with_mpcs>2) h_pt_mpc_3st->Fill(stpt);

	if (has_mpcs_in_st[1] && nst_with_mpcs>1) h_pt_me1_mpc_2st->Fill(stpt);
	if (has_mpcs_in_st[1] && nst_with_mpcs>2) h_pt_me1_mpc_3st->Fill(stpt);
      }


      MatchCSCMuL1::TFCAND * tfc = match->bestTFCAND(match->TFCANDs, bestPtMatch_);
      bool tffid_ok = false;
      if ( tfc ) tffid_ok = ( eta_ok && tfc->l1cand->quality_packed() > 1 );


      // require chambers with at least 4 layers with simhits in at least minNStWith4Hits stations
      if ( nst_with_hits < minNStWith4Hits_ ) continue;
      
      // at least one chamber in ME1 with 4 hits
      if (requireME1With4Hits_ && !has_hits_in_st[1]) continue;
      

      bool eta_high = ( fabs(steta) >= 2.1 &&  fabs(steta) <= 2.4 );
      //bool tffid_high = false;
      //if ( tfc ) tffid_high = ( eta_high && tfc->l1cand->quality_packed() > 1 );

      MatchCSCMuL1::TFCAND * tfcAll = match->bestTFCAND(match->TFCANDsAll, bestPtMatch_);
      bool tffidAll_ok = false;
      if ( tfcAll ) tffidAll_ok = ( eta_ok && tfcAll->l1cand->quality_packed() > 1 );

      MatchCSCMuL1::TFTRACK * tft = match->bestTFTRACK(match->TFTRACKs, bestPtMatch_);
      MatchCSCMuL1::TFTRACK * tftAll = match->bestTFTRACK(match->TFTRACKsAll, bestPtMatch_);
      int nokeey=0;

      //if ( match->TFCANDs.size() && pt_ok && fabs(steta) >= 1.6 &&  fabs(steta) <= 2.1) { 
      if ( match->TFCANDs.size() && pt_ok && fabs(steta) >= 1.65 &&  fabs(steta) <= 2.05) { 
	if (debugINHISTOS && tfc->pt < 7.99 ) cout<<" FAILS_TPT8!"<<endl;
	if ( lookAtTrackCondition_ != 0) {
	  if  ( tfc->pt >= 7.99) nokeey = 1;
	  else nokeey = -1;
	}
      }
      if (lookAtTrackCondition_ != nokeey) continue;

      h_eta_vs_nalct->Fill(steta, match->ALCTs.size());
      h_eta_vs_nclct->Fill(steta, match->CLCTs.size()); 
      h_eta_vs_nlct ->Fill(steta, match->LCTs.size());
      h_eta_vs_nmplct ->Fill(steta, match->MPLCTs.size());
  
      h_pt_vs_nalct->Fill(stpt, match->ALCTs.size());
      h_pt_vs_nclct->Fill(stpt, match->CLCTs.size());
      h_pt_vs_nlct ->Fill(stpt, match->LCTs.size());
      h_pt_vs_nmplct ->Fill(stpt, match->MPLCTs.size());

    }
    
  //============ Initial ==================

  if (tftAll) {
    h_nMplct_vs_nDigiMplct->Fill(tftAll->mplcts.size(), tftAll->trgdigis.size());
    h_qu_vs_nDigiMplct->Fill(tftAll->q_packed, tftAll->trgdigis.size());
  } else {
    h_nMplct_vs_nDigiMplct->Fill(0.,0.);
    h_qu_vs_nDigiMplct->Fill(0.,0.);
  }

  h_ntftrackall_vs_ntftrack->Fill(match->TFTRACKsAll.size(),match->TFTRACKs.size());
  h_ntfcandall_vs_ntfcand->Fill(match->TFCANDsAll.size(),match->TFCANDs.size());

  if (eta_ok) h_pt_vs_ntfcand->Fill(stpt,match->TFCANDs.size());
  if (pt_ok) h_eta_vs_ntfcand->Fill(steta,match->TFCANDs.size());

  if (eta_ok) h_pt_initial->Fill(stpt);
  if (eta_high) h_pth_initial->Fill(stpt);
  if (pt_ok) h_eta_initial->Fill(steta);
  if (etapt_ok) h_phi_initial->Fill(stphi);

  if (eta_1b) h_pt_initial_1b->Fill(stpt);
  if (eta_gem_1b) h_pt_initial_gem_1b->Fill(stpt);

  vector<int> chIds = match->chambersWithHits();
  vector<int> fillIds;
  if (pt_ok) for (size_t ch = 0; ch < chIds.size(); ch++)
	       {
		 CSCDetId chId(chIds[ch]);
		 int csct = getCSCType( chId );
		 h_cscdet_of_chamber->Fill( csct );
		 if(detALCT.find(chIds[ch]) != detALCT.end()) h_cscdet_of_chamber_w_alct->Fill( csct );
		 if(detCLCT.find(chIds[ch]) != detCLCT.end()) h_cscdet_of_chamber_w_clct->Fill( csct );
		 if(detMPLCT.find(chIds[ch]) != detMPLCT.end()) h_cscdet_of_chamber_w_mplct->Fill( csct );

		 if (csct==0 || csct==3) {
		   // check that if the same WG is hit in ME1/b and ME1/a
		   // then fill it only once from ME1/b
		   int wg = match->wireGroupAndStripInChamber(chIds[ch]).first;
		   if (wg>=10 && wg<=18) {
		     if (csct==3) {
		       CSCDetId di(chId.endcap(),chId.station(),1,chId.chamber(),0);
		       if (wg == match->wireGroupAndStripInChamber(di.rawId()).first) continue;
		     }
		   }
		   h_wg_me11_initial->Fill(wg);
		 }
	       }
    
    
  //============ GEM ==================

  SimTrackMatchManager gemcsc_match(*(match->strk), simVertices[match->strk->vertIndex()], gemMatchCfg_, iEvent, iSetup);
  const GEMDigiMatcher& match_gem = gemcsc_match.gemDigis();
  //const CSCStubMatcher& match_lct = gemcsc_match.cscStubs();
    
  int match_has_gem = 0;
  vector<int> match_gem_chambers;
  auto gem_superch_ids = match_gem.superChamberIds();
  for(auto d: gem_superch_ids)
    {
      GEMDetId id(d);
      bool odd = id.chamber() & 1;
      auto digis = match_gem.digisInSuperChamber(d);
      if (digis.size() > 0)
	{
	  match_gem_chambers.push_back(id.chamber());
	  if (odd) match_has_gem |= 1;
	  else     match_has_gem |= 2;
	}
    }

  //csc_ch_ids = match_lct.chamberIdsLCT();
  //for(auto d: csc_ch_ids)
  //{
  // CSCDetId id(d);
  //  bool odd = id.chamber() & 1;
  //  if (odd) match_has_lct |= 1;
  //  else match_has_lct |= 2;
  //}

  if (eta_gem_1b && match_has_gem) h_pt_gem_1b->Fill(stpt);
  if (eta_gem_1b && match_has_gem && has_mplct_me1b) h_pt_lctgem_1b->Fill(stpt);


  //============ ALCTs ==================
  if (debugINHISTOS) cerr<<" check: on to ALCT "<<endl;
  vector<MatchCSCMuL1::ALCT> ME1ALCTsOk;
  bool hasME1alct = 0;
  vector<MatchCSCMuL1::ALCT> rALCTs = match->ALCTsInReadOut();
  if (rALCTs.size()) 
    {
      if (eta_ok) h_pt_after_alct->Fill(stpt);
      if (pt_ok) h_eta_after_alct->Fill(steta);
      if (etapt_ok) h_phi_after_alct->Fill(stphi);
      int minbx[CSC_TYPES]={99,99,99,99,99,99,99,99,99,99};
      if (pt_ok) for (unsigned i=0; i<rALCTs.size();i++)
		   {
		     if (rALCTs[i].inReadOut()==0) continue;
		     int bx = rALCTs[i].getBX() - 6;
		     int bxf = rALCTs[i].trgdigi->getFullBX() - 6;
		     h_eta_vs_bx_after_alct->Fill( steta, bx );
		     h_bx_after_alct->Fill( bx );
		     int csct = getCSCType( rALCTs[i].id );
		     h_bx__alct_cscdet[ csct ]->Fill( bx );
		     h_bxf__alct_cscdet[ csct ]->Fill( bxf );
		     h_dbxbxf__alct_cscdet[ csct ]->Fill( bx-bxf );
		     if (rALCTs[i].deltaOk) {
		       h_bx__alctOk_cscdet[ csct ]->Fill( bx );
		       h_bxf__alctOk_cscdet[ csct ]->Fill( bxf );
		       if (debugINHISTOS && bx<-1) {
			 cout<<" OOW good ALCT: "<<*(rALCTs[i].trgdigi)<<endl;
			 dumpWireDigis(rALCTs[i].id, wiredc);
		       }
		     }
		     if ( fabs(bx) < fabs(minbx[csct]) ) minbx[csct] = bx;
		     h_qu_alct->Fill(rALCTs[i].trgdigi->getQuality());
		     h_qu_vs_bx__alct->Fill(rALCTs[i].trgdigi->getQuality(), bx);
		   }
      if (pt_ok) for (int i=0; i<CSC_TYPES;i++)
		   if (minbx[i]<99) h_bx_min__alct_cscdet[ i ]->Fill( minbx[i] );

      vector<int> chIDs = match->chambersWithALCTs();
      if (pt_ok) h_n_ch_w_alct->Fill(chIDs.size());
      if (pt_ok) for (size_t ch = 0; ch < chIDs.size(); ch++) 
		   {
		     vector<MatchCSCMuL1::ALCT> chalcts = match->chamberALCTs(chIDs[ch]);
		     vector<int> bxsalct = match->bxsWithALCTs(chIDs[ch]);
		     h_n_per_ch_alct->Fill(chalcts.size());
		     h_n_bx_per_ch_alct->Fill(bxsalct.size());
		     CSCDetId chId(chIDs[ch]);
		     int csct = getCSCType( chId );

		     MatchCSCMuL1::ALCT *bestALCT = match->bestALCT( chId );
		     if (bestALCT==0) cout<<"STRANGE: no best ALCT in chamber with ALCTs"<<endl;
		     if (bestALCT and bestALCT->deltaOk) {
		       h_bx__alctOkBest_cscdet[ csct ]->Fill( bestALCT->getBX() - 6 );
		       h_wg_vs_bx__alctOkBest_cscdet[ csct ]->Fill( match->wireGroupAndStripInChamber(chIDs[ch]).first, bestALCT->getBX() - 6);
		     }

		     h_n_per_ch_alct_cscdet[csct]->Fill(chalcts.size());
		     int n_per_bx[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		     for (size_t a = 0; a < chalcts.size(); a++)  n_per_bx[chalcts[a].getBX()]++;

		     for (int b=0;b<16;b++) h_n_per_ch_alct_vs_bx_cscdet[csct]->Fill(n_per_bx[b],b);

		     if (chalcts.size()>2 && debugINHISTOS)
		       {
			 cout<<"~~~~~~~~ WARNING! nALCT = "<<chalcts.size()<<" in ch "<<chIDs[ch]<<" "<<chId<<endl;
			 for (unsigned i=0; i<chalcts.size();i++) cout<<"~~~~~~~~~~ ALCT "<<i<<" "<<*(chalcts[i].trgdigi)<<endl;
		       }

		     if ( chalcts.size()>1 ) for (unsigned i=1; i<chalcts.size();i++)
					       {
						 h_bxdbx_alct_a1_da2->Fill(chalcts[0].getBX() - 6,  chalcts[0].getBX() - chalcts[i].getBX() );
						 h_bxdbx_alct_a1_da2_cscdet[csct]->Fill(chalcts[0].getBX() - 6,  chalcts[0].getBX() - chalcts[i].getBX() );
					       }
		   }

      bool okME1alct = 0, okME1alctg=0;
      if (pt_ok) for (unsigned i=0; i<rALCTs.size();i++)
		   {
		     if (rALCTs[i].id.station() == 1)
		       {
			 if (debugINHISTOS) cout<<" ALCT check: station "<<1<<endl;
			 okME1alct = 1;
			 hasME1alct = 1;
			 if (debugINHISTOS) cout<<" dw="<<rALCTs[i].deltaWire<<" md=["<<minDeltaWire_<<","<< maxDeltaWire_<<"]"<<endl;
			 if (minDeltaWire_ <= rALCTs[i].deltaWire &&
			     rALCTs[i].deltaWire <= maxDeltaWire_)
			   {
			     okME1alctg = 1;
			     ME1ALCTsOk.push_back(rALCTs[i]);
			     if (debugINHISTOS) cout<<" ALCT good "<<1<<endl;
			   }
		       }
		   }
      if(okME1alct)  h_eta_me1_after_alct->Fill(steta);
      if(okME1alctg) {
	h_eta_me1_after_alct_okAlct->Fill(steta);

	vector<int> chIDs = match->chambersWithALCTs();
	for (size_t ch = 0; ch < chIDs.size(); ch++)
	  {
	    CSCDetId chId(chIDs[ch]);
	    int csct = getCSCType( chId );
	    if (!(csct==0 || csct==3)) continue;
	    bool has_alct=0;
	    for (size_t i=0; i<ME1ALCTsOk.size(); i++) if (ME1ALCTsOk[i].id.rawId()==(unsigned int)chIDs[ch]) has_alct=1;
	    if (has_alct==0) continue;
	    // check that if the same WG has ALCT in ME1/b and ME1/a
	    // then fill it only once from ME1/b
	    int wg = match->wireGroupAndStripInChamber(chIDs[ch]).first;
	    if (wg>=10 && wg<=18)
	      {
		if (csct==3) {
		  CSCDetId di(chId.endcap(),chId.station(),1,chId.chamber(),0);
		  bool has_me1b=0;
		  for (size_t i=0; i<ME1ALCTsOk.size(); i++)
		    if (ME1ALCTsOk[i].id==di && wg == match->wireGroupAndStripInChamber(di.rawId()).first ) has_me1b=1;
		  if (has_me1b==1) continue;
		}
	      }
	    h_wg_me11_after_alct_okAlct->Fill(wg);
	  }
      }

      if (debugINHISTOS && pt_ok && steta>2.1 && !okME1alctg)
	{
	  for (size_t ch = 0; ch < chIds.size(); ch++)
	    {
	      CSCDetId chId(chIds[ch]);
	      if (getCSCType( chId )==3) {
		cout<<" no good ME1a okME1alct="<<okME1alct<<endl;
		dumpWireDigis(chId, wiredc);
	      }
	    }
	}

    } // ALCT


      //============ CLCTs ==================

  if (debugINHISTOS) cerr<<" check: on to CLCT "<<endl;
  vector<MatchCSCMuL1::CLCT> ME1CLCTsOk;
  bool hasME1clct = 0;
  vector<MatchCSCMuL1::CLCT> rCLCTs = match->CLCTsInReadOut();
  if (rCLCTs.size()) 
    {
      if (eta_ok) h_pt_after_clct->Fill(stpt);
      if (pt_ok) h_eta_after_clct->Fill(steta);
      if (etapt_ok) h_phi_after_clct->Fill(stphi);
      int minbx[CSC_TYPES]={99,99,99,99,99,99,99,99,99,99};
      if (pt_ok) for (unsigned i=0; i<rCLCTs.size();i++)
		   {
		     int bx = rCLCTs[i].getBX() - 6;
		     int bxf = rCLCTs[i].trgdigi->getFullBX() - 6;
		     h_eta_vs_bx_after_clct->Fill( steta, bx );
		     h_bx_after_clct->Fill( bx );
		     int csct = getCSCType( rCLCTs[i].id );
		     h_bx__clct_cscdet[ csct ]->Fill( bx );
		     h_bxf__clct_cscdet[ csct ]->Fill( bxf );
		     h_dbxbxf__clct_cscdet[ csct ]->Fill( bx-bxf );
		     if (rCLCTs[i].deltaOk) {
		       h_bx__clctOk_cscdet[ csct ]->Fill( bx );
		       h_bxf__clctOk_cscdet[ csct ]->Fill( bxf );
		       h_pt_vs_bend__clctOk_cscdet[csct]->
			 Fill(stpt, abs(pbend[rCLCTs[i].trgdigi->getPattern()]) );
		       h_bend__clctOk_cscdet[csct]->Fill(abs(pbend[rCLCTs[i].trgdigi->getPattern()]));
		     }
		     h_pt_vs_bend__clct_cscdet[csct]->
		       Fill(stpt, abs(pbend[rCLCTs[i].trgdigi->getPattern()]) );
		     if ( fabs(bx) < fabs(minbx[csct]) ) minbx[csct] = bx;
		     h_qu_clct->Fill(rCLCTs[i].trgdigi->getQuality());
		     h_qu_vs_bx__clct->Fill(rCLCTs[i].trgdigi->getQuality(), bx);
		   }
      if (pt_ok) for (int i=0; i<CSC_TYPES;i++)
		   if (minbx[i]<99) h_bx_min__clct_cscdet[ i ]->Fill( minbx[i] );

      vector<int> chIDs = match->chambersWithCLCTs();
      if (pt_ok) h_n_ch_w_clct->Fill(chIDs.size());
      if (pt_ok) for (size_t ch = 0; ch < chIDs.size(); ch++) 
		   {
		     vector<MatchCSCMuL1::CLCT> chcltcs = match->chamberCLCTs(chIDs[ch]);
		     vector<int> bxsclct = match->bxsWithCLCTs(chIDs[ch]);
		     h_n_per_ch_clct->Fill(chcltcs.size());
		     h_n_bx_per_ch_clct->Fill(bxsclct.size());
		     CSCDetId chId(chIDs[ch]);
		     int csct = getCSCType( chId );

		     MatchCSCMuL1::CLCT *bestCLCT = match->bestCLCT( chId );
		     if (bestCLCT==0) cout<<"STRANGE: no best CLCT in chamber with CLCTs"<<endl;
		     if (bestCLCT and bestCLCT->deltaOk) {
		       h_bx__clctOkBest_cscdet[ csct ]->Fill( bestCLCT->trgdigi->getBX() - 6 );
		     }

		     h_n_per_ch_clct_cscdet[csct]->Fill(chcltcs.size());

		     if ( chcltcs.size()>1 ) for (unsigned i=1; i<chcltcs.size();i++)
					       {
						 h_bxdbx_clct_c1_dc2->Fill(chcltcs[0].getBX() - 6,
									   chcltcs[0].getBX() - chcltcs[i].getBX() );
						 h_bxdbx_clct_c1_dc2_cscdet[csct]->Fill(chcltcs[0].getBX() - 6,
											chcltcs[0].getBX() - chcltcs[i].getBX() );
					       }
		   }

      bool okME1clct = 0, okME1clctg=0;
      if (pt_ok) for (unsigned i=0; i<rCLCTs.size();i++)
		   {
		     if (rCLCTs[i].id.station() == 1)
		       {
			 if (debugINHISTOS) cout<<" CLCT check: station "<<1<<endl;
			 okME1clct = 1;
			 hasME1clct = 1;
			 if (debugINHISTOS) cout<<" ds="<<abs(rCLCTs[i].deltaStrip)<<" md="<<minDeltaStrip_<<endl;
			 if (abs(rCLCTs[i].deltaStrip) <= minDeltaStrip_)
			   {
			     okME1clctg = 1;
			     ME1CLCTsOk.push_back(rCLCTs[i]);
			     if (debugINHISTOS) cout<<" CLCT good "<<1<<endl;
			   }
		       }
		   }
      if(okME1clct)  h_eta_me1_after_clct->Fill(steta);
      if(okME1clctg) h_eta_me1_after_clct_okClct->Fill(steta);
    }
    
  if (hasME1alct && hasME1clct) h_eta_me1_after_alctclct->Fill(steta);
  if (ME1ALCTsOk.size() && hasME1clct) h_eta_me1_after_alctclct_okAlct->Fill(steta);
  if (ME1CLCTsOk.size() && hasME1alct) h_eta_me1_after_alctclct_okClct->Fill(steta);
  if (ME1ALCTsOk.size() && ME1CLCTsOk.size()) 
    {
      h_eta_me1_after_alctclct_okAlctClct->Fill(steta);

      vector<int> chIDs = match->chambersWithALCTs();
      for (size_t ch = 0; ch < chIDs.size(); ch++)
	{
	  CSCDetId chId(chIDs[ch]);
	  int csct = getCSCType( chId );
	  if (!(csct==0 || csct==3)) continue;
	  bool has_alct=0;
	  for (size_t i=0; i<ME1ALCTsOk.size(); i++)
	    if (ME1ALCTsOk[i].id.rawId()==(unsigned int)chIDs[ch]) has_alct=1;
	  if (has_alct==0) continue;
	  bool has_clct=0;
	  for (size_t i=0; i<ME1CLCTsOk.size(); i++) 
	    if (ME1CLCTsOk[i].id.rawId()==(unsigned int)chIDs[ch]) has_clct=1;
	  if (has_clct==0) continue;

	  int wg = match->wireGroupAndStripInChamber(chIDs[ch]).first;
	  h_wg_me11_after_alctclct_okAlctClct->Fill(wg);
	}
    }

  if (pt_ok && steta>-2. && steta<-1.7 && ME1ALCTsOk.size() && !hasME1clct )  {
    match->print("badclct: ",1,1,1,1,0,0,0,0);
    CSCDetId gid;
    for (size_t i=0; i<ME1ALCTsOk.size(); i++) if (ME1ALCTsOk[i].id.iChamberType()==2) gid = ME1ALCTsOk[i].id;
    for (CSCCLCTDigiCollection::DigiRangeIterator  cdetUnitIt = clcts->begin(); cdetUnitIt != clcts->end(); cdetUnitIt++)
      {
	const CSCDetId& id = (*cdetUnitIt).first;
	if (id!=gid) continue;
	const CSCCLCTDigiCollection::Range& range = (*cdetUnitIt).second;
	for (CSCCLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++)
	  {
	    if (!(*digiIt).isValid()) continue;
	    cout<<" goodclct!!! "<<id<<"  "<<(*digiIt)<<endl;
	  }
      }
  }


  //============ LCTs ==================
  
  if (debugINHISTOS) cerr<<" check: on to LCT "<<endl;
  vector<MatchCSCMuL1::LCT> ME1LCTsOk;
  vector<MatchCSCMuL1::LCT> rLCTs = match->LCTsInReadOut();
  if (rLCTs.size()) 
    {
      if (eta_ok) h_pt_after_lct->Fill(stpt);
      if (pt_ok) h_eta_after_lct->Fill(steta);
      if (etapt_ok) h_phi_after_lct->Fill(stphi);
      int minbx[CSC_TYPES]={99,99,99,99,99,99,99,99,99,99};
      if (pt_ok) for (unsigned i=0; i<rLCTs.size();i++)
		   {
		     int bx = rLCTs[i].getBX() - 6;
		     h_eta_vs_bx_after_lct->Fill( steta, bx );
		     h_bx_after_lct->Fill( bx );
		     int csct = getCSCType( rLCTs[i].id );
		     h_bx__lct_cscdet[ csct ]->Fill( bx );
		     if ( fabs(bx) < fabs(minbx[csct]) ) minbx[csct] = bx;

		     if (rLCTs[i].alct) h_dBx_LctAlct->Fill ( bx - rLCTs[i].alct->getBX() + 6 );
		     if (rLCTs[i].clct)
		       {
			 h_dBx_LctClct->Fill ( bx - rLCTs[i].clct->getBX() + 6 );
			 h_dBx_LctClct_cscdet[csct]->Fill ( bx - rLCTs[i].clct->getBX() + 6 );
		       }
		     h_qu_lct->Fill(rLCTs[i].trgdigi->getQuality());
		     h_qu_vs_bx__lct->Fill(rLCTs[i].trgdigi->getQuality(), bx);
		     if ( rLCTs[i].clct )
		       h_qu_vs_bxclct__lct->Fill(rLCTs[i].trgdigi->getQuality(), rLCTs[i].clct->getBX() - 6);

		     if (rLCTs[i].alct && rLCTs[i].alct)
		       h_bx_lct__alct_vs_clct_cscdet[csct]->Fill ( rLCTs[i].alct->getBX() - 6,  rLCTs[i].clct->getBX() - 6);
		   }
      if (pt_ok) for (int i=0; i<CSC_TYPES;i++)
		   if (minbx[i]<99) h_bx_min__lct_cscdet[ i ]->Fill( minbx[i] );

      vector<int> chIDs = match->chambersWithLCTs();
      if (pt_ok) h_n_ch_w_lct->Fill(chIDs.size());
      if (pt_ok) for (size_t ch = 0; ch < chIDs.size(); ch++) 
		   {
		     vector<MatchCSCMuL1::LCT> chltcs = match->chamberLCTs(chIDs[ch]);
		     h_n_per_ch_lct->Fill(chltcs.size());
		     vector<int> bxslct = match->bxsWithLCTs(chIDs[ch]);
		     h_n_bx_per_ch_lct->Fill(bxslct.size());
		     CSCDetId chId(chIDs[ch]);
		     int csct = getCSCType( chId );
		     h_n_per_ch_lct_cscdet[csct]->Fill(chltcs.size());
		     if ( chltcs.size()==1 && chltcs[0].clct )
		       {
			 h_dBx_1inCh_LctClct->Fill( chltcs[0].getBX() - chltcs[0].clct->getBX() );
			 h_dBx_1inCh_LctClct_cscdet[csct]->Fill( chltcs[0].getBX() - chltcs[0].clct->getBX() );
		       }
		     if ( chltcs.size()==2 && chltcs[0].clct )
		       {
			 h_dBx_2inCh_LctClct->Fill( chltcs[0].getBX() - chltcs[0].clct->getBX() );
			 h_dBx_2inCh_LctClct_cscdet[csct]->Fill( chltcs[0].getBX() - chltcs[0].clct->getBX() );
		       }
		     if ( chltcs.size()==2 && chltcs[0].clct && chltcs[1].clct )
		       {
			 h_dBx_2inCh_LctClct->Fill( chltcs[1].getBX() - chltcs[1].clct->getBX() );
			 h_dBx_2inCh_LctClct_cscdet[csct]->Fill( chltcs[1].getBX() - chltcs[1].clct->getBX() );
			 h_dBx_2inCh_LctClct2->Fill( chltcs[0].getBX() - chltcs[0].clct->getBX(),
						     chltcs[1].getBX() - chltcs[1].clct->getBX() );
			 h_dBx_2inCh_LctClct2_cscdet[csct]->Fill( chltcs[0].getBX() - chltcs[0].clct->getBX(),
								  chltcs[1].getBX() - chltcs[1].clct->getBX() );
		       }

		     int ltype=0;
		     if ( chltcs.size()==1 && chltcs[0].alct==0 ) ltype=0;
		     else if ( chltcs.size()==1 && chltcs[0].clct==0 ) ltype=1;
		     else if ( chltcs.size()==1 && chltcs[0].alct && chltcs[0].clct ) ltype=2;
		     else if ( chltcs.size()==2 && chltcs[0].alct && chltcs[0].clct && chltcs[1].alct && chltcs[1].clct ) {
		       if ( *(chltcs[0].clct->trgdigi)!= *(chltcs[1].clct->trgdigi)  && *(chltcs[0].alct->trgdigi)== *(chltcs[1].alct->trgdigi)) ltype=3;
		       else if ( *(chltcs[0].clct->trgdigi)== *(chltcs[1].clct->trgdigi)  && *(chltcs[0].alct->trgdigi)!= *(chltcs[1].alct->trgdigi)) ltype=4;
		       else if ( *(chltcs[0].clct->trgdigi)!= *(chltcs[1].clct->trgdigi)  && *(chltcs[0].alct->trgdigi)!= *(chltcs[1].alct->trgdigi)) ltype=5;
		       else ltype=6;
		     }
		     else {
		       ltype=7;
		       if (debugINHISTOS) cout<<" ltype 7 :"<<chltcs.size()<<" "
					      <<(chltcs[0].clct!=0)<<" "<<(chltcs[1].clct!=0)<<" "<<(chltcs[0].alct!=0)<<" "<<(chltcs[1].alct!=0)<<endl;
		     }

		     std::vector<MatchCSCMuL1::ALCT> chaltcs = match->chamberALCTs(chIDs[ch]);
		     //std::vector<MatchCSCMuL1::CLCT> chcltcs = match->chamberCLCTs(chIDs[ch]);
		     if (chltcs.size()==1 && chltcs[0].clct && chaltcs.size()>1) {
		       ltype=10;
		       h_dbx_lct_a1_a2->Fill(chltcs[0].getBX() - chaltcs[0].getBX(),
					     chltcs[0].getBX() - chaltcs[1].getBX());
		       h_dbx_lct_a1_a2_cscdet[csct]->Fill(chltcs[0].getBX() - chaltcs[0].getBX(),
							  chltcs[0].getBX() - chaltcs[1].getBX());
		       h_bx_lct_a1_a2->Fill(chaltcs[0].getBX() - 6 , chaltcs[1].getBX() - 6 );
		       h_bx_lct_a1_a2_cscdet[csct]->Fill(chaltcs[0].getBX() -6 , chaltcs[1].getBX() - 6 );
		     }

		     h_type_lct->Fill(ltype);
		     h_type_lct_cscdet[csct]->Fill(ltype);
		   }
      
      bool okME1lct = 0, okME1alct=0, okME1alctclct=0, okME1clct=0, okME1clctalct=0;
      vector<MatchCSCMuL1::LCT> ME1LCTsOkCLCTNo, ME1LCTsOkCLCTOkALCTNo;
      if (pt_ok) for (unsigned i=0; i<rLCTs.size();i++)
		   {
		     if (rLCTs[i].id.station() == 1)
		       {
			 if (debugINHISTOS) cout<<" LCT check: station "<<1<<endl;
			 okME1lct = 1;
			 if (debugINHISTOS && rLCTs[i].alct) cout<<" dw="<<abs(rLCTs[i].alct->deltaWire)<<" md="<<minDeltaWire_<<endl;
			 if (debugINHISTOS && rLCTs[i].clct) cout<<" ds="<<abs(rLCTs[i].clct->deltaStrip)<<" md="<<minDeltaStrip_<<endl;

			 if (rLCTs[i].alct && rLCTs[i].alct->deltaOk)
			   {
			     okME1alct = 1;
			     if (debugINHISTOS) cout<<" LCT check: alct good "<<1<<endl;
			     if (rLCTs[i].clct && rLCTs[i].clct->deltaOk)
			       {
				 if (debugINHISTOS) cout<<" LCT check: alct-clct good "<<1<<endl;
				 okME1alctclct = 1;
				 //ME1LCTsOk.push_back(rLCTs[i]);
				 //if (debugINHISTOS) cout<<" LCT check: lct pushed "<<1<<endl;
			       }
			   }

			 if (rLCTs[i].clct && rLCTs[i].clct->deltaOk)
			   {
			     okME1clct = 1;
			     if (debugINHISTOS) cout<<" LCT check: clct good "<<1<<endl;
			     if (rLCTs[i].alct && rLCTs[i].alct->deltaOk)
			       {
				 if (debugINHISTOS) cout<<" LCT check: clct-alct good "<<1<<endl;
				 okME1clctalct = 1;
				 ME1LCTsOk.push_back(rLCTs[i]);
				 if (debugINHISTOS) cout<<" LCT check: lct pushed "<<1<<endl;
			       }
			     else if (rLCTs[i].alct) ME1LCTsOkCLCTOkALCTNo.push_back(rLCTs[i]);
			   }
			 else if (rLCTs[i].clct) ME1LCTsOkCLCTNo.push_back(rLCTs[i]);
		       }
		   }
      if(okME1lct) h_eta_me1_after_lct->Fill(steta);
      if(okME1alct) h_eta_me1_after_lct_okAlct->Fill(steta);
      if(okME1clct) h_eta_me1_after_lct_okClct->Fill(steta);
      if(okME1alctclct) 
	{
	  h_eta_me1_after_lct_okAlctClct->Fill(steta);

	  vector<int> chIDs = match->chambersWithLCTs();
	  for (size_t ch = 0; ch < chIDs.size(); ch++)
	    {
	      CSCDetId chId(chIDs[ch]);
	      int csct = getCSCType( chId );
	      if (!(csct==0 || csct==3)) continue;
	      bool has_lct=0;
	      for (size_t i=0; i<ME1LCTsOk.size(); i++)
		if (ME1LCTsOk[i].id.rawId()==(unsigned int)chIDs[ch]) has_lct=1;
	      if (has_lct==0) continue;
	      h_wg_me11_after_lct_okAlctClct->Fill(match->wireGroupAndStripInChamber(chIDs[ch]).first);
	    }
	}
      if(okME1clctalct) h_eta_me1_after_lct_okClctAlct->Fill(steta);
      if (debugINHISTOS) cout<<" LCT check: histo filled "<<1<<endl;
      
      if (pt_ok && ME1ALCTsOk.size() && ME1CLCTsOk.size() && ME1LCTsOk.size()==0 )
	{
	  for(size_t c=0; c<ME1CLCTsOk.size(); c++)
	    for(size_t a=0; a<ME1ALCTsOk.size(); a++)
	      h_bx_me1_aclct_ok_lct_no__bx_alct_vs_dbx_ACLCT->Fill( ME1ALCTsOk[a].getBX(),
								    ME1ALCTsOk[a].getBX() - ME1CLCTsOk[c].getBX());
	  if (debugINHISTOS) {
	    cout<<"Correct ALCT&CLCT but no good LCT"<<endl;
	    for (size_t a=0; a<ME1ALCTsOk.size(); a++)
	      cout<<"ALCT "<<a<<": "<<ME1ALCTsOk[a].id<<"  "<<*(ME1ALCTsOk[a].trgdigi)<<endl;
	    for (size_t c=0; c<ME1CLCTsOk.size(); c++)
	      cout<<"CLCT "<<c<<": "<<ME1CLCTsOk[c].id<<"  "<<*(ME1CLCTsOk[c].trgdigi)<<endl;
	  }
	}
      if (pt_ok && (ME1ALCTsOk.size()==0 || ME1CLCTsOk.size()==0) && ME1LCTsOk.size()>0 )
	{
	  if (debugINHISTOS) {
	    cout<<"Correct LCT but no good ALCT&CLCT"<<endl;
	    for (size_t a=0; a<ME1LCTsOk.size(); a++)
	      cout<<"LCT "<<a<<": "<<ME1LCTsOk[a].id<<"  "<<*(ME1LCTsOk[a].trgdigi)<<endl;
	    match->print("all STUFF",1,1,1,1,1,0);
	  }
	}

      if (pt_ok && okME1lct){
	for (size_t ch = 0; ch < chIDs.size(); ch++)
	  {
	    std::vector<MatchCSCMuL1::ALCT> chaltcs = match->chamberALCTs(chIDs[ch]);
	    std::vector<MatchCSCMuL1::CLCT> chcltcs = match->chamberCLCTs(chIDs[ch]);
	    CSCDetId chId(chIDs[ch]);
	    int csct = getCSCType( chId );
	    int na_per_bx[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	    int nc_per_bx[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	    for (size_t a = 0; a < chaltcs.size(); a++)  na_per_bx[chaltcs[a].getBX()]++;
	    for (size_t c = 0; c < chcltcs.size(); c++)  nc_per_bx[chcltcs[c].getBX()]++;
	    if (!okME1clctalct){
	      for (int b=0;b<16;b++) h_n_per_ch_me1nomatch_alct_vs_bx_cscdet[csct]->Fill(na_per_bx[b],b);
	      h_n_per_ch_me1nomatch_clct_cscdet[csct]->Fill(chcltcs.size());
	    }
	    if (!okME1clct) {
	      for (int b=0;b<16;b++) {
		h_n_per_ch_me1nomatchclct_alct_vs_bx_cscdet[csct]->Fill(na_per_bx[b],b);
		h_n_per_ch_me1nomatchclct_clct_vs_bx_cscdet[csct]->Fill(nc_per_bx[b],b);
	      }
	      h_n_per_ch_me1nomatchclct_clct_cscdet[csct]->Fill(chcltcs.size());
	      //for (size_t c = 0; c < chcltcs.size(); c++) for (size_t a = 0; a < chaltcs.size(); a++)
	      //    h_bx_me11nomatchclct_alct_vs_clct->Fill(chaltcs[a].getBX()-6, chcltcs[c].getBX()-6);
	    }
	    if (okME1clct && !okME1clctalct) {
	      for (int b=0;b<16;b++) {
		h_n_per_ch_me1nomatchalct_alct_vs_bx_cscdet[csct]->Fill(na_per_bx[b],b);
		h_n_per_ch_me1nomatchalct_clct_vs_bx_cscdet[csct]->Fill(nc_per_bx[b],b);
	      }
	      h_n_per_ch_me1nomatchalct_clct_cscdet[csct]->Fill(chcltcs.size());
	      //for (size_t c = 0; c < chcltcs.size(); c++) for (size_t a = 0; a < chaltcs.size(); a++)
	      //    h_bx_me11nomatchalct_alct_vs_clct->Fill(chaltcs[a].getBX()-6, chcltcs[c].getBX()-6);
	    }
	  }
	for (unsigned i=0; i<rLCTs.size();i++)
	  {
	    if (rLCTs[i].id.station() == 1 && rLCTs[i].id.ring() == 1)
	      {
		if ( rLCTs[i].clct && !(rLCTs[i].clct->deltaOk))
		  {
		    if (!okME1clct)
		      h_bx_me11nomatchclct_alct_vs_clct->Fill(rLCTs[i].alct->getBX()-6, rLCTs[i].clct->getBX()-6);
		    if (okME1clct && !okME1clctalct && rLCTs[i].alct && !(rLCTs[i].alct->deltaOk) )
		      h_bx_me11nomatchalct_alct_vs_clct->Fill( rLCTs[i].alct->getBX()-6, rLCTs[i].clct->getBX()-6 );
		  }
	      }
	    if ( rLCTs[i].id.station() == 1 )
	      {
		if ( rLCTs[i].clct && rLCTs[i].clct && rLCTs[i].clct->deltaOk && rLCTs[i].alct->deltaOk) {
		  if (rLCTs[i].id.ring() == 1)
		    h_strip_v_wireg_me1b->Fill(rLCTs[i].clct->trgdigi->getKeyStrip(),rLCTs[i].alct->trgdigi->getKeyWG());
		  if (rLCTs[i].id.ring() == 4)
		    h_strip_v_wireg_me1a->Fill(rLCTs[i].clct->trgdigi->getKeyStrip(),rLCTs[i].alct->trgdigi->getKeyWG());
		}
	      }
	  }
      }
    }

  //============ MPC LCTs ==================

  vector<MatchCSCMuL1::MPLCT> ME1MPLCTsOk;
  //vector<MatchCSCMuL1::MPLCT> ME1MPLCTs2Ok;
  if (rMPLCTs.size()) 
    {
      // count matched 
      set<int> mpcStations;
      for (unsigned i=0; i<rMPLCTs.size();i++)
	{
	  if (!(rMPLCTs[i].deltaOk)) continue;
	  mpcStations.insert(rMPLCTs[i].id.station());
	  if (rMPLCTs[i].id.station() == 1) okME1mplct = 1;
	  if (rMPLCTs[i].id.station() == 2) okME2mplct = 1;
	  if (rMPLCTs[i].id.station() == 3) okME3mplct = 1;
	  if (rMPLCTs[i].id.station() == 4) okME4mplct = 1;
	}
      okNmplct = mpcStations.size();
      
      if (eta_ok) {
	h_pt_after_mpc->Fill(stpt);
	if (okNmplct>1) h_pt_after_mpc_ok_plus->Fill(stpt);
	if (okNmplct>1 && okME1mplct) h_pt_me1_after_mpc_ok_plus->Fill(stpt);
      }
      if (eta_high) {
	h_pth_after_mpc->Fill(stpt);
	if (okNmplct>1) h_pth_after_mpc_ok_plus->Fill(stpt);
	if (okNmplct>1 && okME1mplct) h_pth_me1_after_mpc_ok_plus->Fill(stpt);
      }
      
      if (pt_ok){
	h_eta_after_mpc->Fill(steta);
	if (okNmplct>0) h_eta_after_mpc_ok->Fill(steta);
	if (okNmplct>1) h_eta_after_mpc_ok_plus->Fill(steta);
	if (okNmplct>2) h_eta_after_mpc_ok_plus_3st->Fill(steta);
	if (okNmplct>2 || (fabs(steta)<2.099 && okNmplct>1) ) h_eta_after_mpc_ok_plus_3st1a->Fill(steta);
      }
      
      if (etapt_ok) h_phi_after_mpc->Fill(stphi);
      int minbx[CSC_TYPES]={99,99,99,99,99,99,99,99,99,99};
      if (pt_ok) for (unsigned i=0; i<rMPLCTs.size();i++)
		   {
		     int bx = rMPLCTs[i].getBX() - 6;
		     h_eta_vs_bx_after_mpc->Fill( steta, bx );
		     h_bx_after_mpc->Fill( bx );
		     int csct = getCSCType( rMPLCTs[i].id );
		     h_bx__mpc_cscdet[ csct ]->Fill( bx );
		     if ( fabs(bx) < fabs(minbx[csct]) ) minbx[csct] = bx;
		     h_qu_mplct->Fill(rMPLCTs[i].trgdigi->getQuality());
		     h_qu_vs_bx__mplct->Fill(rMPLCTs[i].trgdigi->getQuality(), bx);
		     h_pattern_mplct->Fill( rMPLCTs[i].trgdigi->getPattern() );
		     h_pattern_mplct_cscdet[ csct ]->Fill( rMPLCTs[i].trgdigi->getPattern() );
		   }
      if (pt_ok) for (int i=0; i<CSC_TYPES;i++)
		   if (minbx[i]<99) h_bx_min__mpc_cscdet[ i ]->Fill( minbx[i] );

      if (pt_ok && nmplct_per_st[0]) h_eta_after_mpc_st1->Fill(steta);
      if (pt_ok && nmplct_per_st_good[0]) h_eta_after_mpc_st1_good->Fill(steta);

      vector<int> chIDs = match->chambersWithMPLCTs();
      if (pt_ok) h_n_ch_w_mplct->Fill(chIDs.size());
      if (pt_ok) for (size_t ch = 0; ch < chIDs.size(); ch++) 
		   {
		     vector<MatchCSCMuL1::MPLCT> chmpltcs = match->chamberMPLCTs(chIDs[ch]);
		     h_n_per_ch_mplct->Fill(chmpltcs.size());
		     CSCDetId chId(chIDs[ch]);
		     int csct = getCSCType( chId );
		     h_n_per_ch_mplct_cscdet[csct]->Fill(chmpltcs.size());
		   }
      
      //if (okME1mplct && okME234mplct) ME1MPLCTs2Ok = ME1MPLCTsOk;
      if(pt_ok){
	if (okME1mplct) {
	  h_eta_me1_after_mplct_okAlctClct->Fill(steta);
	  h_eta_me1_after_mplct_ok->Fill(steta);
	  if (okNmplct>1) h_eta_me1_after_mplct_okAlctClct_plus->Fill(steta);
	}
	if (okME2mplct) h_eta_me2_after_mplct_ok->Fill(steta);
	if (okME3mplct) h_eta_me3_after_mplct_ok->Fill(steta);
	if (okME4mplct) h_eta_me4_after_mplct_ok->Fill(steta);
      }
    }
    
  // TF efficiency drop for high pileup 
  if (rMPLCTs.size() && match->TFCANDs.size()==0 && fabs(steta) >= 1.6 &&  fabs(steta) <= 2.1) 
    {
      //if (eta_ok) h_pt_after_mpc->Fill(stpt);
      //if (pt_ok) h_eta_after_mpc->Fill(steta);
      //if (etapt_ok) h_phi_after_mpc->Fill(stphi);
    }

  //============ TF ==================
    
  if (match->TFTRACKs.size()) 
    {
      if (eta_ok) h_pt_after_tftrack->Fill(stpt);
      if (pt_ok) h_eta_after_tftrack->Fill(steta);
      if (etapt_ok) h_phi_after_tftrack->Fill(stphi);
      for (unsigned t=0; t<match->TFTRACKs.size(); t++) if (match->TFTRACKs[t].debug) match->TFTRACKs[t].print("DEBUG");
    }

  int nTFOk = 0, high_eta_nTFOk=0, nTFstubsOk = 0;
  bool okME1tf = 0;
  float gem_dphi_odd = -99.;
  float gem_dphi_even = -99.;
  if (match->TFCANDs.size()) 
    {
      if (tfc==NULL) cout<<" ALARM: tfc==NULL !!!"<<endl;

      double tfc_pt = tfc->tftrack->pt;
      unsigned ntf_stubs = tfc->tftrack->nStubs();
      bool high_eta_stubs = (tfc->eta<2.0999 || ntf_stubs>=3);
      int tf_mode = tfc->tftrack->mode();

      bool ok_2s123 = (tf_mode != 0xd); // excludes ME1-ME4 stub tf tracks
      bool ok_2s13 = (ok_2s123 && (tf_mode != 0x6)); // excludes ME1-ME2 and ME1-ME4 stub tf tracks

      int qu = tfc->l1cand->quality_packed();
      h_tfpt->Fill(tfc_pt);
      h_tfphi->Fill(tfc->phi);
      h_tfeta->Fill(tfc->eta);
      h_tfbx->Fill(tfc->l1cand->bx());
      h_tfdr->Fill(tfc->dr);

      // count matched 
      set<int> tfStations;
      for (unsigned i=0; i< tfc->tftrack->mplcts.size(); i++)
	{
	  if ( !((tfc->tftrack->mplcts)[i]->deltaOk) ) continue;
	  int st = (tfc->tftrack->mplcts)[i]->id.station();
	  tfStations.insert(st);
	  if (st == 1) 
	    {
	      okME1tf = 1;
	      float tmp_gem_dphi = (tfc->tftrack->mplcts)[i]->trgdigi->getGEMDPhi();
	      bool is_odd = 1  &  (tfc->tftrack->mplcts)[i]->id.chamber();
	      if ( is_odd && fabs(tmp_gem_dphi) < fabs(gem_dphi_odd) ) gem_dphi_odd = tmp_gem_dphi;
	      if (!is_odd && fabs(tmp_gem_dphi) < fabs(gem_dphi_even) ) gem_dphi_even = tmp_gem_dphi;
	    }
	}
      nTFstubsOk = tfStations.size();

      //if (pt_ok && nTFstubsOk>1) nTFOk++;
      //if (pt_ok && nTFstubsOk>1 && okME1tf) nTFOk++;
      if (pt_ok && nTFstubsOk>1 && okME1tf && tfc_pt>=10.) nTFOk++;
      if (pt_ok && nTFstubsOk>1 && okNmplct>1 && high_eta_stubs) high_eta_nTFOk++;
      
      if (eta_ok && pt_ok) 
	{
	  h_tfqu->Fill(qu);
	  h_tfpt_vs_qu->Fill(tfc_pt,qu);
	  h_tf_mode->Fill(tf_mode);
	}
      if (eta_ok) 
	{
	  h_pt_after_tfcand->Fill(stpt);

	  if (tfc_pt>=10.)  h_pt_after_tfcand_pt10->Fill(stpt);
	  if (tfc_pt>=20.)  h_pt_after_tfcand_pt20->Fill(stpt);
	  if (tfc_pt>=40.)  h_pt_after_tfcand_pt40->Fill(stpt);
	  if (tfc_pt>=60.)  h_pt_after_tfcand_pt60->Fill(stpt);

	  h_pt_vs_qu->Fill(stpt,qu);

	  if (tffid_ok)
	    {
	      h_pt_after_tfcand_ok->Fill(stpt);
	      if (tfc_pt>=10.)  h_pt_after_tfcand_pt10_ok->Fill(stpt);
	      if (tfc_pt>=20.)  h_pt_after_tfcand_pt20_ok->Fill(stpt);
	      if (tfc_pt>=40.)  h_pt_after_tfcand_pt40_ok->Fill(stpt);
	      if (tfc_pt>=60.)  h_pt_after_tfcand_pt60_ok->Fill(stpt);

	      h_pt_over_tfpt_resol->Fill( stpt/tfc_pt - 1.);
	      h_pt_over_tfpt_resol_vs_pt->Fill( stpt/tfc_pt - 1., stpt);
	    }
	
	  if (nTFstubsOk>1 && okNmplct>1) {
	    h_pt_after_tfcand_ok_plus->Fill(stpt);
	    if (qu >=1) h_pt_after_tfcand_ok_plus_q[0]->Fill(stpt);
	    if (qu >=2) h_pt_after_tfcand_ok_plus_q[1]->Fill(stpt);
	    if (qu >=3) h_pt_after_tfcand_ok_plus_q[2]->Fill(stpt);
	    if (tfc_pt>=10.) {
	      h_pt_after_tfcand_ok_plus_pt10->Fill(stpt);
	      if (qu >=1) h_pt_after_tfcand_ok_plus_pt10_q[0]->Fill(stpt);
	      if (qu >=2) h_pt_after_tfcand_ok_plus_pt10_q[1]->Fill(stpt);
	      if (qu >=3) h_pt_after_tfcand_ok_plus_pt10_q[2]->Fill(stpt);
	    }
	    if (okME1tf && okME1mplct){
	      h_pt_me1_after_tf_ok_plus->Fill(stpt);
	      if (qu >=1) h_pt_me1_after_tf_ok_plus_q[0]->Fill(stpt);
	      if (qu >=2) h_pt_me1_after_tf_ok_plus_q[1]->Fill(stpt);
	      if (qu >=3) h_pt_me1_after_tf_ok_plus_q[2]->Fill(stpt);
	      if (tfc_pt>=10.) {
		h_pt_me1_after_tf_ok_plus_pt10->Fill(stpt);
		if (qu >=1) h_pt_me1_after_tf_ok_plus_pt10_q[0]->Fill(stpt);
		if (qu >=2) h_pt_me1_after_tf_ok_plus_pt10_q[1]->Fill(stpt);
		if (qu >=3) h_pt_me1_after_tf_ok_plus_pt10_q[2]->Fill(stpt);
	      }
	    }
	  }
	}

      // TF ME1b
      if ( eta_1b && nTFstubsOk >= 2 )
	{
	  const int Nthr = 7;
	  float tfc_pt_thr[Nthr] = {0., 10., 15., 20., 25., 30., 40.};
	  for (int i=0; i<Nthr; ++i) if (tfc_pt >= tfc_pt_thr[i])  h_pt_after_tfcand_eta1b_2s[i]->Fill(stpt);

	  if (okME1tf)
	    {
	      for (int i=0; i<Nthr; ++i) if (tfc_pt >= tfc_pt_thr[i])
					   {
					     h_pt_after_tfcand_eta1b_2s1b[i]->Fill(stpt);
					     if (ok_2s123) h_pt_after_tfcand_eta1b_2s123[i]->Fill(stpt);
					     if (ok_2s13) h_pt_after_tfcand_eta1b_2s13[i]->Fill(stpt);
					   }
	    }

	  if ( nTFstubsOk >= 3 )
	    {
	      for (int i=0; i<Nthr; ++i) if (tfc_pt >= tfc_pt_thr[i])  h_pt_after_tfcand_eta1b_3s[i]->Fill(stpt);

	      if (okME1tf) {
		for (int i=0; i<Nthr; ++i) if (tfc_pt >= tfc_pt_thr[i])  h_pt_after_tfcand_eta1b_3s1b[i]->Fill(stpt);
	      }
	    }
	}

      // TF GEM & ME1b
      if ( eta_gem_1b && match_has_gem && has_mplct_me1b && nTFstubsOk >= 2 && okME1tf )
	{
	  const int Nthr = 7;
	  float tfc_pt_thr[Nthr] = {0., 10., 15., 20., 25., 30., 40.};
        
	  for (int i=0; i<Nthr; ++i) if (tfc_pt >= tfc_pt_thr[i]) 
				       {
					 h_mode_tfcand_gem1b_2s1b_1b[i]->Fill(tf_mode);

					 h_pt_after_tfcand_gem1b_2s1b[i]->Fill(stpt);
					 if (ok_2s123) h_pt_after_tfcand_gem1b_2s123[i]->Fill(stpt);
					 if (ok_2s13) h_pt_after_tfcand_gem1b_2s13[i]->Fill(stpt);

					 if ( nTFstubsOk >= 3 )  h_pt_after_tfcand_gem1b_3s1b[i]->Fill(stpt);
          
					 if (   (gem_dphi_odd < -99. && gem_dphi_even < 99.)  // both just dummy default values
						|| (gem_dphi_odd  > -9. && isGEMDPhiGood(gem_dphi_odd, tfc_pt_thr[i], 1) )  // good dphi odd
						|| (gem_dphi_even > -9. && isGEMDPhiGood(gem_dphi_even, tfc_pt_thr[i], 0) ) ) // good dphi even
					   {
					     h_pt_after_tfcand_dphigem1b_2s1b[i]->Fill(stpt);
					     if (ok_2s123) h_pt_after_tfcand_dphigem1b_2s123[i]->Fill(stpt);
					     if (ok_2s13) h_pt_after_tfcand_dphigem1b_2s13[i]->Fill(stpt);
					     if ( nTFstubsOk >= 3 )  h_pt_after_tfcand_dphigem1b_3s1b[i]->Fill(stpt);
					   }
				       }
	}

      if (eta_high) 
	{
	  h_pth_after_tfcand->Fill(stpt);

	  if (tfc_pt>=10.)  h_pth_after_tfcand_pt10->Fill(stpt);

	  if (tffid_ok)
	    {
	      h_pth_after_tfcand_ok->Fill(stpt);
	      if (tfc_pt>=10.)  h_pth_after_tfcand_pt10_ok->Fill(stpt);

	      h_pth_over_tfpt_resol->Fill( stpt/tfc_pt - 1.);
	      h_pth_over_tfpt_resol_vs_pt->Fill( stpt/tfc_pt - 1., stpt);
	    }
	
	  if (nTFstubsOk>1 && okNmplct>1) {
	    h_pth_after_tfcand_ok_plus->Fill(stpt);
	    if (high_eta_stubs) h_pth_after_tfcand_ok_plus_3st1a->Fill(stpt);
	    if (qu >=1) h_pth_after_tfcand_ok_plus_q[0]->Fill(stpt);
	    if (qu >=2) h_pth_after_tfcand_ok_plus_q[1]->Fill(stpt);
	    if (qu >=3) h_pth_after_tfcand_ok_plus_q[2]->Fill(stpt);
	    if (tfc_pt>=10.) {
	      h_pth_after_tfcand_ok_plus_pt10->Fill(stpt);
	      if (high_eta_stubs) h_pth_after_tfcand_ok_plus_pt10_3st1a->Fill(stpt);
	      if (qu >=1) h_pth_after_tfcand_ok_plus_pt10_q[0]->Fill(stpt);
	      if (qu >=2) h_pth_after_tfcand_ok_plus_pt10_q[1]->Fill(stpt);
	      if (qu >=3) h_pth_after_tfcand_ok_plus_pt10_q[2]->Fill(stpt);
	    }
	    if (okME1tf && okME1mplct){
	      h_pth_me1_after_tf_ok_plus->Fill(stpt);
	      if (high_eta_stubs) h_pth_me1_after_tf_ok_plus_3st1a->Fill(stpt);
	      if (qu >=1) h_pth_me1_after_tf_ok_plus_q[0]->Fill(stpt);
	      if (qu >=2) h_pth_me1_after_tf_ok_plus_q[1]->Fill(stpt);
	      if (qu >=3) h_pth_me1_after_tf_ok_plus_q[2]->Fill(stpt);
	      if (tfc_pt>=10.) {
		h_pth_me1_after_tf_ok_plus_pt10->Fill(stpt);
		if (high_eta_stubs) h_pth_me1_after_tf_ok_plus_pt10_3st1a->Fill(stpt);
		if (qu >=1) h_pth_me1_after_tf_ok_plus_pt10_q[0]->Fill(stpt);
		if (qu >=2) h_pth_me1_after_tf_ok_plus_pt10_q[1]->Fill(stpt);
		if (qu >=3) h_pth_me1_after_tf_ok_plus_pt10_q[2]->Fill(stpt);
	      }
	    }
	  }
	}


      // wor weight calculation
      //if (fabs(steta)>1.25 && fabs(steta)<1.9) {
      if (isME42EtaRegion(steta)) {
	double weight = rateWeight(stpt);
	if (tfc->tftrack->nStubs()>=2) {
	  h_tf_pt_h42_2st->Fill(tfc_pt);
	  h_tf_pt_h42_2st_w->Fill(tfc_pt,weight);
	}
	if (tfc->tftrack->nStubs()>=3) {
	  h_tf_pt_h42_3st->Fill(tfc_pt);
	  h_tf_pt_h42_3st_w->Fill(tfc_pt,weight);
	}
      }
      
      //bool okME1tf = 0;
      //for (unsigned i=0; i< tfc->tftrack->mplcts.size(); i++)  
      //  if ( ((tfc->tftrack->mplcts)[i])->id.station() == 1 && ((tfc->tftrack->mplcts)[i])->lct->deltaOk) okME1tf = 1;
      if(pt_ok && okME1tf) 
	{
	  h_eta_me1_after_tf_ok->Fill(steta);
	  if (tfc_pt>=10.) h_eta_me1_after_tf_ok_pt10->Fill(steta);
	  if (nTFstubsOk>1 && okNmplct>1 && okME1mplct){
	    h_eta_me1_after_tf_ok_plus->Fill(steta);
	    if (high_eta_stubs) h_eta_me1_after_tf_ok_plus_3st1a->Fill(steta);
	    if (qu >=1) h_eta_me1_after_tf_ok_plus_q[0]->Fill(steta);
	    if (qu >=2) h_eta_me1_after_tf_ok_plus_q[1]->Fill(steta);
	    if (qu >=3) h_eta_me1_after_tf_ok_plus_q[2]->Fill(steta);
	    if (tfc_pt>=10.) {
	      h_eta_me1_after_tf_ok_plus_pt10->Fill(steta);
	      if (high_eta_stubs) h_eta_me1_after_tf_ok_plus_pt10_3st1a->Fill(steta);
	      if (qu >=1) h_eta_me1_after_tf_ok_plus_pt10_q[0]->Fill(steta);
	      if (qu >=2) h_eta_me1_after_tf_ok_plus_pt10_q[1]->Fill(steta);
	      if (qu >=3) h_eta_me1_after_tf_ok_plus_pt10_q[2]->Fill(steta);
	    }
	  }
	}
      
      if (pt_ok) 
	{
	  h_eta_after_tfcand->Fill(steta);
	  if (qu >=1 ) h_eta_after_tfcand_q[0]->Fill(steta);
	  if (qu >=2 ) h_eta_after_tfcand_q[1]->Fill(steta);
	  if (qu >=3 ) h_eta_after_tfcand_q[2]->Fill(steta);
	
	  if (nTFstubsOk>0 && okNmplct>0) h_eta_after_tfcand_ok->Fill(steta);
	  if (nTFstubsOk>1 && okNmplct>1) {
	    h_eta_after_tfcand_ok_plus->Fill(steta);
	    if (high_eta_stubs) h_eta_after_tfcand_ok_plus_3st1a->Fill(steta);
	    if (qu >=1) h_eta_after_tfcand_ok_plus_q[0]->Fill(steta);
	    if (qu >=2) h_eta_after_tfcand_ok_plus_q[1]->Fill(steta);
	    if (qu >=3) h_eta_after_tfcand_ok_plus_q[2]->Fill(steta);
	  }
	
	  if (tfc_pt>=10.) {
	    h_eta_after_tfcand_pt10->Fill(steta);
	    if (eta_ok) h_tfqu_pt10->Fill(qu);
	    if (nTFstubsOk>0 && okNmplct>0) h_eta_after_tfcand_ok_pt10->Fill(steta);
	    if (nTFstubsOk>1 && okNmplct>1) {
	      h_eta_after_tfcand_ok_plus_pt10->Fill(steta);
	      if (high_eta_stubs) h_eta_after_tfcand_ok_plus_pt10_3st1a->Fill(steta);
	      if (qu >=1) h_eta_after_tfcand_ok_plus_pt10_q[0]->Fill(steta);
	      if (qu >=2) h_eta_after_tfcand_ok_plus_pt10_q[1]->Fill(steta);
	      if (qu >=3) h_eta_after_tfcand_ok_plus_pt10_q[2]->Fill(steta);
	    }
	  }
	  else  if (eta_ok) h_tfqu_pt10_no->Fill(qu);

	  h_eta_vs_qu->Fill(steta,qu);

	  h_eta_minus_tfeta_resol->Fill(steta - tfc->eta);

	  int nstubs_comm[4]={0};
	  int nstubs_my[4]={0}, nstubs_org[4]={0};
	  for (int st=0; st<MAX_STATIONS; st++)
	    {
	      int etax = (fabs(steta) >= 1.6 &&  fabs(steta) <= 2.1 ) ? 1:0;

	      //for (unsigned i=0; i< rMPLCTs.size(); i++)  if ( (rMPLCTs)[i].id.station() == st+1 )
	      vector<int> stubs_my;
	      for (unsigned i=0; i< tfc->tftrack->mplcts.size(); i++)  if ( ((tfc->tftrack->mplcts)[i])->id.station() == st+1 )
									 {
									   nstubs_my[st]+=1;
									   stubs_my.push_back(i);
									   if (tfc_pt<10.) 
									     {
									       if (etax) h_station_tf_pu_no->Fill(st+1);
									     } 
									   else if (etax) h_station_tf_pu_ok->Fill(st+1);
									 }
	      if (nstubs_my[st] && etax) {
		if (tfc_pt<10.) h_station_tf_pu_no_once->Fill(st+1);
		else h_station_tf_pu_ok_once->Fill(st+1);
	      }
	      vector<int> stubs_org;
	      for (unsigned i=0; i< tfc->tftrack->trgids.size(); i++)  if ( ((tfc->tftrack->trgids)[i]).station() == st+1 )
									 {
									   //CSCCorrelatedLCTDigi
									   nstubs_org[st]+=1;
									   stubs_org.push_back(i);
									   if (tfc_pt<10.) 
									     {
									       if (etax) h_station_tforg_pu_no->Fill(st+1);
									     } 
									   else if (etax) h_station_tforg_pu_ok->Fill(st+1);
									 }
	      if (nstubs_org[st] && etax) {
		if (tfc_pt<10.) h_station_tforg_pu_no_once->Fill(st+1);
		else h_station_tforg_pu_ok_once->Fill(st+1);
	      }

	      vector<int> stubs_comm;
	      for (unsigned i=0; i< stubs_my.size(); i++)
		{
		  for (unsigned j=0; j< stubs_org.size(); j++)
		    {
		      if (  ((tfc->tftrack->mplcts)[i])->id.rawId() != ((tfc->tftrack->trgids)[j]).rawId() ||
			    ((tfc->tftrack->mplcts)[i])->trgdigi->getKeyWG() != ((tfc->tftrack->trgdigis)[j])->getKeyWG() ||
			    ((tfc->tftrack->mplcts)[i])->trgdigi->getStrip() != ((tfc->tftrack->trgdigis)[j])->getStrip() ) continue;
		      stubs_comm.push_back(i);
		    }
		}
	      nstubs_comm[st] = stubs_comm.size();
	    }
	  if (debugINHISTOS) cout<<"diggin001 nstubs_my  ="<<nstubs_my[0]<<" "<<nstubs_my[1]<<" "<<nstubs_my[2]<<" "<<nstubs_my[3]<<endl;
	  if (debugINHISTOS) cout<<"diggin002 nstubs_org ="<<nstubs_org[0]<<" "<<nstubs_org[1]<<" "<<nstubs_org[2]<<" "<<nstubs_org[3]<<endl;
        
	  if (nstubs_my[0])   h_eta_after_tfcand_my_st1->Fill(steta);
	  if (nstubs_org[0])  h_eta_after_tfcand_org_st1->Fill(steta);
	  if (nstubs_comm[0]) h_eta_after_tfcand_comm_st1->Fill(steta);
	  if (nstubs_my[0] && tfc_pt>=10.)   h_eta_after_tfcand_my_st1_pt10->Fill(steta);
	
	  for (unsigned i=0; i< tfc->tftrack->mplcts.size(); i++) 
	    {
	      MatchCSCMuL1::MPLCT *mp = (tfc->tftrack->mplcts)[i];
	      int csct = getCSCType( mp->id );
	      h_tf_stub_bx->Fill( mp->getBX() - 6 );
	      h_tf_stub_bx_cscdet[csct]->Fill( mp->getBX() - 6 );
	      h_tf_stub_qu->Fill( mp->trgdigi->getQuality() );
	      h_tf_stub_qu_cscdet[csct]->Fill( mp->trgdigi->getQuality() );
	      h_tf_stub_qu_vs_bx->Fill( mp->trgdigi->getQuality(), mp->getBX() - 6 );
	      h_tf_stub_qu_vs_bx_cscdet[csct]->Fill( mp->trgdigi->getQuality(), mp->getBX() - 6 );
	      h_tf_stub_csctype->Fill(csct);
	    }
	
	  vector<MatchCSCMuL1::MPLCT> common_stubs;
	  for (unsigned j=0; j< tfc->tftrack->trgdigis.size(); j++) 
	    {
	      CSCDetId dgid = (tfc->tftrack->trgids)[j];
	      int csct = getCSCType( dgid );
	      const CSCCorrelatedLCTDigi * dg = (tfc->tftrack->trgdigis)[j];
	      int unmatched=1;
	      for (unsigned i=0; i< tfc->tftrack->mplcts.size(); i++) 
		{
		  MatchCSCMuL1::MPLCT *mp = (tfc->tftrack->mplcts)[i];
		  const CSCCorrelatedLCTDigi * mdg = mp->trgdigi;
		  if ( dgid != mp->id ) continue;
		  if ( dg->getQuality() != mdg->getQuality() ) continue;
		  if ( dg->getKeyWG()   != mdg->getKeyWG() ) continue;
		  if ( dg->getStrip()   != mdg->getStrip() ) continue;
		  if ( dg->getBX()      != mdg->getBX() ) continue;
		  unmatched = 0;
		  common_stubs.push_back(*mp);
		}
	      h_tf_stub_csctype_org->Fill(csct);
	      if (unmatched) h_tf_stub_csctype_org_unmatch->Fill(csct);
	  
	      h_tf_stub_pattern->Fill(dg->getPattern());
	      h_tf_stub_pattern_cscdet[csct]->Fill(dg->getPattern());
          
	      // looking at CLCT of this digi
	      map<int, vector<CSCCLCTDigi> >::const_iterator mapItr = detCLCT.find(dgid.rawId());
	      if(mapItr != detCLCT.end()) 
		for ( unsigned ii=0; ii<mapItr->second.size(); ii++ )
		  if( dg->getStrip() == ((mapItr->second)[ii]).getKeyStrip() &&
		      dg->getPattern() == ((mapItr->second)[ii]).getPattern()  )
		    {
		      h_tf_stub_bxclct->Fill( ((mapItr->second)[ii]).getBX() - 6 );
		      h_tf_stub_bxclct_cscdet[csct]->Fill( ((mapItr->second)[ii]).getBX() - 6 );
		    }

	    }
	  h_tf_n_uncommon_stubs->Fill( tfc->tftrack->trgdigis.size() - common_stubs.size());
	  h_tf_n_stubs_vs_matchstubs->Fill( tfc->tftrack->trgdigis.size(), tfc->tftrack->mplcts.size());
	  h_tf_n_stubs->Fill( tfc->tftrack->trgdigis.size() );
	  h_tf_n_matchstubs->Fill( tfc->tftrack->mplcts.size() );
	
	}
      if (etapt_ok) 
	{
	  h_phi_after_tfcand->Fill(stphi);
	  h_phi_minus_tfphi_resol->Fill(stphi - tfc->phi);
	}
    }
    
  //============ TF CANDs ==================
    
  if (match->TFCANDs.size()) 
    {
      if (pt_ok && fabs(steta>2.1) && fabs(steta<2.4) && okNmplct>1 && okME1mplct){
	if (fill_debug_tree_) resetDbg(dbg_);
	for (unsigned i=0; i< tfc->tftrack->mplcts.size(); i++)
	  {
	    MatchCSCMuL1::MPLCT* mlct = (tfc->tftrack->mplcts)[i];
	    if (!(mlct->deltaOk)) continue;
          
	    if (mlct->id.station()!=1) continue;
	    h_tf_check_st1_mcStrip_vs_ptbin_all->Fill(mlct->lct->clct->mcStrip,tfc->tftrack->pt_packed);
	    if (mlct->lct->clct->mcStrip > 39) h_tf_check_st1_mcWG_vs_ptbin_all->Fill(mlct->lct->alct->mcWG,tfc->tftrack->pt_packed);
	    if (fill_debug_tree_) 
	      {
		dbg_.evtn = nevt;
		dbg_.trkn = im;
		dbg_.pt = stpt;
		dbg_.eta = steta;
		dbg_.phi = stphi;
		dbg_.tfpt = tfc->tftrack->pt;
		dbg_.tfeta = tfc->tftrack->eta;
		dbg_.tfphi = tfc->tftrack->phi;
		dbg_.tfpt_packed = tfc->tftrack->pt_packed;
		dbg_.tfeta_packed = tfc->tftrack->eta_packed;
		dbg_.tfphi_packed = tfc->tftrack->phi_packed;
		dbg_.nseg = tfc->tftrack->nStubs();
		dbg_.nseg_ok = nTFstubsOk;
		dbg_.meEtap = mlct->meEtap;
		dbg_.mePhip = mlct->mePhip;
		dbg_.mcStrip = mlct->lct->clct->mcStrip;
		dbg_.mcWG    = mlct->lct->alct->mcWG;
		dbg_.strip = mlct->trgdigi->getStrip()/2 + 1;
		dbg_.wg = mlct->trgdigi->getKeyWG();
		dbg_.chamber = mlct->id.chamber();
		dbg_.dPhi12 = tfc->tftrack->dPhi12();
		dbg_.dPhi23 = tfc->tftrack->dPhi23();
		dbg_tree->Fill();
	      }
	  }
      }
      if (pt_ok && fabs(steta>2.1) && fabs(steta<2.4) && okNmplct>1 && okME1mplct && nTFOk==0){
	//if (pt_ok && fabs(steta>2.15) && fabs(steta<2.2) && okNmplct>1 && high_eta_nTFOk==0){
	inefTF = inefTF | (1<<im);

	cout << "badtf: im="<<im<<" steta="<<steta<<" okNmplct="<<okNmplct<<" okME1mplct="<<okME1mplct<<" nTFOk="<<nTFOk<<" high_eta_nTFOk="<<high_eta_nTFOk<<endl;
	bool okME1tf = 0;
	set<int> tfStations;
	bool interesting_strips=0;
	for (unsigned i=0; i< tfc->tftrack->mplcts.size(); i++)
	  {
	    MatchCSCMuL1::MPLCT* mlct = (tfc->tftrack->mplcts)[i];
	    int st = mlct->id.station();
	    cout<<" tf mplct "<<i<<" st="<<st<<" mcstrip="<<mlct->lct->clct->mcStrip<<" ok="<<mlct->deltaOk<<endl;
	    if (!(mlct->deltaOk)) continue;
	    tfStations.insert(st);
	    if (st == 1) {
	      okME1tf = 1;
	      h_tf_check_st1_strip->Fill(mlct->trgdigi->getStrip()/2 + 1);
	      h_tf_check_st1_mcStrip->Fill(mlct->lct->clct->mcStrip);
	      h_tf_check_st1_mcStrip_vs_ptbin->Fill(mlct->lct->clct->mcStrip,tfc->tftrack->pt_packed);
	      if (mlct->trgdigi->getStrip()/2 + 1>41){
		interesting_strips=1;
		h_tf_check_st1_wg->Fill(mlct->trgdigi->getKeyWG());
		h_tf_check_st1_strip->Fill(mlct->trgdigi->getStrip()/2 + 1);
		h_tf_check_st1_mcStrip->Fill(mlct->lct->clct->mcStrip);
		h_tf_check_st1_chamber->Fill(mlct->id.chamber());
	      }
	    }
	  }
	int nTFstubs = tfStations.size();
	if (pt_ok && nTFstubs>1 && okME1tf) nTFOk++;
	cout << "  nTFstubs="<<nTFstubs<<" okME1tf="<<okME1tf<<" nTFOk="<<nTFOk<<endl;
	if (nTFstubs==3 && okNmplct==4) cout<<"inef case 3ok"<<endl;
	else cout<<"inef case other"<<endl;

	if (interesting_strips){
	  h_tf_check_mode->Fill(tfc->tftrack->mode());
	  h_tf_check_bx->Fill(tfc->l1cand->bx());
	  h_tf_check_n_stubs_vs_matched->Fill(tfc->tftrack->nStubs(),nTFstubs);
	}
      }
    }
    
  //============ TF All==================
 
  if (match->TFCANDsAll.size()) 
    {
      if (tfcAll==NULL) cout<<" ALARM: tfcAll==NULL !!!"<<endl;

      if (eta_ok) h_pt_after_tfcand_all->Fill(stpt);

      if (tffidAll_ok) 
	{
	  h_pt_after_tfcand_all_ok->Fill(stpt);
	  if (tfcAll->pt>=10.)  h_pt_after_tfcand_all_pt10_ok->Fill(stpt);
	  if (tfcAll->pt>=20.)  h_pt_after_tfcand_all_pt20_ok->Fill(stpt);
	  if (tfcAll->pt>=40.)  h_pt_after_tfcand_all_pt40_ok->Fill(stpt);
	  if (tfcAll->pt>=60.)  h_pt_after_tfcand_all_pt60_ok->Fill(stpt);
	}
      if (pt_ok) 
	{
	  h_eta_after_tfcand_all->Fill(steta);
	  if (tfcAll->pt>=10.) h_eta_after_tfcand_all_pt10->Fill(steta);
	}
      if (etapt_ok) 
	{
	  h_phi_after_tfcand_all->Fill(stphi);
	}

    }

  //============ GMT Regional ==================

  MatchCSCMuL1::GMTREGCAND * gmtrc = 0;
  if (!lightRun) gmtrc = match->bestGMTREGCAND(match->GMTREGCANDs, bestPtMatch_);
  if (match->GMTREGCANDs.size()) 
    {
      if (gmtrc==NULL) cout<<" ALARM: gmtrc==NULL !!!"<<endl;
      if (eta_ok)   h_pt_after_gmtreg->Fill(stpt);
      if (eta_ok && gmtrc->pt>=10.)  h_pt_after_gmtreg_pt10->Fill(stpt);
      if (pt_ok)    h_eta_after_gmtreg->Fill(steta);
      if (pt_ok  && gmtrc->pt>=10.)  h_eta_after_gmtreg_pt10->Fill(steta);
      if (etapt_ok) h_phi_after_gmtreg->Fill(stphi);
    }
  MatchCSCMuL1::GMTREGCAND * gmtrca = 0;
  if (!lightRun) gmtrca = match->bestGMTREGCAND(match->GMTREGCANDsAll, bestPtMatch_);
  if (match->GMTREGCANDsAll.size()) 
    {
      if (gmtrca==NULL) cout<<" ALARM: gmtrca==NULL !!!"<<endl;
      if (eta_ok)   h_pt_after_gmtreg_all->Fill(stpt);
      if (eta_ok && gmtrca->pt>=10.)  h_pt_after_gmtreg_all_pt10->Fill(stpt);
      if (pt_ok)    h_eta_after_gmtreg_all->Fill(steta);
      if (pt_ok  && gmtrca->pt>=10.)  h_eta_after_gmtreg_all_pt10->Fill(steta);
      if (etapt_ok) h_phi_after_gmtreg_all->Fill(stphi);
    }
  if (match->GMTREGCANDBest.l1reg != NULL) 
    {
      if (eta_ok)   h_pt_after_gmtreg_dr->Fill(stpt);
      if (eta_ok && match->GMTREGCANDBest.pt>=10.)   h_pt_after_gmtreg_dr_pt10->Fill(stpt);
      if (pt_ok)    h_eta_after_gmtreg_dr->Fill(steta);
      if (pt_ok  && match->GMTREGCANDBest.pt>=10.)   h_eta_after_gmtreg_dr_pt10->Fill(steta);
      if (etapt_ok) h_phi_after_gmtreg_dr->Fill(stphi);
    }

  //============ GMT CANDs ==================

  MatchCSCMuL1::GMTCAND * gmtc = 0;
  if (!lightRun) gmtc = match->bestGMTCAND(match->GMTCANDs, bestPtMatch_);
  if (match->GMTCANDs.size()) 
    {
      if (gmtc==NULL) cout<<" ALARM: gmtc==NULL !!!"<<endl;

      h_gmtpt->Fill(gmtc->pt);
      h_gmtphi->Fill(gmtc->phi);
      h_gmteta->Fill(gmtc->eta);
      h_gmtbx->Fill(gmtc->l1gmt->bx());
      h_gmtrank->Fill(gmtc->l1gmt->rank());
      h_gmtqu->Fill(gmtc->l1gmt->quality());
      h_gmtisrpc->Fill(gmtc->l1gmt->isRPC());
      h_gmtdr->Fill(gmtc->dr);

      if (eta_ok)   h_pt_after_gmt->Fill(stpt);
      if (eta_ok && gmtc->pt>=10.)   h_pt_after_gmt_pt10->Fill(stpt);
      if (pt_ok)    h_eta_after_gmt->Fill(steta);
      if (pt_ok  && gmtc->pt>=10.)   h_eta_after_gmt_pt10->Fill(steta);
      if (etapt_ok) h_phi_after_gmt->Fill(stphi);
      

      // GMT ME1b
      if ( eta_1b && gmtc->l1gmt->useInSingleMuonTrigger() )
	{
	  const int Nthr = 7;
	  float tfc_pt_thr[Nthr] = {0., 10., 15., 20., 25., 30., 40.};
	  for (int i=0; i<Nthr; ++i) if (gmtc->pt >= tfc_pt_thr[i]) 
				       {
					 h_pt_after_gmt_eta1b_1mu[i]->Fill(stpt);

					 if (eta_gem_1b && match_has_gem && has_mplct_me1b) 
					   {
					     h_pt_after_gmt_gem1b_1mu[i]->Fill(stpt);

					     if (   (gem_dphi_odd < -99. && gem_dphi_even < 99.)  // both just dummy default values
						    || (gem_dphi_odd  > -9. && isGEMDPhiGood(gem_dphi_odd, tfc_pt_thr[i], 1) )  // good dphi odd
						    || (gem_dphi_even > -9. && isGEMDPhiGood(gem_dphi_even, tfc_pt_thr[i], 0) ) ) // good dphi even
					       {
						 h_pt_after_gmt_dphigem1b_1mu[i]->Fill(stpt);
					       }
					   }
				       }
	}

    }
  MatchCSCMuL1::GMTCAND * gmtca = 0;
  if (!lightRun) gmtca = match->bestGMTCAND(match->GMTCANDsAll, bestPtMatch_);
  if (match->GMTCANDsAll.size()) 
    {
      if (gmtca==NULL) cout<<" ALARM: gmtca==NULL !!!"<<endl;
      if (eta_ok)   h_pt_after_gmt_all->Fill(stpt);
      if (eta_ok && gmtca->pt>=10.)   h_pt_after_gmt_all_pt10->Fill(stpt);
      if (pt_ok)    h_eta_after_gmt_all->Fill(steta);
      if (pt_ok  && gmtca->pt>=10.)   h_eta_after_gmt_all_pt10->Fill(steta);
      if (etapt_ok) h_phi_after_gmt_all->Fill(stphi);
    }

  if (lightRun) match->GMTCANDBest.l1gmt = NULL;
  if (match->GMTCANDBest.l1gmt != NULL) 
    {
      h_gmtxpt->Fill(match->GMTCANDBest.pt);
      h_gmtxphi->Fill(match->GMTCANDBest.phi);
      h_gmtxeta->Fill(match->GMTCANDBest.eta);
      h_gmtxbx->Fill(match->GMTCANDBest.l1gmt->bx());
      h_gmtxrank->Fill(match->GMTCANDBest.l1gmt->rank());
      h_gmtxqu->Fill(match->GMTCANDBest.l1gmt->quality());
      h_gmtxisrpc->Fill(match->GMTCANDBest.l1gmt->isRPC());
      h_gmtxdr->Fill(match->GMTCANDBest.dr);

      if (eta_ok)   h_pt_after_gmt_dr->Fill(stpt);
      if (eta_ok && match->GMTCANDBest.pt>=10.)   h_pt_after_gmt_dr_pt10->Fill(stpt);
      if (pt_ok)    h_eta_after_gmt_dr->Fill(steta);
      if (pt_ok  && match->GMTCANDBest.pt>=10.)   h_eta_after_gmt_dr_pt10->Fill(steta);
      if (etapt_ok) h_phi_after_gmt_dr->Fill(stphi);

      if (fabs(match->GMTCANDBest.eta)<0.9) {
	cout<<"FUUUU!: GMTCANDBest.eta="<<match->GMTCANDBest.eta<<" track: "<<*(match->GMTCANDBest.l1gmt)<<endl;
	cout<<resetiosflags(ios::showpoint | ios::fixed);
      }
      if (match->GMTCANDBest.l1gmt->rank()>205) {
	cout<<"FUUUU!: GMTCANDBest.rank="<<match->GMTCANDBest.l1gmt->rank()<<" track: "<<*(match->GMTCANDBest.l1gmt)<<endl;
	cout<<resetiosflags(ios::showpoint | ios::fixed);
      }
    }
  if (match->GMTCANDBest.l1gmt != NULL && match->GMTCANDs.size()==0) 
    {
      h_gmtxpt_nocsc->Fill(match->GMTCANDBest.pt);
      h_gmtxphi_nocsc->Fill(match->GMTCANDBest.phi);
      h_gmtxeta_nocsc->Fill(match->GMTCANDBest.eta);
      h_gmtxbx_nocsc->Fill(match->GMTCANDBest.l1gmt->bx());
      h_gmtxrank_nocsc->Fill(match->GMTCANDBest.l1gmt->rank());
      h_gmtxqu_nocsc->Fill(match->GMTCANDBest.l1gmt->quality());
      h_gmtxisrpc_nocsc->Fill(match->GMTCANDBest.l1gmt->isRPC());
      h_gmtxdr_nocsc->Fill(match->GMTCANDBest.dr);

      if (eta_ok)   h_pt_after_gmt_dr_nocsc->Fill(stpt);
      if (eta_ok && match->GMTCANDBest.pt>=10.)   h_pt_after_gmt_dr_nocsc_pt10->Fill(stpt);
      if (pt_ok)    h_eta_after_gmt_dr_nocsc->Fill(steta);
      if (pt_ok  && match->GMTCANDBest.pt>=10.)   h_eta_after_gmt_dr_nocsc_pt10->Fill(steta);
      if (etapt_ok) h_phi_after_gmt_dr_nocsc->Fill(stphi);
    }
  if (match->GMTCANDBest.l1gmt != NULL && match->GMTCANDs.size()==0 && match->TFCANDs.size()>0) 
    {
      h_gmtxqu_nogmtreg->Fill(match->GMTCANDBest.l1gmt->quality());
      h_gmtxisrpc_nogmtreg->Fill(match->GMTCANDBest.l1gmt->isRPC());
    }
  if (match->GMTCANDBest.l1gmt != NULL && match->TFCANDs.size()==0 && rMPLCTs.size()>0) 
    {
      h_gmtxqu_notfcand->Fill(match->GMTCANDBest.l1gmt->quality());
      h_gmtxisrpc_notfcand->Fill(match->GMTCANDBest.l1gmt->isRPC());
    }
  if (match->GMTCANDBest.l1gmt != NULL && rMPLCTs.size()==0) 
    {
      h_gmtxqu_nompc->Fill(match->GMTCANDBest.l1gmt->quality());
      h_gmtxisrpc_nompc->Fill(match->GMTCANDBest.l1gmt->isRPC());
    }

  //============ L1EXTRAs ==================

  if (lightRun) match->L1EXTRABest.l1extra = NULL;
  if (match->L1EXTRAs.size()) 
    {
      if (match->L1EXTRABest.l1extra ==NULL) cout<<" ALARM: no L1EXTRABest but #L1EXTRAs="<<match->L1EXTRAs.size()<<endl;
      else if (match->L1EXTRABest.pt >= 10.) 
	{
	  if (eta_ok) h_pt_after_xtra_pt10->Fill(stpt);
	  if (pt_ok)  h_eta_after_xtra_pt10->Fill(steta);
	}
      if (eta_ok)   h_pt_after_xtra->Fill(stpt);
      if (pt_ok)    h_eta_after_xtra->Fill(steta);
      if (etapt_ok) h_phi_after_xtra->Fill(stphi);
    }
  if (match->L1EXTRAsAll.size()) 
    {
      if (match->L1EXTRABest.l1extra ==NULL) cout<<" ALARM: no L1EXTRABest but #L1EXTRAsAll="<<match->L1EXTRAsAll.size()<<endl;
      else if (match->L1EXTRABest.pt >= 10.) 
	{
	  if (eta_ok) h_pt_after_xtra_all_pt10->Fill(stpt);
	  if (pt_ok)  h_eta_after_xtra_all_pt10->Fill(steta);
	}
      if (eta_ok)   h_pt_after_xtra_all->Fill(stpt);
      if (pt_ok)    h_eta_after_xtra_all->Fill(steta);
      if (etapt_ok) h_phi_after_xtra_all->Fill(stphi);
    }
  if (match->L1EXTRABest.l1extra != NULL) 
    {
      h_xtrapt->Fill(match->L1EXTRABest.pt);
      h_xtraphi->Fill(match->L1EXTRABest.phi);
      h_xtraeta->Fill(match->L1EXTRABest.eta);
      //h_xtrabx->Fill(match->L1EXTRABest.l1extra->bx());
      h_xtradr->Fill(match->L1EXTRABest.dr);

      if (eta_ok)   h_pt_after_xtra_dr->Fill(stpt);
      if (eta_ok && match->L1EXTRABest.pt >= 10.)   h_pt_after_xtra_all_pt10->Fill(stpt);
      if (pt_ok)    h_eta_after_xtra_dr->Fill(steta);
      if (pt_ok  && match->L1EXTRABest.pt >= 10.)   h_eta_after_xtra_dr_pt10->Fill(steta);
      if (etapt_ok) h_phi_after_xtra_dr->Fill(stphi);

      if (fabs(match->L1EXTRABest.eta)<0.9) cout<<"FUUUU!: L1EXTRABest.eta="<<match->L1EXTRABest.eta<<" track: "<<match->L1EXTRABest.l1extra->gmtMuonCand()<<endl;
    }

  if (debugINHISTOS) cerr<<" check: on to Ns "<<endl;



  h_n_alct->Fill(rALCTs.size());
  h_n_clct->Fill(rCLCTs.size());
  h_n_lct->Fill(rLCTs.size());
  h_n_mplct->Fill(rMPLCTs.size());
  h_n_tftrack->Fill(match->TFTRACKs.size());
  h_n_tftrack_all->Fill(match->TFTRACKsAll.size());
  h_n_tfcand->Fill(match->TFCANDs.size());
  h_n_tfcand_all->Fill(match->TFCANDsAll.size());
  h_n_gmtregcand->Fill(match->GMTREGCANDs.size());
  h_n_gmtregcand_all->Fill(match->GMTREGCANDsAll.size());
  h_n_gmtcand->Fill(match->GMTCANDs.size());
  h_n_gmtcand_all->Fill(match->GMTCANDsAll.size());
  h_n_xtra->Fill(match->L1EXTRAs.size());
  h_n_xtra_all->Fill(match->L1EXTRAsAll.size());

  if ( (tft && tftAll) &&
       ( tft->phi_packed  != tftAll->phi_packed ||
	 tft->eta_packed  != tftAll->eta_packed   ) )
    cout<<"ALARM: tft != tftAll  "<<tft->eta_packed<<"/"<<tft->phi_packed<<" != "<<tftAll->eta_packed<<"/"<<tftAll->phi_packed<<endl;

  //h_eta_after_tftrack_q[3]->Fill(steta);
  //h_eta_after_tfcand_q[3]->Fill(steta);

}



//=======================================================================
//============================= RATES ===================================


//============ RATE ALCT ==================

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
if (debugRATE) cout<< "----- statring nalct"<<endl;
map< int , vector<const CSCALCTDigi*> > me11alcts;
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
	  {
	    int bx = (*digiIt).getBX();
	    //if ( bx-6 < minBX_ || bx-6 > maxBX_ )
	    if ( bx < minBxALCT_ || bx > maxBxALCT_ )
	      {
		if (debugRATE) cout<<"discarding BX = "<< bx-6 <<endl;
		continue;
	      }

	    // store all ME11 alcts together so we can look at them later
	    // take into accout that 10<=WG<=15 alcts are present in both 1a and 1b
	    if (csct==0) me11alcts[idd.rawId()].push_back(&(*digiIt));
	    if (csct==3 && (*digiIt).getKeyWG() < 10) {
	      CSCDetId id11(idd.endcap(),1,1,idd.chamber());
	      me11alcts[id11.rawId()].push_back(&(*digiIt));
	    }

	    //        if (debugALCT) cout<<"raw ID "<<id.rawId()<<" "<<id<<"    NTrackHitsInChamber  nmhits  alctInfo.size  diff  " 
	    //                           <<trackHitsInChamber.size()<<" "<<nmhits<<" "<<alctInfo.size()<<"  "
	    //                           << nmhits-alctInfo.size() <<endl 
	    //                           << "  "<<(*digiIt)<<endl;
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
	if (nalct_per_ch_bx[b]>0) {
	  ++n_ch_alct_per_bx[b];
	  ++n_ch_alct_per_bx_st[id.station()-1][b];
	  ++n_ch_alct_per_bx_cscdet[csct][b];
	}
      }
  } // loop CSCALCTDigiCollection
//map< CSCDetId , vector<const CSCALCTDigi*> >::const_iterator mapIt = me11alcts.begin();
//for (;mapIt != me11alcts.end(); mapIt++){}
map< int , vector<const CSCALCTDigi*> >::const_iterator aMapIt = me11alcts.begin();
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
for (int b=0;b<16;b++) {
  if (b < minBxALCT_ || b > maxBxALCT_) continue;
  h_rt_nalct_vs_bx->Fill(nalct_per_bx[b],b);
  h_rt_nalct_per_bx->Fill(nalct_per_bx[b]);
  h_rt_n_ch_alct_per_bx->Fill(n_ch_alct_per_bx[b]);
  for (int s=0; s<MAX_STATIONS; s++) 
    h_rt_n_ch_alct_per_bx_st[s]->Fill(n_ch_alct_per_bx_st[s][b]);
  for (int me=0; me<=CSC_TYPES; me++) 
    h_rt_n_ch_alct_per_bx_cscdet[me]->Fill(n_ch_alct_per_bx_cscdet[me][b]);
 }

  
if (debugRATE) cout<< "----- end nalct="<<nalct<<endl;


//============ RATE CLCT ==================

//  map<int, vector<CSCCLCTDigi> > detCLCT;
//  detCLCT.clear();
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
if (debugRATE) cout<< "----- statring nclct"<<endl;
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
		if (debugRATE) cout<<"discarding BX = "<< bx-6 <<endl;
		continue;
	      }
	    //if (debugCLCT) cout<<"raw ID "<<id.rawId()<<" "<<id<<"    NTrackHitsInChamber  nmhits  clctInfo.size  diff  " 
	    //                   <<trackHitsInChamber.size()<<" "<<nmhits<<" "<<clctInfo.size()<<"  "
	    //                   << nmhits-clctInfo.size() <<endl 
	    //                   << "  "<<(*digiIt)<<endl;
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
if (debugRATE) cout<< "----- end nclct="<<nclct<<endl;


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
map< int , vector<const CSCCorrelatedLCTDigi*> > me11lcts;
if (debugRATE) cout<< "----- statring nlct"<<endl;
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
	    //if (debugLCT) cout<< "----- LCT in raw ID "<<id.rawId()<<" "<<id<< " (trig id. " << id.triggerCscId() << ")"<<endl;
	    //if (debugLCT) cout<< " "<< (*digiIt);
	    //if ( bx-6 < minBX_ || bx-6 > maxBX_ )
	    if ( bx < minBxLCT_ || bx > maxBxLCT_ )
	      {
		if (debugRATE) cout<<"discarding BX = "<< bx-6 <<endl;
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

	    map<int, vector<CSCCLCTDigi> >::const_iterator mapItr = detCLCT.find(id.rawId());
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
map< int , vector<const CSCCorrelatedLCTDigi*> >::const_iterator mapIt = me11lcts.begin();
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
for (int s=0; s<MAX_STATIONS; s++)  for (int i=1; i<=12; i++)
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
if (debugRATE) cout<< "----- end nlct="<<nlct<<endl;


//============ RATE MPC LCT ==================

int nmplct=0;
int nmplct_per_bx[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int nmplct_sector_st[MAX_STATIONS][13], nmplct_sector_bx_st[MAX_STATIONS][13][16], nmplct_trigsector_bx_st1[13][16];
for (int s=0; s<MAX_STATIONS; s++) for (int i=0; i<13; i++) {
    nmplct_sector_st[s][i]=0;
    for (int j=0; j<16; j++) { nmplct_sector_bx_st[s][i][j]=0; nmplct_trigsector_bx_st1[i][j]=0; }
  }

if (debugRATE) cout<< "----- statring nmplct"<<endl;
vector<MatchCSCMuL1::MPLCT> rtMPLCTs;
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
	    //if (debugRATE) cout<< "----- MPLCT in raw ID "<<id.rawId()<<" "<<id<< " (trig id. " << id.triggerCscId() << ")"<<endl;
	    //if (debugRATE) cout<<" "<< (*digiIt);
	    int bx = (*digiIt).getBX();
	    //if ( bx-6 < minBX_ || bx-6 > maxBX_ )
	    if ( bx < minBxMPLCT_ || bx > maxBxMPLCT_ )
	      {
		if (debugRATE) cout<<"discarding BX = "<< (*digiIt).getBX()-6 <<endl;
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

		cout<<"DBGSRLUT "<<id.endcap()<<" "<<id.station()<<" "<<id.ring()<<" "<<id.chamber()<<"  "<<(*digiIt).getKeyWG()<<" "<<(*digiIt).getStrip()<<"  "<<etaphi.first<<" "<<etaphi.second<<"  "<<eta_lut<<" "<<phi_lut<<"  "<<etaphi.first - eta_lut<<" "<<deltaPhi(etaphi.second, phi_lut)<<endl;
	      }
	  }
      }
  }
h_rt_nmplct->Fill(nmplct);
for (int b=0;b<16;b++) {
  if ( b < minBxALCT_ || b > maxBxALCT_ ) continue;
  h_rt_nmplct_vs_bx->Fill(nmplct_per_bx[b],b);
 }
for (int s=0; s<MAX_STATIONS; s++)  for (int i=1; i<=12; i++)
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
if (debugRATE) cout<< "----- end nmplct="<<nmplct<<endl;


//============ RATE TF TRACK ==================

int ntftrack=0;
if (debugRATE) cout<< "----- statring ntftrack"<<endl;
vector<MatchCSCMuL1::TFTRACK> rtTFTracks;
//  if (debugTFInef && inefTF) cout<<"#################### TF INEFFICIENCY ALL TFTRACKs:"<<endl;
for ( L1CSCTrackCollection::const_iterator trk = l1Tracks->begin(); trk != l1Tracks->end(); trk++)
  {
    if ( trk->first.bx() < minRateBX_ || trk->first.bx() > maxRateBX_ )
      {
	if (debugRATE) cout<<"discarding BX = "<< trk->first.bx() <<endl;
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
	    if (!((*digiIt).isValid())) cout<<"ALARM!!! match TFCAND to TFTRACK in rates: not valid id="<<id.rawId()<<" "<<id<<endl;
	    bool me1a_case = (defaultME1a && id.station()==1 && id.ring()==1 && (*digiIt).getStrip() > 127);
	    if (me1a_case){
	      CSCDetId id1a(id.endcap(),id.station(),4,id.chamber(),0);
	      cid = id1a;
	    }
	    //if (id.station()==1 && id.ring()==4) cout<<"me1adigi check: "<<(*digiIt)<<" "<<endl;
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
	cout<<"DBGdeta12 "<<d.endcap()<<" "<<d.ring()<<" "<<d.chamber()<<"  "<<stub.getKeyWG()<<" "<<stub.getStrip()<<"  "<<myTFTrk.nStubs(1,1,1,1,1)<<" "<<myTFTrk.pt<<" "<<myTFTrk.eta<<"  "<<etaphi1.first<<" "<<etaphi2.first<<" "<<etaphi1.first-etaphi2.first<<"  "<<etaphi1.second<<" "<<etaphi2.second<<" "<<deltaPhi(etaphi1.second,etaphi2.second)<<endl;

	if ( (etaphi1.first-etaphi2.first) > 0.1) {
	  myTFTrk.print("");
	  cout<<"############### CSCTFSPCoreLogic printout for large deta12 = "<<etaphi1.first-etaphi2.first<< " at "<<d.endcap()<<" "<<d.ring()<<" "<<d.chamber()<<endl;
	  runCSCTFSP(mplcts, dttrigs);
	  cout<<"############### end printout"<<endl;
	}
      }
      else {
	cout<<"myTFTrk.trgids corrupt"<<endl;
	myTFTrk.print("");
      }
    }

    h_rt_tftrack_pt->Fill(myTFTrk.pt);
    h_rt_tftrack_bx->Fill(trk->first.bx());
    h_rt_tftrack_mode->Fill(myTFTrk.mode());
  }
h_rt_ntftrack->Fill(ntftrack);
if (debugRATE) cout<< "----- end ntftrack="<<ntftrack<<endl;


//============ RATE TFCAND ==================

int ntfcand=0, ntfcandpt10=0;
if (debugRATE) cout<< "----- statring ntfcand"<<endl;
vector<MatchCSCMuL1::TFCAND> rtTFCands;
for ( vector< L1MuRegionalCand >::const_iterator trk = l1TfCands->begin(); trk != l1TfCands->end(); trk++)
  {
    if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
      {
	if (debugRATE) cout<<"discarding BX = "<< trk->bx() <<endl;
	continue;
      }
    double sign_eta = ( (trk->eta_packed() & 0x20) == 0) ? 1.:-1;
    //if ( sign_eta<0) continue;
    if (doSelectEtaForGMTRates_ && sign_eta<0) continue;

    MatchCSCMuL1::TFCAND myTFCand;
    myTFCand.init( &*trk , ptLUT, muScales, muPtScale);
    myTFCand.dr = 999.;
    //double tfpt = myTFCand.pt;

    //if (debugRATE) cout<< "----- eta/phi/pt "<<sign_eta<<"*"<<tfeta<<"/"<<tfphi<<"/"<<tfpt<<" "<< int(trk->eta_packed() & 0x1F) <<endl;

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
      cout<<"myTFCand.tftrack == NULL:"<<endl;
      cout<<" cand: "<<trk->pt_packed()<<" "<<trk->eta_packed()<<" "<<trk->phi_packed()<<" "<<trk->bx()<<endl;
      cout<<" trk: "<<endl;
      for (size_t tt = 0; tt<rtTFTracks.size(); tt++)
	cout<<"       "<<rtTFTracks[tt].pt_packed<<" "<<rtTFTracks[tt].eta_packed<<" "<<rtTFTracks[tt].phi_packed<<" "<<rtTFTracks[tt].l1trk->bx()<<endl;
    }

    if (myTFCand.tftrack != NULL) {
      double tfpt = myTFCand.tftrack->pt;
      double tfeta = myTFCand.tftrack->eta;

      if (tfpt>=10.) ntfcandpt10++;
      
      h_rt_tfcand_pt->Fill(tfpt);
    
      unsigned int ntrg_stubs = myTFCand.tftrack->trgdigis.size();
      if (ntrg_stubs!=myTFCand.ids.size())
	cout<<"OBA!!! trgdigis.size()!=ids.size(): "<<ntrg_stubs<<"!="<<myTFCand.ids.size()<<endl;
      if (ntrg_stubs>=2) h_rt_tfcand_pt_2st->Fill(tfpt);
      if (ntrg_stubs>=3) h_rt_tfcand_pt_3st->Fill(tfpt);
      //cout<<"\n nnntf: "<<ntrg_stubs<<" "<<myTFCand.tftrack->nStubs(0,1,1,1,1)<<endl;
      //if (ntrg_stubs != myTFCand.tftrack->nStubs()) myTFCand.tftrack->print("non-equal nstubs!");
      //if (fabs(myTFCand.eta)>1.25 && fabs(myTFCand.eta)<1.9) {
      if (isME42EtaRegion(myTFCand.eta)) {
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
    //else cout<<"Strange: myTFCand.tftrack != NULL"<<endl;
  }
h_rt_ntfcand->Fill(ntfcand);
h_rt_ntfcand_pt10->Fill(ntfcandpt10);
if (debugRATE) cout<< "----- end ntfcand/ntfcandpt10="<<ntfcand<<"/"<<ntfcandpt10<<endl;


//============ RATE GMT REGIONAL ==================

int ngmtcsc=0, ngmtcscpt10=0;
if (debugRATE) cout<< "----- statring ngmt csc"<<endl;
vector<MatchCSCMuL1::GMTREGCAND> rtGMTREGCands;
float max_pt_2s = -1, max_pt_3s = -1, max_pt_2q = -1, max_pt_3q = -1;
float max_pt_2s_eta = -111, max_pt_3s_eta = -111, max_pt_2q_eta = -111, max_pt_3q_eta = -111;
float max_pt_me42_2s = -1, max_pt_me42_3s = -1, max_pt_me42_2q = -1, max_pt_me42_3q = -1;
float max_pt_me42r_2s = -1, max_pt_me42r_3s = -1, max_pt_me42r_2q = -1, max_pt_me42r_3q = -1;

float max_pt_2s_2s1b = -1, max_pt_2s_2s1b_eta = -111; 
float max_pt_2s_no1a = -1;//, max_pt_2s_eta_no1a = -111;
float max_pt_2s_1b = -1;//,   max_pt_2s_eta_1b = -111;
float max_pt_3s_no1a = -1, max_pt_3s_eta_no1a = -111;
float max_pt_3s_1b = -1,   max_pt_3s_eta_1b = -111;

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

MatchCSCMuL1::TFTRACK *trk__max_pt_3s_3s1b_eta = nullptr;
MatchCSCMuL1::TFTRACK *trk__max_pt_2s1b_1b = nullptr;
const CSCCorrelatedLCTDigi * the_me1_stub = nullptr;
CSCDetId the_me1_id;
map<int,int> bx2n;
for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
for ( vector<L1MuRegionalCand>::const_iterator trk = l1GmtCSCCands.begin(); trk != l1GmtCSCCands.end(); trk++)
  {
    if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
      {
	if (debugRATE) cout<<"discarding BX = "<< trk->bx() <<endl;
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

    bool eta_me42 = isME42EtaRegion(myGMTREGCand.eta);
    bool eta_me42r = isME42RPCEtaRegion(myGMTREGCand.eta);
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

	bool eta_me1b = isME1bEtaRegion(myGMTREGCand.eta);
	bool eta_me1b_whole = isME1bEtaRegion(myGMTREGCand.eta, 1.6, 2.14);
	bool eta_no1a = (geta >= 1.2 && geta < 2.14);
      
	n_stubs = myGMTREGCand.nTFStubs;
	size_t n_stubs_id = myGMTREGCand.ids.size();
	//if (n_stubs == n_stubs_id) cout<<"n_stubs good"<<endl;
	if (n_stubs != n_stubs_id) cout<<"n_stubs bad: "<<eta_q<<" "<<n_stubs<<" != "<<n_stubs_id<<" "<< geta  <<endl;

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
      } else { 
      cout<<"GMTCSC match not found pt="<<gpt<<" eta="<<myGMTREGCand.eta<<"  packed: "<<trk->phi_packed()<<" "<<trk->eta_packed()<<endl;
      for (unsigned i=0; i< rtTFCands.size(); i++) cout<<"    "<<rtTFCands[i].l1cand->phi_packed()<<" "<<rtTFCands[i].l1cand->eta_packed();
      cout<<endl;
      cout<<"  all tfcands:";
      for ( vector< L1MuRegionalCand >::const_iterator ctrk = l1TfCands->begin(); ctrk != l1TfCands->end(); ctrk++)
	if (!( ctrk->bx() < minRateBX_ || ctrk->bx() > maxRateBX_ )) cout<<"    "<<ctrk->phi_packed()<<" "<<ctrk->eta_packed();
      cout<<endl;
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
    //  cout<<"weird stubs number "<<myGMTREGCand.ids.size()<<" for q="<<trk->quality()<<endl;
    //  if (myGMTREGCand.tfcand->tftrack != NULL) myGMTREGCand.tfcand->tftrack->print("");
    //  else cout<<"null tftrack!"<<endl;
    //}

    //    if (trk->quality()>=3 && gpt >=40. && isME1bEtaRegion(myGMTREGCand.eta) ) {
    //      cout<<"highpt csctf in ME1b "<<endl;
    //      myGMTREGCand.tfcand->tftrack->print("");
    //    }
    if (has_me1_stub && n_stubs > 2 && gpt >= 30. && geta> 1.6 && geta < 2.15 ) {
      cout<<"highpt csctf in ME1b "<<endl;
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
if (max_pt_3s_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_no1a->Fill(max_pt_3s_eta_no1a);
if (max_pt_3s_2s1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s1b->Fill(max_pt_3s_2s1b_eta);
if (max_pt_3s_2s1b_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s1b_1b->Fill(max_pt_3s_2s1b_eta_1b);
if (max_pt_3s_2s123_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s123_1b->Fill(max_pt_3s_2s123_eta_1b);
if (max_pt_3s_2s13_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s13_1b->Fill(max_pt_3s_2s13_eta_1b);
if (max_pt_3s_2s1b_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s1b_no1a->Fill(max_pt_3s_2s1b_eta_no1a);
if (max_pt_3s_2s123_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s123_no1a->Fill(max_pt_3s_2s123_eta_no1a);
if (max_pt_3s_2s13_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_2s13_no1a->Fill(max_pt_3s_2s13_eta_no1a);
if (max_pt_3s_3s1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1b->Fill(max_pt_3s_3s1b_eta);
if (max_pt_3s_3s1b_1b>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1b_1b->Fill(max_pt_3s_3s1b_eta_1b);
if (max_pt_3s_3s1b_no1a>=20.) h_rt_gmt_csc_ptmax20_eta_3s_3s1b_no1a->Fill(max_pt_3s_3s1b_eta_no1a);
if (max_pt_2q>=20.) h_rt_gmt_csc_ptmax20_eta_2q->Fill(max_pt_2q_eta);
if (max_pt_3q>=20.) h_rt_gmt_csc_ptmax20_eta_3q->Fill(max_pt_3q_eta);

if (max_pt_2s>=30.) h_rt_gmt_csc_ptmax30_eta_2s->Fill(max_pt_2s_eta);
if (max_pt_2s_2s1b>=30.) h_rt_gmt_csc_ptmax30_eta_2s_2s1b->Fill(max_pt_2s_2s1b_eta);
if (max_pt_3s>=30.) h_rt_gmt_csc_ptmax30_eta_3s->Fill(max_pt_3s_eta);
if (max_pt_3s_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_1b->Fill(max_pt_3s_eta_1b);
if (max_pt_3s_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_no1a->Fill(max_pt_3s_eta_no1a);
if (max_pt_3s_2s1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s1b->Fill(max_pt_3s_2s1b_eta);
if (max_pt_3s_2s1b_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s1b_1b->Fill(max_pt_3s_2s1b_eta_1b);
if (max_pt_3s_2s123_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s123_1b->Fill(max_pt_3s_2s123_eta_1b);
if (max_pt_3s_2s13_1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s13_1b->Fill(max_pt_3s_2s13_eta_1b);
if (max_pt_3s_2s1b_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s1b_no1a->Fill(max_pt_3s_2s1b_eta_no1a);
if (max_pt_3s_2s123_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s123_no1a->Fill(max_pt_3s_2s123_eta_no1a);
if (max_pt_3s_2s13_no1a>=30.) h_rt_gmt_csc_ptmax30_eta_3s_2s13_no1a->Fill(max_pt_3s_2s13_eta_no1a);
if (max_pt_3s_3s1b>=30.) h_rt_gmt_csc_ptmax30_eta_3s_3s1b->Fill(max_pt_3s_3s1b_eta);
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
if (debugRATE) cout<< "----- end ngmt csc/ngmtpt10="<<ngmtcsc<<"/"<<ngmtcscpt10<<endl;

if (max_pt_3s_3s1b>=30.) 
  {
    cout<<"filled h_rt_gmt_csc_ptmax30_eta_3s_3s1b eta "<<max_pt_3s_3s1b_eta<<endl;
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
    if (the_me1_stub) cout<<"DBGMODE "<<the_me1_id.endcap()<<" "<<the_me1_id.chamber()<<" "<<trk__max_pt_2s1b_1b->pt<<" "<<trk__max_pt_2s1b_1b->mode()<<" "<<pbend[the_me1_stub->getPattern()] <<" "<<the_me1_stub->getGEMDPhi()<<endl;
  }

int ngmtrpcf=0, ngmtrpcfpt10=0;
if (debugRATE) cout<< "----- statring ngmt rpcf"<<endl;
vector<MatchCSCMuL1::GMTREGCAND> rtGMTRPCfCands;
float max_pt_me42 = -1, max_pt = -1, max_pt_eta = -111;
for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
for ( vector<L1MuRegionalCand>::const_iterator trk = l1GmtRPCfCands.begin(); trk != l1GmtRPCfCands.end(); trk++)
  {
    if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
      {
	if (debugRATE) cout<<"discarding BX = "<< trk->bx() <<endl;
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

    bool eta_me42 = isME42RPCEtaRegion(myGMTREGCand.eta);
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
if (debugRATE) cout<< "----- end ngmt rpcf/ngmtpt10="<<ngmtrpcf<<"/"<<ngmtrpcfpt10<<endl;


int ngmtrpcb=0, ngmtrpcbpt10=0;
if (debugRATE) cout<< "----- statring ngmt rpcb"<<endl;
vector<MatchCSCMuL1::GMTREGCAND> rtGMTRPCbCands;
max_pt = -1, max_pt_eta = -111;
for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
for ( vector<L1MuRegionalCand>::const_iterator trk = l1GmtRPCbCands.begin(); trk != l1GmtRPCbCands.end(); trk++)
  {
    if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
      {
	if (debugRATE) cout<<"discarding BX = "<< trk->bx() <<endl;
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
if (debugRATE) cout<< "----- end ngmt rpcb/ngmtpt10="<<ngmtrpcb<<"/"<<ngmtrpcbpt10<<endl;


int ngmtdt=0, ngmtdtpt10=0;
if (debugRATE) cout<< "----- statring ngmt dt"<<endl;
vector<MatchCSCMuL1::GMTREGCAND> rtGMTDTCands;
max_pt = -1, max_pt_eta = -111;
for (int bx=minRateBX_; bx<=maxRateBX_; bx++) bx2n[bx]=0;
for ( vector<L1MuRegionalCand>::const_iterator trk = l1GmtDTCands.begin(); trk != l1GmtDTCands.end(); trk++)
  {
    if ( trk->bx() < minRateBX_ || trk->bx() > maxRateBX_ )
      {
	if (debugRATE) cout<<"discarding BX = "<< trk->bx() <<endl;
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
if (debugRATE) cout<< "----- end ngmt dt/ngmtpt10="<<ngmtdt<<"/"<<ngmtdtpt10<<endl;


//============ RATE GMT ==================

int ngmt=0;
if (debugRATE) cout<< "----- statring ngmt"<<endl;
vector<MatchCSCMuL1::GMTCAND> rtGMTCands;
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

vector<L1MuGMTReadoutRecord> gmt_records = hl1GmtCands->getRecords();
for ( vector< L1MuGMTReadoutRecord >::const_iterator rItr=gmt_records.begin(); rItr!=gmt_records.end() ; ++rItr )
  {
    if (rItr->getBxInEvent() < minBxGMT_ || rItr->getBxInEvent() > maxBxGMT_) continue;

    vector<L1MuRegionalCand> CSCCands = rItr->getCSCCands();
    vector<L1MuRegionalCand> DTCands  = rItr->getDTBXCands();
    vector<L1MuRegionalCand> RPCfCands = rItr->getFwdRPCCands();
    vector<L1MuRegionalCand> RPCbCands = rItr->getBrlRPCCands();
    vector<L1MuGMTExtendedCand> GMTCands = rItr->getGMTCands();
    for ( vector<L1MuGMTExtendedCand>::const_iterator  muItr = GMTCands.begin() ; muItr != GMTCands.end() ; ++muItr )
      {
	if( muItr->empty() ) continue;

	if ( muItr->bx() < minRateBX_ || muItr->bx() > maxRateBX_ )
	  {
	    if (debugRATE) cout<<"discarding BX = "<< muItr->bx() <<endl;
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
	if (muItr->isFwd() && ( muItr->isMatchedCand() || !muItr->isRPC())) {
	  L1MuRegionalCand rcsc = CSCCands[muItr->getDTCSCIndex()];
	  unsigned my_i = 999;
	  for (unsigned i=0; i< rtGMTREGCands.size(); i++)
	    {
	      if (rcsc.getDataWord()!=rtGMTREGCands[i].l1reg->getDataWord()) continue;
	      my_i = i;
	      break;
	    }
	  if (my_i<99) gmt_csc = &rtGMTREGCands[my_i];
	  else cout<<"DOES NOT EXIST IN rtGMTREGCands! Should not happen!"<<endl;
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
	    else cout<<"DOES NOT EXIST IN rtGMTRPCfCands! Should not happen!"<<endl;
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
	    else cout<<"DOES NOT EXIST IN rtGMTRPCbCands! Should not happen!"<<endl;
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
	    else cout<<"DOES NOT EXIST IN rtGMTDTCands! Should not happen!"<<endl;
	    myGMTCand.regcand = gmt_dt;
	  }

	if ( (gmt_csc != NULL && gmt_rpcf != NULL) && !muItr->isMatchedCand() ) cout<<"csc&rpcf but not matched!"<<endl;

	bool eta_me42 = isME42EtaRegion(myGMTCand.eta);
	bool eta_me42r = isME42RPCEtaRegion(myGMTCand.eta);
	//if (geta>=1.2 && geta<=1.8) eta_me42 = 1;
	bool eta_q = (geta > 1.2);

	bool eta_me1b = isME1bEtaRegion(myGMTCand.eta);
	//bool eta_me1b_whole = isME1bEtaRegion(myGMTCand.eta, 1.6, 2.14);
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
	  if (gtype==0) cout<<"weird: gtype=0 That shouldn't happen!";
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

	if (isCSC && gmt_csc->tfcand != NULL && gmt_csc->tfcand->tftrack == NULL) cout<<"warning: gmt_csc->tfcand->tftrack == NULL"<<endl;
	if (isCSC && gmt_csc->tfcand != NULL && gmt_csc->tfcand->tftrack != NULL && gmt_csc->tfcand->tftrack->l1trk == NULL)
	  cout<<"warning: gmt_csc->tfcand->tftrack->l1trk == NULL"<<endl;
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
if (debugRATE) cout<< "----- end ngmt="<<ngmt<<endl;


for (unsigned int i=0; i<matches.size(); i++) delete matches[i];
matches.clear ();

cleanUp();
return true;
}


// ================================================================================================
void 
SimMuL1::cleanUp()
{
  if (addGhostLCTs_)
    {
      for (size_t i=0; i<ghostLCTs.size();i++) if (ghostLCTs[i]) delete ghostLCTs[i];
      ghostLCTs.clear();
    }
}

// ================================================================================================
void 
SimMuL1::runCSCTFSP(const CSCCorrelatedLCTDigiCollection* mplcts, const L1MuDTChambPhContainer* dttrig)
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
  for (int s=0; s<6; s++) {
    CSCTriggerContainer<csctf::TrackStub> current_e_s = stub_list.get(e+1, s+1);
    if (current_e_s.get().size()>0) {
      cout<<"sector "<<s+1<<":"<<endl<<endl;
      my_SPs[e][s]->run(current_e_s);
    }
  }

}

// ================================================================================================
void 
SimMuL1::propagateToCSCStations(MatchCSCMuL1 *match)
{
  // do not propagate for hight etas
  if (fabs(match->strk->momentum().eta())>2.6) return;

  TrajectoryStateOnSurface tsos;

  // z planes
  int endcap = (match->strk->momentum().eta() >= 0) ? 1 : -1;
  double zME11 = endcap*585.;
  double zME1  = endcap*615.;
  double zME2  = endcap*830.;
  double zME3  = endcap*935.;
  
  // extrapolate to ME1/1 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME11);
  if (tsos.isValid()) {
    math::XYZVectorD vgp( tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z() );
    match->pME11 = vgp;
  }
  // extrapolate to ME1 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME1);
  if (tsos.isValid()) {
    math::XYZVectorD vgp( tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z() );
    match->pME1 = vgp;
  }
  // extrapolate to ME2 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME2);
  if (tsos.isValid()) {
    math::XYZVectorD vgp( tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z() );
    match->pME2 = vgp;
  }
  // extrapolate to ME3 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME3);
  if (tsos.isValid()) {
    math::XYZVectorD vgp( tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z() );
    match->pME3 = vgp;
  }
}

// ================================================================================================
void 
SimMuL1::matchSimTrack2SimHits( MatchCSCMuL1 * match, 
				const SimTrackContainer & simTracks, 
				const SimVertexContainer & simVertices, 
				const PSimHitContainer * allCSCSimHits )
{
  // Matching of SimHits that were created by SimTrack

  // collect all ID of muon SimTrack children
  match->familyIds = fillSimTrackFamilyIds(match->strk->trackId(), simTracks, simVertices);

  // match SimHits to SimTracks
  vector<PSimHit> matchingSimHits = hitsFromSimTrack(match->familyIds, theCSCSimHitMap);
  for (unsigned i=0; i<matchingSimHits.size();i++) {
    if (goodChambersOnly_)
      if ( theStripConditions->isInBadChamber( CSCDetId( matchingSimHits[i].detUnitId() ) ) ) continue; // skip 'bad' chamber
    match->addSimHit(matchingSimHits[i]);
  }

  // checks
  unsigned stNhist = 0;
  for (edm::PSimHitContainer::const_iterator hit = allCSCSimHits->begin();  hit != allCSCSimHits->end();  ++hit) 
    {
      if (hit->trackId() != match->strk->trackId()) continue;
      CSCDetId chId(hit->detUnitId());
      if ( chId.station() == 1 && chId.ring() == 4 && !doME1a_) continue;
      stNhist++;
    }
  if (doStrictSimHitToTrackMatch_ && stNhist != match->simHits.size()) 
    {
      cout <<" ALARM!!! matchSimTrack2SimHits: stNhist != stHits.size()  ---> "<<stNhist <<" != "<<match->simHits.size()<<endl;
      stNhist = 0;
      if (debugALLEVENT) for (edm::PSimHitContainer::const_iterator hit = allCSCSimHits->begin();  hit != allCSCSimHits->end();  ++hit) 
			   if (hit->trackId() == match->strk->trackId()) {
			     CSCDetId chId(hit->detUnitId());
			     if ( !(chId.station() == 1 && chId.ring() == 4 && !doME1a_) ) cout<<"   "<<chId<<"  "<<(*hit)<<" "<<hit->momentumAtEntry()<<" "<<hit->energyLoss()<<" "<<hit->particleType()<<" "<<hit->trackId()<<endl;
			   }
    }

  if (debugALLEVENT) {
    cout<<"--- SimTrack hits: "<< match->simHits.size()<<endl;
    for (unsigned j=0; j<match->simHits.size(); j++) {
      PSimHit & sh = (match->simHits)[j];
      cout<<"   "<<sh<<" "<<sh.exitPoint()<<"  "<<sh.momentumAtEntry()<<" "<<sh.energyLoss()<<" "<<sh.particleType()<<" "<<sh.trackId()<<endl;
    }
  }
}


// ================================================================================================
void
SimMuL1::matchSimTrack2ALCTs(MatchCSCMuL1 *match, 
			     const PSimHitContainer* allCSCSimHits, 
			     const CSCALCTDigiCollection *alcts, 
			     const CSCWireDigiCollection* wiredc )
{
  // tool for matching SimHits to ALCTs
  //CSCAnodeLCTAnalyzer alct_analyzer;
  //alct_analyzer.setDebug();
  //alct_analyzer.setGeometry(cscGeometry);

  if (debugALCT) cout<<"--- ALCT-SimHits ---- begin for trk "<<match->strk->trackId()<<endl;
  
  map<int, vector<CSCALCTDigi> > checkNALCT;
  checkNALCT.clear();

  match->ALCTs.clear();
  for (CSCALCTDigiCollection::DigiRangeIterator  adetUnitIt = alcts->begin(); 
       adetUnitIt != alcts->end(); adetUnitIt++)
    {
      const CSCDetId& id = (*adetUnitIt).first;
      const CSCALCTDigiCollection::Range& range = (*adetUnitIt).second;
      int nm=0;

      //if (id.station()==1&&id.ring()==2) debugALCT=1;
      CSCDetId id1a(id.endcap(),id.station(),4,id.chamber(),0);

      for (CSCALCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
	{
	  checkNALCT[id.rawId()].push_back(*digiIt);
	  nm++;
      
	  if (!(*digiIt).isValid()) continue;

	  bool me1a_all = (defaultME1a && id.station()==1 && id.ring()==1 && (*digiIt).getKeyWG() <= 15);
	  bool me1a_no_overlap = ( me1a_all && (*digiIt).getKeyWG() < 10 );

	  vector<PSimHit> trackHitsInChamber = match->chamberHits(id.rawId());
	  vector<PSimHit> trackHitsInChamber1a;
	  if (me1a_all) trackHitsInChamber1a = match->chamberHits(id1a.rawId());

	  if (trackHitsInChamber.size() + trackHitsInChamber1a.size() == 0 ) // no point to do any matching here
	    {
	      if (debugALCT) cout<<"raw ID "<<id.rawId()<<" "<<id<<"  #"<<nm<<"   no SimHits in chamber from this SimTrack!"<<endl;
	      continue;
	    }

	  if ( (*digiIt).getBX()-6 < minBX_ || (*digiIt).getBX()-6 > maxBX_ )
	    {
	      if (debugALCT) cout<<"discarding BX = "<< (*digiIt).getBX()-6 <<endl;
	      continue;
	    }

	  vector<CSCAnodeLayerInfo> alctInfo;
	  //vector<CSCAnodeLayerInfo> alctInfo = alct_analyzer.getSimInfo(*digiIt, id, wiredc, allCSCSimHits);
	  vector<PSimHit> matchedHits;
	  unsigned nmhits = matchCSCAnodeHits(alctInfo, matchedHits);

	  MatchCSCMuL1::ALCT malct(match);
	  malct.trgdigi = &*digiIt;
	  malct.layerInfo = alctInfo;
	  malct.simHits = matchedHits;
	  malct.id = id;
	  malct.nHitsShared = 0;
	  calculate2DStubsDeltas(match, malct);
	  malct.deltaOk = (minDeltaWire_ <= malct.deltaWire) & (malct.deltaWire <= maxDeltaWire_);

	  vector<CSCAnodeLayerInfo> alctInfo1a;
	  vector<PSimHit> matchedHits1a;
	  unsigned nmhits1a = 0;

	  MatchCSCMuL1::ALCT malct1a(match);
	  if (me1a_all) {
	    //alctInfo1a = alct_analyzer.getSimInfo(*digiIt, id1a, wiredc, allCSCSimHits);
	    nmhits1a = matchCSCAnodeHits(alctInfo, matchedHits1a);

	    malct1a.trgdigi = &*digiIt;
	    malct1a.layerInfo = alctInfo1a;
	    malct1a.simHits = matchedHits1a;
	    malct1a.id = id1a;
	    malct1a.nHitsShared = 0;
	    calculate2DStubsDeltas(match, malct1a);
	    malct1a.deltaOk = (minDeltaWire_ <= malct1a.deltaWire) & (malct1a.deltaWire <= maxDeltaWire_);
	  }

	  if (debugALCT) cout<<"raw ID "<<id.rawId()<<" "<<id<<"  #"<<nm<<"    NTrackHitsInChamber  nmhits  alctInfo.size  diff  "
			     <<trackHitsInChamber.size()<<" "<<nmhits<<" "<<alctInfo.size()<<"  "
			     << nmhits-alctInfo.size() <<endl
			     << "  "<<(*digiIt)<<"  DW="<<malct.deltaWire<<" eta="<<malct.eta;

	  if (nmhits + nmhits1a > 0)
	    {
	      //if (debugALCT) cout<<"  --- matched to ALCT hits: "<<endl;

	      int nHitsMatch = 0;
	      for (unsigned i=0; i<nmhits;i++)
		{
		  //if (debugALCT) cout<<"   "<<matchedHits[i]<<" "<<matchedHits[i].exitPoint()<<"  "
		  //                   <<matchedHits[i].momentumAtEntry()<<" "<<matchedHits[i].energyLoss()<<" "
		  //                   <<matchedHits[i].particleType()<<" "<<matchedHits[i].trackId();
		  //bool wasmatch = 0;
		  for (unsigned j=0; j<trackHitsInChamber.size(); j++)
		    if ( compareSimHits ( matchedHits[i], trackHitsInChamber[j] ) )
		      {
			nHitsMatch++;
			//wasmatch = 1;
		      }
		  //if (debugALCT)  {if (wasmatch) cout<<" --> match!"<<endl;  else cout<<endl;}
		}
	      malct.nHitsShared = nHitsMatch;
	      if( me1a_all ) {
		nHitsMatch = 0;
		for (unsigned i=0; i<nmhits1a;i++) {
		  //bool wasmatch = 0;
		  for (unsigned j=0; j<trackHitsInChamber1a.size(); j++)
		    if ( compareSimHits ( matchedHits1a[i], trackHitsInChamber1a[j] ) )
		      {
			nHitsMatch++;
			//wasmatch = 1;
		      }
		}
		malct1a.nHitsShared = nHitsMatch;
	      }
	    }
	  else if (debugALCT) cout<< "  +++ ALCT warning: no simhits for its digi found!\n";

	  if (debugALCT) cout<<"  nHitsShared="<<malct.nHitsShared<<endl;

	  if(matchAllTrigPrimitivesInChamber_)
	    {
	      // if specified, try DY match
	      bool dymatch = 0;
	      if ( fabs(malct.deltaY)<= minDeltaYAnode_ )
		{
		  if (debugALCT)  for (unsigned i=0; i<trackHitsInChamber.size();i++)
				    cout<<"   DY match: "<<trackHitsInChamber[i]<<" "<<trackHitsInChamber[i].exitPoint()<<"  "
					<<trackHitsInChamber[i].momentumAtEntry()<<" "<<trackHitsInChamber[i].energyLoss()<<" "
					<<trackHitsInChamber[i].particleType()<<" "<<trackHitsInChamber[i].trackId()<<endl;
  
		  if (!me1a_no_overlap) match->ALCTs.push_back(malct);
		  dymatch = true;
		}
	      if ( me1a_all && fabs(malct1a.deltaY)<= minDeltaYAnode_) {
		if (!me1a_no_overlap) match->ALCTs.push_back(malct1a);
		dymatch = true;
	      }
	      if (dymatch) continue;

	      // whole chamber match
	      if ( minDeltaYAnode_ < 0  )
		{
		  if (debugALCT)  for (unsigned i=0; i<trackHitsInChamber.size();i++)
				    cout<<"   chamber match: "<<trackHitsInChamber[i]<<" "<<trackHitsInChamber[i].exitPoint()<<"  "
					<<trackHitsInChamber[i].momentumAtEntry()<<" "<<trackHitsInChamber[i].energyLoss()<<" "
					<<trackHitsInChamber[i].particleType()<<" "<<trackHitsInChamber[i].trackId()<<endl;
  
		  if (!me1a_no_overlap) match->ALCTs.push_back(malct);
		  if (me1a_all) match->ALCTs.push_back(malct1a);
		  continue;
		}
	      continue;
	    }

	  // else proceed with hit2hit matching:
	  if (minNHitsShared_>=0)
	    {
	      if (!me1a_no_overlap && malct.nHitsShared >= minNHitsShared_) {
		if (debugALCT)  cout<<" --> shared hits match!"<<endl;
		match->ALCTs.push_back(malct);
	      }
	      if (me1a_all && malct1a.nHitsShared >= minNHitsShared_) {
		if (debugALCT)  cout<<" --> shared hits match!"<<endl;
		match->ALCTs.push_back(malct);
	      }
	    }

	  // else proceed with deltaWire matching:
	  if (!me1a_no_overlap && minDeltaWire_ <= malct.deltaWire && malct.deltaWire <= maxDeltaWire_){
	    if (debugALCT)  cout<<" --> deltaWire match!"<<endl;
	    match->ALCTs.push_back(malct);
	  }

	  // special case of default emulator with puts all ME11 alcts into ME1b
	  // only for deltaWire matching!
	  if (me1a_all && minDeltaWire_ <= malct1a.deltaWire && malct1a.deltaWire <= maxDeltaWire_){
	    if (debugALCT)  cout<<" --> deltaWire match!"<<endl;
	    match->ALCTs.push_back(malct);
	  }
	}
      //debugALCT=0;
    } // loop CSCALCTDigiCollection
  
  if (debugALCT) for(map<int, vector<CSCALCTDigi> >::const_iterator mapItr = checkNALCT.begin(); mapItr != checkNALCT.end(); ++mapItr)
		   if (mapItr->second.size()>2) {
		     CSCDetId idd(mapItr->first);
		     cout<<"~~~~ checkNALCT WARNING! nALCT = "<<mapItr->second.size()<<" in ch "<< mapItr->first<<" "<<idd<<endl;
		     for (unsigned i=0; i<mapItr->second.size();i++) cout<<"~~~~~~ ALCT "<<i<<" "<<(mapItr->second)[i]<<endl;
		   }

  if (debugALCT) cout<<"--- ALCT-SimHits ---- end"<<endl;
}



// ================================================================================================
void
SimMuL1::matchSimTrack2CLCTs(MatchCSCMuL1 *match, 
			     const PSimHitContainer* allCSCSimHits, 
			     const CSCCLCTDigiCollection *clcts, 
			     const CSCComparatorDigiCollection* compdc )
{
  // tool for matching SimHits to CLCTs
  //CSCCathodeLCTAnalyzer clct_analyzer;
  //clct_analyzer.setDebug();
  //clct_analyzer.setGeometry(cscGeometry);

  if (debugCLCT) cout<<"--- CLCT-SimHits ---- begin for trk "<<match->strk->trackId()<<endl;
  //static const int key_layer = 4; //CSCConstants::KEY_CLCT_LAYER

  map<int, vector<CSCCLCTDigi> > checkNCLCT;
  checkNCLCT.clear();
  
  match->CLCTs.clear();
  for (CSCCLCTDigiCollection::DigiRangeIterator  cdetUnitIt = clcts->begin(); 
       cdetUnitIt != clcts->end(); cdetUnitIt++)
    {
      const CSCDetId& id = (*cdetUnitIt).first;
      const CSCCLCTDigiCollection::Range& range = (*cdetUnitIt).second;
      int nm=0;
      CSCDetId cid = id;

      //if (id.station()==1&&id.ring()==2) debugCLCT=1;

      for (CSCCLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
	{
	  checkNCLCT[id.rawId()].push_back(*digiIt);
	  nm++;
      
	  if (!(*digiIt).isValid()) continue;

	  bool me1a_case = (defaultME1a && id.station()==1 && id.ring()==1 && (*digiIt).getKeyStrip() > 127);
	  if (me1a_case){
	    CSCDetId id1a(id.endcap(),id.station(),4,id.chamber(),0);
	    cid = id1a;
	  }

	  vector<PSimHit> trackHitsInChamber = match->chamberHits(cid.rawId());

	  if (trackHitsInChamber.size()==0) // no point to do any matching here
	    {
	      if (debugCLCT) cout<<"raw ID "<<cid.rawId()<<" "<<cid<<"  #"<<nm<<"   no SimHits in chamber from this SimTrack!"<<endl;
	      continue;
	    }

	  if ( (*digiIt).getBX()-5 < minBX_ || (*digiIt).getBX()-7 > maxBX_ )
	    {
	      if (debugCLCT) cout<<"discarding BX = "<< (*digiIt).getBX()-6 <<endl;
	      continue;
	    }

	  // don't use it anymore: replace with a dummy
	  vector<CSCCathodeLayerInfo> clctInfo;
	  //vector<CSCCathodeLayerInfo> clctInfo = clct_analyzer.getSimInfo(*digiIt, cid, compdc, allCSCSimHits);

	  vector<PSimHit> matchedHits;
	  unsigned nmhits = matchCSCCathodeHits(clctInfo, matchedHits);

	  MatchCSCMuL1::CLCT mclct(match);
	  mclct.trgdigi = &*digiIt;
	  mclct.layerInfo = clctInfo;
	  mclct.simHits = matchedHits;
	  mclct.id = cid;
	  mclct.nHitsShared = 0;
	  calculate2DStubsDeltas(match, mclct);
	  mclct.deltaOk = (abs(mclct.deltaStrip) <= minDeltaStrip_);

	  if (debugCLCT) cout<<"raw ID "<<cid.rawId()<<" "<<cid<<"  #"<<nm<<"    NTrackHitsInChamber  nmhits  clctInfo.size  diff  "
			     <<trackHitsInChamber.size()<<" "<<nmhits<<" "<<clctInfo.size()<<"  "
			     << nmhits-clctInfo.size() <<endl
			     << "  "<<(*digiIt)<<"  DS="<<mclct.deltaStrip<<" phi="<<mclct.phi;

	  if (nmhits > 0)
	    {
	      //if (debugCLCT) cout<<"  --- matched to CLCT hits: "<<endl;

	      int nHitsMatch = 0;
	      for (unsigned i=0; i<nmhits;i++)
		{
		  //if (debugCLCT) cout<<"   "<<matchedHits[i]<<" "<<matchedHits[i].exitPoint()<<"  "
		  //                   <<matchedHits[i].momentumAtEntry()<<" "<<matchedHits[i].energyLoss()<<" "
		  //                   <<matchedHits[i].particleType()<<" "<<matchedHits[i].trackId();
		  //bool wasmatch = 0;
		  for (unsigned j=0; j<trackHitsInChamber.size(); j++)
		    if ( compareSimHits ( matchedHits[i], trackHitsInChamber[j] ) )
		      {
			nHitsMatch++;
			//wasmatch = 1;
		      }
		  //if (debugCLCT)  {if (wasmatch) cout<<" --> match!"<<endl;  else cout<<endl;}
		}
	      mclct.nHitsShared = nHitsMatch;
	    }
	  else if (debugCLCT) cout<< "  +++ CLCT warning: no simhits for its digi found!\n";

	  if (debugCLCT) cout<<"  nHitsShared="<<mclct.nHitsShared<<endl;

	  if(matchAllTrigPrimitivesInChamber_)
	    {
	      if ( fabs(mclct.deltaY)<= minDeltaYCathode_)
		{
		  if (debugCLCT)  for (unsigned i=0; i<trackHitsInChamber.size();i++)
				    cout<<"   DY match: "<<trackHitsInChamber[i]<<" "<<trackHitsInChamber[i].exitPoint()<<"  "
					<<trackHitsInChamber[i].momentumAtEntry()<<" "<<trackHitsInChamber[i].energyLoss()<<" "
					<<trackHitsInChamber[i].particleType()<<" "<<trackHitsInChamber[i].trackId()<<endl;
  
		  match->CLCTs.push_back(mclct);
		  continue;
		}
	      if ( minDeltaYCathode_ < 0  )
		{
		  if (debugCLCT)  for (unsigned i=0; i<trackHitsInChamber.size();i++)
				    cout<<"   chamber match: "<<trackHitsInChamber[i]<<" "<<trackHitsInChamber[i].exitPoint()<<"  "
					<<trackHitsInChamber[i].momentumAtEntry()<<" "<<trackHitsInChamber[i].energyLoss()<<" "
					<<trackHitsInChamber[i].particleType()<<" "<<trackHitsInChamber[i].trackId()<<endl;
  
		  match->CLCTs.push_back(mclct);
		  continue;
		}
	      continue;
	    }
	  // else proceed with hit2hit matching:
 
	  if (mclct.nHitsShared >= minNHitsShared_) {
	    if (debugCLCT)  cout<<" --> shared hits match!"<<endl;
	    match->CLCTs.push_back(mclct);
	  }

	  // else proceed with hit2hit matching:
	  if (minNHitsShared_>=0)
	    {
	      if (mclct.nHitsShared >= minNHitsShared_) {
		if (debugCLCT)  cout<<" --> shared hits match!"<<endl;
		match->CLCTs.push_back(mclct);
	      }
	    }

	  // else proceed with deltaStrip matching:
	  if (abs(mclct.deltaStrip) <= minDeltaStrip_) {
	    if (debugCLCT)  cout<<" --> deltaStrip match!"<<endl;
	    match->CLCTs.push_back(mclct);
	  }
	}
      //debugCLCT=0;

    } // loop CSCCLCTDigiCollection

  if (debugCLCT) for(map<int, vector<CSCCLCTDigi> >::const_iterator mapItr = checkNCLCT.begin(); mapItr != checkNCLCT.end(); ++mapItr)
		   if (mapItr->second.size()>2) {
		     CSCDetId idd(mapItr->first);
		     cout<<"~~~~ checkNCLCT WARNING! nCLCT = "<<mapItr->second.size()<<" in ch "<< mapItr->first<<" "<<idd<<endl;
		     for (unsigned i=0; i<mapItr->second.size();i++) cout<<"~~~~~~ CLCT "<<i<<" "<<(mapItr->second)[i]<<endl;
		   }
  
  if (debugCLCT) cout<<"--- CLCT-SimHits ---- end"<<endl;
}




// ================================================================================================
void
SimMuL1::matchSimTrack2LCTs(MatchCSCMuL1 *match, 
			    const CSCCorrelatedLCTDigiCollection* lcts )
{
  if (debugLCT) cout<<"--- LCT ---- begin"<<endl;
  int nValidLCTs = 0, nCorrelLCTs = 0, nALCTs = 0, nCLCTs = 0;
  match->LCTs.clear();

  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt = lcts->begin(); 
       detUnitIt != lcts->end(); detUnitIt++) 
    {
      const CSCDetId& id = (*detUnitIt).first;
      const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
      CSCDetId cid = id;

      //if (id.station()==1&&id.ring()==2) debugLCT=1;

      for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
	{
	  if (!(*digiIt).isValid()) continue;

	  bool me1a_case = (defaultME1a && id.station()==1 && id.ring()==1 && (*digiIt).getStrip() > 127);
	  if (me1a_case){
	    CSCDetId id1a(id.endcap(),id.station(),4,id.chamber(),0);
	    cid = id1a;
	  }

	  if (debugLCT) cout<< "----- LCT in raw ID "<<cid.rawId()<<" "<<cid<< " (trig id. " << id.triggerCscId() << ")"<<endl;
	  if (debugLCT) cout<< " "<< (*digiIt);
	  nValidLCTs++;

	  if ( (*digiIt).getBX()-6 < minTMBBX_ || (*digiIt).getBX()-6 > maxTMBBX_ )
	    {
	      if (debugLCT) cout<<"discarding BX = "<< (*digiIt).getBX()-6 <<endl;
	      continue;
	    }

	  int quality = (*digiIt).getQuality();

	  //bool alct_valid = (quality != 4 && quality != 5);
	  //bool clct_valid = (quality != 1 && quality != 3);
	  bool alct_valid = (quality != 2);
	  bool clct_valid = (quality != 1);

	  if (debugLCT && !alct_valid) cout<<"  +++ note: valid LCT but not alct_valid: quality = "<<quality<<endl;
	  if (debugLCT && !clct_valid) cout<<"  +++ note: valid LCT but not clct_valid: quality = "<<quality<<endl;

	  int nmalct = 0;
	  MatchCSCMuL1::ALCT *malct = 0;
	  if ( alct_valid )
	    {
	      for (unsigned i=0; i< match->ALCTs.size(); i++)
		if ( cid.rawId() == (match->ALCTs)[i].id.rawId() &&
		     (*digiIt).getKeyWG() == (match->ALCTs)[i].trgdigi->getKeyWG() &&
		     (*digiIt).getBX() == (match->ALCTs)[i].getBX() )
		  {
		    if (debugLCT) cout<< "  ----- ALCT matches LCT: "<<(match->ALCTs)[i].id<<"  "<<*((match->ALCTs)[i].trgdigi) <<endl;
		    malct = &((match->ALCTs)[i]);
		    nmalct++;
		  }
	      if (nmalct>1) cout<<"+++ ALARM in LCT: number of matching ALCTs is more than one: "<<nmalct<<endl;
	    }

	  int nmclct = 0;
	  MatchCSCMuL1::CLCT *mclct = 0;
	  vector<MatchCSCMuL1::CLCT*> vmclct;
	  if ( clct_valid )
	    {
	      for (unsigned i=0; i< match->CLCTs.size(); i++)
		if ( cid.rawId() == (match->CLCTs)[i].id.rawId() &&
		     (*digiIt).getStrip() == (match->CLCTs)[i].trgdigi->getKeyStrip() &&
		     //(*digiIt).getCLCTPattern() == (match->CLCTs)[i].trgdigi->getPattern() )
		     (*digiIt).getPattern() == (match->CLCTs)[i].trgdigi->getPattern() )
		  {
		    if (debugLCT) cout<< "  ----- CLCT matches LCT: "<<(match->CLCTs)[i].id<<"  "<<*((match->CLCTs)[i].trgdigi) <<endl;
		    mclct = &((match->CLCTs)[i]);
		    vmclct.push_back(mclct);
		    nmclct++;
		  }
	      if (nmclct>1) {
		cout<<"+++ ALARM in LCT: number of matching CLCTs is more than one: "<<nmclct<<endl;
		// choose the smallest bx one
		int mbx=999, mnn=0;
		for (int nn=0; nn<nmclct;nn++) if (vmclct[nn]->getBX()<mbx)
						 {
						   mbx=vmclct[nn]->getBX();
						   mnn=nn;
						 }
		mclct = vmclct[mnn];
		cout<<"+++ ALARM in LCT: number of matching CLCTs is more than one: "<<nmclct<<"  choosing one with bx="<<mbx<<endl;
	      }
	    }

	  MatchCSCMuL1::LCT mlct(match);
	  mlct.trgdigi = &*digiIt;
	  mlct.id = cid;
	  mlct.ghost = 0;
	  mlct.deltaOk = 0;

	  // Truly correlated LCTs matched to SimTrack's ALCTs and CLCTs
	  if (alct_valid && clct_valid)
	    {
	      nCorrelLCTs++;
	      //if (nmclct+nmalct > 2) cout<<"+++ ALARM!!! too many matches to LTCs: nmalct="<<nmalct<< "  nmclct="<<nmclct<<endl;
	      if (nmalct && nmclct)
		{
		  mlct.alct = malct;
		  mlct.clct = mclct;
		  mlct.deltaOk = (malct->deltaOk & mclct->deltaOk);
		  match->LCTs.push_back(mlct);
		  if (debugLCT) cout<< "  ------------> LCT matches ALCT & CLCT "<<endl;
		}
	    }
	  // ALCT only LCTs
	  //else if ( alct_valid )
	  //{
	  //  nALCTs++;
	  //  if (nmalct)
	  //  {
	  //    mlct.alct = malct;
	  //    mlct.clct = 0;
	  //    match->LCTs.push_back(mlct);
	  //    if (debugLCT) cout<< "  ------------> LCT matches ALCT only"<<endl;
	  //  }
	  //}
	  // CLCT only LCTs
	  else if ( clct_valid )
	    {
	      nCLCTs++;
	      if (nmclct)
		{
		  mlct.alct = 0;
		  mlct.clct = mclct;
		  match->LCTs.push_back(mlct);
		  if (debugLCT) cout<< "  ------------> LCT matches CLCT only"<<endl;
		}
	    } // if (alct_valid && clct_valid)  ...
	}
      //debugLCT=0;  
    }

  // Adding ghost LCT combinatorics
  vector<MatchCSCMuL1::LCT> ghosts;
  if (addGhostLCTs_)
    {
      vector<int> chIDs = match->chambersWithLCTs();
      for (size_t ch = 0; ch < chIDs.size(); ch++) 
	{
	  vector<MatchCSCMuL1::LCT> chlcts = match->chamberLCTs(chIDs[ch]);
	  if (chlcts.size()<2) continue;
	  if (debugLCT) cout<<"Ghost LCT combinatorics: "<<chlcts.size()<<" in chamber "<<chlcts[0].id<<endl;
	  map<int,vector<MatchCSCMuL1::LCT> > bxlcts;
	  for (size_t t=0; t < chlcts.size(); t++) {
	    int bx=chlcts[t].getBX();
	    bxlcts[bx].push_back(chlcts[t]);
	    if (bxlcts[bx].size() > 2 ) cout<<" Huh!?? "<<" n["<<bx<<"] = 2"<<endl;
	    if (bxlcts[bx].size() == 2)
	      {
		MatchCSCMuL1::LCT lt[2];
		lt[0] = (bxlcts[bx])[0];
		lt[1] = (bxlcts[bx])[1];
		bool sameALCT = ( lt[0].alct->trgdigi->getKeyWG() == lt[1].alct->trgdigi->getKeyWG() );
		bool sameCLCT = ( lt[0].clct->trgdigi->getKeyStrip() == lt[1].clct->trgdigi->getKeyStrip() );
		if (debugLCT) {
		  cout<<" n["<<bx<<"] = 2 sameALCT="<<sameALCT<<" sameCLCT="<<sameCLCT<<endl
		      <<" lct1: "<<*(lt[0].trgdigi)
		      <<" lct2: "<<*(lt[1].trgdigi);
		}
		if (sameALCT||sameCLCT) continue;
	  
		unsigned int q[2];
		q[0]=findQuality(*(lt[0].alct->trgdigi),*(lt[1].clct->trgdigi));
		q[1]=findQuality(*(lt[1].alct->trgdigi),*(lt[0].clct->trgdigi));
		if (debugLCT) cout<<" q0="<<q[0]<<" q1="<<q[1]<<endl;
		int t3=0, t4=1;
		if (q[0]<q[1]) {t3=1;t4=0;}

		CSCCorrelatedLCTDigi* lctd3 = 
		  new  CSCCorrelatedLCTDigi (3, 1, q[t3], lt[t3].trgdigi->getKeyWG(),
					     lt[t4].trgdigi->getStrip(), lt[t4].trgdigi->getPattern(), lt[t4].trgdigi->getBend(),
					     bx, 0, 0, 0, lt[0].trgdigi->getCSCID());
		lctd3->setGEMDPhi( lt[t4].trgdigi->getGEMDPhi() );
		ghostLCTs.push_back(lctd3);
		MatchCSCMuL1::LCT mlct3(match);
		mlct3.trgdigi = lctd3;
		mlct3.alct = lt[t3].alct;
		mlct3.clct = lt[t4].clct;
		mlct3.id = lt[0].id;
		mlct3.ghost = 1;
		mlct3.deltaOk = (mlct3.alct->deltaOk & mlct3.clct->deltaOk);
		ghosts.push_back(mlct3);

		CSCCorrelatedLCTDigi* lctd4 =
		  new  CSCCorrelatedLCTDigi (4, 1, q[t4], lt[t4].trgdigi->getKeyWG(),
					     lt[t3].trgdigi->getStrip(), lt[t3].trgdigi->getPattern(), lt[t3].trgdigi->getBend(),
					     bx, 0, 0, 0, lt[0].trgdigi->getCSCID());
		lctd4->setGEMDPhi( lt[t3].trgdigi->getGEMDPhi() );
		ghostLCTs.push_back(lctd4);
		MatchCSCMuL1::LCT mlct4(match);
		mlct4.trgdigi = lctd4;
		mlct4.alct = lt[t4].alct;
		mlct4.clct = lt[t3].clct;
		mlct4.id = lt[0].id;
		mlct4.ghost = 1;
		mlct4.deltaOk = (mlct4.alct->deltaOk & mlct4.clct->deltaOk);
		ghosts.push_back(mlct4);

		if (debugLCT) cout<<" ghost 3: "<<*lctd3<<" ghost 4: "<<*lctd4;
	      }
	  }
	}
      if (ghosts.size()) match->LCTs.insert( match->LCTs.end(), ghosts.begin(), ghosts.end());
    }
  if (debugLCT) cout<<"--- valid LCTs "<<nValidLCTs<<"  Truly correlated LCTs : "<< nCorrelLCTs <<"  ALCT LCTs : "<< nALCTs <<"  CLCT LCTs : "<< nCLCTs <<"  ghosts:"<<ghosts.size()<<endl;;
  if (debugLCT) cout<<"--- LCT ---- end"<<endl;
}




// ================================================================================================
void
SimMuL1::matchSimTrack2MPLCTs(MatchCSCMuL1 *match, 
			      const CSCCorrelatedLCTDigiCollection* mplcts )
{
  if (debugMPLCT) cout<<"--- MPLCT ---- begin"<<endl;
  int nValidMPLCTs = 0, nCorrelMPLCTs = 0;
  match->MPLCTs.clear();

  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt = mplcts->begin(); 
       detUnitIt != mplcts->end(); detUnitIt++)
    {
      const CSCDetId& id = (*detUnitIt).first;
      const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
      CSCDetId cid = id;

      for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
	{
	  if (!(*digiIt).isValid()) continue;

	  bool me1a_case = (defaultME1a && id.station()==1 && id.ring()==1 && (*digiIt).getStrip() > 127);
	  if (me1a_case){
	    CSCDetId id1a(id.endcap(),id.station(),4,id.chamber(),0);
	    cid = id1a;
	  }

	  if (debugMPLCT) cout<< "----- MPLCT in raw ID "<<cid.rawId()<<" "<<cid<< " (trig id. " << cid.triggerCscId() << ")"<<endl;
	  if (debugMPLCT) cout<<" "<< (*digiIt);
	  nValidMPLCTs++;

	  if ( (*digiIt).getBX()-6 < minTMBBX_ || (*digiIt).getBX()-6 > maxTMBBX_ )
	    {
	      if (debugMPLCT) cout<<"discarding BX = "<< (*digiIt).getBX()-6 <<endl;
	      continue;
	    }

	  int quality = (*digiIt).getQuality();

	  //bool alct_valid = (quality != 4 && quality != 5);
	  //bool clct_valid = (quality != 1 && quality != 3);
	  bool alct_valid = (quality != 2);
	  bool clct_valid = (quality != 1);

	  if (debugMPLCT && !alct_valid) cout<<"+++ note: valid LCT but not alct_valid: quality = "<<quality<<endl;
	  if (debugMPLCT && !clct_valid) cout<<"+++ note: valid LCT but not clct_valid: quality = "<<quality<<endl;

	  // Truly correlated LCTs; for DAQ
	  //if (alct_valid && clct_valid)

	  if (clct_valid)
	    {
	      nCorrelMPLCTs++;
  
	      // match to TMB's LTSs

	      int nmlct = 0;
	      MatchCSCMuL1::LCT *mlct = 0;
	      for (unsigned i=0; i< match->LCTs.size(); i++)
		if ( cid.rawId() == (match->LCTs)[i].id.rawId() &&
		     (*digiIt).getKeyWG() == (match->LCTs)[i].trgdigi->getKeyWG() &&
		     (*digiIt).getStrip() == (match->LCTs)[i].trgdigi->getStrip())
		  {
		    mlct = &((match->LCTs)[i]);
		    nmlct++;
		    if (debugMPLCT) cout<< "  ---------> matched to corresponding LCT"<<endl;
		  }
	      if (nmlct>1) cout<<"+++ Warning in MPLCT: number of matching LCTs is more than one: "<<nmlct<<endl;

	      if (nmlct)
		{
		  MatchCSCMuL1::MPLCT mmplct(match);
		  mmplct.trgdigi = &*digiIt;
		  mmplct.lct = mlct;
		  mmplct.id = cid;
		  mmplct.ghost = 0;
		  mmplct.deltaOk = mmplct.lct->deltaOk;
		  mmplct.meEtap = 0;
		  mmplct.mePhip = 0;
		  match->MPLCTs.push_back(mmplct);
		}
	    }
	}
    }

  // Adding ghost MPC LCT combinatorics
  vector<MatchCSCMuL1::MPLCT> ghosts;
  if (addGhostLCTs_)
    {
      vector<int> chIDs = match->chambersWithMPLCTs();
      for (size_t ch = 0; ch < chIDs.size(); ch++) 
	{
	  vector<MatchCSCMuL1::MPLCT> chmplcts = match->chamberMPLCTs(chIDs[ch]);
	  if (chmplcts.size()<2) continue;
	  if (debugMPLCT) cout<<"Ghost MPLCT combinatorics: "<<chmplcts.size()<<" in chamber "<<chmplcts[0].id<<endl;
	  map<int,vector<MatchCSCMuL1::MPLCT> > bxmplcts;
	  for (size_t t=0; t < chmplcts.size(); t++) {
	    int bx=chmplcts[t].getBX();
	    bxmplcts[bx].push_back(chmplcts[t]);
	    if (bxmplcts[bx].size() > 2 ) cout<<" Huh!?? mpc "<<" n["<<bx<<"] = 2"<<endl;
	    if (bxmplcts[bx].size() == 2)
	      {
		vector<MatchCSCMuL1::LCT*> chlcts = match->chamberLCTsp(chIDs[ch]);
		if (debugMPLCT) cout<<" n["<<bx<<"] = 2 nLCT="<<chlcts.size()<<endl;
		if (chlcts.size()<2) continue;
		for (size_t tc=0; tc < chlcts.size(); tc++)
		  {
		    int bxlct = chlcts[tc]->getBX();
		    if ( bx!=bxlct || chlcts[tc]->ghost==0 ) continue;
		    MatchCSCMuL1::MPLCT mlct(match);
		    mlct.trgdigi = chlcts[tc]->trgdigi;
		    mlct.lct = chlcts[tc];
		    mlct.id = chlcts[tc]->id;
		    mlct.ghost = 1;
		    mlct.deltaOk = mlct.lct->deltaOk;
		    mlct.meEtap = 0;
		    mlct.mePhip = 0;
		    ghosts.push_back(mlct);
	    
		    if (debugMPLCT) cout<<" ghost added: "<<*(chlcts[tc]->trgdigi);
		  }
	      }
	  }
	}
      if (ghosts.size()) match->MPLCTs.insert( match->MPLCTs.end(), ghosts.begin(), ghosts.end());
    }
  
  // add eta and phi from TF srLUTs
  for (unsigned i=0; i<match->MPLCTs.size(); i++)
    {
      MatchCSCMuL1::MPLCT & mplct = match->MPLCTs[i];

      unsigned fpga = (mplct.id.station() == 1) ? CSCTriggerNumbering::triggerSubSectorFromLabels(mplct.id) - 1 : mplct.id.station();
      CSCSectorReceiverLUT* srLUT = srLUTs_[fpga][mplct.id.triggerSector()-1][mplct.id.endcap()-1];

      unsigned cscid = CSCTriggerNumbering::triggerCscIdFromLabels(mplct.id);
      unsigned cscid_special = cscid;
      if (mplct.id.station()==1 && mplct.id.ring()==4) cscid_special = cscid + 9;
      //cout<<" id="<<mplct.id<<" fpga="<<fpga<<" sect="<<mplct.id.triggerSector()<<" e="<<mplct.id.endcap()<<" cscid="<<cscid<<" cscid1="<<cscid_special<<" srLUT="<<srLUT
      //  <<" strip="<<mplct.trgdigi->getStrip()<<" patt="<<mplct.trgdigi->getPattern()<<" q="<<mplct.trgdigi->getQuality()<<" b="<<mplct.trgdigi->getBend()<<endl;
      
      lclphidat lclPhi;
      lclPhi = srLUT->localPhi(mplct.trgdigi->getStrip(), mplct.trgdigi->getPattern(), mplct.trgdigi->getQuality(), mplct.trgdigi->getBend());

      gblphidat gblPhi;
      gblPhi = srLUT->globalPhiME(lclPhi.phi_local, mplct.trgdigi->getKeyWG(), cscid_special);

      gbletadat gblEta;
      gblEta = srLUT->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, mplct.trgdigi->getKeyWG(), cscid);

      mplct.meEtap = gblEta.global_eta;
      mplct.mePhip = gblPhi.global_phi;

      if (debugMPLCT) cout<< "  got srLUTs meEtap="<<mplct.meEtap<<"  mePhip="<<mplct.mePhip<<endl;
    }
  
  if (debugMPLCT) cout<<"--- valid MPLCTs & Truly correlated MPLCTs : "<< nValidMPLCTs <<" "<< nCorrelMPLCTs <<"  ghosts:"<<ghosts.size()<<endl;;
  if (debugMPLCT) cout<<"--- MPLCT ---- end"<<endl;
}


// ================================================================================================
void
SimMuL1::matchSimtrack2TFTRACKs( MatchCSCMuL1 *match,
				 edm::ESHandle< L1MuTriggerScales > &muScales,
				 edm::ESHandle< L1MuTriggerPtScale > &muPtScale,
				 const L1CSCTrackCollection* l1Tracks)
{
  // TrackFinder's track is considered matched if it contains at least one of already matched MPLCTs
  // so, there is a possibility that several TF tracks would be matched
  if (debugTFTRACK) cout<<"--- TFTRACK ---- begin"<<endl;

  match->TFTRACKs.clear();
  for ( L1CSCTrackCollection::const_iterator trk = l1Tracks->begin(); trk != l1Tracks->end(); trk++)
    {

      MatchCSCMuL1::TFTRACK mtftrack(match);
      mtftrack.init( &(trk->first) , ptLUT, muScales, muPtScale);
    
      //mtftrack.dr = deltaR( match->strk->momentum().eta(), normalizedPhi( match->strk->momentum().phi() ), mtftrack.eta, mtftrack.phi );
      mtftrack.dr = match->deltaRSmart( mtftrack.eta, mtftrack.phi );
    
      double degs = mtftrack.phi/M_PI*180.;
      if (degs<0) degs += 360.;
      int Cphi = (int)(degs+5)/10+1;

      if (debugTFTRACK) cout<< "----- L1CSCTrack with  packed: eta="<<mtftrack.eta_packed<<" phi="<<mtftrack.phi_packed
			    <<" pt="<<mtftrack.pt_packed<<" qu="<<mtftrack.q_packed<<"  Cphi="<<Cphi
			    <<"  real: eta="<<mtftrack.eta<<"  phi=" <<mtftrack.phi
			    <<"  pt="<<mtftrack.pt<<"  dr="<<mtftrack.dr<<"  BX="<<mtftrack.l1trk->bx()<<endl;
      //if (debugTFTRACK) cout<< "----- ---- vaues: eta/phi "<<mtftrack.l1trk->etaValue()<<"/"<<mtftrack.l1trk->phiValue()<<"  ptValue "<<mtftrack.l1trk->ptValue()<<endl;

      for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt = trk->second.begin(); 
	   detUnitIt != trk->second.end(); detUnitIt++) 
	{
	  const CSCDetId& id = (*detUnitIt).first;
	  CSCDetId cid = id;
	  const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
	  for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
	    {
	      if (!((*digiIt).isValid())) cout<<"ALARM!!! matchSimtrack2TFTRACKs: L1CSCTrack.MPLCT is not valid id="<<id.rawId()<<" "<<id<<endl;

	      bool me1a_case = (defaultME1a && id.station()==1 && id.ring()==1 && (*digiIt).getStrip() > 127);
	      if (me1a_case){
		CSCDetId id1a(id.endcap(),id.station(),4,id.chamber(),0);
		cid = id1a;
	      }

	      if (debugTFTRACK) cout<< "------- L1CSCTrack.MPLCT in raw ID "<<cid.rawId()<<" "<<cid<<"  BX="<<digiIt->getBX()-6<<endl;

	      mtftrack.trgdigis.push_back( &*digiIt );
	      mtftrack.trgids.push_back( cid );
	      mtftrack.trgetaphis.push_back( intersectionEtaPhi(cid, (*digiIt).getKeyWG(), (*digiIt).getStrip()) );
	      mtftrack.trgstubs.push_back( buildTrackStub((*digiIt), cid) );

	      for (unsigned i=0; i< match->MPLCTs.size(); i++)
		{
		  MatchCSCMuL1::MPLCT & mplct = match->MPLCTs[i];

		  if ( cid.rawId()           != mplct.id.rawId() ||
		       (*digiIt).getKeyWG() != mplct.trgdigi->getKeyWG() ||
		       (*digiIt).getStrip() != mplct.trgdigi->getStrip()   ) continue;

		  mtftrack.mplcts.push_back(&mplct);
		  mtftrack.ids.push_back(mplct.id);
		  if (debugTFTRACK) cout<< "--------->   matched to MPLCTs["<<i<<"]"<<endl;
		  break;
		}
	    }
	}
      if (debugTFTRACK) cout<<"------- # of matched: CSCCorrelatedLCTDigis="<<mtftrack.trgids.size()<<"  MPLCTs="<<mtftrack.ids.size()<<endl;
      if (mtftrack.trgids.size()==1 && mtftrack.l1trk->mb1ID()==0 ){
	char msg[400];
	sprintf(msg, "TF track: nstubs=%lu  nmatchstubs=%lu", mtftrack.trgids.size(), mtftrack.ids.size());
	mtftrack.print(msg);
	cout<<"simtrack's matched MPLCTs:"<<endl;
	for (unsigned i=0; i< match->MPLCTs.size(); i++)
	  {
	    MatchCSCMuL1::MPLCT & mplct = match->MPLCTs[i];
	    if (mplct.deltaOk) cout<<"   "<<mplct.id<<"  wg="<<mplct.trgdigi->getKeyWG()<<"  str="<<mplct.trgdigi->getStrip()<<endl;
	  }
      }
      if (mtftrack.dr < 0.2) match->TFTRACKsAll.push_back(mtftrack);

      mtftrack.deltaOk1 = mtftrack.deltaOk2 = mtftrack.deltaOkME1 = 0;
      if (mtftrack.mplcts.size()) {
	// count matched
	bool okME1tf = 0;
	set<int> tfStations;
	for (unsigned i=0; i< mtftrack.mplcts.size(); i++)
	  {
	    if (!(((mtftrack.mplcts)[i])->deltaOk)) continue;
	    int st = ((mtftrack.mplcts)[i])->id.station();
	    tfStations.insert(st);
	    if (st == 1) okME1tf = 1;
	  }
	int okNtfmpc = tfStations.size();
	if (okNtfmpc>0) mtftrack.deltaOk1 = 1;
	if (okNtfmpc>1) mtftrack.deltaOk2 = 1;
	if (okME1tf) mtftrack.deltaOkME1 = 1;
      
	mtftrack.init( &(trk->first) , ptLUT, muScales, muPtScale);
      
	match->TFTRACKs.push_back(mtftrack);
      }
    
      //else if (debugTFTRACK) cout<<"----- NO MPLCTs found for this L1CSCTrack"<<endl;
    }
  if (debugTFTRACK) cout<<"---- # of matched TFTRACKs = "<<match->TFTRACKs.size()<<"  All = "<<match->TFTRACKsAll.size()<<endl;
  if (debugTFTRACK) cout<<"--- TFTRACK ---- end"<<endl;
}


// ================================================================================================
void
SimMuL1::matchSimtrack2TFCANDs( MatchCSCMuL1 *match,
				edm::ESHandle< L1MuTriggerScales > &muScales,
				edm::ESHandle< L1MuTriggerPtScale > &muPtScale,
				const vector< L1MuRegionalCand > *l1TfCands)
{
  if (debugTFCAND) cout<<"--- TFCAND ---- begin"<<endl;
  for ( vector< L1MuRegionalCand >::const_iterator trk = l1TfCands->begin(); trk != l1TfCands->end(); trk++)
    {

      MatchCSCMuL1::TFCAND mtfcand(match);
      mtfcand.init( &*trk , ptLUT, muScales, muPtScale);

      //mtfcand.dr = deltaR( match->strk->momentum().eta(), normalizedPhi( match->strk->momentum().phi() ), mtfcand.eta, mtfcand.phi );
      mtfcand.dr = match->deltaRSmart( mtfcand.eta, mtfcand.phi );

      if (debugTFCAND) cout<< "----- L1MuRegionalCand with packed eta/phi "<<trk->eta_packed()<<"/"<<trk->phi_packed()<<"    eta="<<mtfcand.eta<<"  phi=" <<mtfcand.phi<<"  pt="<<mtfcand.pt<<"  dr="<<mtfcand.dr<<"  qu="<<trk->quality_packed()<<"  type="<<trk->type_idx()<<endl;
      //if (debugTFCAND) cout<< "----- ---- vaues: eta/phi "<<trk->etaValue()<<"/"<<trk->phiValue()<<"  ptValue "<<trk->ptValue()<<endl;
    
      mtfcand.tftrack = NULL;
      for (unsigned i=0; i< match->TFTRACKs.size(); i++)
	{
	  //if (debugTFCAND) cout<< "------- l1t packed eta phi: "<<l1t_eta_packed<<" "<<l1t_phi_packed<<endl;

	  if ( trk->phi_packed()  != (match->TFTRACKs[i]).phi_packed ||
	       trk->pt_packed()   != (match->TFTRACKs[i]).pt_packed ||
	       trk->eta_packed()  != (match->TFTRACKs[i]).eta_packed   ) continue;

	  mtfcand.tftrack = &(match->TFTRACKs[i]);
	  mtfcand.ids = match->TFTRACKs[i].ids;
	  if (debugTFCAND) cout<< "---------> matched to TFTRACKs["<<i<<"]"<<endl;
	  break;
	}
      if (mtfcand.dr < 0.2) match->TFCANDsAll.push_back(mtfcand);
      if (mtfcand.tftrack != NULL)  match->TFCANDs.push_back(mtfcand);
      //else if (debugTFCAND) cout<<"----- Warning!!! NO TFTRACKs found for this L1MuRegionalCand"<<endl;
    }
  if (debugTFCAND) cout<<"---- # of matched TFCANDs = "<<match->TFCANDs.size()<<"  All = "<<match->TFCANDsAll.size()<<endl;
  if (debugTFCAND) cout<<"--- TFCAND ---- end"<<endl;
}


// ================================================================================================
void
SimMuL1::matchSimtrack2GMTCANDs( MatchCSCMuL1 *match, 
				 edm::ESHandle< L1MuTriggerScales > &muScales,
				 edm::ESHandle< L1MuTriggerPtScale > &muPtScale,
				 const vector< L1MuGMTExtendedCand> &l1GmtCands,
				 const vector<L1MuRegionalCand> &l1GmtCSCCands,
				 const map<int, vector<L1MuRegionalCand> > &l1GmtCSCCandsInBXs)
{
  if (debugGMTCAND) cout<<"--- GMTREGCAND ---- begin"<<endl;

  double ptmatch = -1.;
  MatchCSCMuL1::GMTREGCAND grmatch;
  grmatch.l1reg = NULL;

  for ( vector<L1MuRegionalCand>::const_iterator trk = l1GmtCSCCands.begin(); trk != l1GmtCSCCands.end(); trk++)
    {

      MatchCSCMuL1::GMTREGCAND mcand;
      mcand.init( &*trk , muScales, muPtScale);

      //mcand.dr = deltaR( match->strk->momentum().eta(), normalizedPhi( match->strk->momentum().phi() ), mcand.eta, mcand.phi );
      mcand.dr = match->deltaRSmart( mcand.eta, mcand.phi );

      if (debugGMTCAND) cout<< "----- GMT:L1MuRegionalCand: packed eta/phi/pt "<<mcand.eta_packed<<"/"<<mcand.phi_packed<<"/"<<trk->pt_packed()<<"    eta="<<mcand.eta<<"  phi=" <<mcand.phi<<"  pt="<<mcand.pt<<"\t  dr="<<mcand.dr<<"  qu="<<trk->quality_packed()<<"  type="<<trk->type_idx()<<endl;
      //if (debugGMTCAND) cout<< "----- ---- values: eta/phi "<<trk->etaValue()<<"/"<<trk->phiValue()<<"  ptValue "<<trk->ptValue()<<endl;

      mcand.tfcand = NULL;
      for (unsigned i=0; i< match->TFCANDs.size(); i++)
	{
	  if ( trk->phi_packed()  != (match->TFCANDs[i]).l1cand->phi_packed() ||
	       trk->eta_packed()  != (match->TFCANDs[i]).l1cand->eta_packed()   ) continue;

	  mcand.tfcand = &(match->TFCANDs[i]);
	  mcand.ids = match->TFCANDs[i].ids;
	  if (debugGMTCAND) cout<< "---------> matched to TFCANDs["<<i<<"]"<<endl;
	  break;
	}
      if (mcand.tfcand != NULL)  match->GMTREGCANDs.push_back(mcand);

      mcand.tfcand = NULL;
      for (unsigned i=0; i< match->TFCANDsAll.size(); i++)
	{
	  if ( trk->phi_packed()  != (match->TFCANDsAll[i]).l1cand->phi_packed() ||
	       trk->eta_packed()  != (match->TFCANDsAll[i]).l1cand->eta_packed()   ) continue;

	  mcand.tfcand = &(match->TFCANDsAll[i]);
	  mcand.ids = match->TFCANDsAll[i].ids;
	  if (debugGMTCAND) cout<< "---------> matched to TFCANDsAll["<<i<<"]"<<endl;
	  break;
	}
      if (mcand.tfcand != NULL)  match->GMTREGCANDsAll.push_back(mcand);

      if (mcand.dr < 0.2 && mcand.pt > ptmatch) 
	{
	  ptmatch = mcand.pt;
	  grmatch = mcand;
	  if (debugGMTCAND) cout<< "---------> DR matched"<<endl;
	}
    }
  match->GMTREGCANDBest = grmatch;
  if (debugGMTCAND) cout<<"---- # of matched GMTREGCANDs = "<<match->GMTREGCANDs.size()<<"  All = "<<match->GMTREGCANDsAll.size()<<"  DR = "<<(match->GMTREGCANDBest.l1reg != NULL)<<endl;
  if (debugGMTCAND) cout<<"--- GMTREGCAND ---- end"<<endl;


  if (debugGMTCAND) cout<<"--- GMTCAND ---- begin"<<endl;

  ptmatch = -1.;
  MatchCSCMuL1::GMTCAND gmatch;
  gmatch.l1gmt = NULL;

  for( vector< L1MuGMTExtendedCand >::const_iterator muItr = l1GmtCands.begin() ; muItr != l1GmtCands.end() ; ++muItr)
    {
      if( muItr->empty() ) continue;

      MatchCSCMuL1::GMTCAND mcand;
      mcand.init( &*muItr , muScales, muPtScale);

      //mcand.dr = deltaR( match->strk->momentum().eta(), normalizedPhi( match->strk->momentum().phi() ), mcand.eta, mcand.phi );
      mcand.dr = match->deltaRSmart( mcand.eta, mcand.phi );

      if (debugGMTCAND) cout<< "----- L1MuGMTExtendedCand: packed eta/phi/pt "<<muItr->etaIndex()<<"/"<<muItr->phiIndex()<<"/"<<muItr->ptIndex()<<"    eta="<<mcand.eta<<"  phi=" <<mcand.phi<<"  pt="<<mcand.pt<<"\t  dr="<<mcand.dr<<"  qu="<<muItr->quality()<<"  bx="<<muItr->bx()<<"  q="<<muItr->charge()<<"("<<muItr->charge_valid()<<")  isRPC="<<muItr->isRPC()<<"  rank="<<muItr->rank()<<endl;

      // dataword to match to regional CSC candidates:
      unsigned int csc_dataword = 0;
      if (muItr->isFwd() && ( muItr->isMatchedCand() || !muItr->isRPC()))
	{
	  auto cands = l1GmtCSCCandsInBXs.find(muItr->bx());
	  if (cands != l1GmtCSCCandsInBXs.end())
	    {
	      auto& rcsc = (cands->second)[muItr->getDTCSCIndex()];
	      if (!rcsc.empty()) csc_dataword = rcsc.getDataWord();
	    }
	}

      mcand.regcand = NULL;
      if (csc_dataword) for (unsigned i=0; i< match->GMTREGCANDs.size(); i++)
			  {
			    if ( csc_dataword != match->GMTREGCANDs[i].l1reg->getDataWord() ) continue;

			    mcand.regcand = &(match->GMTREGCANDs[i]);
			    mcand.ids = match->GMTREGCANDs[i].ids;
			    if (debugGMTCAND) cout<< "---------> matched to GMTREGCANDs["<<i<<"]"<<endl;
			    break;
			  }
      if (mcand.regcand != NULL)  match->GMTCANDs.push_back(mcand);

      mcand.regcand = NULL;
      if (csc_dataword) for (unsigned i=0; i< match->GMTREGCANDsAll.size(); i++)
			  {
			    if ( csc_dataword != match->GMTREGCANDsAll[i].l1reg->getDataWord() ) continue;

			    mcand.regcand = &(match->GMTREGCANDsAll[i]);
			    mcand.ids = match->GMTREGCANDsAll[i].ids;
			    if (debugGMTCAND) cout<< "---------> matched to GMTREGCANDsAll["<<i<<"]"<<endl;
			    break;
			  }
      if (mcand.regcand != NULL)  match->GMTCANDsAll.push_back(mcand);

      if (mcand.dr < 0.2 && mcand.pt > ptmatch) 
	{
	  ptmatch = mcand.pt;
	  gmatch = mcand;
	  if (debugGMTCAND) cout<< "---------> DR matched"<<endl;
	}
    }
  match->GMTCANDBest = gmatch;

  if (debugGMTCAND) cout<<"---- # of matched GMTCANDs = "<<match->GMTCANDs.size()<<"  All = "<<match->GMTCANDsAll.size()<<"  DR = "<<(match->GMTCANDBest.l1gmt != NULL)<<endl;
  if (debugGMTCAND) cout<<"--- GMTCAND ---- end"<<endl;
}


// ================================================================================================
void
SimMuL1::matchSimtrack2L1EXTRAs( MatchCSCMuL1 *match,
				 const l1extra::L1MuonParticleCollection* l1Muons)
{
  if (debugL1EXTRA) cout<<"--- L1EXTRA ---- begin"<<endl;

  double ptmatch = -1.;
  MatchCSCMuL1::L1EXTRA xmatch;
  xmatch.l1extra = NULL;

  for ( l1extra::L1MuonParticleCollection::const_iterator mu = l1Muons->begin(); mu != l1Muons->end(); mu++)
    {
      if ( mu->bx() < minBX_ || mu->bx() > maxBX_ ) 
	{
	  if (debugGMTCAND) cout<<"discarding BX = "<< mu->bx() <<endl;
	  continue;
	}

      MatchCSCMuL1::L1EXTRA xtra;
      xtra.l1extra = &*mu;
  
      xtra.phi = normalizedPhi( mu->phi() );
      xtra.eta = mu->eta();
      xtra.pt = mu->pt();
      //xtra.dr = deltaR( match->strk->momentum().eta(), normalizedPhi( match->strk->momentum().phi() ), xtra.eta, xtra.phi );
      xtra.dr = match->deltaRSmart( xtra.eta, xtra.phi );

      if (debugL1EXTRA) cout<< "----- L1MuonParticle with  eta="<<xtra.eta<<"  phi=" <<xtra.phi<<"  pt="<<xtra.pt<<"  dr="<<xtra.dr<<"  bx="<<mu->bx()<<endl;
      //bx="<<mu->bx()

      xtra.gmtcand = NULL;
      for (unsigned i=0; i< match->GMTCANDs.size(); i++)
	{
	  if ( fabs( deltaPhi ( xtra.phi , (match->GMTCANDs[i]).phi ) ) > 0.001 ||
	       fabs( xtra.eta - (match->GMTCANDs[i]).eta ) > 0.001 ||
	       fabs( xtra.pt - (match->GMTCANDs[i]).pt ) > 0.001  ) continue;

	  xtra.gmtcand = &(match->GMTCANDs[i]);
	  //xtra.ids = match->GMTCANDs[i].ids;
	  if (debugL1EXTRA) cout<< "---------> matched to GMTCANDs["<<i<<"]"<<endl;
	  break;
	}
      if (xtra.gmtcand != NULL)  match->L1EXTRAs.push_back(xtra);

      xtra.gmtcand = NULL;
      for (unsigned i=0; i< match->GMTCANDsAll.size(); i++)
	{
	  if ( fabs( deltaPhi ( xtra.phi , (match->GMTCANDsAll[i]).phi ) ) > 0.001 ||
	       fabs( xtra.eta - (match->GMTCANDsAll[i]).eta ) > 0.001 ||
	       fabs( xtra.pt - (match->GMTCANDsAll[i]).pt ) > 0.001  ) continue;

	  xtra.gmtcand = &(match->GMTCANDsAll[i]);
	  //xtra.ids = match->GMTCANDs[i].ids;
	  if (debugL1EXTRA) cout<< "---------> matched to GMTCANDsAll["<<i<<"]"<<endl;
	  break;
	}
      if (xtra.gmtcand != NULL)  match->L1EXTRAsAll.push_back(xtra);

      if (xtra.dr < 0.2 && xtra.pt > ptmatch) 
	{
	  ptmatch = xtra.pt;
	  xmatch = xtra;
	  if (debugL1EXTRA) cout<< "---------> DR matched"<<endl;
	}
    }
  match->L1EXTRABest = xmatch;

  if (debugL1EXTRA) cout<<"---- # of matched GMTCANDs = "<<match->GMTCANDs.size()<<"  All = "<<match->GMTCANDsAll.size()<<"  DR = "<<(match->L1EXTRABest.l1extra != NULL)<<endl;
  if (debugL1EXTRA) cout<<"--- L1EXTRA ---- end"<<endl;
}

// ================================================================================================
vector<unsigned> 
SimMuL1::fillSimTrackFamilyIds(unsigned  id,
			       const SimTrackContainer & simTracks, const SimVertexContainer & simVertices)
{
  int fdebug = 0;
  vector<unsigned> result;
  result.push_back(id);

  if (doStrictSimHitToTrackMatch_) return result;

  if (fdebug)  cout<<"--- fillSimTrackFamilyIds:  id "<<id<<endl;
  for (SimTrackContainer::const_iterator istrk = simTracks.begin(); istrk != simTracks.end(); ++istrk)
    {
      if (fdebug)  cout<<"  --- looking at Id "<<istrk->trackId()<<endl;
      SimTrack lastTr = *istrk;
      bool ischild = 0;
      while (1)
	{
	  if (fdebug)  cout<<"    --- while Id "<<lastTr.trackId()<<endl;
	  if ( lastTr.noVertex() ) break;
	  if (fdebug)  cout<<"      --- vertex "<<lastTr.vertIndex()<<endl;
	  if ( simVertices[lastTr.vertIndex()].noParent() ) break;
	  unsigned parentId = simVertices[lastTr.vertIndex()].parentIndex();
	  if (fdebug)  cout<<"      --- parentId "<<parentId<<endl;
	  if ( parentId == id ) 
	    {
	      if (fdebug)  cout<<"      --- ischild "<<endl;
	      ischild = 1; 
	      break; 
	    }
	  map<unsigned, unsigned >::iterator association = trkId2Index.find( parentId );
	  if ( association == trkId2Index.end() ) 
	    { 
	      if (fdebug)  cout<<"      --- not in trkId2Index "<<endl; 
	      break; 
	    }
	  if (fdebug)  cout<<"      --- lastTrk index "<<association->second<<endl;
	  lastTr = simTracks[ association->second ];
	}
      if (ischild) 
	{
	  result.push_back(istrk->trackId());
	  if (fdebug)  cout<<"  --- child pushed "<<endl;
	}
    }

  if (fdebug)  cout<<"  --- family size = "<<result.size()<<endl;
  return result;
}




// ================================================================================================
vector<PSimHit> 
SimMuL1::hitsFromSimTrack(vector<unsigned> ids, SimHitAnalysis::PSimHitMap &hitMap)
{
  int fdebug = 0;
  
  vector<PSimHit> result;

  if (fdebug)  cout<<"--- hitsFromSimTrack vector size "<<ids.size()<<endl;

  for (size_t id = 0; id < ids.size(); id++)
    {
      vector<PSimHit> resultd = hitsFromSimTrack(ids[id], hitMap);
      result.insert(result.end(), resultd.begin(), resultd.end());
      if (fdebug)  cout<<"  --- n "<<id<<" id "<<ids[id]<<" size "<<resultd.size()<<endl;
    }
  return result;
}



// ================================================================================================
vector<PSimHit> 
SimMuL1::hitsFromSimTrack(unsigned id, SimHitAnalysis::PSimHitMap &hitMap)
{
  int fdebug = 0;
  
  vector<PSimHit> result;
  vector<int> detIds = hitMap.detsWithHits();

  if (fdebug)  cout<<"---- hitsFromSimTrack id "<<id<<endl;
  
  for (size_t di = 0; di < detIds.size(); di++)
    {
      vector<PSimHit> resultd = hitsFromSimTrack(id, detIds[di], hitMap);
      result.insert(result.end(), resultd.begin(), resultd.end());
      if (fdebug)  cout<<"  ---- det "<<detIds[di]<<" size "<<resultd.size()<<endl;
    }
  return result;
}


// ================================================================================================
vector<PSimHit> 
SimMuL1::hitsFromSimTrack(unsigned id, int detId, SimHitAnalysis::PSimHitMap &hitMap)
{
  int fdebug = 0;

  vector<PSimHit> result;

  CSCDetId chId(detId);
  if ( chId.station() == 1 && chId.ring() == 4 && !doME1a_) return result;

  PSimHitContainer hits = hitMap.hits(detId);
  
  for(size_t h = 0; h< hits.size(); h++) if(hits[h].trackId() == id)
					   {
					     result.push_back(hits[h]);

					     if (fdebug)  cout<<"   --- "<<detId<<" -> "<< hits[h] <<" "<<hits[h].exitPoint()<<" "<<hits[h].momentumAtEntry()<<" "<<hits[h].energyLoss()<<" "<<hits[h].particleType()<<" "<<hits[h].trackId()<<endl;
					   }

  return result;
}


// ================================================================================================
int 
SimMuL1::particleType(int simTrack) 
{
  int result = 0;
  vector<PSimHit> hits = hitsFromSimTrack(simTrack,theCSCSimHitMap);
  //if(hits.empty())  hits = dtHitsFromSimTrack(simTrack);
  //if(hits.empty())  hits = rpcHitsFromSimTrack(simTrack);
  if(!hits.empty())  result = hits[0].particleType();
  return result;
}


// ================================================================================================
bool 
SimMuL1::compareSimHits(PSimHit &sh1, PSimHit &sh2)
{
  int fdebug = 0;

  if (fdebug && sh1.detUnitId() == sh2.detUnitId())  {
    cout<<" compare hits in "<<sh1.detUnitId()<<": "<<endl;
    cout<<"   "<<sh1<<" "<<sh1.exitPoint()<<" "<<sh1.momentumAtEntry()<<" "<<sh1.energyLoss()<<" "<<sh1.particleType()<<" "<<sh1.trackId()<<" |"<<sh1.entryPoint().mag()<<" "<<sh1.exitPoint().mag()<<" "<<sh1.momentumAtEntry().mag()<<endl;
    cout<<"   "<<sh2<<" "<<sh2.exitPoint()<<" "<<sh2.momentumAtEntry()<<" "<<sh2.energyLoss()<<" "<<sh2.particleType()<<" "<<sh2.trackId()<<" |"<<sh2.entryPoint().mag()<<" "<<sh2.exitPoint().mag()<<" "<<sh2.momentumAtEntry().mag()<<endl;

    if(sh1.tof()!=sh2.tof()) cout<<" !!tof "<<sh1.tof()-sh2.tof()<<endl;
    if(sh1.entryPoint().mag()!=sh2.entryPoint().mag()) cout<<"  !!ent "<< sh1.entryPoint().mag() - sh2.entryPoint().mag()<<endl;
    if(sh1.exitPoint().mag() != sh2.exitPoint().mag()) cout<<"  !!exi "<< sh1.exitPoint().mag() - sh2.exitPoint().mag()<<endl;
    if(sh1.momentumAtEntry().mag() != sh2.momentumAtEntry().mag()) cout<<"  !!mom "<< sh1.momentumAtEntry().mag() - sh2.momentumAtEntry().mag()<<endl;
    if(sh1.energyLoss() != sh2.energyLoss()) cout<<"  !!los "<<sh1.energyLoss() - sh2.energyLoss() <<endl;
  }
  
  if (
      sh1.detUnitId() == sh2.detUnitId() &&
      sh1.trackId() == sh2.trackId() &&
      sh1.particleType() == sh2.particleType() &&
      sh1.entryPoint().mag() == sh2.entryPoint().mag() &&
      sh1.exitPoint().mag() == sh2.exitPoint().mag() &&
      fabs(sh1. momentumAtEntry().mag() - sh2. momentumAtEntry().mag() ) < 0.0001 &&
      sh1.tof() == sh2.tof() &&
      sh1.energyLoss() == sh2.energyLoss()
      ) return 1;
  
  return 0;
}




// ================================================================================================
unsigned
SimMuL1::matchCSCAnodeHits(const vector<CSCAnodeLayerInfo>& allLayerInfo, 
                           vector<PSimHit> &matchedHit) 
{
  // Match Anode hits in a chamber to SimHits

  // It first tries to look for the SimHit in the key layer.  If it is
  // unsuccessful, it loops over all layers and looks for an associated
  // hits in any one of the layers.  

  //  int fdebug = 0;

  int nhits=0;
  matchedHit.clear();
  
  vector<CSCAnodeLayerInfo>::const_iterator pli;
  for (pli = allLayerInfo.begin(); pli != allLayerInfo.end(); pli++) 
    {
      // For ALCT search, the key layer is the 3rd one, counting from 1.
      if (pli->getId().layer() == CSCConstants::KEY_ALCT_LAYER) 
	{
	  vector<PSimHit> thisLayerHits = pli->getSimHits();
	  if (thisLayerHits.size() > 0) 
	    {
	      // There can be only one RecDigi (and therefore only one SimHit) in a key layer.
	      if (thisLayerHits.size() != 1) 
		{
		  cout<< "+++ Warning in matchCSCAnodeHits: " << thisLayerHits.size()
		      << " SimHits in key layer " << CSCConstants::KEY_ALCT_LAYER
		      << "! +++ \n";
		  for (unsigned i = 0; i < thisLayerHits.size(); i++) 
		    cout<<"      SimHit # " << i <<": "<< thisLayerHits[i] << "\n";
		}
	      matchedHit.push_back(thisLayerHits[0]);
	      nhits++;
	      break;
	    }
	}
    }

  for (pli = allLayerInfo.begin(); pli != allLayerInfo.end(); pli++) 
    {
      if (pli->getId().layer() == CSCConstants::KEY_ALCT_LAYER)  continue;
    
      // if there is any occurrence of simHit size greater that zero, use this.
      if ((pli->getRecDigis()).size() > 0 && (pli->getSimHits()).size() > 0) 
	{
	  vector<PSimHit> thisLayerHits = pli->getSimHits();
	  // There can be several RecDigis and several SimHits in a nonkey layer.
	  //if (thisLayerHits.size() != 1) 
	  //{
	  //  cout<< "+++ Warning in matchCSCAnodeHits: " << thisLayerHits.size()
	  //      << " SimHits in layer " << pli->getId().layer() <<" detID "<<pli->getId().rawId()
	  //      << "! +++ \n";
	  //  for (unsigned i = 0; i < thisLayerHits.size(); i++)
	  //    cout<<"   "<<thisLayerHits[i]<<" "<<thisLayerHits[i].exitPoint()<<"  "<<thisLayerHits[i].momentumAtEntry()
	  //        <<" "<<thisLayerHits[i].energyLoss()<<" "<<thisLayerHits[i].particleType()<<" "<<thisLayerHits[i].trackId()<<endl;
	  //}
	  matchedHit.insert(matchedHit.end(), thisLayerHits.begin(), thisLayerHits.end());
	  nhits += thisLayerHits.size();
	}
    }
  
  return nhits;
}



// ================================================================================================
unsigned
SimMuL1::matchCSCCathodeHits(const vector<CSCCathodeLayerInfo>& allLayerInfo, 
			     vector<PSimHit> &matchedHit) 
{
  // It first tries to look for the SimHit in the key layer.  If it is
  // unsuccessful, it loops over all layers and looks for an associated
  // hits in any one of the layers.  

  //  int fdebug = 0;

  //  static const int key_layer = 4; //CSCConstants::KEY_CLCT_LAYER
  static const int key_layer = CSCConstants::KEY_CLCT_LAYER;
 
  int nhits=0;
  matchedHit.clear();
  
  vector<CSCCathodeLayerInfo>::const_iterator pli;
  for (pli = allLayerInfo.begin(); pli != allLayerInfo.end(); pli++) 
    {
      // For ALCT search, the key layer is the 3rd one, counting from 1.
      if (pli->getId().layer() == key_layer) 
	{
	  vector<PSimHit> thisLayerHits = pli->getSimHits();
	  if (thisLayerHits.size() > 0) 
	    {
	      // There can be only one RecDigi (and therefore only one SimHit) in a key layer.
	      if (thisLayerHits.size() != 1) 
		{
		  cout<< "+++ Warning in matchCSCCathodeHits: " << thisLayerHits.size()
		      << " SimHits in key layer " << key_layer
		      << "! +++ \n";
		  for (unsigned i = 0; i < thisLayerHits.size(); i++) 
		    cout<<"      SimHit # " << i <<": "<< thisLayerHits[i] << "\n";
		}
	      matchedHit.push_back(thisLayerHits[0]);
	      nhits++;
	      break;
	    }
	}
    }

  for (pli = allLayerInfo.begin(); pli != allLayerInfo.end(); pli++) 
    {
      if (pli->getId().layer() == key_layer)  continue;
    
      // if there is any occurrence of simHit size greater that zero, use this.
      if ((pli->getRecDigis()).size() > 0 && (pli->getSimHits()).size() > 0) 
	{
	  vector<PSimHit> thisLayerHits = pli->getSimHits();
	  // There can be several RecDigis and several SimHits in a nonkey layer.
	  //if (thisLayerHits.size() != 1) 
	  //{
	  //  cout<< "+++ Warning in matchCSCCathodeHits: " << thisLayerHits.size()
	  //      << " SimHits in layer " << pli->getId().layer() <<" detID "<<pli->getId().rawId()
	  //      << "! +++ \n";
	  //  for (unsigned i = 0; i < thisLayerHits.size(); i++)
	  //    cout<<"   "<<thisLayerHits[i]<<" "<<thisLayerHits[i].exitPoint()<<"  "<<thisLayerHits[i].momentumAtEntry()
	  //        <<" "<<thisLayerHits[i].energyLoss()<<" "<<thisLayerHits[i].particleType()<<" "<<thisLayerHits[i].trackId()<<endl;
	  //}
	  matchedHit.insert(matchedHit.end(), thisLayerHits.begin(), thisLayerHits.end());
	  nhits += thisLayerHits.size();
	}
    }
  
  return nhits;
}



// ================================================================================================
int 
SimMuL1::calculate2DStubsDeltas(MatchCSCMuL1 *match, MatchCSCMuL1::ALCT &alct)
{
  //** fit muon's hits to a 2D linear stub in a chamber :
  //   wires:   work in 2D plane going through z axis :
  //     z becomes a new x axis, and new y is perpendicular to it
  //    (using SimTrack's position point as well when there is <= 2 mu's hits in chamber)

  int fdebug = 0;

  alct.deltaPhi = M_PI;
  alct.deltaY = 9999.;
  
  CSCDetId keyID(alct.id.rawId()+CSCConstants::KEY_ALCT_LAYER);
  const CSCLayer* csclayer = cscGeometry->layer(keyID);
  //int hitWireG = csclayer->geometry()->wireGroup(csclayer->geometry()->nearestWire(cLP));
  int hitWireG = match->wireGroupAndStripInChamber(alct.id.rawId()).first;
  if (hitWireG<0) return 1;

  alct.deltaWire = hitWireG - alct.trgdigi->getKeyWG() - 1;

  alct.mcWG = hitWireG;

  GlobalPoint gpcwg = csclayer->centerOfWireGroup( alct.trgdigi->getKeyWG()+1 );
  math::XYZVectorD vcwg( gpcwg.x(), gpcwg.y(), gpcwg.z() );
  alct.eta = vcwg.eta();

  if (fdebug) cout<<"    hitWireG = "<<hitWireG<<"    alct.KeyWG = "<<alct.trgdigi->getKeyWG()<<"    deltaWire = "<<alct.deltaWire<<endl;

  return 0;
}


// ================================================================================================
int 
SimMuL1::calculate2DStubsDeltas(MatchCSCMuL1 *match, MatchCSCMuL1::CLCT &clct)
{
  //** fit muon's hits to a 2D linear stub in a chamber :
  //   stripes:  work in 2D cylindrical surface :
  //     z becomes a new x axis, and phi is a new y axis
  //     (using SimTrack stub with no-bend in phi if there is 1 mu's hit in chamber)
  
  int fdebug = 0;
  
  clct.deltaPhi = M_PI;
  clct.deltaY = 9999.;

  //double a = 0., b = 0.;
  //vector<PSimHit> hits = match->chamberHits(clct.id.rawId());

  bool defaultGangedME1a = (defaultME1a && clct.id.station()==1 && (clct.id.ring()==1||clct.id.ring()==4) && clct.trgdigi->getKeyStrip() > 127);
  //if ( defaultGangedME1a ) {
  //  CSCDetId hid(clct.id.endcap(),1,4,clct.id.chamber(),0);
  //  hits = match->chamberHits(hid.rawId());
  //}


  //  delta strip calculation

  // find LocalPoint of the highest energy muon simhit in key layer
  // if no hit in key layer, take the highest energy muon simhit local position

  CSCDetId keyID(clct.id.rawId()+CSCConstants::KEY_CLCT_LAYER);
  const CSCLayer* csclayer = cscGeometry->layer(keyID);
  //int hitStrip = csclayer->geometry()->nearestStrip(cLP);
  int hitStrip = match->wireGroupAndStripInChamber(clct.id.rawId()).second;
  if (hitStrip<0) return 1;

  int stubStrip =  clct.trgdigi->getKeyStrip()/2 + 1;
  int stubStripGeo = stubStrip;

  clct.deltaStrip = hitStrip - stubStrip;

  // shift strip numbers to geometry ME1a space for defaultGangedME1a
  if ( defaultGangedME1a ) {
    stubStrip = (clct.trgdigi->getKeyStrip()-128)/2 + 1;
    stubStripGeo = stubStrip;
    clct.deltaStrip = hitStrip - stubStrip;
  }
  
  clct.mcStrip = hitStrip;
  
  
  // if it's upgraded TMB's ganged ME1a or defaultGangedME1a, find the best delta strip
  if ( (gangedME1a && !defaultME1a && clct.id.station()==1 && clct.id.ring()==4) || defaultGangedME1a) {
    // consider gang#2
    int ds = hitStrip - stubStrip - 16;
    if (abs(ds) < abs(clct.deltaStrip)) {
      clct.deltaStrip = ds;
      stubStripGeo = stubStrip + 16;
    }
    // consider gang#3
    ds = hitStrip - stubStrip - 32;
    if (abs(ds) < abs(clct.deltaStrip)) {
      clct.deltaStrip = ds;
      stubStripGeo = stubStrip + 32;
    }
  }

  GlobalPoint gpcs = csclayer->centerOfStrip( stubStripGeo );
  math::XYZVectorD vcs( gpcs.x(), gpcs.y(), gpcs.z() );
  clct.phi = vcs.phi();

  if (fdebug) cout<<"    hitStrip = "<<hitStrip<<"    alct.KeyStrip = "<<clct.trgdigi->getKeyStrip()<<"    deltaStrip = "<<clct.deltaStrip<<endl;

  return 0;
}



// ================================================================================================
math::XYZVectorD
SimMuL1::cscSimHitGlobalPosition( PSimHit &h )
{
  CSCDetId layerId(h.detUnitId());
  const CSCLayer* csclayer = cscGeometry->layer(layerId);
  //const CSCLayerGeometry* layerGeom = csclayer->geometry();
  GlobalPoint gp = csclayer->toGlobal(h.localPosition());
  math::XYZVectorD vgp( gp.x(), gp.y(), gp.z() );
  return vgp;
}

math::XYZVectorD
SimMuL1::cscSimHitGlobalPositionX0( PSimHit &h )
{
  CSCDetId layerId(h.detUnitId());
  const CSCLayer* csclayer = cscGeometry->layer(layerId);
  //const CSCLayerGeometry* layerGeom = csclayer->geometry();
  const Local3DPoint lp (0.,h.localPosition().y(),h.localPosition().z());
  GlobalPoint gp = csclayer->toGlobal(lp);
  math::XYZVectorD vgp( gp.x(), gp.y(), gp.z() );
  return vgp;
}


// ================================================================================================
//
TrajectoryStateOnSurface
SimMuL1::propagateSimTrackToZ(const SimTrack *track, const SimVertex *vtx, double z)
{
  Plane::PositionType pos(0, 0, z);
  Plane::RotationType rot;
  Plane::PlanePointer myPlane = Plane::build(pos, rot);

  GlobalPoint  innerPoint(vtx->position().x(),  vtx->position().y(),  vtx->position().z());
  GlobalVector innerVec  (track->momentum().x(),  track->momentum().y(),  track->momentum().z());

  FreeTrajectoryState stateStart(innerPoint, innerVec, track->charge(), &*theBField);

  TrajectoryStateOnSurface stateProp = propagatorAlong->propagate(stateStart, *myPlane);
  if (!stateProp.isValid())
    stateProp = propagatorOpposite->propagate(stateStart, *myPlane);

  return stateProp;
}


// ================================================================================================
//
TrajectoryStateOnSurface
SimMuL1::propagateSimTrackToDT(const SimTrack *track, const SimVertex *vtx)
{
  const DetLayer * dt2 = muonGeometry->allDTLayers()[1];
  const BoundCylinder *barrelCylinder = dynamic_cast<const BoundCylinder *>(&dt2->surface());

  GlobalPoint  innerPoint(vtx->position().x(),  vtx->position().y(),  vtx->position().z());
  GlobalVector innerVec  (track->momentum().x(),  track->momentum().y(),  track->momentum().z());
  
  FreeTrajectoryState stateStart(innerPoint, innerVec, track->charge(), &*theBField);

  TrajectoryStateOnSurface stateProp = propagatorAlong->propagate(stateStart, *barrelCylinder);
  if (!stateProp.isValid()) 
    stateProp = propagatorOpposite->propagate(stateStart, *barrelCylinder);

  return stateProp;
}

// ================================================================================================
// 4-bit LCT quality number. Copied from 
// http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.cc?view=markup
unsigned int 
SimMuL1::findQuality(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT)
{
  unsigned int quality = 0;

  // 2008 definition.
  if (!(aLCT.isValid()) || !(cLCT.isValid())) {
    if (aLCT.isValid() && !(cLCT.isValid()))  quality = 1; // no CLCT
    else if (!(aLCT.isValid()) && cLCT.isValid()) quality = 2; // no ALCT
    else quality = 0; // both absent; should never happen.
  }
  else {
    int pattern = cLCT.getPattern();
    if (pattern == 1) quality = 3; // layer-trigger in CLCT
    else {
      // CLCT quality is the number of layers hit minus 3.
      // CLCT quality is the number of layers hit.
      bool a4 = (aLCT.getQuality() >= 1);
      bool c4 = (cLCT.getQuality() >= 4);
      //      quality = 4; "reserved for low-quality muons in future"
      if      (!a4 && !c4) quality = 5; // marginal anode and cathode
      else if ( a4 && !c4) quality = 6; // HQ anode, but marginal cathode
      else if (!a4 &&  c4) quality = 7; // HQ cathode, but marginal anode
      else if ( a4 &&  c4) {
        if (aLCT.getAccelerator()) quality = 8; // HQ muon, but accel ALCT
        else {
          // quality =  9; "reserved for HQ muons with future patterns
          // quality = 10; "reserved for HQ muons with future patterns
          if (pattern == 2 || pattern == 3) quality = 11;
          else if (pattern == 4 || pattern == 5) quality = 12;
          else if (pattern == 6 || pattern == 7) quality = 13;
          else if (pattern == 8 || pattern == 9) quality = 14;
          else if (pattern == 10) quality = 15;
          else cout<< "+++ findQuality: Unexpected CLCT pattern id = "
		   << pattern << "+++"<<endl;
        }
      }
    }
  }
  return quality;
}

// ================================================================================================
// Visualization of wire group digis
void SimMuL1::dumpWireDigis(CSCDetId &id, const CSCWireDigiCollection* wiredc)
{
  // foolproof 1st layer
  int chamberId = id.chamberId().rawId();
  CSCDetId layer1(chamberId+1);

  int numWireGroups = cscGeometry->layer(layer1)->geometry()->numberOfWireGroups();  
  cout<<"Wire digi dump in "<<id<<" nWiregroups " << numWireGroups<<endl;
  
  vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES];
  static int fifo_tbins = 16;
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    const CSCDetId layerId(chamberId + i_layer + 1);
    const CSCWireDigiCollection::Range rwired = wiredc->get(layerId);
    for (CSCWireDigiCollection::const_iterator digiIt = rwired.first;
         digiIt != rwired.second; ++digiIt)
      {
	int i_wire  = digiIt->getWireGroup()-1;
	vector<int> bx_times = digiIt->getTimeBinsOn();
	for (unsigned int i = 0; i < bx_times.size(); i++) {
	  // Comparisons with data show that time bin 0 needs to be skipped.
	  if (bx_times[i] > 0 && bx_times[i] < fifo_tbins) {
	    // For the time being, if there is more than one hit on the same wire,
	    // pick the one which occurred earlier.
	    wire[i_layer][i_wire].push_back(bx_times[i]);
	    break;
	  }
	}
      }
    if (getCSCType(id)==3 && doME1a_){ // add ME1/a
      const CSCDetId layerId_me1a(id.endcap(), id.station(), 4, id.chamber(), i_layer+1);
      const CSCWireDigiCollection::Range rwired1a = wiredc->get(layerId_me1a);
      for (CSCWireDigiCollection::const_iterator digiIt = rwired1a.first;
	   digiIt != rwired1a.second; ++digiIt)
	{
	  int i_wire  = digiIt->getWireGroup()-1;
	  vector<int> bx_times = digiIt->getTimeBinsOn();
	  for (unsigned int i = 0; i < bx_times.size(); i++) {
	    if (bx_times[i] > 0 && bx_times[i] < fifo_tbins) {
	      wire[i_layer][i_wire].push_back(bx_times[i]);
	      break;
	    }
	  }
	}
    }
  }
  
  std::ostringstream strstrm;
  for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
    if (i_wire%10 == 0) {
      if (i_wire < 100) strstrm << i_wire/10;
      else              strstrm << (i_wire-100)/10;
    }
    else                strstrm << " ";
  }
  strstrm << "\n";
  for (int i_wire = 0; i_wire < numWireGroups; i_wire++)
    strstrm << i_wire%10;
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    strstrm << "\n";
    for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
      if (wire[i_layer][i_wire].size() > 0) {
	std::vector<int> bx_times = wire[i_layer][i_wire];
	strstrm << std::hex << bx_times[0] << std::dec;
      }
      else strstrm << ".";
    }
  }
  cout<< strstrm.str() <<endl;
}


// ================================================================================================
// Returns chamber type (0-9) according to the station and ring number
int 
SimMuL1::getCSCType(CSCDetId &id) 
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

int
SimMuL1::isME11(int t)
{
  if (t==0 || t==3) return CSC_TYPES;
  return 0;
}

// Returns chamber type (0-9) according to CSCChamberSpecs type
// 1..10 -> 1/a, 1/b, 1/2, 1/3, 2/1...
int
SimMuL1::getCSCSpecsType(CSCDetId &id)
{
  return cscGeometry->chamber(id)->specs()->chamberType();
}


// ================================================================================================
int
SimMuL1::cscTriggerSubsector(CSCDetId &id)
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
// From Ingo:
// calculates the weight of the event to reproduce a min bias
//spectrum, from G. Wrochna's note CMSS 1997/096
double SimMuL1::rateWeight(double simPT)
{
  double prompt = 1308400. * exp( -0.5 * pow( (log10(simPT)+0.725) / 0.4333, 2 ) );
  // Grzegor's formula is just for hadrons.  Need tomake my own from his plot
  // after the calorimeters
  //double decay = 1.1429e10 * pow( pow(simPT, 1.306) + 0.8251, -3.781);
  // make it 3*10^7 at 1, and 10^5 at 4.  So y = 10^7 pT^(-4.19)
  // But that plot is integrated rate.  Need to take a derivative.
  // This is the first time I've used calculus ever, outside a class.

  double decay = 3e7 * 4.19 * pow(simPT, -5.19);
  return prompt + decay;
}

// ================================================================================================
bool SimMuL1::isME42EtaRegion(float eta)
{
  if (fabs(eta)>=1.2499 && fabs(eta)<=1.8) return true;
  else return false;
}

bool SimMuL1::isME42RPCEtaRegion(float eta)
{
  if (fabs(eta)>=1.2499 && fabs(eta)<=1.6) return true;
  else return false;
}


// ================================================================================================

void SimMuL1::setupTFModeHisto(TH1D* h)
{
  if (h==0) return;
  if (h->GetXaxis()->GetNbins()<16) {
    cout<<"TF mode histogram should have 16 bins, nbins="<<h->GetXaxis()->GetNbins()<<endl;
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

pair<float, float> SimMuL1::intersectionEtaPhi(CSCDetId id, int wg, int hs)
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
  
  return make_pair(csc_gp.eta(), csc_gp.phi());
}

// ================================================================================================

csctf::TrackStub SimMuL1::buildTrackStub(const CSCCorrelatedLCTDigi &d, CSCDetId id)
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

// ================================================================================================

bool SimMuL1::isGEMDPhiGood(double dphi, double tfpt, int is_odd)
{
  // ignore the default/don't-care value of -99
  if (dphi < 9.) return true;

  // the no-matching-gem case value of +99
  if (dphi > 9.) return false;

  // find the first gemPTs_ element that is LARGER then tfpt
  auto ub = upper_bound(gemPTs_.begin(), gemPTs_.end(), tfpt);
  // adjust to get the largest gemPTs_ element that is smaller or equal to tfpt
  if (ub != gemPTs_.begin()) --ub;

  // the index in the vector
  size_t n = ub - gemPTs_.begin();

  return is_odd ? (dphi <= gemDPhisOdd_[n]) : (dphi <= gemDPhisEven_[n]);
}

// ================================================================================================
// ------------ method called once each job just before starting event loop  ------------
void 
SimMuL1::beginJob() {}

// ================================================================================================
// ------------ method called once each job just after ending the event loop  ------------
void 
SimMuL1::endJob() {}


//define this as a plug-in
DEFINE_FWK_MODULE(SimMuL1);
