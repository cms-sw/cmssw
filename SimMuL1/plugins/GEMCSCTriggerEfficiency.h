#ifndef SimMuL1_GEMCSCTriggerEfficiency_h
#define SimMuL1_GEMCSCTriggerEfficiency_h

// system include files
#include <memory>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1.h"
#include "TH2.h"
#include "TTree.h"

//#include "TLorentzVector.h"
//#include "DataFormats/Math/interface/LorentzVector.h"
//#include <CLHEP/Vector/LorentzVector.h>

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
//#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


#include "DataFormats/MuonDetId/interface/CSCDetId.h"
//#include <DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h>

#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"

#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"


//#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "GEMCode/SimMuL1/interface/PSimHitMap.h"
#include "GEMCode/SimMuL1/interface/MuGeometryHelpers.h"

#include "GEMCode/SimMuL1/interface/MatchCSCMuL1.h"

class DTGeometry;
class CSCGeometry;
class RPCGeometry;
class GEMGeometry;
class MuonDetLayerGeometry;

class CSCTFSectorProcessor;
class CSCSectorReceiverLUT;
class CSCTFDTReceiver;

class CSCStripConditions;

class GEMCSCTriggerEfficiency : public edm::EDAnalyzer 
{
public:

  explicit GEMCSCTriggerEfficiency(const edm::ParameterSet&);
  ~GEMCSCTriggerEfficiency();

  enum trig_cscs {MAX_STATIONS = 4, CSC_TYPES = 10};
  //Various useful constants
  static const std::string csc_type[CSC_TYPES+1];
  static const std::string csc_type_[CSC_TYPES+1];
  static const std::string csc_type_a[CSC_TYPES+2];
  static const std::string csc_type_a_[CSC_TYPES+2];
  static const int NCHAMBERS[CSC_TYPES];
  static const int MAX_WG[CSC_TYPES];
  static const int MAX_HS[CSC_TYPES];
//  static const float ptscale[33];
  static const int pbend[CSCConstants::NUM_CLCT_PATTERNS];

  enum pt_thresh {N_PT_THRESHOLDS = 6};
  static const double PT_THRESHOLDS[N_PT_THRESHOLDS];
  static const double PT_THRESHOLDS_FOR_ETA[N_PT_THRESHOLDS];


  int getCSCType(CSCDetId &id);
  int isME11(int t);
  int getCSCSpecsType(CSCDetId &id);
  int cscTriggerSubsector(CSCDetId &id);

  std::vector<unsigned> fillSimTrackFamilyIds(unsigned  index,
             const edm::SimTrackContainer & simTracks, const edm::SimVertexContainer & simVertices);

  std::vector<PSimHit> hitsFromSimTrack(std::vector<unsigned> ids, SimHitAnalysis::PSimHitMap &hitMap);
  std::vector<PSimHit> hitsFromSimTrack(unsigned  id, SimHitAnalysis::PSimHitMap &hitMap);
  std::vector<PSimHit> hitsFromSimTrack(unsigned  id, int detId, SimHitAnalysis::PSimHitMap &hitMap);

  int particleType(int simTrack);
    
  bool compareSimHits(PSimHit &sh1, PSimHit &sh2);

  void propagateToCSCStations(MatchCSCMuL1 *match);

  void matchSimTrack2SimHits( MatchCSCMuL1 *match,
             const edm::SimTrackContainer & simTracks,
             const edm::SimVertexContainer & simVertices,
             const edm::PSimHitContainer* allCSCSimHits);

  unsigned matchCSCAnodeHits(
             const std::vector<CSCAnodeLayerInfo>& allLayerInfo, 
             std::vector<PSimHit> &matchedHit) ;

  unsigned matchCSCCathodeHits(
             const std::vector<CSCCathodeLayerInfo>& allLayerInfo, 
             std::vector<PSimHit> &matchedHit) ;

  void matchSimTrack2ALCTs( MatchCSCMuL1 *match, 
             const edm::PSimHitContainer* allCSCSimHits, 
             const CSCALCTDigiCollection* alcts, 
             const CSCWireDigiCollection* wiredc );

  void matchSimTrack2CLCTs( MatchCSCMuL1 *match, 
             const edm::PSimHitContainer* allCSCSimHits, 
             const CSCCLCTDigiCollection *clcts, 
             const CSCComparatorDigiCollection* compdc );

  void  matchSimTrack2LCTs( MatchCSCMuL1 *match, 
             const CSCCorrelatedLCTDigiCollection* lcts );

  void  matchSimTrack2MPLCTs( MatchCSCMuL1 *match, 
             const CSCCorrelatedLCTDigiCollection* mplcts );

  void  matchSimtrack2TFTRACKs( MatchCSCMuL1 *match,
             edm::ESHandle< L1MuTriggerScales > &muScales,
             edm::ESHandle< L1MuTriggerPtScale > &muPtScale,
             const L1CSCTrackCollection* l1Tracks);

  void  matchSimtrack2TFCANDs( MatchCSCMuL1 *match,
             edm::ESHandle< L1MuTriggerScales > &muScales,
             edm::ESHandle< L1MuTriggerPtScale > &muPtScale,
             const std::vector< L1MuRegionalCand > *l1TfCands);

  void  matchSimtrack2GMTCANDs( MatchCSCMuL1 *match,
             edm::ESHandle< L1MuTriggerScales > &muScales,
             edm::ESHandle< L1MuTriggerPtScale > &muPtScale,
             const std::vector< L1MuGMTExtendedCand> &l1GmtCands,
             const std::vector<L1MuRegionalCand> &l1GmtCSCCands,
             const std::map<int, std::vector<L1MuRegionalCand> > &l1GmtCSCCandsInBXs);


  // fit muon's hits to a 2D linear stub in a chamber :
  //   wires:   work in 2D plane going through z axis :
  //     z becomes a new x axis, and new y is perpendicular to it
  //     (using SimTrack's position point as well when there is <= 2 mu's hits in chamber)
  // fit a 2D stub from SimHits matched to a digi
  // set deltas between SimTrack's and Digi's 2D stubs in (Z,R) -> (x,y) plane
  int calculate2DStubsDeltas(MatchCSCMuL1 *match, MatchCSCMuL1::ALCT &alct);

  // fit muon's hits to a 2D linear stub in a chamber :
  //   stripes:  work in 2D cylindrical surface :
  //     z becomes a new x axis, and phi is a new y axis
  //     (using SimTrack stub with no-bend in phi if there is 1 mu's hit in chamber)
  // fit a 2D stub from SimHits matched to a digi
  // set deltas between SimTrack's and Digi's 2D stubs in (Z,Phi) -> (x,y) plane
  int calculate2DStubsDeltas(MatchCSCMuL1 *match, MatchCSCMuL1::CLCT &clct);


  math::XYZVectorD cscSimHitGlobalPosition ( PSimHit &h );
  math::XYZVectorD cscSimHitGlobalPositionX0( PSimHit &h );

  TrajectoryStateOnSurface propagateSimTrackToZ(const SimTrack *track, const SimVertex *vtx, double z);
  TrajectoryStateOnSurface propagateSimTrackToDT(const SimTrack *track, const SimVertex *vtx);

  // 4-bit LCT quality number
  unsigned int findQuality(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT);

  // Visualization of wire group digis
  void dumpWireDigis(CSCDetId &id, const CSCWireDigiCollection* wiredc);


  // From Ingo:
  // calculates the weight of the event to reproduce a min bias
  //spectrum, from G. Wrochna's note CMSS 1997/096

  static void setupTFModeHisto(TH1D* h);

  std::pair<float, float> intersectionEtaPhi(CSCDetId id, int wg, int hs);
  csctf::TrackStub buildTrackStub(const CSCCorrelatedLCTDigi &d, CSCDetId id);

  void cleanUp();

private:

// methods
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;


  edm::ParameterSet ptLUTset;
  edm::ParameterSet CSCTFSPset;
  CSCTFPtLUT* ptLUT;
  CSCTFSectorProcessor* my_SPs[2][6];
  CSCSectorReceiverLUT* srLUTs_[5][6][2];
  CSCTFDTReceiver* my_dtrc;
  void runCSCTFSP(const CSCCorrelatedLCTDigiCollection*, const L1MuDTChambPhContainer*);
  unsigned long long  muScalesCacheID_;
  unsigned long long  muPtScaleCacheID_;

  edm::ESHandle< L1MuTriggerScales > muScales;
  edm::ESHandle< L1MuTriggerPtScale > muPtScale;

// config parameters:

  bool lightRun;
  bool defaultME1a;

  bool doStrictSimHitToTrackMatch_;
  bool matchAllTrigPrimitivesInChamber_;
  int minNHitsShared_;
  double minDeltaYAnode_;
  double minDeltaYCathode_;
  int minDeltaWire_;
  int maxDeltaWire_;
  int minDeltaStrip_;

  // debugging switches:
  int debugALLEVENT;
  int debugINHISTOS;
  int debugALCT;
  int debugCLCT;
  int debugLCT;
  int debugMPLCT;
  int debugTFTRACK;
  int debugTFCAND;
  int debugGMTCAND;
  int debugL1EXTRA;

  double minSimTrPt_;
  double minSimTrPhi_;
  double maxSimTrPhi_;
  double minSimTrEta_;
  double maxSimTrEta_;
  bool invertSimTrPhiEta_;
  bool bestPtMatch_;
  bool onlyForwardMuons_;

  int minBX_;
  int maxBX_;
  int minTMBBX_;
  int maxTMBBX_;

  int minBxALCT_;
  int maxBxALCT_;
  int minBxCLCT_;
  int maxBxCLCT_;
  int minBxLCT_;
  int maxBxLCT_;
  int minBxMPLCT_;
  int maxBxMPLCT_;
  int minBxGMT_;
  int maxBxGMT_;

  bool centralBxOnlyGMT_;
  bool gangedME1a;
  bool goodChambersOnly_;
  
  int lookAtTrackCondition_;
  
  bool doME1a_, naiveME1a_;

  bool addGhostLCTs_;
  
  bool minNStWith4Hits_;
  bool requireME1With4Hits_;
  
  double minSimTrackDR_;

  mugeo::MuFiducial* mufiducial_;
  
// members
  std::vector<MatchCSCMuL1*> matches;
  
  std::map<unsigned,unsigned> trkId2Index;

  const CSCGeometry* cscGeometry;
  const DTGeometry* dtGeometry;
  const RPCGeometry* rpcGeometry;
  edm::ESHandle<MuonDetLayerGeometry> muonGeometry;

  const GEMGeometry* gemGeometry;

  edm::ParameterSet gemMatchCfg_;
  std::vector<double> gemPTs_, gemDPhisOdd_, gemDPhisEven_;

  bool isGEMDPhiGood(double dphi, double tfpt, int is_odd);

// simhits for matching to simtracks:
  bool simHitsFromCrossingFrame_;
  std::string simHitsModuleName_;
  std::string simHitsCollectionName_;

  SimHitAnalysis::PSimHitMap theCSCSimHitMap;
  //SimHitAnalysis::PSimHitMap theDTSimHitMap;
  //SimHitAnalysis::PSimHitMap theRPCSimHitMap;


  CSCStripConditions * theStripConditions;

  std::vector<const CSCCorrelatedLCTDigi*> ghostLCTs;

  // propagators
  edm::ESHandle<Propagator> propagatorAlong;
  edm::ESHandle<Propagator> propagatorOpposite;
  edm::ESHandle<MagneticField> theBField;

  int nevt;

  TH1D * h_N_mctr;
  TH1D * h_N_simtr;

  TH1D * h_pt_mctr;
  TH1D * h_eta_mctr;
  TH1D * h_phi_mctr;

  TH1D * h_DR_mctr_simtr; 
  TH1D * h_MinDR_mctr_simtr; 

  TH1D * h_DR_2SimTr;
  TH1D * h_DR_2SimTr_looked;
  TH1D * h_DR_2SimTr_after_mpc_ok_plus;
  TH1D * h_DR_2SimTr_after_tfcand_ok_plus;

  TH2D * h_csctype_vs_alct_occup;
  TH2D * h_csctype_vs_clct_occup;
  
  TH2D * h_eta_vs_nalct;
  TH2D * h_eta_vs_nclct;
  TH2D * h_eta_vs_nlct;
  TH2D * h_eta_vs_nmplct;
  
  TH2D * h_pt_vs_nalct;
  TH2D * h_pt_vs_nclct;
  TH2D * h_pt_vs_nlct;
  TH2D * h_pt_vs_nmplct;
  
  TH2D * h_csctype_vs_nlct;
  TH2D * h_csctype_vs_nmplct;

  TH2D * h_eta_vs_nalct_cscstation[MAX_STATIONS];
  TH2D * h_eta_vs_nclct_cscstation[MAX_STATIONS];
  TH2D * h_eta_vs_nlct_cscstation[MAX_STATIONS];

  TH2D * h_eta_vs_nalct_cscstation_ok[MAX_STATIONS];
  TH2D * h_eta_vs_nclct_cscstation_ok[MAX_STATIONS];
  TH2D * h_eta_vs_nlct_cscstation_ok[MAX_STATIONS];
  
  TH2D * h_nmusimhits_vs_nalct_cscdet[CSC_TYPES];
  TH2D * h_nmusimhits_vs_nclct_cscdet[CSC_TYPES];
  TH2D * h_nmusimhits_vs_nlct_cscdet[CSC_TYPES];

  TH1D * h_deltaY__alct_cscdet[CSC_TYPES];
  TH1D * h_deltaY__clct_cscdet[CSC_TYPES];
  TH1D * h_deltaPhi__alct_cscdet[CSC_TYPES];
  TH1D * h_deltaPhi__clct_cscdet[CSC_TYPES];

//  TH1D * h_deltaY__alct_cscdet_nl[CSC_TYPES][7];
//  TH1D * h_deltaY__clct_cscdet_nl[CSC_TYPES][7];
//  TH1D * h_deltaPhi__alct_cscdet_nl[CSC_TYPES][7];
//  TH1D * h_deltaPhi__clct_cscdet_nl[CSC_TYPES][7];


  TH2D * h_ov_nmusimhits_vs_nalct_cscdet[CSC_TYPES];
  TH2D * h_ov_nmusimhits_vs_nclct_cscdet[CSC_TYPES];
  TH2D * h_ov_nmusimhits_vs_nlct_cscdet[CSC_TYPES];

  TH1D * h_ov_deltaY__alct_cscdet[CSC_TYPES];
  TH1D * h_ov_deltaY__clct_cscdet[CSC_TYPES];
  TH1D * h_ov_deltaPhi__alct_cscdet[CSC_TYPES];
  TH1D * h_ov_deltaPhi__clct_cscdet[CSC_TYPES];

//  TH1D * h_ov_deltaY__alct_cscdet_nl[CSC_TYPES][7];
//  TH1D * h_ov_deltaY__clct_cscdet_nl[CSC_TYPES][7];
//  TH1D * h_ov_deltaPhi__alct_cscdet_nl[CSC_TYPES][7];
//  TH1D * h_ov_deltaPhi__clct_cscdet_nl[CSC_TYPES][7];

  TH1D * h_delta__wire_cscdet[CSC_TYPES];
  TH1D * h_delta__strip_cscdet[CSC_TYPES];

  TH1D * h_ov_delta__wire_cscdet[CSC_TYPES];
  TH1D * h_ov_delta__strip_cscdet[CSC_TYPES];

  TH1D * h_bx__alct_cscdet[CSC_TYPES];
  TH1D * h_bx__clct_cscdet[CSC_TYPES];
  TH1D * h_bx__lct_cscdet[CSC_TYPES];
  TH1D * h_bx__mpc_cscdet[CSC_TYPES];

  TH2D * h_bx_lct__alct_vs_clct_cscdet[CSC_TYPES];

  TH1D * h_bx_min__alct_cscdet[CSC_TYPES];
  TH1D * h_bx_min__clct_cscdet[CSC_TYPES];
  TH1D * h_bx_min__lct_cscdet[CSC_TYPES];
  TH1D * h_bx_min__mpc_cscdet[CSC_TYPES];

  TH1D * h_bx__alctOk_cscdet[CSC_TYPES];
  TH1D * h_bx__clctOk_cscdet[CSC_TYPES];
  
  TH1D * h_bx__alctOkBest_cscdet[CSC_TYPES];
  TH1D * h_bx__clctOkBest_cscdet[CSC_TYPES];

  TH2D * h_wg_vs_bx__alctOkBest_cscdet[CSC_TYPES];

  TH1D * h_bxf__alct_cscdet[CSC_TYPES];
  TH1D * h_bxf__clct_cscdet[CSC_TYPES];
  TH1D * h_bxf__alctOk_cscdet[CSC_TYPES];
  TH1D * h_bxf__clctOk_cscdet[CSC_TYPES];

  TH1D * h_dbxbxf__alct_cscdet[CSC_TYPES];
  TH1D * h_dbxbxf__clct_cscdet[CSC_TYPES];

  TH2D * h_bx_me11nomatchclct_alct_vs_clct;
  TH2D * h_bx_me11nomatchalct_alct_vs_clct;

  TH2D * h_bx_me1_aclct_ok_lct_no__bx_alct_vs_dbx_ACLCT;

  TH2D * h_nMplct_vs_nDigiMplct;
  TH2D * h_qu_vs_nDigiMplct;

  TH2D * h_ntftrackall_vs_ntftrack;
  TH2D * h_ntfcandall_vs_ntfcand;

  TH2D * h_eta_vs_ntfcand;
  TH2D * h_pt_vs_ntfcand;

  TH2D * h_pt_vs_qu;
  TH2D * h_eta_vs_qu;
  
  TH1D * h_cscdet_of_chamber;
  TH1D * h_cscdet_of_chamber_w_alct;
  TH1D * h_cscdet_of_chamber_w_clct;
  TH1D * h_cscdet_of_chamber_w_mplct;



  TH1D * h_pt_initial0;
  TH1D * h_pt_initial;
  TH1D * h_pt_initial_1b;
  TH1D * h_pt_initial_gem_1b;
  
  TH1D * h_pt_me1_initial;
  TH1D * h_pt_me2_initial;
  TH1D * h_pt_me3_initial;
  TH1D * h_pt_me4_initial;

  TH1D * h_pt_initial_1st;
  TH1D * h_pt_initial_2st;
  TH1D * h_pt_initial_3st;

  TH1D * h_pt_me1_initial_2st;
  TH1D * h_pt_me1_initial_3st;


  TH1D * h_pt_gem_1b;
  TH1D * h_pt_lctgem_1b;

  TH1D * h_pt_me1_mpc;
  TH1D * h_pt_me2_mpc;
  TH1D * h_pt_me3_mpc;
  TH1D * h_pt_me4_mpc;

  TH1D * h_pt_mpc_1st;
  TH1D * h_pt_mpc_2st;
  TH1D * h_pt_mpc_3st;

  TH1D * h_pt_me1_mpc_2st;
  TH1D * h_pt_me1_mpc_3st;

  TH1D * h_pt_tf_initial0_tfpt[N_PT_THRESHOLDS];
  TH1D * h_pt_tf_initial_tfpt[N_PT_THRESHOLDS];
  TH1D * h_pt_tf_stubs222_tfpt[N_PT_THRESHOLDS];
  TH1D * h_pt_tf_stubs223_tfpt[N_PT_THRESHOLDS];
  TH1D * h_pt_tf_stubs233_tfpt[N_PT_THRESHOLDS];


  TH1D * h_pt_after_alct;
  TH1D * h_pt_after_clct;
  TH1D * h_pt_after_lct;
  TH1D * h_pt_after_mpc;
  TH1D * h_pt_after_mpc_ok_plus;
  TH1D * h_pt_after_tftrack;
  TH1D * h_pt_after_tfcand;
  TH1D * h_pt_after_tfcand_pt10;
  TH1D * h_pt_after_tfcand_pt20;
  TH1D * h_pt_after_tfcand_pt40;
  TH1D * h_pt_after_tfcand_pt60;

  TH1D * h_pt_after_tfcand_ok;
  TH1D * h_pt_after_tfcand_pt10_ok;
  TH1D * h_pt_after_tfcand_pt20_ok;
  TH1D * h_pt_after_tfcand_pt40_ok;
  TH1D * h_pt_after_tfcand_pt60_ok;

  TH1D * h_pt_after_tfcand_ok_plus;
  TH1D * h_pt_after_tfcand_ok_plus_pt10;
  TH1D * h_pt_after_tfcand_ok_plus_q[3];
  TH1D * h_pt_after_tfcand_ok_plus_pt10_q[3];

  TH1D * h_pt_after_tfcand_all;
  TH1D * h_pt_after_tfcand_all_ok;
  TH1D * h_pt_after_tfcand_all_pt10_ok;
  TH1D * h_pt_after_tfcand_all_pt20_ok;
  TH1D * h_pt_after_tfcand_all_pt40_ok;
  TH1D * h_pt_after_tfcand_all_pt60_ok;


  TH1D * h_pt_after_tfcand_eta1b_2s[7];
  TH1D * h_pt_after_tfcand_eta1b_2s1b[7];
  TH1D * h_pt_after_tfcand_eta1b_2s123[7];
  TH1D * h_pt_after_tfcand_eta1b_2s13[7];
  TH1D * h_pt_after_tfcand_eta1b_3s[7];
  TH1D * h_pt_after_tfcand_eta1b_3s1b[7];
  TH1D * h_pt_after_tfcand_gem1b_2s1b[7];
  TH1D * h_pt_after_tfcand_gem1b_2s123[7];
  TH1D * h_pt_after_tfcand_gem1b_2s13[7];
  TH1D * h_pt_after_tfcand_gem1b_3s1b[7];
  TH1D * h_pt_after_tfcand_dphigem1b_2s1b[7];
  TH1D * h_pt_after_tfcand_dphigem1b_2s123[7];
  TH1D * h_pt_after_tfcand_dphigem1b_2s13[7];
  TH1D * h_pt_after_tfcand_dphigem1b_3s1b[7];

  TH1D * h_mode_tfcand_gem1b_2s1b_1b[7];


  TH1D * h_pt_after_gmtreg;
  TH1D * h_pt_after_gmtreg_all;
  TH1D * h_pt_after_gmtreg_dr;
  TH1D * h_pt_after_gmt;
  TH1D * h_pt_after_gmt_all;
  TH1D * h_pt_after_gmt_dr;
  TH1D * h_pt_after_gmt_dr_nocsc;
  TH1D * h_pt_after_gmtreg_pt10;
  TH1D * h_pt_after_gmtreg_all_pt10;
  TH1D * h_pt_after_gmtreg_dr_pt10;
  TH1D * h_pt_after_gmt_pt10;
  TH1D * h_pt_after_gmt_all_pt10;
  TH1D * h_pt_after_gmt_dr_pt10;
  TH1D * h_pt_after_gmt_dr_nocsc_pt10;

  TH1D * h_pt_after_gmt_eta1b_1mu[7];
  TH1D * h_pt_after_gmt_gem1b_1mu[7];
  TH1D * h_pt_after_gmt_dphigem1b_1mu[7];

  TH1D * h_pt_after_xtra;
  TH1D * h_pt_after_xtra_all;
  TH1D * h_pt_after_xtra_dr;
  TH1D * h_pt_after_xtra_pt10;
  TH1D * h_pt_after_xtra_all_pt10;
  TH1D * h_pt_after_xtra_dr_pt10;

  TH1D * h_pt_me1_after_mpc_ok_plus;
  TH1D * h_pt_me1_after_tf_ok_plus;
  TH1D * h_pt_me1_after_tf_ok_plus_pt10;
  TH1D * h_pt_me1_after_tf_ok_plus_q[3];
  TH1D * h_pt_me1_after_tf_ok_plus_pt10_q[3];


  // high eta pt distros
  TH1D * h_pth_initial;
  TH1D * h_pth_after_mpc;
  TH1D * h_pth_after_mpc_ok_plus;
 
  TH1D * h_pth_after_tfcand;
  TH1D * h_pth_after_tfcand_pt10;

  TH1D * h_pth_after_tfcand_ok;
  TH1D * h_pth_after_tfcand_pt10_ok;

  TH1D * h_pth_after_tfcand_ok_plus;
  TH1D * h_pth_after_tfcand_ok_plus_pt10;
  TH1D * h_pth_after_tfcand_ok_plus_q[3];
  TH1D * h_pth_after_tfcand_ok_plus_pt10_q[3];

  TH1D * h_pth_me1_after_mpc_ok_plus;
  TH1D * h_pth_me1_after_tf_ok_plus;
  TH1D * h_pth_me1_after_tf_ok_plus_pt10;
  TH1D * h_pth_me1_after_tf_ok_plus_q[3];
  TH1D * h_pth_me1_after_tf_ok_plus_pt10_q[3];

  TH1D * h_pth_over_tfpt_resol;
  TH2D * h_pth_over_tfpt_resol_vs_pt;


  TH1D * h_pth_after_tfcand_ok_plus_3st1a;
  TH1D * h_pth_after_tfcand_ok_plus_pt10_3st1a;
  TH1D * h_pth_me1_after_tf_ok_plus_3st1a;
  TH1D * h_pth_me1_after_tf_ok_plus_pt10_3st1a;


  // 
  TH1D * h_eta_initial0;
  TH1D * h_eta_initial;
  
  TH1D * h_eta_me1_initial;
  TH1D * h_eta_me2_initial;
  TH1D * h_eta_me3_initial;
  TH1D * h_eta_me4_initial;

  TH1D * h_eta_initial_1st;
  TH1D * h_eta_initial_2st;
  TH1D * h_eta_initial_3st;

  TH1D * h_eta_me1_initial_2st;
  TH1D * h_eta_me1_initial_3st;


  TH1D * h_eta_me1_mpc;
  TH1D * h_eta_me2_mpc;
  TH1D * h_eta_me3_mpc;
  TH1D * h_eta_me4_mpc;

  TH1D * h_eta_mpc_1st;
  TH1D * h_eta_mpc_2st;
  TH1D * h_eta_mpc_3st;

  TH1D * h_eta_me1_mpc_2st;
  TH1D * h_eta_me1_mpc_3st;


  TH1D * h_eta_tf_initial0_tfpt[N_PT_THRESHOLDS];
  TH1D * h_eta_tf_initial_tfpt[N_PT_THRESHOLDS];
  TH1D * h_eta_tf_stubs222_tfpt[N_PT_THRESHOLDS];
  TH1D * h_eta_tf_stubs223_tfpt[N_PT_THRESHOLDS];
  TH1D * h_eta_tf_stubs233_tfpt[N_PT_THRESHOLDS];


  TH1D * h_eta_after_alct;
  TH1D * h_eta_after_clct;
  TH1D * h_eta_after_lct;
  TH1D * h_eta_after_mpc;
  TH1D * h_eta_after_mpc_ok;
  TH1D * h_eta_after_mpc_ok_plus;
  TH1D * h_eta_after_mpc_ok_plus_3st;
  TH1D * h_eta_after_mpc_st1;
  TH1D * h_eta_after_mpc_st1_good;
  TH1D * h_eta_after_tftrack;
  TH1D * h_eta_after_tftrack_q[3];
  TH1D * h_eta_after_tfcand;
  TH1D * h_eta_after_tfcand_q[3];
  TH1D * h_eta_after_tfcand_ok;
  TH1D * h_eta_after_tfcand_ok_plus;
  TH1D * h_eta_after_tfcand_ok_pt10;
  TH1D * h_eta_after_tfcand_ok_plus_pt10;
  TH1D * h_eta_after_tfcand_ok_plus_q[3];
  TH1D * h_eta_after_tfcand_ok_plus_pt10_q[3];
  TH1D * h_eta_after_tfcand_pt10;
  TH1D * h_eta_after_tfcand_all;
  TH1D * h_eta_after_tfcand_all_pt10;
  TH1D * h_eta_after_tfcand_my_st1;
  TH1D * h_eta_after_tfcand_org_st1;
  TH1D * h_eta_after_tfcand_comm_st1;
  TH1D * h_eta_after_tfcand_my_st1_pt10;
  
  TH1D * h_eta_after_gmtreg;
  TH1D * h_eta_after_gmtreg_all;
  TH1D * h_eta_after_gmtreg_dr;
  TH1D * h_eta_after_gmt;
  TH1D * h_eta_after_gmt_all;
  TH1D * h_eta_after_gmt_dr;
  TH1D * h_eta_after_gmt_dr_nocsc;
  TH1D * h_eta_after_gmtreg_pt10;
  TH1D * h_eta_after_gmtreg_all_pt10;
  TH1D * h_eta_after_gmtreg_dr_pt10;
  TH1D * h_eta_after_gmt_pt10;
  TH1D * h_eta_after_gmt_all_pt10;
  TH1D * h_eta_after_gmt_dr_pt10;
  TH1D * h_eta_after_gmt_dr_nocsc_pt10;

  TH1D * h_eta_after_xtra;
  TH1D * h_eta_after_xtra_all;
  TH1D * h_eta_after_xtra_dr;
  TH1D * h_eta_after_xtra_pt10;
  TH1D * h_eta_after_xtra_all_pt10;
  TH1D * h_eta_after_xtra_dr_pt10;

  TH2D * h_eta_vs_bx_after_alct;
  TH2D * h_eta_vs_bx_after_clct;
  TH2D * h_eta_vs_bx_after_lct;
  TH2D * h_eta_vs_bx_after_mpc;

  TH1D * h_eta_me1_after_alct;
  TH1D * h_eta_me1_after_alct_okAlct;
  TH1D * h_eta_me1_after_clct;
  TH1D * h_eta_me1_after_clct_okClct;
  TH1D * h_eta_me1_after_alctclct;
  TH1D * h_eta_me1_after_alctclct_okAlct;
  TH1D * h_eta_me1_after_alctclct_okClct;
  TH1D * h_eta_me1_after_alctclct_okAlctClct;

  TH1D * h_eta_me1_after_lct;
  TH1D * h_eta_me1_after_lct_okAlct;
  TH1D * h_eta_me1_after_lct_okAlctClct;
  TH1D * h_eta_me1_after_lct_okClct;
  TH1D * h_eta_me1_after_lct_okClctAlct;
  TH1D * h_eta_me1_after_mplct_okAlctClct;
  TH1D * h_eta_me1_after_mplct_okAlctClct_plus;
  TH1D * h_eta_me1_after_tf_ok;
  TH1D * h_eta_me1_after_tf_ok_pt10;
  TH1D * h_eta_me1_after_tf_ok_plus;
  TH1D * h_eta_me1_after_tf_ok_plus_pt10;
  TH1D * h_eta_me1_after_tf_ok_plus_q[3];
  TH1D * h_eta_me1_after_tf_ok_plus_pt10_q[3];
  //TH1D * h_eta_me1_after_tf_all;
  //TH1D * h_eta_me1_after_tf_all_pt10;


  TH1D * h_eta_me1_after_mplct_ok;
  TH1D * h_eta_me2_after_mplct_ok;
  TH1D * h_eta_me3_after_mplct_ok;
  TH1D * h_eta_me4_after_mplct_ok;



  TH1D * h_eta_after_mpc_ok_plus_3st1a;
  TH1D * h_eta_after_tfcand_ok_plus_3st1a;
  TH1D * h_eta_after_tfcand_ok_plus_pt10_3st1a;
  TH1D * h_eta_me1_after_tf_ok_plus_3st1a;
  TH1D * h_eta_me1_after_tf_ok_plus_pt10_3st1a;


  TH1D * h_wg_me11_initial;
  TH1D * h_wg_me11_after_alct_okAlct;
  TH1D * h_wg_me11_after_alctclct_okAlctClct;
  TH1D * h_wg_me11_after_lct_okAlctClct;

  TH1D * h_bx_after_alct;
  TH1D * h_bx_after_clct;
  TH1D * h_bx_after_lct;
  TH1D * h_bx_after_mpc;
  
  TH1D * h_phi_initial;
  TH1D * h_phi_after_alct;
  TH1D * h_phi_after_clct;
  TH1D * h_phi_after_lct;
  TH1D * h_phi_after_mpc;
  TH1D * h_phi_after_tftrack;
  TH1D * h_phi_after_tfcand;
  TH1D * h_phi_after_tfcand_all;
  TH1D * h_phi_after_gmtreg;
  TH1D * h_phi_after_gmtreg_all;
  TH1D * h_phi_after_gmtreg_dr;
  TH1D * h_phi_after_gmt;
  TH1D * h_phi_after_gmt_all;
  TH1D * h_phi_after_gmt_dr;
  TH1D * h_phi_after_gmt_dr_nocsc;
  TH1D * h_phi_after_xtra;
  TH1D * h_phi_after_xtra_all;
  TH1D * h_phi_after_xtra_dr;

  TH1D * h_phi_me1_after_alct;
  TH1D * h_phi_me1_after_alct_okAlct;
  TH1D * h_phi_me1_after_clct;
  TH1D * h_phi_me1_after_clct_okClct;
  TH1D * h_phi_me1_after_alctclct;
  TH1D * h_phi_me1_after_alctclct_okAlct;
  TH1D * h_phi_me1_after_alctclct_okClct;
  TH1D * h_phi_me1_after_alctclct_okAlctClct;
  TH1D * h_phi_me1_after_lct;
  TH1D * h_phi_me1_after_lct_okAlct;
  TH1D * h_phi_me1_after_lct_okAlctClct;
  TH1D * h_phi_me1_after_lct_okClct;
  TH1D * h_phi_me1_after_lct_okClctAlct;
  TH1D * h_phi_me1_after_mplct_ok;
  TH1D * h_phi_me1_after_mplct_okAlctClct;
  TH1D * h_phi_me1_after_mplct_okAlctClct_plus;
  TH1D * h_phi_me1_after_tf_ok;

  TH1D * h_qu_alct;
  TH1D * h_qu_clct;
  TH1D * h_qu_lct;
  TH1D * h_qu_mplct;
  TH2D * h_qu_vs_bx__alct;
  TH2D * h_qu_vs_bx__clct;
  TH2D * h_qu_vs_bx__lct;
  TH2D * h_qu_vs_bx__mplct;

  TH2D * h_qu_vs_bxclct__lct;

  TH2D * h_pt_vs_bend__clct_cscdet[CSC_TYPES];
  TH2D * h_pt_vs_bend__clctOk_cscdet[CSC_TYPES];
  TH1D * h_bend__clctOk_cscdet[CSC_TYPES];

  TH1D * h_pattern_mplct;
  TH1D * h_pattern_mplct_cscdet[CSC_TYPES];

  TH1D * h_type_lct;
  TH1D * h_type_lct_cscdet[CSC_TYPES];

  TH2D * h_bxdbx_alct_a1_da2;
  TH2D * h_bxdbx_alct_a1_da2_cscdet[CSC_TYPES];
  TH2D * h_bxdbx_clct_c1_dc2;
  TH2D * h_bxdbx_clct_c1_dc2_cscdet[CSC_TYPES];

  TH2D * h_dbx_lct_a1_a2;
  TH2D * h_dbx_lct_a1_a2_cscdet[CSC_TYPES];
  TH2D * h_bx_lct_a1_a2;
  TH2D * h_bx_lct_a1_a2_cscdet[CSC_TYPES];

  TH1D * h_tf_stub_bx;
  TH1D * h_tf_stub_bx_cscdet[CSC_TYPES];
  TH1D * h_tf_stub_qu;
  TH1D * h_tf_stub_qu_cscdet[CSC_TYPES];
  TH2D * h_tf_stub_qu_vs_bx;
  TH2D * h_tf_stub_qu_vs_bx_cscdet[CSC_TYPES];
  TH1D * h_tf_stub_csctype;
  TH1D * h_tf_stub_csctype_org;
  TH1D * h_tf_stub_csctype_org_unmatch;
  
  TH1D * h_tf_stub_bxclct;
  TH1D * h_tf_stub_bxclct_cscdet[CSC_TYPES];
  
  TH1D * h_tf_stub_pattern;
  TH1D * h_tf_stub_pattern_cscdet[CSC_TYPES];
  
  TH1D * h_tf_n_uncommon_stubs;
  TH1D * h_tf_n_stubs;
  TH1D * h_tf_n_matchstubs;
  TH2D * h_tf_n_stubs_vs_matchstubs;

  TH1D * h_tfpt;
  TH2D * h_tfpt_vs_qu;
  TH1D * h_tfeta;
  TH1D * h_tfphi;
  TH1D * h_tfbx;
  TH1D * h_tfqu;
  TH1D * h_tfdr;
  TH1D * h_tf_mode;

  TH1D * h_tf_pt_h42_2st;
  TH1D * h_tf_pt_h42_3st;
  TH1D * h_tf_pt_h42_2st_w;
  TH1D * h_tf_pt_h42_3st_w;

  TH1D * h_tf_check_mode;
  TH1D * h_tf_check_bx;
  TH2D * h_tf_check_n_stubs_vs_matched;
  TH2D * h_tf_check_st1_mcStrip_vs_ptbin;
  TH2D * h_tf_check_st1_mcStrip_vs_ptbin_all;
  TH2D * h_tf_check_st1_mcWG_vs_ptbin_all;
  TH1D * h_tf_check_st1_wg;
  TH1D * h_tf_check_st1_strip;
  TH1D * h_tf_check_st1_mcStrip;
  TH1D * h_tf_check_st1_chamber;
  //TH2D * h_tf_check_st1_scaledDPhi12_vs_ptbin;
  //TH2D * h_tf_check_st1_scaledDPhi12_vs_ptbin_ok;
  //TH2D * h_tf_check_st1_scaledDPhi23_vs_ptbin;
  //TH2D * h_tf_check_st1_scaledDPhi23_vs_ptbin_ok;

  TH1D * h_gmtpt;
  TH1D * h_gmteta;
  TH1D * h_gmtphi;
  TH1D * h_gmtbx;
  TH1D * h_gmtrank;
  TH1D * h_gmtqu;
  TH1D * h_gmtisrpc;
  TH1D * h_gmtdr;

  TH1D * h_gmtxpt;
  TH1D * h_gmtxeta;
  TH1D * h_gmtxphi;
  TH1D * h_gmtxbx;
  TH1D * h_gmtxrank;
  TH1D * h_gmtxqu;
  TH1D * h_gmtxisrpc;
  TH1D * h_gmtxdr;

  TH1D * h_gmtxpt_nocsc;
  TH1D * h_gmtxeta_nocsc;
  TH1D * h_gmtxphi_nocsc;
  TH1D * h_gmtxbx_nocsc;
  TH1D * h_gmtxrank_nocsc;
  TH1D * h_gmtxqu_nocsc;
  TH1D * h_gmtxisrpc_nocsc;
  TH1D * h_gmtxdr_nocsc;

  TH1D * h_gmtxqu_nogmtreg;
  TH1D * h_gmtxisrpc_nogmtreg;
  TH1D * h_gmtxqu_notfcand;
  TH1D * h_gmtxisrpc_notfcand;
  TH1D * h_gmtxqu_nompc;
  TH1D * h_gmtxisrpc_nompc;

  TH1D * h_xtrapt;
  TH1D * h_xtraeta;
  TH1D * h_xtraphi;
  TH1D * h_xtrabx;
  TH1D * h_xtradr;

  TH1D * h_n_alct;
  TH1D * h_n_clct;
  TH1D * h_n_lct;
  TH1D * h_n_mplct;
  TH1D * h_n_tftrack;
  TH1D * h_n_tftrack_all;
  TH1D * h_n_tfcand;
  TH1D * h_n_tfcand_all;
  TH1D * h_n_gmtregcand;
  TH1D * h_n_gmtregcand_all;
  TH1D * h_n_gmtcand;
  TH1D * h_n_gmtcand_all;
  TH1D * h_n_xtra;
  TH1D * h_n_xtra_all;

  TH1D * h_n_ch_w_alct;
  TH1D * h_n_ch_w_clct;
  TH1D * h_n_ch_w_lct;
  TH1D * h_n_ch_w_mplct;

  TH1D * h_n_bx_per_ch_alct;
  TH1D * h_n_bx_per_ch_clct;
  TH1D * h_n_bx_per_ch_lct;

  TH1D * h_n_per_ch_alct;
  TH1D * h_n_per_ch_clct;
  TH1D * h_n_per_ch_lct;
  TH1D * h_n_per_ch_mplct;
  TH1D * h_n_per_ch_alct_cscdet[CSC_TYPES];
  TH2D * h_n_per_ch_alct_vs_bx_cscdet[CSC_TYPES];
  TH1D * h_n_per_ch_clct_cscdet[CSC_TYPES];
  TH1D * h_n_per_ch_lct_cscdet[CSC_TYPES];
  TH1D * h_n_per_ch_mplct_cscdet[CSC_TYPES];

  TH2D * h_n_per_ch_me1nomatchclct_alct_vs_bx_cscdet[CSC_TYPES];
  TH2D * h_n_per_ch_me1nomatchclct_clct_vs_bx_cscdet[CSC_TYPES];
  TH1D * h_n_per_ch_me1nomatchclct_clct_cscdet[CSC_TYPES];

  TH2D * h_n_per_ch_me1nomatchalct_alct_vs_bx_cscdet[CSC_TYPES];
  TH2D * h_n_per_ch_me1nomatchalct_clct_vs_bx_cscdet[CSC_TYPES];
  TH1D * h_n_per_ch_me1nomatchalct_clct_cscdet[CSC_TYPES];

  TH2D * h_n_per_ch_me1nomatch_alct_vs_bx_cscdet[CSC_TYPES];
  TH1D * h_n_per_ch_me1nomatch_clct_cscdet[CSC_TYPES];


  TH1D * h_station_tf_pu_ok;
  TH1D * h_station_tf_pu_no;
  TH1D * h_station_tf_pu_ok_once;
  TH1D * h_station_tf_pu_no_once;
  TH1D * h_station_tforg_pu_ok;
  TH1D * h_station_tforg_pu_no;
  TH1D * h_station_tforg_pu_ok_once;
  TH1D * h_station_tforg_pu_no_once;
  TH1D * h_tfqu_pt10;
  TH1D * h_tfqu_pt10_no;
  
  TH1D * h_dBx_LctAlct;
  TH1D * h_dBx_LctClct;
  TH1D * h_dBx_1inCh_LctClct;
  TH1D * h_dBx_2inCh_LctClct;
  TH2D * h_dBx_2inCh_LctClct2;

  TH1D * h_dBx_LctClct_cscdet[CSC_TYPES];
  TH1D * h_dBx_1inCh_LctClct_cscdet[CSC_TYPES];
  TH1D * h_dBx_2inCh_LctClct_cscdet[CSC_TYPES];
  TH2D * h_dBx_2inCh_LctClct2_cscdet[CSC_TYPES];

  TH1D * h_pt_over_tfpt_resol;
  TH2D * h_pt_over_tfpt_resol_vs_pt;

  TH1D * h_eta_minus_tfeta_resol;
  TH1D * h_phi_minus_tfphi_resol;

  TH2D * h_strip_v_wireg_me1a;
  TH2D * h_strip_v_wireg_me1b;

  TH1D * h_gmt_mindr; 
  TH1D * h_gmt_dr_maxrank; 
  bool fill_debug_tree_;
  TTree* dbg_tree;
  void bookDbgTTree();
  struct DbgStruct
  {
    int evtn;
    int trkn;
    float pt,eta,phi;
    float tfpt, tfeta, tfphi;
    int tfpt_packed, tfeta_packed, tfphi_packed;
    int nseg, nseg_ok;
    int dPhi12;
    int dPhi23;
    int meEtap, mePhip;
    int mcStrip;
    int mcWG;
    int strip;
    int wg;
    int chamber;
  };
  DbgStruct dbg_;
  void resetDbg(DbgStruct& d);
};

#endif
