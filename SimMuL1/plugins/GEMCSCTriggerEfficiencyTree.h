#ifndef SimMuL1_GEMCSCTriggerEfficiencyTree_h
#define SimMuL1_GEMCSCTriggerEfficiencyTree_h

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
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

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

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"
#include <L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>

#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"

#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"

#include "GEMCode/GEMValidation/src/SimTrackMatchManager.h"
#include "GEMCode/SimMuL1/interface/MuGeometryHelpers.h"


#include "GEMCode/SimMuL1/interface/PSimHitMap.h"
#include "GEMCode/SimMuL1/interface/MatchCSCMuL1.h"

#include "GEMCode/SimMuL1/plugins/Ntuple.h"

class GEMCSCTriggerEfficiencyTree : public edm::EDAnalyzer 
{
public:

  explicit GEMCSCTriggerEfficiencyTree(const edm::ParameterSet&);
  ~GEMCSCTriggerEfficiencyTree();

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
  bool gangedME1a;
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
  int debugRATE;

  double minSimTrPt_;
  double minSimTrPhi_;
  double maxSimTrPhi_;
  double minSimTrEta_;
  double maxSimTrEta_;
  bool invertSimTrPhiEta_;
  bool bestPtMatch_;

  int minBX_;
  int maxBX_;
  int minTMBBX_;
  int maxTMBBX_;
  int minRateBX_;
  int maxRateBX_;

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

  bool doSelectEtaForGMTRates_;

  bool goodChambersOnly_;
  
  int lookAtTrackCondition_;
  
  bool doME1a_, naiveME1a_;

  bool addGhostLCTs_;
  
  bool minNStWith4Hits_;
  bool requireME1With4Hits_;
  
  double minSimTrackDR_;

  
// members
  std::vector<MatchCSCMuL1*> matches;
  
  // map of track Id (in simulation) to the index (in vector)
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

  TTree *tree_eff_;
  MyNtuple etrk_;


  CSCStripConditions * theStripConditions;

  std::vector<const CSCCorrelatedLCTDigi*> ghostLCTs;

  // propagators
  edm::ESHandle<Propagator> propagatorAlong;
  edm::ESHandle<Propagator> propagatorOpposite;
  edm::ESHandle<MagneticField> theBField;


  int nevt;
};

#endif
