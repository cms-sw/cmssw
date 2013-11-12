// -*- C++ -*-
//
// Package:    SimpleMuon
// Class:      SimpleMuon
// 
/**\class SimpleMuon SimpleMuon.cc GEMCode/SimpleMuon/plugins/SimpleMuon.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Sven Dildick
//         Created:  Wed, 06 Nov 2013 20:05:53 GMT
// $Id$
//
//


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

#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "GEMCode/SimMuL1/interface/MatchCSCMuL1.h"
#include "GEMCode/SimMuL1/interface/MuGeometryHelpers.h"
#include "GEMCode/SimMuL1/interface/PSimHitMap.h"
#include "GEMCode/SimMuL1/plugins/Ntuple.h"

#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"


typedef std::vector<std::vector<Float_t> > vvfloat;
typedef std::vector<std::vector<Int_t> > vvint;
typedef std::vector<Float_t> vfloat;
typedef std::vector<Int_t> vint;

//
// class declaration
//

class SimpleMuon : public edm::EDAnalyzer 
{
public:
  explicit SimpleMuon(const edm::ParameterSet&);
  ~SimpleMuon();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  void propagateToCSCStations(MatchCSCMuL1*);
  TrajectoryStateOnSurface propagateSimTrackToZ(const SimTrack*, const SimVertex*, double);
  void matchSimTrack2SimHits(MatchCSCMuL1*, const edm::SimTrackContainer&, 
			     const edm::SimVertexContainer&, const edm::PSimHitContainer*);
  std::vector<unsigned> fillSimTrackFamilyIds(unsigned, const edm::SimTrackContainer &, 
					      const edm::SimVertexContainer &);
  std::vector<PSimHit> hitsFromSimTrack(std::vector<unsigned>, SimHitAnalysis::PSimHitMap &);
  std::vector<PSimHit> hitsFromSimTrack(unsigned, SimHitAnalysis::PSimHitMap &);
  std::vector<PSimHit> hitsFromSimTrack(unsigned, int, SimHitAnalysis::PSimHitMap &);
  void matchSimTrack2ALCTs(MatchCSCMuL1 *, const edm::PSimHitContainer*, 
			   const CSCALCTDigiCollection*, const CSCWireDigiCollection*);
  unsigned matchCSCAnodeHits(const std::vector<CSCAnodeLayerInfo>& , 
			     std::vector<PSimHit> &); 
  bool compareSimHits(PSimHit &, PSimHit &);
  void matchSimTrack2CLCTs( MatchCSCMuL1 *, 
             const edm::PSimHitContainer* , 
             const CSCCLCTDigiCollection *, 
             const CSCComparatorDigiCollection* );
  unsigned
  matchCSCCathodeHits(const std::vector<CSCCathodeLayerInfo>& allLayerInfo, 
		      std::vector<PSimHit> &matchedHit); 

  // fit muon's hits to a 2D linear stub in a chamber :
  //   wires:   work in 2D plane going through z axis :
  //     z becomes a new x axis, and new y is perpendicular to it
  //     (using SimTrack's position point as well when there is <= 2 mu's hits in chamber)
  // fit a 2D stub from SimHits matched to a digi
  // set deltas between SimTrack's and Digi's 2D stubs in (Z,R) -> (x,y) plane
  int calculate2DStubsDeltas(MatchCSCMuL1 *, MatchCSCMuL1::ALCT &);

  // fit muon's hits to a 2D linear stub in a chamber :
  //   stripes:  work in 2D cylindrical surface :
  //     z becomes a new x axis, and phi is a new y axis
  //     (using SimTrack stub with no-bend in phi if there is 1 mu's hit in chamber)
  // fit a 2D stub from SimHits matched to a digi
  // set deltas between SimTrack's and Digi's 2D stubs in (Z,Phi) -> (x,y) plane
  int calculate2DStubsDeltas(MatchCSCMuL1 *match, MatchCSCMuL1::CLCT &clct);

  int getCSCType(const CSCDetId &);

  
  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  
  // ----------member data ---------------------------
  
  const CSCGeometry* cscGeometry;
  edm::ESHandle<MuonDetLayerGeometry> muonGeometry;

  // propagators
  edm::ESHandle<Propagator> propagatorAlong;
  edm::ESHandle<Propagator> propagatorOpposite;
  edm::ESHandle<MagneticField> theBField;

  bool doStrictSimHitToTrackMatch_;

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
  bool doME1a_;
  bool goodChambersOnly_;
  bool gangedME1a;

  bool debugALCT;
  bool debugCLCT;
  bool defaultME1a;
  int minBX_;
  int maxBX_;
  int minDeltaWire_;
  int maxDeltaWire_;
  int minDeltaYAnode_;
  int minNHitsShared_;
  bool matchAllTrigPrimitivesInChamber_;
  int minDeltaStrip_;
  int minDeltaYCathode_;

  SimHitAnalysis::PSimHitMap theCSCSimHitMap;

  CSCStripConditions * theStripConditions;

  std::map<unsigned,unsigned> trkId2Index;

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

  TTree *tree_eff_;
  MyNtuple etrk_;
};

//
// constants, enums and typedefs
//

// ================================================================================================
// class' constants
//
const std::string SimpleMuon::csc_type[CSC_TYPES+1] = 
  { "ME1/1", "ME1/2", "ME1/3", "ME1/a", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2", "ME1/T"};
const std::string SimpleMuon::csc_type_[CSC_TYPES+1] = 
  { "ME11", "ME12", "ME13", "ME1A", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42", "ME1T"};
const std::string SimpleMuon::csc_type_a[CSC_TYPES+2] =
  { "N/A", "ME1/a", "ME1/b", "ME1/2", "ME1/3", "ME2/1", "ME2/2", "ME3/1", "ME3/2", "ME4/1", "ME4/2", "ME1/T"};
const std::string SimpleMuon::csc_type_a_[CSC_TYPES+2] =
  { "NA", "ME1A", "ME1B", "ME12", "ME13", "ME21", "ME22", "ME31", "ME32", "ME41", "ME42", "ME1T"};

const int SimpleMuon::NCHAMBERS[CSC_TYPES] = 
  { 36,  36,  36,  36, 18,  36,  18,  36,  18,  36};

const int SimpleMuon::MAX_WG[CSC_TYPES] = 
  { 48,  64,  32,  48, 112, 64,  96,  64,  96,  64};//max. number of wiregroups

const int SimpleMuon::MAX_HS[CSC_TYPES] = 
  { 128, 160, 128, 96, 160, 160, 160, 160, 160, 160}; // max. # of halfstrips

//const int SimpleMuon::ptype[CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07]= 
//  { -999,  3, -3,  2, -2,  1, -1,  0};  // "signed" pattern (== phiBend)
const int SimpleMuon::pbend[CSCConstants::NUM_CLCT_PATTERNS]= 
  { -999,  -5,  4, -4,  3, -3,  2, -2,  1, -1,  0}; // "signed" pattern (== phiBend)


const double SimpleMuon::PT_THRESHOLDS[N_PT_THRESHOLDS] = {0,10,20,30,40,50};
const double SimpleMuon::PT_THRESHOLDS_FOR_ETA[N_PT_THRESHOLDS] = {10,15,30,40,55,70};


//
// static data member definitions
//

//
// constructors and destructor
//
SimpleMuon::SimpleMuon(const edm::ParameterSet& iConfig)
{
  doStrictSimHitToTrackMatch_ = iConfig.getUntrackedParameter<bool>("doStrictSimHitToTrackMatch", false);

  minBxALCT_ = iConfig.getUntrackedParameter< int >("minBxALCT",5);
  maxBxALCT_ = iConfig.getUntrackedParameter< int >("maxBxALCT",7);
  minBxCLCT_ = iConfig.getUntrackedParameter< int >("minBxCLCT",5);
  maxBxCLCT_ = iConfig.getUntrackedParameter< int >("maxBxCLCT",7);
  minBxLCT_ = iConfig.getUntrackedParameter< int >("minBxLCT",5);
  maxBxLCT_ = iConfig.getUntrackedParameter< int >("maxBxLCT",7);
  minBxMPLCT_ = iConfig.getUntrackedParameter< int >("minBxMPLCT",5);
  maxBxMPLCT_ = iConfig.getUntrackedParameter< int >("maxBxMPLCT",7);

  doME1a_ = iConfig.getUntrackedParameter< bool >("doME1a",false);
  goodChambersOnly_ = iConfig.getUntrackedParameter< bool >("goodChambersOnly",false);

  edm::ParameterSet stripPSet = iConfig.getParameter<edm::ParameterSet>("strips");
  theStripConditions = new CSCDbStripConditions(stripPSet);

  minDeltaWire_    = iConfig.getUntrackedParameter<int>("minDeltaWire", 0);
  maxDeltaWire_    = iConfig.getUntrackedParameter<int>("maxDeltaWire", 2);
  minNHitsShared_ = iConfig.getUntrackedParameter<int>("minNHitsShared_", -1);
  matchAllTrigPrimitivesInChamber_ = iConfig.getUntrackedParameter<bool>("matchAllTrigPrimitivesInChamber", false);
  minDeltaYAnode_    = iConfig.getUntrackedParameter<double>("minDeltaYAnode", -1.);
  minBX_    = iConfig.getUntrackedParameter< int >("minBX",-6);
  maxBX_    = iConfig.getUntrackedParameter< int >("maxBX",6);

  minDeltaYCathode_  = iConfig.getUntrackedParameter<double>("minDeltaYCathode", -1.);
  minDeltaStrip_   = iConfig.getUntrackedParameter<int>("minDeltaStrip", 1);
  gangedME1a = iConfig.getUntrackedParameter<bool>("gangedME1a", false);

  tree_eff_ = etrk_.book(tree_eff_,"efficiency");
  etrk_.initialize();

}


SimpleMuon::~SimpleMuon()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
SimpleMuon::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  etrk_.st_pt.clear();
  etrk_.st_eta.clear();
  etrk_.st_phi.clear();
  etrk_.st_n_csc_simhits.clear();
  etrk_.st_n_alcts.clear();
  etrk_.st_n_alcts_readout.clear();
  etrk_.st_n_clcts.clear();
  etrk_.st_n_tmblcts.clear();
  etrk_.st_n_tmblcts_readout.clear();
  etrk_.st_n_mpclcts.clear();
  etrk_.st_n_mpclcts_readout.clear();
  
  etrk_.csc_alct_valid.clear();
  etrk_.csc_alct_quality.clear();
  etrk_.csc_alct_keywire.clear();
  etrk_.csc_alct_bx.clear();
  etrk_.csc_alct_trknmb.clear();
  etrk_.csc_alct_fullbx.clear();
  etrk_.csc_alct_isGood.clear();
  etrk_.csc_alct_detId.clear();
  etrk_.csc_alct_deltaOk.clear();

  etrk_.csc_clct_valid.clear();
  etrk_.csc_clct_pattern.clear();
  etrk_.csc_clct_quality.clear();
  etrk_.csc_clct_bend.clear();
  etrk_.csc_clct_strip.clear();
  etrk_.csc_clct_bx.clear();
  etrk_.csc_clct_trknmb.clear();
  etrk_.csc_clct_fullbx.clear();
  etrk_.csc_clct_isGood.clear();
  etrk_.csc_clct_detId.clear();
  etrk_.csc_clct_deltaOk.clear();

  etrk_.csc_tmblct_valid.clear();
  etrk_.csc_tmblct_pattern.clear();
  etrk_.csc_tmblct_quality.clear(); 
  etrk_.csc_tmblct_bend.clear();       
  etrk_.csc_tmblct_strip.clear();      
  etrk_.csc_tmblct_bx.clear();
  etrk_.csc_tmblct_trknmb.clear();
  etrk_.csc_tmblct_isAlctGood.clear();
  etrk_.csc_tmblct_isClctGood.clear();
  etrk_.csc_tmblct_detId.clear();
  etrk_.csc_tmblct_gemDPhi.clear();
  etrk_.csc_tmblct_hasGEM.clear();
  etrk_.csc_tmblct_mpclink.clear();


  // ================================================================================================ 

  //                   G E O M E T R Y   A N D   M A G N E T I C   F I E L D 

  // ================================================================================================ 

  // geometry
  edm::ESHandle<CSCGeometry> cscGeom;
  iSetup.get<MuonGeometryRecord>().get(cscGeom);
  iSetup.get<MuonRecoGeometryRecord>().get(muonGeometry);
  cscGeometry = &*cscGeom;

  // csc trigger geometry
  CSCTriggerGeometry::setGeometry(cscGeometry);

  //Get the Magnetic field from the setup
  iSetup.get<IdealMagneticFieldRecord>().get(theBField);

  // Get the propagators
  iSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAnyRK", propagatorAlong);
  iSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAnyOpposite", propagatorOpposite);

  // ================================================================================================ 

  //                                  C O L L E C T I O N S

  // ================================================================================================ 

  // get generator level particle collection
  edm::Handle< reco::GenParticleCollection > hMCCand;
  iEvent.getByLabel("genParticles", hMCCand);
  const reco::GenParticleCollection & cands  = *(hMCCand.product()); 

  /*
  // get PU information
  int bunch_n;
  edm::Handle<std::vector<PileupSummaryInfo> > puInfo;
  iEvent.getByLabel("addPileupInfo", puInfo);
  std::vector<PileupSummaryInfo>::const_iterator PVI;
  bunch_n=0;
  for(PVI = puInfo->begin(); PVI != puInfo->end(); ++PVI) {
    //    pileup.push_back(PVI->getPU_NumInteractions());
    std::cout << "PVI->getPU_NumInteractions() " << PVI->getPU_NumInteractions() << std::endl;
    bunch_n++;
  }
  std::cout << "number of prim verties: " << bunch_n << std::endl;
  */

  // get SimTracks
  edm::Handle< edm::SimTrackContainer > hSimTracks;
  iEvent.getByLabel("g4SimHits", hSimTracks);
  const edm::SimTrackContainer & simTracks = *(hSimTracks.product());

  // get simVertices
  edm::Handle< edm::SimVertexContainer > hSimVertices;
  iEvent.getByLabel("g4SimHits", hSimVertices);
  const edm::SimVertexContainer & simVertices = *(hSimVertices.product());

  // get SimHits
  theCSCSimHitMap.fill(iEvent);

  edm::Handle< edm::PSimHitContainer > MuonCSCHits;
  iEvent.getByLabel("g4SimHits", "MuonCSCHits", MuonCSCHits);
  const edm::PSimHitContainer* allCSCSimHits = MuonCSCHits.product();

  // wire digis
  edm::Handle< CSCWireDigiCollection >       wireDigis;
  iEvent.getByLabel("simMuonCSCDigis","MuonCSCWireDigi",       wireDigis);
  const CSCWireDigiCollection* wiredc = wireDigis.product();

  // strip digis
  edm::Handle< CSCComparatorDigiCollection > compDigis;
  iEvent.getByLabel("simMuonCSCDigis","MuonCSCComparatorDigi", compDigis);
  const CSCComparatorDigiCollection* compdc = compDigis.product();

  // ALCTs 
  edm::Handle< CSCALCTDigiCollection > halcts;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  halcts);
  const CSCALCTDigiCollection* alcts = halcts.product();

  // CLCTs
  edm::Handle< CSCCLCTDigiCollection > hclcts;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  hclcts);
  const CSCCLCTDigiCollection* clcts = hclcts.product();

  // strip&wire matching output  after TMB  
  edm::Handle< CSCCorrelatedLCTDigiCollection > lcts_tmb;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis",  lcts_tmb);
  const CSCCorrelatedLCTDigiCollection* tmblcts = lcts_tmb.product();

  // strip&wire matching output  after MPC sorting
  edm::Handle< CSCCorrelatedLCTDigiCollection > lcts_mpc;
  iEvent.getByLabel("simCscTriggerPrimitiveDigis", "MPCSORTED", lcts_mpc);
  const CSCCorrelatedLCTDigiCollection* mpclcts = lcts_mpc.product();

  //------------------------------------------------------------------------------------------------
  
  // store the CSC trigger primitives in maps of <detId, collection>
  std::map<int,std::vector<CSCALCTDigi> > detALCT;
  detALCT.clear();
  for (CSCALCTDigiCollection::DigiRangeIterator  adetUnitIt = alcts->begin(); adetUnitIt != alcts->end(); adetUnitIt++)
  {
    const CSCDetId& id((*adetUnitIt).first);
    const CSCALCTDigiCollection::Range& range = (*adetUnitIt).second;
    for (CSCALCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
    {
      if ((*digiIt).isValid()) detALCT[id.rawId()].push_back(*digiIt);
    }
  } 

  std::map<int,std::vector<CSCCLCTDigi> > detCLCT;
  detCLCT.clear();
  for (CSCCLCTDigiCollection::DigiRangeIterator adetUnitIt = clcts->begin(); adetUnitIt != clcts->end(); adetUnitIt++)
    {
    const CSCDetId& id((*adetUnitIt).first);
    const CSCCLCTDigiCollection::Range& range = (*adetUnitIt).second;
    for (CSCCLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
    {
      if ((*digiIt).isValid()) detCLCT[id.rawId()].push_back(*digiIt);
    }
  } 

  std::map<int,std::vector<CSCCorrelatedLCTDigi> > detTMBLCT;
  detTMBLCT.clear();
  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt = tmblcts->begin();  detUnitIt != tmblcts->end(); detUnitIt++) 
  {
    const CSCDetId& id((*detUnitIt).first);
    const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
    for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
    {
      if ((*digiIt).isValid()) detTMBLCT[id.rawId()].push_back(*digiIt);
    }
  }

  std::map<int,std::vector<CSCCorrelatedLCTDigi> > detMPCLCT;
  detMPCLCT.clear();
  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt = mpclcts->begin();  detUnitIt != mpclcts->end(); detUnitIt++) 
  {
    const CSCDetId& id((*detUnitIt).first);
    const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
    for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
    {
      if ((*digiIt).isValid()) detMPCLCT[id.rawId()].push_back(*digiIt);
    }
  }


  // ================================================================================================ 
  //
  //                                 M U O N   S E L E C T I O N 
  //
  // ================================================================================================ 

  const bool debug(false);

  // Select the good generator level muons
  std::vector<const reco::GenParticle *> goodGenMuons;
  std::cout << "size of candidate gen particles " << cands.size() << std::endl;
  std::cout << "size of simtracks " << simTracks.size() << std::endl;
  std::cout << "size of simVertices " << simVertices.size() << std::endl;

  for ( size_t ic = 0; ic < cands.size(); ic++ )
  {
    const reco::GenParticle * cand = &(cands[ic]);

    // is this particle a MC muon?    
    if (abs(cand->pdgId()) != 13) continue;
    
    // good MC particle?
    if (cand->status() != 1) continue;
      
    const double mcpt(cand->pt());
    const double mceta(cand->eta());
    const double mcphi(normalizedPhi(cand->phi()));

    // ignore muons with huge eta
    if (fabs(mceta)>10) continue;

    if (debug) std::cout << "Is good MC muon: pt: " << mcpt << ", eta: " << mceta << ", and phi: " << mcphi << std::endl;
    goodGenMuons.push_back(cand);
      
  }
  if (debug) std::cout << "Number of generator level muons " << goodGenMuons.size() << std::endl;

  //------------------------------------------------------------------------------------------------

  // get the primary vertex for this simtrack collection
  int no = 0, primaryVert = -1;
  trkId2Index.clear();
  for (edm::SimTrackContainer::const_iterator istrk = simTracks.begin(); istrk != simTracks.end(); ++istrk)
  {
    // print out: simtrack number, simtrack id, particle index, (px, py, pz, E), vertex index, generator level index (-1 if no generator level particle) 
    //std::cout<<no<<":\t"<<istrk->trackId()<<" "<<*istrk<<std::endl;
    if ( primaryVert == -1 && !(istrk->noVertex()) )
    {
      primaryVert = istrk->vertIndex();
      //std::cout << " -- primary vertex: " << primaryVert << std::endl;
    }
    trkId2Index[istrk->trackId()] = no;
    ++no;
  }
  if ( primaryVert == -1 ) 
  { 
    // No primary vertex found, in non-empty simtrack collection
    std::cout<<">>> WARNING: NO PRIMARY SIMVERTEX! <<<"<<std::endl; 
    if (simTracks.size()>0) return;
  }


  //------------------------------------------------------------------------------------------------

  // select the good simulation level muons
  edm::SimTrackContainer goodSimMuons;
  for (edm::SimTrackContainer::const_iterator track = simTracks.begin(); track != simTracks.end(); ++track)
  {
    int i = track - simTracks.begin();

    // sim track is a muon
    if (abs(track->type()) != 13) continue;
    
    const double sim_pt(sqrt(track->momentum().perp2()));
    const double sim_eta(track->momentum().eta());
    const double sim_phi(normalizedPhi(track->momentum().phi()));

    // ignore muons with very low pt
    if (sim_pt<2.) continue;

    // track has no primary vertex
    if (!(track->vertIndex() == primaryVert)) continue;
    
    // eta selection - has to be in CSC eta !!!
    if (fabs (sim_eta) > 2.5 || fabs (sim_eta) < .8 ) continue;

    // MC matching of SimMuon to GenMuon
    double mc_eta_match = 999, mc_phi_match = 999;
    if (debug) std::cout << "Sim Muon: " << i << std::endl;

    for (unsigned j=0; j<goodGenMuons.size(); ++j)
    {
      if (debug) std::cout << "   MC Muon: " << j << std::endl;
      auto cand(goodGenMuons.at(i));
      double mc_eta(cand->eta());
      double mc_phi(normalizedPhi(cand->phi()));

      const double dr(deltaR(mc_eta, mc_phi, sim_eta, sim_phi));
      if (debug) std::cout << "   dR = " << dr << std::endl;

      //check if the match makes sense
      if (dr < 0.03 && (mc_eta*sim_eta>0))
      {
	mc_eta_match = mc_eta;
	mc_phi_match = mc_phi;
      }
    }
    // ignore the simtrack if there is no GEN level track
    if (mc_eta_match == 999 && mc_phi_match == 999)
    {
      if (debug) std::cout<<">>> WARNING: no matching MC muon for this sim muon! <<<"<<std::endl;      
      continue;	
    }    
    else
    {
      if (debug) std::cout << ">>> INFO: MC muon was matched to Sim muon" << std::endl;
      if (debug) std::cout << ">>> mc_eta = " << mc_eta_match << ", mc_phi = " << mc_phi_match << ", sim_eta = " << sim_eta << ", sim_phi = " << sim_phi << std::endl; 
    }
    // add the muon to the good sim muons
    goodSimMuons.push_back(*track);

    
  }
  if (debug) std::cout << "Number of good simulation level muons " << goodSimMuons.size() << std::endl;


  /*
  // calculate the dR's between all simtracks 
  if (debug) std::cout << "dR between the two good simtracks: "
		       << deltaR(goodSimMuons.at(0).momentum().eta(), normalizedPhi(goodSimMuons.at(0).momentum().phi()),
				 goodSimMuons.at(1).momentum().eta(), normalizedPhi(goodSimMuons.at(1).momentum().phi())) 
		       << std::endl;
  */





  // ================================================================================================ 
  //
  //                      M A I N    L O O P    O V E R   S I M T R A C K S 
  //
  // ================================================================================================ 

  
  for (edm::SimTrackContainer::const_iterator track = goodSimMuons.begin(); track != goodSimMuons.end(); ++track)
  {
    const double track_pt(track->momentum().pt());
    const double track_eta(track->momentum().eta());
    const double track_phi(normalizedPhi(track->momentum().phi()));


    // Extra muon selection
    const bool pt_ok(fabs(track_pt) > 2.);
    const bool eta_ok(1.2 <= fabs(track_eta) && fabs(track_eta) <= 2.5);
    const bool pt_eta_ok(pt_ok && eta_ok);

    if (!pt_eta_ok) continue;

    etrk_.st_pt.push_back(track_pt);
    etrk_.st_eta.push_back(track_eta);
    etrk_.st_phi.push_back(track_phi);

    // create a new matching object for this simtrack 
    MatchCSCMuL1 * match = new MatchCSCMuL1(&*track, &(simVertices[track->vertIndex()]), cscGeometry);
    
    match->muOnly = doStrictSimHitToTrackMatch_;
    match->minBxALCT  = minBxALCT_;
    match->maxBxALCT  = maxBxALCT_;
    match->minBxCLCT  = minBxCLCT_;
    match->maxBxCLCT  = maxBxCLCT_;
    match->minBxLCT   = minBxLCT_;
    match->maxBxLCT   = maxBxLCT_;
    match->minBxMPLCT = minBxMPLCT_;
    match->maxBxMPLCT = maxBxMPLCT_;
    
    // get the approximate position at the CSC stations
    propagateToCSCStations(match);
    
    // match SimHits and do some checks
    matchSimTrack2SimHits(match, simTracks, simVertices, allCSCSimHits);
    
    // match ALCT digis and SimHits;
    // if there are common SimHits in SimTrack, match to SimTrack
    matchSimTrack2ALCTs(match, allCSCSimHits, alcts, wiredc);

    matchSimTrack2CLCTs(match, allCSCSimHits, clcts, compdc);

    etrk_.st_n_csc_simhits.push_back(match->simHits.size());
    etrk_.st_n_alcts.push_back(match->ALCTs.size());
    etrk_.st_n_clcts.push_back(match->CLCTs.size());
//     etrk_.st_n_tmblcts.push_back();
//     etrk_.st_n_mpclcts.push_back();
//     etrk_.st_n_tfTracks.push_back();
//     etrk_.st_n_tfTracksAll.push_back();
//     etrk_.st_n_tfCands.push_back();
//     etrk_.st_n_tfCandsAll.push_back();
//     etrk_.st_n_gmtRegCands.push_back();
//     etrk_.st_n_gmtRegCandsAll.push_back();
//     etrk_.st_n_gmtRegBest.push_back();
//     etrk_.st_n_gmtCands.push_back();
//     etrk_.st_n_gmtCandsAll.push_back();
//     etrk_.st_n_gmtBest.push_back();
//     etrk_.st_n_l1Extra.push_back();
//     etrk_.st_n_l1ExtraAll.push_back();
//     etrk_.st_n_l1ExtraBest.push_back();

// 		 <<" nLCT: "<<match->LCTs.size() 
// 		 <<" nMPLCT: "<<match->MPLCTs.size() 
// 		 <<" TFTRACKs/All: "<<match->TFTRACKs.size() <<"/"<<match->TFTRACKsAll.size()
// 		 <<" TFCANDs/All: "<<match->TFCANDs.size() <<"/"<<match->TFCANDsAll.size()
// 		 <<" GMTREGs/All: "<<match->GMTREGCANDs.size()<<"/"<<match->GMTREGCANDsAll.size()
// 		 <<"  GMTRegBest:"<<(match->GMTREGCANDBest.l1reg != NULL)
// 		 <<" GMTs/All: "<<match->GMTCANDs.size()<<"/"<<match->GMTCANDsAll.size()
// 		 <<" GMTBest:"<<(match->GMTCANDBest.l1gmt != NULL)
// 		 <<" L1EXTRAs/All: "<<match->L1EXTRAs.size()<<"/"<<match->L1EXTRAsAll.size()
// 		 <<" L1EXTRABest:"<<(match->L1EXTRABest.l1extra != NULL)<<std::endl;


    //------------------------------------------------------------------------------------------------
    //                               GEM SimHits
    //------------------------------------------------------------------------------------------------

//     SimTrackMatchManager gemcsc_match(*(match->strk), simVertices[match->strk->vertIndex()], gemMatchCfg_, iEvent, iSetup);
//     const GEMDigiMatcher& match_gem = gemcsc_match.gemDigis();
    
//     int match_has_gem = 0;
//     std::vector<int> match_gem_chambers;
//     auto gem_superch_ids = match_gem.superChamberIds();
//     for(auto d: gem_superch_ids) {
//       GEMDetId id(d);
//       bool odd = id.chamber() & 1;
//       auto digis = match_gem.digisInSuperChamber(d);
//       if (digis.size() > 0) {
// 	match_gem_chambers.push_back(id.chamber());
// 	if (odd) match_has_gem |= 1;
// 	  else     match_has_gem |= 2;
//       }
//     }
      
//     if (eta_gem_1b && match_has_gem) h_pt_gem_1b->Fill(stpt);
//     if (eta_gem_1b && match_has_gem && has_mplct_me1b) h_pt_lctgem_1b->Fill(stpt);
    
    
    //------------------------------------------------------------------------------------------------
    //                               ALCTs in the readout 
    //------------------------------------------------------------------------------------------------
    std::vector<MatchCSCMuL1::ALCT> readoutALCTCollection(match->ALCTsInReadOut());
    etrk_.st_n_alcts_readout.push_back(readoutALCTCollection.size());
    if (readoutALCTCollection.size()==0) {
      std::cout << "WARNING: ALCT Readout collection is empty" << std::endl;
      continue;
    }
    
//     const MatchCSCMuL1::ALCT* bestALCT(match->bestALCT(detId));
//     if (bestALCT==nullptr) { 
//       std::cout << "WARNING: No best ALCT" << std::endl;
//     }
    vint trk_csc_alct_valid;
    vint trk_csc_alct_quality; 
    vint trk_csc_alct_keywire;
    vint trk_csc_alct_bx;
    vint trk_csc_alct_trknmb;
    vint trk_csc_alct_fullbx;
    vint trk_csc_alct_isGood;
    vint trk_csc_alct_detId;
    vint trk_csc_alct_deltaOk;

    trk_csc_alct_valid.clear();
    trk_csc_alct_quality.clear();
    trk_csc_alct_keywire.clear();
    trk_csc_alct_bx.clear();
    trk_csc_alct_trknmb.clear();
    trk_csc_alct_fullbx.clear();
    trk_csc_alct_isGood.clear();
    trk_csc_alct_detId.clear();
    trk_csc_alct_deltaOk.clear();

    for (unsigned i=0; i<readoutALCTCollection.size();i++) {
      auto myALCT(readoutALCTCollection.at(i));
      if (myALCT.inReadOut()==0) continue;
      
      trk_csc_alct_valid.push_back(myALCT.trgdigi->isValid());
      trk_csc_alct_quality.push_back(myALCT.trgdigi->getQuality());
      trk_csc_alct_keywire.push_back(myALCT.trgdigi->getKeyWG());
      trk_csc_alct_bx.push_back(myALCT.getBX()-6);
      trk_csc_alct_trknmb.push_back(myALCT.trgdigi->getTrknmb());
      trk_csc_alct_fullbx.push_back(myALCT.trgdigi->getFullBX()-6);
      trk_csc_alct_isGood.push_back(minDeltaWire_ <= myALCT.deltaWire && myALCT.deltaWire <= maxDeltaWire_);
      trk_csc_alct_detId.push_back(myALCT.id);
      trk_csc_alct_deltaOk.push_back(myALCT.deltaOk);
    }
    etrk_.csc_alct_valid.push_back(trk_csc_alct_valid);
    etrk_.csc_alct_quality.push_back(trk_csc_alct_quality);
    etrk_.csc_alct_keywire.push_back(trk_csc_alct_keywire);
    etrk_.csc_alct_bx.push_back(trk_csc_alct_bx);
    etrk_.csc_alct_trknmb.push_back(trk_csc_alct_trknmb);
    etrk_.csc_alct_fullbx.push_back(trk_csc_alct_fullbx);
    etrk_.csc_alct_isGood.push_back(trk_csc_alct_isGood);
    etrk_.csc_alct_detId.push_back(trk_csc_alct_detId);
    etrk_.csc_alct_deltaOk.push_back(trk_csc_alct_deltaOk);

    //------------------------------------------------------------------------------------------------
    //                               CLCTs in the readout 
    //------------------------------------------------------------------------------------------------

    std::vector<MatchCSCMuL1::CLCT> readoutCLCTCollection(match->CLCTsInReadOut());
    etrk_.st_n_clcts_readout.push_back(readoutCLCTCollection.size());
    if (readoutCLCTCollection.size()==0) {
      std::cout << "WARNING: CLCT Readout collection is empty" << std::endl;
      continue;
    }
    
//     const MatchCSCMuL1::CLCT* bestCLCT(match->bestCLCT(detId));
//     if (bestCLCT==nullptr) { 
//       std::cout << "WARNING: No best CLCT" << std::endl;
//     }
    vint trk_csc_clct_valid;
    vint trk_csc_clct_pattern;
    vint trk_csc_clct_quality; 
    vint trk_csc_clct_bend;       
    vint trk_csc_clct_strip;      
    vint trk_csc_clct_bx;
    vint trk_csc_clct_trknmb;
    vint trk_csc_clct_fullbx;
    vint trk_csc_clct_isGood;
    vint trk_csc_clct_detId;
    vint trk_csc_clct_deltaOk;

    trk_csc_clct_valid.clear();
    trk_csc_clct_pattern.clear();
    trk_csc_clct_quality.clear(); 
    trk_csc_clct_bend.clear();       
    trk_csc_clct_strip.clear();      
    trk_csc_clct_bx.clear();
    trk_csc_clct_trknmb.clear();
    trk_csc_clct_fullbx.clear();
    trk_csc_clct_isGood.clear();
    trk_csc_clct_detId.clear();
    trk_csc_clct_deltaOk.clear();
     
    std::cout << "number of clcts: " << readoutCLCTCollection.size() << std::endl;
    for (unsigned i=0; i<readoutCLCTCollection.size();i++) {
      auto myCLCT(readoutCLCTCollection.at(i));
      if (myCLCT.inReadOut()==0) continue;
      
      trk_csc_clct_valid.push_back(myCLCT.trgdigi->isValid());
      trk_csc_clct_pattern.push_back(myCLCT.trgdigi->getPattern());
      trk_csc_clct_quality.push_back(myCLCT.trgdigi->getQuality());
      trk_csc_clct_bend.push_back(myCLCT.trgdigi->getBend());
      trk_csc_clct_strip.push_back(myCLCT.trgdigi->getStrip());
      trk_csc_clct_bx.push_back(myCLCT.getBX()-6);
      trk_csc_clct_trknmb.push_back(myCLCT.trgdigi->getTrknmb());
      trk_csc_clct_fullbx.push_back(myCLCT.trgdigi->getFullBX()-6);
      trk_csc_clct_isGood.push_back((abs(myCLCT.deltaStrip) <= minDeltaStrip_));
      trk_csc_clct_detId.push_back(myCLCT.id);
      trk_csc_clct_deltaOk.push_back(myCLCT.deltaOk);
    }
    etrk_.csc_clct_valid.push_back(trk_csc_clct_valid);
    etrk_.csc_clct_pattern.push_back(trk_csc_clct_pattern);
    etrk_.csc_clct_quality.push_back(trk_csc_clct_quality);
    etrk_.csc_clct_bend.push_back(trk_csc_clct_bend);
    etrk_.csc_clct_strip.push_back(trk_csc_clct_strip);
    etrk_.csc_clct_bx.push_back(trk_csc_clct_bx);
    etrk_.csc_clct_trknmb.push_back(trk_csc_clct_trknmb);
    etrk_.csc_clct_fullbx.push_back(trk_csc_clct_fullbx);
    etrk_.csc_clct_isGood.push_back(trk_csc_clct_isGood);
    etrk_.csc_clct_detId.push_back(trk_csc_clct_detId);
    etrk_.csc_clct_deltaOk.push_back(trk_csc_clct_deltaOk);

    //------------------------------------------------------------------------------------------------
    //                               LCTs in the readout 
    //------------------------------------------------------------------------------------------------

    std::vector<MatchCSCMuL1::LCT> readoutLCTCollection(match->LCTsInReadOut());
    etrk_.st_n_tmblcts_readout.push_back(readoutLCTCollection.size());
    if (readoutLCTCollection.size()==0) {
      std::cout << "WARNING: LCT Readout collection is empty" << std::endl;
      continue;
    }
    
    //     const MatchCSCMuL1::LCT* bestLCT(match->bestLCT(detId));
    //     if (bestLCT==nullptr) { 
    //       std::cout << "WARNING: No best LCT" << std::endl;
    //     }
    vint trk_csc_tmblct_valid;
    vint trk_csc_tmblct_pattern;
    vint trk_csc_tmblct_quality; 
    vint trk_csc_tmblct_bend;       
    vint trk_csc_tmblct_strip;      
    vint trk_csc_tmblct_bx;
    vint trk_csc_tmblct_trknmb;
    vint trk_csc_tmblct_isAlctGood;
    vint trk_csc_tmblct_isClctGood;
    vint trk_csc_tmblct_detId;
    vfloat trk_csc_tmblct_gemDPhi;
    vint trk_csc_tmblct_hasGEM;
    vint trk_csc_tmblct_mpclink;
    
    trk_csc_tmblct_valid.clear();
    trk_csc_tmblct_pattern.clear();
    trk_csc_tmblct_quality.clear(); 
    trk_csc_tmblct_bend.clear();       
    trk_csc_tmblct_strip.clear();      
    trk_csc_tmblct_bx.clear();
    trk_csc_tmblct_trknmb.clear();
    trk_csc_tmblct_isAlctGood.clear();
    trk_csc_tmblct_isClctGood.clear();
    trk_csc_tmblct_detId.clear();
    trk_csc_tmblct_gemDPhi.clear();
    trk_csc_tmblct_hasGEM.clear();
    trk_csc_tmblct_mpclink.clear();
    
    std::cout << "number of lcts: " << readoutLCTCollection.size() << std::endl;
    for (unsigned i=0; i<readoutLCTCollection.size();i++) {
      auto myLCT(readoutLCTCollection.at(i));
      if (myLCT.inReadOut()==0) continue;
      
      trk_csc_tmblct_valid.push_back(myLCT.trgdigi->isValid());
      trk_csc_tmblct_pattern.push_back(myLCT.trgdigi->getPattern());
      trk_csc_tmblct_quality.push_back(myLCT.trgdigi->getQuality());
      trk_csc_tmblct_bend.push_back(myLCT.trgdigi->getBend());
      trk_csc_tmblct_strip.push_back(myLCT.trgdigi->getStrip());
      trk_csc_tmblct_bx.push_back(myLCT.getBX()-6);
      trk_csc_tmblct_trknmb.push_back(myLCT.trgdigi->getTrknmb());
      trk_csc_tmblct_detId.push_back(myLCT.id);
      trk_csc_tmblct_isAlctGood.push_back(myLCT.alct->deltaOk);
      trk_csc_tmblct_isClctGood.push_back(myLCT.clct->deltaOk);
      trk_csc_tmblct_gemDPhi.push_back(myLCT.trgdigi->getGEMDPhi());
      trk_csc_tmblct_hasGEM.push_back(myLCT.trgdigi->hasGEM());
      trk_csc_tmblct_mpclink.push_back(myLCT.trgdigi->getMPCLink());
    }
    etrk_.csc_tmblct_valid.push_back(trk_csc_tmblct_valid);
    etrk_.csc_tmblct_pattern.push_back(trk_csc_tmblct_pattern);
    etrk_.csc_tmblct_quality.push_back(trk_csc_tmblct_quality); 
    etrk_.csc_tmblct_bend.push_back(trk_csc_tmblct_bend);       
    etrk_.csc_tmblct_strip.push_back(trk_csc_tmblct_strip);      
    etrk_.csc_tmblct_bx.push_back(trk_csc_tmblct_bx);
    etrk_.csc_tmblct_trknmb.push_back(trk_csc_tmblct_trknmb);
    etrk_.csc_tmblct_isAlctGood.push_back(trk_csc_tmblct_isAlctGood);
    etrk_.csc_tmblct_isClctGood.push_back(trk_csc_tmblct_isClctGood);
    etrk_.csc_tmblct_detId.push_back(trk_csc_tmblct_detId);
    etrk_.csc_tmblct_gemDPhi.push_back(trk_csc_tmblct_gemDPhi);
    etrk_.csc_tmblct_hasGEM.push_back(trk_csc_tmblct_hasGEM);
    etrk_.csc_tmblct_mpclink.push_back(trk_csc_tmblct_mpclink);
  }

  tree_eff_->Fill();
  
}


// ------------ method called once each job just before starting event loop  ------------
void 
SimpleMuon::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SimpleMuon::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
SimpleMuon::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
SimpleMuon::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
SimpleMuon::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
SimpleMuon::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SimpleMuon::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


// ================================================================================================
void 
SimpleMuon::propagateToCSCStations(MatchCSCMuL1 *match)
{
  TrajectoryStateOnSurface tsos;

  // z planes
  const int endcap((match->strk->momentum().eta() >= 0) ? 1 : -1);
  const double zME11(endcap*585.);
  const double zME1(endcap*615.);
  const double zME2(endcap*830.);
  const double zME3(endcap*935.);
  
  // extrapolate to ME1/1 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME11);
  if (tsos.isValid()) match->pME11 = math::XYZVectorD(tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z());

  // extrapolate to ME1 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME1);
  if (tsos.isValid()) match->pME1 = math::XYZVectorD(tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z());
     
  // extrapolate to ME2 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME2);
  if (tsos.isValid()) match->pME2 = math::XYZVectorD(tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z());

  // extrapolate to ME3 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME3);
  if (tsos.isValid()) match->pME3 = math::XYZVectorD(tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z());
}

// ================================================================================================
// propagate simtrack to certain Z position
TrajectoryStateOnSurface
SimpleMuon::propagateSimTrackToZ(const SimTrack *track, const SimVertex *vtx, double z)
{
  const Plane::PlanePointer myPlane(Plane::build(Plane::PositionType(0, 0, z), Plane::RotationType()));
  const GlobalPoint  innerPoint(vtx->position().x(),  vtx->position().y(),  vtx->position().z());
  const GlobalVector innerVec  (track->momentum().x(),  track->momentum().y(),  track->momentum().z());
  const FreeTrajectoryState stateStart(innerPoint, innerVec, track->charge(), &*theBField);
  
  TrajectoryStateOnSurface stateProp(propagatorAlong->propagate(stateStart, *myPlane));
  if (!stateProp.isValid()) stateProp = propagatorOpposite->propagate(stateStart, *myPlane);

  return stateProp;
}


// ================================================================================================
// Matching of SimHits that were created by SimTrack
void 
SimpleMuon::matchSimTrack2SimHits(MatchCSCMuL1 * match, 
				  const edm::SimTrackContainer & simTracks, 
				  const edm::SimVertexContainer & simVertices, 
				  const edm::PSimHitContainer * allCSCSimHits)
{
  // collect all ID of muon SimTrack children
  match->familyIds = fillSimTrackFamilyIds(match->strk->trackId(), simTracks, simVertices);

  // match SimHits to SimTracks
  std::vector<PSimHit> matchingSimHits(hitsFromSimTrack(match->familyIds, theCSCSimHitMap));

  std::cout << "number of matching simhits: " << matchingSimHits.size() << std::endl;

  // add the matching simhits to the matching object
  for (unsigned i=0; i<matchingSimHits.size();i++) 
  {
    // only the good chambers?
    if (goodChambersOnly_) 
    {
      // skip the bad chambers
      if (theStripConditions->isInBadChamber(CSCDetId(matchingSimHits[i].detUnitId()))) continue;
    }
    match->addSimHit(matchingSimHits[i]);
  }

  // checks
  unsigned stNhist = 0;
  for (edm::PSimHitContainer::const_iterator hit = allCSCSimHits->begin(); hit != allCSCSimHits->end(); ++hit) 
  {
    // track id has to match
    if (hit->trackId() != match->strk->trackId()) continue;

    // select only certain regions of CSC
    const CSCDetId chId(hit->detUnitId());
    if ( chId.station() == 1 && chId.ring() == 4 && !doME1a_) continue;
    
    stNhist++;
  }
  if (doStrictSimHitToTrackMatch_ && stNhist != match->simHits.size()) 
  {
    std::cout <<" ALARM!!! matchSimTrack2SimHits: stNhist != stHits.size()  ---> "<<stNhist <<" != "<<match->simHits.size()<<std::endl;
    //    stNhist = 0;
//     if (debugALLEVENT) 
//       for (edm::PSimHitContainer::const_iterator hit = allCSCSimHits->begin();  hit != allCSCSimHits->end();  ++hit) 
// 	if (hit->trackId() == match->strk->trackId()) 
// 	  {
// 	    CSCDetId chId(hit->detUnitId());
// 	    if ( !(chId.station() == 1 && chId.ring() == 4 && !doME1a_) ) 
// 	      std::cout<<"   "<<chId<<"  "<<(*hit)<<" "<<hit->momentumAtEntry()<<" "<<hit->energyLoss()<<" "<<hit->particleType()<<" "<<hit->trackId()<<std::endl;
// 	  }
  }
  
}


// ================================================================================================
/*
 * Get the family of child tracks belonging to this track
 *
 * @param id            Track Id of the parent track
 * @param simTracks     The tracks
 * @param simVertices   The vertices
 * @return              The vector with track ids of the child tracks
 *
 */
std::vector<unsigned> 
SimpleMuon::fillSimTrackFamilyIds(unsigned id, const edm::SimTrackContainer & simTracks, 
				  const edm::SimVertexContainer & simVertices)
{
  const bool debug = true;

  std::vector<unsigned> result;
  result.push_back(id);

  if (doStrictSimHitToTrackMatch_) return result;
  
  // get all children for this simtrack
  for (edm::SimTrackContainer::const_iterator track = simTracks.begin(); track != simTracks.end(); ++track)
  {
    SimTrack lastTr = *track;
    bool ischild = 0;

    // keep looping until all children are found
    while (1)
    {
      // track has no vertex
      if (lastTr.noVertex()) break;
      
      // track vertex has no parent
      if (simVertices[lastTr.vertIndex()].noParent()) break;
      
      // get the parent id for the good track
      const unsigned parentId(simVertices[lastTr.vertIndex()].parentIndex());
      if (parentId == id) 
      {
	ischild = 1; 
	break; 
      }
      
      // find the location of the parent track  
      std::map<unsigned, unsigned >::iterator association(trkId2Index.find(parentId));

      // track not in trkId2Index
      if (association == trkId2Index.end()) break;
      
      lastTr = simTracks[association->second];
    }
    // add the track if it is a child
    if (ischild) result.push_back(track->trackId());
  }

  if (debug) std::cout<<"  --- family size = " << result.size() <<std::endl;
  return result;
}


// ================================================================================================
std::vector<PSimHit> 
SimpleMuon::hitsFromSimTrack(std::vector<unsigned> ids, SimHitAnalysis::PSimHitMap &hitMap)
{
  std::vector<PSimHit> result;
  for (size_t id = 0; id < ids.size(); ++id)
  {
    const std::vector<PSimHit> resultd(hitsFromSimTrack(ids[id], hitMap));
    result.insert(result.end(), resultd.begin(), resultd.end());
  }
  return result;
}


// ================================================================================================
std::vector<PSimHit> 
SimpleMuon::hitsFromSimTrack(unsigned id, SimHitAnalysis::PSimHitMap &hitMap)
{
  std::vector<PSimHit> result;
  const std::vector<int> detIds(hitMap.detsWithHits());

  for (size_t di = 0; di < detIds.size(); ++di)
  {
    const std::vector<PSimHit> resultd(hitsFromSimTrack(id, detIds[di], hitMap));
    result.insert(result.end(), resultd.begin(), resultd.end());
  }
  return result;
}


// ================================================================================================
std::vector<PSimHit> 
SimpleMuon::hitsFromSimTrack(unsigned id, int detId, SimHitAnalysis::PSimHitMap &hitMap)
{
  std::vector<PSimHit> result;

  const CSCDetId chId(detId);
  if ( chId.station() == 1 && chId.ring() == 4 && !doME1a_) return result;

  // get the simhits for this detId
  const edm::PSimHitContainer hits(hitMap.hits(detId));
  
  for(size_t h = 0; h< hits.size(); ++h) 
  {
    // add all simhits for which the track id corresponds to the required track id
    if(hits[h].trackId() == id)
    {
      result.push_back(hits[h]);
    }
  }
  return result;
}


// ================================================================================================
void
SimpleMuon::matchSimTrack2ALCTs(MatchCSCMuL1 *match, 
				const edm::PSimHitContainer* allCSCSimHits, 
				const CSCALCTDigiCollection *alcts, 
				const CSCWireDigiCollection* wiredc )
{
  // tool for matching SimHits to ALCTs
  //CSCAnodeLCTAnalyzer alct_analyzer;
  //alct_analyzer.setDebug();
  //alct_analyzer.setGeometry(cscGeometry);

  if (debugALCT) std::cout<<"--- ALCT-SimHits ---- begin for trk "<<match->strk->trackId()<<std::endl;
  
  // map < detId, ALCTCollection > 
  std::map<int, std::vector<CSCALCTDigi> > checkNALCT;
  checkNALCT.clear();

  match->ALCTs.clear();
  for (CSCALCTDigiCollection::DigiRangeIterator adetUnitIt = alcts->begin(); adetUnitIt != alcts->end(); adetUnitIt++)
  {
    const CSCDetId& id = (*adetUnitIt).first;
    const CSCALCTDigiCollection::Range& range = (*adetUnitIt).second;
    int nm=0;
    
    //if (id.station()==1&&id.ring()==2) debugALCT=1;
    // ME1/a has ring number 4???
    CSCDetId id1a(id.endcap(),id.station(),4,id.chamber(),0);
    
    for (CSCALCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; digiIt++) 
    {
     checkNALCT[id.rawId()].push_back(*digiIt);
     nm++;
     
     // ALCT is not valid
     if (!(*digiIt).isValid()) continue;
     
     // how to perform the matching?
     const bool me1a_all(defaultME1a && id.station()==1 && id.ring()==1 && (*digiIt).getKeyWG() <= 15);
     const bool me1a_no_overlap(me1a_all && (*digiIt).getKeyWG() < 10);
     
     std::vector<PSimHit> trackHitsInChamber = match->chamberHits(id.rawId());
     std::vector<PSimHit> trackHitsInChamber1a;
     if (me1a_all) trackHitsInChamber1a = match->chamberHits(id1a.rawId());
     
     // no point to do any matching here
     if (trackHitsInChamber.size() + trackHitsInChamber1a.size() == 0 )
     {
       if (debugALCT) std::cout<<"raw ID "<<id.rawId()<<" "<<id<<"  #"<<nm<<"   no SimHits in chamber from this SimTrack!"<<std::endl;
       continue;
     }
     
     // ALCT BX is not valid
     if ( (*digiIt).getBX()-6 < minBX_ || (*digiIt).getBX()-6 > maxBX_ )
     {
       if (debugALCT) std::cout<<"discarding BX = "<< (*digiIt).getBX()-6 <<std::endl;
       continue;
     }
     
     std::vector<CSCAnodeLayerInfo> alctInfo;
     //std::vector<CSCAnodeLayerInfo> alctInfo = alct_analyzer.getSimInfo(*digiIt, id, wiredc, allCSCSimHits);
     std::vector<PSimHit> matchedHits;
     unsigned nmhits = matchCSCAnodeHits(alctInfo, matchedHits);
     
     MatchCSCMuL1::ALCT malct(match);
     malct.trgdigi = &*digiIt;
     malct.layerInfo = alctInfo;
     malct.simHits = matchedHits;
     malct.id = id;
     malct.nHitsShared = 0;
     calculate2DStubsDeltas(match, malct);
     malct.deltaOk = (minDeltaWire_ <= malct.deltaWire) & (malct.deltaWire <= maxDeltaWire_);
     
     std::vector<CSCAnodeLayerInfo> alctInfo1a;
     std::vector<PSimHit> matchedHits1a;
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
     
     if (debugALCT) std::cout<<"raw ID "<<id.rawId()<<" "<<id<<"  #"<<nm<<"    NTrackHitsInChamber  nmhits  alctInfo.size  diff  "
			     <<trackHitsInChamber.size()<<" "<<nmhits<<" "<<alctInfo.size()<<"  "
			     << nmhits-alctInfo.size() <<std::endl
			     << "  "<<(*digiIt)<<"  DW="<<malct.deltaWire<<" eta="<<malct.eta;
     
     if (nmhits + nmhits1a > 0)
     {
       //if (debugALCT) std::cout<<"  --- matched to ALCT hits: "<<std::endl;
       
       int nHitsMatch = 0;
       for (unsigned i=0; i<nmhits;i++)
	 {
	   //if (debugALCT) std::cout<<"   "<<matchedHits[i]<<" "<<matchedHits[i].exitPoint()<<"  "
	   //                   <<matchedHits[i].momentumAtEntry()<<" "<<matchedHits[i].energyLoss()<<" "
	   //                   <<matchedHits[i].particleType()<<" "<<matchedHits[i].trackId();
	   //bool wasmatch = 0;
	   for (unsigned j=0; j<trackHitsInChamber.size(); j++)
	     if ( compareSimHits ( matchedHits[i], trackHitsInChamber[j] ) )
	       {
		 nHitsMatch++;
		 //wasmatch = 1;
	       }
	   //if (debugALCT)  {if (wasmatch) std::cout<<" --> match!"<<std::endl;  else std::cout<<std::endl;}
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
     else if (debugALCT) std::cout<< "  +++ ALCT warning: no simhits for its digi found!\n";
     
     if (debugALCT) std::cout<<"  nHitsShared="<<malct.nHitsShared<<std::endl;
     
     if(matchAllTrigPrimitivesInChamber_)
       {
	 // if specified, try DY match
	 bool dymatch = 0;
	 if ( fabs(malct.deltaY)<= minDeltaYAnode_ )
	   {
	     if (debugALCT)  for (unsigned i=0; i<trackHitsInChamber.size();i++)
	       std::cout<<"   DY match: "<<trackHitsInChamber[i]<<" "<<trackHitsInChamber[i].exitPoint()<<"  "
			<<trackHitsInChamber[i].momentumAtEntry()<<" "<<trackHitsInChamber[i].energyLoss()<<" "
			<<trackHitsInChamber[i].particleType()<<" "<<trackHitsInChamber[i].trackId()<<std::endl;
	     
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
	       std::cout<<"   chamber match: "<<trackHitsInChamber[i]<<" "<<trackHitsInChamber[i].exitPoint()<<"  "
			<<trackHitsInChamber[i].momentumAtEntry()<<" "<<trackHitsInChamber[i].energyLoss()<<" "
			<<trackHitsInChamber[i].particleType()<<" "<<trackHitsInChamber[i].trackId()<<std::endl;
	     
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
	   if (debugALCT)  std::cout<<" --> shared hits match!"<<std::endl;
	   match->ALCTs.push_back(malct);
	 }
	 if (me1a_all && malct1a.nHitsShared >= minNHitsShared_) {
	   if (debugALCT)  std::cout<<" --> shared hits match!"<<std::endl;
	   match->ALCTs.push_back(malct);
	 }
       }
     
     // else proceed with deltaWire matching:
     if (!me1a_no_overlap && minDeltaWire_ <= malct.deltaWire && malct.deltaWire <= maxDeltaWire_){
       if (debugALCT)  std::cout<<" --> deltaWire match!"<<std::endl;
       match->ALCTs.push_back(malct);
     }
     
     // special case of default emulator with puts all ME11 alcts into ME1b
     // only for deltaWire matching!
     if (me1a_all && minDeltaWire_ <= malct1a.deltaWire && malct1a.deltaWire <= maxDeltaWire_){
       if (debugALCT)  std::cout<<" --> deltaWire match!"<<std::endl;
       match->ALCTs.push_back(malct);
     }
    }
    //debugALCT=0;
  } // loop CSCALCTDigiCollection
  
  if (debugALCT) for(std::map<int, std::vector<CSCALCTDigi> >::const_iterator mapItr = checkNALCT.begin(); mapItr != checkNALCT.end(); ++mapItr)
    if (mapItr->second.size()>2) {
      CSCDetId idd(mapItr->first);
      std::cout<<"~~~~ checkNALCT WARNING! nALCT = "<<mapItr->second.size()<<" in ch "<< mapItr->first<<" "<<idd<<std::endl;
      for (unsigned i=0; i<mapItr->second.size();i++) std::cout<<"~~~~~~ ALCT "<<i<<" "<<(mapItr->second)[i]<<std::endl;
    }
  
  if (debugALCT) std::cout<<"--- ALCT-SimHits ---- end"<<std::endl;
}


// ================================================================================================
unsigned
SimpleMuon::matchCSCAnodeHits(const std::vector<CSCAnodeLayerInfo>& allLayerInfo, 
			      std::vector<PSimHit> &matchedHit) 
{
  // Match Anode hits in a chamber to SimHits

  // It first tries to look for the SimHit in the key layer.  If it is
  // unsuccessful, it loops over all layers and looks for an associated
  // hits in any one of the layers.  

  //  int fdebug = 0;

  int nhits=0;
  matchedHit.clear();
  
  std::vector<CSCAnodeLayerInfo>::const_iterator pli;
  for (pli = allLayerInfo.begin(); pli != allLayerInfo.end(); pli++) 
  {
    // For ALCT search, the key layer is the 3rd one, counting from 1.
    if (pli->getId().layer() == CSCConstants::KEY_ALCT_LAYER) 
    {
      std::vector<PSimHit> thisLayerHits = pli->getSimHits();
      if (thisLayerHits.size() > 0) 
      {
	// There can be only one RecDigi (and therefore only one SimHit) in a key layer.
	if (thisLayerHits.size() != 1) 
	{
	  std::cout<< "+++ Warning in matchCSCAnodeHits: " << thisLayerHits.size()
		   << " SimHits in key layer " << CSCConstants::KEY_ALCT_LAYER
		   << "! +++ \n";
	  for (unsigned i = 0; i < thisLayerHits.size(); i++) 
	    std::cout<<"      SimHit # " << i <<": "<< thisLayerHits[i] << "\n";
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
	  std::vector<PSimHit> thisLayerHits = pli->getSimHits();
	  // There can be several RecDigis and several SimHits in a nonkey layer.
	  //if (thisLayerHits.size() != 1) 
	  //{
	  //  std::cout<< "+++ Warning in matchCSCAnodeHits: " << thisLayerHits.size()
	  //      << " SimHits in layer " << pli->getId().layer() <<" detID "<<pli->getId().rawId()
	  //      << "! +++ \n";
	  //  for (unsigned i = 0; i < thisLayerHits.size(); i++)
	  //    std::cout<<"   "<<thisLayerHits[i]<<" "<<thisLayerHits[i].exitPoint()<<"  "<<thisLayerHits[i].momentumAtEntry()
	  //        <<" "<<thisLayerHits[i].energyLoss()<<" "<<thisLayerHits[i].particleType()<<" "<<thisLayerHits[i].trackId()<<std::endl;
	  //}
	  matchedHit.insert(matchedHit.end(), thisLayerHits.begin(), thisLayerHits.end());
	  nhits += thisLayerHits.size();
	}
    }
  
  return nhits;
}


// ================================================================================================
bool 
SimpleMuon::compareSimHits(PSimHit &sh1, PSimHit &sh2)
{
  int fdebug = 0;

  if (fdebug && sh1.detUnitId() == sh2.detUnitId())  {
    std::cout<<" compare hits in "<<sh1.detUnitId()<<": "<<std::endl;

    std::cout<<"   "<<sh1<<" "<<sh1.exitPoint()<<" "<<sh1.momentumAtEntry()<<" "<<sh1.energyLoss()
	     <<" "<<sh1.particleType()<<" "<<sh1.trackId()<<" |"<<sh1.entryPoint().mag()<<" "
	     <<sh1.exitPoint().mag()<<" "<<sh1.momentumAtEntry().mag()<<std::endl;

    std::cout<<"   "<<sh2<<" "<<sh2.exitPoint()<<" "<<sh2.momentumAtEntry()<<" "<<sh2.energyLoss()
	     <<" "<<sh2.particleType()<<" "<<sh2.trackId()<<" |"<<sh2.entryPoint().mag()<<" "
	     <<sh2.exitPoint().mag()<<" "<<sh2.momentumAtEntry().mag()<<std::endl;

  }
  bool returnValue = true;
  if (sh1.detUnitId() != sh2.detUnitId()) returnValue = false;
  if (sh1.trackId() != sh2.trackId()) returnValue = false;
  if (sh1.particleType() != sh2.particleType()) returnValue = false;
  if (sh1.entryPoint().mag() != sh2.entryPoint().mag()) {
    //std::cout<<"  !!ent "<< sh1.entryPoint().mag() - sh2.entryPoint().mag()<<std::endl;
    returnValue = false;
  }
  if (sh1.exitPoint().mag() != sh2.exitPoint().mag()) {
    //std::cout<<"  !!exi "<< sh1.exitPoint().mag() - sh2.exitPoint().mag()<<std::endl;
    returnValue = false;
  }
  if (fabs(sh1. momentumAtEntry().mag() - sh2. momentumAtEntry().mag() ) > 0.0001) {
    //std::cout<<"  !!mom "<< sh1.momentumAtEntry().mag() - sh2.momentumAtEntry().mag()<<std::endl;
    returnValue = false;
  }
  if (sh1.tof() != sh2.tof()){
    //std::cout<<" !!tof "<<sh1.tof()-sh2.tof()<<std::endl;
    returnValue = false;
  }
  if (sh1.energyLoss() != sh2.energyLoss()) {
    //std::cout<<"  !!los "<<sh1.energyLoss() - sh2.energyLoss() <<std::endl;
    returnValue = false;
  }
  return returnValue;
}


// ================================================================================================
int 
SimpleMuon::calculate2DStubsDeltas(MatchCSCMuL1 *match, MatchCSCMuL1::ALCT &alct)
{
  //** fit muon's hits to a 2D linear stub in a chamber :
  //   wires:   work in 2D plane going through z axis :
  //     z becomes a new x axis, and new y is perpendicular to it
  //    (using SimTrack's position point as well when there is <= 2 mu's hits in chamber)

  int fdebug = 0;

  alct.deltaPhi = M_PI;
  alct.deltaY = 9999.;
  
  const CSCDetId keyID(alct.id.rawId()+CSCConstants::KEY_ALCT_LAYER);
  const CSCLayer* csclayer(cscGeometry->layer(keyID));
  //int hitWireG = csclayer->geometry()->wireGroup(csclayer->geometry()->nearestWire(cLP));
  const int hitWireG(match->wireGroupAndStripInChamber(alct.id.rawId()).first);
  if (hitWireG<0) return 1;

  alct.deltaWire = hitWireG - alct.trgdigi->getKeyWG() - 1;
  alct.mcWG = hitWireG;

  const GlobalPoint gpcwg(csclayer->centerOfWireGroup( alct.trgdigi->getKeyWG()+1));
  const math::XYZVectorD vcwg(gpcwg.x(), gpcwg.y(), gpcwg.z());
  alct.eta = vcwg.eta();

  if (fdebug) std::cout<<"    hitWireG = "<<hitWireG<<"    alct.KeyWG = "<<alct.trgdigi->getKeyWG()<<"    deltaWire = "<<alct.deltaWire<<std::endl;

  return 0;
}

// ================================================================================================
// Returns chamber type (0-9) according to the station and ring number
int 
SimpleMuon::getCSCType(const CSCDetId &id) 
{
  int type = -999;

  if (id.station() == 1) 
  {
    type = (id.triggerCscId()-1)/3;
    if (id.ring() == 4) 
    {
      type = 3;
    }
  }
  else 
  { 
    // stations 2-4
    type = 3 + id.ring() + 2*(id.station()-2);
  }
  assert(type >= 0 && type < CSC_TYPES); // include ME4/2
  return type;
}


// ================================================================================================
void
SimpleMuon::matchSimTrack2CLCTs(MatchCSCMuL1 *match, 
				const edm::PSimHitContainer* allCSCSimHits, 
				const CSCCLCTDigiCollection *clcts, 
				const CSCComparatorDigiCollection* compdc )
{
  // tool for matching SimHits to CLCTs
  //CSCCathodeLCTAnalyzer clct_analyzer;
  //clct_analyzer.setDebug();
  //clct_analyzer.setGeometry(cscGeometry);

  if (debugCLCT) std::cout<<"--- CLCT-SimHits ---- begin for trk "<<match->strk->trackId()<<std::endl;
  //static const int key_layer = 4; //CSCConstants::KEY_CLCT_LAYER

  std::map<int, std::vector<CSCCLCTDigi> > checkNCLCT;
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

	  std::vector<PSimHit> trackHitsInChamber = match->chamberHits(cid.rawId());

	  if (trackHitsInChamber.size()==0) // no point to do any matching here
	    {
	      if (debugCLCT) std::cout<<"raw ID "<<cid.rawId()<<" "<<cid<<"  #"<<nm<<"   no SimHits in chamber from this SimTrack!"<<std::endl;
	      continue;
	    }

	  if ( (*digiIt).getBX()-5 < minBX_ || (*digiIt).getBX()-7 > maxBX_ )
	    {
	      if (debugCLCT) std::cout<<"discarding BX = "<< (*digiIt).getBX()-6 <<std::endl;
	      continue;
	    }

	  // don't use it anymore: replace with a dummy
	  std::vector<CSCCathodeLayerInfo> clctInfo;
	  //std::vector<CSCCathodeLayerInfo> clctInfo = clct_analyzer.getSimInfo(*digiIt, cid, compdc, allCSCSimHits);

	  std::vector<PSimHit> matchedHits;
	  unsigned nmhits = matchCSCCathodeHits(clctInfo, matchedHits);

	  MatchCSCMuL1::CLCT mclct(match);
	  mclct.trgdigi = &*digiIt;
	  mclct.layerInfo = clctInfo;
	  mclct.simHits = matchedHits;
	  mclct.id = cid;
	  mclct.nHitsShared = 0;
	  calculate2DStubsDeltas(match, mclct);
	  mclct.deltaOk = (abs(mclct.deltaStrip) <= minDeltaStrip_);

	  if (debugCLCT) std::cout<<"raw ID "<<cid.rawId()<<" "<<cid<<"  #"<<nm<<"    NTrackHitsInChamber  nmhits  clctInfo.size  diff  "
			     <<trackHitsInChamber.size()<<" "<<nmhits<<" "<<clctInfo.size()<<"  "
			     << nmhits-clctInfo.size() <<std::endl
			     << "  "<<(*digiIt)<<"  DS="<<mclct.deltaStrip<<" phi="<<mclct.phi;

	  if (nmhits > 0)
	    {
	      //if (debugCLCT) std::cout<<"  --- matched to CLCT hits: "<<std::endl;

	      int nHitsMatch = 0;
	      for (unsigned i=0; i<nmhits;i++)
		{
		  //if (debugCLCT) std::cout<<"   "<<matchedHits[i]<<" "<<matchedHits[i].exitPoint()<<"  "
		  //                   <<matchedHits[i].momentumAtEntry()<<" "<<matchedHits[i].energyLoss()<<" "
		  //                   <<matchedHits[i].particleType()<<" "<<matchedHits[i].trackId();
		  //bool wasmatch = 0;
		  for (unsigned j=0; j<trackHitsInChamber.size(); j++)
		    if ( compareSimHits ( matchedHits[i], trackHitsInChamber[j] ) )
		      {
			nHitsMatch++;
			//wasmatch = 1;
		      }
		  //if (debugCLCT)  {if (wasmatch) std::cout<<" --> match!"<<std::endl;  else std::cout<<std::endl;}
		}
	      mclct.nHitsShared = nHitsMatch;
	    }
	  else if (debugCLCT) std::cout<< "  +++ CLCT warning: no simhits for its digi found!\n";

	  if (debugCLCT) std::cout<<"  nHitsShared="<<mclct.nHitsShared<<std::endl;

	  if(matchAllTrigPrimitivesInChamber_)
	    {
	      if ( fabs(mclct.deltaY)<= minDeltaYCathode_)
		{
		  if (debugCLCT)  for (unsigned i=0; i<trackHitsInChamber.size();i++)
				    std::cout<<"   DY match: "<<trackHitsInChamber[i]<<" "<<trackHitsInChamber[i].exitPoint()<<"  "
					<<trackHitsInChamber[i].momentumAtEntry()<<" "<<trackHitsInChamber[i].energyLoss()<<" "
					<<trackHitsInChamber[i].particleType()<<" "<<trackHitsInChamber[i].trackId()<<std::endl;
  
		  match->CLCTs.push_back(mclct);
		  continue;
		}
	      if ( minDeltaYCathode_ < 0  )
		{
		  if (debugCLCT)  for (unsigned i=0; i<trackHitsInChamber.size();i++)
				    std::cout<<"   chamber match: "<<trackHitsInChamber[i]<<" "<<trackHitsInChamber[i].exitPoint()<<"  "
					<<trackHitsInChamber[i].momentumAtEntry()<<" "<<trackHitsInChamber[i].energyLoss()<<" "
					<<trackHitsInChamber[i].particleType()<<" "<<trackHitsInChamber[i].trackId()<<std::endl;
  
		  match->CLCTs.push_back(mclct);
		  continue;
		}
	      continue;
	    }
	  // else proceed with hit2hit matching:
 
	  if (mclct.nHitsShared >= minNHitsShared_) {
	    if (debugCLCT)  std::cout<<" --> shared hits match!"<<std::endl;
	    match->CLCTs.push_back(mclct);
	  }

	  // else proceed with hit2hit matching:
	  if (minNHitsShared_>=0)
	    {
	      if (mclct.nHitsShared >= minNHitsShared_) {
		if (debugCLCT)  std::cout<<" --> shared hits match!"<<std::endl;
		match->CLCTs.push_back(mclct);
	      }
	    }

	  // else proceed with deltaStrip matching:
	  if (abs(mclct.deltaStrip) <= minDeltaStrip_) {
	    if (debugCLCT)  std::cout<<" --> deltaStrip match!"<<std::endl;
	    match->CLCTs.push_back(mclct);
	  }
	}
      //debugCLCT=0;

    } // loop CSCCLCTDigiCollection

  if (debugCLCT) for(std::map<int, std::vector<CSCCLCTDigi> >::const_iterator mapItr = checkNCLCT.begin(); mapItr != checkNCLCT.end(); ++mapItr)
		   if (mapItr->second.size()>2) {
		     CSCDetId idd(mapItr->first);
		     std::cout<<"~~~~ checkNCLCT WARNING! nCLCT = "<<mapItr->second.size()<<" in ch "<< mapItr->first<<" "<<idd<<std::endl;
		     for (unsigned i=0; i<mapItr->second.size();i++) std::cout<<"~~~~~~ CLCT "<<i<<" "<<(mapItr->second)[i]<<std::endl;
		   }
  
  if (debugCLCT) std::cout<<"--- CLCT-SimHits ---- end"<<std::endl;
}


// ================================================================================================
unsigned
SimpleMuon::matchCSCCathodeHits(const std::vector<CSCCathodeLayerInfo>& allLayerInfo, 
				std::vector<PSimHit> &matchedHit) 
{
  // It first tries to look for the SimHit in the key layer.  If it is
  // unsuccessful, it loops over all layers and looks for an associated
  // hits in any one of the layers.  

  //  int fdebug = 0;

  //  static const int key_layer = 4; //CSCConstants::KEY_CLCT_LAYER
  static const int key_layer = CSCConstants::KEY_CLCT_LAYER;
 
  int nhits=0;
  matchedHit.clear();
  
  std::vector<CSCCathodeLayerInfo>::const_iterator pli;
  for (pli = allLayerInfo.begin(); pli != allLayerInfo.end(); pli++) 
    {
      // For ALCT search, the key layer is the 3rd one, counting from 1.
      if (pli->getId().layer() == key_layer) 
	{
	  std::vector<PSimHit> thisLayerHits = pli->getSimHits();
	  if (thisLayerHits.size() > 0) 
	    {
	      // There can be only one RecDigi (and therefore only one SimHit) in a key layer.
	      if (thisLayerHits.size() != 1) 
		{
		  std::cout<< "+++ Warning in matchCSCCathodeHits: " << thisLayerHits.size()
		      << " SimHits in key layer " << key_layer
		      << "! +++ \n";
		  for (unsigned i = 0; i < thisLayerHits.size(); i++) 
		    std::cout<<"      SimHit # " << i <<": "<< thisLayerHits[i] << "\n";
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
	  std::vector<PSimHit> thisLayerHits = pli->getSimHits();
	  // There can be several RecDigis and several SimHits in a nonkey layer.
	  //if (thisLayerHits.size() != 1) 
	  //{
	  //  std::cout<< "+++ Warning in matchCSCCathodeHits: " << thisLayerHits.size()
	  //      << " SimHits in layer " << pli->getId().layer() <<" detID "<<pli->getId().rawId()
	  //      << "! +++ \n";
	  //  for (unsigned i = 0; i < thisLayerHits.size(); i++)
	  //    std::cout<<"   "<<thisLayerHits[i]<<" "<<thisLayerHits[i].exitPoint()<<"  "<<thisLayerHits[i].momentumAtEntry()
	  //        <<" "<<thisLayerHits[i].energyLoss()<<" "<<thisLayerHits[i].particleType()<<" "<<thisLayerHits[i].trackId()<<std::endl;
	  //}
	  matchedHit.insert(matchedHit.end(), thisLayerHits.begin(), thisLayerHits.end());
	  nhits += thisLayerHits.size();
	}
    }
  
  return nhits;
}


// ================================================================================================
int 
SimpleMuon::calculate2DStubsDeltas(MatchCSCMuL1 *match, MatchCSCMuL1::CLCT &clct)
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

  // fit a 2D stub from SimHits matched to a digi


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

  if (fdebug) std::cout<<"    hitStrip = "<<hitStrip<<"    alct.KeyStrip = "<<clct.trgdigi->getKeyStrip()<<"    deltaStrip = "<<clct.deltaStrip<<std::endl;

  return 0;
}


//define this as a plug-in
DEFINE_FWK_MODULE(SimpleMuon);
