#ifndef SimMuL1_MatchCSCMuL1_h
#define SimMuL1_MatchCSCMuL1_h

// system include files
#include <vector>
#include <map>

#include <FWCore/Framework/interface/ESHandle.h>

#include <DataFormats/L1Trigger/interface/L1MuonParticleFwd.h>
#include <DataFormats/L1Trigger/interface/L1MuonParticle.h>

#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>
#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>

#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

#include <L1Trigger/CSCTriggerPrimitives/test/CSCAnodeLCTAnalyzer.h>
#include <L1Trigger/CSCTriggerPrimitives/test/CSCCathodeLCTAnalyzer.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>

#include <DataFormats/L1CSCTrackFinder/interface/L1Track.h>
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h>

#include <DataFormats/L1Trigger/interface/L1MuonParticle.h>

#include <CondFormats/L1TObjects/interface/L1MuTriggerScales.h>
#include <CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h>


//
// class decleration
//
class CSCGeometry;


class MatchCSCMuL1 
{
public:
  MatchCSCMuL1(const SimTrack  *s, const SimVertex *v, const CSCGeometry* g);
  ~MatchCSCMuL1(){};
  
  // SimTrack itself
  const SimTrack  *strk;
  const SimVertex *svtx;

  const CSCGeometry* cscGeometry;
  const GEMGeometry* gemGeometry;
  
  // positions extrapolated to different stations
  math::XYZVectorD pGE11;
  math::XYZVectorD pME11;
  math::XYZVectorD pME1;
  math::XYZVectorD pGE21;
  math::XYZVectorD pME2;
  math::XYZVectorD pME3;
  math::XYZVectorD pME4;
  int keyStation();
  math::XYZVectorD vAtStation(int st);
  math::XYZVectorD vSmart();
  double deltaRAtStation(int station, double to_eta, double to_phi);
  double deltaRSmart(double to_eta, double to_phi);
  
  // geometry
  const GEMGeometry* getGEMGeometry() const {return gemGeometry;}
  const CSCGeometry* getCSCGeometry() const {return cscGeometry;}
  void setGEMGeometry(const GEMGeometry* geom) {gemGeometry = geom;}
  void setCSCGeometry(const CSCGeometry* geom) {cscGeometry = geom;}
  
  // strk's ID is first element, followed by IDs of its children SimTracks
  std::vector<unsigned> familyIds;
  
  // matching SimHits of muon strk and (if !doSimpleSimHitToTrackMatch_) its children 
  void addSimHit( PSimHit & h );
  std::vector<PSimHit> simHits;
  std::map<int, std::vector<PSimHit> > hitsMapLayer;
  std::map<int, std::vector<PSimHit> > hitsMapChamber;

  // if( muOnly == true ) only hits with |particleType|==13 are considered
  
  int nSimHits();
  std::vector<int> detsWithHits();
  std::vector<int> chambersWithHits(int station=0, int ring=0, unsigned minNHits=4);
  // get the detIds that could have been crossed by the simtrack
  std::set<int> cscDetIdsAssociated(int station=0, int ring=0);
  std::set<int> gemDetIdsAssociated(int station=0, int ring=0);
  bool isCSCDetIdAssociated(CSCDetId id);
  bool isGEMDetIdAssociated(CSCDetId id);
  std::vector<PSimHit> layerHits( int detId );
  std::vector<PSimHit> chamberHits( int detId );
  std::vector<PSimHit> allSimHits();
  int numberOfLayersWithHitsInChamber( int detId );
  std::pair<int,int> wireGroupAndStripInChamber( int detId );

  // does simtrack has at least 4 simhits in a particular station and ring?
  // st=0 - any,  st=1,2,3,4 - ME1-4
  // ri=0 - any,  ri=1,2,3 - MEX/1-MEX/3
  bool hasHitsInStation(int st, int ri=0, unsigned minNHits=4); 
  unsigned nStationsWithHits(bool me1=1, bool me2=1, bool me3=1, bool me4=1, unsigned minNHits=4);

  // 
  bool muOnly;
  // primitives' readout window min and max BXs 
  int minBxALCT, maxBxALCT;
  int minBxCLCT, maxBxCLCT;
  int minBxLCT,  maxBxLCT;
  int minBxMPLCT,maxBxMPLCT;


  // matching ALCTs
  struct ALCT 
  {
    ALCT();
    ALCT(MatchCSCMuL1 *m);
    
    int getBX() {return trgdigi->getBX();}
    bool inReadOut();
    
    MatchCSCMuL1 *match; //containing object

    const CSCALCTDigi * trgdigi;
    std::vector<CSCAnodeLayerInfo> layerInfo;
    std::vector<PSimHit> simHits;
    CSCDetId id; // chamber id
    
    int nHitsShared; // # simhits shared with simtrack
    
    double eta;   // center of wire group eta
    double deltaY;     // deltas to SimTrack's 2D stub 
    double deltaPhi;   // in (Z,R) -> (x,y) plane
      
    int deltaWire;// delta to simtrack closest wire
    int mcWG; // simhit's wg #
    bool deltaOk;
  };
  std::vector< ALCT > ALCTs;
  std::vector< ALCT > ALCTsInReadOut();
  std::vector< ALCT > vALCTs(bool readout=true);

  // fit a 2D stub from SimHits matched to a digi
  void linearRegressionALCT( ALCT &alct, double &a, double &b);
  
  ALCT * bestALCT(CSCDetId id, bool readout=true);
  std::vector<int> chambersWithALCTs(bool readout=true);
  std::vector<ALCT> chamberALCTs( int detId, bool readout=true );
  std::vector<int> bxsWithALCTs( int detId, bool readout=true );
  std::vector<ALCT> chamberALCTsInBx( int detId, int bx, bool readout=true );

  // get deltaY and deltaTh 
  

  // matching CLCTs
  struct CLCT 
  {
    CLCT();
    CLCT(MatchCSCMuL1 *m);

    int getBX() {return trgdigi->getBX();}
    bool inReadOut();

    MatchCSCMuL1 *match; //containing object

    const CSCCLCTDigi * trgdigi;
    std::vector<CSCCathodeLayerInfo> layerInfo;
    std::vector<PSimHit> simHits;
    CSCDetId id; // chamber id

    int nHitsShared; // # simhits shared with simtrack

    double phi;    // center of strip phi
    double deltaY;     // deltas to SimTrack's 2D stub 
    double deltaPhi;   // in (Z,Phi) -> (x,y) plane

    int deltaStrip;// delta to simtrack closest strip
    int mcStrip; // simhit's strip # (would not be ganged!)
    bool deltaOk;
  };
  std::vector< CLCT > CLCTs;
  std::vector< CLCT > CLCTsInReadOut();
  std::vector< CLCT > vCLCTs(bool readout=true);

  // fit a 2D stub from SimHits matched to a digi
  void linearRegressionCLCT( CLCT &clct, double &a, double &b);

  CLCT * bestCLCT(CSCDetId id, bool readout=true);
  std::vector<int> chambersWithCLCTs(bool readout=true);
  std::vector<CLCT> chamberCLCTs( int detId, bool readout=true );
  std::vector<int> bxsWithCLCTs( int detId, bool readout=true );
  std::vector<CLCT> chamberCLCTsInBx( int detId, int bx, bool readout=true );

  // matching LCTs
  struct LCT 
  {
    LCT();
    LCT(MatchCSCMuL1 *m);

    int getBX() {return trgdigi->getBX();}
    bool inReadOut();

    MatchCSCMuL1 *match; //containing object

    const CSCCorrelatedLCTDigi * trgdigi;
    ALCT * alct;
    CLCT * clct;
    CSCDetId id; // chamber id
    bool ghost;
    bool deltaOk;
  };
  std::vector< LCT > LCTs;
  std::vector< LCT > LCTsInReadOut();
  std::vector< LCT > vLCTs(bool readout=true);

  std::vector<int> chambersWithLCTs( bool readout=true );
  std::vector<LCT> chamberLCTs( int detId, bool readout=true );
  std::vector<LCT*> chamberLCTsp( int detId, bool readout=true );
  std::vector<int> bxsWithLCTs( int detId, bool readout=true );
  std::vector<LCT> chamberLCTsInBx( int detId, int bx, bool readout=true );
    
  // matching MPLCTs
  struct MPLCT 
  {
    MPLCT();
    MPLCT(MatchCSCMuL1 *m);

    int getBX() {return trgdigi->getBX();}
    bool inReadOut();

    MatchCSCMuL1 *match; //containing object

    const CSCCorrelatedLCTDigi * trgdigi;
    LCT * lct;
    CSCDetId id; // chamber id
    unsigned meEtap;
    unsigned mePhip;
    bool ghost;
    bool deltaOk;
  };
  std::vector< MPLCT > MPLCTs;
  std::vector< MPLCT > MPLCTsInReadOut();
  std::vector< MPLCT > vMPLCTs(bool readout=true);

  std::vector<int> chambersWithMPLCTs( bool readout=true );
  std::vector<MPLCT> chamberMPLCTs( int detId, bool readout=true );
  std::vector<int> bxsWithMPLCTs( int detId, bool readout=true );
  std::vector<MPLCT> chamberMPLCTsInBx( int detId, int bx, bool readout=true );

  // matching TF's tracks
  struct TFTRACK
  {
    TFTRACK();
    TFTRACK(MatchCSCMuL1 *m);
    
    void init(const csc::L1Track *t, CSCTFPtLUT* ptLUT,
         edm::ESHandle< L1MuTriggerScales > &muScales,
         edm::ESHandle< L1MuTriggerPtScale > &muPtScale);
	 
    bool hasStub(int st); // st=0 - MB1, st=1,2,3,4 - ME1-4
    bool hasStubCSCOk(int st); // st=st=1,2,3,4 - ME1-4
    unsigned int nStubs(bool mb1=1, bool me1=1, bool me2=1, bool me3=1, bool me4=1);
    unsigned int nStubsCSCOk(bool mb1=1, bool me1=1, bool me2=1, bool me3=1, bool me4=1);
    bool passStubsMatch(int minLowHStubs=2, int minMidHStubs=2, int minHighHStubs=2);

    unsigned mode() const { return  (0x0f0000 & l1trk->ptLUTAddress())>>16;}
    bool sign() const { return l1trk->sign();}
    unsigned dPhi12() const { return 1*(l1trk->ptLUTAddress() & 0xFF);}
    unsigned dPhi23() const { return 1*( (l1trk->ptLUTAddress() & 0xF00)>>8 );}
    int getBX() {return l1trk->bx();}
    void print(const char msg[300]);

    MatchCSCMuL1 *match; //containing object

    const csc::L1Track * l1trk;
    std::vector < const CSCCorrelatedLCTDigi * > trgdigis;
    std::vector < CSCDetId > trgids;
    std::vector < std::pair<float, float> > trgetaphis;
    std::vector < csctf::TrackStub > trgstubs;
    std::vector < MPLCT* > mplcts;
    std::vector < CSCDetId > ids; // chamber ids
    unsigned phi_packed;
    unsigned eta_packed;
    unsigned pt_packed;
    unsigned q_packed;
    double phi;
    double eta;
    double pt;
    double dr;
    bool deltaOk1;
    bool deltaOk2;
    bool deltaOkME1;
    bool debug;
  };
  std::vector< TFTRACK > TFTRACKs;
  std::vector< TFTRACK > TFTRACKsAll;

  TFTRACK * bestTFTRACK(std::vector< TFTRACK > & trk, bool sortPtFirst=1);

  // matching TF's track candidates after CSC sorter
  struct TFCAND
  {
    TFCAND();
    TFCAND(MatchCSCMuL1 *m);

    void init(const L1MuRegionalCand *t, CSCTFPtLUT* ptLUT,
         edm::ESHandle< L1MuTriggerScales > &muScales,
         edm::ESHandle< L1MuTriggerPtScale > &muPtScale);

    MatchCSCMuL1 *match; //containing object

    const L1MuRegionalCand * l1cand;
    TFTRACK* tftrack;
    std::vector < CSCDetId > ids; // chamber ids
    double phi;
    double eta;
    double pt;
    double dr;
    unsigned nTFStubs;
  };
  std::vector< TFCAND > TFCANDs;
  std::vector< TFCAND > TFCANDsAll;

  TFCAND * bestTFCAND(std::vector< TFCAND > & cands, bool sortPtFirst=1);

  // matching GMT CSC tracks
  struct GMTREGCAND
  {
    void init(const L1MuRegionalCand *t,
         edm::ESHandle< L1MuTriggerScales > &muScales,
         edm::ESHandle< L1MuTriggerPtScale > &muPtScale);
    void print(const char msg[300]);
    const L1MuRegionalCand * l1reg;
    TFCAND* tfcand;
    std::vector< CSCDetId > ids; // chamber ids
    unsigned phi_packed;
    unsigned eta_packed;
    double phi;
    double eta;
    double pt;
    double dr;
    unsigned nTFStubs;
  };
  std::vector< GMTREGCAND > GMTREGCANDs;
  std::vector< GMTREGCAND > GMTREGCANDsAll;
  GMTREGCAND GMTREGCANDBest; // best matched in min DR max Pt, with no regard to any previous matches 

  GMTREGCAND * bestGMTREGCAND(std::vector< GMTREGCAND > & trk, bool sortPtFirst=1);

  // matching GMT tracks
  struct GMTCAND
  {
    void init(const L1MuGMTExtendedCand *t,
         edm::ESHandle< L1MuTriggerScales > &muScales,
         edm::ESHandle< L1MuTriggerPtScale > &muPtScale);
    const L1MuGMTExtendedCand * l1gmt;
    GMTREGCAND* regcand;
    GMTREGCAND* regcand_rpc;
    std::vector< CSCDetId > ids; // chamber ids
    double phi;
    double eta;
    double pt;
    double dr;
    int q;
    int rank;
    bool isCSC, isCSC2s, isCSC3s, isCSC2q, isCSC3q;
    bool isDT;
    bool isRPCf;
    bool isRPCb;
  };
  std::vector< GMTCAND > GMTCANDs;
  std::vector< GMTCAND > GMTCANDsAll;
  GMTCAND GMTCANDBest; // best matched in min DR max Pt, with no regard to any previous matches 

  GMTCAND * bestGMTCAND(std::vector< GMTCAND > & trk, bool sortPtFirst=1);

  // matching trigger muons from l1extra
  struct L1EXTRA
  {
    const l1extra::L1MuonParticle * l1extra;
    GMTCAND* gmtcand;
    double phi;
    double eta;
    double pt;
    double dr;
  };
  std::vector< L1EXTRA > L1EXTRAs;
  std::vector< L1EXTRA > L1EXTRAsAll;
  L1EXTRA L1EXTRABest; // best matched in min DR max Pt, with no regard to any previous matches 

  void print (const char msg[300], bool psimtr=1, bool psimh=1,
              bool palct=1, bool pclct=1, bool plct=1, bool pmplct=1,
	      bool ptftrack=0, bool ptfcand=0);
  void printSimTr (const char msg[300]) { print(msg,1,1,0,0,0,0); }

private:

};

#endif
