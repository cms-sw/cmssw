#ifndef SimMuL1_ALCT_h
#define SimMuL1_ALCT_h

/*
 * Class for matched ALCTs
 */



#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class ALCT
{
 public:
  /// constructor
  ALCT();
  ALCT(const CSCALCTDigi*);
  /// copy constructor
  ALCT(const ACLT&);
  /// destructor
  ~ALCT();
  
  /// get the underlying trigger digi
  const CSCALCTDigi* getTriggerDigi() const {return triggerDigi_;}
  const int getDetId() const {return detId_;}
  const int getBX() const {return bx_;}
  const bool inReadout() const {return inReadout_;}
  const bool isDeltaOk() const {return deltaOk_;}
  const int getNumberHitsShared() const {return nHitsShared_;}
  const int getMCWG() const {return mcWG_;}
  const int getDeltaWire() const {return deltaWire_;}
  const double getDeltaPhi() const {return deltaPhi_;}
  const double getDeltaY() const {return deltaY_;}
  const double getEta const {return eta_;}

  std::vector<CSCAnodeLayerInfo>& getLayerInfo const {return layerInfo_;}
  std::vector<PSimHit>& getSimHits const {return simHits_;}
  
 private:
  /// underlying trigger digi
  const CSCALCTDigi* triggerDigi_;
  
  /// layer info
  std::vector<CSCAnodeLayerInfo> layerInfo_;
  /// matching simhits
  std::vector<PSimHit> simHits_;
  
  /// detector ID
  int detId_;
  /// bunch crossing 
  int bx_;
  /// is it in the readout collection?
  bool inReadout_;
  /// was properly matched
  bool deltaOk_;
  /// number of SimHits shared with SimTrack
  int nHitsShared_;     
  /// SimHit's WG number 
  int mcWG_;
  /// delta to SimTrack closest wire
  int deltaWire_;
  /// in (Z,R) -> (x,y) plane
  double deltaPhi_;
  /// deltas to SimTrack's 2D stub
  double deltaY_;
  /// center of wire group eta
  double eta;  
};

#endif
