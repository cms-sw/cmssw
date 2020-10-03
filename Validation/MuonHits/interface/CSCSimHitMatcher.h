#ifndef Validation_MuonHits_CSCSimHitMatcher_h
#define Validation_MuonHits_CSCSimHitMatcher_h

/**\class CSCSimHitMatcher

   Description: Matching of CSC SimHit to SimTrack

   Author: Sven Dildick (TAMU), Tao Huang (TAMU)
*/

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Validation/MuonHits/interface/MuonSimHitMatcher.h"

class CSCSimHitMatcher : public MuonSimHitMatcher {
public:
  // constructor
  CSCSimHitMatcher(const edm::ParameterSet& iPS, edm::ConsumesCollector&& iC);

  // destructor
  ~CSCSimHitMatcher() {}

  // initialize the event
  void init(const edm::Event& e, const edm::EventSetup& eventSetup);

  // do the matching
  void match(const SimTrack& t, const SimVertex& v);

  // partitions' detIds with SimHits
  std::set<unsigned int> detIds(int type = MuonHitHelper::CSC_ALL) const;

  // chamber detIds with SimHits
  std::set<unsigned int> chamberIds(int type = MuonHitHelper::CSC_ALL) const;

  // CSC station detIds with SimHits
  std::set<unsigned int> chamberIdsStation(int station) const;

  // was there a hit in a particular CSC station?
  bool hitStation(int, int) const;

  // number of stations with hits in at least X layers
  int nStations(int nl = 4) const;

  // #layers with hits
  int nLayersWithHitsInChamber(unsigned int) const;

  // How many CSC chambers with minimum number of layer with simhits did this
  // simtrack get?
  int nCoincidenceChambers(int min_n_layers = 4) const;

  // calculated the fitted position in a given layer for CSC simhits in a
  // chamber
  GlobalPoint simHitPositionKeyLayer(unsigned int chamberid) const;

  // local bending in a CSC chamber
  float LocalBendingInChamber(unsigned int detid) const;
  void fitHitsInChamber(unsigned int detid, float& mean, float& slope) const;

  // calculate average strip number for a provided collection of simhits
  float simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const;

  // calculate average wg number for a provided collection of simhits (for CSC)
  float simHitsMeanWG(const edm::PSimHitContainer& sim_hits) const;

  void chamberIdsToString(const std::set<unsigned int>& set) const;

  // calculate the average position at the second station
  GlobalPoint simHitsMeanPositionStation(int n) const;

  std::set<int> hitStripsInDetId(unsigned int, int margin_n_strips = 0) const;
  std::set<int> hitWiregroupsInDetId(unsigned int, int margin_n_wg = 0) const;

  void camberIdsToString(const std::set<unsigned int>&) const;

private:
  void matchSimHitsToSimTrack();

  void clear();

  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geomToken_;
};

#endif
