#ifndef Validation_MuonHits_DTSimHitMatcher_h
#define Validation_MuonHits_DTSimHitMatcher_h

/**\class DTSimHitMatcher

   Description: Matching of DT SimHit to SimTrack

   Author: Sven Dildick (TAMU), Tao Huang (TAMU)
*/

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Validation/MuonHits/interface/MuonSimHitMatcher.h"

class DTSimHitMatcher : public MuonSimHitMatcher {
public:
  // constructor
  DTSimHitMatcher(const edm::ParameterSet& iPS, edm::ConsumesCollector&& iC);

  // destructor
  ~DTSimHitMatcher() {}

  // initialize the event
  void init(const edm::Event& e, const edm::EventSetup& eventSetup);

  // do the matching
  void match(const SimTrack& t, const SimVertex& v);

  // partitions' detIds with SimHits
  std::set<unsigned int> detIds(int type = MuonHitHelper::DT_ALL) const;

  // chamber detIds with SimHits
  std::set<unsigned int> chamberIds(int type = MuonHitHelper::DT_ALL) const;

  // DT station detIds with SimHits
  std::set<unsigned int> chamberIdsStation(int station) const;

  // DT layer detIds with SimHits
  std::set<unsigned int> layerIds() const;

  // DT super layer detIds with SimHits
  std::set<unsigned int> superlayerIds() const;

  // was there a hit in a particular DT/CSC station?
  bool hitStation(int, int, int) const;

  // number of stations with hits in at least X layers
  int nStations(int nsl = 1, int nl = 3) const;

  // access to DT hits
  int nCellsWithHitsInLayer(unsigned int) const;
  int nLayersWithHitsInSuperLayer(unsigned int) const;
  int nSuperLayersWithHitsInChamber(unsigned int) const;
  int nLayersWithHitsInChamber(unsigned int) const;
  const edm::PSimHitContainer& hitsInLayer(unsigned int) const;
  const edm::PSimHitContainer& hitsInSuperLayer(unsigned int) const;
  const edm::PSimHitContainer& hitsInChamber(unsigned int) const;

  // calculate average wg number for a provided collection of simhits
  float simHitsMeanWire(const edm::PSimHitContainer& sim_hits) const;

  // calculate the average position at the second station
  GlobalPoint simHitsMeanPositionStation(int n) const;

  std::set<unsigned int> hitWiresInDTLayerId(unsigned int, int margin_n_wires = 0) const;       // DT
  std::set<unsigned int> hitWiresInDTSuperLayerId(unsigned int, int margin_n_wires = 0) const;  // DT
  std::set<unsigned int> hitWiresInDTChamberId(unsigned int, int margin_n_wires = 0) const;     // DT

  void dtChamberIdsToString(const std::set<unsigned int>&) const;

private:
  void matchSimHitsToSimTrack();

  std::map<unsigned int, edm::PSimHitContainer> layer_to_hits_;
  std::map<unsigned int, edm::PSimHitContainer> superlayer_to_hits_;

  edm::ESHandle<DTGeometry> dt_geom_;
};

#endif
