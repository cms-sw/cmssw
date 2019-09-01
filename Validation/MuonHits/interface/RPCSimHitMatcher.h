#ifndef Validation_MuonHits_RPCSimHitMatcher_h
#define Validation_MuonHits_RPCSimHitMatcher_h

/**\class RPCSimHitMatcher

   Description: Matching of RPC SimHit to SimTrack

   Author: Sven Dildick (TAMU), Tao Huang (TAMU)
*/

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Validation/MuonHits/interface/MuonSimHitMatcher.h"

class RPCSimHitMatcher : public MuonSimHitMatcher {
public:
  // constructor
  RPCSimHitMatcher(const edm::ParameterSet& iPS, edm::ConsumesCollector&& iC);

  // destructor
  ~RPCSimHitMatcher() {}

  // initialize the event
  void init(const edm::Event& e, const edm::EventSetup& eventSetup);

  // do the matching
  void match(const SimTrack& t, const SimVertex& v);

  // partitions' detIds with SimHits
  std::set<unsigned int> detIds(int type = MuonHitHelper::RPC_ALL) const;

  // chamber detIds with SimHits
  std::set<unsigned int> chamberIds(int type = MuonHitHelper::RPC_ALL) const;

  bool hitStation(int st) const;

  // number of stations with hits
  int nStations() const;

  // calculate average strip number for a provided collection of simhits
  float simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const;

  std::set<int> hitStripsInDetId(unsigned int, int margin_n_strips = 0) const;

private:
  void matchSimHitsToSimTrack();

  edm::ESHandle<RPCGeometry> rpc_geom_;
};

#endif
