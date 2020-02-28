#ifndef Validation_MuonHits_GEMSimHitMatcher_h
#define Validation_MuonHits_GEMSimHitMatcher_h

/**\class GEMSimHitMatcher

   Description: Matching of GEM SimHit to SimTrack

   Author: Sven Dildick (TAMU), Tao Huang (TAMU)
*/

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Validation/MuonHits/interface/MuonSimHitMatcher.h"

class GEMSimHitMatcher : public MuonSimHitMatcher {
public:
  // constructor
  GEMSimHitMatcher(const edm::ParameterSet& iPS, edm::ConsumesCollector&& iC);

  // destructor
  ~GEMSimHitMatcher() {}

  // initialize the event
  void init(const edm::Event& e, const edm::EventSetup& eventSetup);

  // do the matching
  void match(const SimTrack& t, const SimVertex& v);

  // partitions' detIds with SimHits
  std::set<unsigned int> detIds(int gem_type = MuonHitHelper::GEM_ALL) const;

  // chamber detIds with SimHits
  std::set<unsigned int> chamberIds(int gem_type = MuonHitHelper::GEM_ALL) const;

  // GEM detid's with hits in 2 layers of coincidence pads
  // those are layer==1 only detid's
  std::set<unsigned int> detIdsCoincidences() const;

  // GEM superchamber detIds with SimHits
  std::set<unsigned int> superChamberIds() const;

  // GEM superchamber detIds with SimHits 2 layers of coincidence pads
  std::set<unsigned int> superChamberIdsCoincidences() const;

  // simhits from a particular superchamber
  const edm::PSimHitContainer& hitsInSuperChamber(unsigned int) const;

  // was there a hit in a particular station?
  bool hitStation(int, int) const;

  // number of stations with hits in at least X layers
  int nStations(int nl = 2) const;

  // #layers with hits
  int nLayersWithHitsInSuperChamber(unsigned int) const;

  // How many pads with simhits in GEM did this simtrack get?
  int nPadsWithHits() const;

  // How many coincidence pads with simhits in GEM did this simtrack get?
  int nCoincidencePadsWithHits() const;

  // transverse position in GEM
  float simHitsGEMCentralPosition(const edm::PSimHitContainer& sim_hits) const;

  // calculate average strip number for a provided collection of simhits
  float simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const;

  std::set<int> hitStripsInDetId(unsigned int, int margin_n_strips = 0) const;
  std::set<int> hitPadsInDetId(unsigned int) const;
  std::set<int> hitCoPadsInDetId(unsigned int) const;

  // what unique partitions numbers were hit by this simtrack?
  std::set<int> hitPartitions() const;

private:
  void matchSimHitsToSimTrack();

  edm::ESHandle<GEMGeometry> gem_geom_;

  std::map<unsigned int, edm::PSimHitContainer> superchamber_to_hits_;

  // detids with hits in pads
  std::map<unsigned int, std::set<int> > detids_to_pads_;

  // detids with hits in 2-layer pad coincidences
  std::map<unsigned int, std::set<int> > detids_to_copads_;
};

#endif
