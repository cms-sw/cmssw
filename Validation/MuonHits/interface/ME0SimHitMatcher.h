#ifndef Validation_MuonHits_ME0SimHitMatcher_h
#define Validation_MuonHits_ME0SimHitMatcher_h

/**\class ME0SimHitMatcher

   Description: Matching of ME0 SimHit to SimTrack

   Author: Sven Dildick (TAMU), Tao Huang (TAMU)
*/

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Validation/MuonHits/interface/MuonSimHitMatcher.h"

class ME0SimHitMatcher : public MuonSimHitMatcher {
public:
  // constructor
  ME0SimHitMatcher(const edm::ParameterSet& iPS, edm::ConsumesCollector&& iC);

  // destructor
  ~ME0SimHitMatcher() {}

  // initialize the event
  void init(const edm::Event& e, const edm::EventSetup& eventSetup);

  // do the matching
  void match(const SimTrack& t, const SimVertex& v);

  // partitions' detIds with SimHits
  std::set<unsigned int> detIds() const;

  // chamber detIds with SimHits
  std::set<unsigned int> chamberIds() const;

  // ME0 superchamber detIds with SimHits
  std::set<unsigned int> superChamberIds() const;

  // simhits from a particular partition, chamber
  const edm::PSimHitContainer& hitsInSuperChamber(unsigned int) const;

  // #layers with hits
  int nLayersWithHitsInSuperChamber(unsigned int) const;

  // ME0 superchamber detIds with SimHits >=4 layers of coincidence pads
  std::set<unsigned int> superChamberIdsCoincidences(int min_n_layers = 4) const;

  // How many ME0 chambers with minimum number of layer with simhits did this
  // simtrack get?
  int nCoincidenceChambers(int min_n_layers = 4) const;

  // calculate average strip for a provided collection of simhits
  float simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const;

  std::set<int> hitStripsInDetId(unsigned int, int margin_n_strips = 0) const;
  std::set<int> hitPadsInDetId(unsigned int) const;

  // what unique partitions numbers were hit by this simtrack?
  std::set<int> hitPartitions() const;

  // How many pads with simhits in ME0 did this simtrack get?
  int nPadsWithHits() const;

private:
  void matchSimHitsToSimTrack();

  edm::ESHandle<ME0Geometry> me0_geom_;

  // detids with hits in pads
  std::map<unsigned int, std::set<int> > detids_to_pads_;

  // detids with hits in 2-layer pad coincidences
  std::map<unsigned int, std::set<int> > detids_to_copads_;

  std::map<unsigned int, edm::PSimHitContainer> superChamber_to_hits_;
};

#endif
