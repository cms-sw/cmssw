#ifndef Validation_MuonHits_ME0SimHitMatcher_h
#define Validation_MuonHits_ME0SimHitMatcher_h

/**\class ME0SimHitMatcher

   Description: Matching of muon SimHit to SimTrack

   Author: Sven Dildick (TAMU), Tao Huang (TAMU)
*/

#include "Validation/MuonHits/interface/MuonSimHitMatcher.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

class ME0SimHitMatcher : public MuonSimHitMatcher
{
public:

  // constructor
  ME0SimHitMatcher(const edm::ParameterSet& iPS, edm::ConsumesCollector && iC);

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

  // simhits from a particular partition, chamber
  const edm::PSimHitContainer& hitsInSuperChamber(unsigned int) const;

  // detid's with hits in 2 layers of coincidence pads
  std::set<unsigned int> detIdsCoincidences(int min_n_layers = 2) const;

  // ME0 superchamber detIds with SimHits
  std::set<unsigned int> superChamberIds() const;

  // ME0 superchamber detIds with SimHits >=4 layers of coincidence pads
  std::set<unsigned int> superChamberIdsCoincidences(int min_n_layers = 2) const;

  // #layers with hits
  int nLayersWithHitsInSuperChamber(unsigned int) const;

  float simHitsCentralPosition(const edm::PSimHitContainer& sim_hits) const;

  // How many pads with simhits in GEM did this simtrack get?
  int nPadsWithHits() const;

  // How many coincidence pads with simhits in GEM did this simtrack get?
  int nCoincidencePadsWithHits() const;

  // How many ME0 chambers with minimum number of layer with simhits did this simtrack get?
  int nCoincidenceChambers(int min_n_layers = 4) const;

  // calculated the fitted position in a given layer for CSC simhits in a chamber
  GlobalPoint simHitPositionKeyLayer(unsigned int chamberid) const;

  // calculate average strip (strip for GEM/ME0, half-strip for CSC) number for a provided collection of simhits
  float simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const;

  std::set<int> hitStripsInDetId(unsigned int, int margin_n_strips = 0) const;  // GEM/ME0 or CSC
  std::set<int> hitPadsInDetId(unsigned int) const; // GEM

  // what unique partitions numbers were hit by this simtrack?
  std::set<int> hitPartitions() const; // GEM

private:

  void matchSimHitsToSimTrack(std::vector<unsigned int> track_ids, const edm::PSimHitContainer& me0_hits);

  edm::ESHandle<ME0Geometry> me0_geom_;

  // detids with hits in pads
  std::map<unsigned int, std::set<int> > detids_to_pads_;

  // detids with hits in 2-layer pad coincidences
  std::map<unsigned int, std::set<int> > detids_to_copads_;

  std::map<unsigned int, edm::PSimHitContainer > superChamber_to_hits_;
};

#endif
