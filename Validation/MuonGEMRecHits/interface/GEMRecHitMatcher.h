#ifndef Validation_MuonGEMRecHits_GEMRecHitMatcher_h
#define Validation_MuonGEMRecHits_GEMRecHitMatcher_h

/**\class GEMRecHitMatcher

 Description: Matching of RecHits for SimTrack in GEM

 Original Author    : "Vadim Khotilovich"
 Contibuting Author : "Claudio Caputo"

*/

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "Validation/MuonGEMDigis/interface/GEMDigiMatcher.h"

#include <vector>
#include <map>
#include <set>

typedef std::vector<GEMRecHit> GEMRecHitContainer;

class GEMGeometry;
class GEMRecHitMatcher {
public:
  // constructor
  GEMRecHitMatcher(edm::ParameterSet const& iPS, edm::ConsumesCollector&& iC);

  // destructor
  ~GEMRecHitMatcher() {}

  // initialize the event
  void init(const edm::Event& e, const edm::EventSetup& eventSetup);

  // do the matching
  void match(const SimTrack& t, const SimVertex& v);

  // partition GEM detIds with rechits
  std::set<unsigned int> detIds() const;

  // chamber detIds with rechits
  std::set<unsigned int> chamberIds() const;

  // superchamber detIds with rechits
  std::set<unsigned int> superChamberIds() const;

  // GEM recHits from a particular partition, chamber or superchamber
  const GEMRecHitContainer& recHits() const { return recHits_; }
  const GEMRecHitContainer& recHitsInDetId(unsigned int) const;
  const GEMRecHitContainer& recHitsInChamber(unsigned int) const;
  const GEMRecHitContainer& recHitsInSuperChamber(unsigned int) const;

  // #layers with recHits from this simtrack
  int nLayersWithRecHitsInSuperChamber(unsigned int) const;

  /// How many recHits in GEM did this simtrack get in total?
  int nGEMRecHits() const;

  std::set<int> stripNumbersInDetId(unsigned int) const;

  // what unique partitions numbers with recHits from this simtrack?
  std::set<int> partitionNumbers() const;

  // verbose value
  bool verbose() const { return verbose_; }

  // global position of the rechit (based on the first strip hit)
  GlobalPoint recHitPosition(const GEMRecHit& rechit) const;

  // mean position of a rechit collection (all based on the first strip hit)
  GlobalPoint recHitMeanPosition(const GEMRecHitContainer& rechits) const;

  std::shared_ptr<GEMDigiMatcher> gemDigiMatcher() const { return gemDigiMatcher_; }

  bool recHitInContainer(const GEMRecHit& rh, const GEMRecHitContainer& c) const;

  bool isGEMRecHitMatched(const GEMRecHit& thisRh) const;

  bool areGEMRecHitSame(const GEMRecHit& l, const GEMRecHit& r) const;

private:
  void matchRecHitsToSimTrack(const GEMRecHitCollection& recHits);

  edm::EDGetTokenT<GEMRecHitCollection> gemRecHitToken_;
  edm::Handle<GEMRecHitCollection> gemRecHitH_;

  edm::ESHandle<GEMGeometry> gem_geom_;
  const GEMGeometry* gemGeometry_;

  int minBX_, maxBX_;
  bool verbose_;

  std::map<unsigned int, GEMRecHitContainer> detid_to_recHits_;
  std::map<unsigned int, GEMRecHitContainer> chamber_to_recHits_;
  std::map<unsigned int, GEMRecHitContainer> superchamber_to_recHits_;

  const GEMRecHitContainer no_recHits_;
  GEMRecHitContainer recHits_;

  std::shared_ptr<GEMDigiMatcher> gemDigiMatcher_;
};

#endif
