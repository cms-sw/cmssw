#ifndef GEMValidation_GEMRecHitMatcher_h
#define GEMValidation_GEMRecHitMatcher_h

/**\class RecHitMatcher

 Description: Matching of RecHits for SimTrack in GEM

 Original Author    : "Vadim Khotilovich"
 Contibuting Author : "Claudio Caputo"

*/

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "Validation/MuonGEMDigis/interface/GenericDigi.h"
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>

#include <map>
#include <set>
#include <vector>

class SimHitMatcher;
class GEMGeometry;
class GEMRecHitMatcher {
public:
  typedef matching::Digi RecHit;
  typedef matching::DigiContainer RecHitContainer;

  GEMRecHitMatcher(const SimHitMatcher &sh,
                   const edm::Event &,
                   const GEMGeometry &geom,
                   const edm::ParameterSet &cfg,
                   edm::EDGetToken &);

  ~GEMRecHitMatcher();

  // partition GEM detIds with rechits
  std::set<unsigned int> detIds() const;

  // chamber detIds with rechits
  std::set<unsigned int> chamberIds() const;

  // superchamber detIds with rechits
  std::set<unsigned int> superChamberIds() const;

  // GEM recHits from a particular partition, chamber or superchamber
  const RecHitContainer &recHitsInDetId(unsigned int) const;
  const RecHitContainer &recHitsInChamber(unsigned int) const;
  const RecHitContainer &recHitsInSuperChamber(unsigned int) const;

  // #layers with recHits from this simtrack
  int nLayersWithRecHitsInSuperChamber(unsigned int) const;

  /// How many recHits in GEM did this simtrack get in total?
  int nRecHits() const;

  std::set<int> stripNumbersInDetId(unsigned int) const;

  // what unique partitions numbers with recHits from this simtrack?
  std::set<int> partitionNumbers() const;

  // verbose value
  bool verbose() const { return verbose_; }

  GlobalPoint recHitPosition(const RecHit &rechit) const;
  GlobalPoint recHitMeanPosition(const RecHitContainer &rechits) const;

private:
  void init(const edm::Event &);

  void matchRecHitsToSimTrack(const GEMRecHitCollection &recHits);

  edm::Handle<GEMRecHitCollection> gem_rechits_;

  const SimHitMatcher &simhit_matcher_;
  const GEMGeometry &gem_geo_;

  int minBXGEM_, maxBXGEM_;
  bool verbose_;

  int matchDeltaStrip_;

  std::map<unsigned int, RecHitContainer> detid_to_recHits_;
  std::map<unsigned int, RecHitContainer> chamber_to_recHits_;
  std::map<unsigned int, RecHitContainer> superchamber_to_recHits_;

  const RecHitContainer no_recHits_;
};

#endif
