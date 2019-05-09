#ifndef GEMValidation_GEMDigiMatcher_h
#define GEMValidation_GEMDigiMatcher_h

/**\class DigiMatcher

 Description: Matching of Digis for SimTrack in GEM

 Original Author:  "Vadim Khotilovich"
*/

#include "GenericDigi.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>

#include <map>
#include <set>
#include <vector>

class SimHitMatcher;
class GEMGeometry;
class GEMDigiMatcher {
  using DigiContainer = matching::DigiContainer;

public:
  GEMDigiMatcher(const SimHitMatcher &sh,
                 const edm::Event &,
                 const GEMGeometry &geom,
                 const edm::ParameterSet &cfg,
                 edm::EDGetToken &,
                 edm::EDGetToken &,
                 edm::EDGetToken &);

  ~GEMDigiMatcher();

  // partition GEM detIds with digis
  std::set<unsigned int> detIds() const;

  // chamber detIds with digis
  std::set<unsigned int> chamberIds() const;
  // chamber detIds with pads
  std::set<unsigned int> chamberIdsWithPads() const;
  // superchamber detIds with coincidence pads
  std::set<unsigned int> superChamberIdsWithCoPads() const;

  // superchamber detIds with digis
  std::set<unsigned int> superChamberIds() const;
  std::set<unsigned int> superChamberIdsWithPads() const;

  // partition detIds with coincidence pads
  std::set<unsigned int> detIdsWithCoPads() const;

  // GEM digis from a particular partition, chamber or superchamber
  const DigiContainer &digisInDetId(unsigned int) const;
  const DigiContainer &digisInChamber(unsigned int) const;
  const DigiContainer &digisInSuperChamber(unsigned int) const;

  // GEM pads from a particular partition, chamber or superchamber
  const DigiContainer &padsInDetId(unsigned int) const;
  const DigiContainer &padsInChamber(unsigned int) const;
  const DigiContainer &padsInSuperChamber(unsigned int) const;

  // GEM co-pads from a particular partition or superchamber
  const DigiContainer &coPadsInDetId(unsigned int) const;
  const DigiContainer &coPadsInSuperChamber(unsigned int) const;

  // #layers with digis from this simtrack
  int nLayersWithDigisInSuperChamber(unsigned int) const;
  int nLayersWithPadsInSuperChamber(unsigned int) const;

  /// How many pads in GEM did this simtrack get in total?
  int nPads() const;

  /// How many coincidence pads in GEM did this simtrack get in total?
  int nCoPads() const;

  std::set<int> stripNumbersInDetId(unsigned int) const;
  std::set<int> padNumbersInDetId(unsigned int) const;
  std::set<int> coPadNumbersInDetId(unsigned int) const;

  // what unique partitions numbers with digis from this simtrack?
  std::set<int> partitionNumbers() const;
  std::set<int> partitionNumbersWithCoPads() const;

private:
  void init(const edm::Event &);

  void matchDigisToSimTrack(const GEMDigiCollection &digis);
  void matchPadsToSimTrack(const GEMPadDigiCollection &pads);
  void matchCoPadsToSimTrack(const GEMCoPadDigiCollection &co_pads);

  edm::Handle<GEMDigiCollection> gem_digis_;
  edm::Handle<GEMPadDigiCollection> gem_pads_;
  edm::Handle<GEMCoPadDigiCollection> gem_co_pads_;

  const SimHitMatcher &simhit_matcher_;
  const GEMGeometry &gem_geo_;

  int minBXGEM_, maxBXGEM_;
  bool verbose_;

  int matchDeltaStrip_;

  std::map<unsigned int, DigiContainer> detid_to_digis_;
  std::map<unsigned int, DigiContainer> chamber_to_digis_;
  std::map<unsigned int, DigiContainer> superchamber_to_digis_;

  std::map<unsigned int, DigiContainer> detid_to_pads_;
  std::map<unsigned int, DigiContainer> chamber_to_pads_;
  std::map<unsigned int, DigiContainer> superchamber_to_pads_;

  std::map<unsigned int, DigiContainer> detid_to_copads_;
  std::map<unsigned int, DigiContainer> chamber_to_copads_;
  std::map<unsigned int, DigiContainer> superchamber_to_copads_;

  const DigiContainer no_digis_;
};

#endif
