#ifndef Validation_MuonGEMDigis_GEMDigiMatcher_h
#define Validation_MuonGEMDigis_GEMDigiMatcher_h

/**\class DigiMatcher

 Description: Matching of Digis for SimTrack in GEM

 Author: Vadim Khotilovich, Sven Dildick (TAMU), Tao Huang (TAMU)
*/

#include "Validation/MuonHits/interface/GEMSimHitMatcher.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "SimDataFormats/GEMDigiSimLink/interface/GEMDigiSimLink.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include <vector>
#include <map>
#include <set>
#include <unordered_set>

typedef std::vector<GEMDigiSimLink> GEMDigiSimLinkContainer;
typedef std::vector<GEMDigi> GEMDigiContainer;
typedef std::vector<GEMPadDigi> GEMPadDigiContainer;
typedef std::vector<GEMPadDigiCluster> GEMPadDigiClusterContainer;
typedef std::vector<GEMCoPadDigi> GEMCoPadDigiContainer;

class GEMDigiMatcher {
public:
  // constructor
  GEMDigiMatcher(edm::ParameterSet const& iPS, edm::ConsumesCollector&& iC);

  // destructor
  ~GEMDigiMatcher() {}

  // initialize the event
  void init(const edm::Event& e, const edm::EventSetup& eventSetup);

  // do the matching
  void match(const SimTrack& t, const SimVertex& v);

  std::set<unsigned int> detIdsSimLink(int gem_type = MuonHitHelper::GEM_ALL) const;

  // partition GEM detIds with digis
  std::set<unsigned int> detIdsDigi(int gem_type = MuonHitHelper::GEM_ALL) const;
  std::set<unsigned int> detIdsPad(int gem_type = MuonHitHelper::GEM_ALL) const;
  std::set<unsigned int> detIdsCluster(int gem_type = MuonHitHelper::GEM_ALL) const;

  // chamber detIds with digis
  std::set<unsigned int> chamberIdsDigi(int gem_type = MuonHitHelper::GEM_ALL) const;
  std::set<unsigned int> chamberIdsPad(int gem_type = MuonHitHelper::GEM_ALL) const;
  std::set<unsigned int> chamberIdsCluster(int gem_type = MuonHitHelper::GEM_ALL) const;

  // superchamber detIds with digis
  std::set<unsigned int> superChamberIdsDigi(int gem_type = MuonHitHelper::GEM_ALL) const;
  std::set<unsigned int> superChamberIdsPad(int gem_type = MuonHitHelper::GEM_ALL) const;
  std::set<unsigned int> superChamberIdsCluster(int gem_type = MuonHitHelper::GEM_ALL) const;
  std::set<unsigned int> superChamberIdsCoPad(int gem_type = MuonHitHelper::GEM_ALL) const;

  const GEMDigiSimLinkContainer& simLinksInDetId(unsigned int) const;

  // GEM digis from a particular partition, chamber or superchamber
  const GEMDigiContainer& digisInDetId(unsigned int) const;
  const GEMDigiContainer& digisInChamber(unsigned int) const;
  const GEMDigiContainer& digisInSuperChamber(unsigned int) const;

  // GEM pads from a particular partition, chamber or superchamber
  const GEMPadDigiContainer& padsInDetId(unsigned int) const;
  const GEMPadDigiContainer& padsInChamber(unsigned int) const;
  const GEMPadDigiContainer& padsInSuperChamber(unsigned int) const;

  // GEM clusters from a particular partition, chamber or superchamber
  const GEMPadDigiClusterContainer& clustersInDetId(unsigned int) const;
  const GEMPadDigiClusterContainer& clustersInChamber(unsigned int) const;
  const GEMPadDigiClusterContainer& clustersInSuperChamber(unsigned int) const;

  // GEM co-pads from a particular partition or superchamber
  const GEMCoPadDigiContainer& coPadsInSuperChamber(unsigned int) const;

  const std::map<unsigned int, GEMPadDigiContainer> allPads() const { return detid_to_pads_; }

  // #layers with digis from this simtrack
  int nLayersWithDigisInSuperChamber(unsigned int) const;
  int nLayersWithPadsInSuperChamber(unsigned int) const;
  int nLayersWithClustersInSuperChamber(unsigned int) const;

  /// How many pads in GEM did this simtrack get in total?
  int nPads() const;

  /// How many coincidence pads in GEM did this simtrack get in total?
  int nCoPads() const;

  std::set<int> stripNumbersInDetId(unsigned int) const;
  std::set<int> padNumbersInDetId(unsigned int) const;

  int extrapolateHsfromGEMPad(unsigned int, int) const;
  int extrapolateHsfromGEMStrip(unsigned int, int) const;

  // what unique partitions numbers with digis from this simtrack?
  std::set<int> partitionNumbers() const;
  std::set<int> partitionNumbersWithCoPads() const;

  GlobalPoint getGlobalPointDigi(unsigned int rawId, const GEMDigi& d) const;
  GlobalPoint getGlobalPointPad(unsigned int rawId, const GEMPadDigi& tp) const;

  // get the GEMSimHitMatcher
  std::shared_ptr<GEMSimHitMatcher> muonSimHitMatcher() { return muonSimHitMatcher_; }

private:
  void matchDigisSLToSimTrack(const edm::DetSetVector<GEMDigiSimLink>&);
  void matchDigisToSimTrack(const GEMDigiCollection&);
  void matchPadsToSimTrack(const GEMPadDigiCollection&);
  void matchClustersToSimTrack(const GEMPadDigiClusterCollection&);
  void matchCoPadsToSimTrack(const GEMCoPadDigiCollection&);

  void clear();

  edm::EDGetTokenT<edm::DetSetVector<GEMDigiSimLink>> gemSimLinkToken_;
  edm::EDGetTokenT<GEMDigiCollection> gemDigiToken_;
  edm::EDGetTokenT<GEMPadDigiCollection> gemPadToken_;
  edm::EDGetTokenT<GEMPadDigiClusterCollection> gemClusterToken_;
  edm::EDGetTokenT<GEMCoPadDigiCollection> gemCoPadToken_;

  edm::Handle<edm::DetSetVector<GEMDigiSimLink>> gemDigisSLH_;
  edm::Handle<GEMDigiCollection> gemDigisH_;
  edm::Handle<GEMPadDigiCollection> gemPadsH_;
  edm::Handle<GEMPadDigiClusterCollection> gemClustersH_;
  edm::Handle<GEMCoPadDigiCollection> gemCoPadsH_;

  std::shared_ptr<GEMSimHitMatcher> muonSimHitMatcher_;

  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geomToken_;
  const GEMGeometry* gemGeometry_;

  template <class T>
  std::set<unsigned int> selectDetIds(const T&, int) const;

  bool simMuOnly_;
  bool discardEleHits_;

  int minBXDigi_, maxBXDigi_;
  int minBXPad_, maxBXPad_;
  int minBXCluster_, maxBXCluster_;
  int minBXCoPad_, maxBXCoPad_;

  int matchDeltaStrip_;

  bool verboseSimLink_;
  bool verboseDigi_;
  bool verbosePad_;
  bool verboseCluster_;
  bool verboseCoPad_;

  bool matchToSimLink_;

  std::map<unsigned int, GEMDigiSimLinkContainer> detid_to_simLinks_;

  std::map<unsigned int, GEMDigiContainer> detid_to_digis_;
  std::map<unsigned int, GEMDigiContainer> chamber_to_digis_;
  std::map<unsigned int, GEMDigiContainer> superchamber_to_digis_;

  std::map<unsigned int, GEMPadDigiContainer> detid_to_pads_;
  std::map<unsigned int, GEMPadDigiContainer> chamber_to_pads_;
  std::map<unsigned int, GEMPadDigiContainer> superchamber_to_pads_;

  std::map<unsigned int, GEMPadDigiClusterContainer> detid_to_clusters_;
  std::map<unsigned int, GEMPadDigiClusterContainer> chamber_to_clusters_;
  std::map<unsigned int, GEMPadDigiClusterContainer> superchamber_to_clusters_;

  std::map<unsigned int, GEMCoPadDigiContainer> superchamber_to_copads_;

  GEMDigiSimLinkContainer no_gem_simLinks_;
  GEMDigiContainer no_gem_digis_;
  GEMPadDigiContainer no_gem_pads_;
  GEMPadDigiClusterContainer no_gem_clusters_;
  GEMCoPadDigiContainer no_gem_copads_;
};

template <class T>
std::set<unsigned int> GEMDigiMatcher::selectDetIds(const T& digis, int gem_type) const {
  std::set<unsigned int> result;
  for (const auto& p : digis) {
    const auto& id = p.first;
    if (gem_type > 0) {
      GEMDetId detId(id);
      if (MuonHitHelper::toGEMType(detId.station(), detId.ring()) != gem_type)
        continue;
    }
    result.insert(p.first);
  }
  return result;
}

#endif
