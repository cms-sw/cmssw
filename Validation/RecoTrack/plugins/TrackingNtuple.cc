// -*- C++ -*-
//
// Package:    NtupleDump/TrackingNtuple
// Class:      TrackingNtuple
//
/**\class TrackingNtuple TrackingNtuple.cc NtupleDump/TrackingNtuple/plugins/TrackingNtuple.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Giuseppe Cerati
//         Created:  Tue, 25 Aug 2015 13:22:49 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/DynArray.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RZLine.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/interface/TrackingParticleIP.h"
#include "SimTracker/TrackAssociation/interface/trackAssociationChi2.h"
#include "SimTracker/TrackAssociation/interface/trackHitsToClusterRefs.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimTracker/TrackHistory/interface/HistoryBase.h"
#include "HepPDT/ParticleID.hh"

#include "Validation/RecoTrack/interface/trackFromSeedFitFailed.h"

#include "RecoTracker/FinalTrackSelectors/plugins/getBestVertex.h"

#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <utility>

#include "TTree.h"

/*
todo: 
add refitted hit position after track/seed fit
add local angle, path length!
*/

namespace {
  // This pattern is copied from QuickTrackAssociatorByHitsImpl. If
  // further needs arises, it wouldn't hurt to abstract it somehow.
  using TrackingParticleRefKeyToIndex = std::unordered_map<reco::RecoToSimCollection::index_type, size_t>;
  using TrackingVertexRefKeyToIndex = TrackingParticleRefKeyToIndex;
  using SimHitFullKey = std::pair<TrackPSimHitRef::key_type, edm::ProductID>;
  using SimHitRefKeyToIndex = std::map<SimHitFullKey, size_t>;
  using TrackingParticleRefKeyToCount = TrackingParticleRefKeyToIndex;

  std::string subdetstring(int subdet) {
    switch (subdet) {
      case StripSubdetector::TIB:
        return "- TIB";
      case StripSubdetector::TOB:
        return "- TOB";
      case StripSubdetector::TEC:
        return "- TEC";
      case StripSubdetector::TID:
        return "- TID";
      case PixelSubdetector::PixelBarrel:
        return "- PixBar";
      case PixelSubdetector::PixelEndcap:
        return "- PixFwd";
      default:
        return "UNKNOWN TRACKER HIT TYPE";
    }
  }

  struct ProductIDSetPrinter {
    ProductIDSetPrinter(const std::set<edm::ProductID>& set) : set_(set) {}

    void print(std::ostream& os) const {
      for (const auto& item : set_) {
        os << item << " ";
      }
    }

    const std::set<edm::ProductID>& set_;
  };
  std::ostream& operator<<(std::ostream& os, const ProductIDSetPrinter& o) {
    o.print(os);
    return os;
  }
  template <typename T>
  struct ProductIDMapPrinter {
    ProductIDMapPrinter(const std::map<edm::ProductID, T>& map) : map_(map) {}

    void print(std::ostream& os) const {
      for (const auto& item : map_) {
        os << item.first << " ";
      }
    }

    const std::map<edm::ProductID, T>& map_;
  };
  template <typename T>
  auto make_ProductIDMapPrinter(const std::map<edm::ProductID, T>& map) {
    return ProductIDMapPrinter<T>(map);
  }
  template <typename T>
  std::ostream& operator<<(std::ostream& os, const ProductIDMapPrinter<T>& o) {
    o.print(os);
    return os;
  }

  template <typename T>
  struct VectorPrinter {
    VectorPrinter(const std::vector<T>& vec) : vec_(vec) {}

    void print(std::ostream& os) const {
      for (const auto& item : vec_) {
        os << item << " ";
      }
    }

    const std::vector<T>& vec_;
  };
  template <typename T>
  auto make_VectorPrinter(const std::vector<T>& vec) {
    return VectorPrinter<T>(vec);
  }
  template <typename T>
  std::ostream& operator<<(std::ostream& os, const VectorPrinter<T>& o) {
    o.print(os);
    return os;
  }

  void checkProductID(const std::set<edm::ProductID>& set, const edm::ProductID& id, const char* name) {
    if (set.find(id) == set.end())
      throw cms::Exception("Configuration")
          << "Got " << name << " with a hit with ProductID " << id
          << " which does not match to the set of ProductID's for the hits: " << ProductIDSetPrinter(set)
          << ". Usually this is caused by a wrong hit collection in the configuration.";
  }

  template <typename SimLink, typename Func>
  void forEachMatchedSimLink(const edm::DetSet<SimLink>& digiSimLinks, uint32_t channel, Func func) {
    for (const auto& link : digiSimLinks) {
      if (link.channel() == channel) {
        func(link);
      }
    }
  }

  /// No-op function used in the trick of CombineDetId::impl2()
  template <typename... Args>
  void call_nop(Args&&... args) {}

  template <typename... Types>
  class CombineDetId {
  public:
    CombineDetId() {}

    /// Return the raw DetId, assumes that the first type is
    /// DetIdCommon that has operator[]()
    unsigned int operator[](size_t i) const { return std::get<0>(content_)[i]; }

    template <typename... Args>
    void book(Args&&... args) {
      impl([&](auto& vec) { vec.book(std::forward<Args>(args)...); });
    }

    template <typename... Args>
    void push_back(Args&&... args) {
      impl([&](auto& vec) { vec.push_back(std::forward<Args>(args)...); });
    }

    template <typename... Args>
    void resize(Args&&... args) {
      impl([&](auto& vec) { vec.resize(std::forward<Args>(args)...); });
    }

    template <typename... Args>
    void set(Args&&... args) {
      impl([&](auto& vec) { vec.set(std::forward<Args>(args)...); });
    }

    void clear() {
      impl([&](auto& vec) { vec.clear(); });
    }

  private:
    // Trick to not repeate std::index_sequence_for in each of the methods above
    template <typename F>
    void impl(F&& func) {
      impl2(std::index_sequence_for<Types...>{}, std::forward<F>(func));
    }

    // Trick to exploit parameter pack expansion in function call
    // arguments to call a member function for each tuple element
    // (with the same signature). The comma operator is needed to
    // return a value from the expression as an argument for the
    // call_nop.
    template <std::size_t... Is, typename F>
    void impl2(std::index_sequence<Is...>, F&& func) {
      call_nop((func(std::get<Is>(content_)), 0)...);
    }

    std::tuple<Types...> content_;
  };

  std::map<unsigned int, double> chargeFraction(const SiPixelCluster& cluster,
                                                const DetId& detId,
                                                const edm::DetSetVector<PixelDigiSimLink>& digiSimLink) {
    std::map<unsigned int, double> simTrackIdToAdc;

    auto idetset = digiSimLink.find(detId);
    if (idetset == digiSimLink.end())
      return simTrackIdToAdc;

    double adcSum = 0;
    PixelDigiSimLink found;
    for (int iPix = 0; iPix != cluster.size(); ++iPix) {
      const SiPixelCluster::Pixel& pixel = cluster.pixel(iPix);
      adcSum += pixel.adc;
      uint32_t channel = PixelChannelIdentifier::pixelToChannel(pixel.x, pixel.y);
      forEachMatchedSimLink(*idetset, channel, [&](const PixelDigiSimLink& simLink) {
        double& adc = simTrackIdToAdc[simLink.SimTrackId()];
        adc += pixel.adc * simLink.fraction();
      });
    }

    for (auto& pair : simTrackIdToAdc) {
      if (adcSum == 0.)
        pair.second = 0.;
      else
        pair.second /= adcSum;
    }

    return simTrackIdToAdc;
  }

  std::map<unsigned int, double> chargeFraction(const SiStripCluster& cluster,
                                                const DetId& detId,
                                                const edm::DetSetVector<StripDigiSimLink>& digiSimLink) {
    std::map<unsigned int, double> simTrackIdToAdc;

    auto idetset = digiSimLink.find(detId);
    if (idetset == digiSimLink.end())
      return simTrackIdToAdc;

    double adcSum = 0;
    StripDigiSimLink found;
    int first = cluster.firstStrip();
    for (size_t i = 0; i < cluster.amplitudes().size(); ++i) {
      adcSum += cluster.amplitudes()[i];
      forEachMatchedSimLink(*idetset, first + i, [&](const StripDigiSimLink& simLink) {
        double& adc = simTrackIdToAdc[simLink.SimTrackId()];
        adc += cluster.amplitudes()[i] * simLink.fraction();
      });

      for (const auto& pair : simTrackIdToAdc) {
        simTrackIdToAdc[pair.first] = (adcSum != 0. ? pair.second / adcSum : 0.);
      }
    }
    return simTrackIdToAdc;
  }

  std::map<unsigned int, double> chargeFraction(const Phase2TrackerCluster1D& cluster,
                                                const DetId& detId,
                                                const edm::DetSetVector<StripDigiSimLink>& digiSimLink) {
    std::map<unsigned int, double> simTrackIdToAdc;
    throw cms::Exception("LogicError") << "Not possible to use StripDigiSimLink with Phase2TrackerCluster1D! ";
    return simTrackIdToAdc;
  }

  //In the OT, there is no measurement of the charge, so no ADC value.
  //Only in the SSA chip (so in PSs) you have one "threshold" flag that tells you if the charge of at least one strip in the cluster exceeded 1.2 MIPs.
  std::map<unsigned int, double> chargeFraction(const Phase2TrackerCluster1D& cluster,
                                                const DetId& detId,
                                                const edm::DetSetVector<PixelDigiSimLink>& digiSimLink) {
    std::map<unsigned int, double> simTrackIdToAdc;
    return simTrackIdToAdc;
  }

  struct TrackTPMatch {
    int key = -1;
    int countClusters = 0;
  };

  TrackTPMatch findBestMatchingTrackingParticle(const reco::Track& track,
                                                const ClusterTPAssociation& clusterToTPMap,
                                                const TrackingParticleRefKeyToIndex& tpKeyToIndex) {
    struct Count {
      int clusters = 0;
      size_t innermostHit = std::numeric_limits<size_t>::max();
    };

    std::vector<OmniClusterRef> clusters =
        track_associator::hitsToClusterRefs(track.recHitsBegin(), track.recHitsEnd());

    std::unordered_map<int, Count> count;
    for (size_t iCluster = 0, end = clusters.size(); iCluster < end; ++iCluster) {
      const auto& clusterRef = clusters[iCluster];

      auto range = clusterToTPMap.equal_range(clusterRef);
      for (auto ip = range.first; ip != range.second; ++ip) {
        const auto tpKey = ip->second.key();
        if (tpKeyToIndex.find(tpKey) == tpKeyToIndex.end())  // filter out TPs not given as an input
          continue;

        auto& elem = count[tpKey];
        ++elem.clusters;
        elem.innermostHit = std::min(elem.innermostHit, iCluster);
      }
    }

    // In case there are many matches with the same number of clusters,
    // select the one with innermost hit
    TrackTPMatch best;
    int bestCount = 2;  // require >= 3 cluster for the best match
    size_t bestInnermostHit = std::numeric_limits<size_t>::max();
    for (auto& keyCount : count) {
      if (keyCount.second.clusters > bestCount ||
          (keyCount.second.clusters == bestCount && keyCount.second.innermostHit < bestInnermostHit)) {
        best.key = keyCount.first;
        best.countClusters = bestCount = keyCount.second.clusters;
        bestInnermostHit = keyCount.second.innermostHit;
      }
    }

    LogTrace("TrackingNtuple") << "findBestMatchingTrackingParticle key " << best.key;

    return best;
  }

  TrackTPMatch findMatchingTrackingParticleFromFirstHit(const reco::Track& track,
                                                        const ClusterTPAssociation& clusterToTPMap,
                                                        const TrackingParticleRefKeyToIndex& tpKeyToIndex) {
    TrackTPMatch best;

    std::vector<OmniClusterRef> clusters =
        track_associator::hitsToClusterRefs(track.recHitsBegin(), track.recHitsEnd());
    if (clusters.empty()) {
      return best;
    }

    auto operateCluster = [&](const auto& clusterRef, const auto& func) {
      auto range = clusterToTPMap.equal_range(clusterRef);
      for (auto ip = range.first; ip != range.second; ++ip) {
        const auto tpKey = ip->second.key();
        if (tpKeyToIndex.find(tpKey) == tpKeyToIndex.end())  // filter out TPs not given as an input
          continue;
        func(tpKey);
      }
    };

    std::vector<unsigned int>
        validTPs;  // first cluster can be associated to multiple TPs, use vector as set as this should be small
    auto iCluster = clusters.begin();
    operateCluster(*iCluster, [&](unsigned int tpKey) { validTPs.push_back(tpKey); });
    if (validTPs.empty()) {
      return best;
    }
    ++iCluster;
    ++best.countClusters;

    std::vector<bool> foundTPs(validTPs.size(), false);
    for (auto iEnd = clusters.end(); iCluster != iEnd; ++iCluster) {
      const auto& clusterRef = *iCluster;

      // find out to which first-cluster TPs this cluster is matched to
      operateCluster(clusterRef, [&](unsigned int tpKey) {
        auto found = std::find(cbegin(validTPs), cend(validTPs), tpKey);
        if (found != cend(validTPs)) {
          foundTPs[std::distance(cbegin(validTPs), found)] = true;
        }
      });

      // remove the non-found TPs
      auto iTP = validTPs.size();
      do {
        --iTP;

        if (!foundTPs[iTP]) {
          validTPs.erase(validTPs.begin() + iTP);
          foundTPs.erase(foundTPs.begin() + iTP);
        }
      } while (iTP > 0);
      if (!validTPs.empty()) {
        // for multiple TPs the "first one" is a bit arbitrary, but
        // I hope it is rare that a track would have many
        // consecutive hits matched to two TPs
        best.key = validTPs[0];
      } else {
        break;
      }

      std::fill(begin(foundTPs), end(foundTPs), false);
      ++best.countClusters;
    }

    // Reqquire >= 3 clusters for a match
    return best.countClusters >= 3 ? best : TrackTPMatch();
  }
}  // namespace

//
// class declaration
//

class TrackingNtuple : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TrackingNtuple(const edm::ParameterSet&);
  ~TrackingNtuple() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void clearVariables();

  enum class HitType { Pixel = 0, Strip = 1, Glued = 2, Invalid = 3, Phase2OT = 4, Unknown = 99 };

  // This gives the "best" classification of a reco hit
  // In case of reco hit mathing to multiple sim, smaller number is
  // considered better
  // To be kept in synch with class HitSimType in ntuple.py
  enum class HitSimType { Signal = 0, ITPileup = 1, OOTPileup = 2, Noise = 3, Unknown = 99 };

  using MVACollection = std::vector<float>;
  using QualityMaskCollection = std::vector<unsigned char>;

  using PixelMaskContainer = edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster>>;
  using StripMaskContainer = edm::ContainerMask<edmNew::DetSetVector<SiStripCluster>>;

  struct TPHitIndex {
    TPHitIndex(unsigned int tp = 0, unsigned int simHit = 0, float to = 0, unsigned int id = 0)
        : tpKey(tp), simHitIdx(simHit), tof(to), detId(id) {}
    unsigned int tpKey;
    unsigned int simHitIdx;
    float tof;
    unsigned int detId;
  };
  static bool tpHitIndexListLess(const TPHitIndex& i, const TPHitIndex& j) { return (i.tpKey < j.tpKey); }
  static bool tpHitIndexListLessSort(const TPHitIndex& i, const TPHitIndex& j) {
    if (i.tpKey == j.tpKey) {
      if (edm::isNotFinite(i.tof) && edm::isNotFinite(j.tof)) {
        return i.detId < j.detId;
      }
      return i.tof < j.tof;  // works as intended if either one is NaN
    }
    return i.tpKey < j.tpKey;
  }

  void fillBeamSpot(const reco::BeamSpot& bs);
  void fillPixelHits(const edm::Event& iEvent,
                     const ClusterTPAssociation& clusterToTPMap,
                     const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                     const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                     const edm::DetSetVector<PixelDigiSimLink>& digiSimLink,
                     const TransientTrackingRecHitBuilder& theTTRHBuilder,
                     const TrackerTopology& tTopo,
                     const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                     std::set<edm::ProductID>& hitProductIds);

  void fillStripRphiStereoHits(const edm::Event& iEvent,
                               const ClusterTPAssociation& clusterToTPMap,
                               const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                               const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                               const edm::DetSetVector<StripDigiSimLink>& digiSimLink,
                               const TransientTrackingRecHitBuilder& theTTRHBuilder,
                               const TrackerTopology& tTopo,
                               const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                               std::set<edm::ProductID>& hitProductIds);

  void fillStripMatchedHits(const edm::Event& iEvent,
                            const TransientTrackingRecHitBuilder& theTTRHBuilder,
                            const TrackerTopology& tTopo,
                            std::vector<std::pair<int, int>>& monoStereoClusterList);

  size_t addStripMatchedHit(const SiStripMatchedRecHit2D& hit,
                            const TransientTrackingRecHitBuilder& theTTRHBuilder,
                            const TrackerTopology& tTopo,
                            const std::vector<std::pair<uint64_t, StripMaskContainer const*>>& stripMasks,
                            std::vector<std::pair<int, int>>& monoStereoClusterList);

  void fillPhase2OTHits(const edm::Event& iEvent,
                        const ClusterTPAssociation& clusterToTPMap,
                        const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                        const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                        const edm::DetSetVector<PixelDigiSimLink>& digiSimLink,
                        const TransientTrackingRecHitBuilder& theTTRHBuilder,
                        const TrackerTopology& tTopo,
                        const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                        std::set<edm::ProductID>& hitProductIds);

  void fillSeeds(const edm::Event& iEvent,
                 const TrackingParticleRefVector& tpCollection,
                 const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                 const reco::BeamSpot& bs,
                 const reco::TrackToTrackingParticleAssociator& associatorByHits,
                 const ClusterTPAssociation& clusterToTPMap,
                 const TransientTrackingRecHitBuilder& theTTRHBuilder,
                 const MagneticField& theMF,
                 const TrackerTopology& tTopo,
                 std::vector<std::pair<int, int>>& monoStereoClusterList,
                 const std::set<edm::ProductID>& hitProductIds,
                 std::map<edm::ProductID, size_t>& seedToCollIndex);

  void fillTracks(const edm::RefToBaseVector<reco::Track>& tracks,
                  const TrackingParticleRefVector& tpCollection,
                  const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                  const TrackingParticleRefKeyToCount& tpKeyToClusterCount,
                  const MagneticField& mf,
                  const reco::BeamSpot& bs,
                  const reco::VertexCollection& vertices,
                  const reco::TrackToTrackingParticleAssociator& associatorByHits,
                  const ClusterTPAssociation& clusterToTPMap,
                  const TransientTrackingRecHitBuilder& theTTRHBuilder,
                  const TrackerTopology& tTopo,
                  const std::set<edm::ProductID>& hitProductIds,
                  const std::map<edm::ProductID, size_t>& seedToCollIndex,
                  const std::vector<const MVACollection*>& mvaColls,
                  const std::vector<const QualityMaskCollection*>& qualColls);

  void fillSimHits(const TrackerGeometry& tracker,
                   const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                   const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                   const TrackerTopology& tTopo,
                   SimHitRefKeyToIndex& simHitRefKeyToIndex,
                   std::vector<TPHitIndex>& tpHitList);

  void fillTrackingParticles(const edm::Event& iEvent,
                             const edm::EventSetup& iSetup,
                             const edm::RefToBaseVector<reco::Track>& tracks,
                             const reco::BeamSpot& bs,
                             const TrackingParticleRefVector& tpCollection,
                             const TrackingVertexRefKeyToIndex& tvKeyToIndex,
                             const reco::TrackToTrackingParticleAssociator& associatorByHits,
                             const std::vector<TPHitIndex>& tpHitList,
                             const TrackingParticleRefKeyToCount& tpKeyToClusterCount);

  void fillTrackingParticlesForSeeds(const TrackingParticleRefVector& tpCollection,
                                     const reco::SimToRecoCollection& simRecColl,
                                     const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                     const unsigned int seedOffset);

  void fillVertices(const reco::VertexCollection& vertices, const edm::RefToBaseVector<reco::Track>& tracks);

  void fillTrackingVertices(const TrackingVertexRefVector& trackingVertices,
                            const TrackingParticleRefKeyToIndex& tpKeyToIndex);

  struct SimHitData {
    std::vector<int> matchingSimHit;
    std::vector<float> chargeFraction;
    std::vector<float> xySignificance;
    std::vector<int> bunchCrossing;
    std::vector<int> event;
    HitSimType type = HitSimType::Unknown;
  };

  template <typename SimLink>
  SimHitData matchCluster(const OmniClusterRef& cluster,
                          DetId hitId,
                          int clusterKey,
                          const TransientTrackingRecHit::RecHitPointer& ttrh,
                          const ClusterTPAssociation& clusterToTPMap,
                          const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                          const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                          const edm::DetSetVector<SimLink>& digiSimLinks,
                          const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                          HitType hitType);

  // ----------member data ---------------------------
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
  const edm::ESGetToken<ParametersDefinerForTP, TrackAssociatorRecord> paramsDefineToken_;

  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> seedTokens_;
  std::vector<edm::EDGetTokenT<std::vector<SeedStopInfo>>> seedStopInfoTokens_;
  edm::EDGetTokenT<edm::View<reco::Track>> trackToken_;
  std::vector<std::tuple<edm::EDGetTokenT<MVACollection>, edm::EDGetTokenT<QualityMaskCollection>>>
      mvaQualityCollectionTokens_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;
  edm::EDGetTokenT<TrackingParticleRefVector> trackingParticleRefToken_;
  edm::EDGetTokenT<ClusterTPAssociation> clusterTPMapToken_;
  edm::EDGetTokenT<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitTPMapToken_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trackAssociatorToken_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> pixelSimLinkToken_;
  edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink>> stripSimLinkToken_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> siphase2OTSimLinksToken_;
  bool includeStripHits_, includePhase2OTHits_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> stripMatchedRecHitToken_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> phase2OTRecHitToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  edm::EDGetTokenT<TrackingVertexCollection> trackingVertexToken_;
  edm::EDGetTokenT<edm::ValueMap<unsigned int>> tpNLayersToken_;
  edm::EDGetTokenT<edm::ValueMap<unsigned int>> tpNPixelLayersToken_;
  edm::EDGetTokenT<edm::ValueMap<unsigned int>> tpNStripStereoLayersToken_;

  std::vector<std::pair<unsigned int, edm::EDGetTokenT<PixelMaskContainer>>> pixelUseMaskTokens_;
  std::vector<std::pair<unsigned int, edm::EDGetTokenT<StripMaskContainer>>> stripUseMaskTokens_;

  std::string builderName_;
  std::string parametersDefinerName_;
  const bool includeSeeds_;
  const bool addSeedCurvCov_;
  const bool includeAllHits_;
  const bool includeMVA_;
  const bool includeTrackingParticles_;
  const bool includeOOT_;
  const bool keepEleSimHits_;
  const bool saveSimHitsP3_;
  const bool simHitBySignificance_;

  HistoryBase tracer_;

  TTree* t;

  // DetId branches
#define BOOK(name) tree->Branch((prefix + "_" + #name).c_str(), &name);
  class DetIdCommon {
  public:
    DetIdCommon(){};

    unsigned int operator[](size_t i) const { return detId[i]; }

    void book(const std::string& prefix, TTree* tree) {
      BOOK(detId);
      BOOK(subdet);
      BOOK(layer);
      BOOK(side);
      BOOK(module);
    }

    void push_back(const TrackerTopology& tTopo, const DetId& id) {
      detId.push_back(id.rawId());
      subdet.push_back(id.subdetId());
      layer.push_back(tTopo.layer(id));
      module.push_back(tTopo.module(id));

      unsigned short s = 0;
      switch (id.subdetId()) {
        case StripSubdetector::TIB:
          s = tTopo.tibSide(id);
          break;
        case StripSubdetector::TOB:
          s = tTopo.tobSide(id);
          break;
        default:
          s = tTopo.side(id);
      }
      side.push_back(s);
    }

    void resize(size_t size) {
      detId.resize(size);
      subdet.resize(size);
      layer.resize(size);
      side.resize(size);
      module.resize(size);
    }

    void set(size_t index, const TrackerTopology& tTopo, const DetId& id) {
      detId[index] = id.rawId();
      subdet[index] = id.subdetId();
      layer[index] = tTopo.layer(id);
      side[index] = tTopo.side(id);
      module[index] = tTopo.module(id);
    }

    void clear() {
      detId.clear();
      subdet.clear();
      layer.clear();
      side.clear();
      module.clear();
    }

  private:
    std::vector<unsigned int> detId;
    std::vector<unsigned short> subdet;
    std::vector<unsigned short> layer;  // or disk/wheel
    std::vector<unsigned short> side;
    std::vector<unsigned short> module;
  };

  class DetIdPixelOnly {
  public:
    DetIdPixelOnly() {}

    void book(const std::string& prefix, TTree* tree) {
      BOOK(ladder);
      BOOK(blade);
      BOOK(panel);
    }

    void push_back(const TrackerTopology& tTopo, const DetId& id) {
      const bool isBarrel = id.subdetId() == PixelSubdetector::PixelBarrel;
      ladder.push_back(isBarrel ? tTopo.pxbLadder(id) : 0);
      blade.push_back(isBarrel ? 0 : tTopo.pxfBlade(id));
      panel.push_back(isBarrel ? 0 : tTopo.pxfPanel(id));
    }

    void clear() {
      ladder.clear();
      blade.clear();
      panel.clear();
    }

  private:
    std::vector<unsigned short> ladder;
    std::vector<unsigned short> blade;
    std::vector<unsigned short> panel;
  };

  class DetIdOTCommon {
  public:
    DetIdOTCommon() {}

    void book(const std::string& prefix, TTree* tree) {
      BOOK(order);
      BOOK(ring);
      BOOK(rod);
    }

    void push_back(const TrackerTopology& tTopo, const DetId& id) {
      const auto parsed = parse(tTopo, id);
      order.push_back(parsed.order);
      ring.push_back(parsed.ring);
      rod.push_back(parsed.rod);
    }

    void resize(size_t size) {
      order.resize(size);
      ring.resize(size);
      rod.resize(size);
    }

    void set(size_t index, const TrackerTopology& tTopo, const DetId& id) {
      const auto parsed = parse(tTopo, id);
      order[index] = parsed.order;
      ring[index] = parsed.ring;
      rod[index] = parsed.rod;
    }

    void clear() {
      order.clear();
      ring.clear();
      rod.clear();
    }

  private:
    struct Parsed {
      // use int here instead of short to avoid compilation errors due
      // to narrowing conversion (less boilerplate than explicit static_casts)
      unsigned int order = 0;
      unsigned int ring = 0;
      unsigned int rod = 0;
    };
    Parsed parse(const TrackerTopology& tTopo, const DetId& id) const {
      switch (id.subdetId()) {
        case StripSubdetector::TIB:
          return Parsed{tTopo.tibOrder(id), 0, 0};
        case StripSubdetector::TID:
          return Parsed{tTopo.tidOrder(id), tTopo.tidRing(id), 0};
        case StripSubdetector::TOB:
          return Parsed{0, 0, tTopo.tobRod(id)};
        case StripSubdetector::TEC:
          return Parsed{tTopo.tecOrder(id), tTopo.tecRing(id), 0};
        default:
          return Parsed{};
      };
    }

    std::vector<unsigned short> order;
    std::vector<unsigned short> ring;
    std::vector<unsigned short> rod;
  };

  class DetIdStripOnly {
  public:
    DetIdStripOnly() {}

    void book(const std::string& prefix, TTree* tree) {
      BOOK(string);
      BOOK(petalNumber);
      BOOK(isStereo);
      BOOK(isRPhi);
      BOOK(isGlued);
    }

    void push_back(const TrackerTopology& tTopo, const DetId& id) {
      const auto parsed = parse(tTopo, id);
      string.push_back(parsed.string);
      petalNumber.push_back(parsed.petalNumber);
      isStereo.push_back(tTopo.isStereo(id));
      isRPhi.push_back(tTopo.isRPhi(id));
      isGlued.push_back(parsed.glued);
    }

    void resize(size_t size) {
      string.resize(size);
      petalNumber.resize(size);
      isStereo.resize(size);
      isRPhi.resize(size);
      isGlued.resize(size);
    }

    void set(size_t index, const TrackerTopology& tTopo, const DetId& id) {
      const auto parsed = parse(tTopo, id);
      string[index] = parsed.string;
      petalNumber[index] = parsed.petalNumber;
      isStereo[index] = tTopo.isStereo(id);
      isRPhi[index] = tTopo.isRPhi(id);
      isGlued[index] = parsed.glued;
    }

    void clear() {
      string.clear();
      isStereo.clear();
      isRPhi.clear();
      isGlued.clear();
      petalNumber.clear();
    }

  private:
    struct Parsed {
      // use int here instead of short to avoid compilation errors due
      // to narrowing conversion (less boilerplate than explicit static_casts)
      unsigned int string = 0;
      unsigned int petalNumber = 0;
      bool glued = false;
    };
    Parsed parse(const TrackerTopology& tTopo, const DetId& id) const {
      switch (id.subdetId()) {
        case StripSubdetector::TIB:
          return Parsed{tTopo.tibString(id), 0, tTopo.tibIsDoubleSide(id)};
        case StripSubdetector::TID:
          return Parsed{0, 0, tTopo.tidIsDoubleSide(id)};
        case StripSubdetector::TOB:
          return Parsed{0, 0, tTopo.tobIsDoubleSide(id)};
        case StripSubdetector::TEC:
          return Parsed{0, tTopo.tecPetalNumber(id), tTopo.tecIsDoubleSide(id)};
        default:
          return Parsed{};
      }
    }

    std::vector<unsigned short> string;
    std::vector<unsigned short> petalNumber;
    std::vector<unsigned short> isStereo;
    std::vector<unsigned short> isRPhi;
    std::vector<unsigned short> isGlued;
  };

  class DetIdPhase2OTOnly {
  public:
    DetIdPhase2OTOnly() {}

    void book(const std::string& prefix, TTree* tree) {
      BOOK(isLower);
      BOOK(isUpper);
      BOOK(isStack);
    }

    void push_back(const TrackerTopology& tTopo, const DetId& id) {
      isLower.push_back(tTopo.isLower(id));
      isUpper.push_back(tTopo.isUpper(id));
      isStack.push_back(tTopo.stack(id) ==
                        0);  // equivalent to *IsDoubleSide() but without the hardcoded layer+ring requirements
    }

    void clear() {
      isLower.clear();
      isUpper.clear();
      isStack.clear();
    }

  private:
    std::vector<unsigned short> isLower;
    std::vector<unsigned short> isUpper;
    std::vector<unsigned short> isStack;
  };
#undef BOOK

  using DetIdPixel = CombineDetId<DetIdCommon, DetIdPixelOnly>;
  using DetIdStrip = CombineDetId<DetIdCommon, DetIdOTCommon, DetIdStripOnly>;
  using DetIdPhase2OT = CombineDetId<DetIdCommon, DetIdOTCommon, DetIdPhase2OTOnly>;
  using DetIdAll = CombineDetId<DetIdCommon, DetIdPixelOnly, DetIdOTCommon, DetIdStripOnly>;
  using DetIdAllPhase2 = CombineDetId<DetIdCommon, DetIdPixelOnly, DetIdOTCommon, DetIdPhase2OTOnly>;

  // event
  edm::RunNumber_t ev_run;
  edm::LuminosityBlockNumber_t ev_lumi;
  edm::EventNumber_t ev_event;

  ////////////////////
  // tracks
  // (first) index runs through tracks
  std::vector<float> trk_px;
  std::vector<float> trk_py;
  std::vector<float> trk_pz;
  std::vector<float> trk_pt;
  std::vector<float> trk_inner_px;
  std::vector<float> trk_inner_py;
  std::vector<float> trk_inner_pz;
  std::vector<float> trk_inner_pt;
  std::vector<float> trk_outer_px;
  std::vector<float> trk_outer_py;
  std::vector<float> trk_outer_pz;
  std::vector<float> trk_outer_pt;
  std::vector<float> trk_eta;
  std::vector<float> trk_lambda;
  std::vector<float> trk_cotTheta;
  std::vector<float> trk_phi;
  std::vector<float> trk_dxy;
  std::vector<float> trk_dz;
  std::vector<float> trk_dxyPV;
  std::vector<float> trk_dzPV;
  std::vector<float> trk_dxyClosestPV;
  std::vector<float> trk_dzClosestPV;
  std::vector<float> trk_ptErr;
  std::vector<float> trk_etaErr;
  std::vector<float> trk_lambdaErr;
  std::vector<float> trk_phiErr;
  std::vector<float> trk_dxyErr;
  std::vector<float> trk_dzErr;
  std::vector<float> trk_refpoint_x;
  std::vector<float> trk_refpoint_y;
  std::vector<float> trk_refpoint_z;
  std::vector<float> trk_nChi2;
  std::vector<float> trk_nChi2_1Dmod;
  std::vector<float> trk_ndof;
  std::vector<std::vector<float>> trk_mvas;
  std::vector<std::vector<unsigned short>> trk_qualityMasks;
  std::vector<int> trk_q;
  std::vector<unsigned int> trk_nValid;
  std::vector<unsigned int> trk_nLost;
  std::vector<unsigned int> trk_nInactive;
  std::vector<unsigned int> trk_nPixel;
  std::vector<unsigned int> trk_nStrip;
  std::vector<unsigned int> trk_nOuterLost;
  std::vector<unsigned int> trk_nInnerLost;
  std::vector<unsigned int> trk_nOuterInactive;
  std::vector<unsigned int> trk_nInnerInactive;
  std::vector<unsigned int> trk_nPixelLay;
  std::vector<unsigned int> trk_nStripLay;
  std::vector<unsigned int> trk_n3DLay;
  std::vector<unsigned int> trk_nLostLay;
  std::vector<unsigned int> trk_nCluster;
  std::vector<unsigned int> trk_algo;
  std::vector<unsigned int> trk_originalAlgo;
  std::vector<decltype(reco::TrackBase().algoMaskUL())> trk_algoMask;
  std::vector<unsigned short> trk_stopReason;
  std::vector<short> trk_isHP;
  std::vector<int> trk_seedIdx;
  std::vector<int> trk_vtxIdx;
  std::vector<short> trk_isTrue;
  std::vector<int> trk_bestSimTrkIdx;
  std::vector<float> trk_bestSimTrkShareFrac;
  std::vector<float> trk_bestSimTrkShareFracSimDenom;
  std::vector<float> trk_bestSimTrkShareFracSimClusterDenom;
  std::vector<float> trk_bestSimTrkNChi2;
  std::vector<int> trk_bestFromFirstHitSimTrkIdx;
  std::vector<float> trk_bestFromFirstHitSimTrkShareFrac;
  std::vector<float> trk_bestFromFirstHitSimTrkShareFracSimDenom;
  std::vector<float> trk_bestFromFirstHitSimTrkShareFracSimClusterDenom;
  std::vector<float> trk_bestFromFirstHitSimTrkNChi2;
  std::vector<std::vector<float>> trk_simTrkShareFrac;  // second index runs through matched TrackingParticles
  std::vector<std::vector<float>> trk_simTrkNChi2;      // second index runs through matched TrackingParticles
  std::vector<std::vector<int>> trk_simTrkIdx;          // second index runs through matched TrackingParticles
  std::vector<std::vector<int>> trk_hitIdx;             // second index runs through hits
  std::vector<std::vector<int>> trk_hitType;            // second index runs through hits
  ////////////////////
  // sim tracks
  // (first) index runs through TrackingParticles
  std::vector<int> sim_event;
  std::vector<int> sim_bunchCrossing;
  std::vector<int> sim_pdgId;
  std::vector<std::vector<int>> sim_genPdgIds;
  std::vector<int> sim_isFromBHadron;
  std::vector<float> sim_px;
  std::vector<float> sim_py;
  std::vector<float> sim_pz;
  std::vector<float> sim_pt;
  std::vector<float> sim_eta;
  std::vector<float> sim_phi;
  std::vector<float> sim_pca_pt;
  std::vector<float> sim_pca_eta;
  std::vector<float> sim_pca_lambda;
  std::vector<float> sim_pca_cotTheta;
  std::vector<float> sim_pca_phi;
  std::vector<float> sim_pca_dxy;
  std::vector<float> sim_pca_dz;
  std::vector<int> sim_q;
  // numbers of sim hits/layers
  std::vector<unsigned int> sim_nValid;
  std::vector<unsigned int> sim_nPixel;
  std::vector<unsigned int> sim_nStrip;
  std::vector<unsigned int> sim_nLay;
  std::vector<unsigned int> sim_nPixelLay;
  std::vector<unsigned int> sim_n3DLay;
  // number of sim hits as calculated in TrackingTruthAccumulator
  std::vector<unsigned int> sim_nTrackerHits;
  // number of clusters associated to TP
  std::vector<unsigned int> sim_nRecoClusters;
  // links to other objects
  std::vector<std::vector<int>> sim_trkIdx;          // second index runs through matched tracks
  std::vector<std::vector<float>> sim_trkShareFrac;  // second index runs through matched tracks
  std::vector<std::vector<int>> sim_seedIdx;         // second index runs through matched seeds
  std::vector<int> sim_parentVtxIdx;
  std::vector<std::vector<int>> sim_decayVtxIdx;  // second index runs through decay vertices
  std::vector<std::vector<int>> sim_simHitIdx;    // second index runs through SimHits
  ////////////////////
  // pixel hits
  // (first) index runs through hits
  std::vector<short> pix_isBarrel;
  DetIdPixel pix_detId;
  std::vector<std::vector<int>> pix_trkIdx;            // second index runs through tracks containing this hit
  std::vector<std::vector<int>> pix_seeIdx;            // second index runs through seeds containing this hit
  std::vector<std::vector<int>> pix_simHitIdx;         // second index runs through SimHits inducing this hit
  std::vector<std::vector<float>> pix_xySignificance;  // second index runs through SimHits inducing this hit
  std::vector<std::vector<float>> pix_chargeFraction;  // second index runs through SimHits inducing this hit
  std::vector<unsigned short> pix_simType;
  std::vector<float> pix_x;
  std::vector<float> pix_y;
  std::vector<float> pix_z;
  std::vector<float> pix_xx;
  std::vector<float> pix_xy;
  std::vector<float> pix_yy;
  std::vector<float> pix_yz;
  std::vector<float> pix_zz;
  std::vector<float> pix_zx;
  std::vector<float>
      pix_radL;  //http://cmslxr.fnal.gov/lxr/source/DataFormats/GeometrySurface/interface/MediumProperties.h
  std::vector<float> pix_bbxi;
  std::vector<int> pix_clustSizeCol;
  std::vector<int> pix_clustSizeRow;
  std::vector<uint64_t> pix_usedMask;
  ////////////////////
  // strip hits
  // (first) index runs through hits
  std::vector<short> str_isBarrel;
  DetIdStrip str_detId;
  std::vector<std::vector<int>> str_trkIdx;            // second index runs through tracks containing this hit
  std::vector<std::vector<int>> str_seeIdx;            // second index runs through seeds containing this hit
  std::vector<std::vector<int>> str_simHitIdx;         // second index runs through SimHits inducing this hit
  std::vector<std::vector<float>> str_xySignificance;  // second index runs through SimHits inducing this hit
  std::vector<std::vector<float>> str_chargeFraction;  // second index runs through SimHits inducing this hit
  std::vector<unsigned short> str_simType;
  std::vector<float> str_x;
  std::vector<float> str_y;
  std::vector<float> str_z;
  std::vector<float> str_xx;
  std::vector<float> str_xy;
  std::vector<float> str_yy;
  std::vector<float> str_yz;
  std::vector<float> str_zz;
  std::vector<float> str_zx;
  std::vector<float>
      str_radL;  //http://cmslxr.fnal.gov/lxr/source/DataFormats/GeometrySurface/interface/MediumProperties.h
  std::vector<float> str_bbxi;
  std::vector<float> str_chargePerCM;
  std::vector<int> str_clustSize;
  std::vector<uint64_t> str_usedMask;
  ////////////////////
  // strip matched hits
  // (first) index runs through hits
  std::vector<short> glu_isBarrel;
  DetIdStrip glu_detId;
  std::vector<int> glu_monoIdx;
  std::vector<int> glu_stereoIdx;
  std::vector<std::vector<int>> glu_seeIdx;  // second index runs through seeds containing this hit
  std::vector<float> glu_x;
  std::vector<float> glu_y;
  std::vector<float> glu_z;
  std::vector<float> glu_xx;
  std::vector<float> glu_xy;
  std::vector<float> glu_yy;
  std::vector<float> glu_yz;
  std::vector<float> glu_zz;
  std::vector<float> glu_zx;
  std::vector<float>
      glu_radL;  //http://cmslxr.fnal.gov/lxr/source/DataFormats/GeometrySurface/interface/MediumProperties.h
  std::vector<float> glu_bbxi;
  std::vector<float> glu_chargePerCM;
  std::vector<int> glu_clustSizeMono;
  std::vector<int> glu_clustSizeStereo;
  std::vector<uint64_t> glu_usedMaskMono;
  std::vector<uint64_t> glu_usedMaskStereo;
  ////////////////////
  // phase2 Outer Tracker hits
  // (first) index runs through hits
  std::vector<short> ph2_isBarrel;
  DetIdPhase2OT ph2_detId;
  std::vector<std::vector<int>> ph2_trkIdx;            // second index runs through tracks containing this hit
  std::vector<std::vector<int>> ph2_seeIdx;            // second index runs through seeds containing this hit
  std::vector<std::vector<int>> ph2_simHitIdx;         // second index runs through SimHits inducing this hit
  std::vector<std::vector<float>> ph2_xySignificance;  // second index runs through SimHits inducing this hit
  //std::vector<std::vector<float>> ph2_chargeFraction; // Not supported at the moment for Phase2
  std::vector<unsigned short> ph2_simType;
  std::vector<float> ph2_x;
  std::vector<float> ph2_y;
  std::vector<float> ph2_z;
  std::vector<float> ph2_xx;
  std::vector<float> ph2_xy;
  std::vector<float> ph2_yy;
  std::vector<float> ph2_yz;
  std::vector<float> ph2_zz;
  std::vector<float> ph2_zx;
  std::vector<float>
      ph2_radL;  //http://cmslxr.fnal.gov/lxr/source/DataFormats/GeometrySurface/interface/MediumProperties.h
  std::vector<float> ph2_bbxi;

  ////////////////////
  // invalid (missing/inactive/etc) hits
  // (first) index runs through hits
  std::vector<short> inv_isBarrel;
  DetIdAll inv_detId;
  DetIdAllPhase2 inv_detId_phase2;
  std::vector<unsigned short> inv_type;
  ////////////////////
  // sim hits
  // (first) index runs through hits
  DetIdAll simhit_detId;
  DetIdAllPhase2 simhit_detId_phase2;
  std::vector<float> simhit_x;
  std::vector<float> simhit_y;
  std::vector<float> simhit_z;
  std::vector<float> simhit_px;
  std::vector<float> simhit_py;
  std::vector<float> simhit_pz;
  std::vector<int> simhit_particle;
  std::vector<short> simhit_process;
  std::vector<float> simhit_eloss;
  std::vector<float> simhit_tof;
  //std::vector<unsigned int> simhit_simTrackId; // can be useful for debugging, but not much of general interest
  std::vector<int> simhit_simTrkIdx;
  std::vector<std::vector<int>> simhit_hitIdx;   // second index runs through induced reco hits
  std::vector<std::vector<int>> simhit_hitType;  // second index runs through induced reco hits
  ////////////////////
  // beam spot
  float bsp_x;
  float bsp_y;
  float bsp_z;
  float bsp_sigmax;
  float bsp_sigmay;
  float bsp_sigmaz;
  ////////////////////
  // seeds
  // (first) index runs through seeds
  std::vector<short> see_fitok;
  std::vector<float> see_px;
  std::vector<float> see_py;
  std::vector<float> see_pz;
  std::vector<float> see_pt;
  std::vector<float> see_eta;
  std::vector<float> see_phi;
  std::vector<float> see_dxy;
  std::vector<float> see_dz;
  std::vector<float> see_ptErr;
  std::vector<float> see_etaErr;
  std::vector<float> see_phiErr;
  std::vector<float> see_dxyErr;
  std::vector<float> see_dzErr;
  std::vector<float> see_chi2;
  std::vector<float> see_statePt;
  std::vector<float> see_stateTrajX;
  std::vector<float> see_stateTrajY;
  std::vector<float> see_stateTrajPx;
  std::vector<float> see_stateTrajPy;
  std::vector<float> see_stateTrajPz;
  std::vector<float> see_stateTrajGlbX;
  std::vector<float> see_stateTrajGlbY;
  std::vector<float> see_stateTrajGlbZ;
  std::vector<float> see_stateTrajGlbPx;
  std::vector<float> see_stateTrajGlbPy;
  std::vector<float> see_stateTrajGlbPz;
  std::vector<std::vector<float>> see_stateCurvCov;
  std::vector<int> see_q;
  std::vector<unsigned int> see_nValid;
  std::vector<unsigned int> see_nPixel;
  std::vector<unsigned int> see_nGlued;
  std::vector<unsigned int> see_nStrip;
  std::vector<unsigned int> see_nPhase2OT;
  std::vector<unsigned int> see_nCluster;
  std::vector<unsigned int> see_algo;
  std::vector<unsigned short> see_stopReason;
  std::vector<unsigned short> see_nCands;
  std::vector<int> see_trkIdx;
  std::vector<short> see_isTrue;
  std::vector<int> see_bestSimTrkIdx;
  std::vector<float> see_bestSimTrkShareFrac;
  std::vector<int> see_bestFromFirstHitSimTrkIdx;
  std::vector<float> see_bestFromFirstHitSimTrkShareFrac;
  std::vector<std::vector<float>> see_simTrkShareFrac;  // second index runs through matched TrackingParticles
  std::vector<std::vector<int>> see_simTrkIdx;          // second index runs through matched TrackingParticles
  std::vector<std::vector<int>> see_hitIdx;             // second index runs through hits
  std::vector<std::vector<int>> see_hitType;            // second index runs through hits
  //seed algo offset, index runs through iterations
  std::vector<unsigned int> see_offset;

  ////////////////////
  // Vertices
  // (first) index runs through vertices
  std::vector<float> vtx_x;
  std::vector<float> vtx_y;
  std::vector<float> vtx_z;
  std::vector<float> vtx_xErr;
  std::vector<float> vtx_yErr;
  std::vector<float> vtx_zErr;
  std::vector<float> vtx_ndof;
  std::vector<float> vtx_chi2;
  std::vector<short> vtx_fake;
  std::vector<short> vtx_valid;
  std::vector<std::vector<int>> vtx_trkIdx;  // second index runs through tracks used in the vertex fit

  ////////////////////
  // Tracking vertices
  // (first) index runs through TrackingVertices
  std::vector<int> simvtx_event;
  std::vector<int> simvtx_bunchCrossing;
  std::vector<unsigned int> simvtx_processType;  // only from first SimVertex of TrackingVertex
  std::vector<float> simvtx_x;
  std::vector<float> simvtx_y;
  std::vector<float> simvtx_z;
  std::vector<std::vector<int>> simvtx_sourceSimIdx;    // second index runs through source TrackingParticles
  std::vector<std::vector<int>> simvtx_daughterSimIdx;  // second index runs through daughter TrackingParticles
  std::vector<int> simpv_idx;
};

//
// constructors and destructor
//
TrackingNtuple::TrackingNtuple(const edm::ParameterSet& iConfig)
    : mfToken_(esConsumes()),
      ttrhToken_(esConsumes(edm::ESInputTag("", iConfig.getUntrackedParameter<std::string>("TTRHBuilder")))),
      tTopoToken_(esConsumes()),
      tGeomToken_(esConsumes()),
      paramsDefineToken_(
          esConsumes(edm::ESInputTag("", iConfig.getUntrackedParameter<std::string>("parametersDefiner")))),
      trackToken_(consumes<edm::View<reco::Track>>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
      clusterTPMapToken_(consumes<ClusterTPAssociation>(iConfig.getUntrackedParameter<edm::InputTag>("clusterTPMap"))),
      simHitTPMapToken_(consumes<SimHitTPAssociationProducer::SimHitTPAssociationList>(
          iConfig.getUntrackedParameter<edm::InputTag>("simHitTPMap"))),
      trackAssociatorToken_(consumes<reco::TrackToTrackingParticleAssociator>(
          iConfig.getUntrackedParameter<edm::InputTag>("trackAssociator"))),
      pixelSimLinkToken_(consumes<edm::DetSetVector<PixelDigiSimLink>>(
          iConfig.getUntrackedParameter<edm::InputTag>("pixelDigiSimLink"))),
      stripSimLinkToken_(consumes<edm::DetSetVector<StripDigiSimLink>>(
          iConfig.getUntrackedParameter<edm::InputTag>("stripDigiSimLink"))),
      siphase2OTSimLinksToken_(consumes<edm::DetSetVector<PixelDigiSimLink>>(
          iConfig.getUntrackedParameter<edm::InputTag>("phase2OTSimLink"))),
      includeStripHits_(!iConfig.getUntrackedParameter<edm::InputTag>("stripDigiSimLink").label().empty()),
      includePhase2OTHits_(!iConfig.getUntrackedParameter<edm::InputTag>("phase2OTSimLink").label().empty()),
      beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getUntrackedParameter<edm::InputTag>("beamSpot"))),
      pixelRecHitToken_(
          consumes<SiPixelRecHitCollection>(iConfig.getUntrackedParameter<edm::InputTag>("pixelRecHits"))),
      stripRphiRecHitToken_(
          consumes<SiStripRecHit2DCollection>(iConfig.getUntrackedParameter<edm::InputTag>("stripRphiRecHits"))),
      stripStereoRecHitToken_(
          consumes<SiStripRecHit2DCollection>(iConfig.getUntrackedParameter<edm::InputTag>("stripStereoRecHits"))),
      stripMatchedRecHitToken_(consumes<SiStripMatchedRecHit2DCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("stripMatchedRecHits"))),
      phase2OTRecHitToken_(consumes<Phase2TrackerRecHit1DCollectionNew>(
          iConfig.getUntrackedParameter<edm::InputTag>("phase2OTRecHits"))),
      vertexToken_(consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("vertices"))),
      trackingVertexToken_(
          consumes<TrackingVertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("trackingVertices"))),
      tpNLayersToken_(consumes<edm::ValueMap<unsigned int>>(
          iConfig.getUntrackedParameter<edm::InputTag>("trackingParticleNlayers"))),
      tpNPixelLayersToken_(consumes<edm::ValueMap<unsigned int>>(
          iConfig.getUntrackedParameter<edm::InputTag>("trackingParticleNpixellayers"))),
      tpNStripStereoLayersToken_(consumes<edm::ValueMap<unsigned int>>(
          iConfig.getUntrackedParameter<edm::InputTag>("trackingParticleNstripstereolayers"))),
      includeSeeds_(iConfig.getUntrackedParameter<bool>("includeSeeds")),
      addSeedCurvCov_(iConfig.getUntrackedParameter<bool>("addSeedCurvCov")),
      includeAllHits_(iConfig.getUntrackedParameter<bool>("includeAllHits")),
      includeMVA_(iConfig.getUntrackedParameter<bool>("includeMVA")),
      includeTrackingParticles_(iConfig.getUntrackedParameter<bool>("includeTrackingParticles")),
      includeOOT_(iConfig.getUntrackedParameter<bool>("includeOOT")),
      keepEleSimHits_(iConfig.getUntrackedParameter<bool>("keepEleSimHits")),
      saveSimHitsP3_(iConfig.getUntrackedParameter<bool>("saveSimHitsP3")),
      simHitBySignificance_(iConfig.getUntrackedParameter<bool>("simHitBySignificance")) {
  if (includeSeeds_) {
    seedTokens_ =
        edm::vector_transform(iConfig.getUntrackedParameter<std::vector<edm::InputTag>>("seedTracks"),
                              [&](const edm::InputTag& tag) { return consumes<edm::View<reco::Track>>(tag); });
    seedStopInfoTokens_ =
        edm::vector_transform(iConfig.getUntrackedParameter<std::vector<edm::InputTag>>("trackCandidates"),
                              [&](const edm::InputTag& tag) { return consumes<std::vector<SeedStopInfo>>(tag); });
    if (seedTokens_.size() != seedStopInfoTokens_.size()) {
      throw cms::Exception("Configuration") << "Got " << seedTokens_.size() << " seed collections, but "
                                            << seedStopInfoTokens_.size() << " track candidate collections";
    }
  }

  if (includeAllHits_) {
    if (includeStripHits_ && includePhase2OTHits_) {
      throw cms::Exception("Configuration")
          << "Both stripDigiSimLink and phase2OTSimLink are set, please set only either one (this information is used "
             "to infer if you're running phase0/1 or phase2 detector)";
    }
    if (!includeStripHits_ && !includePhase2OTHits_) {
      throw cms::Exception("Configuration")
          << "Neither stripDigiSimLink or phase2OTSimLink are set, please set either one.";
    }

    auto const& maskVPset = iConfig.getUntrackedParameterSetVector("clusterMasks");
    pixelUseMaskTokens_.reserve(maskVPset.size());
    stripUseMaskTokens_.reserve(maskVPset.size());
    for (auto const& mask : maskVPset) {
      auto index = mask.getUntrackedParameter<unsigned int>("index");
      assert(index < 64);
      pixelUseMaskTokens_.emplace_back(index,
                                       consumes<PixelMaskContainer>(mask.getUntrackedParameter<edm::InputTag>("src")));
      if (includeStripHits_)
        stripUseMaskTokens_.emplace_back(
            index, consumes<StripMaskContainer>(mask.getUntrackedParameter<edm::InputTag>("src")));
    }
  }

  const bool tpRef = iConfig.getUntrackedParameter<bool>("trackingParticlesRef");
  const auto tpTag = iConfig.getUntrackedParameter<edm::InputTag>("trackingParticles");
  if (tpRef) {
    trackingParticleRefToken_ = consumes<TrackingParticleRefVector>(tpTag);
  } else {
    trackingParticleToken_ = consumes<TrackingParticleCollection>(tpTag);
  }

  tracer_.depth(-2);  // as in SimTracker/TrackHistory/src/TrackClassifier.cc

  if (includeMVA_) {
    mvaQualityCollectionTokens_ = edm::vector_transform(
        iConfig.getUntrackedParameter<std::vector<std::string>>("trackMVAs"), [&](const std::string& tag) {
          return std::make_tuple(consumes<MVACollection>(edm::InputTag(tag, "MVAValues")),
                                 consumes<QualityMaskCollection>(edm::InputTag(tag, "QualityMasks")));
        });
  }

  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  t = fs->make<TTree>("tree", "tree");

  t->Branch("event", &ev_event);
  t->Branch("lumi", &ev_lumi);
  t->Branch("run", &ev_run);

  //tracks
  t->Branch("trk_px", &trk_px);
  t->Branch("trk_py", &trk_py);
  t->Branch("trk_pz", &trk_pz);
  t->Branch("trk_pt", &trk_pt);
  t->Branch("trk_inner_px", &trk_inner_px);
  t->Branch("trk_inner_py", &trk_inner_py);
  t->Branch("trk_inner_pz", &trk_inner_pz);
  t->Branch("trk_inner_pt", &trk_inner_pt);
  t->Branch("trk_outer_px", &trk_outer_px);
  t->Branch("trk_outer_py", &trk_outer_py);
  t->Branch("trk_outer_pz", &trk_outer_pz);
  t->Branch("trk_outer_pt", &trk_outer_pt);
  t->Branch("trk_eta", &trk_eta);
  t->Branch("trk_lambda", &trk_lambda);
  t->Branch("trk_cotTheta", &trk_cotTheta);
  t->Branch("trk_phi", &trk_phi);
  t->Branch("trk_dxy", &trk_dxy);
  t->Branch("trk_dz", &trk_dz);
  t->Branch("trk_dxyPV", &trk_dxyPV);
  t->Branch("trk_dzPV", &trk_dzPV);
  t->Branch("trk_dxyClosestPV", &trk_dxyClosestPV);
  t->Branch("trk_dzClosestPV", &trk_dzClosestPV);
  t->Branch("trk_ptErr", &trk_ptErr);
  t->Branch("trk_etaErr", &trk_etaErr);
  t->Branch("trk_lambdaErr", &trk_lambdaErr);
  t->Branch("trk_phiErr", &trk_phiErr);
  t->Branch("trk_dxyErr", &trk_dxyErr);
  t->Branch("trk_dzErr", &trk_dzErr);
  t->Branch("trk_refpoint_x", &trk_refpoint_x);
  t->Branch("trk_refpoint_y", &trk_refpoint_y);
  t->Branch("trk_refpoint_z", &trk_refpoint_z);
  t->Branch("trk_nChi2", &trk_nChi2);
  t->Branch("trk_nChi2_1Dmod", &trk_nChi2_1Dmod);
  t->Branch("trk_ndof", &trk_ndof);
  if (includeMVA_) {
    trk_mvas.resize(mvaQualityCollectionTokens_.size());
    trk_qualityMasks.resize(mvaQualityCollectionTokens_.size());
    if (!trk_mvas.empty()) {
      t->Branch("trk_mva", &(trk_mvas[0]));
      t->Branch("trk_qualityMask", &(trk_qualityMasks[0]));
      for (size_t i = 1; i < trk_mvas.size(); ++i) {
        t->Branch(("trk_mva" + std::to_string(i + 1)).c_str(), &(trk_mvas[i]));
        t->Branch(("trk_qualityMask" + std::to_string(i + 1)).c_str(), &(trk_qualityMasks[i]));
      }
    }
  }
  t->Branch("trk_q", &trk_q);
  t->Branch("trk_nValid", &trk_nValid);
  t->Branch("trk_nLost", &trk_nLost);
  t->Branch("trk_nInactive", &trk_nInactive);
  t->Branch("trk_nPixel", &trk_nPixel);
  t->Branch("trk_nStrip", &trk_nStrip);
  t->Branch("trk_nOuterLost", &trk_nOuterLost);
  t->Branch("trk_nInnerLost", &trk_nInnerLost);
  t->Branch("trk_nOuterInactive", &trk_nOuterInactive);
  t->Branch("trk_nInnerInactive", &trk_nInnerInactive);
  t->Branch("trk_nPixelLay", &trk_nPixelLay);
  t->Branch("trk_nStripLay", &trk_nStripLay);
  t->Branch("trk_n3DLay", &trk_n3DLay);
  t->Branch("trk_nLostLay", &trk_nLostLay);
  t->Branch("trk_nCluster", &trk_nCluster);
  t->Branch("trk_algo", &trk_algo);
  t->Branch("trk_originalAlgo", &trk_originalAlgo);
  t->Branch("trk_algoMask", &trk_algoMask);
  t->Branch("trk_stopReason", &trk_stopReason);
  t->Branch("trk_isHP", &trk_isHP);
  if (includeSeeds_) {
    t->Branch("trk_seedIdx", &trk_seedIdx);
  }
  t->Branch("trk_vtxIdx", &trk_vtxIdx);
  if (includeTrackingParticles_) {
    t->Branch("trk_simTrkIdx", &trk_simTrkIdx);
    t->Branch("trk_simTrkShareFrac", &trk_simTrkShareFrac);
    t->Branch("trk_simTrkNChi2", &trk_simTrkNChi2);
    t->Branch("trk_bestSimTrkIdx", &trk_bestSimTrkIdx);
    t->Branch("trk_bestFromFirstHitSimTrkIdx", &trk_bestFromFirstHitSimTrkIdx);
  } else {
    t->Branch("trk_isTrue", &trk_isTrue);
  }
  t->Branch("trk_bestSimTrkShareFrac", &trk_bestSimTrkShareFrac);
  t->Branch("trk_bestSimTrkShareFracSimDenom", &trk_bestSimTrkShareFracSimDenom);
  t->Branch("trk_bestSimTrkShareFracSimClusterDenom", &trk_bestSimTrkShareFracSimClusterDenom);
  t->Branch("trk_bestSimTrkNChi2", &trk_bestSimTrkNChi2);
  t->Branch("trk_bestFromFirstHitSimTrkShareFrac", &trk_bestFromFirstHitSimTrkShareFrac);
  t->Branch("trk_bestFromFirstHitSimTrkShareFracSimDenom", &trk_bestFromFirstHitSimTrkShareFracSimDenom);
  t->Branch("trk_bestFromFirstHitSimTrkShareFracSimClusterDenom", &trk_bestFromFirstHitSimTrkShareFracSimClusterDenom);
  t->Branch("trk_bestFromFirstHitSimTrkNChi2", &trk_bestFromFirstHitSimTrkNChi2);
  if (includeAllHits_) {
    t->Branch("trk_hitIdx", &trk_hitIdx);
    t->Branch("trk_hitType", &trk_hitType);
  }
  if (includeTrackingParticles_) {
    //sim tracks
    t->Branch("sim_event", &sim_event);
    t->Branch("sim_bunchCrossing", &sim_bunchCrossing);
    t->Branch("sim_pdgId", &sim_pdgId);
    t->Branch("sim_genPdgIds", &sim_genPdgIds);
    t->Branch("sim_isFromBHadron", &sim_isFromBHadron);
    t->Branch("sim_px", &sim_px);
    t->Branch("sim_py", &sim_py);
    t->Branch("sim_pz", &sim_pz);
    t->Branch("sim_pt", &sim_pt);
    t->Branch("sim_eta", &sim_eta);
    t->Branch("sim_phi", &sim_phi);
    t->Branch("sim_pca_pt", &sim_pca_pt);
    t->Branch("sim_pca_eta", &sim_pca_eta);
    t->Branch("sim_pca_lambda", &sim_pca_lambda);
    t->Branch("sim_pca_cotTheta", &sim_pca_cotTheta);
    t->Branch("sim_pca_phi", &sim_pca_phi);
    t->Branch("sim_pca_dxy", &sim_pca_dxy);
    t->Branch("sim_pca_dz", &sim_pca_dz);
    t->Branch("sim_q", &sim_q);
    t->Branch("sim_nValid", &sim_nValid);
    t->Branch("sim_nPixel", &sim_nPixel);
    t->Branch("sim_nStrip", &sim_nStrip);
    t->Branch("sim_nLay", &sim_nLay);
    t->Branch("sim_nPixelLay", &sim_nPixelLay);
    t->Branch("sim_n3DLay", &sim_n3DLay);
    t->Branch("sim_nTrackerHits", &sim_nTrackerHits);
    t->Branch("sim_nRecoClusters", &sim_nRecoClusters);
    t->Branch("sim_trkIdx", &sim_trkIdx);
    t->Branch("sim_trkShareFrac", &sim_trkShareFrac);
    if (includeSeeds_) {
      t->Branch("sim_seedIdx", &sim_seedIdx);
    }
    t->Branch("sim_parentVtxIdx", &sim_parentVtxIdx);
    t->Branch("sim_decayVtxIdx", &sim_decayVtxIdx);
    if (includeAllHits_) {
      t->Branch("sim_simHitIdx", &sim_simHitIdx);
    }
  }
  if (includeAllHits_) {
    //pixels
    t->Branch("pix_isBarrel", &pix_isBarrel);
    pix_detId.book("pix", t);
    t->Branch("pix_trkIdx", &pix_trkIdx);
    if (includeSeeds_) {
      t->Branch("pix_seeIdx", &pix_seeIdx);
    }
    if (includeTrackingParticles_) {
      t->Branch("pix_simHitIdx", &pix_simHitIdx);
      if (simHitBySignificance_) {
        t->Branch("pix_xySignificance", &pix_xySignificance);
      }
      t->Branch("pix_chargeFraction", &pix_chargeFraction);
      t->Branch("pix_simType", &pix_simType);
    }
    t->Branch("pix_x", &pix_x);
    t->Branch("pix_y", &pix_y);
    t->Branch("pix_z", &pix_z);
    t->Branch("pix_xx", &pix_xx);
    t->Branch("pix_xy", &pix_xy);
    t->Branch("pix_yy", &pix_yy);
    t->Branch("pix_yz", &pix_yz);
    t->Branch("pix_zz", &pix_zz);
    t->Branch("pix_zx", &pix_zx);
    t->Branch("pix_radL", &pix_radL);
    t->Branch("pix_bbxi", &pix_bbxi);
    t->Branch("pix_clustSizeCol", &pix_clustSizeCol);
    t->Branch("pix_clustSizeRow", &pix_clustSizeRow);
    t->Branch("pix_usedMask", &pix_usedMask);
    //strips
    if (includeStripHits_) {
      t->Branch("str_isBarrel", &str_isBarrel);
      str_detId.book("str", t);
      t->Branch("str_trkIdx", &str_trkIdx);
      if (includeSeeds_) {
        t->Branch("str_seeIdx", &str_seeIdx);
      }
      if (includeTrackingParticles_) {
        t->Branch("str_simHitIdx", &str_simHitIdx);
        if (simHitBySignificance_) {
          t->Branch("str_xySignificance", &str_xySignificance);
        }
        t->Branch("str_chargeFraction", &str_chargeFraction);
        t->Branch("str_simType", &str_simType);
      }
      t->Branch("str_x", &str_x);
      t->Branch("str_y", &str_y);
      t->Branch("str_z", &str_z);
      t->Branch("str_xx", &str_xx);
      t->Branch("str_xy", &str_xy);
      t->Branch("str_yy", &str_yy);
      t->Branch("str_yz", &str_yz);
      t->Branch("str_zz", &str_zz);
      t->Branch("str_zx", &str_zx);
      t->Branch("str_radL", &str_radL);
      t->Branch("str_bbxi", &str_bbxi);
      t->Branch("str_chargePerCM", &str_chargePerCM);
      t->Branch("str_clustSize", &str_clustSize);
      t->Branch("str_usedMask", &str_usedMask);
      //matched hits
      t->Branch("glu_isBarrel", &glu_isBarrel);
      glu_detId.book("glu", t);
      t->Branch("glu_monoIdx", &glu_monoIdx);
      t->Branch("glu_stereoIdx", &glu_stereoIdx);
      if (includeSeeds_) {
        t->Branch("glu_seeIdx", &glu_seeIdx);
      }
      t->Branch("glu_x", &glu_x);
      t->Branch("glu_y", &glu_y);
      t->Branch("glu_z", &glu_z);
      t->Branch("glu_xx", &glu_xx);
      t->Branch("glu_xy", &glu_xy);
      t->Branch("glu_yy", &glu_yy);
      t->Branch("glu_yz", &glu_yz);
      t->Branch("glu_zz", &glu_zz);
      t->Branch("glu_zx", &glu_zx);
      t->Branch("glu_radL", &glu_radL);
      t->Branch("glu_bbxi", &glu_bbxi);
      t->Branch("glu_chargePerCM", &glu_chargePerCM);
      t->Branch("glu_clustSizeMono", &glu_clustSizeMono);
      t->Branch("glu_clustSizeStereo", &glu_clustSizeStereo);
      t->Branch("glu_usedMaskMono", &glu_usedMaskMono);
      t->Branch("glu_usedMaskStereo", &glu_usedMaskStereo);
    }
    //phase2 OT
    if (includePhase2OTHits_) {
      t->Branch("ph2_isBarrel", &ph2_isBarrel);
      ph2_detId.book("ph2", t);
      t->Branch("ph2_trkIdx", &ph2_trkIdx);
      if (includeSeeds_) {
        t->Branch("ph2_seeIdx", &ph2_seeIdx);
      }
      if (includeTrackingParticles_) {
        t->Branch("ph2_simHitIdx", &ph2_simHitIdx);
        if (simHitBySignificance_) {
          t->Branch("ph2_xySignificance", &ph2_xySignificance);
        }
        t->Branch("ph2_simType", &ph2_simType);
      }
      t->Branch("ph2_x", &ph2_x);
      t->Branch("ph2_y", &ph2_y);
      t->Branch("ph2_z", &ph2_z);
      t->Branch("ph2_xx", &ph2_xx);
      t->Branch("ph2_xy", &ph2_xy);
      t->Branch("ph2_yy", &ph2_yy);
      t->Branch("ph2_yz", &ph2_yz);
      t->Branch("ph2_zz", &ph2_zz);
      t->Branch("ph2_zx", &ph2_zx);
      t->Branch("ph2_radL", &ph2_radL);
      t->Branch("ph2_bbxi", &ph2_bbxi);
      t->Branch("ph2_bbxi", &ph2_bbxi);
    }
    //invalid hits
    t->Branch("inv_isBarrel", &inv_isBarrel);
    if (includeStripHits_)
      inv_detId.book("inv", t);
    else
      inv_detId_phase2.book("inv", t);
    t->Branch("inv_type", &inv_type);
    //simhits
    if (includeTrackingParticles_) {
      if (includeStripHits_)
        simhit_detId.book("simhit", t);
      else
        simhit_detId_phase2.book("simhit", t);
      t->Branch("simhit_x", &simhit_x);
      t->Branch("simhit_y", &simhit_y);
      t->Branch("simhit_z", &simhit_z);
      if (saveSimHitsP3_) {
        t->Branch("simhit_px", &simhit_px);
        t->Branch("simhit_py", &simhit_py);
        t->Branch("simhit_pz", &simhit_pz);
      }
      t->Branch("simhit_particle", &simhit_particle);
      t->Branch("simhit_process", &simhit_process);
      t->Branch("simhit_eloss", &simhit_eloss);
      t->Branch("simhit_tof", &simhit_tof);
      t->Branch("simhit_simTrkIdx", &simhit_simTrkIdx);
      t->Branch("simhit_hitIdx", &simhit_hitIdx);
      t->Branch("simhit_hitType", &simhit_hitType);
    }
  }
  //beam spot
  t->Branch("bsp_x", &bsp_x, "bsp_x/F");
  t->Branch("bsp_y", &bsp_y, "bsp_y/F");
  t->Branch("bsp_z", &bsp_z, "bsp_z/F");
  t->Branch("bsp_sigmax", &bsp_sigmax, "bsp_sigmax/F");
  t->Branch("bsp_sigmay", &bsp_sigmay, "bsp_sigmay/F");
  t->Branch("bsp_sigmaz", &bsp_sigmaz, "bsp_sigmaz/F");
  if (includeSeeds_) {
    //seeds
    t->Branch("see_fitok", &see_fitok);
    t->Branch("see_px", &see_px);
    t->Branch("see_py", &see_py);
    t->Branch("see_pz", &see_pz);
    t->Branch("see_pt", &see_pt);
    t->Branch("see_eta", &see_eta);
    t->Branch("see_phi", &see_phi);
    t->Branch("see_dxy", &see_dxy);
    t->Branch("see_dz", &see_dz);
    t->Branch("see_ptErr", &see_ptErr);
    t->Branch("see_etaErr", &see_etaErr);
    t->Branch("see_phiErr", &see_phiErr);
    t->Branch("see_dxyErr", &see_dxyErr);
    t->Branch("see_dzErr", &see_dzErr);
    t->Branch("see_chi2", &see_chi2);
    t->Branch("see_statePt", &see_statePt);
    t->Branch("see_stateTrajX", &see_stateTrajX);
    t->Branch("see_stateTrajY", &see_stateTrajY);
    t->Branch("see_stateTrajPx", &see_stateTrajPx);
    t->Branch("see_stateTrajPy", &see_stateTrajPy);
    t->Branch("see_stateTrajPz", &see_stateTrajPz);
    t->Branch("see_stateTrajGlbX", &see_stateTrajGlbX);
    t->Branch("see_stateTrajGlbY", &see_stateTrajGlbY);
    t->Branch("see_stateTrajGlbZ", &see_stateTrajGlbZ);
    t->Branch("see_stateTrajGlbPx", &see_stateTrajGlbPx);
    t->Branch("see_stateTrajGlbPy", &see_stateTrajGlbPy);
    t->Branch("see_stateTrajGlbPz", &see_stateTrajGlbPz);
    if (addSeedCurvCov_) {
      t->Branch("see_stateCurvCov", &see_stateCurvCov);
    }
    t->Branch("see_q", &see_q);
    t->Branch("see_nValid", &see_nValid);
    t->Branch("see_nPixel", &see_nPixel);
    t->Branch("see_nGlued", &see_nGlued);
    t->Branch("see_nStrip", &see_nStrip);
    t->Branch("see_nPhase2OT", &see_nPhase2OT);
    t->Branch("see_nCluster", &see_nCluster);
    t->Branch("see_algo", &see_algo);
    t->Branch("see_stopReason", &see_stopReason);
    t->Branch("see_nCands", &see_nCands);
    t->Branch("see_trkIdx", &see_trkIdx);
    if (includeTrackingParticles_) {
      t->Branch("see_simTrkIdx", &see_simTrkIdx);
      t->Branch("see_simTrkShareFrac", &see_simTrkShareFrac);
      t->Branch("see_bestSimTrkIdx", &see_bestSimTrkIdx);
      t->Branch("see_bestFromFirstHitSimTrkIdx", &see_bestFromFirstHitSimTrkIdx);
    } else {
      t->Branch("see_isTrue", &see_isTrue);
    }
    t->Branch("see_bestSimTrkShareFrac", &see_bestSimTrkShareFrac);
    t->Branch("see_bestFromFirstHitSimTrkShareFrac", &see_bestFromFirstHitSimTrkShareFrac);
    if (includeAllHits_) {
      t->Branch("see_hitIdx", &see_hitIdx);
      t->Branch("see_hitType", &see_hitType);
    }
    //seed algo offset
    t->Branch("see_offset", &see_offset);
  }

  //vertices
  t->Branch("vtx_x", &vtx_x);
  t->Branch("vtx_y", &vtx_y);
  t->Branch("vtx_z", &vtx_z);
  t->Branch("vtx_xErr", &vtx_xErr);
  t->Branch("vtx_yErr", &vtx_yErr);
  t->Branch("vtx_zErr", &vtx_zErr);
  t->Branch("vtx_ndof", &vtx_ndof);
  t->Branch("vtx_chi2", &vtx_chi2);
  t->Branch("vtx_fake", &vtx_fake);
  t->Branch("vtx_valid", &vtx_valid);
  t->Branch("vtx_trkIdx", &vtx_trkIdx);

  // tracking vertices
  t->Branch("simvtx_event", &simvtx_event);
  t->Branch("simvtx_bunchCrossing", &simvtx_bunchCrossing);
  t->Branch("simvtx_processType", &simvtx_processType);
  t->Branch("simvtx_x", &simvtx_x);
  t->Branch("simvtx_y", &simvtx_y);
  t->Branch("simvtx_z", &simvtx_z);
  t->Branch("simvtx_sourceSimIdx", &simvtx_sourceSimIdx);
  t->Branch("simvtx_daughterSimIdx", &simvtx_daughterSimIdx);

  t->Branch("simpv_idx", &simpv_idx);

  //t->Branch("" , &);
}

TrackingNtuple::~TrackingNtuple() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//
void TrackingNtuple::clearVariables() {
  ev_run = 0;
  ev_lumi = 0;
  ev_event = 0;

  //tracks
  trk_px.clear();
  trk_py.clear();
  trk_pz.clear();
  trk_pt.clear();
  trk_inner_px.clear();
  trk_inner_py.clear();
  trk_inner_pz.clear();
  trk_inner_pt.clear();
  trk_outer_px.clear();
  trk_outer_py.clear();
  trk_outer_pz.clear();
  trk_outer_pt.clear();
  trk_eta.clear();
  trk_lambda.clear();
  trk_cotTheta.clear();
  trk_phi.clear();
  trk_dxy.clear();
  trk_dz.clear();
  trk_dxyPV.clear();
  trk_dzPV.clear();
  trk_dxyClosestPV.clear();
  trk_dzClosestPV.clear();
  trk_ptErr.clear();
  trk_etaErr.clear();
  trk_lambdaErr.clear();
  trk_phiErr.clear();
  trk_dxyErr.clear();
  trk_dzErr.clear();
  trk_refpoint_x.clear();
  trk_refpoint_y.clear();
  trk_refpoint_z.clear();
  trk_nChi2.clear();
  trk_nChi2_1Dmod.clear();
  trk_ndof.clear();
  for (auto& mva : trk_mvas) {
    mva.clear();
  }
  for (auto& mask : trk_qualityMasks) {
    mask.clear();
  }
  trk_q.clear();
  trk_nValid.clear();
  trk_nLost.clear();
  trk_nInactive.clear();
  trk_nPixel.clear();
  trk_nStrip.clear();
  trk_nOuterLost.clear();
  trk_nInnerLost.clear();
  trk_nOuterInactive.clear();
  trk_nInnerInactive.clear();
  trk_nPixelLay.clear();
  trk_nStripLay.clear();
  trk_n3DLay.clear();
  trk_nLostLay.clear();
  trk_nCluster.clear();
  trk_algo.clear();
  trk_originalAlgo.clear();
  trk_algoMask.clear();
  trk_stopReason.clear();
  trk_isHP.clear();
  trk_seedIdx.clear();
  trk_vtxIdx.clear();
  trk_isTrue.clear();
  trk_bestSimTrkIdx.clear();
  trk_bestSimTrkShareFrac.clear();
  trk_bestSimTrkShareFracSimDenom.clear();
  trk_bestSimTrkShareFracSimClusterDenom.clear();
  trk_bestSimTrkNChi2.clear();
  trk_bestFromFirstHitSimTrkIdx.clear();
  trk_bestFromFirstHitSimTrkShareFrac.clear();
  trk_bestFromFirstHitSimTrkShareFracSimDenom.clear();
  trk_bestFromFirstHitSimTrkShareFracSimClusterDenom.clear();
  trk_bestFromFirstHitSimTrkNChi2.clear();
  trk_simTrkIdx.clear();
  trk_simTrkShareFrac.clear();
  trk_simTrkNChi2.clear();
  trk_hitIdx.clear();
  trk_hitType.clear();
  //sim tracks
  sim_event.clear();
  sim_bunchCrossing.clear();
  sim_pdgId.clear();
  sim_genPdgIds.clear();
  sim_isFromBHadron.clear();
  sim_px.clear();
  sim_py.clear();
  sim_pz.clear();
  sim_pt.clear();
  sim_eta.clear();
  sim_phi.clear();
  sim_pca_pt.clear();
  sim_pca_eta.clear();
  sim_pca_lambda.clear();
  sim_pca_cotTheta.clear();
  sim_pca_phi.clear();
  sim_pca_dxy.clear();
  sim_pca_dz.clear();
  sim_q.clear();
  sim_nValid.clear();
  sim_nPixel.clear();
  sim_nStrip.clear();
  sim_nLay.clear();
  sim_nPixelLay.clear();
  sim_n3DLay.clear();
  sim_nTrackerHits.clear();
  sim_nRecoClusters.clear();
  sim_trkIdx.clear();
  sim_seedIdx.clear();
  sim_trkShareFrac.clear();
  sim_parentVtxIdx.clear();
  sim_decayVtxIdx.clear();
  sim_simHitIdx.clear();
  //pixels
  pix_isBarrel.clear();
  pix_detId.clear();
  pix_trkIdx.clear();
  pix_seeIdx.clear();
  pix_simHitIdx.clear();
  pix_xySignificance.clear();
  pix_chargeFraction.clear();
  pix_simType.clear();
  pix_x.clear();
  pix_y.clear();
  pix_z.clear();
  pix_xx.clear();
  pix_xy.clear();
  pix_yy.clear();
  pix_yz.clear();
  pix_zz.clear();
  pix_zx.clear();
  pix_radL.clear();
  pix_bbxi.clear();
  pix_clustSizeCol.clear();
  pix_clustSizeRow.clear();
  pix_usedMask.clear();
  //strips
  str_isBarrel.clear();
  str_detId.clear();
  str_trkIdx.clear();
  str_seeIdx.clear();
  str_simHitIdx.clear();
  str_xySignificance.clear();
  str_chargeFraction.clear();
  str_simType.clear();
  str_x.clear();
  str_y.clear();
  str_z.clear();
  str_xx.clear();
  str_xy.clear();
  str_yy.clear();
  str_yz.clear();
  str_zz.clear();
  str_zx.clear();
  str_radL.clear();
  str_bbxi.clear();
  str_chargePerCM.clear();
  str_clustSize.clear();
  str_usedMask.clear();
  //matched hits
  glu_isBarrel.clear();
  glu_detId.clear();
  glu_monoIdx.clear();
  glu_stereoIdx.clear();
  glu_seeIdx.clear();
  glu_x.clear();
  glu_y.clear();
  glu_z.clear();
  glu_xx.clear();
  glu_xy.clear();
  glu_yy.clear();
  glu_yz.clear();
  glu_zz.clear();
  glu_zx.clear();
  glu_radL.clear();
  glu_bbxi.clear();
  glu_chargePerCM.clear();
  glu_clustSizeMono.clear();
  glu_clustSizeStereo.clear();
  glu_usedMaskMono.clear();
  glu_usedMaskStereo.clear();
  //phase2 OT
  ph2_isBarrel.clear();
  ph2_detId.clear();
  ph2_trkIdx.clear();
  ph2_seeIdx.clear();
  ph2_xySignificance.clear();
  ph2_simHitIdx.clear();
  ph2_simType.clear();
  ph2_x.clear();
  ph2_y.clear();
  ph2_z.clear();
  ph2_xx.clear();
  ph2_xy.clear();
  ph2_yy.clear();
  ph2_yz.clear();
  ph2_zz.clear();
  ph2_zx.clear();
  ph2_radL.clear();
  ph2_bbxi.clear();
  //invalid hits
  inv_isBarrel.clear();
  inv_detId.clear();
  inv_detId_phase2.clear();
  inv_type.clear();
  // simhits
  simhit_detId.clear();
  simhit_detId_phase2.clear();
  simhit_x.clear();
  simhit_y.clear();
  simhit_z.clear();
  simhit_px.clear();
  simhit_py.clear();
  simhit_pz.clear();
  simhit_particle.clear();
  simhit_process.clear();
  simhit_eloss.clear();
  simhit_tof.clear();
  //simhit_simTrackId.clear();
  simhit_simTrkIdx.clear();
  simhit_hitIdx.clear();
  simhit_hitType.clear();
  //beamspot
  bsp_x = -9999.;
  bsp_y = -9999.;
  bsp_z = -9999.;
  bsp_sigmax = -9999.;
  bsp_sigmay = -9999.;
  bsp_sigmaz = -9999.;
  //seeds
  see_fitok.clear();
  see_px.clear();
  see_py.clear();
  see_pz.clear();
  see_pt.clear();
  see_eta.clear();
  see_phi.clear();
  see_dxy.clear();
  see_dz.clear();
  see_ptErr.clear();
  see_etaErr.clear();
  see_phiErr.clear();
  see_dxyErr.clear();
  see_dzErr.clear();
  see_chi2.clear();
  see_statePt.clear();
  see_stateTrajX.clear();
  see_stateTrajY.clear();
  see_stateTrajPx.clear();
  see_stateTrajPy.clear();
  see_stateTrajPz.clear();
  see_stateTrajGlbX.clear();
  see_stateTrajGlbY.clear();
  see_stateTrajGlbZ.clear();
  see_stateTrajGlbPx.clear();
  see_stateTrajGlbPy.clear();
  see_stateTrajGlbPz.clear();
  see_stateCurvCov.clear();
  see_q.clear();
  see_nValid.clear();
  see_nPixel.clear();
  see_nGlued.clear();
  see_nStrip.clear();
  see_nPhase2OT.clear();
  see_nCluster.clear();
  see_algo.clear();
  see_stopReason.clear();
  see_nCands.clear();
  see_trkIdx.clear();
  see_isTrue.clear();
  see_bestSimTrkIdx.clear();
  see_bestSimTrkShareFrac.clear();
  see_bestFromFirstHitSimTrkIdx.clear();
  see_bestFromFirstHitSimTrkShareFrac.clear();
  see_simTrkIdx.clear();
  see_simTrkShareFrac.clear();
  see_hitIdx.clear();
  see_hitType.clear();
  //seed algo offset
  see_offset.clear();

  // vertices
  vtx_x.clear();
  vtx_y.clear();
  vtx_z.clear();
  vtx_xErr.clear();
  vtx_yErr.clear();
  vtx_zErr.clear();
  vtx_ndof.clear();
  vtx_chi2.clear();
  vtx_fake.clear();
  vtx_valid.clear();
  vtx_trkIdx.clear();

  // Tracking vertices
  simvtx_event.clear();
  simvtx_bunchCrossing.clear();
  simvtx_processType.clear();
  simvtx_x.clear();
  simvtx_y.clear();
  simvtx_z.clear();
  simvtx_sourceSimIdx.clear();
  simvtx_daughterSimIdx.clear();
  simpv_idx.clear();
}

// ------------ method called for each event  ------------
void TrackingNtuple::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace reco;
  using namespace std;

  const auto& mf = iSetup.getData(mfToken_);
  const auto& theTTRHBuilder = &iSetup.getData(ttrhToken_);
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);
  const TrackerGeometry& tracker = iSetup.getData(tGeomToken_);

  edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator;
  iEvent.getByToken(trackAssociatorToken_, theAssociator);
  const reco::TrackToTrackingParticleAssociator& associatorByHits = *theAssociator;

  LogDebug("TrackingNtuple") << "Analyzing new event";

  //initialize tree variables
  clearVariables();

  // FIXME: we really need to move to edm::View for reading the
  // TrackingParticles... Unfortunately it has non-trivial
  // consequences on the associator/association interfaces etc.
  TrackingParticleRefVector tmpTP;
  const TrackingParticleRefVector* tmpTPptr = nullptr;
  edm::Handle<TrackingParticleCollection> TPCollectionH;
  edm::Handle<TrackingParticleRefVector> TPCollectionHRefVector;

  if (!trackingParticleToken_.isUninitialized()) {
    iEvent.getByToken(trackingParticleToken_, TPCollectionH);
    for (size_t i = 0, size = TPCollectionH->size(); i < size; ++i) {
      tmpTP.push_back(TrackingParticleRef(TPCollectionH, i));
    }
    tmpTPptr = &tmpTP;
  } else {
    iEvent.getByToken(trackingParticleRefToken_, TPCollectionHRefVector);
    tmpTPptr = TPCollectionHRefVector.product();
  }
  const TrackingParticleRefVector& tpCollection = *tmpTPptr;

  // Fill mapping from Ref::key() to index
  TrackingParticleRefKeyToIndex tpKeyToIndex;
  for (size_t i = 0; i < tpCollection.size(); ++i) {
    tpKeyToIndex[tpCollection[i].key()] = i;
  }

  // tracking vertices
  edm::Handle<TrackingVertexCollection> htv;
  iEvent.getByToken(trackingVertexToken_, htv);
  const TrackingVertexCollection& tvs = *htv;

  // Fill mapping from Ref::key() to index
  TrackingVertexRefVector tvRefs;
  TrackingVertexRefKeyToIndex tvKeyToIndex;
  for (size_t i = 0; i < tvs.size(); ++i) {
    const TrackingVertex& v = tvs[i];
    if (!includeOOT_ && v.eventId().bunchCrossing() != 0)  // Ignore OOTPU
      continue;
    tvKeyToIndex[i] = tvRefs.size();
    tvRefs.push_back(TrackingVertexRef(htv, i));
  }

  //get association maps, etc.
  Handle<ClusterTPAssociation> pCluster2TPListH;
  iEvent.getByToken(clusterTPMapToken_, pCluster2TPListH);
  const ClusterTPAssociation& clusterToTPMap = *pCluster2TPListH;
  edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
  iEvent.getByToken(simHitTPMapToken_, simHitsTPAssoc);

  // TP -> cluster count
  TrackingParticleRefKeyToCount tpKeyToClusterCount;
  for (const auto& clusterTP : clusterToTPMap) {
    tpKeyToClusterCount[clusterTP.second.key()] += 1;
  }

  // SimHit key -> index mapping
  SimHitRefKeyToIndex simHitRefKeyToIndex;

  //make a list to link TrackingParticles to its simhits
  std::vector<TPHitIndex> tpHitList;

  // Count the number of reco cluster per TP

  std::set<edm::ProductID> hitProductIds;
  std::map<edm::ProductID, size_t> seedCollToOffset;

  ev_run = iEvent.id().run();
  ev_lumi = iEvent.id().luminosityBlock();
  ev_event = iEvent.id().event();

  // Digi->Sim links for pixels and strips
  edm::Handle<edm::DetSetVector<PixelDigiSimLink>> pixelDigiSimLinksHandle;
  iEvent.getByToken(pixelSimLinkToken_, pixelDigiSimLinksHandle);
  const auto& pixelDigiSimLinks = *pixelDigiSimLinksHandle;

  edm::Handle<edm::DetSetVector<StripDigiSimLink>> stripDigiSimLinksHandle;
  iEvent.getByToken(stripSimLinkToken_, stripDigiSimLinksHandle);

  // Phase2 OT DigiSimLink
  edm::Handle<edm::DetSetVector<PixelDigiSimLink>> siphase2OTSimLinksHandle;
  iEvent.getByToken(siphase2OTSimLinksToken_, siphase2OTSimLinksHandle);

  //beamspot
  Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotToken_, recoBeamSpotHandle);
  BeamSpot const& bs = *recoBeamSpotHandle;
  fillBeamSpot(bs);

  //prapare list to link matched hits to collection
  vector<pair<int, int>> monoStereoClusterList;
  if (includeAllHits_) {
    // simhits, only if TPs are saved as well
    if (includeTrackingParticles_) {
      fillSimHits(tracker, tpKeyToIndex, *simHitsTPAssoc, tTopo, simHitRefKeyToIndex, tpHitList);
    }

    //pixel hits
    fillPixelHits(iEvent,
                  clusterToTPMap,
                  tpKeyToIndex,
                  *simHitsTPAssoc,
                  pixelDigiSimLinks,
                  *theTTRHBuilder,
                  tTopo,
                  simHitRefKeyToIndex,
                  hitProductIds);

    //strip hits
    if (includeStripHits_) {
      LogDebug("TrackingNtuple") << "foundStripSimLink";
      const auto& stripDigiSimLinks = *stripDigiSimLinksHandle;
      fillStripRphiStereoHits(iEvent,
                              clusterToTPMap,
                              tpKeyToIndex,
                              *simHitsTPAssoc,
                              stripDigiSimLinks,
                              *theTTRHBuilder,
                              tTopo,
                              simHitRefKeyToIndex,
                              hitProductIds);

      //matched hits
      fillStripMatchedHits(iEvent, *theTTRHBuilder, tTopo, monoStereoClusterList);
    }

    if (includePhase2OTHits_) {
      LogDebug("TrackingNtuple") << "foundPhase2OTSimLinks";
      const auto& phase2OTSimLinks = *siphase2OTSimLinksHandle;
      fillPhase2OTHits(iEvent,
                       clusterToTPMap,
                       tpKeyToIndex,
                       *simHitsTPAssoc,
                       phase2OTSimLinks,
                       *theTTRHBuilder,
                       tTopo,
                       simHitRefKeyToIndex,
                       hitProductIds);
    }
  }

  //seeds
  if (includeSeeds_) {
    fillSeeds(iEvent,
              tpCollection,
              tpKeyToIndex,
              bs,
              associatorByHits,
              clusterToTPMap,
              *theTTRHBuilder,
              mf,
              tTopo,
              monoStereoClusterList,
              hitProductIds,
              seedCollToOffset);
  }

  //tracks
  edm::Handle<edm::View<reco::Track>> tracksHandle;
  iEvent.getByToken(trackToken_, tracksHandle);
  const edm::View<reco::Track>& tracks = *tracksHandle;
  // The associator interfaces really need to be fixed...
  edm::RefToBaseVector<reco::Track> trackRefs;
  for (edm::View<Track>::size_type i = 0; i < tracks.size(); ++i) {
    trackRefs.push_back(tracks.refAt(i));
  }
  std::vector<const MVACollection*> mvaColls;
  std::vector<const QualityMaskCollection*> qualColls;
  if (includeMVA_) {
    edm::Handle<MVACollection> hmva;
    edm::Handle<QualityMaskCollection> hqual;

    for (const auto& tokenTpl : mvaQualityCollectionTokens_) {
      iEvent.getByToken(std::get<0>(tokenTpl), hmva);
      iEvent.getByToken(std::get<1>(tokenTpl), hqual);

      mvaColls.push_back(hmva.product());
      qualColls.push_back(hqual.product());
      if (mvaColls.back()->size() != tracks.size()) {
        throw cms::Exception("Configuration")
            << "Inconsistency in track collection and MVA sizes. Track collection has " << tracks.size()
            << " tracks, whereas the MVA " << (mvaColls.size() - 1) << " has " << mvaColls.back()->size()
            << " entries. Double-check your configuration.";
      }
      if (qualColls.back()->size() != tracks.size()) {
        throw cms::Exception("Configuration")
            << "Inconsistency in track collection and quality mask sizes. Track collection has " << tracks.size()
            << " tracks, whereas the quality mask " << (qualColls.size() - 1) << " has " << qualColls.back()->size()
            << " entries. Double-check your configuration.";
      }
    }
  }

  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vertexToken_, vertices);

  fillTracks(trackRefs,
             tpCollection,
             tpKeyToIndex,
             tpKeyToClusterCount,
             mf,
             bs,
             *vertices,
             associatorByHits,
             clusterToTPMap,
             *theTTRHBuilder,
             tTopo,
             hitProductIds,
             seedCollToOffset,
             mvaColls,
             qualColls);

  //tracking particles
  //sort association maps with simHits
  std::sort(tpHitList.begin(), tpHitList.end(), tpHitIndexListLessSort);
  fillTrackingParticles(
      iEvent, iSetup, trackRefs, bs, tpCollection, tvKeyToIndex, associatorByHits, tpHitList, tpKeyToClusterCount);

  // vertices
  fillVertices(*vertices, trackRefs);

  // tracking vertices
  fillTrackingVertices(tvRefs, tpKeyToIndex);

  t->Fill();
}

void TrackingNtuple::fillBeamSpot(const reco::BeamSpot& bs) {
  bsp_x = bs.x0();
  bsp_y = bs.y0();
  bsp_z = bs.x0();
  bsp_sigmax = bs.BeamWidthX();
  bsp_sigmay = bs.BeamWidthY();
  bsp_sigmaz = bs.sigmaZ();
}

namespace {
  template <typename SimLink>
  struct GetCluster;
  template <>
  struct GetCluster<PixelDigiSimLink> {
    static const SiPixelCluster& call(const OmniClusterRef& cluster) { return cluster.pixelCluster(); }
  };
  template <>
  struct GetCluster<StripDigiSimLink> {
    static const SiStripCluster& call(const OmniClusterRef& cluster) { return cluster.stripCluster(); }
  };
}  // namespace

template <typename SimLink>
TrackingNtuple::SimHitData TrackingNtuple::matchCluster(
    const OmniClusterRef& cluster,
    DetId hitId,
    int clusterKey,
    const TransientTrackingRecHit::RecHitPointer& ttrh,
    const ClusterTPAssociation& clusterToTPMap,
    const TrackingParticleRefKeyToIndex& tpKeyToIndex,
    const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
    const edm::DetSetVector<SimLink>& digiSimLinks,
    const SimHitRefKeyToIndex& simHitRefKeyToIndex,
    HitType hitType) {
  SimHitData ret;

  std::map<unsigned int, double> simTrackIdToChargeFraction;
  if (hitType == HitType::Phase2OT)
    simTrackIdToChargeFraction = chargeFraction(cluster.phase2OTCluster(), hitId, digiSimLinks);
  else
    simTrackIdToChargeFraction = chargeFraction(GetCluster<SimLink>::call(cluster), hitId, digiSimLinks);

  float h_x = 0, h_y = 0;
  float h_xx = 0, h_xy = 0, h_yy = 0;
  if (simHitBySignificance_) {
    h_x = ttrh->localPosition().x();
    h_y = ttrh->localPosition().y();
    h_xx = ttrh->localPositionError().xx();
    h_xy = ttrh->localPositionError().xy();
    h_yy = ttrh->localPositionError().yy();
  }

  ret.type = HitSimType::Noise;
  auto range = clusterToTPMap.equal_range(cluster);
  if (range.first != range.second) {
    for (auto ip = range.first; ip != range.second; ++ip) {
      const TrackingParticleRef& trackingParticle = ip->second;

      // Find out if the cluster is from signal/ITPU/OOTPU
      const auto event = trackingParticle->eventId().event();
      const auto bx = trackingParticle->eventId().bunchCrossing();
      HitSimType type = HitSimType::OOTPileup;
      if (bx == 0) {
        type = (event == 0 ? HitSimType::Signal : HitSimType::ITPileup);
      } else
        type = HitSimType::OOTPileup;
      ret.type = static_cast<HitSimType>(std::min(static_cast<int>(ret.type), static_cast<int>(type)));

      // Limit to only input TrackingParticles (usually signal+ITPU)
      auto tpIndex = tpKeyToIndex.find(trackingParticle.key());
      if (tpIndex == tpKeyToIndex.end())
        continue;

      //now get the corresponding sim hit
      std::pair<TrackingParticleRef, TrackPSimHitRef> simHitTPpairWithDummyTP(trackingParticle, TrackPSimHitRef());
      //SimHit is dummy: for simHitTPAssociationListGreater sorting only the TP is needed
      auto range = std::equal_range(simHitsTPAssoc.begin(),
                                    simHitsTPAssoc.end(),
                                    simHitTPpairWithDummyTP,
                                    SimHitTPAssociationProducer::simHitTPAssociationListGreater);
      bool foundSimHit = false;
      bool foundElectron = false;
      int foundElectrons = 0;
      int foundNonElectrons = 0;
      for (auto ip = range.first; ip != range.second; ++ip) {
        TrackPSimHitRef TPhit = ip->second;
        DetId dId = DetId(TPhit->detUnitId());
        if (dId.rawId() == hitId.rawId()) {
          // skip electron SimHits for non-electron TPs also here
          if (std::abs(TPhit->particleType()) == 11 && std::abs(trackingParticle->pdgId()) != 11) {
            foundElectrons++;
          } else {
            foundNonElectrons++;
          }
        }
      }

      float minSignificance = 1e12;
      if (simHitBySignificance_) {  //save the best matching hit

        int simHitKey = -1;
        edm::ProductID simHitID;
        for (auto ip = range.first; ip != range.second; ++ip) {
          TrackPSimHitRef TPhit = ip->second;
          DetId dId = DetId(TPhit->detUnitId());
          if (dId.rawId() == hitId.rawId()) {
            // skip electron SimHits for non-electron TPs also here
            if (std::abs(TPhit->particleType()) == 11 && std::abs(trackingParticle->pdgId()) != 11) {
              foundElectron = true;
              if (!keepEleSimHits_)
                continue;
            }

            float sx = TPhit->localPosition().x();
            float sy = TPhit->localPosition().y();
            float dx = sx - h_x;
            float dy = sy - h_y;
            float sig = (dx * dx * h_yy - 2 * dx * dy * h_xy + dy * dy * h_xx) / (h_xx * h_yy - h_xy * h_xy);

            if (sig < minSignificance) {
              minSignificance = sig;
              foundSimHit = true;
              simHitKey = TPhit.key();
              simHitID = TPhit.id();
            }
          }
        }  //loop over matching hits

        auto simHitIndex = simHitRefKeyToIndex.at(std::make_pair(simHitKey, simHitID));
        ret.matchingSimHit.push_back(simHitIndex);

        double chargeFraction = 0.;
        for (const SimTrack& simtrk : trackingParticle->g4Tracks()) {
          auto found = simTrackIdToChargeFraction.find(simtrk.trackId());
          if (found != simTrackIdToChargeFraction.end()) {
            chargeFraction += found->second;
          }
        }
        ret.xySignificance.push_back(minSignificance);
        ret.chargeFraction.push_back(chargeFraction);

        // only for debug prints
        ret.bunchCrossing.push_back(bx);
        ret.event.push_back(event);

        simhit_hitIdx[simHitIndex].push_back(clusterKey);
        simhit_hitType[simHitIndex].push_back(static_cast<int>(hitType));

      } else {  //save all matching hits
        for (auto ip = range.first; ip != range.second; ++ip) {
          TrackPSimHitRef TPhit = ip->second;
          DetId dId = DetId(TPhit->detUnitId());
          if (dId.rawId() == hitId.rawId()) {
            // skip electron SimHits for non-electron TPs also here
            if (std::abs(TPhit->particleType()) == 11 && std::abs(trackingParticle->pdgId()) != 11) {
              foundElectron = true;
              if (!keepEleSimHits_)
                continue;
              if (foundNonElectrons > 0)
                continue;  //prioritize: skip electrons if non-electrons are present
            }

            foundSimHit = true;
            auto simHitKey = TPhit.key();
            auto simHitID = TPhit.id();

            auto simHitIndex = simHitRefKeyToIndex.at(std::make_pair(simHitKey, simHitID));
            ret.matchingSimHit.push_back(simHitIndex);

            double chargeFraction = 0.;
            for (const SimTrack& simtrk : trackingParticle->g4Tracks()) {
              auto found = simTrackIdToChargeFraction.find(simtrk.trackId());
              if (found != simTrackIdToChargeFraction.end()) {
                chargeFraction += found->second;
              }
            }
            ret.xySignificance.push_back(minSignificance);
            ret.chargeFraction.push_back(chargeFraction);

            // only for debug prints
            ret.bunchCrossing.push_back(bx);
            ret.event.push_back(event);

            simhit_hitIdx[simHitIndex].push_back(clusterKey);
            simhit_hitType[simHitIndex].push_back(static_cast<int>(hitType));
          }
        }
      }  //if/else simHitBySignificance_
      if (!foundSimHit) {
        // In case we didn't find a simhit because of filtered-out
        // electron SimHit, just ignore the missing SimHit.
        if (foundElectron && !keepEleSimHits_)
          continue;

        auto ex = cms::Exception("LogicError")
                  << "Did not find SimHit for reco hit DetId " << hitId.rawId() << " for TP " << trackingParticle.key()
                  << " bx:event " << bx << ":" << event << " PDGid " << trackingParticle->pdgId() << " q "
                  << trackingParticle->charge() << " p4 " << trackingParticle->p4() << " nG4 "
                  << trackingParticle->g4Tracks().size() << ".\nFound SimHits from detectors ";
        for (auto ip = range.first; ip != range.second; ++ip) {
          TrackPSimHitRef TPhit = ip->second;
          DetId dId = DetId(TPhit->detUnitId());
          ex << dId.rawId() << " ";
        }
        if (trackingParticle->eventId().event() != 0) {
          ex << "\nSince this is a TrackingParticle from pileup, check that you're running the pileup mixing in "
                "playback mode.";
        }
        throw ex;
      }
    }
  }

  return ret;
}

void TrackingNtuple::fillSimHits(const TrackerGeometry& tracker,
                                 const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                 const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                                 const TrackerTopology& tTopo,
                                 SimHitRefKeyToIndex& simHitRefKeyToIndex,
                                 std::vector<TPHitIndex>& tpHitList) {
  for (const auto& assoc : simHitsTPAssoc) {
    auto tpKey = assoc.first.key();

    // SimHitTPAssociationList can contain more TrackingParticles than
    // what are given to this EDAnalyzer, so we can filter those out here.
    auto found = tpKeyToIndex.find(tpKey);
    if (found == tpKeyToIndex.end())
      continue;
    const auto tpIndex = found->second;

    // skip non-tracker simhits (mostly muons)
    const auto& simhit = *(assoc.second);
    auto detId = DetId(simhit.detUnitId());
    if (detId.det() != DetId::Tracker)
      continue;

    // Skip electron SimHits for non-electron TrackingParticles to
    // filter out delta rays. The delta ray hits just confuse. If we
    // need them later, let's add them as a separate "collection" of
    // hits of a TP
    const TrackingParticle& tp = *(assoc.first);
    if (!keepEleSimHits_ && std::abs(simhit.particleType()) == 11 && std::abs(tp.pdgId()) != 11)
      continue;

    auto simHitKey = std::make_pair(assoc.second.key(), assoc.second.id());

    if (simHitRefKeyToIndex.find(simHitKey) != simHitRefKeyToIndex.end()) {
      for (const auto& assoc2 : simHitsTPAssoc) {
        if (std::make_pair(assoc2.second.key(), assoc2.second.id()) == simHitKey) {
#ifdef EDM_ML_DEBUG
          auto range1 = std::equal_range(simHitsTPAssoc.begin(),
                                         simHitsTPAssoc.end(),
                                         std::make_pair(assoc.first, TrackPSimHitRef()),
                                         SimHitTPAssociationProducer::simHitTPAssociationListGreater);
          auto range2 = std::equal_range(simHitsTPAssoc.begin(),
                                         simHitsTPAssoc.end(),
                                         std::make_pair(assoc2.first, TrackPSimHitRef()),
                                         SimHitTPAssociationProducer::simHitTPAssociationListGreater);

          LogTrace("TrackingNtuple") << "Earlier TP " << assoc2.first.key() << " SimTrack Ids";
          for (const auto& simTrack : assoc2.first->g4Tracks()) {
            edm::LogPrint("TrackingNtuple") << " SimTrack " << simTrack.trackId() << " BX:event "
                                            << simTrack.eventId().bunchCrossing() << ":" << simTrack.eventId().event();
          }
          for (auto iHit = range2.first; iHit != range2.second; ++iHit) {
            LogTrace("TrackingNtuple") << " SimHit " << iHit->second.key() << " " << iHit->second.id() << " tof "
                                       << iHit->second->tof() << " trackId " << iHit->second->trackId() << " BX:event "
                                       << iHit->second->eventId().bunchCrossing() << ":"
                                       << iHit->second->eventId().event();
          }
          LogTrace("TrackingNtuple") << "Current TP " << assoc.first.key() << " SimTrack Ids";
          for (const auto& simTrack : assoc.first->g4Tracks()) {
            edm::LogPrint("TrackingNtuple") << " SimTrack " << simTrack.trackId() << " BX:event "
                                            << simTrack.eventId().bunchCrossing() << ":" << simTrack.eventId().event();
          }
          for (auto iHit = range1.first; iHit != range1.second; ++iHit) {
            LogTrace("TrackingNtuple") << " SimHit " << iHit->second.key() << " " << iHit->second.id() << " tof "
                                       << iHit->second->tof() << " trackId " << iHit->second->trackId() << " BX:event "
                                       << iHit->second->eventId().bunchCrossing() << ":"
                                       << iHit->second->eventId().event();
          }
#endif

          throw cms::Exception("LogicError")
              << "Got second time the SimHit " << simHitKey.first << " of " << simHitKey.second
              << ", first time with TrackingParticle " << assoc2.first.key() << ", now with " << tpKey;
        }
      }
      throw cms::Exception("LogicError") << "Got second time the SimHit " << simHitKey.first << " of "
                                         << simHitKey.second << ", now with TrackingParticle " << tpKey
                                         << ", but I didn't find the first occurrance!";
    }

    auto det = tracker.idToDetUnit(detId);
    if (!det)
      throw cms::Exception("LogicError") << "Did not find a det unit for DetId " << simhit.detUnitId()
                                         << " from tracker geometry";

    const auto pos = det->surface().toGlobal(simhit.localPosition());
    const float tof = simhit.timeOfFlight();

    const auto simHitIndex = simhit_x.size();
    simHitRefKeyToIndex[simHitKey] = simHitIndex;

    if (includeStripHits_)
      simhit_detId.push_back(tTopo, detId);
    else
      simhit_detId_phase2.push_back(tTopo, detId);
    simhit_x.push_back(pos.x());
    simhit_y.push_back(pos.y());
    simhit_z.push_back(pos.z());
    if (saveSimHitsP3_) {
      const auto mom = det->surface().toGlobal(simhit.momentumAtEntry());
      simhit_px.push_back(mom.x());
      simhit_py.push_back(mom.y());
      simhit_pz.push_back(mom.z());
    }
    simhit_particle.push_back(simhit.particleType());
    simhit_process.push_back(simhit.processType());
    simhit_eloss.push_back(simhit.energyLoss());
    simhit_tof.push_back(tof);
    //simhit_simTrackId.push_back(simhit.trackId());

    simhit_simTrkIdx.push_back(tpIndex);

    simhit_hitIdx.emplace_back();   // filled in matchCluster
    simhit_hitType.emplace_back();  // filled in matchCluster

    tpHitList.emplace_back(tpKey, simHitIndex, tof, simhit.detUnitId());
  }
}

void TrackingNtuple::fillPixelHits(const edm::Event& iEvent,
                                   const ClusterTPAssociation& clusterToTPMap,
                                   const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                   const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                                   const edm::DetSetVector<PixelDigiSimLink>& digiSimLink,
                                   const TransientTrackingRecHitBuilder& theTTRHBuilder,
                                   const TrackerTopology& tTopo,
                                   const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                                   std::set<edm::ProductID>& hitProductIds) {
  std::vector<std::pair<uint64_t, PixelMaskContainer const*>> pixelMasks;
  pixelMasks.reserve(pixelUseMaskTokens_.size());
  for (const auto& itoken : pixelUseMaskTokens_) {
    edm::Handle<PixelMaskContainer> aH;
    iEvent.getByToken(itoken.second, aH);
    pixelMasks.emplace_back(1 << itoken.first, aH.product());
  }
  auto pixUsedMask = [&pixelMasks](size_t key) {
    uint64_t mask = 0;
    for (auto const& m : pixelMasks) {
      if (m.second->mask(key))
        mask |= m.first;
    }
    return mask;
  };

  edm::Handle<SiPixelRecHitCollection> pixelHits;
  iEvent.getByToken(pixelRecHitToken_, pixelHits);
  for (auto it = pixelHits->begin(); it != pixelHits->end(); it++) {
    const DetId hitId = it->detId();
    for (auto hit = it->begin(); hit != it->end(); hit++) {
      TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder.build(&*hit);

      hitProductIds.insert(hit->cluster().id());

      const int key = hit->cluster().key();
      const int lay = tTopo.layer(hitId);

      pix_isBarrel.push_back(hitId.subdetId() == 1);
      pix_detId.push_back(tTopo, hitId);
      pix_trkIdx.emplace_back();  // filled in fillTracks
      pix_seeIdx.emplace_back();  // filled in fillSeeds
      pix_x.push_back(ttrh->globalPosition().x());
      pix_y.push_back(ttrh->globalPosition().y());
      pix_z.push_back(ttrh->globalPosition().z());
      pix_xx.push_back(ttrh->globalPositionError().cxx());
      pix_xy.push_back(ttrh->globalPositionError().cyx());
      pix_yy.push_back(ttrh->globalPositionError().cyy());
      pix_yz.push_back(ttrh->globalPositionError().czy());
      pix_zz.push_back(ttrh->globalPositionError().czz());
      pix_zx.push_back(ttrh->globalPositionError().czx());
      pix_radL.push_back(ttrh->surface()->mediumProperties().radLen());
      pix_bbxi.push_back(ttrh->surface()->mediumProperties().xi());
      pix_clustSizeCol.push_back(hit->cluster()->sizeY());
      pix_clustSizeRow.push_back(hit->cluster()->sizeX());
      pix_usedMask.push_back(pixUsedMask(hit->firstClusterRef().key()));

      LogTrace("TrackingNtuple") << "pixHit cluster=" << key << " subdId=" << hitId.subdetId() << " lay=" << lay
                                 << " rawId=" << hitId.rawId() << " pos =" << ttrh->globalPosition();
      if (includeTrackingParticles_) {
        SimHitData simHitData = matchCluster(hit->firstClusterRef(),
                                             hitId,
                                             key,
                                             ttrh,
                                             clusterToTPMap,
                                             tpKeyToIndex,
                                             simHitsTPAssoc,
                                             digiSimLink,
                                             simHitRefKeyToIndex,
                                             HitType::Pixel);
        pix_simHitIdx.push_back(simHitData.matchingSimHit);
        pix_simType.push_back(static_cast<int>(simHitData.type));
        pix_xySignificance.push_back(simHitData.xySignificance);
        pix_chargeFraction.push_back(simHitData.chargeFraction);
        LogTrace("TrackingNtuple") << " nMatchingSimHit=" << simHitData.matchingSimHit.size();
        if (!simHitData.matchingSimHit.empty()) {
          const auto simHitIdx = simHitData.matchingSimHit[0];
          LogTrace("TrackingNtuple") << " firstMatchingSimHit=" << simHitIdx << " simHitPos="
                                     << GlobalPoint(simhit_x[simHitIdx], simhit_y[simHitIdx], simhit_z[simHitIdx])
                                     << " energyLoss=" << simhit_eloss[simHitIdx]
                                     << " particleType=" << simhit_particle[simHitIdx]
                                     << " processType=" << simhit_process[simHitIdx]
                                     << " bunchCrossing=" << simHitData.bunchCrossing[0]
                                     << " event=" << simHitData.event[0];
        }
      }
    }
  }
}

void TrackingNtuple::fillStripRphiStereoHits(const edm::Event& iEvent,
                                             const ClusterTPAssociation& clusterToTPMap,
                                             const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                             const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                                             const edm::DetSetVector<StripDigiSimLink>& digiSimLink,
                                             const TransientTrackingRecHitBuilder& theTTRHBuilder,
                                             const TrackerTopology& tTopo,
                                             const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                                             std::set<edm::ProductID>& hitProductIds) {
  std::vector<std::pair<uint64_t, StripMaskContainer const*>> stripMasks;
  stripMasks.reserve(stripUseMaskTokens_.size());
  for (const auto& itoken : stripUseMaskTokens_) {
    edm::Handle<StripMaskContainer> aH;
    iEvent.getByToken(itoken.second, aH);
    stripMasks.emplace_back(1 << itoken.first, aH.product());
  }
  auto strUsedMask = [&stripMasks](size_t key) {
    uint64_t mask = 0;
    for (auto const& m : stripMasks) {
      if (m.second->mask(key))
        mask |= m.first;
    }
    return mask;
  };

  //index strip hit branches by cluster index
  edm::Handle<SiStripRecHit2DCollection> rphiHits;
  iEvent.getByToken(stripRphiRecHitToken_, rphiHits);
  edm::Handle<SiStripRecHit2DCollection> stereoHits;
  iEvent.getByToken(stripStereoRecHitToken_, stereoHits);
  int totalStripHits = rphiHits->dataSize() + stereoHits->dataSize();
  str_isBarrel.resize(totalStripHits);
  str_detId.resize(totalStripHits);
  str_trkIdx.resize(totalStripHits);  // filled in fillTracks
  str_seeIdx.resize(totalStripHits);  // filled in fillSeeds
  str_simHitIdx.resize(totalStripHits);
  str_simType.resize(totalStripHits);
  str_chargeFraction.resize(totalStripHits);
  str_x.resize(totalStripHits);
  str_y.resize(totalStripHits);
  str_z.resize(totalStripHits);
  str_xx.resize(totalStripHits);
  str_xy.resize(totalStripHits);
  str_yy.resize(totalStripHits);
  str_yz.resize(totalStripHits);
  str_zz.resize(totalStripHits);
  str_zx.resize(totalStripHits);
  str_xySignificance.resize(totalStripHits);
  str_chargeFraction.resize(totalStripHits);
  str_radL.resize(totalStripHits);
  str_bbxi.resize(totalStripHits);
  str_chargePerCM.resize(totalStripHits);
  str_clustSize.resize(totalStripHits);
  str_usedMask.resize(totalStripHits);

  auto fill = [&](const SiStripRecHit2DCollection& hits, const char* name) {
    for (const auto& detset : hits) {
      const DetId hitId = detset.detId();
      for (const auto& hit : detset) {
        TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder.build(&hit);

        hitProductIds.insert(hit.cluster().id());

        const int key = hit.cluster().key();
        const int lay = tTopo.layer(hitId);
        str_isBarrel[key] = (hitId.subdetId() == StripSubdetector::TIB || hitId.subdetId() == StripSubdetector::TOB);
        str_detId.set(key, tTopo, hitId);
        str_x[key] = ttrh->globalPosition().x();
        str_y[key] = ttrh->globalPosition().y();
        str_z[key] = ttrh->globalPosition().z();
        str_xx[key] = ttrh->globalPositionError().cxx();
        str_xy[key] = ttrh->globalPositionError().cyx();
        str_yy[key] = ttrh->globalPositionError().cyy();
        str_yz[key] = ttrh->globalPositionError().czy();
        str_zz[key] = ttrh->globalPositionError().czz();
        str_zx[key] = ttrh->globalPositionError().czx();
        str_radL[key] = ttrh->surface()->mediumProperties().radLen();
        str_bbxi[key] = ttrh->surface()->mediumProperties().xi();
        str_chargePerCM[key] = siStripClusterTools::chargePerCM(hitId, hit.firstClusterRef().stripCluster());
        str_clustSize[key] = hit.cluster()->amplitudes().size();
        str_usedMask[key] = strUsedMask(key);
        LogTrace("TrackingNtuple") << name << " cluster=" << key << " subdId=" << hitId.subdetId() << " lay=" << lay
                                   << " rawId=" << hitId.rawId() << " pos =" << ttrh->globalPosition();

        if (includeTrackingParticles_) {
          SimHitData simHitData = matchCluster(hit.firstClusterRef(),
                                               hitId,
                                               key,
                                               ttrh,
                                               clusterToTPMap,
                                               tpKeyToIndex,
                                               simHitsTPAssoc,
                                               digiSimLink,
                                               simHitRefKeyToIndex,
                                               HitType::Strip);
          str_simHitIdx[key] = simHitData.matchingSimHit;
          str_simType[key] = static_cast<int>(simHitData.type);
          str_xySignificance[key] = simHitData.xySignificance;
          str_chargeFraction[key] = simHitData.chargeFraction;
          LogTrace("TrackingNtuple") << " nMatchingSimHit=" << simHitData.matchingSimHit.size();
          if (!simHitData.matchingSimHit.empty()) {
            const auto simHitIdx = simHitData.matchingSimHit[0];
            LogTrace("TrackingNtuple") << " firstMatchingSimHit=" << simHitIdx << " simHitPos="
                                       << GlobalPoint(simhit_x[simHitIdx], simhit_y[simHitIdx], simhit_z[simHitIdx])
                                       << " simHitPos="
                                       << GlobalPoint(simhit_x[simHitIdx], simhit_y[simHitIdx], simhit_z[simHitIdx])
                                       << " energyLoss=" << simhit_eloss[simHitIdx]
                                       << " particleType=" << simhit_particle[simHitIdx]
                                       << " processType=" << simhit_process[simHitIdx]
                                       << " bunchCrossing=" << simHitData.bunchCrossing[0]
                                       << " event=" << simHitData.event[0];
          }
        }
      }
    }
  };

  fill(*rphiHits, "stripRPhiHit");
  fill(*stereoHits, "stripStereoHit");
}

size_t TrackingNtuple::addStripMatchedHit(const SiStripMatchedRecHit2D& hit,
                                          const TransientTrackingRecHitBuilder& theTTRHBuilder,
                                          const TrackerTopology& tTopo,
                                          const std::vector<std::pair<uint64_t, StripMaskContainer const*>>& stripMasks,
                                          std::vector<std::pair<int, int>>& monoStereoClusterList) {
  auto strUsedMask = [&stripMasks](size_t key) {
    uint64_t mask = 0;
    for (auto const& m : stripMasks) {
      if (m.second->mask(key))
        mask |= m.first;
    }
    return mask;
  };

  TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder.build(&hit);
  const auto hitId = hit.geographicalId();
  const int lay = tTopo.layer(hitId);
  monoStereoClusterList.emplace_back(hit.monoHit().cluster().key(), hit.stereoHit().cluster().key());
  glu_isBarrel.push_back((hitId.subdetId() == StripSubdetector::TIB || hitId.subdetId() == StripSubdetector::TOB));
  glu_detId.push_back(tTopo, hitId);
  glu_monoIdx.push_back(hit.monoHit().cluster().key());
  glu_stereoIdx.push_back(hit.stereoHit().cluster().key());
  glu_seeIdx.emplace_back();  // filled in fillSeeds
  glu_x.push_back(ttrh->globalPosition().x());
  glu_y.push_back(ttrh->globalPosition().y());
  glu_z.push_back(ttrh->globalPosition().z());
  glu_xx.push_back(ttrh->globalPositionError().cxx());
  glu_xy.push_back(ttrh->globalPositionError().cyx());
  glu_yy.push_back(ttrh->globalPositionError().cyy());
  glu_yz.push_back(ttrh->globalPositionError().czy());
  glu_zz.push_back(ttrh->globalPositionError().czz());
  glu_zx.push_back(ttrh->globalPositionError().czx());
  glu_radL.push_back(ttrh->surface()->mediumProperties().radLen());
  glu_bbxi.push_back(ttrh->surface()->mediumProperties().xi());
  glu_chargePerCM.push_back(siStripClusterTools::chargePerCM(hitId, hit.firstClusterRef().stripCluster()));
  glu_clustSizeMono.push_back(hit.monoHit().cluster()->amplitudes().size());
  glu_clustSizeStereo.push_back(hit.stereoHit().cluster()->amplitudes().size());
  glu_usedMaskMono.push_back(strUsedMask(hit.monoHit().cluster().key()));
  glu_usedMaskStereo.push_back(strUsedMask(hit.stereoHit().cluster().key()));
  LogTrace("TrackingNtuple") << "stripMatchedHit"
                             << " cluster0=" << hit.stereoHit().cluster().key()
                             << " cluster1=" << hit.monoHit().cluster().key() << " subdId=" << hitId.subdetId()
                             << " lay=" << lay << " rawId=" << hitId.rawId() << " pos =" << ttrh->globalPosition();
  return glu_isBarrel.size() - 1;
}

void TrackingNtuple::fillStripMatchedHits(const edm::Event& iEvent,
                                          const TransientTrackingRecHitBuilder& theTTRHBuilder,
                                          const TrackerTopology& tTopo,
                                          std::vector<std::pair<int, int>>& monoStereoClusterList) {
  std::vector<std::pair<uint64_t, StripMaskContainer const*>> stripMasks;
  stripMasks.reserve(stripUseMaskTokens_.size());
  for (const auto& itoken : stripUseMaskTokens_) {
    edm::Handle<StripMaskContainer> aH;
    iEvent.getByToken(itoken.second, aH);
    stripMasks.emplace_back(1 << itoken.first, aH.product());
  }

  edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
  iEvent.getByToken(stripMatchedRecHitToken_, matchedHits);
  for (auto it = matchedHits->begin(); it != matchedHits->end(); it++) {
    for (auto hit = it->begin(); hit != it->end(); hit++) {
      addStripMatchedHit(*hit, theTTRHBuilder, tTopo, stripMasks, monoStereoClusterList);
    }
  }
}

void TrackingNtuple::fillPhase2OTHits(const edm::Event& iEvent,
                                      const ClusterTPAssociation& clusterToTPMap,
                                      const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                      const SimHitTPAssociationProducer::SimHitTPAssociationList& simHitsTPAssoc,
                                      const edm::DetSetVector<PixelDigiSimLink>& digiSimLink,
                                      const TransientTrackingRecHitBuilder& theTTRHBuilder,
                                      const TrackerTopology& tTopo,
                                      const SimHitRefKeyToIndex& simHitRefKeyToIndex,
                                      std::set<edm::ProductID>& hitProductIds) {
  edm::Handle<Phase2TrackerRecHit1DCollectionNew> phase2OTHits;
  iEvent.getByToken(phase2OTRecHitToken_, phase2OTHits);
  for (auto it = phase2OTHits->begin(); it != phase2OTHits->end(); it++) {
    const DetId hitId = it->detId();
    for (auto hit = it->begin(); hit != it->end(); hit++) {
      TransientTrackingRecHit::RecHitPointer ttrh = theTTRHBuilder.build(&*hit);

      hitProductIds.insert(hit->cluster().id());

      const int key = hit->cluster().key();
      const int lay = tTopo.layer(hitId);

      ph2_isBarrel.push_back(hitId.subdetId() == 1);
      ph2_detId.push_back(tTopo, hitId);
      ph2_trkIdx.emplace_back();  // filled in fillTracks
      ph2_seeIdx.emplace_back();  // filled in fillSeeds
      ph2_x.push_back(ttrh->globalPosition().x());
      ph2_y.push_back(ttrh->globalPosition().y());
      ph2_z.push_back(ttrh->globalPosition().z());
      ph2_xx.push_back(ttrh->globalPositionError().cxx());
      ph2_xy.push_back(ttrh->globalPositionError().cyx());
      ph2_yy.push_back(ttrh->globalPositionError().cyy());
      ph2_yz.push_back(ttrh->globalPositionError().czy());
      ph2_zz.push_back(ttrh->globalPositionError().czz());
      ph2_zx.push_back(ttrh->globalPositionError().czx());
      ph2_radL.push_back(ttrh->surface()->mediumProperties().radLen());
      ph2_bbxi.push_back(ttrh->surface()->mediumProperties().xi());

      LogTrace("TrackingNtuple") << "phase2 OT cluster=" << key << " subdId=" << hitId.subdetId() << " lay=" << lay
                                 << " rawId=" << hitId.rawId() << " pos =" << ttrh->globalPosition();

      if (includeTrackingParticles_) {
        SimHitData simHitData = matchCluster(hit->firstClusterRef(),
                                             hitId,
                                             key,
                                             ttrh,
                                             clusterToTPMap,
                                             tpKeyToIndex,
                                             simHitsTPAssoc,
                                             digiSimLink,
                                             simHitRefKeyToIndex,
                                             HitType::Phase2OT);
        ph2_xySignificance.push_back(simHitData.xySignificance);
        ph2_simHitIdx.push_back(simHitData.matchingSimHit);
        ph2_simType.push_back(static_cast<int>(simHitData.type));
        LogTrace("TrackingNtuple") << " nMatchingSimHit=" << simHitData.matchingSimHit.size();
        if (!simHitData.matchingSimHit.empty()) {
          const auto simHitIdx = simHitData.matchingSimHit[0];
          LogTrace("TrackingNtuple") << " firstMatchingSimHit=" << simHitIdx << " simHitPos="
                                     << GlobalPoint(simhit_x[simHitIdx], simhit_y[simHitIdx], simhit_z[simHitIdx])
                                     << " energyLoss=" << simhit_eloss[simHitIdx]
                                     << " particleType=" << simhit_particle[simHitIdx]
                                     << " processType=" << simhit_process[simHitIdx]
                                     << " bunchCrossing=" << simHitData.bunchCrossing[0]
                                     << " event=" << simHitData.event[0];
        }
      }
    }
  }
}

void TrackingNtuple::fillSeeds(const edm::Event& iEvent,
                               const TrackingParticleRefVector& tpCollection,
                               const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                               const reco::BeamSpot& bs,
                               const reco::TrackToTrackingParticleAssociator& associatorByHits,
                               const ClusterTPAssociation& clusterToTPMap,
                               const TransientTrackingRecHitBuilder& theTTRHBuilder,
                               const MagneticField& theMF,
                               const TrackerTopology& tTopo,
                               std::vector<std::pair<int, int>>& monoStereoClusterList,
                               const std::set<edm::ProductID>& hitProductIds,
                               std::map<edm::ProductID, size_t>& seedCollToOffset) {
  TSCBLBuilderNoMaterial tscblBuilder;
  for (size_t iColl = 0; iColl < seedTokens_.size(); ++iColl) {
    const auto& seedToken = seedTokens_[iColl];

    edm::Handle<edm::View<reco::Track>> seedTracksHandle;
    iEvent.getByToken(seedToken, seedTracksHandle);
    const auto& seedTracks = *seedTracksHandle;

    if (seedTracks.empty())
      continue;

    edm::EDConsumerBase::Labels labels;
    labelsForToken(seedToken, labels);

    const auto& seedStopInfoToken = seedStopInfoTokens_[iColl];
    edm::Handle<std::vector<SeedStopInfo>> seedStopInfoHandle;
    iEvent.getByToken(seedStopInfoToken, seedStopInfoHandle);
    const auto& seedStopInfos = *seedStopInfoHandle;
    if (seedTracks.size() != seedStopInfos.size()) {
      edm::EDConsumerBase::Labels labels2;
      labelsForToken(seedStopInfoToken, labels2);

      throw cms::Exception("LogicError") << "Got " << seedTracks.size() << " seeds, but " << seedStopInfos.size()
                                         << " seed stopping infos for collections " << labels.module << ", "
                                         << labels2.module;
    }

    std::vector<std::pair<uint64_t, StripMaskContainer const*>> stripMasks;
    stripMasks.reserve(stripUseMaskTokens_.size());
    for (const auto& itoken : stripUseMaskTokens_) {
      edm::Handle<StripMaskContainer> aH;
      iEvent.getByToken(itoken.second, aH);
      stripMasks.emplace_back(1 << itoken.first, aH.product());
    }

    // The associator interfaces really need to be fixed...
    edm::RefToBaseVector<reco::Track> seedTrackRefs;
    for (edm::View<reco::Track>::size_type i = 0; i < seedTracks.size(); ++i) {
      seedTrackRefs.push_back(seedTracks.refAt(i));
    }
    reco::RecoToSimCollection recSimColl = associatorByHits.associateRecoToSim(seedTrackRefs, tpCollection);
    reco::SimToRecoCollection simRecColl = associatorByHits.associateSimToReco(seedTrackRefs, tpCollection);

    TString label = labels.module;
    //format label to match algoName
    label.ReplaceAll("seedTracks", "");
    label.ReplaceAll("Seeds", "");
    label.ReplaceAll("muonSeeded", "muonSeededStep");
    //for HLT seeds
    label.ReplaceAll("FromPixelTracks", "");
    label.ReplaceAll("PFLowPixel", "");
    label.ReplaceAll("hltDoubletRecovery", "pixelPairStep");  //random choice
    int algo = reco::TrackBase::algoByName(label.Data());

    edm::ProductID id = seedTracks[0].seedRef().id();
    const auto offset = see_fitok.size();
    auto inserted = seedCollToOffset.emplace(id, offset);
    if (!inserted.second)
      throw cms::Exception("Configuration")
          << "Trying to add seeds with ProductID " << id << " for a second time from collection " << labels.module
          << ", seed algo " << label << ". Typically this is caused by a configuration problem.";
    see_offset.push_back(offset);

    LogTrace("TrackingNtuple") << "NEW SEED LABEL: " << label << " size: " << seedTracks.size() << " algo=" << algo
                               << " ProductID " << id;

    for (size_t iSeed = 0; iSeed < seedTrackRefs.size(); ++iSeed) {
      const auto& seedTrackRef = seedTrackRefs[iSeed];
      const auto& seedTrack = *seedTrackRef;
      const auto& seedRef = seedTrack.seedRef();
      const auto& seed = *seedRef;

      const auto seedStopInfo = seedStopInfos[iSeed];

      if (seedRef.id() != id)
        throw cms::Exception("LogicError")
            << "All tracks in 'TracksFromSeeds' collection should point to seeds in the same collection. Now the "
               "element 0 had ProductID "
            << id << " while the element " << seedTrackRef.key() << " had " << seedTrackRef.id()
            << ". The source collection is " << labels.module << ".";

      std::vector<int> tpIdx;
      std::vector<float> sharedFraction;
      auto foundTPs = recSimColl.find(seedTrackRef);
      if (foundTPs != recSimColl.end()) {
        for (const auto& tpQuality : foundTPs->val) {
          tpIdx.push_back(tpKeyToIndex.at(tpQuality.first.key()));
          sharedFraction.push_back(tpQuality.second);
        }
      }

      // Search for a best-matching TrackingParticle for a seed
      const int nHits = seedTrack.numberOfValidHits();
      const auto clusters = track_associator::hitsToClusterRefs(
          seedTrack.recHitsBegin(),
          seedTrack.recHitsEnd());  // TODO: this function is called 3 times per track, try to reduce
      const int nClusters = clusters.size();
      const auto bestKeyCount = findBestMatchingTrackingParticle(seedTrack, clusterToTPMap, tpKeyToIndex);
      const float bestShareFrac =
          nClusters > 0 ? static_cast<float>(bestKeyCount.countClusters) / static_cast<float>(nClusters) : 0;
      // Another way starting from the first hit of the seed
      const auto bestFirstHitKeyCount =
          findMatchingTrackingParticleFromFirstHit(seedTrack, clusterToTPMap, tpKeyToIndex);
      const float bestFirstHitShareFrac =
          nClusters > 0 ? static_cast<float>(bestFirstHitKeyCount.countClusters) / static_cast<float>(nClusters) : 0;

      const bool seedFitOk = !trackFromSeedFitFailed(seedTrack);
      const int charge = seedTrack.charge();
      const float pt = seedFitOk ? seedTrack.pt() : 0;
      const float eta = seedFitOk ? seedTrack.eta() : 0;
      const float phi = seedFitOk ? seedTrack.phi() : 0;

      const auto seedIndex = see_fitok.size();

      see_fitok.push_back(seedFitOk);

      see_px.push_back(seedFitOk ? seedTrack.px() : 0);
      see_py.push_back(seedFitOk ? seedTrack.py() : 0);
      see_pz.push_back(seedFitOk ? seedTrack.pz() : 0);
      see_pt.push_back(pt);
      see_eta.push_back(eta);
      see_phi.push_back(phi);
      see_q.push_back(charge);
      see_nValid.push_back(nHits);

      see_dxy.push_back(seedFitOk ? seedTrack.dxy(bs.position()) : 0);
      see_dz.push_back(seedFitOk ? seedTrack.dz(bs.position()) : 0);
      see_ptErr.push_back(seedFitOk ? seedTrack.ptError() : 0);
      see_etaErr.push_back(seedFitOk ? seedTrack.etaError() : 0);
      see_phiErr.push_back(seedFitOk ? seedTrack.phiError() : 0);
      see_dxyErr.push_back(seedFitOk ? seedTrack.dxyError() : 0);
      see_dzErr.push_back(seedFitOk ? seedTrack.dzError() : 0);
      see_algo.push_back(algo);
      see_stopReason.push_back(seedStopInfo.stopReasonUC());
      see_nCands.push_back(seedStopInfo.candidatesPerSeed());

      const auto& state = seedTrack.seedRef()->startingState();
      const auto& pos = state.parameters().position();
      const auto& mom = state.parameters().momentum();
      see_statePt.push_back(state.pt());
      see_stateTrajX.push_back(pos.x());
      see_stateTrajY.push_back(pos.y());
      see_stateTrajPx.push_back(mom.x());
      see_stateTrajPy.push_back(mom.y());
      see_stateTrajPz.push_back(mom.z());

      ///the following is useful for analysis in global coords at seed hit surface
      TransientTrackingRecHit::RecHitPointer lastRecHit = theTTRHBuilder.build(&*(seed.recHits().end() - 1));
      TrajectoryStateOnSurface tsos =
          trajectoryStateTransform::transientState(seed.startingState(), lastRecHit->surface(), &theMF);
      auto const& stateGlobal = tsos.globalParameters();
      see_stateTrajGlbX.push_back(stateGlobal.position().x());
      see_stateTrajGlbY.push_back(stateGlobal.position().y());
      see_stateTrajGlbZ.push_back(stateGlobal.position().z());
      see_stateTrajGlbPx.push_back(stateGlobal.momentum().x());
      see_stateTrajGlbPy.push_back(stateGlobal.momentum().y());
      see_stateTrajGlbPz.push_back(stateGlobal.momentum().z());
      if (addSeedCurvCov_) {
        auto const& stateCcov = tsos.curvilinearError().matrix();
        std::vector<float> cov(15);
        auto covP = cov.begin();
        for (auto const val : stateCcov)
          *(covP++) = val;  //row-major
        see_stateCurvCov.push_back(std::move(cov));
      }

      see_trkIdx.push_back(-1);  // to be set correctly in fillTracks
      if (includeTrackingParticles_) {
        see_simTrkIdx.push_back(tpIdx);
        see_simTrkShareFrac.push_back(sharedFraction);
        see_bestSimTrkIdx.push_back(bestKeyCount.key >= 0 ? tpKeyToIndex.at(bestKeyCount.key) : -1);
        see_bestFromFirstHitSimTrkIdx.push_back(
            bestFirstHitKeyCount.key >= 0 ? tpKeyToIndex.at(bestFirstHitKeyCount.key) : -1);
      } else {
        see_isTrue.push_back(!tpIdx.empty());
      }
      see_bestSimTrkShareFrac.push_back(bestShareFrac);
      see_bestFromFirstHitSimTrkShareFrac.push_back(bestFirstHitShareFrac);

      /// Hmm, the following could make sense instead of plain failing if propagation to beam line fails
      /*
      TransientTrackingRecHit::RecHitPointer lastRecHit = theTTRHBuilder.build(&*(seed.recHits().second-1));
      TrajectoryStateOnSurface state = trajectoryStateTransform::transientState( itSeed->startingState(), lastRecHit->surface(), &theMF);
      float pt  = state.globalParameters().momentum().perp();
      float eta = state.globalParameters().momentum().eta();
      float phi = state.globalParameters().momentum().phi();
      see_px      .push_back( state.globalParameters().momentum().x() );
      see_py      .push_back( state.globalParameters().momentum().y() );
      see_pz      .push_back( state.globalParameters().momentum().z() );
      */

      std::vector<int> hitIdx;
      std::vector<int> hitType;

      for (auto const& hit : seed.recHits()) {
        TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder.build(&hit);
        int subid = recHit->geographicalId().subdetId();
        if (subid == (int)PixelSubdetector::PixelBarrel || subid == (int)PixelSubdetector::PixelEndcap) {
          const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&*recHit);
          const auto& clusterRef = bhit->firstClusterRef();
          const auto clusterKey = clusterRef.cluster_pixel().key();
          if (includeAllHits_) {
            checkProductID(hitProductIds, clusterRef.id(), "seed");
            pix_seeIdx[clusterKey].push_back(seedIndex);
          }
          hitIdx.push_back(clusterKey);
          hitType.push_back(static_cast<int>(HitType::Pixel));
        } else if (subid == (int)StripSubdetector::TOB || subid == (int)StripSubdetector::TID ||
                   subid == (int)StripSubdetector::TIB || subid == (int)StripSubdetector::TEC) {
          if (trackerHitRTTI::isMatched(*recHit)) {
            const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D*>(&*recHit);
            if (includeAllHits_) {
              checkProductID(hitProductIds, matchedHit->monoClusterRef().id(), "seed");
              checkProductID(hitProductIds, matchedHit->stereoClusterRef().id(), "seed");
            }
            int monoIdx = matchedHit->monoClusterRef().key();
            int stereoIdx = matchedHit->stereoClusterRef().key();

            std::vector<std::pair<int, int>>::iterator pos =
                find(monoStereoClusterList.begin(), monoStereoClusterList.end(), std::make_pair(monoIdx, stereoIdx));
            size_t gluedIndex = -1;
            if (pos != monoStereoClusterList.end()) {
              gluedIndex = std::distance(monoStereoClusterList.begin(), pos);
            } else {
              // We can encounter glued hits not in the input
              // SiStripMatchedRecHit2DCollection, e.g. via muon
              // outside-in seeds (or anything taking hits from
              // MeasurementTrackerEvent). So let's add them here.
              gluedIndex = addStripMatchedHit(*matchedHit, theTTRHBuilder, tTopo, stripMasks, monoStereoClusterList);
            }

            if (includeAllHits_)
              glu_seeIdx[gluedIndex].push_back(seedIndex);
            hitIdx.push_back(gluedIndex);
            hitType.push_back(static_cast<int>(HitType::Glued));
          } else {
            const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&*recHit);
            const auto& clusterRef = bhit->firstClusterRef();
            unsigned int clusterKey;
            if (clusterRef.isPhase2()) {
              clusterKey = clusterRef.cluster_phase2OT().key();
            } else {
              clusterKey = clusterRef.cluster_strip().key();
            }

            if (includeAllHits_) {
              checkProductID(hitProductIds, clusterRef.id(), "seed");
              if (clusterRef.isPhase2()) {
                ph2_seeIdx[clusterKey].push_back(seedIndex);
              } else {
                str_seeIdx[clusterKey].push_back(seedIndex);
              }
            }

            hitIdx.push_back(clusterKey);
            if (clusterRef.isPhase2()) {
              hitType.push_back(static_cast<int>(HitType::Phase2OT));
            } else {
              hitType.push_back(static_cast<int>(HitType::Strip));
            }
          }
        } else {
          LogTrace("TrackingNtuple") << " not pixel and not Strip detector";
        }
      }
      see_hitIdx.push_back(hitIdx);
      see_hitType.push_back(hitType);
      see_nPixel.push_back(std::count(hitType.begin(), hitType.end(), static_cast<int>(HitType::Pixel)));
      see_nGlued.push_back(std::count(hitType.begin(), hitType.end(), static_cast<int>(HitType::Glued)));
      see_nStrip.push_back(std::count(hitType.begin(), hitType.end(), static_cast<int>(HitType::Strip)));
      see_nPhase2OT.push_back(std::count(hitType.begin(), hitType.end(), static_cast<int>(HitType::Phase2OT)));
      see_nCluster.push_back(nClusters);
      //the part below is not strictly needed
      float chi2 = -1;
      if (nHits == 2) {
        TransientTrackingRecHit::RecHitPointer recHit0 = theTTRHBuilder.build(&*(seed.recHits().begin()));
        TransientTrackingRecHit::RecHitPointer recHit1 = theTTRHBuilder.build(&*(seed.recHits().begin() + 1));
        std::vector<GlobalPoint> gp(2);
        std::vector<GlobalError> ge(2);
        gp[0] = recHit0->globalPosition();
        ge[0] = recHit0->globalPositionError();
        gp[1] = recHit1->globalPosition();
        ge[1] = recHit1->globalPositionError();
        LogTrace("TrackingNtuple")
            << "seed " << seedTrackRef.key() << " pt=" << pt << " eta=" << eta << " phi=" << phi << " q=" << charge
            << " - PAIR - ids: " << recHit0->geographicalId().rawId() << " " << recHit1->geographicalId().rawId()
            << " hitpos: " << gp[0] << " " << gp[1] << " trans0: "
            << (recHit0->transientHits().size() > 1 ? recHit0->transientHits()[0]->globalPosition()
                                                    : GlobalPoint(0, 0, 0))
            << " "
            << (recHit0->transientHits().size() > 1 ? recHit0->transientHits()[1]->globalPosition()
                                                    : GlobalPoint(0, 0, 0))
            << " trans1: "
            << (recHit1->transientHits().size() > 1 ? recHit1->transientHits()[0]->globalPosition()
                                                    : GlobalPoint(0, 0, 0))
            << " "
            << (recHit1->transientHits().size() > 1 ? recHit1->transientHits()[1]->globalPosition()
                                                    : GlobalPoint(0, 0, 0))
            << " eta,phi: " << gp[0].eta() << "," << gp[0].phi();
      } else if (nHits == 3) {
        TransientTrackingRecHit::RecHitPointer recHit0 = theTTRHBuilder.build(&*(seed.recHits().begin()));
        TransientTrackingRecHit::RecHitPointer recHit1 = theTTRHBuilder.build(&*(seed.recHits().begin() + 1));
        TransientTrackingRecHit::RecHitPointer recHit2 = theTTRHBuilder.build(&*(seed.recHits().begin() + 2));
        declareDynArray(GlobalPoint, 4, gp);
        declareDynArray(GlobalError, 4, ge);
        declareDynArray(bool, 4, bl);
        gp[0] = recHit0->globalPosition();
        ge[0] = recHit0->globalPositionError();
        int subid0 = recHit0->geographicalId().subdetId();
        bl[0] = (subid0 == StripSubdetector::TIB || subid0 == StripSubdetector::TOB ||
                 subid0 == (int)PixelSubdetector::PixelBarrel);
        gp[1] = recHit1->globalPosition();
        ge[1] = recHit1->globalPositionError();
        int subid1 = recHit1->geographicalId().subdetId();
        bl[1] = (subid1 == StripSubdetector::TIB || subid1 == StripSubdetector::TOB ||
                 subid1 == (int)PixelSubdetector::PixelBarrel);
        gp[2] = recHit2->globalPosition();
        ge[2] = recHit2->globalPositionError();
        int subid2 = recHit2->geographicalId().subdetId();
        bl[2] = (subid2 == StripSubdetector::TIB || subid2 == StripSubdetector::TOB ||
                 subid2 == (int)PixelSubdetector::PixelBarrel);
        RZLine rzLine(gp, ge, bl);
        float seed_chi2 = rzLine.chi2();
        //float seed_pt = state.globalParameters().momentum().perp();
        float seed_pt = pt;
        LogTrace("TrackingNtuple")
            << "seed " << seedTrackRef.key() << " pt=" << pt << " eta=" << eta << " phi=" << phi << " q=" << charge
            << " - TRIPLET - ids: " << recHit0->geographicalId().rawId() << " " << recHit1->geographicalId().rawId()
            << " " << recHit2->geographicalId().rawId() << " hitpos: " << gp[0] << " " << gp[1] << " " << gp[2]
            << " trans0: "
            << (recHit0->transientHits().size() > 1 ? recHit0->transientHits()[0]->globalPosition()
                                                    : GlobalPoint(0, 0, 0))
            << " "
            << (recHit0->transientHits().size() > 1 ? recHit0->transientHits()[1]->globalPosition()
                                                    : GlobalPoint(0, 0, 0))
            << " trans1: "
            << (recHit1->transientHits().size() > 1 ? recHit1->transientHits()[0]->globalPosition()
                                                    : GlobalPoint(0, 0, 0))
            << " "
            << (recHit1->transientHits().size() > 1 ? recHit1->transientHits()[1]->globalPosition()
                                                    : GlobalPoint(0, 0, 0))
            << " trans2: "
            << (recHit2->transientHits().size() > 1 ? recHit2->transientHits()[0]->globalPosition()
                                                    : GlobalPoint(0, 0, 0))
            << " "
            << (recHit2->transientHits().size() > 1 ? recHit2->transientHits()[1]->globalPosition()
                                                    : GlobalPoint(0, 0, 0))
            << " local: "
            << recHit2->localPosition()
            //<< " tsos pos, mom: " << state.globalPosition()<<" "<<state.globalMomentum()
            << " eta,phi: " << gp[0].eta() << "," << gp[0].phi() << " pt,chi2: " << seed_pt << "," << seed_chi2;
        chi2 = seed_chi2;
      }
      see_chi2.push_back(chi2);
    }

    fillTrackingParticlesForSeeds(tpCollection, simRecColl, tpKeyToIndex, offset);
  }
}

void TrackingNtuple::fillTracks(const edm::RefToBaseVector<reco::Track>& tracks,
                                const TrackingParticleRefVector& tpCollection,
                                const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                const TrackingParticleRefKeyToCount& tpKeyToClusterCount,
                                const MagneticField& mf,
                                const reco::BeamSpot& bs,
                                const reco::VertexCollection& vertices,
                                const reco::TrackToTrackingParticleAssociator& associatorByHits,
                                const ClusterTPAssociation& clusterToTPMap,
                                const TransientTrackingRecHitBuilder& theTTRHBuilder,
                                const TrackerTopology& tTopo,
                                const std::set<edm::ProductID>& hitProductIds,
                                const std::map<edm::ProductID, size_t>& seedCollToOffset,
                                const std::vector<const MVACollection*>& mvaColls,
                                const std::vector<const QualityMaskCollection*>& qualColls) {
  reco::RecoToSimCollection recSimColl = associatorByHits.associateRecoToSim(tracks, tpCollection);
  edm::EDConsumerBase::Labels labels;
  labelsForToken(trackToken_, labels);
  LogTrace("TrackingNtuple") << "NEW TRACK LABEL: " << labels.module;

  auto pvPosition = vertices[0].position();

  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    const auto& itTrack = tracks[iTrack];
    int charge = itTrack->charge();
    float pt = itTrack->pt();
    float eta = itTrack->eta();
    const double lambda = itTrack->lambda();
    float chi2 = itTrack->normalizedChi2();
    float ndof = itTrack->ndof();
    float phi = itTrack->phi();
    int nHits = itTrack->numberOfValidHits();
    const reco::HitPattern& hp = itTrack->hitPattern();

    const auto& tkParam = itTrack->parameters();
    auto tkCov = itTrack->covariance();
    tkCov.Invert();

    // Standard track-TP matching
    int nSimHits = 0;
    bool isSimMatched = false;
    std::vector<int> tpIdx;
    std::vector<float> sharedFraction;
    std::vector<float> tpChi2;
    auto foundTPs = recSimColl.find(itTrack);
    if (foundTPs != recSimColl.end()) {
      if (!foundTPs->val.empty()) {
        nSimHits = foundTPs->val[0].first->numberOfTrackerHits();
        isSimMatched = true;
      }
      for (const auto& tpQuality : foundTPs->val) {
        tpIdx.push_back(tpKeyToIndex.at(tpQuality.first.key()));
        sharedFraction.push_back(tpQuality.second);
        tpChi2.push_back(track_associator::trackAssociationChi2(tkParam, tkCov, *(tpCollection[tpIdx.back()]), mf, bs));
      }
    }

    // Search for a best-matching TrackingParticle for a track
    const auto clusters = track_associator::hitsToClusterRefs(
        itTrack->recHitsBegin(),
        itTrack->recHitsEnd());  // TODO: this function is called 3 times per track, try to reduce
    const int nClusters = clusters.size();

    const auto bestKeyCount = findBestMatchingTrackingParticle(*itTrack, clusterToTPMap, tpKeyToIndex);
    const float bestShareFrac = static_cast<float>(bestKeyCount.countClusters) / static_cast<float>(nClusters);
    float bestShareFracSimDenom = 0;
    float bestShareFracSimClusterDenom = 0;
    float bestChi2 = -1;
    if (bestKeyCount.key >= 0) {
      bestShareFracSimDenom =
          static_cast<float>(bestKeyCount.countClusters) /
          static_cast<float>(tpCollection[tpKeyToIndex.at(bestKeyCount.key)]->numberOfTrackerHits());
      bestShareFracSimClusterDenom =
          static_cast<float>(bestKeyCount.countClusters) / static_cast<float>(tpKeyToClusterCount.at(bestKeyCount.key));
      bestChi2 = track_associator::trackAssociationChi2(
          tkParam, tkCov, *(tpCollection[tpKeyToIndex.at(bestKeyCount.key)]), mf, bs);
    }
    // Another way starting from the first hit of the track
    const auto bestFirstHitKeyCount = findMatchingTrackingParticleFromFirstHit(*itTrack, clusterToTPMap, tpKeyToIndex);
    const float bestFirstHitShareFrac =
        static_cast<float>(bestFirstHitKeyCount.countClusters) / static_cast<float>(nClusters);
    float bestFirstHitShareFracSimDenom = 0;
    float bestFirstHitShareFracSimClusterDenom = 0;
    float bestFirstHitChi2 = -1;
    if (bestFirstHitKeyCount.key >= 0) {
      bestFirstHitShareFracSimDenom =
          static_cast<float>(bestFirstHitKeyCount.countClusters) /
          static_cast<float>(tpCollection[tpKeyToIndex.at(bestFirstHitKeyCount.key)]->numberOfTrackerHits());
      bestFirstHitShareFracSimClusterDenom = static_cast<float>(bestFirstHitKeyCount.countClusters) /
                                             static_cast<float>(tpKeyToClusterCount.at(bestFirstHitKeyCount.key));
      bestFirstHitChi2 = track_associator::trackAssociationChi2(
          tkParam, tkCov, *(tpCollection[tpKeyToIndex.at(bestFirstHitKeyCount.key)]), mf, bs);
    }

    float chi2_1Dmod = chi2;
    int count1dhits = 0;
    for (auto iHit = itTrack->recHitsBegin(), iEnd = itTrack->recHitsEnd(); iHit != iEnd; ++iHit) {
      const TrackingRecHit& hit = **iHit;
      if (hit.isValid() && typeid(hit) == typeid(SiStripRecHit1D))
        ++count1dhits;
    }
    if (count1dhits > 0) {
      chi2_1Dmod = (chi2 + count1dhits) / (ndof + count1dhits);
    }

    Point bestPV = getBestVertex(*itTrack, vertices);

    trk_px.push_back(itTrack->px());
    trk_py.push_back(itTrack->py());
    trk_pz.push_back(itTrack->pz());
    trk_pt.push_back(pt);
    trk_inner_px.push_back(itTrack->innerMomentum().x());
    trk_inner_py.push_back(itTrack->innerMomentum().y());
    trk_inner_pz.push_back(itTrack->innerMomentum().z());
    trk_inner_pt.push_back(itTrack->innerMomentum().rho());
    trk_outer_px.push_back(itTrack->outerMomentum().x());
    trk_outer_py.push_back(itTrack->outerMomentum().y());
    trk_outer_pz.push_back(itTrack->outerMomentum().z());
    trk_outer_pt.push_back(itTrack->outerMomentum().rho());
    trk_eta.push_back(eta);
    trk_lambda.push_back(lambda);
    trk_cotTheta.push_back(1 / tan(M_PI * 0.5 - lambda));
    trk_phi.push_back(phi);
    trk_dxy.push_back(itTrack->dxy(bs.position()));
    trk_dz.push_back(itTrack->dz(bs.position()));
    trk_dxyPV.push_back(itTrack->dxy(pvPosition));
    trk_dzPV.push_back(itTrack->dz(pvPosition));
    trk_dxyClosestPV.push_back(itTrack->dxy(bestPV));
    trk_dzClosestPV.push_back(itTrack->dz(bestPV));
    trk_ptErr.push_back(itTrack->ptError());
    trk_etaErr.push_back(itTrack->etaError());
    trk_lambdaErr.push_back(itTrack->lambdaError());
    trk_phiErr.push_back(itTrack->phiError());
    trk_dxyErr.push_back(itTrack->dxyError());
    trk_dzErr.push_back(itTrack->dzError());
    trk_refpoint_x.push_back(itTrack->vx());
    trk_refpoint_y.push_back(itTrack->vy());
    trk_refpoint_z.push_back(itTrack->vz());
    trk_nChi2.push_back(chi2);
    trk_nChi2_1Dmod.push_back(chi2_1Dmod);
    trk_ndof.push_back(ndof);
    trk_q.push_back(charge);
    trk_nValid.push_back(hp.numberOfValidHits());
    trk_nLost.push_back(hp.numberOfLostHits(reco::HitPattern::TRACK_HITS));
    trk_nInactive.push_back(hp.trackerLayersTotallyOffOrBad(reco::HitPattern::TRACK_HITS));
    trk_nPixel.push_back(hp.numberOfValidPixelHits());
    trk_nStrip.push_back(hp.numberOfValidStripHits());
    trk_nOuterLost.push_back(hp.numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS));
    trk_nInnerLost.push_back(hp.numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS));
    trk_nOuterInactive.push_back(hp.trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_OUTER_HITS));
    trk_nInnerInactive.push_back(hp.trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_INNER_HITS));
    trk_nPixelLay.push_back(hp.pixelLayersWithMeasurement());
    trk_nStripLay.push_back(hp.stripLayersWithMeasurement());
    trk_n3DLay.push_back(hp.numberOfValidStripLayersWithMonoAndStereo() + hp.pixelLayersWithMeasurement());
    trk_nLostLay.push_back(hp.trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS));
    trk_nCluster.push_back(nClusters);
    trk_algo.push_back(itTrack->algo());
    trk_originalAlgo.push_back(itTrack->originalAlgo());
    trk_algoMask.push_back(itTrack->algoMaskUL());
    trk_stopReason.push_back(itTrack->stopReason());
    trk_isHP.push_back(itTrack->quality(reco::TrackBase::highPurity));
    if (includeMVA_) {
      for (size_t i = 0; i < trk_mvas.size(); ++i) {
        trk_mvas[i].push_back((*(mvaColls[i]))[iTrack]);
        trk_qualityMasks[i].push_back((*(qualColls[i]))[iTrack]);
      }
    }
    if (includeSeeds_) {
      auto offset = seedCollToOffset.find(itTrack->seedRef().id());
      if (offset == seedCollToOffset.end()) {
        throw cms::Exception("Configuration")
            << "Track algo '" << reco::TrackBase::algoName(itTrack->algo()) << "' originalAlgo '"
            << reco::TrackBase::algoName(itTrack->originalAlgo()) << "' refers to seed collection "
            << itTrack->seedRef().id()
            << ", but that seed collection is not given as an input. The following collections were given as an input "
            << make_ProductIDMapPrinter(seedCollToOffset);
      }

      const auto seedIndex = offset->second + itTrack->seedRef().key();
      trk_seedIdx.push_back(seedIndex);
      if (see_trkIdx[seedIndex] != -1) {
        throw cms::Exception("LogicError") << "Track index has already been set for seed " << seedIndex << " to "
                                           << see_trkIdx[seedIndex] << "; was trying to set it to " << iTrack;
      }
      see_trkIdx[seedIndex] = iTrack;
    }
    trk_vtxIdx.push_back(-1);  // to be set correctly in fillVertices
    if (includeTrackingParticles_) {
      trk_simTrkIdx.push_back(tpIdx);
      trk_simTrkShareFrac.push_back(sharedFraction);
      trk_simTrkNChi2.push_back(tpChi2);
      trk_bestSimTrkIdx.push_back(bestKeyCount.key >= 0 ? tpKeyToIndex.at(bestKeyCount.key) : -1);
      trk_bestFromFirstHitSimTrkIdx.push_back(bestFirstHitKeyCount.key >= 0 ? tpKeyToIndex.at(bestFirstHitKeyCount.key)
                                                                            : -1);
    } else {
      trk_isTrue.push_back(!tpIdx.empty());
    }
    trk_bestSimTrkShareFrac.push_back(bestShareFrac);
    trk_bestSimTrkShareFracSimDenom.push_back(bestShareFracSimDenom);
    trk_bestSimTrkShareFracSimClusterDenom.push_back(bestShareFracSimClusterDenom);
    trk_bestSimTrkNChi2.push_back(bestChi2);
    trk_bestFromFirstHitSimTrkShareFrac.push_back(bestFirstHitShareFrac);
    trk_bestFromFirstHitSimTrkShareFracSimDenom.push_back(bestFirstHitShareFracSimDenom);
    trk_bestFromFirstHitSimTrkShareFracSimClusterDenom.push_back(bestFirstHitShareFracSimClusterDenom);
    trk_bestFromFirstHitSimTrkNChi2.push_back(bestFirstHitChi2);

    LogTrace("TrackingNtuple") << "Track #" << itTrack.key() << " with q=" << charge << ", pT=" << pt
                               << " GeV, eta: " << eta << ", phi: " << phi << ", chi2=" << chi2 << ", Nhits=" << nHits
                               << ", algo=" << itTrack->algoName(itTrack->algo()).c_str()
                               << " hp=" << itTrack->quality(reco::TrackBase::highPurity)
                               << " seed#=" << itTrack->seedRef().key() << " simMatch=" << isSimMatched
                               << " nSimHits=" << nSimHits
                               << " sharedFraction=" << (sharedFraction.empty() ? -1 : sharedFraction[0])
                               << " tpIdx=" << (tpIdx.empty() ? -1 : tpIdx[0]);
    std::vector<int> hitIdx;
    std::vector<int> hitType;

    for (auto i = itTrack->recHitsBegin(); i != itTrack->recHitsEnd(); i++) {
      TransientTrackingRecHit::RecHitPointer hit = theTTRHBuilder.build(&**i);
      DetId hitId = hit->geographicalId();
      LogTrace("TrackingNtuple") << "hit #" << std::distance(itTrack->recHitsBegin(), i)
                                 << " subdet=" << hitId.subdetId();
      if (hitId.det() != DetId::Tracker)
        continue;

      LogTrace("TrackingNtuple") << " " << subdetstring(hitId.subdetId()) << " " << tTopo.layer(hitId);

      if (hit->isValid()) {
        //ugly... but works
        const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&*hit);
        const auto& clusterRef = bhit->firstClusterRef();
        unsigned int clusterKey;
        if (clusterRef.isPixel()) {
          clusterKey = clusterRef.cluster_pixel().key();
        } else if (clusterRef.isPhase2()) {
          clusterKey = clusterRef.cluster_phase2OT().key();
        } else {
          clusterKey = clusterRef.cluster_strip().key();
        }

        LogTrace("TrackingNtuple") << " id: " << hitId.rawId() << " - globalPos =" << hit->globalPosition()
                                   << " cluster=" << clusterKey << " clusterRef ID=" << clusterRef.id()
                                   << " eta,phi: " << hit->globalPosition().eta() << "," << hit->globalPosition().phi();
        if (includeAllHits_) {
          checkProductID(hitProductIds, clusterRef.id(), "track");
          if (clusterRef.isPixel()) {
            pix_trkIdx[clusterKey].push_back(iTrack);
          } else if (clusterRef.isPhase2()) {
            ph2_trkIdx[clusterKey].push_back(iTrack);
          } else {
            str_trkIdx[clusterKey].push_back(iTrack);
          }
        }

        hitIdx.push_back(clusterKey);
        if (clusterRef.isPixel()) {
          hitType.push_back(static_cast<int>(HitType::Pixel));
        } else if (clusterRef.isPhase2()) {
          hitType.push_back(static_cast<int>(HitType::Phase2OT));
        } else {
          hitType.push_back(static_cast<int>(HitType::Strip));
        }
      } else {
        LogTrace("TrackingNtuple") << " - invalid hit";

        hitIdx.push_back(inv_isBarrel.size());
        hitType.push_back(static_cast<int>(HitType::Invalid));

        inv_isBarrel.push_back(hitId.subdetId() == 1);
        if (includeStripHits_)
          inv_detId.push_back(tTopo, hitId);
        else
          inv_detId_phase2.push_back(tTopo, hitId);
        inv_type.push_back(hit->getType());
      }
    }

    trk_hitIdx.push_back(hitIdx);
    trk_hitType.push_back(hitType);
  }
}

void TrackingNtuple::fillTrackingParticles(const edm::Event& iEvent,
                                           const edm::EventSetup& iSetup,
                                           const edm::RefToBaseVector<reco::Track>& tracks,
                                           const reco::BeamSpot& bs,
                                           const TrackingParticleRefVector& tpCollection,
                                           const TrackingVertexRefKeyToIndex& tvKeyToIndex,
                                           const reco::TrackToTrackingParticleAssociator& associatorByHits,
                                           const std::vector<TPHitIndex>& tpHitList,
                                           const TrackingParticleRefKeyToCount& tpKeyToClusterCount) {
  const ParametersDefinerForTP* parametersDefiner = &iSetup.getData(paramsDefineToken_);

  // Number of 3D layers for TPs
  edm::Handle<edm::ValueMap<unsigned int>> tpNLayersH;
  iEvent.getByToken(tpNLayersToken_, tpNLayersH);
  const auto& nLayers_tPCeff = *tpNLayersH;

  iEvent.getByToken(tpNPixelLayersToken_, tpNLayersH);
  const auto& nPixelLayers_tPCeff = *tpNLayersH;

  iEvent.getByToken(tpNStripStereoLayersToken_, tpNLayersH);
  const auto& nStripMonoAndStereoLayers_tPCeff = *tpNLayersH;

  reco::SimToRecoCollection simRecColl = associatorByHits.associateSimToReco(tracks, tpCollection);

  for (const TrackingParticleRef& tp : tpCollection) {
    LogTrace("TrackingNtuple") << "tracking particle pt=" << tp->pt() << " eta=" << tp->eta() << " phi=" << tp->phi();
    bool isRecoMatched = false;
    std::vector<int> tkIdx;
    std::vector<float> sharedFraction;
    auto foundTracks = simRecColl.find(tp);
    if (foundTracks != simRecColl.end()) {
      isRecoMatched = true;
      for (const auto& trackQuality : foundTracks->val) {
        sharedFraction.push_back(trackQuality.second);
        tkIdx.push_back(trackQuality.first.key());
      }
    }

    sim_genPdgIds.emplace_back();
    for (const auto& genRef : tp->genParticles()) {
      if (genRef.isNonnull())
        sim_genPdgIds.back().push_back(genRef->pdgId());
    }

    bool isFromBHadron = false;
    // Logic is similar to SimTracker/TrackHistory
    if (tracer_.evaluate(tp)) {  // ignore TP if history can not be traced
      // following is from TrackClassifier::processesAtGenerator()
      HistoryBase::RecoGenParticleTrail const& recoGenParticleTrail = tracer_.recoGenParticleTrail();
      for (const auto& particle : recoGenParticleTrail) {
        HepPDT::ParticleID particleID(particle->pdgId());
        if (particleID.hasBottom()) {
          isFromBHadron = true;
          break;
        }
      }
    }

    LogTrace("TrackingNtuple") << "matched to tracks = " << make_VectorPrinter(tkIdx)
                               << " isRecoMatched=" << isRecoMatched;
    sim_event.push_back(tp->eventId().event());
    sim_bunchCrossing.push_back(tp->eventId().bunchCrossing());
    sim_pdgId.push_back(tp->pdgId());
    sim_isFromBHadron.push_back(isFromBHadron);
    sim_px.push_back(tp->px());
    sim_py.push_back(tp->py());
    sim_pz.push_back(tp->pz());
    sim_pt.push_back(tp->pt());
    sim_eta.push_back(tp->eta());
    sim_phi.push_back(tp->phi());
    sim_q.push_back(tp->charge());
    sim_trkIdx.push_back(tkIdx);
    sim_trkShareFrac.push_back(sharedFraction);
    sim_parentVtxIdx.push_back(tvKeyToIndex.at(tp->parentVertex().key()));
    std::vector<int> decayIdx;
    for (const auto& v : tp->decayVertices())
      decayIdx.push_back(tvKeyToIndex.at(v.key()));
    sim_decayVtxIdx.push_back(decayIdx);

    //Calcualte the impact parameters w.r.t. PCA
    TrackingParticle::Vector momentum = parametersDefiner->momentum(iEvent, iSetup, tp);
    TrackingParticle::Point vertex = parametersDefiner->vertex(iEvent, iSetup, tp);
    auto dxySim = TrackingParticleIP::dxy(vertex, momentum, bs.position());
    auto dzSim = TrackingParticleIP::dz(vertex, momentum, bs.position());
    const double lambdaSim = M_PI / 2 - momentum.theta();
    sim_pca_pt.push_back(std::sqrt(momentum.perp2()));
    sim_pca_eta.push_back(momentum.Eta());
    sim_pca_lambda.push_back(lambdaSim);
    sim_pca_cotTheta.push_back(1 / tan(M_PI * 0.5 - lambdaSim));
    sim_pca_phi.push_back(momentum.phi());
    sim_pca_dxy.push_back(dxySim);
    sim_pca_dz.push_back(dzSim);

    std::vector<int> hitIdx;
    int nPixel = 0, nStrip = 0;
    auto rangeHit = std::equal_range(tpHitList.begin(), tpHitList.end(), TPHitIndex(tp.key()), tpHitIndexListLess);
    for (auto ip = rangeHit.first; ip != rangeHit.second; ++ip) {
      auto type = HitType::Unknown;
      if (!simhit_hitType[ip->simHitIdx].empty())
        type = static_cast<HitType>(simhit_hitType[ip->simHitIdx][0]);
      LogTrace("TrackingNtuple") << "simhit=" << ip->simHitIdx << " type=" << static_cast<int>(type);
      hitIdx.push_back(ip->simHitIdx);
      const auto detid = DetId(includeStripHits_ ? simhit_detId[ip->simHitIdx] : simhit_detId_phase2[ip->simHitIdx]);
      if (detid.det() != DetId::Tracker) {
        throw cms::Exception("LogicError") << "Encountered SimHit for TP " << tp.key() << " with DetId "
                                           << detid.rawId() << " whose det() is not Tracker but " << detid.det();
      }
      const auto subdet = detid.subdetId();
      switch (subdet) {
        case PixelSubdetector::PixelBarrel:
        case PixelSubdetector::PixelEndcap:
          ++nPixel;
          break;
        case StripSubdetector::TIB:
        case StripSubdetector::TID:
        case StripSubdetector::TOB:
        case StripSubdetector::TEC:
          ++nStrip;
          break;
        default:
          throw cms::Exception("LogicError") << "Encountered SimHit for TP " << tp.key() << " with DetId "
                                             << detid.rawId() << " whose subdet is not recognized, is " << subdet;
      };
    }
    sim_nValid.push_back(hitIdx.size());
    sim_nPixel.push_back(nPixel);
    sim_nStrip.push_back(nStrip);

    const auto nSimLayers = nLayers_tPCeff[tp];
    const auto nSimPixelLayers = nPixelLayers_tPCeff[tp];
    const auto nSimStripMonoAndStereoLayers = nStripMonoAndStereoLayers_tPCeff[tp];
    sim_nLay.push_back(nSimLayers);
    sim_nPixelLay.push_back(nSimPixelLayers);
    sim_n3DLay.push_back(nSimPixelLayers + nSimStripMonoAndStereoLayers);

    sim_nTrackerHits.push_back(tp->numberOfTrackerHits());
    auto found = tpKeyToClusterCount.find(tp.key());
    sim_nRecoClusters.push_back(found != cend(tpKeyToClusterCount) ? found->second : 0);

    sim_simHitIdx.push_back(hitIdx);
  }
}

// called from fillSeeds
void TrackingNtuple::fillTrackingParticlesForSeeds(const TrackingParticleRefVector& tpCollection,
                                                   const reco::SimToRecoCollection& simRecColl,
                                                   const TrackingParticleRefKeyToIndex& tpKeyToIndex,
                                                   const unsigned int seedOffset) {
  if (sim_seedIdx.empty())  // first call
    sim_seedIdx.resize(tpCollection.size());

  for (const auto& keyVal : simRecColl) {
    const auto& tpRef = keyVal.key;
    auto found = tpKeyToIndex.find(tpRef.key());
    if (found == tpKeyToIndex.end())
      throw cms::Exception("Assert") << __FILE__ << ":" << __LINE__ << " fillTrackingParticlesForSeeds: tpRef.key() "
                                     << tpRef.key() << " not found from tpKeyToIndex. tpKeyToIndex size "
                                     << tpKeyToIndex.size();
    const auto tpIndex = found->second;
    for (const auto& pair : keyVal.val) {
      const auto& seedRef = pair.first->seedRef();
      sim_seedIdx[tpIndex].push_back(seedOffset + seedRef.key());
    }
  }
}

void TrackingNtuple::fillVertices(const reco::VertexCollection& vertices,
                                  const edm::RefToBaseVector<reco::Track>& tracks) {
  for (size_t iVertex = 0, size = vertices.size(); iVertex < size; ++iVertex) {
    const reco::Vertex& vertex = vertices[iVertex];
    vtx_x.push_back(vertex.x());
    vtx_y.push_back(vertex.y());
    vtx_z.push_back(vertex.z());
    vtx_xErr.push_back(vertex.xError());
    vtx_yErr.push_back(vertex.yError());
    vtx_zErr.push_back(vertex.zError());
    vtx_chi2.push_back(vertex.chi2());
    vtx_ndof.push_back(vertex.ndof());
    vtx_fake.push_back(vertex.isFake());
    vtx_valid.push_back(vertex.isValid());

    std::vector<int> trkIdx;
    for (auto iTrack = vertex.tracks_begin(); iTrack != vertex.tracks_end(); ++iTrack) {
      // Ignore link if vertex was fit from a track collection different from the input
      if (iTrack->id() != tracks.id())
        continue;

      trkIdx.push_back(iTrack->key());

      if (trk_vtxIdx[iTrack->key()] != -1) {
        throw cms::Exception("LogicError") << "Vertex index has already been set for track " << iTrack->key() << " to "
                                           << trk_vtxIdx[iTrack->key()] << "; was trying to set it to " << iVertex;
      }
      trk_vtxIdx[iTrack->key()] = iVertex;
    }
    vtx_trkIdx.push_back(trkIdx);
  }
}

void TrackingNtuple::fillTrackingVertices(const TrackingVertexRefVector& trackingVertices,
                                          const TrackingParticleRefKeyToIndex& tpKeyToIndex) {
  int current_event = -1;
  for (const auto& ref : trackingVertices) {
    const TrackingVertex v = *ref;
    if (v.eventId().event() != current_event) {
      // next PV
      current_event = v.eventId().event();
      simpv_idx.push_back(simvtx_x.size());
    }

    unsigned int processType = std::numeric_limits<unsigned int>::max();
    if (!v.g4Vertices().empty()) {
      processType = v.g4Vertices()[0].processType();
    }

    simvtx_event.push_back(v.eventId().event());
    simvtx_bunchCrossing.push_back(v.eventId().bunchCrossing());
    simvtx_processType.push_back(processType);

    simvtx_x.push_back(v.position().x());
    simvtx_y.push_back(v.position().y());
    simvtx_z.push_back(v.position().z());

    auto fill = [&](const TrackingParticleRefVector& tps, std::vector<int>& idx) {
      for (const auto& tpRef : tps) {
        auto found = tpKeyToIndex.find(tpRef.key());
        if (found != tpKeyToIndex.end()) {
          idx.push_back(found->second);
        }
      }
    };

    std::vector<int> sourceIdx;
    std::vector<int> daughterIdx;
    fill(v.sourceTracks(), sourceIdx);
    fill(v.daughterTracks(), daughterIdx);

    simvtx_sourceSimIdx.push_back(sourceIdx);
    simvtx_daughterSimIdx.push_back(daughterIdx);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TrackingNtuple::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<edm::InputTag>>(
      "seedTracks",
      std::vector<edm::InputTag>{edm::InputTag("seedTracksinitialStepSeeds"),
                                 edm::InputTag("seedTracksdetachedTripletStepSeeds"),
                                 edm::InputTag("seedTrackspixelPairStepSeeds"),
                                 edm::InputTag("seedTrackslowPtTripletStepSeeds"),
                                 edm::InputTag("seedTracksmixedTripletStepSeeds"),
                                 edm::InputTag("seedTrackspixelLessStepSeeds"),
                                 edm::InputTag("seedTrackstobTecStepSeeds"),
                                 edm::InputTag("seedTracksjetCoreRegionalStepSeeds"),
                                 edm::InputTag("seedTracksmuonSeededSeedsInOut"),
                                 edm::InputTag("seedTracksmuonSeededSeedsOutIn")});
  desc.addUntracked<std::vector<edm::InputTag>>(
      "trackCandidates",
      std::vector<edm::InputTag>{edm::InputTag("initialStepTrackCandidates"),
                                 edm::InputTag("detachedTripletStepTrackCandidates"),
                                 edm::InputTag("pixelPairStepTrackCandidates"),
                                 edm::InputTag("lowPtTripletStepTrackCandidates"),
                                 edm::InputTag("mixedTripletStepTrackCandidates"),
                                 edm::InputTag("pixelLessStepTrackCandidates"),
                                 edm::InputTag("tobTecStepTrackCandidates"),
                                 edm::InputTag("jetCoreRegionalStepTrackCandidates"),
                                 edm::InputTag("muonSeededTrackCandidatesInOut"),
                                 edm::InputTag("muonSeededTrackCandidatesOutIn")});
  desc.addUntracked<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.addUntracked<std::vector<std::string>>("trackMVAs", std::vector<std::string>{{"generalTracks"}});

  edm::ParameterSetDescription cMaskDesc;
  cMaskDesc.addUntracked<unsigned int>("index");
  cMaskDesc.addUntracked<edm::InputTag>("src");
  std::vector<edm::ParameterSet> cMasks;
  auto addMask = [&cMasks](reco::Track::TrackAlgorithm algo) {
    edm::ParameterSet ps;
    ps.addUntrackedParameter<unsigned int>("index", static_cast<unsigned int>(algo));
    ps.addUntrackedParameter<edm::InputTag>("src", {reco::Track::algoName(algo) + "Clusters"});
    cMasks.push_back(ps);
  };
  addMask(reco::Track::detachedQuadStep);
  addMask(reco::Track::highPtTripletStep);
  addMask(reco::Track::detachedTripletStep);
  addMask(reco::Track::lowPtQuadStep);
  addMask(reco::Track::lowPtTripletStep);
  addMask(reco::Track::mixedTripletStep);
  addMask(reco::Track::pixelLessStep);
  addMask(reco::Track::pixelPairStep);
  addMask(reco::Track::tobTecStep);
  desc.addVPSetUntracked("clusterMasks", cMaskDesc, cMasks);

  desc.addUntracked<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
  desc.addUntracked<bool>("trackingParticlesRef", false);
  desc.addUntracked<edm::InputTag>("clusterTPMap", edm::InputTag("tpClusterProducer"));
  desc.addUntracked<edm::InputTag>("simHitTPMap", edm::InputTag("simHitTPAssocProducer"));
  desc.addUntracked<edm::InputTag>("trackAssociator", edm::InputTag("quickTrackAssociatorByHits"));
  desc.addUntracked<edm::InputTag>("pixelDigiSimLink", edm::InputTag("simSiPixelDigis"));
  desc.addUntracked<edm::InputTag>("stripDigiSimLink", edm::InputTag("simSiStripDigis"));
  desc.addUntracked<edm::InputTag>("phase2OTSimLink", edm::InputTag(""));
  desc.addUntracked<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.addUntracked<edm::InputTag>("pixelRecHits", edm::InputTag("siPixelRecHits"));
  desc.addUntracked<edm::InputTag>("stripRphiRecHits", edm::InputTag("siStripMatchedRecHits", "rphiRecHit"));
  desc.addUntracked<edm::InputTag>("stripStereoRecHits", edm::InputTag("siStripMatchedRecHits", "stereoRecHit"));
  desc.addUntracked<edm::InputTag>("stripMatchedRecHits", edm::InputTag("siStripMatchedRecHits", "matchedRecHit"));
  desc.addUntracked<edm::InputTag>("phase2OTRecHits", edm::InputTag("siPhase2RecHits"));
  desc.addUntracked<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.addUntracked<edm::InputTag>("trackingVertices", edm::InputTag("mix", "MergedTrackTruth"));
  desc.addUntracked<edm::InputTag>("trackingParticleNlayers",
                                   edm::InputTag("trackingParticleNumberOfLayersProducer", "trackerLayers"));
  desc.addUntracked<edm::InputTag>("trackingParticleNpixellayers",
                                   edm::InputTag("trackingParticleNumberOfLayersProducer", "pixelLayers"));
  desc.addUntracked<edm::InputTag>("trackingParticleNstripstereolayers",
                                   edm::InputTag("trackingParticleNumberOfLayersProducer", "stripStereoLayers"));
  desc.addUntracked<std::string>("TTRHBuilder", "WithTrackAngle");
  desc.addUntracked<std::string>("parametersDefiner", "LhcParametersDefinerForTP");
  desc.addUntracked<bool>("includeSeeds", false);
  desc.addUntracked<bool>("addSeedCurvCov", false);
  desc.addUntracked<bool>("includeAllHits", false);
  desc.addUntracked<bool>("includeMVA", true);
  desc.addUntracked<bool>("includeTrackingParticles", true);
  desc.addUntracked<bool>("includeOOT", false);
  desc.addUntracked<bool>("keepEleSimHits", false);
  desc.addUntracked<bool>("saveSimHitsP3", false);
  desc.addUntracked<bool>("simHitBySignificance", false);
  descriptions.add("trackingNtuple", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackingNtuple);
