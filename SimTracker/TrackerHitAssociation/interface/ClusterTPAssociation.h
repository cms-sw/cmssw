#ifndef SimTracker_TrackerHitAssociation_ClusterTPAssociation_h
#define SimTracker_TrackerHitAssociation_ClusterTPAssociation_h

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/HandleBase.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "FWCore/Utilities/interface/VecArray.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include <vector>
#include <utility>
#include <algorithm>

/**
 * Maps OmniClusterRefs to TrackingParticleRefs
 *
 * Assumes that the TrackingParticleRefs point to a single
 * TrackingParticle collection.
 */
class ClusterTPAssociation {
public:
  using key_type = OmniClusterRef;
  using mapped_type = TrackingParticleRef;
  using value_type = std::pair<key_type, mapped_type>;
  using map_type = std::vector<value_type>;
  using const_iterator = typename map_type::const_iterator;
  using range = std::pair<const_iterator, const_iterator>;

  ClusterTPAssociation() {}
  explicit ClusterTPAssociation(const edm::HandleBase& mappedHandle): ClusterTPAssociation(mappedHandle.id()) {}
  explicit ClusterTPAssociation(const edm::ProductID& mappedProductId): mappedProductId_(mappedProductId) {}

  void emplace_back(const OmniClusterRef& cluster, const TrackingParticleRef& tp) {
    checkMappedProductID(tp);
    auto foundKeyID = std::find(std::begin(keyProductIDs_), std::end(keyProductIDs_), cluster.id());
    if(foundKeyID == std::end(keyProductIDs_)) {
      keyProductIDs_.emplace_back(cluster.id());
    }
    map_.emplace_back(cluster, tp);
  }
  void sortAndUnique() {
    std::sort(map_.begin(), map_.end(), compareSort);
    auto last = std::unique(map_.begin(), map_.end());
    map_.erase(last, map_.end());
    map_.shrink_to_fit();
  }
  void swap(ClusterTPAssociation& other) {
    map_.swap(other.map_);
    mappedProductId_.swap(other.mappedProductId_);
  }

  bool empty() const { return map_.empty(); }
  size_t size() const { return map_.size(); }

  const_iterator begin()  const { return map_.begin(); }
  const_iterator cbegin() const { return map_.cbegin(); }
  const_iterator end()    const { return map_.end(); }
  const_iterator cend()   const { return map_.end(); }

  range equal_range(const OmniClusterRef& key) const {
    checkKeyProductID(key);
    return std::equal_range(map_.begin(), map_.end(), value_type(key, TrackingParticleRef()), compare);
  }
  
  const map_type& map() const { return map_; }

  void checkKeyProductID(const OmniClusterRef& key) const { checkKeyProductID(key.id()); }
  void checkKeyProductID(const edm::ProductID& id) const;

  void checkMappedProductID(const edm::HandleBase& mappedHandle) const { checkMappedProductID(mappedHandle.id()); }
  void checkMappedProductID(const TrackingParticleRef& tp) const { checkMappedProductID(tp.id()); }
  void checkMappedProductID(const edm::ProductID& id) const;

private:
  static bool compare(const value_type& i, const value_type& j) {
    return i.first.rawIndex() > j.first.rawIndex();
  }

  static bool compareSort(const value_type& i, const value_type& j) {
    // For sorting compare also TrackingParticle keys in order to
    // remove duplicate matches
    return compare(i, j) || (i.first.rawIndex() == j.first.rawIndex() && i.second.key() > j.second.key());
  }

  map_type map_;
  edm::VecArray<edm::ProductID, 2> keyProductIDs_;
  edm::ProductID mappedProductId_;
};

#endif
