#ifndef SimTracker_TrackerHitAssociation_ClusterTPAssociation_h
#define SimTracker_TrackerHitAssociation_ClusterTPAssociation_h

#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include <vector>
#include <utility>
#include <algorithm>

class ClusterTPAssociation {
public:
  using key_type = OmniClusterRef;
  using mapped_type = TrackingParticleRef;
  using value_type = std::pair<key_type, mapped_type>;
  using map_type = std::vector<value_type>;
  using const_iterator = typename map_type::const_iterator;
  using range = std::pair<const_iterator, const_iterator>;

  ClusterTPAssociation() {}

  void emplace_back(const OmniClusterRef& cluster, const TrackingParticleRef& tp) {
    map_.emplace_back(cluster, tp);
  }
  void sort() { std::sort(map_.begin(), map_.end(), compare); }

  bool empty() const { return map_.empty(); }
  size_t size() const { return map_.size(); }

  const_iterator begin()  const { return map_.begin(); }
  const_iterator cbegin() const { return map_.cbegin(); }
  const_iterator end()    const { return map_.end(); }
  const_iterator cend()   const { return map_.end(); }

  range equal_range(const OmniClusterRef& key) const {
    return std::equal_range(map_.begin(), map_.end(), value_type(key, TrackingParticleRef()), compare);
  }

private:
  static bool compare(const value_type& i, const value_type& j) {
    return i.first.rawIndex() > j.first.rawIndex();
  }

  map_type map_;
};

#endif
