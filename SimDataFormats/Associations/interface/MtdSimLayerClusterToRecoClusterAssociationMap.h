#ifndef SimDataFormats_Associations_MtdSimLayerClusterToRecoClusterAssociationMap_h
#define SimDataFormats_Associations_MtdSimLayerClusterToRecoClusterAssociationMap_h

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/HandleBase.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"

#include <vector>
#include <utility>
#include <algorithm>

/**
 * Maps MtdSimLayerCluserRef to FTLClusterRef
 *
 */
class MtdSimLayerClusterToRecoClusterAssociationMap {
public:
  using key_type = MtdSimLayerClusterRef;
  using mapped_type = FTLClusterRef;
  using value_type = std::pair<key_type, std::vector<mapped_type>>;
  using map_type = std::vector<value_type>;
  using const_iterator = typename map_type::const_iterator;
  using range = std::pair<const_iterator, const_iterator>;

  /// Constructor
  MtdSimLayerClusterToRecoClusterAssociationMap();
  /// Destructor
  ~MtdSimLayerClusterToRecoClusterAssociationMap();

  void emplace_back(const MtdSimLayerClusterRef& simClus, std::vector<FTLClusterRef>& recoClusVect) {
    map_.emplace_back(simClus, recoClusVect);
  }

  void post_insert() { std::sort(map_.begin(), map_.end(), compare); }

  bool empty() const { return map_.empty(); }
  size_t size() const { return map_.size(); }

  const_iterator begin() const { return map_.begin(); }
  const_iterator cbegin() const { return map_.cbegin(); }
  const_iterator end() const { return map_.end(); }
  const_iterator cend() const { return map_.cend(); }

  range equal_range(const MtdSimLayerClusterRef& key) const {
    return std::equal_range(map_.begin(), map_.end(), value_type(key, std::vector<FTLClusterRef>()), compare);
  }

  const map_type& map() const { return map_; }

private:
  static bool compare(const value_type& i, const value_type& j) {
    const auto& i_hAndE = (i.first)->hits_and_energies();
    const auto& j_hAndE = (j.first)->hits_and_energies();

    auto imin = std::min_element(i_hAndE.begin(),
                                 i_hAndE.end(),
                                 [](std::pair<uint64_t, float> a, std::pair<uint64_t, float> b) { return a < b; });

    auto jmin = std::min_element(j_hAndE.begin(),
                                 j_hAndE.end(),
                                 [](std::pair<uint64_t, float> a, std::pair<uint64_t, float> b) { return a < b; });

    return (*imin < *jmin);
  }

  map_type map_;
};

#endif
