#ifndef SimDataFormats_Associations_MtdSimMergedClusterToRecoMergedClusterAssociationMap_h
#define SimDataFormats_Associations_MtdSimMergedClusterToRecoMergedClusterAssociationMap_h

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/HandleBase.h"
#include "DataFormats/FTLRecHit/interface/FTLMergedClusterCollections.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedClusterFwd.h"

#include <vector>
#include <utility>
#include <algorithm>

/**
 * Maps MtdSimMergedClusterRef to FTLMergedClusterRef
 *
 */
class MtdSimMergedClusterToRecoMergedClusterAssociationMap {
public:
  using key_type = MtdSimMergedClusterRef;
  using mapped_type = FTLMergedClusterRef;
  using value_type = std::pair<key_type, std::vector<mapped_type>>;
  using map_type = std::vector<value_type>;
  using const_iterator = typename map_type::const_iterator;
  using range = std::pair<const_iterator, const_iterator>;

  /// Constructor
  MtdSimMergedClusterToRecoMergedClusterAssociationMap();
  /// Destructor
  ~MtdSimMergedClusterToRecoMergedClusterAssociationMap();

  void emplace_back(const MtdSimMergedClusterRef& simClus, std::vector<FTLMergedClusterRef>& recoClusVect) {
    map_.emplace_back(simClus, recoClusVect);
  }

  void post_insert() { std::sort(map_.begin(), map_.end(), compare); }

  bool empty() const { return map_.empty(); }
  size_t size() const { return map_.size(); }

  const_iterator begin() const { return map_.begin(); }
  const_iterator cbegin() const { return map_.cbegin(); }
  const_iterator end() const { return map_.end(); }
  const_iterator cend() const { return map_.cend(); }

  range equal_range(const MtdSimMergedClusterRef& key) const {
    auto result =
        std::equal_range(map_.begin(), map_.end(), value_type(key, std::vector<FTLMergedClusterRef>()), compare);
    return result;
  }

  const map_type& map() const { return map_; }

private:
  static bool compare(const value_type& i, const value_type& j) {
    const auto& i_detIds = (i.first)->detIds();
    const auto& j_detIds = (j.first)->detIds();

    if (i_detIds.empty() || j_detIds.empty()) {
      return i_detIds.size() < j_detIds.size();
    }

    auto imin = std::min_element(i_detIds.begin(), i_detIds.end(), [](DetId a, DetId b) { return a < b; });

    auto jmin = std::min_element(j_detIds.begin(), j_detIds.end(), [](DetId a, DetId b) { return a < b; });

    bool result = (*imin < *jmin);
    return result;
  }

  map_type map_;
};

#endif
