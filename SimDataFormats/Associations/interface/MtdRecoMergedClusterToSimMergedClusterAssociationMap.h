#ifndef SimDataFormats_Associations_MtdRecoMergedClusterToSimMergedClusterAssociationMap_h
#define SimDataFormats_Associations_MtdRecoMergedClusterToSimMergedClusterAssociationMap_h

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/HandleBase.h"
#include "DataFormats/FTLRecHit/interface/FTLMergedClusterCollections.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedClusterFwd.h"

#include <vector>
#include <utility>
#include <algorithm>

/**
 * Maps FTLMergedClusterRef to MtdSimMergedClusterRef
 *
 */
class MtdRecoMergedClusterToSimMergedClusterAssociationMap {
public:
  using key_type = FTLMergedClusterRef;
  using mapped_type = MtdSimMergedClusterRef;
  using value_type = std::pair<key_type, std::vector<mapped_type>>;
  using map_type = std::vector<value_type>;
  using const_iterator = typename map_type::const_iterator;
  using range = std::pair<const_iterator, const_iterator>;

  /// Constructor
  MtdRecoMergedClusterToSimMergedClusterAssociationMap();
  /// Destructor
  ~MtdRecoMergedClusterToSimMergedClusterAssociationMap();

  void emplace_back(const FTLMergedClusterRef& recoClus, std::vector<MtdSimMergedClusterRef>& simClusVect) {
    map_.emplace_back(recoClus, simClusVect);
  }

  void post_insert() { std::sort(map_.begin(), map_.end(), compare); }

  bool empty() const { return map_.empty(); }
  size_t size() const { return map_.size(); }

  const_iterator begin() const { return map_.begin(); }
  const_iterator cbegin() const { return map_.cbegin(); }
  const_iterator end() const { return map_.end(); }
  const_iterator cend() const { return map_.cend(); }

  range equal_range(const FTLMergedClusterRef& key) const {
    return std::equal_range(map_.begin(), map_.end(), value_type(key, std::vector<MtdSimMergedClusterRef>()), compare);
  }

  const map_type& map() const { return map_; }

private:
  static bool compare(const value_type& i, const value_type& j) { return (i.first < j.first); }

  map_type map_;
};

#endif
