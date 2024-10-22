#ifndef SimDataFormats_Associations_MtdRecoClusterToSimLayerClusterAssociationMap_h
#define SimDataFormats_Associations_MtdRecoClusterToSimLayerClusterAssociationMap_h

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/HandleBase.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"

#include <vector>
#include <utility>
#include <algorithm>

/**
 * Maps FTLCluserRef to SimLayerClusterRef
 *
 */
class MtdRecoClusterToSimLayerClusterAssociationMap {
public:
  using key_type = FTLClusterRef;
  using mapped_type = MtdSimLayerClusterRef;
  using value_type = std::pair<key_type, std::vector<mapped_type>>;
  using map_type = std::vector<value_type>;
  using const_iterator = typename map_type::const_iterator;
  using range = std::pair<const_iterator, const_iterator>;

  /// Constructor
  MtdRecoClusterToSimLayerClusterAssociationMap();
  /// Destructor
  ~MtdRecoClusterToSimLayerClusterAssociationMap();

  void emplace_back(const FTLClusterRef& recoClus, std::vector<MtdSimLayerClusterRef>& simClusVect) {
    map_.emplace_back(recoClus, simClusVect);
  }

  void post_insert() { std::sort(map_.begin(), map_.end(), compare); }

  bool empty() const { return map_.empty(); }
  size_t size() const { return map_.size(); }

  const_iterator begin() const { return map_.begin(); }
  const_iterator cbegin() const { return map_.cbegin(); }
  const_iterator end() const { return map_.end(); }
  const_iterator cend() const { return map_.cend(); }

  range equal_range(const FTLClusterRef& key) const {
    return std::equal_range(map_.begin(), map_.end(), value_type(key, std::vector<MtdSimLayerClusterRef>()), compare);
  }

  const map_type& map() const { return map_; }

private:
  static bool compare(const value_type& i, const value_type& j) { return (i.first < j.first); }

  map_type map_;
};

#endif
