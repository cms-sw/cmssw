#ifndef CaloAnalysis_SimClusterFwd_h
#define CaloAnalysis_SimClusterFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

class SimCluster;
std::ostream &operator<<(std::ostream &s, SimCluster const &tp);

typedef std::vector<SimCluster> SimClusterCollection;
typedef edm::Ref<SimClusterCollection> SimClusterRef;
typedef edm::RefVector<SimClusterCollection> SimClusterRefVector;
typedef edm::RefProd<SimClusterCollection> SimClusterRefProd;
typedef edm::RefVector<SimClusterCollection> SimClusterContainer;

std::ostream &operator<<(std::ostream &s, SimCluster const &tp);

namespace simcluster_utils {
  extern const std::unordered_map<std::string, std::vector<DetId::Detector>> DetIdMap;
  std::vector<DetId::Detector> check_and_join_detids(const std::vector<std::string> &dets_v);
}  // namespace simcluster_utils

#endif
