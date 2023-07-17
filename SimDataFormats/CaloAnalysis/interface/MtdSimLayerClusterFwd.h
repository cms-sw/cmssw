#ifndef CaloAnalysis_MtdSimLayerClusterFwd_h
#define CaloAnalysis_MtdSimLayerClusterFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

class MtdSimLayerCluster;
typedef std::vector<MtdSimLayerCluster> MtdSimLayerClusterCollection;
typedef edm::Ref<MtdSimLayerClusterCollection> MtdSimLayerClusterRef;
typedef edm::RefVector<MtdSimLayerClusterCollection> MtdSimLayerClusterRefVector;
typedef edm::RefProd<MtdSimLayerClusterCollection> MtdSimLayerClusterRefProd;
typedef edm::RefVector<MtdSimLayerClusterCollection> MtdSimLayerClusterContainer;

std::ostream &operator<<(std::ostream &s, MtdSimLayerCluster const &tp);

#endif
