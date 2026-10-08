#ifndef CaloAnalysis_MtdSimMergedClusterFwd_h
#define CaloAnalysis_MtdSimMergedClusterFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedCluster.h"

typedef std::vector<MtdSimMergedCluster> MtdSimMergedClusterCollection;
typedef edm::Ref<MtdSimMergedClusterCollection> MtdSimMergedClusterRef;
typedef edm::RefVector<MtdSimMergedClusterCollection> MtdSimMergedClusterRefVector;
typedef edm::RefProd<MtdSimMergedClusterCollection> MtdSimMergedClusterRefProd;
typedef edm::RefVector<MtdSimMergedClusterCollection> MtdSimMergedClusterContainer;

std::ostream &operator<<(std::ostream &s, MtdSimMergedCluster const &tp);

#endif
