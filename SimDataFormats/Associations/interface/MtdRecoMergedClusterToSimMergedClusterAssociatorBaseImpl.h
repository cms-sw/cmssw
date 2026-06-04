#ifndef SimDataFormats_Associations_MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl_h
#define SimDataFormats_Associations_MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FTLRecHit/interface/FTLMergedClusterCollections.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedClusterFwd.h"
#include "SimDataFormats/Associations/interface/MtdRecoMergedClusterToSimMergedClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/MtdSimMergedClusterToRecoMergedClusterAssociationMap.h"

namespace reco {

  using MergedRecoToSimCollectionMtd = MtdRecoMergedClusterToSimMergedClusterAssociationMap;
  using MergedSimToRecoCollectionMtd = MtdSimMergedClusterToRecoMergedClusterAssociationMap;

  class MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl {
  public:
    /// Constructor
    MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl();
    /// Destructor
    virtual ~MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl();

    /// Associate MtdRecoMergedCluster to MtdSimMergedClusters
    virtual reco::MergedRecoToSimCollectionMtd associateRecoToSim(
        const edm::Handle<FTLMergedClusterCollection> &btlRecoClusH,
        const edm::Handle<FTLMergedClusterCollection> &etlRecoClusH,
        const edm::Handle<MtdSimMergedClusterCollection> &simClusH) const;

    /// Associate MtdSimMergedClusters to MtdRecoMergedClusters
    virtual reco::MergedSimToRecoCollectionMtd associateSimToReco(
        const edm::Handle<FTLMergedClusterCollection> &btlRecoClusH,
        const edm::Handle<FTLMergedClusterCollection> &etlRecoClusH,
        const edm::Handle<MtdSimMergedClusterCollection> &simClusH) const;
  };
}  // namespace reco

#endif
