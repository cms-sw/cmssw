#ifndef SimDataFormats_Associations_MtdRecoClusterToSimLayerClusterAssociatorBaseImpl_h
#define SimDataFormats_Associations_MtdRecoClusterToSimLayerClusterAssociatorBaseImpl_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"
#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToRecoClusterAssociationMap.h"

namespace reco {

  using RecoToSimCollectionMtd = MtdRecoClusterToSimLayerClusterAssociationMap;
  using SimToRecoCollectionMtd = MtdSimLayerClusterToRecoClusterAssociationMap;

  class MtdRecoClusterToSimLayerClusterAssociatorBaseImpl {
  public:
    /// Constructor
    MtdRecoClusterToSimLayerClusterAssociatorBaseImpl();
    /// Destructor
    virtual ~MtdRecoClusterToSimLayerClusterAssociatorBaseImpl();

    /// Associate a MtdRecoCluster to MtdSimLayerClusters
    virtual reco::RecoToSimCollectionMtd associateRecoToSim(
        const edm::Handle<FTLClusterCollection> &btlRecoClusH,
        const edm::Handle<FTLClusterCollection> &etlRecoClusH,
        const edm::Handle<MtdSimLayerClusterCollection> &simClusH) const;

    /// Associate a MtdSimLayerClusters to MtdRecoClusters
    virtual reco::SimToRecoCollectionMtd associateSimToReco(
        const edm::Handle<FTLClusterCollection> &btlRecoClusH,
        const edm::Handle<FTLClusterCollection> &etlRecoClusH,
        const edm::Handle<MtdSimLayerClusterCollection> &simClusH) const;
  };
}  // namespace reco

#endif
