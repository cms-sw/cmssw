#ifndef SimDataFormats_Associations_LayerClusterToSimClusterAssociatorBaseImplT_h
#define SimDataFormats_Associations_LayerClusterToSimClusterAssociatorBaseImplT_h

/** \class LayerClusterToSimClusterAssociatorBaseImplT
 *
 * Base class for LayerClusterToSimClusterAssociators.  Methods take as input
 * the handle of LayerClusters and the SimCluster collections and return an
 * AssociationMap (oneToManyWithQuality)
 *
 *  \author Marco Rovere
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

namespace ticl {

  template <typename CLUSTER>
  using SimToRecoCollectionWithSimClustersT =
      edm::AssociationMap<edm::OneToManyWithQualityGeneric<SimClusterCollection, CLUSTER, std::pair<float, float>>>;
  template <typename CLUSTER>
  using RecoToSimCollectionWithSimClustersT =
      edm::AssociationMap<edm::OneToManyWithQualityGeneric<CLUSTER, SimClusterCollection, float>>;

  template <typename CLUSTER>
  class LayerClusterToSimClusterAssociatorBaseImplT {
  public:
    /// Constructor
    LayerClusterToSimClusterAssociatorBaseImplT();
    /// Destructor
    virtual ~LayerClusterToSimClusterAssociatorBaseImplT();

    /// Associate a LayerCluster to SimClusters
    virtual RecoToSimCollectionWithSimClustersT<CLUSTER> associateRecoToSim(
        const edm::Handle<CLUSTER> &cCH, const edm::Handle<SimClusterCollection> &sCCH) const;

    /// Associate a SimCluster to LayerClusters
    virtual SimToRecoCollectionWithSimClustersT<CLUSTER> associateSimToReco(
        const edm::Handle<CLUSTER> &cCH, const edm::Handle<SimClusterCollection> &sCCH) const;
  };
}  // namespace ticl

extern template class ticl::LayerClusterToSimClusterAssociatorBaseImplT<reco::CaloClusterCollection>;
extern template class ticl::LayerClusterToSimClusterAssociatorBaseImplT<reco::PFClusterCollection>;

#endif
