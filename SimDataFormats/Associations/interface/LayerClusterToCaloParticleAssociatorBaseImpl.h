#ifndef SimDataFormats_Associations_LayerClusterToCaloParticleAssociatorBaseImpl_h
#define SimDataFormats_Associations_LayerClusterToCaloParticleAssociatorBaseImpl_h

/** \class LayerClusterToCaloParticleAssociatorBaseImpl
 *
 * Base class for LayerClusterToCaloParticleAssociators.  Methods take as input
 * the handle of LayerClusters and the CaloParticle collections and return an
 * AssociationMap (oneToManyWithQuality)
 *
 *  \author Marco Rovere
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterCollection.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

namespace ticl {

  template <typename CLUSTER>
  using SimToRecoCollectionT =
      edm::AssociationMap<edm::OneToManyWithQualityGeneric<CaloParticleCollection, CLUSTER, std::pair<float, float>>>;
  template <typename CLUSTER>
  using RecoToSimCollectionT =
      edm::AssociationMap<edm::OneToManyWithQualityGeneric<CLUSTER, CaloParticleCollection, float>>;

  template <typename CLUSTER>
  class LayerClusterToCaloParticleAssociatorBaseImplT {
  public:
    /// Constructor
    LayerClusterToCaloParticleAssociatorBaseImplT();
    /// Destructor
    virtual ~LayerClusterToCaloParticleAssociatorBaseImplT();

    /// Associate a LayerCluster to CaloParticles
    virtual ticl::RecoToSimCollectionT<CLUSTER> associateRecoToSim(
        const edm::Handle<CLUSTER> &cCH, const edm::Handle<CaloParticleCollection> &cPCH) const;

    /// Associate a CaloParticle to LayerClusters
    virtual ticl::SimToRecoCollectionT<CLUSTER> associateSimToReco(
        const edm::Handle<CLUSTER> &cCH, const edm::Handle<CaloParticleCollection> &cPCH) const;
  };
}  // namespace ticl

extern template class ticl::LayerClusterToCaloParticleAssociatorBaseImplT<reco::CaloClusterCollection>;
extern template class ticl::LayerClusterToCaloParticleAssociatorBaseImplT<reco::PFClusterCollection>;

#endif
