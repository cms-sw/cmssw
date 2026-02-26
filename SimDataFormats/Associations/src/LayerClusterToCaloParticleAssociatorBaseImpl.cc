// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociatorBaseImpl.h"

namespace ticl {
  template <typename CLUSTER>
  LayerClusterToCaloParticleAssociatorBaseImplT<CLUSTER>::LayerClusterToCaloParticleAssociatorBaseImplT() {}
  template <typename CLUSTER>
  LayerClusterToCaloParticleAssociatorBaseImplT<CLUSTER>::~LayerClusterToCaloParticleAssociatorBaseImplT() {}

  template <typename CLUSTER>
  ticl::RecoToSimCollectionT<CLUSTER> LayerClusterToCaloParticleAssociatorBaseImplT<CLUSTER>::associateRecoToSim(
      const edm::Handle<CLUSTER> &cCCH, const edm::Handle<CaloParticleCollection> &cPCH) const {
    return ticl::RecoToSimCollectionT<CLUSTER>();
  }

  template <typename CLUSTER>
  ticl::SimToRecoCollectionT<CLUSTER> LayerClusterToCaloParticleAssociatorBaseImplT<CLUSTER>::associateSimToReco(
      const edm::Handle<CLUSTER> &cCCH, const edm::Handle<CaloParticleCollection> &cPCH) const {
    return ticl::SimToRecoCollectionT<CLUSTER>();
  }
}  // namespace ticl

template class ticl::LayerClusterToCaloParticleAssociatorBaseImplT<reco::CaloClusterCollection>;
template class ticl::LayerClusterToCaloParticleAssociatorBaseImplT<reco::PFClusterCollection>;
