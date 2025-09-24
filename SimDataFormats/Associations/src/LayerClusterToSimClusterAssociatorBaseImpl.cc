// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociatorBaseImpl.h"

namespace ticl {
  template <typename CLUSTER>
  LayerClusterToSimClusterAssociatorBaseImplT<CLUSTER>::LayerClusterToSimClusterAssociatorBaseImplT() {}
  template <typename CLUSTER>
  LayerClusterToSimClusterAssociatorBaseImplT<CLUSTER>::~LayerClusterToSimClusterAssociatorBaseImplT() {}

  template <typename CLUSTER>
  RecoToSimCollectionWithSimClustersT<CLUSTER> LayerClusterToSimClusterAssociatorBaseImplT<CLUSTER>::associateRecoToSim(
      const edm::Handle<CLUSTER> &cCCH, const edm::Handle<SimClusterCollection> &sCCH) const {
    return RecoToSimCollectionWithSimClustersT<CLUSTER>();
  }

  template <typename CLUSTER>
  SimToRecoCollectionWithSimClustersT<CLUSTER> LayerClusterToSimClusterAssociatorBaseImplT<CLUSTER>::associateSimToReco(
      const edm::Handle<CLUSTER> &cCCH, const edm::Handle<SimClusterCollection> &sCCH) const {
    return SimToRecoCollectionWithSimClustersT<CLUSTER>();
  }

  template class ticl::LayerClusterToSimClusterAssociatorBaseImplT<reco::CaloClusterCollection>;
  template class ticl::LayerClusterToSimClusterAssociatorBaseImplT<reco::PFClusterCollection>;

}  // namespace ticl
