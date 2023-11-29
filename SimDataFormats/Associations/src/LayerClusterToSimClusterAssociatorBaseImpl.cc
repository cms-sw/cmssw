// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociatorBaseImpl.h"

namespace ticl {
  LayerClusterToSimClusterAssociatorBaseImpl::LayerClusterToSimClusterAssociatorBaseImpl(){};
  LayerClusterToSimClusterAssociatorBaseImpl::~LayerClusterToSimClusterAssociatorBaseImpl(){};

  ticl::RecoToSimCollectionWithSimClusters LayerClusterToSimClusterAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<reco::CaloClusterCollection> &cCCH, const edm::Handle<SimClusterCollection> &sCCH) const {
    return ticl::RecoToSimCollectionWithSimClusters();
  }

  ticl::SimToRecoCollectionWithSimClusters LayerClusterToSimClusterAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<reco::CaloClusterCollection> &cCCH, const edm::Handle<SimClusterCollection> &sCCH) const {
    return ticl::SimToRecoCollectionWithSimClusters();
  }

}  // namespace ticl
