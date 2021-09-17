// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociatorBaseImpl.h"

namespace hgcal {
  LayerClusterToSimClusterAssociatorBaseImpl::LayerClusterToSimClusterAssociatorBaseImpl(){};
  LayerClusterToSimClusterAssociatorBaseImpl::~LayerClusterToSimClusterAssociatorBaseImpl(){};

  hgcal::RecoToSimCollectionWithSimClusters LayerClusterToSimClusterAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<reco::CaloClusterCollection> &cCCH, const edm::Handle<SimClusterCollection> &sCCH) const {
    return hgcal::RecoToSimCollectionWithSimClusters();
  }

  hgcal::SimToRecoCollectionWithSimClusters LayerClusterToSimClusterAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<reco::CaloClusterCollection> &cCCH, const edm::Handle<SimClusterCollection> &sCCH) const {
    return hgcal::SimToRecoCollectionWithSimClusters();
  }

}  // namespace hgcal
