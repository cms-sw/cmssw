// Original Author: Marco Rovere
//

#include "LayerClusterAssociatorByEnergyScoreImpl.h"

LayerClusterAssociatorByEnergyScoreImpl::LayerClusterAssociatorByEnergyScoreImpl() {}

hgcal::RecoToSimCollection LayerClusterAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<reco::CaloClusterCollection> &cCH, const edm::Handle<CaloParticleCollection> &cPCH) const {
  return hgcal::RecoToSimCollection();
}

hgcal::SimToRecoCollection LayerClusterAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<reco::CaloClusterCollection> &cCH, const edm::Handle<CaloParticleCollection> &cPCH) const {
  return hgcal::SimToRecoCollection();
}
