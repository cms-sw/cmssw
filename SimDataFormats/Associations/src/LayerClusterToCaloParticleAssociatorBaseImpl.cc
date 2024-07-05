// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociatorBaseImpl.h"

namespace ticl {
  LayerClusterToCaloParticleAssociatorBaseImpl::LayerClusterToCaloParticleAssociatorBaseImpl(){};
  LayerClusterToCaloParticleAssociatorBaseImpl::~LayerClusterToCaloParticleAssociatorBaseImpl(){};

  ticl::RecoToSimCollection LayerClusterToCaloParticleAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<reco::CaloClusterCollection> &cCCH, const edm::Handle<CaloParticleCollection> &cPCH) const {
    return ticl::RecoToSimCollection();
  }

  ticl::SimToRecoCollection LayerClusterToCaloParticleAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<reco::CaloClusterCollection> &cCCH, const edm::Handle<CaloParticleCollection> &cPCH) const {
    return ticl::SimToRecoCollection();
  }

}  // namespace ticl
