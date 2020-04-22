// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociatorBaseImpl.h"

namespace hgcal {
  LayerClusterToCaloParticleAssociatorBaseImpl::LayerClusterToCaloParticleAssociatorBaseImpl(){};
  LayerClusterToCaloParticleAssociatorBaseImpl::~LayerClusterToCaloParticleAssociatorBaseImpl(){};

  hgcal::RecoToSimCollection LayerClusterToCaloParticleAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<reco::CaloClusterCollection> &cCCH, const edm::Handle<CaloParticleCollection> &cPCH) const {
    return hgcal::RecoToSimCollection();
  }

  hgcal::SimToRecoCollection LayerClusterToCaloParticleAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<reco::CaloClusterCollection> &cCCH, const edm::Handle<CaloParticleCollection> &cPCH) const {
    return hgcal::SimToRecoCollection();
  }

}  // namespace hgcal
