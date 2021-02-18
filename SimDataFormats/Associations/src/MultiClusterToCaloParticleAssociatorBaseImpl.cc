// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/MultiClusterToCaloParticleAssociatorBaseImpl.h"

namespace hgcal {
  MultiClusterToCaloParticleAssociatorBaseImpl::MultiClusterToCaloParticleAssociatorBaseImpl(){};
  MultiClusterToCaloParticleAssociatorBaseImpl::~MultiClusterToCaloParticleAssociatorBaseImpl(){};

  hgcal::RecoToSimCollectionWithMultiClusters MultiClusterToCaloParticleAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<reco::HGCalMultiClusterCollection> &cCCH,
      const edm::Handle<CaloParticleCollection> &cPCH) const {
    return hgcal::RecoToSimCollectionWithMultiClusters();
  }

  hgcal::SimToRecoCollectionWithMultiClusters MultiClusterToCaloParticleAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<reco::HGCalMultiClusterCollection> &cCCH,
      const edm::Handle<CaloParticleCollection> &cPCH) const {
    return hgcal::SimToRecoCollectionWithMultiClusters();
  }

}  // namespace hgcal
