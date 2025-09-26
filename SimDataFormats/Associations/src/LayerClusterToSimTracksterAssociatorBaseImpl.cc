// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociatorBaseImpl.h"

namespace ticl {
  LayerClusterToSimTracksterAssociatorBaseImpl::LayerClusterToSimTracksterAssociatorBaseImpl() {}
  LayerClusterToSimTracksterAssociatorBaseImpl::~LayerClusterToSimTracksterAssociatorBaseImpl() {}

  ticl::RecoToSimTracksterCollection LayerClusterToSimTracksterAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<reco::CaloClusterCollection> &cCCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const ticl::RecoToSimCollectionT<reco::CaloClusterCollection> &lCToCPs,
      const edm::Handle<SimClusterCollection> &sCCH,
      const ticl::RecoToSimCollectionWithSimClustersT<reco::CaloClusterCollection> &lCToSCs) const {
    return ticl::RecoToSimTracksterCollection();
  }

  ticl::SimTracksterToRecoCollection LayerClusterToSimTracksterAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<reco::CaloClusterCollection> &cCCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const ticl::SimToRecoCollectionT<reco::CaloClusterCollection> &cPToLCs,
      const edm::Handle<SimClusterCollection> &sCCH,
      const ticl::SimToRecoCollectionWithSimClustersT<reco::CaloClusterCollection> &sCToLCs) const {
    return ticl::SimTracksterToRecoCollection();
  }

}  // namespace ticl
