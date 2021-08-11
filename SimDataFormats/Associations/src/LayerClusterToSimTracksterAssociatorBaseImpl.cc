// Original Author: Leonardo Cristella

#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociatorBaseImpl.h"

namespace hgcal {
  LayerClusterToSimTracksterAssociatorBaseImpl::LayerClusterToSimTracksterAssociatorBaseImpl(){};
  LayerClusterToSimTracksterAssociatorBaseImpl::~LayerClusterToSimTracksterAssociatorBaseImpl(){};

  hgcal::RecoToSimTracksterCollection LayerClusterToSimTracksterAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<reco::CaloClusterCollection> &cCCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const hgcal::RecoToSimCollection &lCToCPs,
      const edm::Handle<SimClusterCollection> &sCCH,
      const hgcal::RecoToSimCollectionWithSimClusters &lCToSCs) const {
    return hgcal::RecoToSimTracksterCollection();
  }

  hgcal::SimTracksterToRecoCollection LayerClusterToSimTracksterAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<reco::CaloClusterCollection> &cCCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const hgcal::SimToRecoCollection &cPToLCs,
      const edm::Handle<SimClusterCollection> &sCCH,
      const hgcal::SimToRecoCollectionWithSimClusters &sCToLCs) const {
    return hgcal::SimTracksterToRecoCollection();
  }

}  // namespace hgcal
