// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociatorBaseImpl.h"

namespace hgcal {
  LayerClusterToSimTracksterAssociatorBaseImpl::LayerClusterToSimTracksterAssociatorBaseImpl(){};
  LayerClusterToSimTracksterAssociatorBaseImpl::~LayerClusterToSimTracksterAssociatorBaseImpl(){};

  hgcal::RecoToSimCollection LayerClusterToSimTracksterAssociatorBaseImpl::associateRecoToSim(
      const edm::Handle<reco::CaloClusterCollection> &cCCH, const edm::Handle<ticl::TracksterCollection> &cPCH) const {
    return hgcal::RecoToSimCollection();
  }

  hgcal::SimToRecoCollection LayerClusterToSimTracksterAssociatorBaseImpl::associateSimToReco(
      const edm::Handle<reco::CaloClusterCollection> &cCCH, const edm::Handle<CaloParticleCollection> &cPCH) const {
    return hgcal::SimToRecoCollection();
  }

}  // namespace hgcal
