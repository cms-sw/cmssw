#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociatorBaseImpl.h"

namespace reco {
  MtdSimLayerClusterToTPAssociatorBaseImpl::MtdSimLayerClusterToTPAssociatorBaseImpl(){};
  MtdSimLayerClusterToTPAssociatorBaseImpl::~MtdSimLayerClusterToTPAssociatorBaseImpl(){};

  reco::SimToTPCollectionMtd MtdSimLayerClusterToTPAssociatorBaseImpl::associateSimToTP(
      const edm::Handle<MtdSimLayerClusterCollection> &simClusH,
      const edm::Handle<TrackingParticleCollection> &trackingParticleH) const {
    return reco::SimToTPCollectionMtd();
  }

  reco::TPToSimCollectionMtd MtdSimLayerClusterToTPAssociatorBaseImpl::associateTPToSim(
      const edm::Handle<MtdSimLayerClusterCollection> &simClusH,
      const edm::Handle<TrackingParticleCollection> &trackingParticleH) const {
    return reco::TPToSimCollectionMtd();
  }

}  // namespace reco
