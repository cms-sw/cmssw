#include "SimDataFormats/Associations/interface/MtdSimMergedClusterToTPAssociatorBaseImpl.h"

namespace reco {
  MtdSimMergedClusterToTPAssociatorBaseImpl::MtdSimMergedClusterToTPAssociatorBaseImpl() {}
  MtdSimMergedClusterToTPAssociatorBaseImpl::~MtdSimMergedClusterToTPAssociatorBaseImpl() {}

  reco::MergedSimToTPCollectionMtd MtdSimMergedClusterToTPAssociatorBaseImpl::associateSimToTP(
      const edm::Handle<MtdSimMergedClusterCollection> &simClusH,
      const edm::Handle<TrackingParticleCollection> &trackingParticleH) const {
    return reco::MergedSimToTPCollectionMtd();
  }

  reco::TPToMergedSimCollectionMtd MtdSimMergedClusterToTPAssociatorBaseImpl::associateTPToSim(
      const edm::Handle<MtdSimMergedClusterCollection> &simClusH,
      const edm::Handle<TrackingParticleCollection> &trackingParticleH) const {
    return reco::TPToMergedSimCollectionMtd();
  }

}  // namespace reco
