#include <vector>
#include <memory>

#include "SimDataFormats/Associations/interface/MtdSimMergedClusterToTPAssociator.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociator.h"

namespace edm {
  class EDProductGetter;
}

class MtdSimMergedClusterToTPAssociatorByTrackIdImpl : public reco::MtdSimMergedClusterToTPAssociatorBaseImpl {
public:
  explicit MtdSimMergedClusterToTPAssociatorByTrackIdImpl(edm::EDProductGetter const &,
                                                          reco::SimToTPCollectionMtd,
                                                          reco::TPToSimCollectionMtd);

  reco::MergedSimToTPCollectionMtd associateSimToTP(
      const edm::Handle<MtdSimMergedClusterCollection> &simClusH,
      const edm::Handle<TrackingParticleCollection> &trackingParticleH) const override;

  reco::TPToMergedSimCollectionMtd associateTPToSim(
      const edm::Handle<MtdSimMergedClusterCollection> &simClusH,
      const edm::Handle<TrackingParticleCollection> &trackingParticleH) const override;

private:
  edm::EDProductGetter const *productGetter_;

  reco::SimToTPCollectionMtd simToTPMap_;
  reco::TPToSimCollectionMtd tpToSimMap_;
};
