#include <vector>
#include <memory>

#include "Geometry/MTDGeometryBuilder/interface/MTDGeomUtil.h"
#include "SimDataFormats/Associations/interface/MtdRecoMergedClusterToSimMergedClusterAssociator.h"

#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociator.h"

namespace edm {
  class EDProductGetter;
}

class MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl
    : public reco::MtdRecoMergedClusterToSimMergedClusterAssociatorBaseImpl {
public:
  explicit MtdRecoMergedClusterToSimMergedClusterAssociatorByHitsImpl(edm::EDProductGetter const &,
                                                                      mtd::MTDGeomUtil &,
                                                                      reco::SimToRecoCollectionMtd,
                                                                      reco::RecoToSimCollectionMtd);

  reco::MergedRecoToSimCollectionMtd associateRecoToSim(
      const edm::Handle<FTLMergedClusterCollection> &btlRecoClusH,
      const edm::Handle<FTLMergedClusterCollection> &etlRecoClusH,
      const edm::Handle<MtdSimMergedClusterCollection> &simClusH) const override;

  reco::MergedSimToRecoCollectionMtd associateSimToReco(
      const edm::Handle<FTLMergedClusterCollection> &btlRecoClusH,
      const edm::Handle<FTLMergedClusterCollection> &etlRecoClusH,
      const edm::Handle<MtdSimMergedClusterCollection> &simClusH) const override;

private:
  edm::EDProductGetter const *productGetter_;
  mtd::MTDGeomUtil geomTools_;

  reco::SimToRecoCollectionMtd simToRecoMap_;
  reco::RecoToSimCollectionMtd recoToSimMap_;
};
