// Original Author: Marco Rovere

#include <vector>
#include <map>
#include <unordered_map>
#include <memory>  // shared_ptr

#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace edm {
  class EDProductGetter;
}

class LCToSimTSAssociatorByEnergyScoreImpl : public hgcal::LayerClusterToSimTracksterAssociatorBaseImpl {
public:
  explicit LCToSimTSAssociatorByEnergyScoreImpl(edm::EDProductGetter const &,
                                             bool,
                                             std::shared_ptr<hgcal::RecHitTools>);

  hgcal::RecoToSimCollection associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                const edm::Handle<ticl::TracksterCollection> &sTCH) const override;

  hgcal::SimToRecoCollection associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                const edm::Handle<CaloParticleCollection> &cPCH) const override;

private:
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> recHitTools_;
  unsigned layers_;
  edm::EDProductGetter const *productGetter_;
};
