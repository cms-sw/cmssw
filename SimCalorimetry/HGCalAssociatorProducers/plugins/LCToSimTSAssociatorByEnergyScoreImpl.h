// Original Author: Leonardo Cristella

#include <vector>
#include <map>
#include <unordered_map>
#include <memory>  // shared_ptr

#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"

namespace edm {
  class EDProductGetter;
}

class LCToSimTSAssociatorByEnergyScoreImpl : public hgcal::LayerClusterToSimTracksterAssociatorBaseImpl {
public:
  explicit LCToSimTSAssociatorByEnergyScoreImpl(edm::EDProductGetter const &,
                                             bool);

  hgcal::RecoToSimTracksterCollection associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                const edm::Handle<ticl::TracksterCollection> &sTCH,
                                                const edm::Handle<CaloParticleCollection>& cPCH,
                                                const hgcal::RecoToSimCollection &lCToCPs,
                                                const edm::Handle<SimClusterCollection>& sCCH,
                                                const hgcal::RecoToSimCollectionWithSimClusters &lCToSCs) const override;

  hgcal::SimTracksterToRecoCollection associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                const edm::Handle<ticl::TracksterCollection> &sTCH,
                                                const edm::Handle<CaloParticleCollection>& cPCH,
                                                const hgcal::SimToRecoCollection &cPToLCs,
                                                const edm::Handle<SimClusterCollection>& sCCH,
                                                const hgcal::SimToRecoCollectionWithSimClusters &sCToLCs) const override;

private:
  const bool hardScatterOnly_;
  edm::EDProductGetter const *productGetter_;
};
