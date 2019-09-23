// Original Author: Marco Rovere

#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"

class LayerClusterAssociatorByEnergyScoreImpl : public hgcal::LayerClusterToCaloParticleAssociatorBaseImpl {
public:
  explicit LayerClusterAssociatorByEnergyScoreImpl();

  hgcal::RecoToSimCollection associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                const edm::Handle<CaloParticleCollection> &cPCH) const override;

  hgcal::SimToRecoCollection associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                const edm::Handle<CaloParticleCollection> &cPCH) const override;
};
