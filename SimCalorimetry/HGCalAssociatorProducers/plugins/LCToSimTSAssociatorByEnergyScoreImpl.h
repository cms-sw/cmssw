// Original Author: Leonardo Cristella
#ifndef SimCalorimetry_HGCalAssociatorProducers_LCToSimTSAssociatorByEnergyScoreImpl_h
#define SimCalorimetry_HGCalAssociatorProducers_LCToSimTSAssociatorByEnergyScoreImpl_h

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

class LCToSimTSAssociatorByEnergyScoreImpl : public ticl::LayerClusterToSimTracksterAssociatorBaseImpl {
public:
  explicit LCToSimTSAssociatorByEnergyScoreImpl(edm::EDProductGetter const &);

  ticl::RecoToSimTracksterCollection associateRecoToSim(
      const edm::Handle<reco::CaloClusterCollection> &cCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const ticl::RecoToSimCollection &lCToCPs,
      const edm::Handle<SimClusterCollection> &sCCH,
      const ticl::RecoToSimCollectionWithSimClusters &lCToSCs) const override;

  ticl::SimTracksterToRecoCollection associateSimToReco(
      const edm::Handle<reco::CaloClusterCollection> &cCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const ticl::SimToRecoCollection &cPToLCs,
      const edm::Handle<SimClusterCollection> &sCCH,
      const ticl::SimToRecoCollectionWithSimClusters &sCToLCs) const override;

private:
  edm::EDProductGetter const *productGetter_;
};

#endif
