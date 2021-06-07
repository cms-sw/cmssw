// Original Author: Marco Rovere
//

#include "LCToSimTSAssociatorByEnergyScoreImpl.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

LCToSimTSAssociatorByEnergyScoreImpl::LCToSimTSAssociatorByEnergyScoreImpl(
    edm::EDProductGetter const& productGetter,
    bool hardScatterOnly,
    std::shared_ptr<hgcal::RecHitTools> recHitTools)
    : hardScatterOnly_(hardScatterOnly), recHitTools_(recHitTools), productGetter_(&productGetter) {
  layers_ = recHitTools_->lastLayerBH();
}


hgcal::RecoToSimCollection LCToSimTSAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<ticl::TracksterCollection> &sTCH) const {
  hgcal::RecoToSimCollection returnValue(productGetter_);
/*
  for (size_t lcId = 0; lcId < cCCH.size(); ++lcId) {
    for (const auto tst : sTCH) {
      if (tst.seedID() == )




    for (auto& cpPair : cpsInLayerCluster[lcId]) {
      LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
          << "layerCluster Id: \t" << lcId << "\t CP id: \t" << cpPair.first << "\t score \t" << cpPair.second << "\n";
      // Fill AssociationMap
      returnValue.insert(edm::Ref<reco::CaloClusterCollection>(cCCH, lcId),  // Ref to LC
                         std::make_pair(edm::Ref<CaloParticleCollection>(cPCH, cpPair.first),
                                        cpPair.second)  // Pair <Ref to CP, score>
      );
    }
  }
*/
  return returnValue;
}

hgcal::SimToRecoCollection LCToSimTSAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<CaloParticleCollection>& cPCH) const {
  hgcal::SimToRecoCollection returnValue(productGetter_);
/*
  const auto& links = makeConnections(cCCH, cPCH);
  const auto& cPOnLayer = std::get<1>(links);
  for (size_t cpId = 0; cpId < cPOnLayer.size(); ++cpId) {
    for (size_t layerId = 0; layerId < cPOnLayer[cpId].size(); ++layerId) {
      for (auto& lcPair : cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore) {
        returnValue.insert(
            edm::Ref<CaloParticleCollection>(cPCH, cpId),                              // Ref to CP
            std::make_pair(edm::Ref<reco::CaloClusterCollection>(cCCH, lcPair.first),  // Pair <Ref to LC,
                           std::make_pair(lcPair.second.first, lcPair.second.second))  // pair <energy, score> >
        );
      }
    }
  }
*/
  return returnValue;
}
