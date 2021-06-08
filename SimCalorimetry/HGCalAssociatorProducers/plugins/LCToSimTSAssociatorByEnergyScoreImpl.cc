// Original Author: Marco Rovere
//

#include "LCToSimTSAssociatorByEnergyScoreImpl.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

LCToSimTSAssociatorByEnergyScoreImpl::LCToSimTSAssociatorByEnergyScoreImpl(
    edm::EDProductGetter const& productGetter,
    bool hardScatterOnly)
    : hardScatterOnly_(hardScatterOnly), productGetter_(&productGetter) {
}


hgcal::RecoToSimTracksterCollection LCToSimTSAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<ticl::TracksterCollection> &sTCH,
    const edm::Handle<CaloParticleCollection>& cPCH, const hgcal::RecoToSimCollection &lCToCPs,
    const edm::Handle<SimClusterCollection>& sCCH, const hgcal::RecoToSimCollectionWithSimClusters &lCToSCs) const {
  hgcal::RecoToSimTracksterCollection returnValue(productGetter_);

  const auto simTracksters = *sTCH.product();

  for (size_t lcId = 0; lcId < cCCH.product()->size(); ++lcId) {
    const edm::Ref<reco::CaloClusterCollection> lcRef(cCCH, lcId);
    for (size_t tsId = 0; tsId < simTracksters.size(); ++tsId) {
      if (simTracksters[tsId].seedID() == cPCH.id()) {
        const auto& cpIt = lCToCPs.find(lcRef);
        if (cpIt == lCToCPs.end()) {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
            << "layerCluster Id " << lcId << " not found in CaloParticle association map\n";
          continue;
        }

        const edm::Ref<CaloParticleCollection> cpRef(cPCH, simTracksters[tsId].seedIndex());
        const auto& cps = cpIt->val;
        const auto cpPair = std::find_if(std::begin(cps), std::end(cps),
                                         [&cpRef](const std::pair<edm::Ref<CaloParticleCollection>, float>& p) {
                         return p.first == cpRef;
                       });
        if (cpPair == cps.end()) {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
            << "CaloParticle Id " << simTracksters[tsId].seedIndex() << " not found in LayerCluster association map\n";
          continue;
        }
        else {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
            << "layerCluster Id: \t" << lcId << "\t CP Id: \t" << cpPair->first.index() << "\t score \t" << cpPair->second << "\n";
          // Fill AssociationMap
          returnValue.insert(lcRef,  // Ref to LC
                             std::make_pair(edm::Ref<ticl::TracksterCollection>(sTCH, tsId), // Pair <Ref to TS, score>
                                            cpPair->second));
        }
      } else if (simTracksters[tsId].seedID() == sCCH.id()) {
        const auto& scIt = lCToSCs.find(lcRef);
        if (scIt == lCToSCs.end()) {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
            << "layerCluster Id " << lcId << " not found in SimCluster association map\n";
          continue;
        }

        const edm::Ref<SimClusterCollection> scRef(sCCH, simTracksters[tsId].seedIndex());
        const auto& scs = scIt->val;
        const auto scPair = std::find_if(std::begin(scs), std::end(scs),
                                         [&scRef](const std::pair<edm::Ref<SimClusterCollection>, float>& p) {
                         return p.first == scRef;
                       });
        if (scPair == scs.end()) {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
            << "SimCluster Id " << simTracksters[tsId].seedIndex() << " not found in LayerCluster association map\n";
          continue;
        }
        else {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
            << "layerCluster Id: \t" << lcId << "\t SC Id: \t" << scPair->first.index() << "\t score \t" << scPair->second << "\n";
          // Fill AssociationMap
          returnValue.insert(lcRef,  // Ref to LC
                             std::make_pair(edm::Ref<ticl::TracksterCollection>(sTCH, tsId), // Pair <Ref to TS, score>
                                            scPair->second));
        }
      }
    } // end loop over simTracksters
  } // end loop over layerClusters

  return returnValue;
}

hgcal::SimTracksterToRecoCollection LCToSimTSAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<ticl::TracksterCollection> &sTCH) const {
  hgcal::SimTracksterToRecoCollection returnValue(productGetter_);
/*
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
