// Original Author: Leonardo Cristella
//

#include "LCToSimTSAssociatorByEnergyScoreImpl.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

LCToSimTSAssociatorByEnergyScoreImpl::LCToSimTSAssociatorByEnergyScoreImpl(edm::EDProductGetter const& productGetter)
    : productGetter_(&productGetter) {}

hgcal::RecoToSimTracksterCollection LCToSimTSAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<reco::CaloClusterCollection>& cCCH,
    const edm::Handle<ticl::TracksterCollection>& sTCH,
    const edm::Handle<CaloParticleCollection>& cPCH,
    const hgcal::RecoToSimCollection& lCToCPs,
    const edm::Handle<SimClusterCollection>& sCCH,
    const hgcal::RecoToSimCollectionWithSimClusters& lCToSCs) const {
  hgcal::RecoToSimTracksterCollection returnValue(productGetter_);

  const auto simTracksters = *sTCH.product();

  for (size_t lcId = 0; lcId < cCCH.product()->size(); ++lcId) {
    const edm::Ref<reco::CaloClusterCollection> lcRef(cCCH, lcId);
    for (size_t tsId = 0; tsId < simTracksters.size(); ++tsId) {
      if (simTracksters[tsId].seedID() == cPCH.id()) {
        const auto& cpIt = lCToCPs.find(lcRef);
        if (cpIt == lCToCPs.end()) {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
              << "LayerCluster Id " << lcId << " not found in CaloParticle association map\n";
          continue;
        }

        const edm::Ref<CaloParticleCollection> cpRef(cPCH, simTracksters[tsId].seedIndex());
        const auto& cps = cpIt->val;
        const auto cpPair = std::find_if(
            std::begin(cps), std::end(cps), [&cpRef](const std::pair<edm::Ref<CaloParticleCollection>, float>& p) {
              return p.first == cpRef;
            });
        if (cpPair == cps.end()) {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl") << "CaloParticle Id " << simTracksters[tsId].seedIndex()
                                                           << " not found in LayerCluster association map\n";
          continue;
        } else {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
              << "LayerCluster Id: \t" << lcId << "\t CaloParticle Id: \t" << cpPair->first.index() << "\t score \t"
              << cpPair->second << "\n";
          // Fill AssociationMap
          returnValue.insert(lcRef,                                                           // Ref to LC
                             std::make_pair(edm::Ref<ticl::TracksterCollection>(sTCH, tsId),  // Pair <Ref to TS, score>
                                            cpPair->second));
        }
      } else if (simTracksters[tsId].seedID() == sCCH.id()) {
        const auto& scIt = lCToSCs.find(lcRef);
        if (scIt == lCToSCs.end()) {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
              << "LayerCluster Id " << lcId << " not found in SimCluster association map\n";
          continue;
        }

        const edm::Ref<SimClusterCollection> scRef(sCCH, simTracksters[tsId].seedIndex());
        const auto& scs = scIt->val;
        const auto scPair = std::find_if(
            std::begin(scs), std::end(scs), [&scRef](const std::pair<edm::Ref<SimClusterCollection>, float>& p) {
              return p.first == scRef;
            });
        if (scPair == scs.end()) {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
              << "SimCluster Id " << simTracksters[tsId].seedIndex() << " not found in LayerCluster association map\n";
          continue;
        } else {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
              << "LayerCluster Id: \t" << lcId << "\t SimCluster Id: \t" << scPair->first.index() << "\t score \t"
              << scPair->second << "\n";
          // Fill AssociationMap
          returnValue.insert(lcRef,                                                           // Ref to LC
                             std::make_pair(edm::Ref<ticl::TracksterCollection>(sTCH, tsId),  // Pair <Ref to TS, score>
                                            scPair->second));
        }
      } else {
        LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
            << "The seedID " << simTracksters[tsId].seedID() << " of SimTrackster " << tsId
            << " is neither a CaloParticle nor a SimCluster!\n";
      }
    }  // end loop over simTracksters
  }    // end loop over layerClusters

  return returnValue;
}

hgcal::SimTracksterToRecoCollection LCToSimTSAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<reco::CaloClusterCollection>& cCCH,
    const edm::Handle<ticl::TracksterCollection>& sTCH,
    const edm::Handle<CaloParticleCollection>& cPCH,
    const hgcal::SimToRecoCollection& cPToLCs,
    const edm::Handle<SimClusterCollection>& sCCH,
    const hgcal::SimToRecoCollectionWithSimClusters& sCToLCs) const {
  hgcal::SimTracksterToRecoCollection returnValue(productGetter_);

  const auto simTracksters = *sTCH.product();
  for (size_t tsId = 0; tsId < simTracksters.size(); ++tsId) {
    if (simTracksters[tsId].seedID() == cPCH.id()) {
      const auto cpId = simTracksters[tsId].seedIndex();
      const edm::Ref<CaloParticleCollection> cpRef(cPCH, cpId);
      const auto& lcIt = cPToLCs.find(cpRef);
      if (lcIt == cPToLCs.end()) {
        LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
            << "CaloParticle Id " << cpId << " not found in LayerCluster association map\n";
        continue;
      }

      const auto& lcs = lcIt->val;
      for (size_t lcId = 0; lcId < lcs.size(); ++lcId) {
        const edm::Ref<reco::CaloClusterCollection> lcRef(cCCH, lcId);
        const auto lcPair =
            std::find_if(std::begin(lcs),
                         std::end(lcs),
                         [&lcRef](const std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>& p) {
                           return p.first == lcRef;
                         });
        if (lcPair == lcs.end()) {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
              << "LayerCluster Id " << lcId << " not found in CaloParticle association map\n";
          continue;
        } else {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
              << "CaloParticle Id: \t" << cpId << "\t LayerCluster Id: \t" << lcPair->first.index() << "\t score \t"
              << lcPair->second.second << "\n";
          // Fill AssociationMap
          returnValue.insert(
              edm::Ref<ticl::TracksterCollection>(sTCH, tsId),                             // Ref to TS
              std::make_pair(lcRef,                                                        // Pair <Ref to LC,
                             std::make_pair(lcPair->second.first, lcPair->second.second))  // pair <energy, score> >
          );
        }
      }
    } else if (simTracksters[tsId].seedID() == sCCH.id()) {
      const auto scId = simTracksters[tsId].seedIndex();
      const edm::Ref<SimClusterCollection> scRef(sCCH, scId);
      const auto& lcIt = sCToLCs.find(scRef);
      if (lcIt == sCToLCs.end()) {
        LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
            << "SimCluster Id " << scId << " not found in LayerCluster association map\n";
        continue;
      }

      const auto& lcs = lcIt->val;
      for (size_t lcId = 0; lcId < lcs.size(); ++lcId) {
        const edm::Ref<reco::CaloClusterCollection> lcRef(cCCH, lcId);
        const auto lcPair =
            std::find_if(std::begin(lcs),
                         std::end(lcs),
                         [&lcRef](const std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>& p) {
                           return p.first == lcRef;
                         });
        if (lcPair == lcs.end()) {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
              << "LayerCluster Id " << lcId << " not found in SimCluster association map\n";
          continue;
        } else {
          LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
              << "SimCluster Id: \t" << scId << "\t LayerCluster Id: \t" << lcPair->first.index() << "\t score \t"
              << lcPair->second.second << "\n";
          // Fill AssociationMap
          returnValue.insert(
              edm::Ref<ticl::TracksterCollection>(sTCH, tsId),                             // Ref to TS
              std::make_pair(lcRef,                                                        // Pair <Ref to LC,
                             std::make_pair(lcPair->second.first, lcPair->second.second))  // pair <energy, score> >
          );
        }
      }
    } else {
      LogDebug("LCToSimTSAssociatorByEnergyScoreImpl")
          << "The seedID " << simTracksters[tsId].seedID() << " of SimTrackster " << tsId
          << " is neither a CaloParticle nor a SimCluster!\n";
    }

  }  // end loop over simTracksters
  return returnValue;
}
