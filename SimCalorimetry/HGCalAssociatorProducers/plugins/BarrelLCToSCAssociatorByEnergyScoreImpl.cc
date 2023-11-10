#include "BarrelLCToSCAssociatorByEnergyScoreImpl.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

BarrelLCToSCAssociatorByEnergyScoreImpl::BarrelLCToSCAssociatorByEnergyScoreImpl(
    edm::EDProductGetter const& productGetter,
    bool hardScatterOnly,
    const std::unordered_map<DetId, const reco::PFRecHit*>* hitMap)
    : hardScatterOnly_(hardScatterOnly), hitMap_(hitMap), productGetter_(&productGetter) {
  layers_ = 6;
}

ticl::association BarrelLCToSCAssociatorByEnergyScoreImpl::makeConnections(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<SimClusterCollection>& sCCH) const {
  const auto& clusters = *cCCH.product();
  const auto& simClusters = *sCCH.product();
  auto nLayerClusters = clusters.size();
  auto nSimClusters = simClusters.size();
  std::vector<size_t> sCIndices;
  for (unsigned int scId = 0; scId < nSimClusters; ++scId) {
    if (hardScatterOnly_ && (simClusters[scId].g4Tracks()[0].eventId().event() != 0 or
                             simClusters[scId].g4Tracks()[0].eventId().bunchCrossing() != 0)) {
      continue;
    }
    sCIndices.emplace_back(scId);
  }
  nSimClusters = sCIndices.size();
  ticl::simClusterToLayerCluster lcsInSimCluster;
  lcsInSimCluster.resize(nSimClusters);

  for (unsigned int i = 0; i < nSimClusters; ++i) {
    lcsInSimCluster[i].resize(layers_);
    for (unsigned int j = 0; j < layers_; ++j) {
      lcsInSimCluster[i][j].simClusterId = i;
      lcsInSimCluster[i][j].energy = 0.f;
      lcsInSimCluster[i][j].hits_and_fractions.clear();
    }
  }

  std::unordered_map<DetId, std::vector<ticl::detIdInfoInCluster>> detIdToSimClusterId_Map;
  for (const auto& scId : sCIndices) {
    const auto& hits_and_fractions = simClusters[scId].hits_and_fractions();
    for (const auto& it_haf : hits_and_fractions) {
      const auto hitid = (it_haf.first);
      auto scLayerId = 0;
      if ((DetId(hitid)).det() == DetId::Hcal) {
        scLayerId = (HcalDetId(hitid)).depth();
        if ((DetId(hitid)).subdetId() == HcalSubdetector::HcalOuter)
          scLayerId += 1;
      }
      const auto itcheck = hitMap_->find(hitid);
      if (itcheck != hitMap_->end()) {
        auto hit_find_it = detIdToSimClusterId_Map.find(hitid);
        if (hit_find_it == detIdToSimClusterId_Map.end()) {
          detIdToSimClusterId_Map[hitid] = std::vector<ticl::detIdInfoInCluster>();
        }
        detIdToSimClusterId_Map[hitid].emplace_back(scId, it_haf.second);

        const reco::PFRecHit* hit = itcheck->second;
        lcsInSimCluster[scId][scLayerId].energy += it_haf.second * hit->energy();
        lcsInSimCluster[scId][scLayerId].hits_and_fractions.emplace_back(hitid, it_haf.second);
      }
    }
  }
  std::unordered_map<DetId, std::vector<ticl::detIdInfoInCluster>> detIdToLayerClusterId_Map;
  ticl::layerClusterToSimCluster scsInLayerCluster;
  scsInLayerCluster.resize(nLayerClusters);

  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    const std::vector<std::pair<DetId, float>>& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    const auto firstHitDetId = hits_and_fractions[0].first;
    int lcLayerId = 0;
    if (DetId(firstHitDetId).det() == DetId::Hcal) {
      lcLayerId = (HcalDetId(firstHitDetId)).depth();
      if ((DetId(firstHitDetId)).subdetId() == HcalSubdetector::HcalOuter)
        lcLayerId += 1;
    }

    for (unsigned int hitId = 0; hitId < numberOfHitsInLC; ++hitId) {
      const auto rh_detid = hits_and_fractions[hitId].first;
      const auto rhFraction = hits_and_fractions[hitId].second;

      auto hit_find_in_LC = detIdToLayerClusterId_Map.find(rh_detid);
      if (hit_find_in_LC == detIdToLayerClusterId_Map.end()) {
        detIdToLayerClusterId_Map[rh_detid] = std::vector<ticl::detIdInfoInCluster>();
      }
      detIdToLayerClusterId_Map[rh_detid].emplace_back(lcId, rhFraction);

      auto hit_find_in_SC = detIdToSimClusterId_Map.find(rh_detid);

      if (hit_find_in_SC != detIdToSimClusterId_Map.end()) {
        const auto itcheck = hitMap_->find(rh_detid);
        const reco::PFRecHit* hit = itcheck->second;

        for (auto& h : hit_find_in_SC->second) {
          lcsInSimCluster[h.clusterId][lcLayerId].layerClusterIdToEnergyAndScore[lcId].first +=
              h.fraction * hit->energy();
          scsInLayerCluster[lcId].emplace_back(h.clusterId, 0.f);
        }
      }
    }
  }

  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    std::sort(scsInLayerCluster[lcId].begin(), scsInLayerCluster[lcId].end());
    auto last = std::unique(scsInLayerCluster[lcId].begin(), scsInLayerCluster[lcId].end());
    scsInLayerCluster[lcId].erase(last, scsInLayerCluster[lcId].end());
    const auto& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();

    if (clusters[lcId].energy() == 0. && !scsInLayerCluster[lcId].empty()) {
      for (auto& scPair : scsInLayerCluster[lcId]) {
        scPair.second = 1;
      }
      continue;
    }

    float invLayerClusterEnergyWeight = 0.f;
    for (auto const& haf : hits_and_fractions) {
      invLayerClusterEnergyWeight +=
          (haf.second * hitMap_->at(haf.first)->energy()) * (haf.second * hitMap_->at(haf.first)->energy());
    }
    invLayerClusterEnergyWeight = 1.f / invLayerClusterEnergyWeight;
    for (unsigned int i = 0; i < numberOfHitsInLC; ++i) {
      DetId rh_detid = hits_and_fractions[i].first;
      float rhFraction = hits_and_fractions[i].second;

      bool hitWithSC = (detIdToSimClusterId_Map.find(rh_detid) != detIdToSimClusterId_Map.end());

      auto itcheck = hitMap_->find(rh_detid);
      const reco::PFRecHit* hit = itcheck->second;
      float hitEnergyWeight = hit->energy() * hit->energy();

      for (auto& scPair : scsInLayerCluster[lcId]) {
        float scFraction = 0.f;
        if (hitWithSC) {
          auto findHitIt = std::find(detIdToSimClusterId_Map[rh_detid].begin(),
                                     detIdToSimClusterId_Map[rh_detid].end(),
                                     ticl::detIdInfoInCluster{scPair.first, 0.f});
          if (findHitIt != detIdToSimClusterId_Map[rh_detid].end())
            scFraction = findHitIt->fraction;
        }
        scPair.second += std::min(std::pow(rhFraction - scFraction, 2), std::pow(rhFraction, 2)) * hitEnergyWeight *
                         invLayerClusterEnergyWeight;
      }
    }
  }
  for (const auto& scId : sCIndices) {
    for (unsigned int layerId = 0; layerId < layers_; ++layerId) {
      unsigned int SCNumberOfHits = lcsInSimCluster[scId][layerId].hits_and_fractions.size();
      if (SCNumberOfHits == 0)
        continue;
      float invSCEnergyWeight = 0.f;
      for (auto const& haf : lcsInSimCluster[scId][layerId].hits_and_fractions) {
        invSCEnergyWeight += std::pow(haf.second * hitMap_->at(haf.first)->energy(), 2);
      }
      invSCEnergyWeight = 1.f / invSCEnergyWeight;
      for (unsigned int i = 0; i < SCNumberOfHits; ++i) {
        auto& sc_hitDetId = lcsInSimCluster[scId][layerId].hits_and_fractions[i].first;
        auto& scFraction = lcsInSimCluster[scId][layerId].hits_and_fractions[i].second;

        bool hitWithLC = false;
        if (scFraction == 0.f)
          continue;
        auto hit_find_in_LC = detIdToLayerClusterId_Map.find(sc_hitDetId);
        if (hit_find_in_LC != detIdToLayerClusterId_Map.end())
          hitWithLC = true;
        auto itcheck = hitMap_->find(sc_hitDetId);
        const reco::PFRecHit* hit = itcheck->second;
        float hitEnergyWeight = hit->energy() * hit->energy();
        for (auto& lcPair : lcsInSimCluster[scId][layerId].layerClusterIdToEnergyAndScore) {
          unsigned int layerClusterId = lcPair.first;
          float lcFraction = 0.f;

          if (hitWithLC) {
            auto findHitIt = std::find(detIdToLayerClusterId_Map[sc_hitDetId].begin(),
                                       detIdToLayerClusterId_Map[sc_hitDetId].end(),
                                       ticl::detIdInfoInCluster{layerClusterId, 0.f});
            if (findHitIt != detIdToLayerClusterId_Map[sc_hitDetId].end())
              lcFraction = findHitIt->fraction;
          }
          lcPair.second.second += std::min(std::pow(lcFraction - scFraction, 2), std::pow(scFraction, 2)) *
                                  hitEnergyWeight * invSCEnergyWeight;
        }
      }
    }
  }
  return {scsInLayerCluster, lcsInSimCluster};
}

ticl::RecoToSimCollectionWithSimClusters BarrelLCToSCAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<SimClusterCollection>& sCCH) const {
  ticl::RecoToSimCollectionWithSimClusters returnValue(productGetter_);
  const auto& links = makeConnections(cCCH, sCCH);

  const auto& scsInLayerCluster = std::get<0>(links);
  for (size_t lcId = 0; lcId < scsInLayerCluster.size(); ++lcId) {
    for (const auto& scPair : scsInLayerCluster[lcId]) {
      returnValue.insert(edm::Ref<reco::CaloClusterCollection>(cCCH, lcId),
                         std::make_pair(edm::Ref<SimClusterCollection>(sCCH, scPair.first), scPair.second));
      returnValue.insert(edm::Ref<reco::CaloClusterCollection>(cCCH, lcId),
                         std::make_pair(edm::Ref<SimClusterCollection>(sCCH, scPair.first), scPair.second));
    }
  }
  return returnValue;
}

ticl::SimToRecoCollectionWithSimClusters BarrelLCToSCAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<SimClusterCollection>& sCCH) const {
  ticl::SimToRecoCollectionWithSimClusters returnValue(productGetter_);
  const auto& links = makeConnections(cCCH, sCCH);

  const auto& lcsInSimCluster = std::get<1>(links);
  for (size_t scId = 0; scId < lcsInSimCluster.size(); ++scId) {
    for (size_t layerId = 0; layerId < lcsInSimCluster[scId].size(); ++layerId) {
      for (const auto& lcPair : lcsInSimCluster[scId][layerId].layerClusterIdToEnergyAndScore) {
        returnValue.insert(edm::Ref<SimClusterCollection>(sCCH, scId),
                           std::make_pair(edm::Ref<reco::CaloClusterCollection>(cCCH, lcPair.first),
                                          std::make_pair(lcPair.second.first, lcPair.second.second)));
      }
    }
  }
  return returnValue;
}
