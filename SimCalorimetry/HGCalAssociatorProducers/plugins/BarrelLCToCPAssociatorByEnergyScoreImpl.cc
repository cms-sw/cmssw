#include "BarrelLCToCPAssociatorByEnergyScoreImpl.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "SimCalorimetry/HGCalAssociatorProducers/interface/AssociatorTools.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

BarrelLCToCPAssociatorByEnergyScoreImpl::BarrelLCToCPAssociatorByEnergyScoreImpl(
    edm::EDProductGetter const& productGetter,
    bool hardScatterOnly,
    const std::unordered_map<DetId, const reco::PFRecHit*>* hitMap)
    : hardScatterOnly_(hardScatterOnly), hitMap_(hitMap), productGetter_(&productGetter) {
  layers_ = 6;
}

ticl::association BarrelLCToCPAssociatorByEnergyScoreImpl::makeConnections(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<CaloParticleCollection>& cPCH) const {
  const auto& clusters = *cCCH.product();
  const auto& caloParticles = *cPCH.product();
  auto nLayerClusters = clusters.size();

  std::vector<size_t> cPIndices;
  removeCPFromPU(caloParticles, cPIndices, hardScatterOnly_);
  auto nCaloParticles = cPIndices.size();

  ticl::caloParticleToLayerCluster cPOnLayer;
  cPOnLayer.resize(nCaloParticles);
  for (unsigned int i = 0; i < nCaloParticles; ++i) {
    cPOnLayer[i].resize(layers_);
    for (unsigned int j = 0; j < layers_; ++j) {
      cPOnLayer[i][j].caloParticleId = i;
      cPOnLayer[i][j].energy = 0.f;
      cPOnLayer[i][j].hits_and_fractions.clear();
    }
  }

  std::unordered_map<DetId, std::vector<ticl::detIdInfoInCluster>> detIdToCaloParticleId_Map;
  for (const auto& cpId : cPIndices) {
    const SimClusterRefVector& simClusterRefVector = caloParticles[cpId].simClusters();
    for (const auto& it_sc : simClusterRefVector) {
      const SimCluster& simCluster = (*(it_sc));
      const auto& hits_and_fractions = simCluster.hits_and_fractions();
      for (const auto& it_haf : hits_and_fractions) {
        const auto hitid = (it_haf.first);
        DetId hitId(hitid);
        int cpLayerId = 0;
        if (hitId.det() == DetId::Hcal)
          cpLayerId = (HcalDetId(hitid)).depth();
        else if (hitId.subdetId() == HcalSubdetector::HcalOuter)
          cpLayerId = (HcalDetId(hitid)).depth() + 1;
        const auto itcheck = hitMap_->find(hitid);
        if (itcheck != hitMap_->end()) {
          auto hit_find_it = detIdToCaloParticleId_Map.find(hitid);
          if (hit_find_it == detIdToCaloParticleId_Map.end()) {
            detIdToCaloParticleId_Map[hitid] = std::vector<ticl::detIdInfoInCluster>();
            detIdToCaloParticleId_Map[hitid].emplace_back(cpId, it_haf.second);
          } else {
            auto findHitIt = std::find(detIdToCaloParticleId_Map[hitid].begin(),
                                       detIdToCaloParticleId_Map[hitid].end(),
                                       ticl::detIdInfoInCluster{cpId, it_haf.second});
            if (findHitIt != detIdToCaloParticleId_Map[hitid].end()) {
              findHitIt->fraction += it_haf.second;
            } else {
              detIdToCaloParticleId_Map[hitid].emplace_back(cpId, it_haf.second);
            }
          }
          const reco::PFRecHit* hit = itcheck->second;
          cPOnLayer[cpId][cpLayerId].energy += it_haf.second * hit->energy();
          auto& haf = cPOnLayer[cpId][cpLayerId].hits_and_fractions;
          auto found = std::find_if(
              std::begin(haf), std::end(haf), [&hitid](const std::pair<DetId, float>& v) { return v.first == hitid; });
          if (found != haf.end()) {
            found->second += it_haf.second;
          } else {
            cPOnLayer[cpId][cpLayerId].hits_and_fractions.emplace_back(hitid, it_haf.second);
          }
        }
      }
    }
  }

#ifdef EDM_ML_DEBUG
  LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl") << "cPOnLayer INFO" << std::endl;
  for (size_t cp = 0; cp < cPOnLayer.size(); ++cp) {
    LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl") << "For CaloParticle Idx: " << cp << " we have: " << std::endl;
    for (size_t cpp = 0; cpp < cPOnLayer[cp].size(); ++cpp) {
      LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl") << "  On Layer: " << cpp << " we have:" << std::endl;
      LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl")
          << "    CaloParticleIdx: " << cPOnLayer[cp][cpp].caloParticleId << std::endl;
      LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl")
          << "    Energy:          " << cPOnLayer[cp][cpp].energy << std::endl;
      double tot_energy = 0.;
      for (auto const& haf : cPOnLayer[cp][cpp].hits_and_fractions) {
        LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl")
            << "      Hits/fraction/energy: " << (uint32_t)haf.first << "/" << haf.second << "/"
            << haf.second * hitMap_->at(haf.first)->energy() << std::endl;
        tot_energy += haf.second * hitMap_->at(haf.first)->energy();
      }
      LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl") << "    Tot Sum haf: " << tot_energy << std::endl;
      for (auto const& lc : cPOnLayer[cp][cpp].layerClusterIdToEnergyAndScore) {
        LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl") << "      lcIdx/energy/score: " << lc.first << "/"
                                                            << lc.second.first << "/" << lc.second.second << std::endl;
      }
    }
  }

  LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl") << "detIdToCaloParticleId_Map INFO" << std::endl;
  for (auto const& cp : detIdToCaloParticleId_Map) {
    LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl")
        << "For detId: " << (uint32_t)cp.first
        << " we have found the following connections with CaloParticles:" << std::endl;
    for (auto const& cpp : cp.second) {
      LogDebug("BarrelLCToCPAssociatorByEnergyScoreImpl")
          << "  CaloParticle Id: " << cpp.clusterId << " with fraction: " << cpp.fraction
          << " and energy: " << cpp.fraction * hitMap_->at(cp.first)->energy() << std::endl;
    }
  }
#endif

  std::unordered_map<DetId, std::vector<ticl::detIdInfoInCluster>> detIdToLayerClusterId_Map;
  ticl::layerClusterToCaloParticle cpsInLayerCluster;
  cpsInLayerCluster.resize(nLayerClusters);

  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    const std::vector<std::pair<DetId, float>>& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitInLC = hits_and_fractions.size();
    const auto firstHitDetId = hits_and_fractions[0].first;
    int lcLayerId = 0;
    if (firstHitDetId.det() == DetId::Hcal)
      lcLayerId = (HcalDetId(firstHitDetId)).depth();
    else if (firstHitDetId.subdetId() == HcalSubdetector::HcalOuter)
      lcLayerId = (HcalDetId(firstHitDetId)).depth() + 1;

    for (unsigned int hitId = 0; hitId < numberOfHitInLC; ++hitId) {
      const auto rh_detid = hits_and_fractions[hitId].first;
      const auto rhFraction = hits_and_fractions[hitId].second;

      auto hit_find_in_LC = detIdToLayerClusterId_Map.find(rh_detid);
      if (hit_find_in_LC == detIdToLayerClusterId_Map.end()) {
        detIdToLayerClusterId_Map[rh_detid] = std::vector<ticl::detIdInfoInCluster>();
      }
      detIdToLayerClusterId_Map[rh_detid].emplace_back(lcId, rhFraction);

      auto hit_find_in_CP = detIdToCaloParticleId_Map.find(rh_detid);

      if (hit_find_in_CP != detIdToCaloParticleId_Map.end()) {
        const auto itcheck = hitMap_->find(rh_detid);
        const reco::PFRecHit* hit = itcheck->second;
        for (auto& h : hit_find_in_CP->second) {
          cPOnLayer[h.clusterId][lcLayerId].layerClusterIdToEnergyAndScore[lcId].first += h.fraction * hit->energy();
          cpsInLayerCluster[lcId].emplace_back(h.clusterId, 0.f);
        }
      }
    }
  }

  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    std::sort(cpsInLayerCluster[lcId].begin(), cpsInLayerCluster[lcId].end());
    auto last = std::unique(cpsInLayerCluster[lcId].begin(), cpsInLayerCluster[lcId].end());
    cpsInLayerCluster[lcId].erase(last, cpsInLayerCluster[lcId].end());
    const auto& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();

    if (clusters[lcId].energy() == 0. && !cpsInLayerCluster[lcId].empty()) {
      for (auto& cpPair : cpsInLayerCluster[lcId]) {
        cpPair.second = 1.;
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

      bool hitWithNoCP = (detIdToCaloParticleId_Map.find(rh_detid) == detIdToCaloParticleId_Map.end());

      auto itcheck = hitMap_->find(rh_detid);
      const reco::PFRecHit* hit = itcheck->second;
      float hitEnergyWeight = hit->energy() * hit->energy();

      for (auto& cpPair : cpsInLayerCluster[lcId]) {
        float cpFraction = 0.f;
        if (!hitWithNoCP) {
          auto findHitIt = std::find(detIdToCaloParticleId_Map[rh_detid].begin(),
                                     detIdToCaloParticleId_Map[rh_detid].end(),
                                     ticl::detIdInfoInCluster{cpPair.first, 0.f});
          if (findHitIt != detIdToCaloParticleId_Map[rh_detid].end())
            cpFraction = findHitIt->fraction;
        }
        cpPair.second += std::min(std::pow(rhFraction - cpFraction, 2), std::pow(rhFraction, 2)) * hitEnergyWeight *
                         invLayerClusterEnergyWeight;
      }
    }
  }

  for (const auto& cpId : cPIndices) {
    for (unsigned int layerId = 0; layerId < layers_; ++layerId) {
      unsigned int CPNumberOfHits = cPOnLayer[cpId][layerId].hits_and_fractions.size();
      if (CPNumberOfHits == 0)
        continue;
      float invCPEnergyWeight = 0.;
      for (auto const& haf : cPOnLayer[cpId][layerId].hits_and_fractions) {
        invCPEnergyWeight += std::pow(haf.second * hitMap_->at(haf.first)->energy(), 2);
      }
      invCPEnergyWeight = 1.f / invCPEnergyWeight;
      for (unsigned int i = 0; i < CPNumberOfHits; ++i) {
        auto& cp_hitDetId = cPOnLayer[cpId][layerId].hits_and_fractions[i].first;
        auto& cpFraction = cPOnLayer[cpId][layerId].hits_and_fractions[i].second;

        bool hitWithNoLC = false;
        if (cpFraction == 0.f)
          continue;
        auto hit_find_in_LC = detIdToLayerClusterId_Map.find(cp_hitDetId);
        if (hit_find_in_LC == detIdToLayerClusterId_Map.end())
          hitWithNoLC = true;
        auto itcheck = hitMap_->find(cp_hitDetId);
        const reco::PFRecHit* hit = itcheck->second;
        float hitEnergyWeight = hit->energy() * hit->energy();
        for (auto& lcPair : cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore) {
          unsigned int layerClusterId = lcPair.first;
          float lcFraction = 0.f;

          if (!hitWithNoLC) {
            auto findHitIt = std::find(detIdToLayerClusterId_Map[cp_hitDetId].begin(),
                                       detIdToLayerClusterId_Map[cp_hitDetId].end(),
                                       ticl::detIdInfoInCluster{layerClusterId, 0.f});
            if (findHitIt != detIdToLayerClusterId_Map[cp_hitDetId].end())
              lcFraction = findHitIt->fraction;
          }
          lcPair.second.second += std::min(std::pow(lcFraction - cpFraction, 2), std::pow(cpFraction, 2)) *
                                  hitEnergyWeight * invCPEnergyWeight;
        }
      }
    }
  }
  return {cpsInLayerCluster, cPOnLayer};
}

ticl::RecoToSimCollection BarrelLCToCPAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<CaloParticleCollection>& cPCH) const {
  ticl::RecoToSimCollection returnValue(productGetter_);
  const auto& links = makeConnections(cCCH, cPCH);

  const auto& cpsInLayerCluster = std::get<0>(links);
  for (size_t lcId = 0; lcId < cpsInLayerCluster.size(); ++lcId) {
    for (const auto& cpPair : cpsInLayerCluster[lcId]) {
      returnValue.insert(edm::Ref<reco::CaloClusterCollection>(cCCH, lcId),
                         std::make_pair(edm::Ref<CaloParticleCollection>(cPCH, cpPair.first), cpPair.second));
    }
  }
  return returnValue;
}

ticl::SimToRecoCollection BarrelLCToCPAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<CaloParticleCollection>& cPCH) const {
  ticl::SimToRecoCollection returnValue(productGetter_);
  const auto& links = makeConnections(cCCH, cPCH);

  const auto& cPOnLayer = std::get<1>(links);
  for (size_t cpId = 0; cpId < cPOnLayer.size(); ++cpId) {
    for (size_t layerId = 0; layerId < cPOnLayer[cpId].size(); ++layerId) {
      for (const auto& lcPair : cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore) {
        returnValue.insert(edm::Ref<CaloParticleCollection>(cPCH, cpId),
                           std::make_pair(edm::Ref<reco::CaloClusterCollection>(cCCH, lcPair.first),
                                          std::make_pair(lcPair.second.first, lcPair.second.second)));
      }
    }
  }
  return returnValue;
}
