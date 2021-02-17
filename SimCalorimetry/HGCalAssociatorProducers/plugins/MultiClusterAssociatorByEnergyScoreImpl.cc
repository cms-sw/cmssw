// Original Author: Leonardo Cristella
//

#include "MultiClusterAssociatorByEnergyScoreImpl.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

MultiClusterAssociatorByEnergyScoreImpl::MultiClusterAssociatorByEnergyScoreImpl(
    edm::EDProductGetter const& productGetter,
    bool hardScatterOnly,
    std::shared_ptr<hgcal::RecHitTools> recHitTools,
    const std::unordered_map<DetId, const HGCRecHit*>*& hitMap)
    : hardScatterOnly_(hardScatterOnly), recHitTools_(recHitTools), hitMap_(hitMap), productGetter_(&productGetter) {
  layers_ = recHitTools_->lastLayerBH();
}

hgcal::association MultiClusterAssociatorByEnergyScoreImpl::makeConnections(
    const edm::Handle<reco::HGCalMultiClusterCollection>& mCCH, const edm::Handle<CaloParticleCollection>& cPCH) const {
  // 1. Extract collections and filter CaloParticles, if required
  const auto& clusters = *mCCH.product();
  const auto& caloParticles = *cPCH.product();
  auto nMultiClusters = clusters.size();
  //Consider CaloParticles coming from the hard scatterer, excluding the PU contribution.
  auto nCaloParticles = caloParticles.size();
  std::vector<size_t> cPIndices;
  //Consider CaloParticles coming from the hard scatterer
  //excluding the PU contribution and save the indices.
  for (unsigned int cpId = 0; cpId < nCaloParticles; ++cpId) {
    if (hardScatterOnly_ && (caloParticles[cpId].g4Tracks()[0].eventId().event() != 0 or
                             caloParticles[cpId].g4Tracks()[0].eventId().bunchCrossing() != 0)) {
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << "Excluding CaloParticles from event: " << caloParticles[cpId].g4Tracks()[0].eventId().event()
          << " with BX: " << caloParticles[cpId].g4Tracks()[0].eventId().bunchCrossing() << std::endl;
      continue;
    }
    cPIndices.emplace_back(cpId);
  }
  nCaloParticles = cPIndices.size();

  // Initialize cPOnLayer. To be returned outside, since it contains the
  // information to compute the CaloParticle-To-MultiCluster score.
  hgcal::caloParticleToMultiCluster cPOnLayer;
  cPOnLayer.resize(nCaloParticles);
  for (unsigned int i = 0; i < nCaloParticles; ++i) {
    cPOnLayer[i].resize(layers_ * 2);
    for (unsigned int j = 0; j < layers_ * 2; ++j) {
      cPOnLayer[i][j].caloParticleId = i;
      cPOnLayer[i][j].energy = 0.f;
      cPOnLayer[i][j].hits_and_fractions.clear();
      //cPOnLayer[i][j].multiClusterIdToEnergyAndScore.reserve(nMultiClusters); // Not necessary but may improve performance
    }
  }

  // Fill detIdToCaloParticleId_Map and update cPOnLayer
  std::unordered_map<DetId, std::vector<hgcal::detIdInfoInCluster>> detIdToCaloParticleId_Map;
  for (const auto& cpId : cPIndices) {
    const SimClusterRefVector& simClusterRefVector = caloParticles[cpId].simClusters();
    for (const auto& it_sc : simClusterRefVector) {
      const SimCluster& simCluster = (*(it_sc));
      const auto& hits_and_fractions = simCluster.hits_and_fractions();
      for (const auto& it_haf : hits_and_fractions) {
        const auto hitid = (it_haf.first);
        const auto cpLayerId =
            recHitTools_->getLayerWithOffset(hitid) + layers_ * ((recHitTools_->zside(hitid) + 1) >> 1) - 1;
        const auto itcheck = hitMap_->find(hitid);
        if (itcheck != hitMap_->end()) {
          auto hit_find_it = detIdToCaloParticleId_Map.find(hitid);
          if (hit_find_it == detIdToCaloParticleId_Map.end()) {
            detIdToCaloParticleId_Map[hitid] = std::vector<hgcal::detIdInfoInCluster>();
            detIdToCaloParticleId_Map[hitid].emplace_back(cpId, it_haf.second);
          } else {
            auto findHitIt = std::find(detIdToCaloParticleId_Map[hitid].begin(),
                                       detIdToCaloParticleId_Map[hitid].end(),
                                       hgcal::detIdInfoInCluster{cpId, it_haf.second});
            if (findHitIt != detIdToCaloParticleId_Map[hitid].end()) {
              findHitIt->fraction += it_haf.second;
            } else {
              detIdToCaloParticleId_Map[hitid].emplace_back(cpId, it_haf.second);
            }
          }
          const HGCRecHit* hit = itcheck->second;
          cPOnLayer[cpId][cpLayerId].energy += it_haf.second * hit->energy();
          // We need to compress the hits and fractions in order to have a
          // reasonable score between CP and LC. Imagine, for example, that a
          // CP has detID X used by 2 SimClusters with different fractions. If
          // a single LC uses X with fraction 1 and is compared to the 2
          // contributions separately, it will be assigned a score != 0, which
          // is wrong.
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
  LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "cPOnLayer INFO" << std::endl;
  for (size_t cp = 0; cp < cPOnLayer.size(); ++cp) {
    LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "For CaloParticle Idx: " << cp << " we have: " << std::endl;
    for (size_t cpp = 0; cpp < cPOnLayer[cp].size(); ++cpp) {
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "  On Layer: " << cpp << " we have:" << std::endl;
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << "    CaloParticleIdx: " << cPOnLayer[cp][cpp].caloParticleId << std::endl;
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << "    Energy:          " << cPOnLayer[cp][cpp].energy << std::endl;
      double tot_energy = 0.;
      for (auto const& haf : cPOnLayer[cp][cpp].hits_and_fractions) {
        LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
            << "      Hits/fraction/energy: " << (uint32_t)haf.first << "/" << haf.second << "/"
            << haf.second * hitMap_->at(haf.first)->energy() << std::endl;
        tot_energy += haf.second * hitMap_->at(haf.first)->energy();
      }
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "    Tot Sum haf: " << tot_energy << std::endl;
      for (auto const& lc : cPOnLayer[cp][cpp].multiClusterIdToEnergyAndScore) {
        LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "      lcIdx/energy/score: " << lc.first << "/"
                                                            << lc.second.first << "/" << lc.second.second << std::endl;
      }
    }
  }

  LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "detIdToCaloParticleId_Map INFO" << std::endl;
  for (auto const& cp : detIdToCaloParticleId_Map) {
    LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
        << "For detId: " << (uint32_t)cp.first
        << " we have found the following connections with CaloParticles:" << std::endl;
    for (auto const& cpp : cp.second) {
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << "  CaloParticle Id: " << cpp.clusterId << " with fraction: " << cpp.fraction
          << " and energy: " << cpp.fraction * hitMap_->at(cp.first)->energy() << std::endl;
    }
  }
#endif

  // Fill detIdToMultiClusterId_Map and cpsInMultiCluster; update cPOnLayer
  std::unordered_map<DetId, std::vector<hgcal::detIdInfoInCluster>> detIdToMultiClusterId_Map;
  // this contains the ids of the caloparticles contributing with at least one
  // hit to the layer cluster and the reconstruction error. To be returned
  // since this contains the information to compute the
  // MultiCluster-To-CaloParticle score.
  hgcal::multiClusterToCaloParticle cpsInMultiCluster;
  cpsInMultiCluster.resize(nMultiClusters);

  for (unsigned int lcId = 0; lcId < nMultiClusters; ++lcId) {
    const std::vector<std::pair<DetId, float>>& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    const auto firstHitDetId = hits_and_fractions[0].first;
    int lcLayerId =
        recHitTools_->getLayerWithOffset(firstHitDetId) + layers_ * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;

    for (unsigned int hitId = 0; hitId < numberOfHitsInLC; hitId++) {
      const auto rh_detid = hits_and_fractions[hitId].first;
      const auto rhFraction = hits_and_fractions[hitId].second;

      auto hit_find_in_LC = detIdToMultiClusterId_Map.find(rh_detid);
      if (hit_find_in_LC == detIdToMultiClusterId_Map.end()) {
        detIdToMultiClusterId_Map[rh_detid] = std::vector<hgcal::detIdInfoInCluster>();
      }
      detIdToMultiClusterId_Map[rh_detid].emplace_back(lcId, rhFraction);

      auto hit_find_in_CP = detIdToCaloParticleId_Map.find(rh_detid);

      if (hit_find_in_CP != detIdToCaloParticleId_Map.end()) {
        const auto itcheck = hitMap_->find(rh_detid);
        const HGCRecHit* hit = itcheck->second;
        for (auto& h : hit_find_in_CP->second) {
          cPOnLayer[h.clusterId][lcLayerId].multiClusterIdToEnergyAndScore[lcId].first += h.fraction * hit->energy();
          cpsInMultiCluster[lcId].emplace_back(h.clusterId, 0.f);
        }
      }
    }  // End loop over hits on a MultiCluster
  }    // End of loop over MultiClusters

#ifdef EDM_ML_DEBUG
  for (unsigned int lcId = 0; lcId < nMultiClusters; ++lcId) {
    const auto& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    const auto firstHitDetId = hits_and_fractions[0].first;
    int lcLayerId =
        recHitTools_->getLayerWithOffset(firstHitDetId) + layers_ * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;

    // This vector will store, for each hit in the Layercluster, the index of
    // the CaloParticle that contributed the most, in terms of energy, to it.
    // Special values are:
    //
    // -2  --> the reconstruction fraction of the RecHit is 0 (used in the past to monitor Halo Hits)
    // -3  --> same as before with the added condition that no CaloParticle has been linked to this RecHit
    // -1  --> the reco fraction is >0, but no CaloParticle has been linked to it
    // >=0 --> index of the linked CaloParticle
    std::vector<int> hitsToCaloParticleId(numberOfHitsInLC);
    // This will store the index of the CaloParticle linked to the MultiCluster that has the most number of hits in common.
    int maxCPId_byNumberOfHits = -1;
    // This will store the maximum number of shared hits between a Layercluster andd a CaloParticle
    unsigned int maxCPNumberOfHitsInLC = 0;
    // This will store the index of the CaloParticle linked to the MultiCluster that has the most energy in common.
    int maxCPId_byEnergy = -1;
    // This will store the maximum number of shared energy between a Layercluster and a CaloParticle
    float maxEnergySharedLCandCP = 0.f;
    // This will store the fraction of the MultiCluster energy shared with the best(energy) CaloParticle: e_shared/lc_energy
    float energyFractionOfLCinCP = 0.f;
    // This will store the fraction of the CaloParticle energy shared with the MultiCluster: e_shared/cp_energy
    float energyFractionOfCPinLC = 0.f;
    std::unordered_map<unsigned, unsigned> occurrencesCPinLC;
    unsigned int numberOfNoiseHitsInLC = 0;
    std::unordered_map<unsigned, float> CPEnergyInLC;

    for (unsigned int hitId = 0; hitId < numberOfHitsInLC; hitId++) {
      const auto rh_detid = hits_and_fractions[hitId].first;
      const auto rhFraction = hits_and_fractions[hitId].second;

      auto hit_find_in_CP = detIdToCaloParticleId_Map.find(rh_detid);

      // if the fraction is zero or the hit does not belong to any calo
      // particle, set the caloparticleId for the hit to -1 this will
      // contribute to the number of noise hits

      // MR Remove the case in which the fraction is 0, since this could be a
      // real hit that has been marked as halo.
      if (rhFraction == 0.) {
        hitsToCaloParticleId[hitId] = -2;
      }
      if (hit_find_in_CP == detIdToCaloParticleId_Map.end()) {
        hitsToCaloParticleId[hitId] -= 1;
      } else {
        const auto itcheck = hitMap_->find(rh_detid);
        const HGCRecHit* hit = itcheck->second;
        auto maxCPEnergyInLC = 0.f;
        auto maxCPId = -1;
        for (auto& h : hit_find_in_CP->second) {
          CPEnergyInLC[h.clusterId] += h.fraction * hit->energy();
          // Keep track of which CaloParticle ccontributed the most, in terms
          // of energy, to this specific MultiCluster.
          if (CPEnergyInLC[h.clusterId] > maxCPEnergyInLC) {
            maxCPEnergyInLC = CPEnergyInLC[h.clusterId];
            maxCPId = h.clusterId;
          }
        }
        hitsToCaloParticleId[hitId] = maxCPId;
      }
    }  // End loop over hits on a MultiCluster

    for (const auto& c : hitsToCaloParticleId) {
      if (c < 0) {
        numberOfNoiseHitsInLC++;
      } else {
        occurrencesCPinLC[c]++;
      }
    }

    for (const auto& c : occurrencesCPinLC) {
      if (c.second > maxCPNumberOfHitsInLC) {
        maxCPId_byNumberOfHits = c.first;
        maxCPNumberOfHitsInLC = c.second;
      }
    }

    for (const auto& c : CPEnergyInLC) {
      if (c.second > maxEnergySharedLCandCP) {
        maxCPId_byEnergy = c.first;
        maxEnergySharedLCandCP = c.second;
      }
    }

    float totalCPEnergyOnLayer = 0.f;
    if (maxCPId_byEnergy >= 0) {
      totalCPEnergyOnLayer = cPOnLayer[maxCPId_byEnergy][lcLayerId].energy;
      energyFractionOfCPinLC = maxEnergySharedLCandCP / totalCPEnergyOnLayer;
      if (clusters[lcId].energy() > 0.f) {
        energyFractionOfLCinCP = maxEnergySharedLCandCP / clusters[lcId].energy();
      }
    }

    LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << std::setw(10) << "LayerId:"
                                                        << "\t" << std::setw(12) << "multiCluster"
                                                        << "\t" << std::setw(10) << "lc energy"
                                                        << "\t" << std::setw(5) << "nhits"
                                                        << "\t" << std::setw(12) << "noise hits"
                                                        << "\t" << std::setw(22) << "maxCPId_byNumberOfHits"
                                                        << "\t" << std::setw(8) << "nhitsCP"
                                                        << "\t" << std::setw(13) << "maxCPId_byEnergy"
                                                        << "\t" << std::setw(20) << "maxEnergySharedLCandCP"
                                                        << "\t" << std::setw(22) << "totalCPEnergyOnLayer"
                                                        << "\t" << std::setw(22) << "energyFractionOfLCinCP"
                                                        << "\t" << std::setw(25) << "energyFractionOfCPinLC"
                                                        << "\t"
                                                        << "\n";
    LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
        << std::setw(10) << lcLayerId << "\t" << std::setw(12) << lcId << "\t" << std::setw(10)
        << clusters[lcId].energy() << "\t" << std::setw(5) << numberOfHitsInLC << "\t" << std::setw(12)
        << numberOfNoiseHitsInLC << "\t" << std::setw(22) << maxCPId_byNumberOfHits << "\t" << std::setw(8)
        << maxCPNumberOfHitsInLC << "\t" << std::setw(13) << maxCPId_byEnergy << "\t" << std::setw(20)
        << maxEnergySharedLCandCP << "\t" << std::setw(22) << totalCPEnergyOnLayer << "\t" << std::setw(22)
        << energyFractionOfLCinCP << "\t" << std::setw(25) << energyFractionOfCPinLC << "\n";
  }  // End of loop over MultiClusters

  LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "Improved cPOnLayer INFO" << std::endl;
  for (size_t cp = 0; cp < cPOnLayer.size(); ++cp) {
    LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "For CaloParticle Idx: " << cp << " we have: " << std::endl;
    for (size_t cpp = 0; cpp < cPOnLayer[cp].size(); ++cpp) {
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "  On Layer: " << cpp << " we have:" << std::endl;
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << "    CaloParticleIdx: " << cPOnLayer[cp][cpp].caloParticleId << std::endl;
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << "    Energy:          " << cPOnLayer[cp][cpp].energy << std::endl;
      double tot_energy = 0.;
      for (auto const& haf : cPOnLayer[cp][cpp].hits_and_fractions) {
        LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
            << "      Hits/fraction/energy: " << (uint32_t)haf.first << "/" << haf.second << "/"
            << haf.second * hitMap_->at(haf.first)->energy() << std::endl;
        tot_energy += haf.second * hitMap_->at(haf.first)->energy();
      }
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "    Tot Sum haf: " << tot_energy << std::endl;
      for (auto const& lc : cPOnLayer[cp][cpp].multiClusterIdToEnergyAndScore) {
        LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "      lcIdx/energy/score: " << lc.first << "/"
                                                            << lc.second.first << "/" << lc.second.second << std::endl;
      }
    }
  }

  LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "Improved detIdToCaloParticleId_Map INFO" << std::endl;
  for (auto const& cp : detIdToCaloParticleId_Map) {
    LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
        << "For detId: " << (uint32_t)cp.first
        << " we have found the following connections with CaloParticles:" << std::endl;
    for (auto const& cpp : cp.second) {
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << "  CaloParticle Id: " << cpp.clusterId << " with fraction: " << cpp.fraction
          << " and energy: " << cpp.fraction * hitMap_->at(cp.first)->energy() << std::endl;
    }
  }
#endif

  // Update cpsInMultiCluster; compute the score MultiCluster-to-CaloParticle,
  // together with the returned AssociationMap
  for (unsigned int lcId = 0; lcId < nMultiClusters; ++lcId) {
    // find the unique caloparticles id contributing to the layer clusters
    std::sort(cpsInMultiCluster[lcId].begin(), cpsInMultiCluster[lcId].end());
    auto last = std::unique(cpsInMultiCluster[lcId].begin(), cpsInMultiCluster[lcId].end());
    cpsInMultiCluster[lcId].erase(last, cpsInMultiCluster[lcId].end());
    const auto& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    // If a reconstructed MultiCluster has energy 0 but is linked to a
    // CaloParticle, assigned score 1
    if (clusters[lcId].energy() == 0. && !cpsInMultiCluster[lcId].empty()) {
      for (auto& cpPair : cpsInMultiCluster[lcId]) {
        cpPair.second = 1.;
        LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "multiClusterId : \t " << lcId << "\t CP id : \t"
                                                            << cpPair.first << "\t score \t " << cpPair.second << "\n";
      }
      continue;
    }

    // Compute the correct normalization
    float invMultiClusterEnergyWeight = 0.f;
    for (auto const& haf : clusters[lcId].hitsAndFractions()) {
      invMultiClusterEnergyWeight +=
          (haf.second * hitMap_->at(haf.first)->energy()) * (haf.second * hitMap_->at(haf.first)->energy());
    }
    invMultiClusterEnergyWeight = 1.f / invMultiClusterEnergyWeight;
    for (unsigned int i = 0; i < numberOfHitsInLC; ++i) {
      DetId rh_detid = hits_and_fractions[i].first;
      float rhFraction = hits_and_fractions[i].second;

      bool hitWithNoCP = (detIdToCaloParticleId_Map.find(rh_detid) == detIdToCaloParticleId_Map.end());

      auto itcheck = hitMap_->find(rh_detid);
      const HGCRecHit* hit = itcheck->second;
      float hitEnergyWeight = hit->energy() * hit->energy();

      for (auto& cpPair : cpsInMultiCluster[lcId]) {
        float cpFraction = 0.f;
        if (!hitWithNoCP) {
          auto findHitIt = std::find(detIdToCaloParticleId_Map[rh_detid].begin(),
                                     detIdToCaloParticleId_Map[rh_detid].end(),
                                     hgcal::detIdInfoInCluster{cpPair.first, 0.f});
          if (findHitIt != detIdToCaloParticleId_Map[rh_detid].end())
            cpFraction = findHitIt->fraction;
        }
        cpPair.second +=
            (rhFraction - cpFraction) * (rhFraction - cpFraction) * hitEnergyWeight * invMultiClusterEnergyWeight;
      }
    }  // End of loop over Hits within a MultiCluster
#ifdef EDM_ML_DEBUG
    if (cpsInMultiCluster[lcId].empty())
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "multiCluster Id: \t" << lcId << "\tCP id:\t-1 "
                                                          << "\t score \t-1"
                                                          << "\n";
#endif
  }  // End of loop over MultiClusters

  // Compute the CaloParticle-To-MultiCluster score
  for (const auto& cpId : cPIndices) {
    for (unsigned int layerId = 0; layerId < layers_ * 2; ++layerId) {
      unsigned int CPNumberOfHits = cPOnLayer[cpId][layerId].hits_and_fractions.size();
      if (CPNumberOfHits == 0)
        continue;
#ifdef EDM_ML_DEBUG
      int lcWithMaxEnergyInCP = -1;
      float maxEnergyLCinCP = 0.f;
      float CPenergy = cPOnLayer[cpId][layerId].energy;
      float CPEnergyFractionInLC = 0.f;
      for (auto& lc : cPOnLayer[cpId][layerId].multiClusterIdToEnergyAndScore) {
        if (lc.second.first > maxEnergyLCinCP) {
          maxEnergyLCinCP = lc.second.first;
          lcWithMaxEnergyInCP = lc.first;
        }
      }
      if (CPenergy > 0.f)
        CPEnergyFractionInLC = maxEnergyLCinCP / CPenergy;

      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << std::setw(8) << "LayerId:\t" << std::setw(12) << "caloparticle\t" << std::setw(15) << "cp total energy\t"
          << std::setw(15) << "cpEnergyOnLayer\t" << std::setw(14) << "CPNhitsOnLayer\t" << std::setw(18)
          << "lcWithMaxEnergyInCP\t" << std::setw(15) << "maxEnergyLCinCP\t" << std::setw(20) << "CPEnergyFractionInLC"
          << "\n";
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << std::setw(8) << layerId << "\t" << std::setw(12) << cpId << "\t" << std::setw(15)
          << caloParticles[cpId].energy() << "\t" << std::setw(15) << CPenergy << "\t" << std::setw(14)
          << CPNumberOfHits << "\t" << std::setw(18) << lcWithMaxEnergyInCP << "\t" << std::setw(15) << maxEnergyLCinCP
          << "\t" << std::setw(20) << CPEnergyFractionInLC << "\n";
#endif
      // Compute the correct normalization
      float invCPEnergyWeight = 0.f;
      for (auto const& haf : cPOnLayer[cpId][layerId].hits_and_fractions) {
        invCPEnergyWeight += std::pow(haf.second * hitMap_->at(haf.first)->energy(), 2);
      }
      invCPEnergyWeight = 1.f / invCPEnergyWeight;
      for (unsigned int i = 0; i < CPNumberOfHits; ++i) {
        auto& cp_hitDetId = cPOnLayer[cpId][layerId].hits_and_fractions[i].first;
        auto& cpFraction = cPOnLayer[cpId][layerId].hits_and_fractions[i].second;

        bool hitWithNoLC = false;
        if (cpFraction == 0.f)
          continue;  //hopefully this should never happen
        auto hit_find_in_LC = detIdToMultiClusterId_Map.find(cp_hitDetId);
        if (hit_find_in_LC == detIdToMultiClusterId_Map.end())
          hitWithNoLC = true;
        auto itcheck = hitMap_->find(cp_hitDetId);
        const HGCRecHit* hit = itcheck->second;
        float hitEnergyWeight = hit->energy() * hit->energy();
        for (auto& lcPair : cPOnLayer[cpId][layerId].multiClusterIdToEnergyAndScore) {
          unsigned int multiClusterId = lcPair.first;
          float lcFraction = 0.f;

          if (!hitWithNoLC) {
            auto findHitIt = std::find(detIdToMultiClusterId_Map[cp_hitDetId].begin(),
                                       detIdToMultiClusterId_Map[cp_hitDetId].end(),
                                       hgcal::detIdInfoInCluster{multiClusterId, 0.f});
            if (findHitIt != detIdToMultiClusterId_Map[cp_hitDetId].end())
              lcFraction = findHitIt->fraction;
          }
          lcPair.second.second +=
              (lcFraction - cpFraction) * (lcFraction - cpFraction) * hitEnergyWeight * invCPEnergyWeight;
#ifdef EDM_ML_DEBUG
          LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
              << "cpDetId:\t" << (uint32_t)cp_hitDetId << "\tmultiClusterId:\t" << multiClusterId << "\t"
              << "lcfraction,cpfraction:\t" << lcFraction << ", " << cpFraction << "\t"
              << "hitEnergyWeight:\t" << hitEnergyWeight << "\t"
              << "current score:\t" << lcPair.second.second << "\t"
              << "invCPEnergyWeight:\t" << invCPEnergyWeight << "\n";
#endif
        }  // End of loop over MultiClusters linked to hits of this CaloParticle
      }    // End of loop over hits of CaloParticle on a Layer
#ifdef EDM_ML_DEBUG
      if (cPOnLayer[cpId][layerId].multiClusterIdToEnergyAndScore.empty())
        LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "CP Id: \t" << cpId << "\tLC id:\t-1 "
                                                            << "\t score \t-1"
                                                            << "\n";

      for (const auto& lcPair : cPOnLayer[cpId][layerId].multiClusterIdToEnergyAndScore) {
        LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
            << "CP Id: \t" << cpId << "\t LC id: \t" << lcPair.first << "\t score \t" << lcPair.second.second
            << "\t shared energy:\t" << lcPair.second.first << "\t shared energy fraction:\t"
            << (lcPair.second.first / CPenergy) << "\n";
      }
#endif
    }
  }
  return {cpsInMultiCluster, cPOnLayer};
}

hgcal::RecoToSimCollectionWithMultiClusters MultiClusterAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<reco::HGCalMultiClusterCollection>& mCCH, const edm::Handle<CaloParticleCollection>& cPCH) const {
  hgcal::RecoToSimCollectionWithMultiClusters returnValue(productGetter_);
  const auto& links = makeConnections(mCCH, cPCH);

  const auto& cpsInMultiCluster = std::get<0>(links);
  for (size_t lcId = 0; lcId < cpsInMultiCluster.size(); ++lcId) {
    for (auto& cpPair : cpsInMultiCluster[lcId]) {
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << "multiCluster Id: \t" << lcId << "\t CP id: \t" << cpPair.first << "\t score \t" << cpPair.second << "\n";
      // Fill AssociationMap
      returnValue.insert(edm::Ref<reco::HGCalMultiClusterCollection>(mCCH, lcId),  // Ref to LC
                         std::make_pair(edm::Ref<CaloParticleCollection>(cPCH, cpPair.first),
                                        cpPair.second)  // Pair <Ref to CP, score>
      );
    }
  }
  return returnValue;
}

hgcal::SimToRecoCollectionWithMultiClusters MultiClusterAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<reco::HGCalMultiClusterCollection>& mCCH, const edm::Handle<CaloParticleCollection>& cPCH) const {
  hgcal::SimToRecoCollectionWithMultiClusters returnValue(productGetter_);
  const auto& links = makeConnections(mCCH, cPCH);
  const auto& cPOnLayer = std::get<1>(links);
  for (size_t cpId = 0; cpId < cPOnLayer.size(); ++cpId) {
    for (size_t layerId = 0; layerId < cPOnLayer[cpId].size(); ++layerId) {
      for (auto& lcPair : cPOnLayer[cpId][layerId].multiClusterIdToEnergyAndScore) {
        returnValue.insert(
            edm::Ref<CaloParticleCollection>(cPCH, cpId),                              // Ref to CP
            std::make_pair(edm::Ref<reco::HGCalMultiClusterCollection>(mCCH, lcPair.first),  // Pair <Ref to LC,
                           std::make_pair(lcPair.second.first, lcPair.second.second))  // pair <energy, score> >
        );
      }
    }
  }
  return returnValue;
}
