// Original Author: Marco Rovere
//

#include "LCToCPAssociatorByEnergyScoreImpl.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "SimCalorimetry/HGCalAssociatorProducers/interface/AssociatorTools.h"

LCToCPAssociatorByEnergyScoreImpl::LCToCPAssociatorByEnergyScoreImpl(
    edm::EDProductGetter const& productGetter,
    bool hardScatterOnly,
    std::shared_ptr<hgcal::RecHitTools> recHitTools,
    const std::unordered_map<DetId, const HGCRecHit*>* hitMap)
    : hardScatterOnly_(hardScatterOnly), recHitTools_(recHitTools), hitMap_(hitMap), productGetter_(&productGetter) {
  layers_ = recHitTools_->lastLayerBH();
}

hgcal::association LCToCPAssociatorByEnergyScoreImpl::makeConnections(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<CaloParticleCollection>& cPCH) const {
  // Get collections
  const auto& clusters = *cCCH.product();
  const auto& caloParticles = *cPCH.product();
  auto nLayerClusters = clusters.size();

  //Consider CaloParticles coming from the hard scatterer, excluding the PU contribution and save the indices.
  std::vector<size_t> cPIndices;
  removeCPFromPU(caloParticles, cPIndices, hardScatterOnly_);
  auto nCaloParticles = cPIndices.size();

  // Initialize cPOnLayer. It contains the caloParticleOnLayer structure for all CaloParticles in each layer and
  // among other the information to compute the CaloParticle-To-LayerCluster score. It is one of the two objects that
  // build the output of the makeConnections function.
  // cPOnLayer[cpId][layerId]
  hgcal::caloParticleToLayerCluster cPOnLayer;
  cPOnLayer.resize(nCaloParticles);
  for (unsigned int i = 0; i < nCaloParticles; ++i) {
    cPOnLayer[i].resize(layers_ * 2);
    for (unsigned int j = 0; j < layers_ * 2; ++j) {
      cPOnLayer[i][j].caloParticleId = i;
      cPOnLayer[i][j].energy = 0.f;
      cPOnLayer[i][j].hits_and_fractions.clear();
      //cPOnLayer[i][j].layerClusterIdToEnergyAndScore.reserve(nLayerClusters); // Not necessary but may improve performance
    }
  }

  // Fill detIdToCaloParticleId_Map and update cPOnLayer
  // The detIdToCaloParticleId_Map is used to connect a hit Detid (key) with all the CaloParticles that
  // contributed to that hit by storing the CaloParticle id and the fraction of the hit. Observe here
  // that all the different contributions of the same CaloParticle to a single hit (coming from their
  // internal SimClusters) are merged into a single entry with the fractions properly summed.
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
  LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "cPOnLayer INFO" << std::endl;
  for (size_t cp = 0; cp < cPOnLayer.size(); ++cp) {
    LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "For CaloParticle Idx: " << cp << " we have: " << std::endl;
    for (size_t cpp = 0; cpp < cPOnLayer[cp].size(); ++cpp) {
      LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "  On Layer: " << cpp << " we have:" << std::endl;
      LogDebug("LCToCPAssociatorByEnergyScoreImpl")
          << "    CaloParticleIdx: " << cPOnLayer[cp][cpp].caloParticleId << std::endl;
      LogDebug("LCToCPAssociatorByEnergyScoreImpl")
          << "    Energy:          " << cPOnLayer[cp][cpp].energy << std::endl;
      double tot_energy = 0.;
      for (auto const& haf : cPOnLayer[cp][cpp].hits_and_fractions) {
        LogDebug("LCToCPAssociatorByEnergyScoreImpl")
            << "      Hits/fraction/energy: " << (uint32_t)haf.first << "/" << haf.second << "/"
            << haf.second * hitMap_->at(haf.first)->energy() << std::endl;
        tot_energy += haf.second * hitMap_->at(haf.first)->energy();
      }
      LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "    Tot Sum haf: " << tot_energy << std::endl;
      for (auto const& lc : cPOnLayer[cp][cpp].layerClusterIdToEnergyAndScore) {
        LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "      lcIdx/energy/score: " << lc.first << "/"
                                                      << lc.second.first << "/" << lc.second.second << std::endl;
      }
    }
  }

  LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "detIdToCaloParticleId_Map INFO" << std::endl;
  for (auto const& cp : detIdToCaloParticleId_Map) {
    LogDebug("LCToCPAssociatorByEnergyScoreImpl")
        << "For detId: " << (uint32_t)cp.first
        << " we have found the following connections with CaloParticles:" << std::endl;
    for (auto const& cpp : cp.second) {
      LogDebug("LCToCPAssociatorByEnergyScoreImpl")
          << "  CaloParticle Id: " << cpp.clusterId << " with fraction: " << cpp.fraction
          << " and energy: " << cpp.fraction * hitMap_->at(cp.first)->energy() << std::endl;
    }
  }
#endif

  // Fill detIdToLayerClusterId_Map and cpsInLayerCluster; update cPOnLayer
  std::unordered_map<DetId, std::vector<hgcal::detIdInfoInCluster>> detIdToLayerClusterId_Map;
  // this contains the ids of the caloparticles contributing with at least one
  // hit to the layer cluster and the reconstruction error. To be returned
  // since this contains the information to compute the
  // LayerCluster-To-CaloParticle score.
  hgcal::layerClusterToCaloParticle cpsInLayerCluster;
  cpsInLayerCluster.resize(nLayerClusters);

  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    const std::vector<std::pair<DetId, float>>& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    const auto firstHitDetId = hits_and_fractions[0].first;
    int lcLayerId =
        recHitTools_->getLayerWithOffset(firstHitDetId) + layers_ * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;

    for (unsigned int hitId = 0; hitId < numberOfHitsInLC; hitId++) {
      const auto rh_detid = hits_and_fractions[hitId].first;
      const auto rhFraction = hits_and_fractions[hitId].second;

      auto hit_find_in_LC = detIdToLayerClusterId_Map.find(rh_detid);
      if (hit_find_in_LC == detIdToLayerClusterId_Map.end()) {
        detIdToLayerClusterId_Map[rh_detid] = std::vector<hgcal::detIdInfoInCluster>();
      }
      detIdToLayerClusterId_Map[rh_detid].emplace_back(lcId, rhFraction);

      auto hit_find_in_CP = detIdToCaloParticleId_Map.find(rh_detid);

      if (hit_find_in_CP != detIdToCaloParticleId_Map.end()) {
        const auto itcheck = hitMap_->find(rh_detid);
        const HGCRecHit* hit = itcheck->second;
        for (auto& h : hit_find_in_CP->second) {
          cPOnLayer[h.clusterId][lcLayerId].layerClusterIdToEnergyAndScore[lcId].first += h.fraction * hit->energy();
          cpsInLayerCluster[lcId].emplace_back(h.clusterId, 0.f);
        }
      }
    }  // End loop over hits on a LayerCluster
  }    // End of loop over LayerClusters

#ifdef EDM_ML_DEBUG
  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
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
    // This will store the index of the CaloParticle linked to the LayerCluster that has the most number of hits in common.
    int maxCPId_byNumberOfHits = -1;
    // This will store the maximum number of shared hits between a Layercluster andd a CaloParticle
    unsigned int maxCPNumberOfHitsInLC = 0;
    // This will store the index of the CaloParticle linked to the LayerCluster that has the most energy in common.
    int maxCPId_byEnergy = -1;
    // This will store the maximum number of shared energy between a Layercluster and a CaloParticle
    float maxEnergySharedLCandCP = 0.f;
    // This will store the fraction of the LayerCluster energy shared with the best(energy) CaloParticle: e_shared/lc_energy
    float energyFractionOfLCinCP = 0.f;
    // This will store the fraction of the CaloParticle energy shared with the LayerCluster: e_shared/cp_energy
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
          // of energy, to this specific LayerCluster.
          if (CPEnergyInLC[h.clusterId] > maxCPEnergyInLC) {
            maxCPEnergyInLC = CPEnergyInLC[h.clusterId];
            maxCPId = h.clusterId;
          }
        }
        hitsToCaloParticleId[hitId] = maxCPId;
      }
    }  // End loop over hits on a LayerCluster

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

    LogDebug("LCToCPAssociatorByEnergyScoreImpl")
        << std::setw(10) << "LayerId:\t" << std::setw(12) << "layerCluster\t" << std::setw(10) << "lc energy\t"
        << std::setw(5) << "nhits\t" << std::setw(12) << "noise hits\t" << std::setw(22) << "maxCPId_byNumberOfHits\t"
        << std::setw(8) << "nhitsCP\t" << std::setw(13) << "maxCPId_byEnergy\t" << std::setw(20)
        << "maxEnergySharedLCandCP\t" << std::setw(22) << "totalCPEnergyOnLayer\t" << std::setw(22)
        << "energyFractionOfLCinCP\t" << std::setw(25) << "energyFractionOfCPinLC\t"
        << "\n";
    LogDebug("LCToCPAssociatorByEnergyScoreImpl")
        << std::setw(10) << lcLayerId << "\t" << std::setw(12) << lcId << "\t" << std::setw(10)
        << clusters[lcId].energy() << "\t" << std::setw(5) << numberOfHitsInLC << "\t" << std::setw(12)
        << numberOfNoiseHitsInLC << "\t" << std::setw(22) << maxCPId_byNumberOfHits << "\t" << std::setw(8)
        << maxCPNumberOfHitsInLC << "\t" << std::setw(13) << maxCPId_byEnergy << "\t" << std::setw(20)
        << maxEnergySharedLCandCP << "\t" << std::setw(22) << totalCPEnergyOnLayer << "\t" << std::setw(22)
        << energyFractionOfLCinCP << "\t" << std::setw(25) << energyFractionOfCPinLC << "\n";
  }  // End of loop over LayerClusters

  LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "Improved cPOnLayer INFO" << std::endl;
  for (size_t cp = 0; cp < cPOnLayer.size(); ++cp) {
    LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "For CaloParticle Idx: " << cp << " we have: " << std::endl;
    for (size_t cpp = 0; cpp < cPOnLayer[cp].size(); ++cpp) {
      LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "  On Layer: " << cpp << " we have:" << std::endl;
      LogDebug("LCToCPAssociatorByEnergyScoreImpl")
          << "    CaloParticleIdx: " << cPOnLayer[cp][cpp].caloParticleId << std::endl;
      LogDebug("LCToCPAssociatorByEnergyScoreImpl")
          << "    Energy:          " << cPOnLayer[cp][cpp].energy << std::endl;
      double tot_energy = 0.;
      for (auto const& haf : cPOnLayer[cp][cpp].hits_and_fractions) {
        LogDebug("LCToCPAssociatorByEnergyScoreImpl")
            << "      Hits/fraction/energy: " << (uint32_t)haf.first << "/" << haf.second << "/"
            << haf.second * hitMap_->at(haf.first)->energy() << std::endl;
        tot_energy += haf.second * hitMap_->at(haf.first)->energy();
      }
      LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "    Tot Sum haf: " << tot_energy << std::endl;
      for (auto const& lc : cPOnLayer[cp][cpp].layerClusterIdToEnergyAndScore) {
        LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "      lcIdx/energy/score: " << lc.first << "/"
                                                      << lc.second.first << "/" << lc.second.second << std::endl;
      }
    }
  }

  LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "Improved detIdToCaloParticleId_Map INFO" << std::endl;
  for (auto const& cp : detIdToCaloParticleId_Map) {
    LogDebug("LCToCPAssociatorByEnergyScoreImpl")
        << "For detId: " << (uint32_t)cp.first
        << " we have found the following connections with CaloParticles:" << std::endl;
    for (auto const& cpp : cp.second) {
      LogDebug("LCToCPAssociatorByEnergyScoreImpl")
          << "  CaloParticle Id: " << cpp.clusterId << " with fraction: " << cpp.fraction
          << " and energy: " << cpp.fraction * hitMap_->at(cp.first)->energy() << std::endl;
    }
  }
#endif

  // Update cpsInLayerCluster; compute the score LayerCluster-to-CaloParticle,
  // together with the returned AssociationMap
  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    // find the unique caloparticles id contributing to the layer clusters
    std::sort(cpsInLayerCluster[lcId].begin(), cpsInLayerCluster[lcId].end());
    auto last = std::unique(cpsInLayerCluster[lcId].begin(), cpsInLayerCluster[lcId].end());
    cpsInLayerCluster[lcId].erase(last, cpsInLayerCluster[lcId].end());
    const auto& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    // If a reconstructed LayerCluster has energy 0 but is linked to a
    // CaloParticle, assigned score 1
    if (clusters[lcId].energy() == 0. && !cpsInLayerCluster[lcId].empty()) {
      for (auto& cpPair : cpsInLayerCluster[lcId]) {
        cpPair.second = 1.;
        LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "layerClusterId : \t " << lcId << "\t CP id : \t"
                                                      << cpPair.first << "\t score \t " << cpPair.second << "\n";
      }
      continue;
    }

    // Compute the correct normalization
    // It is the inverse of the denominator of the LCToCP score formula. Observe that this is the sum of the squares.
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
      const HGCRecHit* hit = itcheck->second;
      float hitEnergyWeight = hit->energy() * hit->energy();

      for (auto& cpPair : cpsInLayerCluster[lcId]) {
        float cpFraction = 0.f;
        if (!hitWithNoCP) {
          auto findHitIt = std::find(detIdToCaloParticleId_Map[rh_detid].begin(),
                                     detIdToCaloParticleId_Map[rh_detid].end(),
                                     hgcal::detIdInfoInCluster{cpPair.first, 0.f});
          if (findHitIt != detIdToCaloParticleId_Map[rh_detid].end())
            cpFraction = findHitIt->fraction;
        }
        cpPair.second += std::min(std::pow(rhFraction - cpFraction, 2), std::pow(rhFraction, 2)) * hitEnergyWeight *
                         invLayerClusterEnergyWeight;
      }  //End of loop over CaloParticles related the this LayerCluster.
    }    // End of loop over Hits within a LayerCluster
#ifdef EDM_ML_DEBUG
    if (cpsInLayerCluster[lcId].empty())
      LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "layerCluster Id: \t" << lcId << "\tCP id:\t-1 "
                                                    << "\t score \t-1\n";
#endif
  }  // End of loop over LayerClusters

  // Compute the CaloParticle-To-LayerCluster score
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
      for (auto& lc : cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore) {
        if (lc.second.first > maxEnergyLCinCP) {
          maxEnergyLCinCP = lc.second.first;
          lcWithMaxEnergyInCP = lc.first;
        }
      }
      if (CPenergy > 0.f)
        CPEnergyFractionInLC = maxEnergyLCinCP / CPenergy;

      LogDebug("LCToCPAssociatorByEnergyScoreImpl")
          << std::setw(8) << "LayerId:\t" << std::setw(12) << "caloparticle\t" << std::setw(15) << "cp total energy\t"
          << std::setw(15) << "cpEnergyOnLayer\t" << std::setw(14) << "CPNhitsOnLayer\t" << std::setw(18)
          << "lcWithMaxEnergyInCP\t" << std::setw(15) << "maxEnergyLCinCP\t" << std::setw(20) << "CPEnergyFractionInLC"
          << "\n";
      LogDebug("LCToCPAssociatorByEnergyScoreImpl")
          << std::setw(8) << layerId << "\t" << std::setw(12) << cpId << "\t" << std::setw(15)
          << caloParticles[cpId].energy() << "\t" << std::setw(15) << CPenergy << "\t" << std::setw(14)
          << CPNumberOfHits << "\t" << std::setw(18) << lcWithMaxEnergyInCP << "\t" << std::setw(15) << maxEnergyLCinCP
          << "\t" << std::setw(20) << CPEnergyFractionInLC << "\n";
#endif
      // Compute the correct normalization. Observe that this is the sum of the squares.
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
        auto hit_find_in_LC = detIdToLayerClusterId_Map.find(cp_hitDetId);
        if (hit_find_in_LC == detIdToLayerClusterId_Map.end())
          hitWithNoLC = true;
        auto itcheck = hitMap_->find(cp_hitDetId);
        const HGCRecHit* hit = itcheck->second;
        float hitEnergyWeight = hit->energy() * hit->energy();
        for (auto& lcPair : cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore) {
          unsigned int layerClusterId = lcPair.first;
          float lcFraction = 0.f;

          if (!hitWithNoLC) {
            auto findHitIt = std::find(detIdToLayerClusterId_Map[cp_hitDetId].begin(),
                                       detIdToLayerClusterId_Map[cp_hitDetId].end(),
                                       hgcal::detIdInfoInCluster{layerClusterId, 0.f});
            if (findHitIt != detIdToLayerClusterId_Map[cp_hitDetId].end())
              lcFraction = findHitIt->fraction;
          }
          lcPair.second.second += std::min(std::pow(lcFraction - cpFraction, 2), std::pow(cpFraction, 2)) *
                                  hitEnergyWeight * invCPEnergyWeight;
#ifdef EDM_ML_DEBUG
          LogDebug("LCToCPAssociatorByEnergyScoreImpl")
              << "cpDetId:\t" << (uint32_t)cp_hitDetId << "\tlayerClusterId:\t" << layerClusterId << "\t"
              << "lcfraction,cpfraction:\t" << lcFraction << ", " << cpFraction << "\t"
              << "hitEnergyWeight:\t" << hitEnergyWeight << "\t"
              << "current score:\t" << lcPair.second.second << "\t"
              << "invCPEnergyWeight:\t" << invCPEnergyWeight << "\n";
#endif
        }  // End of loop over LayerClusters linked to hits of this CaloParticle
      }    // End of loop over hits of CaloParticle on a Layer
#ifdef EDM_ML_DEBUG
      if (cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore.empty())
        LogDebug("LCToCPAssociatorByEnergyScoreImpl") << "CP Id: \t" << cpId << "\tLC id:\t-1 "
                                                      << "\t score \t-1\n";

      for (const auto& lcPair : cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore) {
        LogDebug("LCToCPAssociatorByEnergyScoreImpl")
            << "CP Id: \t" << cpId << "\t LC id: \t" << lcPair.first << "\t score \t" << lcPair.second.second
            << "\t shared energy:\t" << lcPair.second.first << "\t shared energy fraction:\t"
            << (lcPair.second.first / CPenergy) << "\n";
      }
#endif
    }  // End of loop over layers
  }    // End of loop over CaloParticles

  return {cpsInLayerCluster, cPOnLayer};
}

hgcal::RecoToSimCollection LCToCPAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<CaloParticleCollection>& cPCH) const {
  hgcal::RecoToSimCollection returnValue(productGetter_);
  const auto& links = makeConnections(cCCH, cPCH);

  const auto& cpsInLayerCluster = std::get<0>(links);
  for (size_t lcId = 0; lcId < cpsInLayerCluster.size(); ++lcId) {
    for (auto& cpPair : cpsInLayerCluster[lcId]) {
      LogDebug("LCToCPAssociatorByEnergyScoreImpl")
          << "layerCluster Id: \t" << lcId << "\t CP id: \t" << cpPair.first << "\t score \t" << cpPair.second << "\n";
      // Fill AssociationMap
      returnValue.insert(edm::Ref<reco::CaloClusterCollection>(cCCH, lcId),  // Ref to LC
                         std::make_pair(edm::Ref<CaloParticleCollection>(cPCH, cpPair.first),
                                        cpPair.second)  // Pair <Ref to CP, score>
      );
    }
  }
  return returnValue;
}

hgcal::SimToRecoCollection LCToCPAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<CaloParticleCollection>& cPCH) const {
  hgcal::SimToRecoCollection returnValue(productGetter_);
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
  return returnValue;
}
