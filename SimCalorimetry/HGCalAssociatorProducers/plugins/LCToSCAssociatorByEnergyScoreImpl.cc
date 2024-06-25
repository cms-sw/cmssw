#include "LCToSCAssociatorByEnergyScoreImpl.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

template <typename HIT>
LCToSCAssociatorByEnergyScoreImpl<HIT>::LCToSCAssociatorByEnergyScoreImpl(
    edm::EDProductGetter const& productGetter,
    bool hardScatterOnly,
    std::shared_ptr<hgcal::RecHitTools> recHitTools,
    const std::unordered_map<DetId, const unsigned int>* hitMap,
    std::vector<const HIT*>& hits)
    : hardScatterOnly_(hardScatterOnly),
      recHitTools_(recHitTools),
      hitMap_(hitMap),
      productGetter_(&productGetter),
      hits_(hits) {
  if constexpr (std::is_same_v<HIT, HGCRecHit>)
    layers_ = recHitTools_->lastLayerBH();
  else
    layers_ = 6;  //EB + 4 HB + HO
}

template <typename HIT>
ticl::association LCToSCAssociatorByEnergyScoreImpl<HIT>::makeConnections(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<SimClusterCollection>& sCCH) const {
  // Get collections
  const auto& clusters = *cCCH.product();
  const auto& simClusters = *sCCH.product();
  auto nLayerClusters = clusters.size();

  //There shouldn't be any SimTracks from different crossings, but maybe they will be added later.
  //At the moment there should be one SimTrack in each SimCluster.
  auto nSimClusters = simClusters.size();
  std::vector<size_t> sCIndices;
  for (unsigned int scId = 0; scId < nSimClusters; ++scId) {
    if (hardScatterOnly_ && (simClusters[scId].g4Tracks()[0].eventId().event() != 0 or
                             simClusters[scId].g4Tracks()[0].eventId().bunchCrossing() != 0)) {
      LogDebug("LCToSCAssociatorByEnergyScoreImpl")
          << "Excluding SimCluster from event: " << simClusters[scId].g4Tracks()[0].eventId().event()
          << " with BX: " << simClusters[scId].g4Tracks()[0].eventId().bunchCrossing() << std::endl;
      continue;
    }
    sCIndices.emplace_back(scId);
  }
  nSimClusters = sCIndices.size();

  // Initialize lcsInSimCluster. It contains the simClusterOnLayer structure for all simClusters in each layer and
  // among other the information to compute the SimCluster-To-LayerCluster score. It is one of the two objects that
  // build the output of the makeConnections function.
  // lcsInSimCluster[scId][layerId]
  ticl::simClusterToLayerCluster lcsInSimCluster;
  lcsInSimCluster.resize(nSimClusters);
  for (unsigned int i = 0; i < nSimClusters; ++i) {
    lcsInSimCluster[i].resize(layers_ * 2);
    for (unsigned int j = 0; j < layers_ * 2; ++j) {
      lcsInSimCluster[i][j].simClusterId = i;
      lcsInSimCluster[i][j].energy = 0.f;
      lcsInSimCluster[i][j].hits_and_fractions.clear();
    }
  }

  // Fill detIdToSimClusterId_Map and update lcsInSimCluster
  // The detIdToSimClusterId_Map is used to connect a hit Detid (key) with all the SimClusters that
  // contributed to that hit by storing the SimCluster id and the fraction of the hit. Observe here
  // that in contrast to the CaloParticle case there is no merging and summing of the fractions, which
  // in the CaloParticle's case was necessary due to the multiple SimClusters of a single CaloParticle.
  std::unordered_map<DetId, std::vector<ticl::detIdInfoInCluster>> detIdToSimClusterId_Map;
  for (const auto& scId : sCIndices) {
    std::vector<std::pair<uint32_t, float>> hits_and_fractions = simClusters[scId].hits_and_fractions();
    if constexpr (std::is_same_v<HIT, HGCRecHit>)
      hits_and_fractions = simClusters[scId].endcap_hits_and_fractions();
    else
      hits_and_fractions = simClusters[scId].barrel_hits_and_fractions();
    for (const auto& it_haf : hits_and_fractions) {
      const auto hitid = (it_haf.first);
      unsigned int scLayerId = recHitTools_->getLayer(hitid);
      if constexpr (std::is_same_v<HIT, HGCRecHit>)
        scLayerId += layers_ * ((recHitTools_->zside(hitid) + 1) >> 1) - 1;
      const auto itcheck = hitMap_->find(hitid);
      if (itcheck != hitMap_->end()) {
        auto hit_find_it = detIdToSimClusterId_Map.find(hitid);
        if (hit_find_it == detIdToSimClusterId_Map.end()) {
          detIdToSimClusterId_Map[hitid] = std::vector<ticl::detIdInfoInCluster>();
        }
        detIdToSimClusterId_Map[hitid].emplace_back(scId, it_haf.second);
        const HIT* hit = hits_[itcheck->second];
        lcsInSimCluster[scId][scLayerId].energy += it_haf.second * hit->energy();
        lcsInSimCluster[scId][scLayerId].hits_and_fractions.emplace_back(hitid, it_haf.second);
      }
    }
  }  // end of loop over SimClusters

#ifdef EDM_ML_DEBUG
  LogDebug("LCToSCAssociatorByEnergyScoreImpl")
      << "lcsInSimCluster INFO (Only SimCluster filled at the moment)" << std::endl;
  LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "    # of clusters :          " << nLayerClusters << std::endl;
  for (size_t sc = 0; sc < lcsInSimCluster.size(); ++sc) {
    LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "For SimCluster Idx: " << sc << " we have: " << std::endl;
    for (size_t sclay = 0; sclay < lcsInSimCluster[sc].size(); ++sclay) {
      LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "  On Layer: " << sclay << " we have:" << std::endl;
      LogDebug("LCToSCAssociatorByEnergyScoreImpl")
          << "    SimClusterIdx: " << lcsInSimCluster[sc][sclay].simClusterId << std::endl;
      LogDebug("LCToSCAssociatorByEnergyScoreImpl")
          << "    Energy:          " << lcsInSimCluster[sc][sclay].energy << std::endl;
      double tot_energy = 0.;
      for (auto const& haf : lcsInSimCluster[sc][sclay].hits_and_fractions) {
        const HIT* hit = hits_[hitMap_->at(haf.first)];
        LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "      Hits/fraction/energy: " << (uint32_t)haf.first << "/"
                                                      << haf.second << "/" << haf.second * hit->energy() << std::endl;
        tot_energy += haf.second * hit->energy();
      }
      LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "    Tot Sum haf: " << tot_energy << std::endl;
      for (auto const& lc : lcsInSimCluster[sc][sclay].layerClusterIdToEnergyAndScore) {
        LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "      lcIdx/energy/score: " << lc.first << "/"
                                                      << lc.second.first << "/" << lc.second.second << std::endl;
      }
    }
  }

  LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "detIdToSimClusterId_Map INFO" << std::endl;
  for (auto const& sc : detIdToSimClusterId_Map) {
    LogDebug("LCToSCAssociatorByEnergyScoreImpl")
        << "For detId: " << (uint32_t)sc.first
        << " we have found the following connections with SimClusters:" << std::endl;
    // At this point here if you activate the printing you will notice cases where in a
    // specific detId there are more that one SimClusters contributing with fractions less than 1.
    // This is important since it effects the score computation, since the fraction is also in the
    // denominator of the score formula.
    const HIT* hit = hits_[hitMap_->at(sc.first)];
    for (auto const& sclu : sc.second) {
      LogDebug("LCToSCAssociatorByEnergyScoreImpl")
          << "  SimCluster Id: " << sclu.clusterId << " with fraction: " << sclu.fraction
          << " and energy: " << sclu.fraction * hit->energy() << std::endl;
    }
  }
#endif

  // Fill detIdToLayerClusterId_Map and scsInLayerCluster; update lcsInSimCluster
  // The detIdToLayerClusterId_Map is used to connect a hit Detid (key) with all the LayerClusters that
  // contributed to that hit by storing the LayerCluster id and the fraction of the corresponding hit.
  std::unordered_map<DetId, std::vector<ticl::detIdInfoInCluster>> detIdToLayerClusterId_Map;
  // scsInLayerCluster together with lcsInSimCluster are the two objects that are used to build the
  // output of the makeConnections function. scsInLayerCluster connects a LayerCluster with
  // all the SimClusters that share at least one cell with the LayerCluster and for each pair (LC,SC)
  // it stores the score.
  ticl::layerClusterToSimCluster scsInLayerCluster;  //[lcId][scId]->(score)
  scsInLayerCluster.resize(nLayerClusters);

  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    const std::vector<std::pair<DetId, float>>& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    const auto firstHitDetId = hits_and_fractions[0].first;
    int lcLayerId = recHitTools_->getLayer(firstHitDetId);
    if constexpr (std::is_same_v<HIT, HGCRecHit>)
      lcLayerId += layers_ * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;
    for (unsigned int hitId = 0; hitId < numberOfHitsInLC; hitId++) {
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
        const HIT* hit = hits_[itcheck->second];
        //Loops through all the simclusters that have the layer cluster rechit under study
        //Here is time to update the lcsInSimCluster and connect the SimCluster with all
        //the layer clusters that have the current rechit detid matched.
        for (auto& h : hit_find_in_SC->second) {
          //lcsInSimCluster[simclusterId][layerId][layerclusterId]-> (energy,score)
          //SC_i - > LC_j, LC_k, ...
          lcsInSimCluster[h.clusterId][lcLayerId].layerClusterIdToEnergyAndScore[lcId].first +=
              h.fraction * hit->energy();
          //LC_i -> SC_j, SC_k, ...
          scsInLayerCluster[lcId].emplace_back(h.clusterId, 0.f);
        }
      }
    }  // End loop over hits on a LayerCluster
  }    // End of loop over LayerClusters

#ifdef EDM_ML_DEBUG
  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    const auto& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    const auto firstHitDetId = hits_and_fractions[0].first;
    int lcLayerId = recHitTools_->getLayer(firstHitDetId);
    if constexpr (std::is_same_v<HIT, HGCRecHit>)
      lcLayerId += layers_ * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;
    // This vector will store, for each hit in the Layercluster, the index of
    // the SimCluster that contributed the most, in terms of energy, to it.
    // Special values are:
    //
    // -2  --> the reconstruction fraction of the RecHit is 0 (used in the past to monitor Halo Hits)
    // -3  --> same as before with the added condition that no SimCluster has been linked to this RecHit
    // -1  --> the reco fraction is >0, but no SimCluster has been linked to it
    // >=0 --> index of the linked SimCluster
    std::vector<int> hitsToSimClusterId(numberOfHitsInLC);
    // This will store the index of the SimCluster linked to the LayerCluster that has the most number of hits in common.
    int maxSCId_byNumberOfHits = -1;
    // This will store the maximum number of shared hits between a Layercluster and a SimCluster
    unsigned int maxSCNumberOfHitsInLC = 0;
    // This will store the index of the SimCluster linked to the LayerCluster that has the most energy in common.
    int maxSCId_byEnergy = -1;
    // This will store the maximum number of shared energy between a Layercluster and a SimCluster
    float maxEnergySharedLCandSC = 0.f;
    // This will store the fraction of the LayerCluster energy shared with the best(energy) SimCluster: e_shared/lc_energy
    float energyFractionOfLCinSC = 0.f;
    // This will store the fraction of the SimCluster energy shared with the LayerCluster: e_shared/sc_energy
    float energyFractionOfSCinLC = 0.f;
    std::unordered_map<unsigned, unsigned> occurrencesSCinLC;
    unsigned int numberOfNoiseHitsInLC = 0;
    std::unordered_map<unsigned, float> SCEnergyInLC;

    for (unsigned int hitId = 0; hitId < numberOfHitsInLC; hitId++) {
      const auto rh_detid = hits_and_fractions[hitId].first;
      const auto rhFraction = hits_and_fractions[hitId].second;

      auto hit_find_in_SC = detIdToSimClusterId_Map.find(rh_detid);

      // if the fraction is zero or the hit does not belong to any simcluster
      // set the caloparticleId for the hit to -1 this will
      // contribute to the number of noise hits

      // MR Remove the case in which the fraction is 0, since this could be a
      // real hit that has been marked as halo.
      if (rhFraction == 0.) {
        hitsToSimClusterId[hitId] = -2;
      }
      //Now check if there are SimClusters linked to this rechit of the layercluster
      if (hit_find_in_SC == detIdToSimClusterId_Map.end()) {
        hitsToSimClusterId[hitId] -= 1;
      } else {
        const auto itcheck = hitMap_->find(rh_detid);
        const HIT* hit = hits_[itcheck->second];
        auto maxSCEnergyInLC = 0.f;
        auto maxSCId = -1;
        //Loop through all the linked SimClusters
        for (auto& h : hit_find_in_SC->second) {
          SCEnergyInLC[h.clusterId] += h.fraction * hit->energy();
          // Keep track of which SimCluster ccontributed the most, in terms
          // of energy, to this specific LayerCluster.
          if (SCEnergyInLC[h.clusterId] > maxSCEnergyInLC) {
            maxSCEnergyInLC = SCEnergyInLC[h.clusterId];
            maxSCId = h.clusterId;
          }
        }
        hitsToSimClusterId[hitId] = maxSCId;
      }
    }  // End loop over hits on a LayerCluster

    for (const auto& c : hitsToSimClusterId) {
      if (c < 0) {
        numberOfNoiseHitsInLC++;
      } else {
        occurrencesSCinLC[c]++;
      }
    }

    for (const auto& c : occurrencesSCinLC) {
      if (c.second > maxSCNumberOfHitsInLC) {
        maxSCId_byNumberOfHits = c.first;
        maxSCNumberOfHitsInLC = c.second;
      }
    }

    for (const auto& c : SCEnergyInLC) {
      if (c.second > maxEnergySharedLCandSC) {
        maxSCId_byEnergy = c.first;
        maxEnergySharedLCandSC = c.second;
      }
    }

    float totalSCEnergyOnLayer = 0.f;
    if (maxSCId_byEnergy >= 0) {
      totalSCEnergyOnLayer = lcsInSimCluster[maxSCId_byEnergy][lcLayerId].energy;
      energyFractionOfSCinLC = maxEnergySharedLCandSC / totalSCEnergyOnLayer;
      if (clusters[lcId].energy() > 0.f) {
        energyFractionOfLCinSC = maxEnergySharedLCandSC / clusters[lcId].energy();
      }
    }

    LogDebug("LCToSCAssociatorByEnergyScoreImpl") << std::setw(10) << "LayerId:"
                                                  << "\t" << std::setw(12) << "layerCluster"
                                                  << "\t" << std::setw(10) << "lc energy"
                                                  << "\t" << std::setw(5) << "nhits"
                                                  << "\t" << std::setw(12) << "noise hits"
                                                  << "\t" << std::setw(22) << "maxSCId_byNumberOfHits"
                                                  << "\t" << std::setw(8) << "nhitsSC"
                                                  << "\t" << std::setw(13) << "maxSCId_byEnergy"
                                                  << "\t" << std::setw(20) << "maxEnergySharedLCandSC"
                                                  << "\t" << std::setw(22) << "totalSCEnergyOnLayer"
                                                  << "\t" << std::setw(22) << "energyFractionOfLCinSC"
                                                  << "\t" << std::setw(25) << "energyFractionOfSCinLC"
                                                  << "\t"
                                                  << "\n";
    LogDebug("LCToSCAssociatorByEnergyScoreImpl")
        << std::setw(10) << lcLayerId << "\t" << std::setw(12) << lcId << "\t" << std::setw(10)
        << clusters[lcId].energy() << "\t" << std::setw(5) << numberOfHitsInLC << "\t" << std::setw(12)
        << numberOfNoiseHitsInLC << "\t" << std::setw(22) << maxSCId_byNumberOfHits << "\t" << std::setw(8)
        << maxSCNumberOfHitsInLC << "\t" << std::setw(13) << maxSCId_byEnergy << "\t" << std::setw(20)
        << maxEnergySharedLCandSC << "\t" << std::setw(22) << totalSCEnergyOnLayer << "\t" << std::setw(22)
        << energyFractionOfLCinSC << "\t" << std::setw(25) << energyFractionOfSCinLC << "\n";
  }  // End of loop over LayerClusters

  LogDebug("LCToSCAssociatorByEnergyScoreImpl")
      << "Improved lcsInSimCluster INFO (Now containing the linked layer clusters id and energy - score still empty)"
      << std::endl;
  for (size_t sc = 0; sc < lcsInSimCluster.size(); ++sc) {
    LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "For SimCluster Idx: " << sc << " we have: " << std::endl;
    for (size_t sclay = 0; sclay < lcsInSimCluster[sc].size(); ++sclay) {
      LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "  On Layer: " << sclay << " we have:" << std::endl;
      LogDebug("LCToSCAssociatorByEnergyScoreImpl")
          << "    SimClusterIdx: " << lcsInSimCluster[sc][sclay].simClusterId << std::endl;
      LogDebug("LCToSCAssociatorByEnergyScoreImpl")
          << "    Energy:          " << lcsInSimCluster[sc][sclay].energy << std::endl;
      double tot_energy = 0.;
      for (auto const& haf : lcsInSimCluster[sc][sclay].hits_and_fractions) {
        const HIT* hit = hits_[hitMap_->at(haf.first)];
        LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "      Hits/fraction/energy: " << (uint32_t)haf.first << "/"
                                                      << haf.second << "/" << haf.second * hit->energy() << std::endl;
        tot_energy += haf.second * hit->energy();
      }
      LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "    Tot Sum haf: " << tot_energy << std::endl;
      for (auto const& lc : lcsInSimCluster[sc][sclay].layerClusterIdToEnergyAndScore) {
        LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "      lcIdx/energy/score: " << lc.first << "/"
                                                      << lc.second.first << "/" << lc.second.second << std::endl;
      }
    }
  }

  LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "Improved detIdToSimClusterId_Map INFO" << std::endl;
  for (auto const& sc : detIdToSimClusterId_Map) {
    const HIT* hit = hits_[hitMap_->at(sc.first)];
    LogDebug("LCToSCAssociatorByEnergyScoreImpl")
        << "For detId: " << (uint32_t)sc.first
        << " we have found the following connections with SimClusters:" << std::endl;
    for (auto const& sclu : sc.second) {
      LogDebug("LCToSCAssociatorByEnergyScoreImpl")
          << "  SimCluster Id: " << sclu.clusterId << " with fraction: " << sclu.fraction
          << " and energy: " << sclu.fraction * hit->energy() << std::endl;
    }
  }
#endif

  // Update scsInLayerCluster; compute the score LayerCluster-to-SimCluster,
  // together with the returned AssociationMap
  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    // The simclusters contributing to the layer clusters should already be unique.
    // find the unique simclusters id contributing to the layer clusters
    std::sort(scsInLayerCluster[lcId].begin(), scsInLayerCluster[lcId].end());
    auto last = std::unique(scsInLayerCluster[lcId].begin(), scsInLayerCluster[lcId].end());
    scsInLayerCluster[lcId].erase(last, scsInLayerCluster[lcId].end());
    const auto& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    // If a reconstructed LayerCluster has energy 0 but is linked to a
    // SimCluster, assigned score 1
    if (clusters[lcId].energy() == 0. && !scsInLayerCluster[lcId].empty()) {
      for (auto& scPair : scsInLayerCluster[lcId]) {
        scPair.second = 1.;
        LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "layerClusterId : \t " << lcId << "\t SC id : \t"
                                                      << scPair.first << "\t score \t " << scPair.second << "\n";
      }
      continue;
    }

    // Compute the correct normalization.
    // It is the inverse of the denominator of the LCToSC score formula. Observe that this is the sum of the squares.
    float invLayerClusterEnergyWeight = 0.f;
    for (auto const& haf : hits_and_fractions) {
      const HIT* hit = hits_[hitMap_->at(haf.first)];
      invLayerClusterEnergyWeight += (haf.second * hit->energy()) * (haf.second * hit->energy());
    }
    invLayerClusterEnergyWeight = 1.f / invLayerClusterEnergyWeight;
    for (unsigned int i = 0; i < numberOfHitsInLC; ++i) {
      DetId rh_detid = hits_and_fractions[i].first;
      float rhFraction = hits_and_fractions[i].second;

      bool hitWithSC = (detIdToSimClusterId_Map.find(rh_detid) != detIdToSimClusterId_Map.end());

      auto itcheck = hitMap_->find(rh_detid);
      const HIT* hit = hits_[itcheck->second];
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
#ifdef EDM_ML_DEBUG
        LogDebug("LCToSCAssociatorByEnergyScoreImpl")
            << "rh_detid:\t" << (uint32_t)rh_detid << "\tlayerClusterId:\t" << lcId << "\t"
            << "rhfraction,scfraction:\t" << rhFraction << ", " << scFraction << "\t"
            << "hitEnergyWeight:\t" << hitEnergyWeight << "\t"
            << "current score:\t" << scPair.second << "\t"
            << "invLayerClusterEnergyWeight:\t" << invLayerClusterEnergyWeight << "\n";
#endif
      }
    }  // End of loop over Hits within a LayerCluster
#ifdef EDM_ML_DEBUG
    if (scsInLayerCluster[lcId].empty())
      LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "layerCluster Id: \t" << lcId << "\tSC id:\t-1 "
                                                    << "\t score \t-1"
                                                    << "\n";
#endif
  }  // End of loop over LayerClusters

  // Compute the SimCluster-To-LayerCluster score
  for (const auto& scId : sCIndices) {
    for (unsigned int layerId = 0; layerId < layers_ * 2; ++layerId) {
      unsigned int SCNumberOfHits = lcsInSimCluster[scId][layerId].hits_and_fractions.size();
      if (SCNumberOfHits == 0)
        continue;
#ifdef EDM_ML_DEBUG
      int lcWithMaxEnergyInSC = -1;
      //energy of the most energetic LC from all that were linked to SC
      float maxEnergyLCinSC = 0.f;
      //Energy of the SC scId on layer layerId that was reconstructed.
      float SCenergy = lcsInSimCluster[scId][layerId].energy;
      //most energetic LC from all LCs linked to SC over SC energy.
      float SCEnergyFractionInLC = 0.f;
      for (auto& lc : lcsInSimCluster[scId][layerId].layerClusterIdToEnergyAndScore) {
        if (lc.second.first > maxEnergyLCinSC) {
          maxEnergyLCinSC = lc.second.first;
          lcWithMaxEnergyInSC = lc.first;
        }
      }
      if (SCenergy > 0.f)
        SCEnergyFractionInLC = maxEnergyLCinSC / SCenergy;

      LogDebug("LCToSCAssociatorByEnergyScoreImpl")
          << std::setw(8) << "LayerId:\t" << std::setw(12) << "simcluster\t" << std::setw(15) << "sc total energy\t"
          << std::setw(15) << "scEnergyOnLayer\t" << std::setw(14) << "SCNhitsOnLayer\t" << std::setw(18)
          << "lcWithMaxEnergyInSC\t" << std::setw(15) << "maxEnergyLCinSC\t" << std::setw(20) << "SCEnergyFractionInLC"
          << "\n";
      LogDebug("LCToSCAssociatorByEnergyScoreImpl")
          << std::setw(8) << layerId << "\t" << std::setw(12) << scId << "\t" << std::setw(15)
          << simClusters[scId].energy() << "\t" << std::setw(15) << SCenergy << "\t" << std::setw(14) << SCNumberOfHits
          << "\t" << std::setw(18) << lcWithMaxEnergyInSC << "\t" << std::setw(15) << maxEnergyLCinSC << "\t"
          << std::setw(20) << SCEnergyFractionInLC << "\n";
#endif
      // Compute the correct normalization. Observe that this is the sum of the squares.
      float invSCEnergyWeight = 0.f;
      for (auto const& haf : lcsInSimCluster[scId][layerId].hits_and_fractions) {
        const HIT* hit = hits_[hitMap_->at(haf.first)];
        invSCEnergyWeight += std::pow(haf.second * hit->energy(), 2);
      }
      invSCEnergyWeight = 1.f / invSCEnergyWeight;
      for (unsigned int i = 0; i < SCNumberOfHits; ++i) {
        auto& sc_hitDetId = lcsInSimCluster[scId][layerId].hits_and_fractions[i].first;
        auto& scFraction = lcsInSimCluster[scId][layerId].hits_and_fractions[i].second;

        bool hitWithLC = false;
        if (scFraction == 0.f)
          continue;  //hopefully this should never happen
        auto hit_find_in_LC = detIdToLayerClusterId_Map.find(sc_hitDetId);
        if (hit_find_in_LC != detIdToLayerClusterId_Map.end())
          hitWithLC = true;
        auto itcheck = hitMap_->find(sc_hitDetId);
        const HIT* hit = hits_[itcheck->second];
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
#ifdef EDM_ML_DEBUG
          LogDebug("LCToSCAssociatorByEnergyScoreImpl")
              << "scDetId:\t" << (uint32_t)sc_hitDetId << "\tlayerClusterId:\t" << layerClusterId << "\t"
              << "lcfraction,scfraction:\t" << lcFraction << ", " << scFraction << "\t"
              << "hitEnergyWeight:\t" << hitEnergyWeight << "\t"
              << "current score:\t" << lcPair.second.second << "\t"
              << "invSCEnergyWeight:\t" << invSCEnergyWeight << "\n";
#endif
        }  // End of loop over LayerClusters linked to hits of this SimCluster
      }    // End of loop over hits of SimCluster on a Layer
#ifdef EDM_ML_DEBUG
      if (lcsInSimCluster[scId][layerId].layerClusterIdToEnergyAndScore.empty())
        LogDebug("LCToSCAssociatorByEnergyScoreImpl") << "SC Id: \t" << scId << "\tLC id:\t-1 "
                                                      << "\t score \t-1"
                                                      << "\n";

      for (const auto& lcPair : lcsInSimCluster[scId][layerId].layerClusterIdToEnergyAndScore) {
        LogDebug("LCToSCAssociatorByEnergyScoreImpl")
            << "SC Id: \t" << scId << "\t LC id: \t" << lcPair.first << "\t score \t" << lcPair.second.second
            << "\t shared energy:\t" << lcPair.second.first << "\t shared energy fraction:\t"
            << (lcPair.second.first / SCenergy) << "\n";
      }
#endif
    }  // End of loop over layers
  }    // End of loop over SimClusters

  return {scsInLayerCluster, lcsInSimCluster};
}

template <typename HIT>
ticl::RecoToSimCollectionWithSimClusters LCToSCAssociatorByEnergyScoreImpl<HIT>::associateRecoToSim(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<SimClusterCollection>& sCCH) const {
  ticl::RecoToSimCollectionWithSimClusters returnValue(productGetter_);
  const auto& links = makeConnections(cCCH, sCCH);

  const auto& scsInLayerCluster = std::get<0>(links);
  for (size_t lcId = 0; lcId < scsInLayerCluster.size(); ++lcId) {
    for (auto& scPair : scsInLayerCluster[lcId]) {
      LogDebug("LCToSCAssociatorByEnergyScoreImpl")
          << "layerCluster Id: \t" << lcId << "\t SC id: \t" << scPair.first << "\t score \t" << scPair.second << "\n";
      // Fill AssociationMap
      returnValue.insert(edm::Ref<reco::CaloClusterCollection>(cCCH, lcId),  // Ref to LC
                         std::make_pair(edm::Ref<SimClusterCollection>(sCCH, scPair.first),
                                        scPair.second)  // Pair <Ref to SC, score>
      );
    }
  }
  return returnValue;
}

template <typename HIT>
ticl::SimToRecoCollectionWithSimClusters LCToSCAssociatorByEnergyScoreImpl<HIT>::associateSimToReco(
    const edm::Handle<reco::CaloClusterCollection>& cCCH, const edm::Handle<SimClusterCollection>& sCCH) const {
  ticl::SimToRecoCollectionWithSimClusters returnValue(productGetter_);
  const auto& links = makeConnections(cCCH, sCCH);
  const auto& lcsInSimCluster = std::get<1>(links);
  for (size_t scId = 0; scId < lcsInSimCluster.size(); ++scId) {
    for (size_t layerId = 0; layerId < lcsInSimCluster[scId].size(); ++layerId) {
      for (auto& lcPair : lcsInSimCluster[scId][layerId].layerClusterIdToEnergyAndScore) {
        returnValue.insert(
            edm::Ref<SimClusterCollection>(sCCH, scId),                                // Ref to SC
            std::make_pair(edm::Ref<reco::CaloClusterCollection>(cCCH, lcPair.first),  // Pair <Ref to LC,
                           std::make_pair(lcPair.second.first, lcPair.second.second))  // pair <energy, score> >
        );
      }
    }
  }
  return returnValue;
}

template class LCToSCAssociatorByEnergyScoreImpl<HGCRecHit>;
template class LCToSCAssociatorByEnergyScoreImpl<reco::PFRecHit>;
