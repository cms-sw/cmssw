#include "TSToSCAssociatorByEnergyScoreImpl.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

TSToSCAssociatorByEnergyScoreImpl::TSToSCAssociatorByEnergyScoreImpl(
    edm::EDProductGetter const& productGetter,
    bool hardScatterOnly,
    std::shared_ptr<hgcal::RecHitTools> recHitTools,
    const std::unordered_map<DetId, const HGCRecHit*>* hitMap)
    : hardScatterOnly_(hardScatterOnly), recHitTools_(recHitTools), hitMap_(hitMap), productGetter_(&productGetter) {
  layers_ = recHitTools_->lastLayerBH();
}

hgcal::association TSToSCAssociatorByEnergyScoreImpl::makeConnections(
    const edm::Handle<ticl::TracksterCollection>& tCH,
    const edm::Handle<reco::CaloClusterCollection>& lCCH,
    const edm::Handle<SimClusterCollection>& sCCH) const {
  // Get collections
  const auto& tracksters = *tCH.product();
  const auto& layerClusters = *lCCH.product();
  const auto& simClusters = *sCCH.product();
  auto nTracksters = tracksters.size();

  //There shouldn't be any SimTracks from different crossings, but maybe they will be added later.
  //At the moment there should be one SimTrack in each SimCluster.
  auto nSimClusters = simClusters.size();
  std::vector<size_t> sCIndices;
  for (unsigned int scId = 0; scId < nSimClusters; ++scId) {
    if (hardScatterOnly_ && (simClusters[scId].g4Tracks()[0].eventId().event() != 0 or
                             simClusters[scId].g4Tracks()[0].eventId().bunchCrossing() != 0)) {
      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << "Excluding SimCluster from event: " << simClusters[scId].g4Tracks()[0].eventId().event()
          << " with BX: " << simClusters[scId].g4Tracks()[0].eventId().bunchCrossing() << std::endl;
      continue;
    }
    sCIndices.emplace_back(scId);
  }
  nSimClusters = sCIndices.size();

  // Initialize tssInSimCluster. To be returned outside, since it contains the
  // information to compute the SimCluster-To-Trackster score.
  // tssInSimCluster[scId]:
  hgcal::simClusterToTrackster tssInSimCluster;
  tssInSimCluster.resize(nSimClusters);
  for (unsigned int i = 0; i < nSimClusters; ++i) {
    tssInSimCluster[i].simClusterId = i;
    tssInSimCluster[i].energy = 0.f;
    tssInSimCluster[i].hits_and_fractions.clear();
  }

  // Fill detIdToSimClusterId_Map and update tssInSimCluster
  std::unordered_map<DetId, std::vector<hgcal::detIdInfoInCluster>> detIdToSimClusterId_Map;
  for (const auto& scId : sCIndices) {
    const auto& hits_and_fractions = simClusters[scId].hits_and_fractions();
    for (const auto& it_haf : hits_and_fractions) {
      const auto hitid = it_haf.first;
      const auto itcheck = hitMap_->find(hitid);
      if (itcheck != hitMap_->end()) {
        const auto hit_find_it = detIdToSimClusterId_Map.find(hitid);
        if (hit_find_it == detIdToSimClusterId_Map.end()) {
          detIdToSimClusterId_Map[hitid] = std::vector<hgcal::detIdInfoInCluster>();
        }
        detIdToSimClusterId_Map[hitid].emplace_back(scId, it_haf.second);

        const HGCRecHit* hit = itcheck->second;
        tssInSimCluster[scId].energy += it_haf.second * hit->energy();
        tssInSimCluster[scId].hits_and_fractions.emplace_back(hitid, it_haf.second);
      }
    }
  }  // end of loop over SimClusters

#ifdef EDM_ML_DEBUG
  LogDebug("TSToSCAssociatorByEnergyScoreImpl")
      << "tssInSimCluster INFO (Only SimCluster filled at the moment)" << std::endl;
  for (size_t sc = 0; sc < tssInSimCluster.size(); ++sc) {
    LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "For SimCluster Idx: " << sc << " we have: " << std::endl;
    LogDebug("TSToSCAssociatorByEnergyScoreImpl")
        << "\tSimClusterIdx:\t" << tssInSimCluster[sc].simClusterId << std::endl;
    LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "\tEnergy:\t" << tssInSimCluster[sc].energy << std::endl;
    LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "\t# of clusters:\t" << layerClusters.size() << std::endl;
    double tot_energy = 0.;
    for (auto const& haf : tssInSimCluster[sc].hits_and_fractions) {
      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << "\tHits/fraction/energy: " << (uint32_t)haf.first << "/" << haf.second << "/"
          << haf.second * hitMap_->at(haf.first)->energy() << std::endl;
      tot_energy += haf.second * hitMap_->at(haf.first)->energy();
    }
    LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "\tTot Sum haf: " << tot_energy << std::endl;
    for (auto const& ts : tssInSimCluster[sc].tracksterIdToEnergyAndScore) {
      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << "\ttsIdx/energy/score: " << ts.first << "/" << ts.second.first << "/" << ts.second.second << std::endl;
    }
  }

  LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "detIdToSimClusterId_Map INFO" << std::endl;
  for (auto const& detId : detIdToSimClusterId_Map) {
    LogDebug("TSToSCAssociatorByEnergyScoreImpl")
        << "For detId: " << (uint32_t)detId.first
        << " we have found the following connections with SimClusters:" << std::endl;
    for (auto const& sc : detId.second) {
      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << "\tSimCluster Id: " << sc.clusterId << " with fraction: " << sc.fraction
          << " and energy: " << sc.fraction * hitMap_->at(detId.first)->energy() << std::endl;
    }
  }
#endif

  // Fill detIdToLayerClusterId_Map and scsInTrackster; update tssInSimCluster
  std::unordered_map<DetId, std::vector<hgcal::detIdInfoInCluster>> detIdToLayerClusterId_Map;
  // this contains the ids of the simclusters contributing with at least one
  // hit to the Trackster. To be returned since this contains the information
  // to compute the Trackster-To-SimCluster score.
  hgcal::tracksterToSimCluster scsInTrackster;  //[tsId][scId]->(energy,score)
  scsInTrackster.resize(nTracksters);

  for (unsigned int tsId = 0; tsId < nTracksters; ++tsId) {
    for (unsigned int i = 0; i < tracksters[tsId].vertices().size(); ++i) {
      const auto lcId = tracksters[tsId].vertices(i);
      const auto lcFractionInTs = 1.f / tracksters[tsId].vertex_multiplicity(i);

      const std::vector<std::pair<DetId, float>>& hits_and_fractions = layerClusters[lcId].hitsAndFractions();
      unsigned int numberOfHitsInLC = hits_and_fractions.size();

      for (unsigned int hitId = 0; hitId < numberOfHitsInLC; hitId++) {
        const auto rh_detid = hits_and_fractions[hitId].first;
        const auto rhFraction = hits_and_fractions[hitId].second;

        const auto hit_find_in_LC = detIdToLayerClusterId_Map.find(rh_detid);
        if (hit_find_in_LC == detIdToLayerClusterId_Map.end()) {
          detIdToLayerClusterId_Map[rh_detid] = std::vector<hgcal::detIdInfoInCluster>();
        }
        detIdToLayerClusterId_Map[rh_detid].emplace_back(lcId, rhFraction);

        const auto hit_find_in_SC = detIdToSimClusterId_Map.find(rh_detid);

        if (hit_find_in_SC != detIdToSimClusterId_Map.end()) {
          const auto itcheck = hitMap_->find(rh_detid);
          const HGCRecHit* hit = itcheck->second;
          //Loops through all the simclusters that have the layer cluster rechit under study
          //Here is time to update the tssInSimCluster and connect the SimCluster with all
          //the Tracksters that have the current rechit detid matched.
          for (const auto& h : hit_find_in_SC->second) {
            //tssInSimCluster[simclusterId][layerclusterId]-> (energy,score)
            //SC_i - > TS_j, TS_k, ...
            tssInSimCluster[h.clusterId].tracksterIdToEnergyAndScore[tsId].first +=
                lcFractionInTs * h.fraction * hit->energy();
            //TS_i -> SC_j, SC_k, ...
            scsInTrackster[tsId].emplace_back(h.clusterId, 0.f);
          }
        }
      }  // End loop over hits on a LayerCluster
    }    // End loop over LayerClusters in Trackster
  }      // End of loop over Tracksters

#ifdef EDM_ML_DEBUG
  for (unsigned int tsId = 0; tsId < nTracksters; ++tsId) {
    for (const auto& lcId : tracksters[tsId].vertices()) {
      const auto& hits_and_fractions = layerClusters[lcId].hitsAndFractions();
      unsigned int numberOfHitsInLC = hits_and_fractions.size();

      // This vector will store, for each hit in the Layercluster, the index of
      // the SimCluster that contributed the most, in terms of energy, to it.
      // Special values are:
      //
      // -2  --> the reconstruction fraction of the RecHit is 0 (used in the past to monitor Halo Hits)
      // -3  --> same as before with the added condition that no SimCluster has been linked to this RecHit
      // -1  --> the reco fraction is >0, but no SimCluster has been linked to it
      // >=0 --> index of the linked SimCluster
      std::vector<int> hitsToSimClusterId(numberOfHitsInLC);
      // This will store the index of the SimCluster linked to the LayerCluster that has the largest number of hits in common.
      int maxSCId_byNumberOfHits = -1;
      // This will store the maximum number of shared hits between a LayerCluster and a SimCluster
      unsigned int maxSCNumberOfHitsInLC = 0;
      // This will store the index of the SimCluster linked to the LayerCluster that has the largest energy in common.
      int maxSCId_byEnergy = -1;
      // This will store the maximum number of shared energy between a LayerCluster and a SimCluster
      float maxEnergySharedLCandSC = 0.f;
      // This will store the fraction of the LayerCluster energy shared with the best(energy) SimCluster: e_shared/lc_energy
      float energyFractionOfLCinSC = 0.f;
      // This will store the fraction of the SimCluster energy shared with the Trackster: e_shared/sc_energy
      float energyFractionOfSCinLC = 0.f;
      std::unordered_map<unsigned, unsigned> occurrencesSCinLC;
      unsigned int numberOfNoiseHitsInLC = 0;
      std::unordered_map<unsigned, float> SCEnergyInLC;

      for (unsigned int hitId = 0; hitId < numberOfHitsInLC; hitId++) {
        const auto rh_detid = hits_and_fractions[hitId].first;
        const auto rhFraction = hits_and_fractions[hitId].second;

        const auto hit_find_in_SC = detIdToSimClusterId_Map.find(rh_detid);

        // if the fraction is zero or the hit does not belong to any SimCluster,
        // set the SimCluster Id for the hit to -1; this will
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
          const HGCRecHit* hit = itcheck->second;
          auto maxSCEnergyInLC = 0.f;
          auto maxSCId = -1;
          //Loop through all the linked SimClusters
          for (const auto& h : hit_find_in_SC->second) {
            SCEnergyInLC[h.clusterId] += h.fraction * hit->energy();
            // Keep track of which SimCluster contributed the most, in terms
            // of energy, to this specific Layer Cluster.
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
        totalSCEnergyOnLayer = tssInSimCluster[maxSCId_byEnergy].energy;
        energyFractionOfSCinLC = maxEnergySharedLCandSC / totalSCEnergyOnLayer;
        if (tracksters[tsId].raw_energy() > 0.f) {
          energyFractionOfLCinSC = maxEnergySharedLCandSC / tracksters[tsId].raw_energy();
        }
      }

      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << std::setw(12) << "TracksterID:\t" << std::setw(12) << "layerCluster\t" << std::setw(10) << "lc energy\t"
          << std::setw(5) << "nhits\t" << std::setw(12) << "noise hits\t" << std::setw(22) << "maxSCId_byNumberOfHits\t"
          << std::setw(8) << "nhitsSC\t" << std::setw(13) << "maxSCId_byEnergy\t" << std::setw(20)
          << "maxEnergySharedLCandSC\t" << std::setw(22) << "totalSCEnergyOnLayer\t" << std::setw(22)
          << "energyFractionOfLCinSC\t" << std::setw(25) << "energyFractionOfSCinLC\t"
          << "\n";
      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << std::setw(12) << tsId << "\t" << std::setw(12) << lcId << "\t" << std::setw(10)
          << tracksters[tsId].raw_energy() << "\t" << std::setw(5) << numberOfHitsInLC << "\t" << std::setw(12)
          << numberOfNoiseHitsInLC << "\t" << std::setw(22) << maxSCId_byNumberOfHits << "\t" << std::setw(8)
          << maxSCNumberOfHitsInLC << "\t" << std::setw(13) << maxSCId_byEnergy << "\t" << std::setw(20)
          << maxEnergySharedLCandSC << "\t" << std::setw(22) << totalSCEnergyOnLayer << "\t" << std::setw(22)
          << energyFractionOfLCinSC << "\t" << std::setw(25) << energyFractionOfSCinLC << "\n";
    }  // End of loop over LayerClusters in Trackster
  }    // End of loop over Tracksters

  LogDebug("TSToSCAssociatorByEnergyScoreImpl")
      << "Improved tssInSimCluster INFO (Now containing the linked tracksters id and energy - score still empty)"
      << std::endl;
  for (size_t sc = 0; sc < tssInSimCluster.size(); ++sc) {
    LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "For SimCluster Idx: " << sc << " we have: " << std::endl;
    LogDebug("TSToSCAssociatorByEnergyScoreImpl")
        << "    SimClusterIdx: " << tssInSimCluster[sc].simClusterId << std::endl;
    LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "\tEnergy:\t" << tssInSimCluster[sc].energy << std::endl;
    double tot_energy = 0.;
    for (auto const& haf : tssInSimCluster[sc].hits_and_fractions) {
      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << "\tHits/fraction/energy: " << (uint32_t)haf.first << "/" << haf.second << "/"
          << haf.second * hitMap_->at(haf.first)->energy() << std::endl;
      tot_energy += haf.second * hitMap_->at(haf.first)->energy();
    }
    LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "\tTot Sum haf: " << tot_energy << std::endl;
    for (auto const& ts : tssInSimCluster[sc].tracksterIdToEnergyAndScore) {
      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << "\ttsIdx/energy/score: " << ts.first << "/" << ts.second.first << "/" << ts.second.second << std::endl;
    }
  }

  LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "Improved detIdToSimClusterId_Map INFO" << std::endl;
  for (auto const& sc : detIdToSimClusterId_Map) {
    LogDebug("TSToSCAssociatorByEnergyScoreImpl")
        << "For detId: " << (uint32_t)sc.first
        << " we have found the following connections with SimClusters:" << std::endl;
    for (auto const& sclu : sc.second) {
      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << "  SimCluster Id: " << sclu.clusterId << " with fraction: " << sclu.fraction
          << " and energy: " << sclu.fraction * hitMap_->at(sc.first)->energy() << std::endl;
    }
  }
#endif

  // Update scsInTrackster; compute the score Trackster-to-SimCluster,
  // together with the returned AssociationMap
  for (unsigned int tsId = 0; tsId < nTracksters; ++tsId) {
    // The SimClusters contributing to the Trackster's LayerClusters should already be unique.
    // find the unique SimClusters id contributing to the Trackster's LayerClusters
    std::sort(scsInTrackster[tsId].begin(), scsInTrackster[tsId].end());
    auto last = std::unique(scsInTrackster[tsId].begin(), scsInTrackster[tsId].end());
    scsInTrackster[tsId].erase(last, scsInTrackster[tsId].end());

    // If a reconstructed Trackster has energy 0 but is linked to a
    // SimCluster, assigned score 1
    if (tracksters[tsId].raw_energy() == 0. && !scsInTrackster[tsId].empty()) {
      for (auto& scPair : scsInTrackster[tsId]) {
        scPair.second = 1.;
        LogDebug("TSToSCAssociatorByEnergyScoreImpl")
            << "TracksterId:\t " << tsId << "\tSC id:\t" << scPair.first << "\tscore\t " << scPair.second << "\n";
      }
      continue;
    }

    float invTracksterEnergyWeight = 0.f;
    for (unsigned int i = 0; i < tracksters[tsId].vertices().size(); ++i) {
      const auto lcId = tracksters[tsId].vertices(i);
      const auto lcFractionInTs = 1.f / tracksters[tsId].vertex_multiplicity(i);

      const auto& hits_and_fractions = layerClusters[lcId].hitsAndFractions();
      // Compute the correct normalization
      for (auto const& haf : hits_and_fractions) {
        invTracksterEnergyWeight += (lcFractionInTs * haf.second * hitMap_->at(haf.first)->energy()) *
                                    (lcFractionInTs * haf.second * hitMap_->at(haf.first)->energy());
      }
    }
    invTracksterEnergyWeight = 1.f / invTracksterEnergyWeight;

    for (unsigned int i = 0; i < tracksters[tsId].vertices().size(); ++i) {
      const auto lcId = tracksters[tsId].vertices(i);
      const auto lcFractionInTs = 1.f / tracksters[tsId].vertex_multiplicity(i);

      const auto& hits_and_fractions = layerClusters[lcId].hitsAndFractions();
      unsigned int numberOfHitsInLC = hits_and_fractions.size();
      for (unsigned int i = 0; i < numberOfHitsInLC; ++i) {
        DetId rh_detid = hits_and_fractions[i].first;
        float rhFraction = hits_and_fractions[i].second * lcFractionInTs;

        const bool hitWithSC = (detIdToSimClusterId_Map.find(rh_detid) != detIdToSimClusterId_Map.end());

        const auto itcheck = hitMap_->find(rh_detid);
        const HGCRecHit* hit = itcheck->second;
        float hitEnergyWeight = hit->energy() * hit->energy();

        for (auto& scPair : scsInTrackster[tsId]) {
          float scFraction = 0.f;
          if (hitWithSC) {
            const auto findHitIt = std::find(detIdToSimClusterId_Map[rh_detid].begin(),
                                             detIdToSimClusterId_Map[rh_detid].end(),
                                             hgcal::detIdInfoInCluster{scPair.first, 0.f});
            if (findHitIt != detIdToSimClusterId_Map[rh_detid].end())
              scFraction = findHitIt->fraction;
          }
          scPair.second +=
              (rhFraction - scFraction) * (rhFraction - scFraction) * hitEnergyWeight * invTracksterEnergyWeight;
#ifdef EDM_ML_DEBUG
          LogDebug("TSToSCAssociatorByEnergyScoreImpl")
              << "rh_detid:\t" << (uint32_t)rh_detid << "\ttracksterId:\t" << tsId << "\t"
              << "rhfraction,scFraction:\t" << rhFraction << ", " << scFraction << "\t"
              << "hitEnergyWeight:\t" << hitEnergyWeight << "\t"
              << "current score:\t" << scPair.second << "\t"
              << "invTracksterEnergyWeight:\t" << invTracksterEnergyWeight << "\n";
#endif
        }
      }  // End of loop over Hits within a LayerCluster
    }    // End of loop over LayerClusters in Trackster

#ifdef EDM_ML_DEBUG
    if (scsInTrackster[tsId].empty())
      LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "trackster Id:\t" << tsId << "\tSC id:\t-1"
                                                    << "\tscore\t-1\n";
#endif
  }  // End of loop over Tracksters

  // Compute the SimCluster-To-Trackster score
  for (const auto& scId : sCIndices) {
    float invSCEnergyWeight = 0.f;

    const unsigned int SCNumberOfHits = tssInSimCluster[scId].hits_and_fractions.size();
    if (SCNumberOfHits == 0)
      continue;
#ifdef EDM_ML_DEBUG
    int tsWithMaxEnergyInSC = -1;
    //energy of the most energetic TS from all that were linked to SC
    float maxEnergyTSinSC = 0.f;
    float SCenergy = tssInSimCluster[scId].energy;
    //most energetic TS from all TSs linked to SC over SC energy.
    float SCEnergyFractionInTS = 0.f;
    for (const auto& ts : tssInSimCluster[scId].tracksterIdToEnergyAndScore) {
      if (ts.second.first > maxEnergyTSinSC) {
        maxEnergyTSinSC = ts.second.first;
        tsWithMaxEnergyInSC = ts.first;
      }
    }
    if (SCenergy > 0.f)
      SCEnergyFractionInTS = maxEnergyTSinSC / SCenergy;

    LogDebug("TSToSCAssociatorByEnergyScoreImpl")
        << std::setw(12) << "simcluster\t" << std::setw(15) << "sc total energy\t" << std::setw(15)
        << "scEnergyOnLayer\t" << std::setw(14) << "SCNhitsOnLayer\t" << std::setw(18) << "tsWithMaxEnergyInSC\t"
        << std::setw(15) << "maxEnergyTSinSC\t" << std::setw(20) << "SCEnergyFractionInTS"
        << "\n";
    LogDebug("TSToSCAssociatorByEnergyScoreImpl")
        << std::setw(12) << scId << "\t" << std::setw(15) << simClusters[scId].energy() << "\t" << std::setw(15)
        << SCenergy << "\t" << std::setw(14) << SCNumberOfHits << "\t" << std::setw(18) << tsWithMaxEnergyInSC << "\t"
        << std::setw(15) << maxEnergyTSinSC << "\t" << std::setw(20) << SCEnergyFractionInTS << "\n";
#endif
    // Compute the correct normalization
    for (auto const& haf : tssInSimCluster[scId].hits_and_fractions) {
      invSCEnergyWeight += std::pow(haf.second * hitMap_->at(haf.first)->energy(), 2);
    }
    invSCEnergyWeight = 1.f / invSCEnergyWeight;

    for (unsigned int i = 0; i < SCNumberOfHits; ++i) {
      auto& sc_hitDetId = tssInSimCluster[scId].hits_and_fractions[i].first;
      auto& scFraction = tssInSimCluster[scId].hits_and_fractions[i].second;

      bool hitWithLC = false;
      if (scFraction == 0.f)
        continue;  // hopefully this should never happen
      const auto hit_find_in_LC = detIdToLayerClusterId_Map.find(sc_hitDetId);
      if (hit_find_in_LC != detIdToLayerClusterId_Map.end())
        hitWithLC = true;
      const auto itcheck = hitMap_->find(sc_hitDetId);
      const HGCRecHit* hit = itcheck->second;
      float hitEnergyWeight = hit->energy() * hit->energy();
      for (auto& tsPair : tssInSimCluster[scId].tracksterIdToEnergyAndScore) {
        unsigned int tsId = tsPair.first;
        float tsFraction = 0.f;

        for (unsigned int i = 0; i < tracksters[tsId].vertices().size(); ++i) {
          const auto lcId = tracksters[tsId].vertices(i);
          const auto lcFractionInTs = 1.f / tracksters[tsId].vertex_multiplicity(i);

          if (hitWithLC) {
            const auto findHitIt = std::find(detIdToLayerClusterId_Map[sc_hitDetId].begin(),
                                             detIdToLayerClusterId_Map[sc_hitDetId].end(),
                                             hgcal::detIdInfoInCluster{lcId, 0.f});
            if (findHitIt != detIdToLayerClusterId_Map[sc_hitDetId].end())
              tsFraction = findHitIt->fraction * lcFractionInTs;
          }
          tsPair.second.second +=
              (tsFraction - scFraction) * (tsFraction - scFraction) * hitEnergyWeight * invSCEnergyWeight;
#ifdef EDM_ML_DEBUG
          LogDebug("TSToSCAssociatorByEnergyScoreImpl")
              << "SCDetId:\t" << (uint32_t)sc_hitDetId << "\tTracksterId:\t" << tsId << "\t"
              << "tsFraction, scFraction:\t" << tsFraction << ", " << scFraction << "\t"
              << "hitEnergyWeight:\t" << hitEnergyWeight << "\t"
              << "current score:\t" << tsPair.second.second << "\t"
              << "invSCEnergyWeight:\t" << invSCEnergyWeight << "\n";
#endif
        }  // End of loop over Trackster's LayerClusters
      }    // End of loop over Tracksters linked to hits of this SimCluster
    }      // End of loop over hits of SimCluster on a Layer
#ifdef EDM_ML_DEBUG
    if (tssInSimCluster[scId].tracksterIdToEnergyAndScore.empty())
      LogDebug("TSToSCAssociatorByEnergyScoreImpl") << "SC Id:\t" << scId << "\tTS id:\t-1 "
                                                    << "\tscore\t-1\n";

    for (const auto& tsPair : tssInSimCluster[scId].tracksterIdToEnergyAndScore) {
      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << "SC Id: \t" << scId << "\t TS id: \t" << tsPair.first << "\t score \t" << tsPair.second.second
          << "\t shared energy:\t" << tsPair.second.first << "\t shared energy fraction:\t"
          << (tsPair.second.first / SCenergy) << "\n";
    }
#endif
  }  // End loop over SimCluster indices
  return {scsInTrackster, tssInSimCluster};
}

hgcal::RecoToSimCollectionTracksters TSToSCAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<ticl::TracksterCollection>& tCH,
    const edm::Handle<reco::CaloClusterCollection>& lCCH,
    const edm::Handle<SimClusterCollection>& sCCH) const {
  hgcal::RecoToSimCollectionTracksters returnValue(productGetter_);
  const auto& links = makeConnections(tCH, lCCH, sCCH);

  const auto& scsInTrackster = std::get<0>(links);
  for (size_t tsId = 0; tsId < scsInTrackster.size(); ++tsId) {
    for (auto& scPair : scsInTrackster[tsId]) {
      LogDebug("TSToSCAssociatorByEnergyScoreImpl")
          << "Trackster Id:\t" << tsId << "\tSimCluster id:\t" << scPair.first << "\tscore:\t" << scPair.second << "\n";
      // Fill AssociationMap
      returnValue.insert(edm::Ref<ticl::TracksterCollection>(tCH, tsId),  // Ref to TS
                         std::make_pair(edm::Ref<SimClusterCollection>(sCCH, scPair.first),
                                        scPair.second)  // Pair <Ref to SC, score>
      );
    }
  }
  return returnValue;
}

hgcal::SimToRecoCollectionTracksters TSToSCAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<ticl::TracksterCollection>& tCH,
    const edm::Handle<reco::CaloClusterCollection>& lCCH,
    const edm::Handle<SimClusterCollection>& sCCH) const {
  hgcal::SimToRecoCollectionTracksters returnValue(productGetter_);
  const auto& links = makeConnections(tCH, lCCH, sCCH);
  const auto& tssInSimCluster = std::get<1>(links);
  for (size_t scId = 0; scId < tssInSimCluster.size(); ++scId) {
    for (auto& tsPair : tssInSimCluster[scId].tracksterIdToEnergyAndScore) {
      returnValue.insert(
          edm::Ref<SimClusterCollection>(sCCH, scId),                                // Ref to SC
          std::make_pair(edm::Ref<ticl::TracksterCollection>(tCH, tsPair.first),     // Pair <Ref to TS,
                         std::make_pair(tsPair.second.first, tsPair.second.second))  // pair <energy, score> >
      );
    }
  }
  return returnValue;
}
