#include "TSToSimTSAssociatorByEnergyScoreImpl.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

TSToSimTSAssociatorByEnergyScoreImpl::TSToSimTSAssociatorByEnergyScoreImpl(
    edm::EDProductGetter const& productGetter,
    bool hardScatterOnly,
    std::shared_ptr<hgcal::RecHitTools> recHitTools,
    const std::unordered_map<DetId, const HGCRecHit*>* hitMap)
    : hardScatterOnly_(hardScatterOnly), recHitTools_(recHitTools), hitMap_(hitMap), productGetter_(&productGetter) {
  layers_ = recHitTools_->lastLayerBH();
}

hgcal::association TSToSimTSAssociatorByEnergyScoreImpl::makeConnections(
    const edm::Handle<ticl::TracksterCollection>& tCH,
    const edm::Handle<reco::CaloClusterCollection>& lCCH,
    const edm::Handle<ticl::TracksterCollection>& sTCH) const {
  // Get collections
  const auto& tracksters = *tCH.product();
  const auto& layerClusters = *lCCH.product();
  const auto& simTracksters = *sTCH.product();
  auto nTracksters = tracksters.size();

  //There shouldn't be any SimTrackster without vertices, but maybe they will be added later.
  auto nSimTracksters = simTracksters.size();
  std::vector<size_t> sTIndices;
  for (unsigned int stId = 0; stId < nSimTracksters; ++stId) {
    if (simTracksters[stId].vertices().empty()) {
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
          << "Excluding SimTrackster " << stId << " witH no vertices!" << std::endl;
      continue;
    }
    sTIndices.emplace_back(stId);
  }
  nSimTracksters = sTIndices.size();

  // Initialize tssInSimTrackster. It contains the simTracksterOnLayer structure for all CaloParticles in each layer and
  // among other the information to compute the SimTrackster-To-Trackster score. It is one of the two objects that
  // build the output of the makeConnections function.
  // tssInSimTrackster[stId]
  hgcal::simTracksterToTrackster tssInSimTrackster;
  tssInSimTrackster.resize(nSimTracksters);
  for (unsigned int i = 0; i < nSimTracksters; ++i) {
    tssInSimTrackster[i].simTracksterId = i;
    tssInSimTrackster[i].energy = 0.f;
    tssInSimTrackster[i].lcs_and_fractions.clear();
  }

  // Fill lcToSimTracksterId_Map and update tssInSimTrackster
  // The lcToSimTracksterId_Map is used to connect a LayerCluster, via its id (key), with all the SimTracksters that
  // contributed to that LayerCluster by storing the SimTrackster id and the fraction of the LayerCluster's energy
  // in which the SimTrackster contributed.
  std::unordered_map<int, std::vector<hgcal::lcInfoInTrackster>> lcToSimTracksterId_Map;
  for (const auto& stId : sTIndices) {
    const auto& lcs = simTracksters[stId].vertices();
    int lcInSimTst = 0;
    for (const auto& lcId : lcs) {
      const auto fraction = 1.f / simTracksters[stId].vertex_multiplicity(lcInSimTst++);

      const auto lc_find_it = lcToSimTracksterId_Map.find(lcId);
      if (lc_find_it == lcToSimTracksterId_Map.end()) {
        lcToSimTracksterId_Map[lcId] = std::vector<hgcal::lcInfoInTrackster>();
      }
      lcToSimTracksterId_Map[lcId].emplace_back(stId, fraction);

      tssInSimTrackster[stId].energy += fraction * layerClusters[lcId].energy();
      tssInSimTrackster[stId].lcs_and_fractions.emplace_back(lcId, fraction);
    }
  }  // end of loop over SimTracksters

#ifdef EDM_ML_DEBUG
  LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
      << "tssInSimTrackster INFO (Only SimTrackster filled at the moment)" << std::endl;
  for (size_t st = 0; st < tssInSimTrackster.size(); ++st) {
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "For SimTrackster Idx: " << st << " we have: " << std::endl;
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
        << "\tSimTracksterIdx:\t" << tssInSimTrackster[st].simTracksterId << std::endl;
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "\tEnergy:\t" << tssInSimTrackster[st].energy << std::endl;
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "\t# of clusters:\t" << layerClusters.size() << std::endl;
    double tot_energy = 0.;
    for (auto const& lcaf : tssInSimTrackster[st].lcs_and_fractions) {
      const auto lcEn = lcaf.second * layerClusters[lcaf.first].energy();
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
          << "\tLC/fraction/energy: " << (uint32_t)lcaf.first << "/" << lcaf.second << "/" << lcEn << std::endl;
      tot_energy += lcEn;
    }
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "\tTot Sum lcaf: " << tot_energy << std::endl;
    for (auto const& ts : tssInSimTrackster[st].tracksterIdToEnergyAndScore) {
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
          << "\ttsIdx/energy/score: " << ts.first << "/" << ts.second.first << "/" << ts.second.second << std::endl;
    }
  }

  LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "lcToSimTracksterId_Map INFO" << std::endl;
  for (auto const& lc : lcToSimTracksterId_Map) {
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
        << "For lcId: " << (uint32_t)lc.first
        << " we have found the following connections with SimTracksters:" << std::endl;
    for (auto const& st : lc.second) {
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
          << "\tSimTrackster Id: " << st.clusterId << " with fraction: " << st.fraction
          << " and energy: " << st.fraction * layerClusters[lc.first].energy() << std::endl;
    }
  }
#endif

  // Fill stsInTrackster and update tssInSimTrackster
  // this contains the ids of the simTracksters contributing with at least one
  // hit to the Trackster. To be returned since this contains the information
  // to compute the Trackster-To-SimTrackster score.
  hgcal::tracksterToSimTrackster stsInTrackster;  // tsId->(stId,score)
  stsInTrackster.resize(nTracksters);

  for (unsigned int tsId = 0; tsId < nTracksters; ++tsId) {
    for (unsigned int i = 0; i < tracksters[tsId].vertices().size(); ++i) {
      const auto lcId = tracksters[tsId].vertices(i);
      const auto lcFractionInTs = 1.f / tracksters[tsId].vertex_multiplicity(i);

      const auto lc_find_in_ST = lcToSimTracksterId_Map.find(lcId);

      if (lc_find_in_ST != lcToSimTracksterId_Map.end()) {
        //Loops through all the simTracksters that contain the layer cluster under study
        //Here is time to update the tssInSimTrackster and connect the SimTrackster with all
        //the Tracksters that have the current layer cluster matched.
        for (const auto& st : lc_find_in_ST->second) {
          //tssInSimTrackster[simTracksterId][layerclusterId]-> (energy,score)
          //ST_i - > TS_j, TS_k, ...
          tssInSimTrackster[st.clusterId].tracksterIdToEnergyAndScore[tsId].first +=
              lcFractionInTs * st.fraction * layerClusters[lcId].energy();
          //TS_i -> ST_j, ST_k, ...
          stsInTrackster[tsId].emplace_back(st.clusterId, std::make_pair(0.f, 0.f));
        }
      }
    }  // End loop over LayerClusters in Trackster
  }    // End of loop over Tracksters

#ifdef EDM_ML_DEBUG
  for (unsigned int tsId = 0; tsId < nTracksters; ++tsId) {
    for (const auto& lcId : tracksters[tsId].vertices()) {
      const auto& hits_and_fractions = layerClusters[lcId].hitsAndFractions();
      unsigned int numberOfHitsInLC = hits_and_fractions.size();

      // This vector will store, for each hit in the Layercluster, the index of
      // the SimTrackster that contributed the most, in terms of energy, to it.
      // Special values are:
      //
      // -2  --> the reconstruction fraction of the RecHit is 0 (used in the past to monitor Halo Hits)
      // -3  --> same as before with the added condition that no SimTrackster has been linked to this RecHit
      // -1  --> the reco fraction is >0, but no SimTrackster has been linked to it
      // >=0 --> index of the linked SimTrackster
      std::vector<int> hitsToSimTracksterId(numberOfHitsInLC);
      // This will store the index of the SimTrackster linked to the LayerCluster that has the largest number of hits in common.
      int maxSTId_byNumberOfHits = -1;
      // This will store the maximum number of shared hits between a LayerCluster and a SimTrackster
      unsigned int maxSTNumberOfHitsInLC = 0;
      // This will store the index of the SimTrackster linked to the LayerCluster that has the largest energy in common.
      int maxSTId_byEnergy = -1;
      // This will store the maximum number of shared energy between a LayerCluster and a SimTrackster
      float maxEnergySharedLCandST = 0.f;
      // This will store the fraction of the LayerCluster energy shared with the best(energy) SimTrackster: e_shared/lc_energy
      float energyFractionOfLCinST = 0.f;
      // This will store the fraction of the SimTrackster energy shared with the Trackster: e_shared/sc_energy
      float energyFractionOfSTinLC = 0.f;
      std::unordered_map<unsigned, unsigned> occurrencesSTinLC;
      unsigned int numberOfNoiseHitsInLC = 0;
      std::unordered_map<unsigned, float> STEnergyInLC;

      const auto lc_find_in_ST = lcToSimTracksterId_Map.find(lcId);
      for (unsigned int hitId = 0; hitId < numberOfHitsInLC; hitId++) {
        const auto rhFraction = hits_and_fractions[hitId].second;
        // if the fraction is zero or the hit does not belong to any SimTrackster,
        // set the SimTrackster Id for the hit to -1; this will
        // contribute to the number of noise hits

        // MR Remove the case in which the fraction is 0, since this could be a
        // real hit that has been marked as halo.
        if (rhFraction == 0.) {
          hitsToSimTracksterId[hitId] = -2;
        }
        //Now check if there are SimTracksters linked to this rechit of the layercluster
        if (lc_find_in_ST == lcToSimTracksterId_Map.end()) {
          hitsToSimTracksterId[hitId] -= 1;
        } else {
          auto maxSTEnergyInLC = 0.f;
          auto maxSTId = -1;
          //Loop through all the linked SimTracksters
          for (const auto& st : lc_find_in_ST->second) {
            const auto stId = st.clusterId;
            STEnergyInLC[stId] += st.fraction * layerClusters[lcId].energy();
            // Keep track of which SimTrackster contributed the most, in terms
            // of energy, to this specific Layer Cluster.
            if (STEnergyInLC[stId] > maxSTEnergyInLC) {
              maxSTEnergyInLC = STEnergyInLC[stId];
              maxSTId = stId;
            }
          }
          hitsToSimTracksterId[hitId] = maxSTId;
        }
      }  // End loop over hits on a LayerCluster

      for (const auto& c : hitsToSimTracksterId) {
        if (c < 0) {
          numberOfNoiseHitsInLC++;
        } else {
          occurrencesSTinLC[c]++;
        }
      }

      for (const auto& c : occurrencesSTinLC) {
        if (c.second > maxSTNumberOfHitsInLC) {
          maxSTId_byNumberOfHits = c.first;
          maxSTNumberOfHitsInLC = c.second;
        }
      }

      for (const auto& c : STEnergyInLC) {
        if (c.second > maxEnergySharedLCandST) {
          maxSTId_byEnergy = c.first;
          maxEnergySharedLCandST = c.second;
        }
      }

      float totalSTEnergyOnLayer = 0.f;
      if (maxSTId_byEnergy >= 0) {
        totalSTEnergyOnLayer = tssInSimTrackster[maxSTId_byEnergy].energy;
        energyFractionOfSTinLC = maxEnergySharedLCandST / totalSTEnergyOnLayer;
        if (tracksters[tsId].raw_energy() > 0.f) {
          energyFractionOfLCinST = maxEnergySharedLCandST / tracksters[tsId].raw_energy();
        }
      }

      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
          << std::setw(12) << "TracksterID:\t" << std::setw(12) << "layerCluster\t" << std::setw(10) << "lc energy\t"
          << std::setw(5) << "nhits\t" << std::setw(12) << "noise hits\t" << std::setw(22) << "maxSTId_byNumberOfHits\t"
          << std::setw(8) << "nhitsST\t" << std::setw(13) << "maxSTId_byEnergy\t" << std::setw(20)
          << "maxEnergySharedLCandST\t" << std::setw(22) << "totalSTEnergyOnLayer\t" << std::setw(22)
          << "energyFractionOfLCinST\t" << std::setw(25) << "energyFractionOfSTinLC\t"
          << "\n";
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
          << std::setw(12) << tsId << "\t" << std::setw(12) << lcId << "\t" << std::setw(10)
          << tracksters[tsId].raw_energy() << "\t" << std::setw(5) << numberOfHitsInLC << "\t" << std::setw(12)
          << numberOfNoiseHitsInLC << "\t" << std::setw(22) << maxSTId_byNumberOfHits << "\t" << std::setw(8)
          << maxSTNumberOfHitsInLC << "\t" << std::setw(13) << maxSTId_byEnergy << "\t" << std::setw(20)
          << maxEnergySharedLCandST << "\t" << std::setw(22) << totalSTEnergyOnLayer << "\t" << std::setw(22)
          << energyFractionOfLCinST << "\t" << std::setw(25) << energyFractionOfSTinLC << "\n";
    }  // End of loop over LayerClusters in Trackster
  }    // End of loop over Tracksters

  LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
      << "Improved tssInSimTrackster INFO (Now containing the linked tracksters id and energy - score still empty)"
      << std::endl;
  for (size_t st = 0; st < tssInSimTrackster.size(); ++st) {
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "For SimTrackster Idx: " << st << " we have: " << std::endl;
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
        << "    SimTracksterIdx: " << tssInSimTrackster[st].simTracksterId << std::endl;
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "\tEnergy:\t" << tssInSimTrackster[st].energy << std::endl;
    double tot_energy = 0.;
    for (auto const& lcaf : tssInSimTrackster[st].lcs_and_fractions) {
      const auto lcEn = lcaf.second * layerClusters[lcaf.first].energy();
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
          << "\tLC/fraction/energy: " << (uint32_t)lcaf.first << "/" << lcaf.second << "/" << lcEn << std::endl;
      tot_energy += lcEn;
    }
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "\tTot Sum lcaf: " << tot_energy << std::endl;
    for (auto const& ts : tssInSimTrackster[st].tracksterIdToEnergyAndScore) {
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
          << "\ttsIdx/energy/score: " << ts.first << "/" << ts.second.first << "/" << ts.second.second << std::endl;
    }
  }

  LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "Improved lcToSimTracksterId_Map INFO" << std::endl;
  for (auto const& lc : lcToSimTracksterId_Map) {
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
        << "For lcId: " << (uint32_t)lc.first
        << " we have found the following connections with SimTracksters:" << std::endl;
    for (auto const& st : lc.second) {
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
          << "  SimTrackster Id: " << st.clusterId << " with fraction: " << st.fraction
          << " and energy: " << st.fraction * layerClusters[lc.first].energy() << std::endl;
    }
  }
#endif

  // Update stsInTrackster; compute the score Trackster-to-SimTrackster,
  // together with the returned AssociationMap
  for (unsigned int tsId = 0; tsId < nTracksters; ++tsId) {
    // The SimTracksters contributing to the Trackster's LayerClusters should already be unique.
    // find the unique SimTracksters id contributing to the Trackster's LayerClusters
    std::sort(stsInTrackster[tsId].begin(), stsInTrackster[tsId].end());
    auto last = std::unique(stsInTrackster[tsId].begin(), stsInTrackster[tsId].end());
    stsInTrackster[tsId].erase(last, stsInTrackster[tsId].end());

    // If a reconstructed Trackster has energy 0 but is linked to a
    // SimTrackster, assigned score 1
    if (tracksters[tsId].raw_energy() == 0. && !stsInTrackster[tsId].empty()) {
      for (auto& stPair : stsInTrackster[tsId]) {
        stPair.second.second = 1.;
        LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
            << "TracksterId:\t " << tsId << "\tST id:\t" << stPair.first << "\tenergy" << stPair.second.first
            << "\tscore\t " << stPair.second.second << "\n";
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
        invTracksterEnergyWeight += std::pow(lcFractionInTs * haf.second * hitMap_->at(haf.first)->energy(), 2);
      }
    }
    invTracksterEnergyWeight = 1.f / invTracksterEnergyWeight;

    for (unsigned int i = 0; i < tracksters[tsId].vertices().size(); ++i) {
      const auto lcId = tracksters[tsId].vertices(i);
      const auto lcFractionInTs = 1.f / tracksters[tsId].vertex_multiplicity(i);

      const bool lcWithST = (lcToSimTracksterId_Map.find(lcId) != lcToSimTracksterId_Map.end());

      float lcEnergyWeight = pow(layerClusters[lcId].energy(), 2);

      for (auto& stPair : stsInTrackster[tsId]) {
        float stFraction = 0.f;
        if (lcWithST) {
          const auto findLCIt = std::find(lcToSimTracksterId_Map[lcId].begin(),
                                          lcToSimTracksterId_Map[lcId].end(),
                                          hgcal::lcInfoInTrackster{stPair.first, 0.f});
          if (findLCIt != lcToSimTracksterId_Map[lcId].end()) {
            stFraction = findLCIt->fraction;
          }
        }
        stPair.second.second +=
            (lcFractionInTs - stFraction) * (lcFractionInTs - stFraction) * lcEnergyWeight * invTracksterEnergyWeight;
#ifdef EDM_ML_DEBUG
        LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
            << "lcId:\t" << (uint32_t)lcId << "\ttracksterId:\t" << tsId << "\ttsFraction,stFraction:\t"
            << lcFractionInTs << ", " << stFraction << "\tlcEnergyWeight:\t" << lcEnergyWeight << "\tcurrent score:\t"
            << stPair.second.second << "\tinvTracksterEnergyWeight:\t" << invTracksterEnergyWeight << "\n";
#endif
      }
    }  // End of loop over LayerClusters in Trackster

#ifdef EDM_ML_DEBUG
    if (stsInTrackster[tsId].empty())
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "trackster Id:\t" << tsId << "\tST id:\t-1"
                                                       << "\tscore\t-1\n";
#endif
  }  // End of loop over Tracksters

  // Compute the SimTrackster-To-Trackster score
  for (const auto& stId : sTIndices) {
    float invSTEnergyWeight = 0.f;

    const unsigned int STNumberOfLCs = tssInSimTrackster[stId].lcs_and_fractions.size();
    if (STNumberOfLCs == 0)
      continue;

#ifdef EDM_ML_DEBUG
    int tsWithMaxEnergyInST = -1;
    //energy of the most energetic TS from all that were linked to ST
    float maxEnergyTSinST = 0.f;
    float STenergy = tssInSimTrackster[stId].energy;
    //most energetic TS from all TSs linked to ST over ST energy.
    float STEnergyFractionInTS = 0.f;
    for (const auto& ts : tssInSimTrackster[stId].tracksterIdToEnergyAndScore) {
      if (ts.second.first > maxEnergyTSinST) {
        maxEnergyTSinST = ts.second.first;
        tsWithMaxEnergyInST = ts.first;
      }
    }
    if (STenergy > 0.f)
      STEnergyFractionInTS = maxEnergyTSinST / STenergy;

    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
        << std::setw(12) << "simTrackster\t" << std::setw(15) << "st total energy\t" << std::setw(15)
        << "stEnergyOnLayer\t" << std::setw(14) << "STNhitsOnLayer\t" << std::setw(18) << "tsWithMaxEnergyInST\t"
        << std::setw(15) << "maxEnergyTSinST\t" << std::setw(20) << "STEnergyFractionInTS"
        << "\n";
    LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
        << std::setw(12) << stId << "\t" << std::setw(15) << simTracksters[stId].raw_energy() << "\t" << std::setw(15)
        << STenergy << "\t" << std::setw(14) << STNumberOfLCs << "\t" << std::setw(18) << tsWithMaxEnergyInST << "\t"
        << std::setw(15) << maxEnergyTSinST << "\t" << std::setw(20) << STEnergyFractionInTS << "\n";
#endif

    // Compute the correct normalization
    for (auto const& lcaf : tssInSimTrackster[stId].lcs_and_fractions) {
      invSTEnergyWeight += std::pow(lcaf.second * layerClusters[lcaf.first].energy(), 2);
    }
    invSTEnergyWeight = 1.f / invSTEnergyWeight;

    for (unsigned int i = 0; i < STNumberOfLCs; ++i) {
      auto& st_lcId = tssInSimTrackster[stId].lcs_and_fractions[i].first;
      auto& st_lcFraction = tssInSimTrackster[stId].lcs_and_fractions[i].second;

      if (st_lcFraction == 0.f)
        continue;  // hopefully this should never happen
      float lcEnergyWeight = pow(layerClusters[st_lcId].energy(), 2);
      for (auto& tsPair : tssInSimTrackster[stId].tracksterIdToEnergyAndScore) {
        unsigned int tsId = tsPair.first;

        for (unsigned int i = 0; i < tracksters[tsId].vertices().size(); ++i) {
          const auto tsFraction = 1.f / tracksters[tsId].vertex_multiplicity(i);

          tsPair.second.second +=
              (tsFraction - st_lcFraction) * (tsFraction - st_lcFraction) * lcEnergyWeight * invSTEnergyWeight;
#ifdef EDM_ML_DEBUG
          LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
              << "STLCId:\t" << (uint32_t)st_lcId << "\tTracksterId:\t" << tsId << "\t"
              << "tsFraction, stFraction:\t" << tsFraction << ", " << st_lcFraction << "\t"
              << "lcEnergyWeight:\t" << lcEnergyWeight << "\t"
              << "current score:\t" << tsPair.second.second << "\t"
              << "invSTEnergyWeight:\t" << invSTEnergyWeight << "\n";
#endif
        }  // End of loop over Trackster's LayerClusters
      }    // End of loop over Tracksters linked to hits of this SimTrackster
    }      // End of loop over hits of SimTrackster on a Layer
#ifdef EDM_ML_DEBUG
    if (tssInSimTrackster[stId].tracksterIdToEnergyAndScore.empty())
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "ST Id:\t" << stId << "\tTS id:\t-1 "
                                                       << "\tscore\t-1\n";

    for (const auto& tsPair : tssInSimTrackster[stId].tracksterIdToEnergyAndScore) {
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl")
          << "ST Id: \t" << stId << "\t TS id: \t" << tsPair.first << "\t score \t" << tsPair.second.second
          << "\t shared energy:\t" << tsPair.second.first << "\t shared energy fraction:\t"
          << (tsPair.second.first / STenergy) << "\n";
    }
#endif
  }  // End loop over SimTrackster indices
  return {stsInTrackster, tssInSimTrackster};
}

hgcal::RecoToSimCollectionSimTracksters TSToSimTSAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<ticl::TracksterCollection>& tCH,
    const edm::Handle<reco::CaloClusterCollection>& lCCH,
    const edm::Handle<ticl::TracksterCollection>& sTCH) const {
  hgcal::RecoToSimCollectionSimTracksters returnValue(productGetter_);
  const auto& links = makeConnections(tCH, lCCH, sTCH);

  const auto& stsInTrackster = std::get<0>(links);
  for (size_t tsId = 0; tsId < stsInTrackster.size(); ++tsId) {
    for (auto& stPair : stsInTrackster[tsId]) {
      LogDebug("TSToSimTSAssociatorByEnergyScoreImpl") << "Trackster Id:\t" << tsId << "\tSimTrackster id:\t"
                                                       << stPair.first << "\tscore:\t" << stPair.second << "\n";
      // Fill AssociationMap
      returnValue.insert(
          edm::Ref<ticl::TracksterCollection>(tCH, tsId),                            // Ref to TS
          std::make_pair(edm::Ref<ticl::TracksterCollection>(sTCH, stPair.first),    //Pair <Refo to TS>
                         std::make_pair(stPair.second.first, stPair.second.second))  // Pair <energy, score>
      );
    }
  }
  return returnValue;
}

hgcal::SimToRecoCollectionSimTracksters TSToSimTSAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<ticl::TracksterCollection>& tCH,
    const edm::Handle<reco::CaloClusterCollection>& lCCH,
    const edm::Handle<ticl::TracksterCollection>& sTCH) const {
  hgcal::SimToRecoCollectionSimTracksters returnValue(productGetter_);
  const auto& links = makeConnections(tCH, lCCH, sTCH);
  const auto& tssInSimTrackster = std::get<1>(links);
  for (size_t stId = 0; stId < tssInSimTrackster.size(); ++stId) {
    for (auto& tsPair : tssInSimTrackster[stId].tracksterIdToEnergyAndScore) {
      returnValue.insert(
          edm::Ref<ticl::TracksterCollection>(sTCH, stId),                           // Ref to ST
          std::make_pair(edm::Ref<ticl::TracksterCollection>(tCH, tsPair.first),     // Pair <Ref to TS,
                         std::make_pair(tsPair.second.first, tsPair.second.second))  // pair <energy, score> >
      );
    }
  }
  return returnValue;
}
