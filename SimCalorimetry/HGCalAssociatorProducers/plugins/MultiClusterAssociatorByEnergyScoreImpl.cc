// Original Author: Leonardo Cristella
//

#include "MultiClusterAssociatorByEnergyScoreImpl.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimCalorimetry/HGCalAssociatorProducers/interface/AssociatorTools.h"

#include <cfloat>

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
  const auto& mClusters = *mCCH.product();
  const auto& caloParticles = *cPCH.product();
  auto nMultiClusters = mClusters.size();
  //Consider CaloParticles coming from the hard scatterer, excluding the PU contribution.
  std::vector<size_t> cPIndices;
  //Consider CaloParticles coming from the hard scatterer
  //excluding the PU contribution and save the indices.
  removeCPFromPU(caloParticles, cPIndices, false);
  auto nCaloParticles = cPIndices.size();

  std::vector<size_t> cPSelectedIndices;
  removeCPFromPU(caloParticles, cPSelectedIndices, true);

  //cPOnLayer[caloparticle][layer]
  //This defines a "caloParticle on layer" concept. It is only filled in case
  //that caloParticle has a reconstructed hit related via detid. So, a cPOnLayer[i][j] connects a
  //specific caloParticle i in layer j with:
  //1. the sum of all recHits energy times fraction of the relevant simHit in layer j related to that caloParticle i.
  //2. the hits and fractions of that caloParticle i in layer j.
  //3. the layer clusters with matched recHit id.
  hgcal::caloParticleToMultiCluster cPOnLayer;
  cPOnLayer.resize(nCaloParticles);
  for (unsigned int i = 0; i < nCaloParticles; ++i) {
    auto cpIndex = cPIndices[i];
    cPOnLayer[cpIndex].resize(layers_ * 2);
    for (unsigned int j = 0; j < layers_ * 2; ++j) {
      cPOnLayer[cpIndex][j].caloParticleId = cpIndex;
      cPOnLayer[cpIndex][j].energy = 0.f;
      cPOnLayer[cpIndex][j].hits_and_fractions.clear();
    }
  }

  std::unordered_map<DetId, std::vector<hgcal::detIdInfoInCluster>> detIdToCaloParticleId_Map;
  // Fill detIdToCaloParticleId_Map and update cPOnLayer
  for (const auto& cpId : cPIndices) {
    //take sim clusters
    const SimClusterRefVector& simClusterRefVector = caloParticles[cpId].simClusters();
    //loop through sim clusters
    for (const auto& it_sc : simClusterRefVector) {
      const SimCluster& simCluster = (*(it_sc));
      const auto& hits_and_fractions = simCluster.hits_and_fractions();
      for (const auto& it_haf : hits_and_fractions) {
        const auto hitid = (it_haf.first);
        const auto cpLayerId =
            recHitTools_->getLayerWithOffset(hitid) + layers_ * ((recHitTools_->zside(hitid) + 1) >> 1) - 1;
        const auto itcheck = hitMap_->find(hitid);
        if (itcheck != hitMap_->end()) {
          //Since the current hit from sim cluster has a reconstructed hit with the same detid,
          //make a map that will connect a detid with:
          //1. the caloParticles that have a simcluster with sim hits in that cell via caloParticle id.
          //2. the sum of all simHits fractions that contributes to that detid.
          //So, keep in mind that in case of multiple caloParticles contributing in the same cell
          //the fraction is the sum over all caloParticles. So, something like:
          //detid: (caloParticle 1, sum of hits fractions in that detid over all cp) , (caloParticle 2, sum of hits fractions in that detid over all cp), (caloParticle 3, sum of hits fractions in that detid over all cp) ...
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
          //Since the current hit from sim cluster has a reconstructed hit with the same detid,
          //fill the cPOnLayer[caloparticle][layer] object with energy (sum of all recHits energy times fraction
          //of the relevant simHit) and keep the hit (detid and fraction) that contributed.
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
      }  // end of loop through simHits
    }    // end of loop through simclusters
  }      // end of loop through caloParticles

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
      for (auto const& mc : cPOnLayer[cp][cpp].multiClusterIdToEnergyAndScore) {
        LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "      mcIdx/energy/score: " << mc.first << "/"
                                                            << mc.second.first << "/" << mc.second.second << std::endl;
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
  std::unordered_map<DetId, std::vector<hgcal::detIdInfoInMultiCluster>> detIdToMultiClusterId_Map;

  // this contains the ids of the caloParticles contributing with at least one hit to the multiCluster and the reconstruction error
  //cpsInMultiCluster[multicluster][CPids]
  //Connects a multiCluster with all related caloParticles.
  hgcal::multiClusterToCaloParticle cpsInMultiCluster;
  cpsInMultiCluster.resize(nMultiClusters);

  //Loop through multiClusters
  for (unsigned int mcId = 0; mcId < nMultiClusters; ++mcId) {
    const auto& hits_and_fractions = mClusters[mcId].hitsAndFractions();
    if (!hits_and_fractions.empty()) {
      std::unordered_map<unsigned, float> CPEnergyInMCL;
      int maxCPId_byNumberOfHits = -1;
      unsigned int maxCPNumberOfHitsInMCL = 0;
      int maxCPId_byEnergy = -1;
      float maxEnergySharedMCLandCP = 0.f;
      float energyFractionOfMCLinCP = 0.f;
      float energyFractionOfCPinMCL = 0.f;

      //In case of matched rechit-simhit, so matched
      //caloparticle-layercluster-multicluster, we count and save the number of
      //recHits related to the maximum energy CaloParticle out of all
      //CaloParticles related to that layer cluster and multiCluster.

      std::unordered_map<unsigned, unsigned> occurrencesCPinMCL;
      unsigned int numberOfNoiseHitsInMCL = 0;
      unsigned int numberOfHitsInMCL = 0;

      //number of hits related to that cluster
      unsigned int numberOfHitsInLC = hits_and_fractions.size();
      numberOfHitsInMCL += numberOfHitsInLC;
      std::unordered_map<unsigned, float> CPEnergyInLC;

      //hitsToCaloParticleId is a vector of ints, one for each recHit of the
      //layer cluster under study. If negative, there is no simHit from any CaloParticle related.
      //If positive, at least one CaloParticle has been found with matched simHit.
      //In more detail:
      // 1. hitsToCaloParticleId[hitId] = -3
      //    TN:  These represent Halo Cells(N) that have not been
      //    assigned to any CaloParticle (hence the T).
      // 2. hitsToCaloParticleId[hitId] = -2
      //    FN: There represent Halo Cells(N) that have been assigned
      //    to a CaloParticle (hence the F, since those should have not been marked as halo)
      // 3. hitsToCaloParticleId[hitId] = -1
      //    FP: These represent Real Cells(P) that have not been
      //    assigned to any CaloParticle (hence the F, since these are fakes)
      // 4. hitsToCaloParticleId[hitId] >= 0
      //    TP There represent Real Cells(P) that have been assigned
      //    to a CaloParticle (hence the T)

      std::vector<int> hitsToCaloParticleId(numberOfHitsInLC);
      //det id of the first hit just to make the lcLayerId variable
      //which maps the layers in -z: 0->51 and in +z: 52->103
      const auto firstHitDetId = hits_and_fractions[0].first;
      int lcLayerId = recHitTools_->getLayerWithOffset(firstHitDetId) +
                      layers_ * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;

      //Loop though the hits of the layer cluster under study
      for (unsigned int hitId = 0; hitId < numberOfHitsInLC; hitId++) {
        const auto rh_detid = hits_and_fractions[hitId].first;
        const auto rhFraction = hits_and_fractions[hitId].second;

        //Since the hit is belonging to the layer cluster, it must also be in the recHits map.
        const auto itcheck = hitMap_->find(rh_detid);
        const auto hit = itcheck->second;

        //Make a map that will connect a detid (that belongs to a recHit of the layer cluster under study,
        //no need to save others) with:
        //1. the layer clusters that have recHits in that detid
        //2. the fraction of the recHit of each layer cluster that contributes to that detid.
        //So, something like:
        //detid: (layer cluster 1, hit fraction) , (layer cluster 2, hit fraction), (layer cluster 3, hit fraction) ...
        //here comparing with the caloParticle map above
        auto hit_find_in_LC = detIdToMultiClusterId_Map.find(rh_detid);
        if (hit_find_in_LC == detIdToMultiClusterId_Map.end()) {
          detIdToMultiClusterId_Map[rh_detid] = std::vector<hgcal::detIdInfoInMultiCluster>();
        }
        detIdToMultiClusterId_Map[rh_detid].emplace_back(hgcal::detIdInfoInMultiCluster{mcId, mcId, rhFraction});

        // Check whether the recHit of the layer cluster under study has a sim hit in the same cell
        auto hit_find_in_CP = detIdToCaloParticleId_Map.find(rh_detid);

        // If the fraction is zero or the hit does not belong to any calo
        // particle, set the caloParticleId for the hit to -1 and this will
        // contribute to the number of noise hits
        if (rhFraction == 0.) {  // this could be a real hit that has been marked as halo
          hitsToCaloParticleId[hitId] = -2;
        }
        if (hit_find_in_CP == detIdToCaloParticleId_Map.end()) {
          hitsToCaloParticleId[hitId] -= 1;
        } else {
          auto maxCPEnergyInLC = 0.f;
          auto maxCPId = -1;
          for (auto& h : hit_find_in_CP->second) {
            auto shared_fraction = std::min(rhFraction, h.fraction);
            //We are in the case where there are caloParticles with simHits connected via detid with the recHit under study
            //So, from all layers clusters, find the recHits that are connected with a caloParticle and save/calculate the
            //energy of that caloParticle as the sum over all recHits of the recHits energy weighted
            //by the caloParticle's fraction related to that recHit.
            CPEnergyInMCL[h.clusterId] += shared_fraction * hit->energy();
            //Same but for layer clusters for the cell association per layer
            CPEnergyInLC[h.clusterId] += shared_fraction * hit->energy();
            //Here cPOnLayer[caloparticle][layer] described above is set
            //Here for multiClusters with matched recHit, the CP fraction times hit energy is added and saved
            cPOnLayer[h.clusterId][lcLayerId].multiClusterIdToEnergyAndScore[mcId].first +=
                shared_fraction * hit->energy();
            cPOnLayer[h.clusterId][lcLayerId].multiClusterIdToEnergyAndScore[mcId].second = FLT_MAX;
            //cpsInMultiCluster[multicluster][CPids]
            //Connects a multiCluster with all related caloParticles
            cpsInMultiCluster[mcId].emplace_back(h.clusterId, FLT_MAX);
            //From all CaloParticles related to a layer cluster, we save id and energy of the caloParticle
            //that after simhit-rechit matching in layer has the maximum energy.
            if (shared_fraction > maxCPEnergyInLC) {
              //energy is used only here. cpid is saved for multiClusters
              maxCPEnergyInLC = CPEnergyInLC[h.clusterId];
              maxCPId = h.clusterId;
            }
          }
          //Keep in mind here maxCPId could be zero. So, below ask for negative not including zero to count noise.
          hitsToCaloParticleId[hitId] = maxCPId;
        }

      }  //end of loop through recHits of the layer cluster.

      //Loop through all recHits to count how many of them are noise and how many are matched.
      //In case of matched rechit-simhit, we count and save the number of recHits related to the maximum energy CaloParticle.
      for (auto c : hitsToCaloParticleId) {
        if (c < 0) {
          numberOfNoiseHitsInMCL++;
        } else {
          occurrencesCPinMCL[c]++;
        }
      }

      //Below from all maximum energy CaloParticles, we save the one with the largest amount
      //of related recHits.
      for (auto& c : occurrencesCPinMCL) {
        if (c.second > maxCPNumberOfHitsInMCL) {
          maxCPId_byNumberOfHits = c.first;
          maxCPNumberOfHitsInMCL = c.second;
        }
      }

      //Find the CaloParticle that has the maximum energy shared with the multiCluster under study.
      for (auto& c : CPEnergyInMCL) {
        if (c.second > maxEnergySharedMCLandCP) {
          maxCPId_byEnergy = c.first;
          maxEnergySharedMCLandCP = c.second;
        }
      }
      //The energy of the CaloParticle that found to have the maximum energy shared with the multiCluster under study.
      float totalCPEnergyFromLayerCP = 0.f;
      if (maxCPId_byEnergy >= 0) {
        //Loop through all layers
        for (unsigned int j = 0; j < layers_ * 2; ++j) {
          totalCPEnergyFromLayerCP = totalCPEnergyFromLayerCP + cPOnLayer[maxCPId_byEnergy][j].energy;
        }
        energyFractionOfCPinMCL = maxEnergySharedMCLandCP / totalCPEnergyFromLayerCP;
        if (mClusters[mcId].energy() > 0.f) {
          energyFractionOfMCLinCP = maxEnergySharedMCLandCP / mClusters[mcId].energy();
        }
      }

      LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << std::setw(12) << "multiCluster"
                                                          << "\t" << std::setw(10) << "mulcl energy"
                                                          << "\t" << std::setw(5) << "nhits"
                                                          << "\t" << std::setw(12) << "noise hits"
                                                          << "\t" << std::setw(22) << "maxCPId_byNumberOfHits"
                                                          << "\t" << std::setw(8) << "nhitsCP"
                                                          << "\t" << std::setw(16) << "maxCPId_byEnergy"
                                                          << "\t" << std::setw(23) << "maxEnergySharedMCLandCP"
                                                          << "\t" << std::setw(22) << "totalCPEnergyFromAllLayerCP"
                                                          << "\t" << std::setw(22) << "energyFractionOfMCLinCP"
                                                          << "\t" << std::setw(25) << "energyFractionOfCPinMCL"
                                                          << "\t" << std::endl;
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << std::setw(12) << mcId << "\t" << std::setw(10) << mClusters[mcId].energy() << "\t" << std::setw(5)
          << numberOfHitsInMCL << "\t" << std::setw(12) << numberOfNoiseHitsInMCL << "\t" << std::setw(22)
          << maxCPId_byNumberOfHits << "\t" << std::setw(8) << maxCPNumberOfHitsInMCL << "\t" << std::setw(16)
          << maxCPId_byEnergy << "\t" << std::setw(23) << maxEnergySharedMCLandCP << "\t" << std::setw(22)
          << totalCPEnergyFromLayerCP << "\t" << std::setw(22) << energyFractionOfMCLinCP << "\t" << std::setw(25)
          << energyFractionOfCPinMCL << std::endl;
    }
  }  // end of loop through multiClusters

  // Update cpsInMultiCluster; compute the score MultiCluster-to-CaloParticle,
  // together with the returned AssociationMap
  for (unsigned int mcId = 0; mcId < nMultiClusters; ++mcId) {
    // find the unique caloParticles id contributing to the multilusters
    std::sort(cpsInMultiCluster[mcId].begin(), cpsInMultiCluster[mcId].end());
    auto last = std::unique(cpsInMultiCluster[mcId].begin(), cpsInMultiCluster[mcId].end());
    cpsInMultiCluster[mcId].erase(last, cpsInMultiCluster[mcId].end());

    const auto& hits_and_fractions = mClusters[mcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    if (numberOfHitsInLC > 0) {
      if (mClusters[mcId].energy() == 0. && !cpsInMultiCluster[mcId].empty()) {
        //Loop through all CaloParticles contributing to multiCluster mcId.
        for (auto& cpPair : cpsInMultiCluster[mcId]) {
          //In case of a multiCluster with zero energy but related CaloParticles the score is set to 1.
          cpPair.second = 1.;
          LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
              << "multiClusterId : \t " << mcId << "\t CP id : \t" << cpPair.first << "\t score \t " << cpPair.second
              << "\n";
        }
        continue;
      }

      // Compute the correct normalization
      float invMultiClusterEnergyWeight = 0.f;
      for (auto const& haf : mClusters[mcId].hitsAndFractions()) {
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

        for (auto& cpPair : cpsInMultiCluster[mcId]) {
          unsigned int multiClusterId = cpPair.first;
          float cpFraction = 0.f;
          if (!hitWithNoCP) {
            auto findHitIt = std::find(detIdToCaloParticleId_Map[rh_detid].begin(),
                                       detIdToCaloParticleId_Map[rh_detid].end(),
                                       hgcal::detIdInfoInCluster{multiClusterId, 0.f});
            if (findHitIt != detIdToCaloParticleId_Map[rh_detid].end())
              cpFraction = findHitIt->fraction;
          }
          if (cpPair.second == FLT_MAX) {
            cpPair.second = 0.f;
          }
          cpPair.second +=
              (rhFraction - cpFraction) * (rhFraction - cpFraction) * hitEnergyWeight * invMultiClusterEnergyWeight;
        }
      }  // End of loop over Hits within a MultiCluster
#ifdef EDM_ML_DEBUG
      //In case of a multiCluster with some energy but none related CaloParticles print some info.
      if (cpsInMultiCluster[mcId].empty())
        LogDebug("MultiClusterAssociatorByEnergyScoreImpl") << "multiCluster Id: \t" << mcId << "\tCP id:\t-1 "
                                                            << "\t score \t-1"
                                                            << "\n";
#endif
    }
  }  // End of loop over MultiClusters

  // Compute the CaloParticle-To-MultiCluster score
  for (const auto& cpId : cPSelectedIndices) {
    for (unsigned int layerId = 0; layerId < layers_ * 2; ++layerId) {
      unsigned int CPNumberOfHits = cPOnLayer[cpId][layerId].hits_and_fractions.size();
      if (CPNumberOfHits == 0)
        continue;
#ifdef EDM_ML_DEBUG
      int mcWithMaxEnergyInCP = -1;
      float maxEnergyMCLperlayerinCP = 0.f;
      float CPenergy = cPOnLayer[cpId][layerId].energy;
      float CPEnergyFractionInMCLperlayer = 0.f;
      for (auto& mc : cPOnLayer[cpId][layerId].multiClusterIdToEnergyAndScore) {
        if (mc.second.first > maxEnergyMCLperlayerinCP) {
          maxEnergyMCLperlayerinCP = mc.second.first;
          mcWithMaxEnergyInCP = mc.first;
        }
      }
      if (CPenergy > 0.f)
        CPEnergyFractionInMCLperlayer = maxEnergyMCLperlayerinCP / CPenergy;

      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << std::setw(8) << "LayerId:\t" << std::setw(12) << "caloparticle\t" << std::setw(15) << "cp total energy\t"
          << std::setw(15) << "cpEnergyOnLayer\t" << std::setw(14) << "CPNhitsOnLayer\t" << std::setw(18)
          << "mcWithMaxEnergyInCP\t" << std::setw(15) << "maxEnergyMCLinCP\t" << std::setw(20)
          << "CPEnergyFractionInMCL"
          << "\n";
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << std::setw(8) << layerId << "\t" << std::setw(12) << cpId << "\t" << std::setw(15)
          << caloParticles[cpId].energy() << "\t" << std::setw(15) << CPenergy << "\t" << std::setw(14)
          << CPNumberOfHits << "\t" << std::setw(18) << mcWithMaxEnergyInCP << "\t" << std::setw(15)
          << maxEnergyMCLperlayerinCP << "\t" << std::setw(20) << CPEnergyFractionInMCLperlayer << "\n";
#endif

      for (unsigned int i = 0; i < CPNumberOfHits; ++i) {
        auto& cp_hitDetId = cPOnLayer[cpId][layerId].hits_and_fractions[i].first;
        auto& cpFraction = cPOnLayer[cpId][layerId].hits_and_fractions[i].second;

        bool hitWithNoMCL = false;
        if (cpFraction == 0.f)
          continue;  //hopefully this should never happen
        auto hit_find_in_MCL = detIdToMultiClusterId_Map.find(cp_hitDetId);
        if (hit_find_in_MCL == detIdToMultiClusterId_Map.end())
          hitWithNoMCL = true;
        auto itcheck = hitMap_->find(cp_hitDetId);
        const HGCRecHit* hit = itcheck->second;
        float hitEnergyWeight = hit->energy() * hit->energy();
        for (auto& mcPair : cPOnLayer[cpId][layerId].multiClusterIdToEnergyAndScore) {
          unsigned int multiClusterId = mcPair.first;
          float mcFraction = 0.f;

          if (!hitWithNoMCL) {
            auto findHitIt = std::find(detIdToMultiClusterId_Map[cp_hitDetId].begin(),
                                       detIdToMultiClusterId_Map[cp_hitDetId].end(),
                                       hgcal::detIdInfoInMultiCluster{multiClusterId, 0, 0.f});
            if (findHitIt != detIdToMultiClusterId_Map[cp_hitDetId].end())
              mcFraction = findHitIt->fraction;
          }
          //Observe here that we do not divide as before by the layer cluster energy weight. We should sum first
          //over all layers and divide with the total CP energy over all layers.
          if (mcPair.second.second == FLT_MAX) {
            mcPair.second.second = 0.f;
          }
          mcPair.second.second += (mcFraction - cpFraction) * (mcFraction - cpFraction) * hitEnergyWeight;
#ifdef EDM_ML_DEBUG
          LogDebug("HGCalValidator") << "multiClusterId:\t" << multiClusterId << "\tmcfraction,cpfraction:\t"
                                     << mcFraction << ", " << cpFraction << "\thitEnergyWeight:\t" << hitEnergyWeight
                                     << "\tcurrent score numerator:\t" << mcPair.second.second << "\n";
#endif
        }  // End of loop over MultiClusters linked to hits of this CaloParticle
      }    // End of loop over hits of CaloParticle on a Layer
#ifdef EDM_ML_DEBUG
      if (cPOnLayer[cpId][layerId].multiClusterIdToEnergyAndScore.empty())
        LogDebug("HGCalValidator") << "CP Id: \t" << cpId << "\t MCL id:\t-1 "
                                   << "\t layer \t " << layerId << " Sub score in \t -1"
                                   << "\n";

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
  for (size_t mcId = 0; mcId < cpsInMultiCluster.size(); ++mcId) {
    for (auto& cpPair : cpsInMultiCluster[mcId]) {
      LogDebug("MultiClusterAssociatorByEnergyScoreImpl")
          << "multiCluster Id: \t" << mcId << "\t CP id: \t" << cpPair.first << "\t score \t" << cpPair.second << "\n";
      // Fill AssociationMap
      returnValue.insert(edm::Ref<reco::HGCalMultiClusterCollection>(mCCH, mcId),  // Ref to MC
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
      for (auto& mcPair : cPOnLayer[cpId][layerId].multiClusterIdToEnergyAndScore) {
        returnValue.insert(
            edm::Ref<CaloParticleCollection>(cPCH, cpId),                                    // Ref to CP
            std::make_pair(edm::Ref<reco::HGCalMultiClusterCollection>(mCCH, mcPair.first),  // Pair <Ref to MC,
                           std::make_pair(mcPair.second.first, mcPair.second.second))        // pair <energy, score> >
        );
      }
    }
  }
  return returnValue;
}
