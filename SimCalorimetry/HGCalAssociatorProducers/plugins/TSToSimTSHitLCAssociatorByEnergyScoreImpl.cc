#include <cfloat>

#include "TSToSimTSHitLCAssociatorByEnergyScoreImpl.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

TSToSimTSHitLCAssociatorByEnergyScoreImpl::TSToSimTSHitLCAssociatorByEnergyScoreImpl(
    edm::EDProductGetter const& productGetter,
    bool hardScatterOnly,
    std::shared_ptr<hgcal::RecHitTools> recHitTools,
    const std::unordered_map<DetId, const HGCRecHit*>* hitMap)
    : hardScatterOnly_(hardScatterOnly), recHitTools_(recHitTools), hitMap_(hitMap), productGetter_(&productGetter) {
  layers_ = recHitTools_->lastLayerBH();
}

hgcal::association_t TSToSimTSHitLCAssociatorByEnergyScoreImpl::makeConnections(
    const edm::Handle<ticl::TracksterCollection>& tCH,
    const edm::Handle<reco::CaloClusterCollection>& lCCH,
    const edm::Handle<SimClusterCollection>& sCCH,
    const edm::Handle<CaloParticleCollection>& cPCH,
    const edm::Handle<ticl::TracksterCollection>& sTCH) const {
  // Get collections
  const auto& tracksters = *tCH.product();
  const auto& layerClusters = *lCCH.product();
  const auto& sC = *sCCH.product();
  const auto& cP = *cPCH.product();
  const auto cPHandle_id = cPCH.id();
  const auto& simTSs = *sTCH.product();
  const auto nTracksters = tracksters.size();
  const auto nSimTracksters = simTSs.size();

  std::unordered_map<DetId, std::vector<std::pair<int, float>>> detIdSimTSId_Map;
  std::unordered_map<DetId, std::vector<std::pair<int, float>>> detIdSimClusterId_Map;
  std::unordered_map<DetId, std::vector<std::pair<int, float>>> detIdCaloParticleId_Map;
  std::unordered_map<DetId, std::vector<std::pair<int, float>>> detIdToRecoTSId_Map;

  hgcal::sharedEnergyAndScore_t recoToSim_sharedEnergyAndScore;
  hgcal::sharedEnergyAndScore_t simToReco_sharedEnergyAndScore;

  recoToSim_sharedEnergyAndScore.resize(nTracksters);
  simToReco_sharedEnergyAndScore.resize(nSimTracksters);

  for (size_t i = 0; i < nTracksters; ++i)
    recoToSim_sharedEnergyAndScore[i].resize(nSimTracksters);
  for (size_t i = 0; i < nSimTracksters; ++i)
    simToReco_sharedEnergyAndScore[i].resize(nTracksters);

  // fill sim maps

  for (size_t i = 0; i < sC.size(); ++i) {
    for (const auto& haf : sC[i].hits_and_fractions()) {
      detIdSimClusterId_Map[haf.first].emplace_back(i, haf.second);
    }
  }

  for (size_t i = 0; i < cP.size(); ++i) {
    for (const auto& sc : cP[i].simClusters()) {
      for (const auto& haf : sc->hits_and_fractions()) {
        auto hitId = haf.first;
        auto found = std::find_if(detIdCaloParticleId_Map[hitId].begin(),
                                  detIdCaloParticleId_Map[hitId].end(),
                                  [=](const std::pair<int, float>& v) { return v.first == static_cast<int>(i); });
        if (found == detIdCaloParticleId_Map[hitId].end())
          detIdCaloParticleId_Map[haf.first].emplace_back(i, haf.second);
        else
          found->second += haf.second;
      }
    }
  }

  for (size_t i = 0; i < nSimTracksters; ++i) {
    const auto& lcsInSimTrackster = simTSs[i].vertices();
    const auto& multiplicities = simTSs[i].vertex_multiplicity();
    for (size_t j = 0; j < lcsInSimTrackster.size(); ++j) {
      assert(multiplicities[j] > 0.f);
      const auto& v = lcsInSimTrackster[j];
      float fraction = 1.f / multiplicities[j];

      for (const auto& haf : layerClusters[v].hitsAndFractions()) {
        detIdSimTSId_Map[haf.first].emplace_back(i, haf.second * fraction);
      }
    }
  }

  // fill reco map

  for (size_t i = 0; i < nTracksters; ++i) {
    const auto& lcsInSimTrackster = tracksters[i].vertices();
    const auto& multiplicities = tracksters[i].vertex_multiplicity();
    for (size_t j = 0; j < lcsInSimTrackster.size(); ++j) {
      assert(multiplicities[j] > 0.f);
      const auto& v = lcsInSimTrackster[j];
      float fraction = 1.f / multiplicities[j];
      for (const auto& haf : layerClusters[v].hitsAndFractions()) {
        detIdToRecoTSId_Map[haf.first].emplace_back(i, haf.second * fraction);
      }
    }
  }

  std::vector<float> denominator_simToReco(nSimTracksters, 0.f);
  std::vector<std::vector<float>> numerator_simToReco(nSimTracksters);
  std::vector<std::vector<float>> sharedEnergy(nSimTracksters);

  for (size_t i = 0; i < nSimTracksters; ++i) {
    numerator_simToReco[i].resize(nTracksters, 0.f);
    sharedEnergy[i].resize(nTracksters, 0.f);

    const auto seedIndex = simTSs[i].seedIndex();
    const auto& lcsInSimTrackster = simTSs[i].vertices();

    for (const auto& v : lcsInSimTrackster) {
      for (const auto& haf : layerClusters[v].hitsAndFractions()) {
        const auto hitId = haf.first;
        float simFraction = 0.f;

        std::vector<std::pair<int, float>>::iterator found;
        if (simTSs[i].seedID() == cPHandle_id) {
          found = std::find_if(detIdSimTSId_Map[hitId].begin(),
                               detIdSimTSId_Map[hitId].end(),
                               [=](const std::pair<int, float>& v) { return v.first == seedIndex; });
          if (found != detIdSimTSId_Map[hitId].end()) {
            const auto iLC = std::find(simTSs[i].vertices().begin(), simTSs[i].vertices().end(), v);
            const auto lcFraction =
                1.f / simTSs[i].vertex_multiplicity(std::distance(std::begin(simTSs[i].vertices()), iLC));
            simFraction = found->second * lcFraction;
          }
        } else {
          found = std::find_if(detIdSimClusterId_Map[hitId].begin(),
                               detIdSimClusterId_Map[hitId].end(),
                               [=](const std::pair<int, float>& v) { return v.first == seedIndex; });
          if (found != detIdSimClusterId_Map[hitId].end()) {
            simFraction = found->second;
          }
        }

        float hitEnergy = hitMap_->find(hitId)->second->energy();
        float hitEnergySquared = hitEnergy * hitEnergy;
        float simFractionSquared = simFraction * simFraction;
        denominator_simToReco[i] += simFractionSquared * hitEnergySquared;
        for (size_t j = 0; j < nTracksters; ++j) {
          float recoFraction = 0.f;

          auto found_reco =
              std::find_if(detIdToRecoTSId_Map[hitId].begin(),
                           detIdToRecoTSId_Map[hitId].end(),
                           [=](const std::pair<int, float>& v) { return v.first == static_cast<int>(j); });
          if (found_reco != detIdToRecoTSId_Map[hitId].end())
            recoFraction = found_reco->second;
          numerator_simToReco[i][j] +=
              std::min(simFractionSquared, (simFraction - recoFraction) * (simFraction - recoFraction)) *
              hitEnergySquared;
          sharedEnergy[i][j] += std::min(simFraction, recoFraction) * hitEnergy;
        }
      }
    }
  }

  std::vector<float> denominator_recoToSim(nTracksters, 0.f);
  std::vector<std::vector<float>> numerator_recoToSim(nTracksters);

  for (unsigned int i = 0; i < nTracksters; ++i) {
    numerator_recoToSim[i].resize(nSimTracksters, 0.f);
    const auto& lcsInTrackster = tracksters[i].vertices();
    for (const auto& v : lcsInTrackster) {
      for (const auto& haf : layerClusters[v].hitsAndFractions()) {
        const auto hitId = haf.first;
        float recoFraction = 0.f;

        auto found = std::find_if(detIdToRecoTSId_Map[hitId].begin(),
                                  detIdToRecoTSId_Map[hitId].end(),
                                  [=](const std::pair<int, float>& v) { return v.first == static_cast<int>(i); });
        if (found != detIdToRecoTSId_Map[hitId].end())
          recoFraction = found->second;

        float hitEnergy = hitMap_->find(hitId)->second->energy();
        float hitEnergySquared = hitEnergy * hitEnergy;
        float recoFractionSquared = recoFraction * recoFraction;
        denominator_recoToSim[i] += recoFractionSquared * hitEnergySquared;

        for (size_t j = 0; j < nSimTracksters; ++j) {
          float simFraction = 0.f;

          auto found_sim = std::find_if(detIdSimTSId_Map[hitId].begin(),
                                        detIdSimTSId_Map[hitId].end(),
                                        [=](const std::pair<int, float>& v) { return v.first == static_cast<int>(j); });
          if (found_sim != detIdSimTSId_Map[hitId].end())
            simFraction = found_sim->second;
          numerator_recoToSim[i][j] +=
              std::min(recoFractionSquared, (simFraction - recoFraction) * (simFraction - recoFraction)) *
              hitEnergySquared;
        }
      }
    }
  }

  // compute score

  for (size_t i = 0; i < nSimTracksters; ++i) {
    for (size_t j = 0; j < nTracksters; ++j) {
      simToReco_sharedEnergyAndScore[i][j].first = sharedEnergy[i][j];
      simToReco_sharedEnergyAndScore[i][j].second = numerator_simToReco[i][j] / denominator_simToReco[i];
      recoToSim_sharedEnergyAndScore[j][i].first = sharedEnergy[i][j];
      recoToSim_sharedEnergyAndScore[j][i].second = numerator_recoToSim[j][i] / denominator_recoToSim[j];
    }
  }

  return {recoToSim_sharedEnergyAndScore, simToReco_sharedEnergyAndScore};
}

hgcal::RecoToSimCollectionSimTracksters TSToSimTSHitLCAssociatorByEnergyScoreImpl::associateRecoToSim(
    const edm::Handle<ticl::TracksterCollection>& tCH,
    const edm::Handle<reco::CaloClusterCollection>& lCCH,
    const edm::Handle<SimClusterCollection>& sCCH,
    const edm::Handle<CaloParticleCollection>& cPCH,
    const edm::Handle<ticl::TracksterCollection>& sTCH) const {
  hgcal::RecoToSimCollectionSimTracksters returnValue(productGetter_);
  const auto& links = makeConnections(tCH, lCCH, sCCH, cPCH, sTCH);
  const auto& recoToSim_sharedEnergyAndScore = std::get<0>(links);
  for (std::size_t tsId = 0; tsId < recoToSim_sharedEnergyAndScore.size(); ++tsId) {
    std::size_t numSimTracksters = recoToSim_sharedEnergyAndScore[tsId].size();
    for (std::size_t simTsId = 0; simTsId < numSimTracksters; ++simTsId) {
      LogDebug("TSToSimTSHitLCAssociatorByEnergyScoreImpl")
          << " Trackster Id:\t" << tsId << "\tSimTrackster id:\t" << recoToSim_sharedEnergyAndScore[tsId][simTsId].first
          << "\tscore:\t" << recoToSim_sharedEnergyAndScore[tsId][simTsId].second << "\n";
      // Fill AssociationMap
      returnValue.insert(
          edm::Ref<ticl::TracksterCollection>(tCH, tsId),  // Ref to TS
          std::make_pair(
              edm::Ref<ticl::TracksterCollection>(sTCH, simTsId),
              std::make_pair(recoToSim_sharedEnergyAndScore[tsId][simTsId].first,
                             recoToSim_sharedEnergyAndScore[tsId][simTsId].second))  // Pair <Ref to ST, score>
      );
    }
  }
  return returnValue;
}

hgcal::SimToRecoCollectionSimTracksters TSToSimTSHitLCAssociatorByEnergyScoreImpl::associateSimToReco(
    const edm::Handle<ticl::TracksterCollection>& tCH,
    const edm::Handle<reco::CaloClusterCollection>& lCCH,
    const edm::Handle<SimClusterCollection>& sCCH,
    const edm::Handle<CaloParticleCollection>& cPCH,
    const edm::Handle<ticl::TracksterCollection>& sTCH) const {
  hgcal::SimToRecoCollectionSimTracksters returnValue(productGetter_);
  const auto& links = makeConnections(tCH, lCCH, sCCH, cPCH, sTCH);
  const auto& simToReco_sharedEnergyAndScore = std::get<1>(links);
  for (std::size_t simTsId = 0; simTsId < simToReco_sharedEnergyAndScore.size(); ++simTsId) {
    std::size_t numTracksters = simToReco_sharedEnergyAndScore[simTsId].size();
    for (std::size_t tsId = 0; tsId < numTracksters; ++tsId) {
      LogDebug("TSToSimTSHitLCAssociatorByEnergyScoreImpl")
          << "Trackster Id:\t" << tsId << "\tSimTrackster id:\t" << simTsId << " Shared energy "
          << simToReco_sharedEnergyAndScore[simTsId][tsId].first << "\tscore:\t"
          << simToReco_sharedEnergyAndScore[simTsId][tsId].second << "\n";
      // Fill AssociationMap
      returnValue.insert(edm::Ref<ticl::TracksterCollection>(sTCH, simTsId),
                         std::make_pair(edm::Ref<ticl::TracksterCollection>(tCH, tsId),
                                        std::make_pair(simToReco_sharedEnergyAndScore[simTsId][tsId].first,
                                                       simToReco_sharedEnergyAndScore[simTsId][tsId].second)));
    }
  }
  return returnValue;
}
