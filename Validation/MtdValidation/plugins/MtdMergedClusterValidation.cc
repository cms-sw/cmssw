#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/CommonTopologies/interface/GlobalTrackingGeometry.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociatorBaseImpl.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <iostream>
#include <cmath>
#include <CLHEP/Units/SystemOfUnits.h>
#include "DataFormats/Math/interface/GeantUnits.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedClusterFwd.h"
#include "DataFormats/FTLRecHit/interface/FTLMergedClusterCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociatorBaseImpl.h"

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class MtdMergedClusterValidation : public DQMEDAnalyzer {
public:
  explicit MtdMergedClusterValidation(const edm::ParameterSet&);
  ~MtdMergedClusterValidation() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const std::string folder_;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  edm::EDGetTokenT<FTLMergedClusterCollection> mergedClustersToken_;
  edm::EDGetTokenT<FTLClusterCollection> clustersToken_;

  edm::EDGetTokenT<MtdSimMergedClusterCollection> simMergedClustersToken_;
  edm::EDGetTokenT<MtdSimLayerClusterCollection> simClustersToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;

  edm::EDGetTokenT<reco::SimToTPCollectionMtd> sim2tpAssociationMapToken_;
  edm::EDGetTokenT<MtdRecoClusterToSimLayerClusterAssociationMap> r2sAssociationMapToken_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

  // RECO
  int totalMC2DFilled_ = 0;
  int totalMCClusterNotFound_ = 0;
  int totalMCGeometryFailed_ = 0;
  int totalMCWrongSize_ = 0;

  MonitorElement* h_mc_energy_;
  MonitorElement* h_mc_time_;
  MonitorElement* h_mc_timeError_;
  MonitorElement* h_mc_x_;
  MonitorElement* h_mc_y_;
  MonitorElement* h_mc_eta_;
  MonitorElement* h_mc_nClusters_;
  MonitorElement* h_cluster_energy_;
  MonitorElement* h_cluster_time_;
  MonitorElement* h_mc_cluster_distance_phi_;
  MonitorElement* h_mc_cluster_distance_eta_;
  MonitorElement* h_mc_cluster_distance_z_;
  MonitorElement* h_mc_cluster_distance_2D_;
  MonitorElement* h_mc_cluster_hitProdType_2D;
  MonitorElement* h_mc_cluster_distance_iphi_;
  MonitorElement* h_mc_cluster_distance_ieta_;
  MonitorElement* h_mc_cluster_distance_i2D_;

  MonitorElement* h_cluster_energy_eff_;
  MonitorElement* h_cluster_time_eff_;
  MonitorElement* h_comp_energy_;
  MonitorElement* h_comp_time_;

  // single-cluster check
  MonitorElement* h_single_dx_;
  MonitorElement* h_single_dy_;
  MonitorElement* h_single_dt_;
  MonitorElement* h_single_de_;
  MonitorElement* h_single_dt_outlier_;
  MonitorElement* h_single_de_outlier_;
  MonitorElement* h_mc_energy_minus_sumInputs_;

  MonitorElement* h_eta_adjacent_pairs_;
  MonitorElement* h_eta_merged_pairs_;
  MonitorElement* h_eta_merging_fraction_;

  MonitorElement* h_time_res_etaphi_[6][6];  // need assoc. map

  MonitorElement* h_mc_energy_vs_time_;
  MonitorElement* h_mc_xy_;
  MonitorElement* h_mc_energy_vs_nClusters_;
  MonitorElement* h_merging_fraction_;

  MonitorElement* h_merging_efficiency_;
  MonitorElement* h_merging_efficiency_vs_eta_;
  MonitorElement* h_eta_sameTrackID_pairs_;
  MonitorElement* h_eta_merged_sameTrackID_pairs_;

  // SIM
  MonitorElement* h_simmc_energy_;
  MonitorElement* h_simmc_logEnergy_;
  MonitorElement* h_simmc_time_;
  MonitorElement* h_simmc_x_;
  MonitorElement* h_simmc_y_;
  MonitorElement* h_simmc_nClusters_;
  MonitorElement* h_simmc_n_;

  MonitorElement* h_simmc_logEnergy_perCluster_;
  MonitorElement* h_simmc_time_perCluster_;
  MonitorElement* h_simmc_clusterType_;

  MonitorElement* h_simmc_xy_;
  MonitorElement* h_simmc_energy_vs_time_;
  MonitorElement* h_simmc_energy_vs_nClusters_;
  MonitorElement* h_simmc_primaryPt_vs_nClusters_;
  MonitorElement* h_simmc_primaryPt_vs_energy_;
  MonitorElement* h_simmc_primaryEnergy_vs_energy_;
  MonitorElement* h_simmc_primaryEnergy_vs_nClusters_;

  int evt_run_, evt_event_;

  // RECO
  int mc_n_;
  std::vector<float> mc_energy_, mc_time_, mc_timeError_, mc_x_, mc_y_;
  std::vector<int> mc_nClusters_;
  std::vector<uint32_t> mc_seedId_;
  std::vector<std::vector<uint32_t>> mc_clusterIds_;

  int cluster_n_;
  std::vector<float> cluster_energy_, cluster_time_, cluster_x_, cluster_y_;
  std::vector<uint32_t> cluster_detId_;
  std::vector<bool> cluster_inMergedCluster_;

  // SIM
  int simmc_n_;
  std::vector<float> simmc_energy_, simmc_time_, simmc_timeError_, simmc_x_, simmc_y_;
  std::vector<int> simmc_nClusters_;
  std::vector<std::vector<uint32_t>> simmc_iphi_perCluster_, simmc_ieta_perCluster_;
  std::vector<std::vector<float>> simmc_energy_perCluster_, simmc_time_perCluster_;
  std::vector<std::vector<uint32_t>> simmc_clusterType_;

  // Primary particle information per mergedcluster
  std::vector<float> simmc_primary_energy_, simmc_primary_et_, simmc_primary_phi_, simmc_primary_eta_;
  std::vector<int> simmc_primary_pdgId_;

  int totalAdjacentPairs_;
  int totalMergedPairs_;

  int sameTrackIdPairs_;
  int mergedSameTrackIDPairs_;
};

MtdMergedClusterValidation::MtdMergedClusterValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      mtdgeoToken_(esConsumes<MTDGeometry, MTDDigiGeometryRecord>()),
      mtdtopoToken_(esConsumes<MTDTopology, MTDTopologyRcd>()) {
  mergedClustersToken_ = consumes<FTLMergedClusterCollection>(iConfig.getParameter<edm::InputTag>("mergedClusters"));
  clustersToken_ = consumes<FTLClusterCollection>(iConfig.getParameter<edm::InputTag>("clusters"));

  simMergedClustersToken_ =
      consumes<MtdSimMergedClusterCollection>(iConfig.getParameter<edm::InputTag>("simMergedClusters"));
  simClustersToken_ = consumes<MtdSimLayerClusterCollection>(iConfig.getParameter<edm::InputTag>("simLayerClusters"));

  sim2tpAssociationMapToken_ =
      consumes<reco::SimToTPCollectionMtd>(iConfig.getParameter<edm::InputTag>("sim2tpAssociationMapTag"));
  r2sAssociationMapToken_ = consumes<MtdRecoClusterToSimLayerClusterAssociationMap>(
      iConfig.getParameter<edm::InputTag>("r2sAssociationMapTag"));

  totalAdjacentPairs_ = 0;
  totalMergedPairs_ = 0;
  sameTrackIdPairs_ = 0;
  mergedSameTrackIDPairs_ = 0;
}

void MtdMergedClusterValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace geant_units::operators;  // for energy conversion

  edm::Handle<FTLMergedClusterCollection> mergedClustersHandle;
  iEvent.getByToken(mergedClustersToken_, mergedClustersHandle);

  edm::Handle<FTLClusterCollection> clustersHandle;
  iEvent.getByToken(clustersToken_, clustersHandle);

  edm::Handle<MtdSimMergedClusterCollection> simMergedClustersHandle;
  iEvent.getByToken(simMergedClustersToken_, simMergedClustersHandle);

  edm::Handle<MtdSimLayerClusterCollection> mtdSimLCHandle;
  iEvent.getByToken(simClustersToken_, mtdSimLCHandle);

  edm::Handle<reco::SimToTPCollectionMtd> sim2tpAssociationMapHandle;
  iEvent.getByToken(sim2tpAssociationMapToken_, sim2tpAssociationMapHandle);
  const reco::SimToTPCollectionMtd& sim2tpAssociationMap = *sim2tpAssociationMapHandle;

  edm::Handle<MtdRecoClusterToSimLayerClusterAssociationMap> r2sAssociationMapHandle;
  iEvent.getByToken(r2sAssociationMapToken_, r2sAssociationMapHandle);
  const MtdRecoClusterToSimLayerClusterAssociationMap& r2sAssociationMap = *r2sAssociationMapHandle;

  if (!mergedClustersHandle.isValid() || !clustersHandle.isValid() || !simMergedClustersHandle.isValid() ||
      !mtdSimLCHandle.isValid()) {
    edm::LogWarning("MtdMergedClusterValidation") << "Invalid handles!"
                                                  << "  mergedClustersHandle: " << mergedClustersHandle.isValid()
                                                  << "  clustersHandle: " << clustersHandle.isValid()
                                                  << "  simMergedClustersHandle: " << simMergedClustersHandle.isValid()
                                                  << "  mtdSimLCHandle: " << mtdSimLCHandle.isValid();
    return;
  }

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  auto geomHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geomHandle.product();

  evt_run_ = iEvent.id().run();
  evt_event_ = iEvent.id().event();

  mc_energy_.clear();
  mc_time_.clear();
  mc_timeError_.clear();
  mc_x_.clear();
  mc_y_.clear();
  mc_nClusters_.clear();
  mc_seedId_.clear();
  mc_clusterIds_.clear();
  cluster_energy_.clear();
  cluster_time_.clear();
  cluster_x_.clear();
  cluster_y_.clear();
  cluster_detId_.clear();
  cluster_inMergedCluster_.clear();

  simmc_energy_.clear();
  simmc_time_.clear();
  simmc_x_.clear();
  simmc_y_.clear();
  simmc_nClusters_.clear();
  simmc_iphi_perCluster_.clear();
  simmc_ieta_perCluster_.clear();
  simmc_energy_perCluster_.clear();
  simmc_time_perCluster_.clear();
  simmc_clusterType_.clear();
  simmc_primary_energy_.clear();
  simmc_primary_et_.clear();
  simmc_primary_phi_.clear();
  simmc_primary_eta_.clear();
  simmc_primary_pdgId_.clear();

  // flatten detsetvector
  std::vector<const FTLMergedCluster*> mergedVec;
  mergedVec.reserve(mergedClustersHandle->size());
  for (const auto& detSet : *mergedClustersHandle) {
    for (const auto& mc : detSet) {
      mergedVec.push_back(&mc);
    }
  }
  mc_n_ = mergedVec.size();

  simmc_n_ = simMergedClustersHandle->size();

  std::set<uint32_t> clusterDetIds;
  std::map<uint32_t, int> detIdToSCIndex;
  std::unordered_map<const FTLCluster*, int> clusterPtrToSCIndex;
  int scIndex = 0;

  std::map<BTLDetId, const FTLCluster*> clusterMap;
  std::vector<const FTLCluster*> validClusters;
  cluster_n_ = 0;

  // ---------------------- //
  // -------- RECO -------- //
  // ---------------------- //

  // build map
  for (const auto& detSet : *clustersHandle) {
    for (const auto& cluster : detSet) {
      clusterMap[cluster.id()] = &cluster;  // ALL clusters in map
    }
  }

  for (const auto& detSet : *clustersHandle) {
    for (const auto& cluster : detSet) {
      if (cluster.energy() < 1.0)
        continue;

      cluster_n_++;
      validClusters.push_back(&cluster);
      //clusterMap[cluster.id()] = &cluster;

      h_cluster_energy_->Fill(cluster.energy());
      h_cluster_time_->Fill(cluster.time());
    }
  }

  // debug
  LogDebug("MtdMergedClusterValidation") << "Valid clusters for event " << evt_event_ << ":";
  for (const auto* cluster : validClusters) {
    LogTrace("MtdMergedClusterValidation") << "  DetId " << cluster->id().rawId();
  }

  std::unordered_map<uint32_t, std::vector<const FTLCluster*>> clustersByDetId;
  clustersByDetId.reserve(validClusters.size());
  for (const auto* c : validClusters) {
    clustersByDetId[c->id().rawId()].push_back(c);
  }

  std::unordered_set<const FTLCluster*> consumedRecoClusters;
  consumedRecoClusters.reserve(2 * mergedClustersHandle->size());

  auto pickBestMatch = [](const std::vector<const FTLCluster*>& v, double mcTime) -> const FTLCluster* {
    if (v.empty())
      return nullptr;
    const FTLCluster* best = v.front();
    double bestDt = std::abs(v.front()->time() - mcTime);
    for (const auto* c : v) {
      double dt = std::abs(c->time() - mcTime);
      if (dt < bestDt) {
        best = c;
        bestDt = dt;
      }
    }
    return best;
  };

  // process merged clusters and mark constituent clusters
  for (size_t mcIndex = 0; mcIndex < mergedVec.size(); ++mcIndex) {
    const FTLMergedCluster* mc = mergedVec[mcIndex];

    LogDebug("MtdMergedClusterValidation") << "Merged Cluster # " << mcIndex << *mc;

    const MTDGeomDet* mcdet = geom->idToDet(mc->id());
    if (mcdet) {
      GlobalPoint mc_global_point = mcdet->surface().toGlobal(LocalPoint(mc->x(), mc->y(), 0));
      h_mc_eta_->Fill(mc_global_point.eta());
    }

    h_mc_energy_->Fill(mc->energy());
    h_mc_time_->Fill(mc->time());
    h_mc_timeError_->Fill(mc->timeError());
    h_mc_x_->Fill(mc->x());
    h_mc_y_->Fill(mc->y());
    h_mc_nClusters_->Fill(mc->nClusters());
    h_mc_energy_vs_time_->Fill(mc->time(), mc->energy());
    h_mc_xy_->Fill(mc->x(), mc->y());
    h_mc_energy_vs_nClusters_->Fill(mc->nClusters(), mc->energy());

    mc_energy_.push_back(mc->energy());
    mc_time_.push_back(mc->time());
    mc_timeError_.push_back(mc->timeError());
    mc_x_.push_back(mc->x());
    mc_y_.push_back(mc->y());
    mc_nClusters_.push_back(mc->nClusters());
    mc_seedId_.push_back(mc->id().rawId());

    // da budemo sigurni
    bool seedFound = false;
    for (const auto& did : mc->clusterIds()) {
      if (did == mc->id()) {
        seedFound = true;
        break;
      }
    }
    if (!seedFound) {
      edm::LogWarning("MtdMergedClusterValidation")
          << "[WARN] Validation: merged cluster seed " << mc->id().rawId()
          << " not found among clusterIds() (nClusters=" << mc->nClusters() << ")";
    }

    std::set<uint32_t> ids;
    for (const auto& did : mc->clusterIds()) {
      if (!ids.insert(did.rawId()).second) {
        edm::LogWarning("MtdMergedClusterValidation")
            << "[WARN] Validation: duplicate detId " << did.rawId() << " in merged cluster seed " << mc->id().rawId();
      }
      if (clusterMap.find(BTLDetId(did)) == clusterMap.end()) {
        LogDebug("MtdMergedClusterValidation")
            << "[WARN] Validation: clusterId " << did.rawId() << " references reco DetId " << did.rawId()
            << " not found in input clusters";
      }
    }

    if (mc->nClusters() == 1) {
      uint32_t detRaw = mc->clusterIds()[0].rawId();
      auto itVec = clustersByDetId.find(detRaw);
      if (itVec != clustersByDetId.end() && !itVec->second.empty()) {
        const FTLCluster* original = pickBestMatch(itVec->second, mc->time());
        if (original == nullptr)
          original = itVec->second.front();
#ifdef EDM_ML_DEBUG
        for (int ihit = 0; ihit < original->size(); ++ihit) {
          auto thisHit = original->hit(ihit);
          LogTrace("MtdMergedClusterValidation")
              << "Cluster hit " << ihit << " row/col = " << thisHit.x() << " " << thisHit.y()
              << " energy = " << thisHit.energy() << " time = " << thisHit.time() << " +/- " << thisHit.time_error();
          int hit_row = original->minHitRow() + original->hitOffset()[ihit * 2];
          int hit_col = original->minHitCol() + original->hitOffset()[ihit * 2 + 1];
          if (hit_row != thisHit.x() || hit_col != thisHit.y()) {
            edm::LogWarning("MtdMergedClusterValidation")
                << "Index in cluster memory not consistent, row/col = " << hit_row << " " << hit_col;
          }
        }
#endif
        float dx = mc->x() - original->x();
        float dy = mc->y() - original->y();
        h_single_dx_->Fill(dx);
        h_single_dy_->Fill(dy);
        float dt = mc->time() - original->time();
        float de = mc->energy() - original->energy();
        h_single_dt_->Fill(dt);
        h_single_de_->Fill(de);
      } else {
        LogDebug("MtdMergedClusterValidation")
            << "[WARN] Validation: single-cluster merged cluster references reco DetId " << mc->clusterIds()[0].rawId()
            << " not found in input clusters";
      }
    }
    // energy consistency check
    double sumInputs = 0.0;
    std::vector<std::vector<const FTLCluster*>> candidates;
    candidates.reserve(mc->nClusters());
    bool anyEmpty = false;

    for (const auto& detId : mc->clusterIds()) {
      uint32_t dr = detId.rawId();
      auto it = clustersByDetId.find(dr);
      if (it == clustersByDetId.end() || it->second.empty()) {
        candidates.emplace_back();
        anyEmpty = true;
      } else {
        candidates.emplace_back(it->second.begin(), it->second.end());
      }
    }

    if (anyEmpty) {
      sumInputs = 0.0;
    } else {
      size_t nComb = 1;
      for (const auto& v : candidates) {
        nComb *= std::max<size_t>(1, v.size());
        if (nComb > 2000)
          break;
      }

      if (nComb <= 2000) {
        double bestSum = 0.0;
        double bestDiff = 1e9;
        std::vector<const FTLCluster*> current;
        current.resize(candidates.size(), nullptr);

        std::function<void(size_t)> dfs = [&](size_t idx) {
          if (idx == candidates.size()) {
            double s = 0.0;
            for (const auto* c : current)
              s += c->energy();
            double diff = std::abs(s - mc->energy());
            if (diff < bestDiff) {
              bestDiff = diff;
              bestSum = s;
            }
            return;
          }
          for (const auto* cand : candidates[idx]) {
            current[idx] = cand;
            dfs(idx + 1);
          }
        };
        dfs(0);
        sumInputs = bestSum;
      } else {
        for (const auto& detId : mc->clusterIds()) {
          uint32_t dr = detId.rawId();
          auto it2 = clustersByDetId.find(dr);
          if (it2 != clustersByDetId.end() && !it2->second.empty()) {
            const FTLCluster* best = pickBestMatch(it2->second, mc->time());
            if (best)
              sumInputs += best->energy();
          }
        }
      }
    }
    h_mc_energy_minus_sumInputs_->Fill(mc->energy() - sumInputs);
    float deltaE = mc->energy() - static_cast<float>(sumInputs);
    if (std::abs(deltaE) > 1e-3f) {
      if (std::abs(deltaE) > 0.5f) {
        LogDebug("MtdMergedClusterValidation")
            << "[OUTLIER] Merged energy differs from sum(inputs) by " << deltaE << " MeV for seed " << mc->id().rawId()
            << " nClusters=" << mc->nClusters();
        LogDebug("MtdMergedClusterValidation") << "  MergedCluster: id=" << mc->id().rawId() << " E=" << mc->energy()
                                               << " t=" << mc->time() << " nIds=" << mc->nClusters();
        for (const auto& detId : mc->clusterIds()) {
          uint32_t dr = detId.rawId();
          LogDebug("MtdMergedClusterValidation") << "  Constituent DetId: " << dr;
          auto itVec = clustersByDetId.find(dr);
          if (itVec == clustersByDetId.end() || itVec->second.empty()) {
            LogDebug("MtdMergedClusterValidation") << "    NO candidates found in reco collection for this DetId\n";
            continue;
          }
          // list all candidates
          int idx = 0;
          for (const auto* cand : itVec->second) {
            LogDebug("MtdMergedClusterValidation")
                << "    cand[" << idx++ << "] ptr=" << cand << " E=" << cand->energy() << " t=" << cand->time()
                << " tErr=" << cand->timeError();
          }
          // show chosen best (by pickBestMatch used above)
          const FTLCluster* chosen = pickBestMatch(itVec->second, mc->time());
          if (chosen) {
            LogDebug("MtdMergedClusterValidation") << "    chosen ptr=" << chosen << " E=" << chosen->energy()
                                                   << " t=" << chosen->time() << " tErr=" << chosen->timeError();
          } else {
            LogDebug("MtdMergedClusterValidation") << "    pickBestMatch returned nullptr\n";
          }
        }  // per-DetId loop
        LogDebug("MtdMergedClusterValidation") << "  End debug for merged seed " << mc->id().rawId() << "\n";
      }
    }
    LogDebug("MtdMergedClusterValidation") << "Filling eff histogram: E=" << mc->energy() << " T=" << mc->time();
    h_cluster_energy_eff_->Fill(mc->energy());
    h_cluster_time_eff_->Fill(mc->time());

    const auto& cluIds = mc->clusterIds();

    //LogDebug("MtdMergedClusterValidation") << "[DEBUG] MergedCluster created with " << mc.nClusters()
    //  << " clusters, IDs: ";
    for (const auto& id : mc->clusterIds()) {
      LogDebug("MtdMergedClusterValidation") << id.rawId() << " ";
    }
    LogDebug("MtdMergedClusterValidation");
    if (cluIds.size() == 2) {
      std::vector<GlobalPoint> clusterPositions;

      for (const auto& detId : cluIds) {
        auto it = clusterMap.find(BTLDetId(detId));
        if (it != clusterMap.end()) {
          const FTLCluster* cluster = it->second;
          BTLDetId cluId = cluster->id();

          const MTDGeomDet* thedet = geom->idToDet(cluId);
          if (thedet) {
            GlobalPoint global_point = thedet->surface().toGlobal(LocalPoint(cluster->x(), cluster->y(), 0));
            clusterPositions.push_back(global_point);
          } else {
            LogDebug("MtdMergedClusterValidation") << "Warning: no geometry for cluster detid " << cluId.rawId();
          }
        } else {
          LogDebug("MtdMergedClusterValidation")
              << "Warning: cluster detid " << detId.rawId() << " not found in cluster map.";
        }
      }

      if (clusterPositions.size() == 2) {
        double dz = clusterPositions[0].z() - clusterPositions[1].z();
        h_mc_cluster_distance_z_->Fill(dz);
        double deta = clusterPositions[0].eta() - clusterPositions[1].eta();
        double dphi = clusterPositions[0].phi() - clusterPositions[1].phi();
        if (dphi > M_PI)
          dphi -= 2 * M_PI;
        else if (dphi < -M_PI)
          dphi += 2 * M_PI;
        h_mc_cluster_distance_eta_->Fill(deta);
        h_mc_cluster_distance_phi_->Fill(dphi);
        h_mc_cluster_distance_2D_->Fill(dphi, dz);
      }
    }

    if (cluIds.size() == 2) {
      std::vector<std::pair<uint32_t, uint32_t>> indices;
      for (const auto& detId : cluIds) {
        std::pair<uint32_t, uint32_t> idx =
            topology->btlIndex(BTLDetId(detId).geographicalId(BTLDetId::CrysLayout::v4).rawId());
        indices.push_back(idx);
      }

      if (indices.size() == 2) {
        int deltaIeta = static_cast<int>(indices[0].second) - static_cast<int>(indices[1].second);
        int deltaIphi = static_cast<int>(indices[0].first) - static_cast<int>(indices[1].first);
        h_mc_cluster_distance_ieta_->Fill(deltaIeta);
        h_mc_cluster_distance_iphi_->Fill(deltaIphi);
        h_mc_cluster_distance_i2D_->Fill(deltaIphi, deltaIeta);

        h_mc_cluster_distance_ieta_->Fill(-deltaIeta);
        h_mc_cluster_distance_iphi_->Fill(-deltaIphi);
        h_mc_cluster_distance_i2D_->Fill(-deltaIphi, -deltaIeta);
      }
    }

    std::vector<uint32_t> clusterIds;
    for (const auto& detId : mc->clusterIds()) {
      clusterIds.push_back(detId.rawId());
      //clusterDetIds.insert(detId.rawId()); // mark these as merged
      //detIdToSCIndex[detId.rawId()] = scIndex; // ovo je bilo sa staron kolekcijon
    }
    // s novom kolekcijon je ovo dalje:
    if (mc->nClusters() > 1) {
      for (const auto& detId : mc->clusterIds()) {
        detIdToSCIndex[detId.rawId()] = scIndex;  // mark these as true merged
      }
    }
    // i onda ide isto sta i prije:
    mc_clusterIds_.push_back(clusterIds);
    scIndex++;

    //debug detIdtoscindex
    LogDebug("MtdMergedClusterValidation") << "detIdToSCIndex contents for event " << evt_event_ << ":";
    for (const auto& entry : detIdToSCIndex) {
      LogDebug("MtdMergedClusterValidation") << "  DetId " << entry.first << " -> MergedCluster " << entry.second;
    }

    for (size_t scIndex = 0; scIndex < mergedVec.size(); ++scIndex) {
      const FTLMergedCluster* mc = mergedVec[scIndex];
      for (const auto& detId : mc->clusterIds()) {
        auto it = clustersByDetId.find(detId.rawId());
        if (it != clustersByDetId.end()) {
          const FTLCluster* best = pickBestMatch(it->second, mc->time());
          if (best) {
            clusterPtrToSCIndex[best] = scIndex;
            consumedRecoClusters.insert(best);
          }
        }
      }
    }
  }

  std::set<uint32_t> countedDetIds;
  for (const auto* cluster : validClusters) {
    if (countedDetIds.count(cluster->id().rawId()))
      continue;
    countedDetIds.insert(cluster->id().rawId());
    bool inMergedCluster = detIdToSCIndex.count(cluster->id().rawId()) > 0;

    cluster_energy_.push_back(cluster->energy());
    cluster_time_.push_back(cluster->time());
    cluster_x_.push_back(cluster->x());
    cluster_y_.push_back(cluster->y());
    cluster_detId_.push_back(cluster->id().rawId());
    cluster_inMergedCluster_.push_back(inMergedCluster);

    if (!inMergedCluster) {
      //LogDebug("MtdMergedClusterValidation") << "[DEBUG] Filling eff with STANDALONE cluster: E=" << cluster->energy()
      //      << " MeV, T=" << cluster->time() << " ns" ;
      h_cluster_energy_eff_->Fill(cluster->energy());
      h_cluster_time_eff_->Fill(cluster->time());
    }
  }

  LogDebug("MtdMergedClusterValidation") << "MergedClusters and their constituent DetIds for event " << evt_event_
                                         << ":";
  for (size_t i = 0; i < mergedVec.size(); ++i) {
    const FTLMergedCluster* mc = mergedVec[i];
    LogDebug("MtdMergedClusterValidation") << "  MergedCluster " << i << ": ";
    for (const auto& detId : mc->clusterIds())
      LogDebug("MtdMergedClusterValidation") << detId.rawId() << " ";
    LogDebug("MtdMergedClusterValidation");
  }

  std::set<uint32_t> validDetIds;
  for (const auto* cluster : validClusters) {
    validDetIds.insert(cluster->id().rawId());
  }

  for (const auto& entry : detIdToSCIndex) {
    if (validDetIds.find(entry.first) == validDetIds.end()) {
      LogDebug("MtdMergedClusterValidation")
          << "WARNING: DetId " << entry.first << " in detIdToSCIndex but not in validClusters!";
    }
  }
  for (const auto& detId : validDetIds) {
    if (detIdToSCIndex.find(detId) == detIdToSCIndex.end()) {
      LogDebug("MtdMergedClusterValidation")
          << "WARNING: DetId " << detId << " in validClusters but not in detIdToSCIndex!";
    }
  }
  for (const auto& pair : clustersByDetId) {
    if (pair.second.size() > 1) {
      LogDebug("MtdMergedClusterValidation") << "WARNING: Multiple clusters with DetId " << pair.first;
    }
  }

  std::set<std::pair<uint32_t, uint32_t>> processedPairs;

  int adjacentPairs = 0;
  int mergedPairs = 0;
  int sameTrackIDPairs = 0;
  int mergedSameTrackIDPairs = 0;

  if (evt_event_ <= 3) {
    LogDebug("MtdMergedClusterValidation") << "Checking " << validClusters.size() << " clusters for adjacency:";
  }

  //WIP
  for (const auto* cluster : validClusters) {
    BTLDetId cluId = cluster->id();

    // get eta of the cluster
    const MTDGeomDet* thedet = geom->idToDet(cluId);
    if (!thedet)
      continue;
    GlobalPoint global_point = thedet->surface().toGlobal(LocalPoint(cluster->x(), cluster->y(), 0));
    double cluster_eta = global_point.eta();

    // topology indices
    std::pair<uint32_t, uint32_t> indices = topology->btlIndex(cluId.geographicalId(BTLDetId::CrysLayout::v4).rawId());
    uint32_t iphi = indices.first;
    uint32_t ieta = indices.second;

    if (iphi == std::numeric_limits<uint32_t>::max() || ieta == std::numeric_limits<uint32_t>::max()) {
      continue;
    }

    // Check edge hits for eta direction
    bool edgeHitIn0 = false;
    bool edgeHitIn15 = false;
    for (int i = 0; i < cluster->size(); ++i) {
      auto hit = cluster->hit(i);
      int hit_col = hit.y();
      if (hit_col == 0)
        edgeHitIn0 = true;
      else if (hit_col == 15)
        edgeHitIn15 = true;
    }
    bool hasEdgeHit = edgeHitIn0 || edgeHitIn15;

    //eta
    if (hasEdgeHit) {
      std::vector<int> etaOffsets = {1, -1};
      for (int etaOffset : etaOffsets) {
        uint32_t adjDetIdRaw = topology->btlidFromIndex(iphi, ieta + etaOffset);
        if (adjDetIdRaw == 0)
          continue;

        BTLDetId adjDetId(adjDetIdRaw);
        auto it = clusterMap.find(adjDetId);
        if (it == clusterMap.end())
          continue;

        const FTLCluster* adjCluster = it->second;

        // Check for opposite edge hit in neighbor
        bool hasOppositeEdgeHit = false;
        for (int j = 0; j < adjCluster->size(); ++j) {
          auto hit = adjCluster->hit(j);
          int hit_col = hit.y();
          if ((edgeHitIn0 && hit_col == 15) || (edgeHitIn15 && hit_col == 0)) {
            hasOppositeEdgeHit = true;
            break;
          }
        }

        if (hasOppositeEdgeHit) {
          std::pair<uint32_t, uint32_t> pairKey = (cluId.rawId() < adjDetId.rawId())
                                                      ? std::make_pair(cluId.rawId(), adjDetId.rawId())
                                                      : std::make_pair(adjDetId.rawId(), cluId.rawId());

          // Skip if already processed
          if (processedPairs.find(pairKey) != processedPairs.end()) {
            continue;
          }
          processedPairs.insert(pairKey);

          adjacentPairs++;
          h_eta_adjacent_pairs_->Fill(cluster_eta);

          edm::Ref<edmNew::DetSetVector<FTLCluster>, FTLCluster> currentClusterRef =
              edmNew::makeRefTo(clustersHandle, cluster);
          edm::Ref<edmNew::DetSetVector<FTLCluster>, FTLCluster> adjacentClusterRef =
              edmNew::makeRefTo(clustersHandle, adjCluster);

          std::unordered_set<int> currentTrackIDs;
          auto currentSimClustersRange = r2sAssociationMap.equal_range(currentClusterRef);

          // find track ids
          for (auto assocIt = currentSimClustersRange.first; assocIt != currentSimClustersRange.second; ++assocIt) {
            const auto& simClusterRefs = assocIt->second;
            for (const auto& simClusterRef : simClusterRefs) {
              //if (simClusterRef->hitProdType() != 0) continue; // Only direct matches

              auto tpRefs = sim2tpAssociationMap.find(simClusterRef);
              if (tpRefs != sim2tpAssociationMap.end()) {
                for (const auto& tpRef : tpRefs->val) {
                  for (const auto& g4Track : tpRef->g4Tracks()) {
                    currentTrackIDs.insert(g4Track.trackId());
                  }
                }
              }
            }
          }

          std::unordered_set<int> adjacentTrackIDs;
          auto adjacentSimClustersRange = r2sAssociationMap.equal_range(adjacentClusterRef);

          for (auto assocIt = adjacentSimClustersRange.first; assocIt != adjacentSimClustersRange.second; ++assocIt) {
            const auto& simClusterRefs = assocIt->second;
            for (const auto& simClusterRef : simClusterRefs) {
              //if (simClusterRef->hitProdType() != 0) continue;

              auto tpRefs = sim2tpAssociationMap.find(simClusterRef);
              if (tpRefs != sim2tpAssociationMap.end()) {
                for (const auto& tpRef : tpRefs->val) {
                  for (const auto& g4Track : tpRef->g4Tracks()) {
                    adjacentTrackIDs.insert(g4Track.trackId());
                  }
                }
              }
            }
          }

          // do they share trackid?
          bool hasMatchingTrackID = false;
          for (const auto& trackID : currentTrackIDs) {
            if (adjacentTrackIDs.find(trackID) != adjacentTrackIDs.end()) {
              hasMatchingTrackID = true;
              break;
            }
          }

          // Check if both clusters are in the same MergedCluster
          uint32_t detId1 = cluId.rawId();
          uint32_t detId2 = adjDetId.rawId();

          auto it1 = detIdToSCIndex.find(detId1);
          auto it2 = detIdToSCIndex.find(detId2);

          // check if the detids of both clusters in the pair are present in the same mc, not if the pointers are the same
          bool bothInSC = (it1 != detIdToSCIndex.end()) && (it2 != detIdToSCIndex.end());
          bool sameIndexMerged = bothInSC && (it1->second == it2->second);

          if (sameIndexMerged) {
            mergedPairs++;
            h_eta_merged_pairs_->Fill(cluster_eta);
          }

          if (hasMatchingTrackID) {
            sameTrackIDPairs++;
            h_eta_sameTrackID_pairs_->Fill(cluster_eta);

            if (sameIndexMerged) {
              mergedSameTrackIDPairs++;
              h_eta_merged_sameTrackID_pairs_->Fill(cluster_eta);
            }
          }
#ifdef EDM_ML_DEBUG
          if (hasMatchingTrackID && !sameIndexMerged) {
            LogDebug("MtdMergedClusterValidation") << "Not merged but same TrackID clusters: ";

            if (clusterPtrToSCIndex.count(cluster)) {
              LogDebug("MtdMergedClusterValidation")
                  << "  Cluster " << cluster->id().rawId() << " is in MergedCluster " << clusterPtrToSCIndex[cluster];
            } else {
              LogDebug("MtdMergedClusterValidation")
                  << "  Cluster " << cluster->id().rawId() << " is NOT in any MergedCluster";
            }
            if (clusterPtrToSCIndex.count(adjCluster)) {
              LogDebug("MtdMergedClusterValidation") << "  Cluster " << adjCluster->id().rawId()
                                                     << " is in MergedCluster " << clusterPtrToSCIndex[adjCluster];
            } else {
              LogDebug("MtdMergedClusterValidation")
                  << "  Cluster " << adjCluster->id().rawId() << " is NOT in any MergedCluster";
            }
            LogDebug("MtdMergedClusterValidation") << "  Cluster 1 edge hits: ";
            for (int i = 0; i < cluster->size(); ++i) {
              auto hit = cluster->hit(i);
              if (hit.y() == 0 || hit.y() == 15) {
                LogDebug("MtdMergedClusterValidation") << hit.y() << " ";
              }
            }
            LogDebug("MtdMergedClusterValidation");

            LogDebug("MtdMergedClusterValidation") << "  Cluster 2 edge hits: ";
            for (int j = 0; j < adjCluster->size(); ++j) {
              auto hit = adjCluster->hit(j);
              if (hit.y() == 0 || hit.y() == 15) {
                LogDebug("MtdMergedClusterValidation") << hit.y() << " ";
              }
            }
            LogDebug("MtdMergedClusterValidation");

            LogDebug("MtdMergedClusterValidation")
                << "  Cluster 1 time: " << cluster->time() << " ± " << cluster->timeError();
            LogDebug("MtdMergedClusterValidation")
                << "  Cluster 2 time: " << adjCluster->time() << " ± " << adjCluster->timeError();
            double dt = std::abs(cluster->time() - adjCluster->time());
            double combinedError = std::sqrt(std::pow(cluster->timeError(), 2) + std::pow(adjCluster->timeError(), 2));
            LogDebug("MtdMergedClusterValidation") << "  Δt = " << dt << ", threshold = " << (10 * combinedError);

            LogDebug("MtdMergedClusterValidation") << "  Cluster 1 ieta, iphi: " << ieta << ", " << iphi;
            auto adjIndices = topology->btlIndex(adjDetId.geographicalId(BTLDetId::CrysLayout::v4).rawId());
            LogDebug("MtdMergedClusterValidation")
                << "  Cluster 2 ieta, iphi: " << adjIndices.second << ", " << adjIndices.first;

            LogDebug("MtdMergedClusterValidation") << "  Cluster 1 energy: " << cluster->energy();
            LogDebug("MtdMergedClusterValidation") << "  Cluster 2 energy: " << adjCluster->energy();
          }
#endif

          if (evt_event_ <= 3 && adjacentPairs <= 10) {
            LogDebug("MtdMergedClusterValidation")
                << "  ETA adjacent pair " << adjacentPairs << ": " << detId1 << " <-> " << detId2
                << ", bothInSC=" << bothInSC << ", sameIndex=" << sameIndexMerged;
            if (bothInSC) {
              LogDebug("MtdMergedClusterValidation") << " (SC indices: " << it1->second << "," << it2->second << ")";
            }
            LogDebug("MtdMergedClusterValidation");
          }
        }
      }
    }

    //phi
    /*std::vector<int> phiOffsets = {1, -1};
        for (int phiOffset : phiOffsets) {
            uint32_t adjDetIdRaw = topology->btlidFromIndex(iphi + phiOffset, ieta);
            if (adjDetIdRaw == 0) continue;
            
            BTLDetId adjDetId(adjDetIdRaw);
            auto it = clusterMap.find(adjDetId);
            if (it == clusterMap.end()) continue;

            std::pair<uint32_t, uint32_t> pairKey = 
                (cluId.rawId() < adjDetId.rawId()) ? 
                std::make_pair(cluId.rawId(), adjDetId.rawId()) :
                std::make_pair(adjDetId.rawId(), cluId.rawId());
            
            // Skip if already processed
            if (processedPairs.find(pairKey) != processedPairs.end()) {
                continue;
            }
            processedPairs.insert(pairKey);
            
            adjacentPairs++;

            h_eta_adjacent_pairs_->Fill(cluster_eta);

            // Check if both clusters are in the same MergedCluster
            uint32_t detId1 = cluId.rawId();
            uint32_t detId2 = adjDetId.rawId();
            
            auto it1 = detIdToSCIndex.find(detId1);
            auto it2 = detIdToSCIndex.find(detId2);
            
            bool bothInSC = (it1 != detIdToSCIndex.end()) && (it2 != detIdToSCIndex.end());
            bool sameIndexMerged = bothInSC && (it1->second == it2->second);
            
            if (sameIndexMerged) {
                mergedPairs++;

                h_eta_merged_pairs_->Fill(cluster_eta);
            }
            
            if (evt_event_ <= 3 && adjacentPairs <= 10) {
                LogDebug("MtdMergedClusterValidation") << "  PHI adjacent pair " << adjacentPairs << ": " 
                          << detId1 << " <-> " << detId2 
                          << ", bothInSC=" << bothInSC
                          << ", sameIndex=" << sameIndexMerged;
                if (bothInSC) {
                    LogDebug("MtdMergedClusterValidation") << " (SC indices: " << it1->second << "," << it2->second << ")";
                }
                LogDebug("MtdMergedClusterValidation") ;
            }
        }*/
  }

  totalAdjacentPairs_ += adjacentPairs;
  totalMergedPairs_ += mergedPairs;
  sameTrackIdPairs_ += sameTrackIDPairs;
  mergedSameTrackIDPairs_ += mergedSameTrackIDPairs;

  LogDebug("MtdMergedClusterValidation") << "Event " << evt_event_ << ": " << mc_n_ << " MergedClusters, " << cluster_n_
                                         << " individual clusters";
  LogDebug("MtdMergedClusterValidation") << "Adjacent pairs: " << adjacentPairs << ", Merged pairs: " << mergedPairs;
  LogDebug("MtdMergedClusterValidation") << "Same TrackID pairs: " << sameTrackIDPairs
                                         << ", Merged same TrackID pairs: " << mergedSameTrackIDPairs;

  if (h_eta_adjacent_pairs_->getTH1F()->GetEntries() > 0) {
    h_eta_merging_fraction_->getTH1F()->Divide(
        h_eta_merged_pairs_->getTH1F(), h_eta_adjacent_pairs_->getTH1F(), 1, 1, "B");
  }

  if (h_eta_sameTrackID_pairs_->getTH1F()->GetEntries() > 0) {
    h_merging_efficiency_->getTH1F()->Divide(
        h_eta_merged_sameTrackID_pairs_->getTH1F(), h_eta_sameTrackID_pairs_->getTH1F(), 1, 1, "B");
  }

  if (h_cluster_energy_eff_->getTH1F()->GetEntries() > 0) {
    h_comp_energy_->getTH1F()->Add(h_cluster_energy_eff_->getTH1F());
    h_comp_energy_->getTH1F()->Add(h_cluster_energy_->getTH1F(), -1);
  }

  if (h_cluster_time_eff_->getTH1F()->GetEntries() > 0) {
    h_comp_time_->getTH1F()->Add(h_cluster_time_eff_->getTH1F());
    h_comp_time_->getTH1F()->Add(h_cluster_time_->getTH1F(), -1);
  }

  TH1F* h_energy_ind = h_cluster_energy_->getTH1F();
  TH1F* h_energy_eff = h_cluster_energy_eff_->getTH1F();

  double eventIndividualEnergy = 0.0;
  double eventEffectiveEnergy = 0.0;

  for (int i = 1; i <= h_energy_ind->GetNbinsX(); ++i) {
    eventIndividualEnergy += h_energy_ind->GetBinContent(i) * h_energy_ind->GetBinCenter(i);
    eventEffectiveEnergy += h_energy_eff->GetBinContent(i) * h_energy_eff->GetBinCenter(i);
  }

  // ---------------------- //
  // -------- SIM --------- //
  // ---------------------- //

  for (const auto& simmc : *simMergedClustersHandle) {
    LogDebug("MtdMergedClusterValidation") << "\tSC: time = " << simmc.simTime();
    LogDebug("MtdMergedClusterValidation") << "\tSC: position: " << simmc.simPos();
    LogDebug("MtdMergedClusterValidation") << "\tSC: energy = " << simmc.simEnergy();
    LogDebug("MtdMergedClusterValidation") << "\tSC: #clusters = " << simmc.clusters().size();

    auto energy = convertUnitsTo(0.001_MeV, simmc.simEnergy());  // convert energy from GeV to MeV
    auto time = simmc.simTime();
    auto nClusters = simmc.clusters().size();

    h_simmc_energy_->Fill(energy);
    h_simmc_logEnergy_->Fill(log10(energy));
    h_simmc_time_->Fill(time);
    h_simmc_nClusters_->Fill(nClusters);

    h_simmc_energy_vs_nClusters_->Fill(nClusters, energy);
    h_simmc_energy_vs_time_->Fill(time, energy);

    // retrieve detId (-> geometry) of earliest cluster
    BTLDetId detId = simmc.simDetId();
    DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));

    const MTDGeomDet* thedet = geom->idToDet(geoId);
    // Convert simLC local position to global position
    GlobalPoint global_point(0, 0, 0);
    if (thedet != nullptr) {
      // get global position of cluster
      global_point = thedet->toGlobal(simmc.simPos());
    } else {
      LogDebug("MtdMergedClusterValidation") << "\tSC: WARNING - no geometry for detId " << detId.rawId();
      global_point = GlobalPoint(-999, -999, -999);
    }

    h_simmc_x_->Fill(global_point.x());
    h_simmc_y_->Fill(global_point.y());

    h_simmc_xy_->Fill(global_point.x(), global_point.y());

    // Tree filling
    simmc_energy_.push_back(energy);
    simmc_time_.push_back(time);
    // simmc_timeError_.push_back(simmc.timeError());
    simmc_x_.push_back(global_point.x());
    simmc_y_.push_back(global_point.y());
    simmc_nClusters_.push_back(nClusters);

    // Collect per-cluster information for this mergedcluster
    std::vector<uint32_t> iphi_perCluster;
    std::vector<uint32_t> ieta_perCluster;
    std::vector<float> energy_perCluster;
    std::vector<float> time_perCluster;
    std::vector<uint32_t> clusterType_perCluster;
    if (simmc.clusters().size() > 1) {
      h_mc_cluster_hitProdType_2D->Fill((*simmc.clusters().at(0)).hitProdType(),
                                        (*simmc.clusters().at(1)).hitProdType());
    }
    // Access individual clusters from the mergedcluster
    for (const auto& cluster_ref : simmc.clusters()) {
      const auto& cluster = *cluster_ref;

      // Get detId from first hit (following MtdSimMergedClusterProducer pattern)
      if (cluster.detIds_and_rows().empty())
        continue;
      if (MTDDetId(cluster.detIds_and_rows()[0].first).mtdSubDetector() == 2)
        continue;  // Skip clusters from ETL

      BTLDetId clusterDetId(cluster.detIds_and_rows()[0].first);

      // Get topology indices for this cluster
      std::pair<uint32_t, uint32_t> cluster_indices =
          topology->btlIndex(clusterDetId.geographicalId(BTLDetId::CrysLayout::v4).rawId());
      iphi_perCluster.push_back(cluster_indices.first);
      ieta_perCluster.push_back(cluster_indices.second);

      // Store cluster energy and time using MtdSimLayerCluster methods
      energy_perCluster.push_back(convertUnitsTo(0.001_MeV, cluster.simLCEnergy()));  // convert GeV to MeV
      time_perCluster.push_back(cluster.simLCTime());

      // Store cluster type (hitProdType)
      clusterType_perCluster.push_back(cluster.hitProdType());

      // Fill histograms
      h_simmc_logEnergy_perCluster_->Fill(log10(convertUnitsTo(0.001_MeV, cluster.simLCEnergy())));
      h_simmc_time_perCluster_->Fill(cluster.simLCTime());
      h_simmc_clusterType_->Fill(cluster.hitProdType());

      LogDebug("MtdMergedClusterValidation")
          << "\t\t cluster detId: " << clusterDetId.rawId() << " iPhi: " << cluster_indices.first
          << " iEta: " << cluster_indices.second << " E: " << convertUnitsTo(0.001_MeV, cluster.simLCEnergy()) << " MeV"
          << " t: " << cluster.simLCTime() << " ns"
          << " hitProdType: " << cluster.hitProdType();
    }

    simmc_iphi_perCluster_.push_back(iphi_perCluster);
    simmc_ieta_perCluster_.push_back(ieta_perCluster);
    simmc_energy_perCluster_.push_back(energy_perCluster);
    simmc_time_perCluster_.push_back(time_perCluster);
    simmc_clusterType_.push_back(clusterType_perCluster);

    // Find primary tracking particle for this mergedcluster
    float primary_energy = -999.0;
    float primary_et = -999.0;
    float primary_phi = -999.0;
    float primary_eta = -999.0;
    int primary_pdgId = 0;

    // Take any tracking particle and climb back to original ancestor
    if (!simmc.trackingParticles().empty()) {
      auto tp_ref = simmc.trackingParticles()[0];  // Take first TP

      LogDebug("MtdMergedClusterValidation")
          << "\t\t Starting TP: pdgId = " << tp_ref->pdgId() << ", status = " << tp_ref->status()
          << ", E = " << tp_ref->energy() << " GeV"
          << ", #g4tracks = " << tp_ref->g4Tracks().size();

      // Climb back to original ancestor
      TrackingParticleRef current = tp_ref;

      while (true) {
        // Check if this particle has parent vertices
        const auto& parentVertices = current->parentVertex();
        if (parentVertices.isNull() || !parentVertices.isAvailable()) {
          // No parent vertex - this is the root ancestor
          LogDebug("MtdMergedClusterValidation") << "\t\t Found root ancestor (no parent vertex)";
          break;
        }

        // Get the parent tracks from the parent vertex
        const auto& parentTracks = parentVertices->sourceTracks();
        if (parentTracks.empty()) {
          // No parent tracks - this is the root ancestor
          LogDebug("MtdMergedClusterValidation") << "\t\t Found root ancestor (no parent tracks)";
          break;
        }

        // Move to the first parent
        current = parentTracks[0];
        LogDebug("MtdMergedClusterValidation")
            << "\t\t Moving to parent: pdgId = " << current->pdgId() << ", status = " << current->status();
      }

      // Now check if the ancestor passes primary particle criteria
      const auto& ancestor = *current;

      LogDebug("MtdMergedClusterValidation")
          << "\t\t Final ancestor: pdgId = " << ancestor.pdgId() << ", status = " << ancestor.status()
          << ", E = " << ancestor.energy() << " GeV"
          << ", #g4tracks = " << ancestor.g4Tracks().size();

      // Apply primary particle criteria to ancestor
      bool isPrimary = true;
      if (ancestor.status() != 1)
        isPrimary = false;
      if (!ancestor.g4Tracks().empty()) {
        if (ancestor.g4Tracks().front().vertIndex() != 0)
          isPrimary = false;
      } else {
        isPrimary = false;
      }

      if (isPrimary) {
        // This ancestor passes the criteria - extract its properties
        primary_energy = ancestor.energy();  // GeV
        primary_et = ancestor.pt();          // GeV
        primary_phi = ancestor.phi();
        primary_eta = ancestor.eta();
        primary_pdgId = ancestor.pdgId();

        LogDebug("MtdMergedClusterValidation")
            << "\t\t Found valid primary ancestor: E=" << primary_energy << " GeV, ET=" << primary_et
            << " GeV, phi=" << primary_phi << ", eta=" << primary_eta << ", pdgId=" << primary_pdgId;
      } else {
        LogDebug("MtdMergedClusterValidation") << "\t\t Ancestor does not pass primary criteria";
      }
    }

    h_simmc_primaryPt_vs_nClusters_->Fill(nClusters, primary_et);
    h_simmc_primaryPt_vs_energy_->Fill(energy, primary_et);
    h_simmc_primaryEnergy_vs_energy_->Fill(energy, primary_energy);
    h_simmc_primaryEnergy_vs_nClusters_->Fill(nClusters, primary_energy);

    simmc_primary_energy_.push_back(primary_energy);
    simmc_primary_et_.push_back(primary_et);
    simmc_primary_phi_.push_back(primary_phi);
    simmc_primary_eta_.push_back(primary_eta);
    simmc_primary_pdgId_.push_back(primary_pdgId);

    // Print detailed info for first few events
    if (evt_event_ <= 3) {
      LogDebug("MtdMergedClusterValidation") << "MergedCluster: MeV, t=" << time << " ns;"
                                             << " E = " << energy << " MeV;"
                                             << " pos = (" << simmc.simPos().x() << "," << simmc.simPos().y() << ")";
      LogDebug("MtdMergedClusterValidation") << "  Made from " << nClusters << " clusters:";
      for (const auto& detId : simmc.detIds()) {
        LogDebug("MtdMergedClusterValidation") << "    DetId: " << detId.rawId();
      }
    }
  }

  h_simmc_n_->Fill(simmc_n_);

  // tree_->getTTree()->Fill();

  if (evt_event_ <= 3) {
    LogDebug("MtdMergedClusterValidation")
        << "Event " << evt_event_ << ": " << mc_n_ << " MergedClusters, " << cluster_n_ << " individual clusters";
    LogDebug("MtdMergedClusterValidation") << "Adjacent pairs: " << adjacentPairs << ", Merged pairs: " << mergedPairs;
  }
}

void MtdMergedClusterValidation::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(folder_);

  // Book all histograms
  h_mc_energy_ = ibooker.book1D("h_mc_energy", "MergedCluster Energy;Energy [MeV];Count", 100, 0, 50);
  h_mc_time_ = ibooker.book1D("h_mc_time", "MergedCluster Time;Time [ns];Count", 100, -5, 20);
  h_mc_timeError_ = ibooker.book1D("h_mc_timeError", "MergedCluster Time Error;Time Error [ns];Count", 100, 0, 1);
  h_mc_x_ = ibooker.book1D("h_mc_x", "MergedCluster X;X [mm];Count", 100, -200, 200);
  h_mc_y_ = ibooker.book1D("h_mc_y", "MergedCluster Y;Y [mm];Count", 100, -200, 200);
  h_mc_eta_ = ibooker.book1D("h_mc_eta", "MergedCluster Eta;#eta;Count", 100, -1.5, 1.5);
  h_mc_nClusters_ = ibooker.book1D("h_mc_nClusters", "Number of Clusters;N_{clusters};Count", 10, 0.5, 10.5);

  // Cluster histograms for comparison
  h_cluster_energy_ = ibooker.book1D("h_cluster_energy", "Cluster Energy;Energy [MeV];Count", 100, 0, 50);
  h_cluster_time_ = ibooker.book1D("h_cluster_time", "Cluster Time;Time [ns];Count", 100, -5, 20);
  h_comp_energy_ = ibooker.book1D("h_comp_energy", "Cluster vs MergedCluster Energy;Energy [MeV];Count", 100, 0, 50);
  h_comp_time_ = ibooker.book1D("h_comp_time", "Cluster vs MergedCluster Time;Time [ns];Count", 100, -5, 20);

  // sanity checks
  h_cluster_energy_eff_ = ibooker.book1D("h_cluster_energy_eff", "Cluster Energy;Energy [MeV];Efficiency", 100, 0, 50);
  h_cluster_time_eff_ = ibooker.book1D("h_cluster_time_eff", "Cluster Time;Time [ns];Efficiency", 100, -5, 20);

  h_mc_cluster_distance_phi_ = ibooker.book1D("h_mc_cluster_distance_phi",
                                              "Distance in Phi Between Clusters in MergedCluster;#Delta'phi;Entries",
                                              50,
                                              -0.1,
                                              0.1);
  h_mc_cluster_distance_eta_ = ibooker.book1D("h_mc_cluster_distance_eta",
                                              "Distance in Eta Between Clusters in MergedCluster;#Delta'eta;Entries",
                                              50,
                                              -0.1,
                                              0.1);

  h_mc_cluster_distance_z_ = ibooker.book1D("h_mc_cluster_distance_z",
                                            "Distance in Z Between Clusters in MergedCluster;#Delta'z [mm];Entries",
                                            100,
                                            -50.,
                                            50.);
  h_mc_cluster_distance_2D_ =
      ibooker.book2D("h_mc_cluster_distance_2D",
                     "Distance in Eta vs Phi Between Clusters in MergedCluster;#Delta'phi;#Delta'z",
                     50,
                     -0.1,
                     0.1,
                     100,
                     -50.,
                     50.);
  h_mc_cluster_hitProdType_2D = ibooker.book2D("h_mc_cluster_hitProdType_2D",
                                               "HitProdType of clusters in MergedCluster;HitProdType;HitProdType",
                                               4,
                                               -0.5,
                                               3.5,
                                               4,
                                               -0.5,
                                               3.5);

  h_mc_cluster_distance_ieta_ = ibooker.book1D(
      "h_mc_cluster_distance_ieta", "Distance in iEta Between Clusters in MergedCluster;#Delta iEta;Entries", 10, -5, 5);
  h_mc_cluster_distance_iphi_ = ibooker.book1D(
      "h_mc_cluster_distance_iphi", "Distance in iPhi Between Clusters in MergedCluster;#Delta iPhi;Entries", 10, -5, 5);
  h_mc_cluster_distance_i2D_ =
      ibooker.book2D("h_mc_cluster_distance_i2D",
                     "Distance in iEta vs iPhi Between Clusters in MergedCluster;#Delta iPhi;#Delta iEta",
                     11,
                     -5,
                     5,
                     11,
                     -5,
                     5);

  // 2D histograms
  h_mc_energy_vs_time_ = ibooker.book2D(
      "h_mc_energy_vs_time", "MergedCluster Energy vs Time;Time [ns];Energy [MeV]", 100, -5, 20, 100, 0, 50);
  h_mc_xy_ = ibooker.book2D("h_mc_xy", "MergedCluster Position;X [mm];Y [mm]", 100, -200, 200, 100, -200, 200);
  h_mc_energy_vs_nClusters_ = ibooker.book2D(
      "h_mc_energy_vs_nClusters_", "Energy vs N Clusters;N_{clusters};Energy [MeV]", 10, 0.5, 10.5, 100, 0, 50);
  //h_merging_efficiency_ = ibooker.book2D("h_merging_efficiency", "Merging Efficiency;Cluster Energy [MeV];Merged?", 50, 0, 50, 2, -0.5, 1.5);

  // efficiency
  h_eta_adjacent_pairs_ =
      ibooker.book1D("h_eta_adjacent_pairs", "Adjacent Pairs vs Eta;#eta;Number of Adjacent Pairs", 100, -1.5, 1.5);
  h_eta_merged_pairs_ =
      ibooker.book1D("h_eta_merged_pairs", "Merged Pairs vs Eta;#eta;Number of Merged Pairs", 100, -1.5, 1.5);
  h_eta_merging_fraction_ =
      ibooker.book1D("h_eta_merging_fraction", "Merging Fraction vs Eta;#eta;Fraction", 100, -1.5, 1.5);

  h_eta_sameTrackID_pairs_ =
      ibooker.book1D("h_eta_sameTrackID_pairs", "Pairs with Same TrackID vs Eta;#eta;Number of Pairs", 100, -1.5, 1.5);
  h_eta_merged_sameTrackID_pairs_ = ibooker.book1D(
      "h_eta_merged_sameTrackID_pairs", "Merged Pairs with Same TrackID vs Eta;#eta;Number of Pairs", 100, -1.5, 1.5);
  h_merging_efficiency_ =
      ibooker.book1D("h_merging_efficiency", "Merging Efficiency vs Eta;#eta;Efficiency", 100, -1.5, 1.5);

  //*** eta phi resolution map ***
  for (int phi_idx = 0; phi_idx < 6; phi_idx++) {
    for (int eta_idx = 0; eta_idx < 6; eta_idx++) {
      float phi_min = -1.5 + phi_idx * 0.5;
      float phi_max = phi_min + 0.5;
      float eta_min = -1.5 + eta_idx * 0.5;
      float eta_max = eta_min + 0.5;

      std::string timeResName = "BtlTimeResPhi" + std::to_string(static_cast<int>(phi_min * 10)) + "to" +
                                std::to_string(static_cast<int>(phi_max * 10)) + "Eta" +
                                std::to_string(static_cast<int>(eta_min * 10)) + "to" +
                                std::to_string(static_cast<int>(eta_max * 10));
      std::string timeResTitle = "BTL time resolution;t_{RECO} - t_{SIM} [ns]";
      h_time_res_etaphi_[phi_idx][eta_idx] = ibooker.book1D(timeResName.c_str(), timeResTitle.c_str(), 100, -0.5, 0.5);
    }
  }

  // single cluster checks
  h_single_dx_ = ibooker.book1D("h_single_dx", "Merged - Cluster X (single); #Delta X [mm]; Entries", 50, -.5, .5);
  h_single_dy_ = ibooker.book1D("h_single_dy", "Merged - Cluster Y (single); #Delta Y [mm]; Entries", 50, -.5, .5);
  h_single_dt_ =
      ibooker.book1D("h_single_dt", "Merged - Cluster Time (single); #Delta Time [ns]; Entries", 50, -.5, .5);
  h_single_de_ = ibooker.book1D("h_single_de", "Merged - Cluster Energy (single); Energy Ratio; Entries", 50, -0.1, .1);
  h_single_dt_outlier_ =
      ibooker.book1D("h_single_dt_outlier", "Outlier Δt (single);#Delta t [ns];Entries", 50, -10, 10);
  h_single_de_outlier_ =
      ibooker.book1D("h_single_de_outlier", "Outlier ΔE (single);#Delta E [MeV];Entries", 50, -50, 50);
  // energy consistency: merged - sum(inputs)
  h_mc_energy_minus_sumInputs_ =
      ibooker.book1D("h_mc_energy_minus_sumInputs", "Merged E - sum(input E);ΔE [MeV];Entries", 100, -1.0, 1.0);

  // ------------------- //
  // ------- SIM ------- //
  // ------------------- //

  // 1D histograms
  h_simmc_energy_ = ibooker.book1D("h_simmc_energy", "MergedCluster Energy;Energy [MeV];Count", 50, 0, 50);
  h_simmc_logEnergy_ =
      ibooker.book1D("h_simmc_logEnergy", "MergedCluster Log(Energy);Log(Energy [MeV]);Count", 100, -3, 3);
  h_simmc_time_ = ibooker.book1D("h_simmc_time", "MergedCluster Time;Time [ns];Count", 50, 0, 30);
  h_simmc_x_ = ibooker.book1D("h_simmc_x", "MergedCluster X;X [mm];Count", 50, -120, 120);
  h_simmc_y_ = ibooker.book1D("h_simmc_y", "MergedCluster Y;Y [mm];Count", 50, -120, 120);
  h_simmc_nClusters_ = ibooker.book1D("h_simmc_nClusters", "Number of Clusters;N_{clusters};Count", 11, -0.5, 10.5);
  h_simmc_n_ = ibooker.book1D("h_simmc_n", "Number of MergedClusters;N_{MergedClusters};Count", 51, -0.5, 50.5);

  // 1D histograms -- per simLC
  h_simmc_logEnergy_perCluster_ =
      ibooker.book1D("h_simmc_logEnergy_perCluster", "Cluster Log(Energy);Log(Energy [MeV]);Count", 100, -3, 3);
  h_simmc_time_perCluster_ = ibooker.book1D("h_simmc_time_perCluster", "Cluster Time;Time [ns];Count", 50, 0, 30);
  h_simmc_clusterType_ = ibooker.book1D("h_simmc_clusterType", "Cluster Type;Type;Count", 4, -0.5, 3.5);

  // 2D histograms
  h_simmc_xy_ = ibooker.book2D("h_simmc_xy", "MergedCluster Position;X [mm];Y [mm]", 100, -120, 120, 100, -120, 120);
  h_simmc_energy_vs_time_ = ibooker.book2D(
      "h_simmc_energy_vs_time", "MergedCluster Energy vs Time;Time [ns];Energy [MeV]", 30, 4, 27, 20, 0, 1);
  h_simmc_energy_vs_nClusters_ = ibooker.book2D(
      "h_simmc_energy_vs_nClusters", "Energy vs N Clusters;N_{clusters};Energy [MeV]", 4, -0.5, 3.5, 20, 0, 1);
  h_simmc_primaryPt_vs_nClusters_ = ibooker.book2D("h_simmc_primaryPt_vs_nClusters",
                                                   "Primary Particle pT vs N Clusters;N_{clusters};p_{T} [GeV]",
                                                   4,
                                                   -0.5,
                                                   3.5,
                                                   20,
                                                   0,
                                                   20);
  h_simmc_primaryPt_vs_energy_ =
      ibooker.book2D("h_simmc_primaryPt_vs_energy",
                     "Primary Particle pT vs MergedCluster Energy;MergedCluster Energy [MeV];p_{T} [GeV]",
                     20,
                     0,
                     1,
                     20,
                     0,
                     11);
  h_simmc_primaryEnergy_vs_energy_ = ibooker.book2D(
      "h_simmc_primaryEnergy_vs_energy",
      "Primary Particle Energy vs MergedCluster Energy;MergedCluster Energy [MeV];Primary Particle Energy [GeV]",
      20,
      0,
      1,
      20,
      0,
      40);
  h_simmc_primaryEnergy_vs_nClusters_ =
      ibooker.book2D("h_simmc_primaryEnergy_vs_nClusters",
                     "Primary Particle Energy vs N Clusters;N_{clusters};Primary Particle Energy [GeV]",
                     4,
                     -0.5,
                     3.5,
                     20,
                     0,
                     40);
}

void MtdMergedClusterValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "MTD/MergedClusters");
  desc.add<edm::InputTag>("mergedClusters", edm::InputTag("mtdMergedClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("clusters", edm::InputTag("mtdClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("simMergedClusters", edm::InputTag("mtdSimMergedClusterProducer"));
  desc.add<edm::InputTag>("simLayerClusters", edm::InputTag("mix", "MergedMtdTruthLC"));
  desc.add<edm::InputTag>("sim2tpAssociationMapTag", edm::InputTag("mtdSimLayerClusterToTPAssociation", ""));
  desc.add<edm::InputTag>("r2sAssociationMapTag", edm::InputTag("mtdRecoClusterToSimLayerClusterAssociation", ""));

  descriptions.add("mtdMergedClusterValid", desc);
}

DEFINE_FWK_MODULE(MtdMergedClusterValidation);
