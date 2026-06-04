#ifndef Validation_MtdValidation_MergedClusterValidation_h
#define Validation_MtdValidation_MergedClusterValidation_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "DataFormats/FTLRecHit/interface/FTLMergedClusterCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociatorBaseImpl.h"

#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimMergedClusterFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class MergedClusterValidation : public DQMEDAnalyzer {
public:
  explicit MergedClusterValidation(const edm::ParameterSet&);
  ~MergedClusterValidation() override = default;

private:
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

#endif
