#define EDM_ML_DEBUG

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

#include "SimDataFormats/Associations/interface/MtdRecoMergedClusterToSimMergedClusterAssociationMap.h"
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
  edm::EDGetTokenT<MtdRecoMergedClusterToSimMergedClusterAssociationMap> r2sAssociationMapToken_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

  MonitorElement* h_mc_nClusters_;
  MonitorElement* h_mc_nRecHits_merged_;
  MonitorElement* h_mc_energy_merged_;
  MonitorElement* h_mc_eta_total_;
  MonitorElement* h_mc_eta_merged_;
  MonitorElement* h_mc_time_merged_;
  MonitorElement* h_mc_timeError_merged_;
  MonitorElement* h_mc_x_merged_;
  MonitorElement* h_mc_y_merged_;
  MonitorElement* h_mc_nMatched_merged_;

  MonitorElement* h_mc_nRecHits_unmerged_;
  MonitorElement* h_mc_energy_unmerged_;
  MonitorElement* h_mc_eta_unmerged_;
  MonitorElement* h_mc_time_unmerged_;
  MonitorElement* h_mc_timeError_unmerged_;
  MonitorElement* h_mc_x_unmerged_;
  MonitorElement* h_mc_y_unmerged_;
  MonitorElement* h_mc_nMatched_unmerged_;

  MonitorElement* h_mc_comp_energy_merged_;
  MonitorElement* h_mc_comp_energy_unmerged_;
  MonitorElement* h_mc_comp_time_merged_;
  MonitorElement* h_mc_comp_time_unmerged_;

  MonitorElement* h_eta_merging_fraction_;

  //comparison between reco and sim

  MonitorElement* h_mc_vs_sim_energyres_merged_;
  MonitorElement* h_mc_vs_sim_energyres_unmerged_;
  MonitorElement* h_mc_vs_sim_timeres_merged_;
  MonitorElement* h_mc_vs_sim_timeres_unmerged_;

  // sim

  MonitorElement* h_simmc_nSimClusters_;
  MonitorElement* h_simmc_nSimHits_merged_;
  MonitorElement* h_simmc_energy_merged_;
  MonitorElement* h_simmc_eta_merged_;
  MonitorElement* h_simmc_eta_total_;
  MonitorElement* h_simmc_eta_unmerged_;
  MonitorElement* h_simmc_time_merged_;
  MonitorElement* h_simmc_x_merged_;
  MonitorElement* h_simmc_y_merged_;
  MonitorElement* h_simmc_nSimHits_unmerged_;
  MonitorElement* h_simmc_energy_unmerged_;
  MonitorElement* h_simmc_time_unmerged_;
  MonitorElement* h_simmc_x_unmerged_;
  MonitorElement* h_simmc_y_unmerged_;
  MonitorElement* h_simmc_hitProdType_merged_;
  MonitorElement* h_simmc_hitProdType_unmerged_;
  MonitorElement* h_simmc_merging_fraction_eta_;

  MonitorElement* h_simmc_cluster_hitProdType_2D;
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
  r2sAssociationMapToken_ = consumes<MtdRecoMergedClusterToSimMergedClusterAssociationMap>(
      iConfig.getParameter<edm::InputTag>("r2sAssociationMapTag"));
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

  const auto& r2sAssociationMap = iEvent.get(r2sAssociationMapToken_);

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

  //loop on reco merged clusters
  for (const auto& detSet : *mergedClustersHandle) {
    for (const auto& mc : detSet) {
      h_mc_nClusters_->Fill(mc.nClusters());
      h_mc_nRecHits_merged_->Fill(mc.nHits());
      const MTDGeomDet* mcdet = geom->idToDet(mc.id());

      if (mc.nClusters() > 1) {
        h_mc_nRecHits_merged_->Fill(mc.nHits());
        if (mcdet) {
          GlobalPoint mc_global_point = mcdet->surface().toGlobal(LocalPoint(mc.x(), mc.y(), 0));
          h_mc_eta_merged_->Fill(mc_global_point.eta());
          h_eta_merging_fraction_->Fill(mc_global_point.eta(), 1.0);
        }
        h_mc_energy_merged_->Fill(mc.energy());
        h_mc_time_merged_->Fill(mc.time());
        h_mc_timeError_merged_->Fill(mc.timeError());
        h_mc_x_merged_->Fill(mc.x());
        h_mc_y_merged_->Fill(mc.y());
        for (size_t i_hit = 0; i_hit < mc.nHits(); i_hit++) {
          h_mc_comp_time_merged_->Fill(mc.hTime(i_hit) - mc.time());
          h_mc_comp_energy_merged_->Fill(mc.hEnergy(i_hit) - mc.energy());
        }
      } else {
        h_mc_nRecHits_unmerged_->Fill(mc.nHits());
        if (mcdet) {
          GlobalPoint mc_global_point = mcdet->surface().toGlobal(LocalPoint(mc.x(), mc.y(), 0));
          h_mc_eta_unmerged_->Fill(mc_global_point.eta());
          h_eta_merging_fraction_->Fill(mc_global_point.eta(), 0.0);
        }
        h_mc_energy_unmerged_->Fill(mc.energy());
        h_mc_time_unmerged_->Fill(mc.time());
        h_mc_timeError_unmerged_->Fill(mc.timeError());
        h_mc_x_unmerged_->Fill(mc.x());
        h_mc_y_unmerged_->Fill(mc.y());
        for (size_t i_hit = 0; i_hit < mc.nHits(); i_hit++) {
          h_mc_comp_time_unmerged_->Fill(mc.hTime(i_hit) - mc.time());
          h_mc_comp_energy_unmerged_->Fill(mc.hEnergy(i_hit) - mc.energy());
        }
      }

      //get matched sim merged clusters
      edm::Ref<edmNew::DetSetVector<FTLMergedCluster>, FTLMergedCluster> mcRef =
          edmNew::makeRefTo(mergedClustersHandle, &mc);
      auto range = r2sAssociationMap.equal_range(mcRef);
      if (range.first != range.second) {
        std::vector<MtdSimMergedClusterRef> simClustersRefs =
            (*range.first).second;  // the range of itp.first, itp.second should be always 1

        for (size_t it_matched_sim_mc = 0; it_matched_sim_mc < simClustersRefs.size(); it_matched_sim_mc++) {
          auto sim_mc_ref = simClustersRefs[it_matched_sim_mc];
          if (sim_mc_ref.isNonnull()) {
            const MtdSimMergedCluster& sim_mc = *sim_mc_ref;
            if (mc.nClusters() > 1) {
              h_mc_vs_sim_energyres_merged_->Fill(
                  (mc.energy() - geant_units::operators::convertUnitsTo(0.001_MeV, sim_mc.simEnergy())));
              h_mc_vs_sim_timeres_merged_->Fill((mc.time() - sim_mc.simTime()));
              if (it_matched_sim_mc == 0) {
                h_mc_nMatched_merged_->Fill(simClustersRefs.size());
              }

            } else {
              h_mc_vs_sim_energyres_unmerged_->Fill(
                  (mc.energy() - geant_units::operators::convertUnitsTo(0.001_MeV, sim_mc.simEnergy())));
              h_mc_vs_sim_timeres_unmerged_->Fill((mc.time() - sim_mc.simTime()));
              if (it_matched_sim_mc == 0) {
                h_mc_nMatched_unmerged_->Fill(simClustersRefs.size());
              }
            }
          }
        }
      }
    }

  }  //end loop on reco MC
  auto topologyMode = MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode());
  // loop on sim MC
  for (const auto& simmc : *simMergedClustersHandle) {
    if (MTDDetId(simmc.simDetId()).mtdSubDetector() == 2)
      continue;                                                  // Skip clusters from ETL
    auto energy = convertUnitsTo(0.001_MeV, simmc.simEnergy());  // convert energy from GeV to MeV
    auto time = simmc.simTime();
    auto nClusters = simmc.clusters().size();
    h_simmc_nSimClusters_->Fill(nClusters);
    BTLDetId detId = simmc.simDetId();
    DetId geoId = detId.geographicalId(topologyMode);

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

    if (nClusters > 1) {
      h_simmc_energy_merged_->Fill(energy);
      h_simmc_time_merged_->Fill(time);
      h_simmc_nSimHits_merged_->Fill(simmc.hitTimesAndPositions().size());
      h_simmc_eta_merged_->Fill(global_point.eta());
      h_simmc_merging_fraction_eta_->Fill(global_point.eta(), 1.0);
      h_simmc_x_merged_->Fill(global_point.x());
      h_simmc_y_merged_->Fill(global_point.y());
      h_simmc_hitProdType_merged_->Fill(simmc.hitProdType());
      if (nClusters == 2) {
        h_simmc_cluster_hitProdType_2D->Fill((*simmc.clusters().at(0)).hitProdType(),
                                             (*simmc.clusters().at(1)).hitProdType());
      }
    } else {
      h_simmc_energy_unmerged_->Fill(energy);
      h_simmc_time_unmerged_->Fill(time);
      h_simmc_nSimHits_merged_->Fill(simmc.hitTimesAndPositions().size());
      h_simmc_eta_unmerged_->Fill(global_point.eta());
      h_simmc_merging_fraction_eta_->Fill(global_point.eta(), 1.0);

      h_simmc_x_unmerged_->Fill(global_point.x());
      h_simmc_y_unmerged_->Fill(global_point.y());
      h_simmc_hitProdType_unmerged_->Fill(simmc.hitProdType());
    }
  }  // end of loop on sim MC
}

void MtdMergedClusterValidation::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(folder_);

  // Book all histograms
  h_mc_nClusters_ =
      ibooker.book1D("h_mc_nClusters", "Number of FTLClusters in reco MergedCluster;N_{clusters};Count", 4, 0.5, 4.5);

  h_mc_nRecHits_merged_ = ibooker.book1D("h_mc_nRecHits_merged",
                                         "Number of FTLRecHits in reco MergedCluster, >1 FTLCluster;N_{RecHits};Count",
                                         32,
                                         0.5,
                                         32.5);
  h_mc_nRecHits_unmerged_ =
      ibooker.book1D("h_mc_nRecHits_unmerged",
                     "Number of FTLRecHits in reco MergedCluster, =1 FTLCluster;N_{RecHits};Count",
                     32,
                     0.5,
                     32.5);

  h_mc_energy_merged_ =
      ibooker.book1D("h_mc_energy_merged", "MergedCluster Energy, >1 FTLCluster;Energy [MeV];Count", 100, 0, 50);
  h_mc_energy_unmerged_ =
      ibooker.book1D("h_mc_energy_unmerged", "MergedCluster Energy, =1 FTLCluster;Energy [MeV];Count", 100, 0, 50);

  h_mc_time_merged_ =
      ibooker.book1D("h_mc_time_merged", "MergedCluster Time, >1 FTLCluster;Time [ns];Count", 100, -5, 20);
  h_mc_time_unmerged_ =
      ibooker.book1D("h_mc_time_unmerged", "MergedCluster Time, =1 FTLCluster;Time [ns];Count", 100, -5, 20);

  h_mc_timeError_merged_ = ibooker.book1D(
      "h_mc_timeError_merged", "MergedCluster Time Error, >1 FTLCluster;Time Error [ns];Count", 100, 0, 1);
  h_mc_timeError_unmerged_ = ibooker.book1D(
      "h_mc_timeError_unmerged", "MergedCluster Time Error, =1 FTLCluster;Time Error [ns];Count", 100, 0, 1);

  h_mc_x_merged_ = ibooker.book1D("h_mc_x_merged", "MergedCluster X, >1 FTLCluster;X [mm];Count", 20, -4, 4);
  h_mc_x_unmerged_ = ibooker.book1D("h_mc_x_unmerged", "MergedCluster X, =1 FTLCluster;X [mm];Count", 20, -4, 4);

  h_mc_y_merged_ = ibooker.book1D("h_mc_y_merged", "MergedCluster Y, >1 FTLCluster;Y [mm];Count", 20, -4, 4);
  h_mc_y_unmerged_ = ibooker.book1D("h_mc_y_unmerged", "MergedCluster Y, =1 FTLCluster;Y [mm];Count", 20, -4, 4);

  h_mc_eta_total_ = ibooker.book1D("h_mc_eta_total", "MergedClusterEta, total;#eta;Count", 50, -1.5, 1.5);
  h_mc_eta_merged_ = ibooker.book1D("h_mc_eta_merged", "MergedClusterEta, >1 FTLCluster;#eta;Count", 50, -1.5, 1.5);
  h_mc_eta_unmerged_ = ibooker.book1D("h_mc_eta_unmerged", "MergedClusterEta, =1 FTLCluster;#eta;Count", 50, -1.5, 1.5);
  h_eta_merging_fraction_ = ibooker.bookProfile(
      "h_eta_merging_fraction", "Merging Fraction vs Eta;#eta;Fraction", 50, -1.5, 1.5, 0.0, 1.0, "");

  h_mc_time_unmerged_ =
      ibooker.book1D("h_mc_time_unmerged", "MergedCluster Time, =1 FTLCluster;Time [ns];Count", 100, -5, 20);
  h_mc_timeError_unmerged_ = ibooker.book1D(
      "h_mc_timeError_unmerged", "MergedCluster Time Error, =1 FTLCluster;Time Error [ns];Count", 100, 0, 1);

  h_mc_comp_energy_merged_ =
      ibooker.book1D("h_mc_comp_energy_merged",
                     "#Delta E(recHit - MC),  >1 FTLCluster;E(recHit)- E(MergedCluster) [MeV];Count",
                     50,
                     -50,
                     0);
  h_mc_comp_energy_unmerged_ =
      ibooker.book1D("h_mc_comp_energy_unmerged",
                     "#Delta E(recHit - MC),  =1 FTLCluster;E(recHit)- E(MergedCluster) [MeV];Count",
                     50,
                     -50,
                     0);

  h_mc_comp_time_merged_ =
      ibooker.book1D("h_mc_comp_time_merged",
                     "#Delta t(recHit - MC),  >1 FTLCluster;t(recHit)- t(MergedCluster) [ns];Count",
                     50,
                     -1,
                     1);
  h_mc_comp_time_unmerged_ =
      ibooker.book1D("h_mc_comp_time_unmerged",
                     "#Delta t(recHit - MC),  =1 FTLCluster;t(recHit)- t(MergedCluster) [ns];Count",
                     50,
                     -1,
                     1);

  h_mc_nMatched_merged_ =
      ibooker.book1D("h_mc_nMatched_merged",
                     "N. of matched SimMergedClusters per RecoMergedCluster,  >1 FTLCluster;nMatched;Count",
                     6,
                     -0.5,
                     5.5);
  h_mc_nMatched_unmerged_ =
      ibooker.book1D("h_mc_nMatched_unmerged",
                     "N. of matched SimMergedClusters per RecoMergedCluster,  =1 FTLCluster;nMatched;Count",
                     6,
                     -0.5,
                     5.5);

  h_mc_vs_sim_energyres_merged_ = ibooker.book1D(
      "h_mc_vs_sim_energyres_merged", "#Delta E(reco - sim),  >1 FTLCluster;E(reco)- E(sim) [MeV];Count", 50, -50, 50);
  h_mc_vs_sim_energyres_unmerged_ = ibooker.book1D(
      "h_mc_vs_sim_energyres_unmerged", "#Delta E(reco - sim),  =1 FTLCluster;E(reco)- E(sim) [MeV];Count", 50, -50, 50);
  h_mc_vs_sim_timeres_merged_ = ibooker.book1D(
      "h_mc_vs_sim_timeres_merged", "#Delta t(reco - sim),  >1 FTLCluster;t(reco)- t(sim) [ns];Count", 50, -1, 1);
  h_mc_vs_sim_timeres_unmerged_ = ibooker.book1D(
      "h_mc_vs_sim_timeres_unmerged", "#Delta t(reco - sim),  =1 FTLCluster;t(reco)- t(sim) [ns];Count", 50, -1, 1);

  // sim

  //2D

  h_simmc_nSimClusters_ =
      ibooker.book1D("h_simmc_nSimClusters_",
                     "N. of MtdSimLayerClusters in MergedCluster;N. of MtdSimLayerClusters in MergedCluster;Count",
                     6,
                     -0.5,
                     5.5);
  h_simmc_nSimHits_merged_ = ibooker.book1D(
      "h_simmc_nSimHits_merged", "N. of SimHits in MergedCluster;N. of SimHits in MergedCluster;Count", 20, -0.5, 19.5);
  h_simmc_nSimHits_unmerged_ = ibooker.book1D("h_simmc_nSimHits_unmerged",
                                              "N. of SimHits in MergedCluster;N. of SimHits in MergedCluster;Count",
                                              20,
                                              -0.5,
                                              19.5);
  h_simmc_energy_merged_ =
      ibooker.book1D("h_simmc_energy_merged", "Energy of MergedCluster;Energy [MeV];Count", 100, 0, 100);
  h_simmc_energy_unmerged_ =
      ibooker.book1D("h_simmc_energy_unmerged", "Energy of MergedCluster;Energy [MeV];Count", 100, 0, 100);
  h_simmc_time_merged_ = ibooker.book1D("h_simmc_time_merged", "Time of MergedCluster;Time [ns];Count", 100, -5, 20);
  h_simmc_time_unmerged_ =
      ibooker.book1D("h_simmc_time_unmerged", "Time of MergedCluster;Time [ns];Count", 100, -5, 20);
  h_simmc_x_merged_ = ibooker.book1D("h_simmc_x_merged", "MergedCluster X, >1 FTLCluster;X [mm];Count", 20, -4, 4);
  h_simmc_x_unmerged_ = ibooker.book1D("h_simmc_x_unmerged", "MergedCluster X, =1 FTLCluster;X [mm];Count", 20, -4, 4);
  h_simmc_y_merged_ = ibooker.book1D("h_simmc_y_merged", "MergedCluster Y, >1 FTLCluster;Y [mm];Count", 20, -4, 4);
  h_simmc_y_unmerged_ = ibooker.book1D("h_simmc_y_unmerged", "MergedCluster Y, =1 FTLCluster;Y [mm];Count", 20, -4, 4);
  h_simmc_eta_merged_ =
      ibooker.book1D("h_simmc_eta_merged", "MergedClusterEta, >1 FTLCluster;#eta;Count", 50, -1.5, 1.5);
  h_simmc_eta_unmerged_ =
      ibooker.book1D("h_simmc_eta_unmerged", "MergedClusterEta, =1 FTLCluster;#eta;Count", 50, -1.5, 1.5);
  h_simmc_eta_total_ =
      ibooker.book1D("h_simmc_eta_total", "MergedClusterEta, all FTLClusters;#eta;Count", 50, -1.5, 1.5);
  h_simmc_merging_fraction_eta_ = ibooker.bookProfile(
      "h_simmc_merging_fraction_eta", "Merging Fraction of MergedCluster;#eta;Fraction", 50, -1.5, 1.5, 0.0, 1.0, "");
  h_simmc_hitProdType_merged_ =
      ibooker.book1D("h_simmc_clusterType_merged", "hitProdType of MergedCluster;hitProdType;Count", 4, -0.5, 3.5);
  h_simmc_hitProdType_unmerged_ =
      ibooker.book1D("h_simmc_clusterType_unmerged", "hitProdType of MergedCluster;hitProdType;Count", 4, -0.5, 3.5);

  h_simmc_cluster_hitProdType_2D = ibooker.book2D("h_simmc_cluster_hitProdType_2D",
                                                  "HitProdType of clusters in MergedCluster;HitProdType;HitProdType",
                                                  4,
                                                  -0.5,
                                                  3.5,
                                                  4,
                                                  -0.5,
                                                  3.5);
}

void MtdMergedClusterValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "MTD/MergedClusters");
  desc.add<edm::InputTag>("mergedClusters", edm::InputTag("mtdMergedClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("clusters", edm::InputTag("mtdClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("simMergedClusters", edm::InputTag("mtdSimMergedClusterProducer"));
  desc.add<edm::InputTag>("simLayerClusters", edm::InputTag("mix", "MergedMtdTruthLC"));
  desc.add<edm::InputTag>("sim2tpAssociationMapTag", edm::InputTag("mtdSimLayerClusterToTPAssociation", ""));
  desc.add<edm::InputTag>("r2sAssociationMapTag",
                          edm::InputTag("mtdRecoMergedClusterToSimMergedClusterAssociation", ""));

  descriptions.add("mtdMergedClusterValid", desc);
}

DEFINE_FWK_MODULE(MtdMergedClusterValidation);
