// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      EtlLocalRecoValidation
//
/**\class EtlLocalRecoValidation EtlLocalRecoValidation.cc Validation/MtdValidation/plugins/EtlLocalRecoValidation.cc

 Description: ETL RECO hits and clusters validation

 Implementation:
     [Notes on implementation]
*/

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTypes.h"

#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociationMap.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomUtil.h"

#include "RecoLocalFastTime/Records/interface/MTDCPERecord.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDClusterParameterEstimator.h"

#include "MTDHit.h"

class EtlLocalRecoValidation : public DQMEDAnalyzer {
public:
  explicit EtlLocalRecoValidation(const edm::ParameterSet&);
  ~EtlLocalRecoValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  bool isSameCluster(const FTLCluster&, const FTLCluster&);

  // ------------ member data ------------

  const std::string folder_;
  const float hitMinEnergy2Dis_;
  const bool optionalPlots_;
  const bool uncalibRecHitsPlots_;
  const double hitMinAmplitude_;

  edm::EDGetTokenT<FTLRecHitCollection> etlRecHitsToken_;
  edm::EDGetTokenT<FTLUncalibratedRecHitCollection> etlUncalibRecHitsToken_;
  edm::EDGetTokenT<CrossingFrame<PSimHit>> etlSimHitsToken_;
  edm::EDGetTokenT<FTLClusterCollection> etlRecCluToken_;
  edm::EDGetTokenT<MTDTrackingDetSetVector> mtdTrackingHitToken_;
  edm::EDGetTokenT<MtdRecoClusterToSimLayerClusterAssociationMap> r2sAssociationMapToken_;

  const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  const edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
  const edm::ESGetToken<MTDClusterParameterEstimator, MTDCPERecord> cpeToken_;

  // --- histograms declaration

  MonitorElement* meNhits_[4];
  MonitorElement* meHitTime_[4];
  MonitorElement* meHitTimeError_[4];

  MonitorElement* meOccupancy_[4];

  MonitorElement* meLocalOccupancy_[2];
  MonitorElement* meHitXlocal_[2];
  MonitorElement* meHitYlocal_[2];

  MonitorElement* meHitX_[4];
  MonitorElement* meHitY_[4];
  MonitorElement* meHitZ_[4];
  MonitorElement* meHitPhi_[4];
  MonitorElement* meHitEta_[4];

  MonitorElement* meHitTvsPhi_[4];
  MonitorElement* meHitTvsEta_[4];

  MonitorElement* meCluTime_[4];
  MonitorElement* meCluTimeError_[4];
  MonitorElement* meCluPhi_[4];
  MonitorElement* meCluEta_[4];
  MonitorElement* meCluHits_[4];
  MonitorElement* meCluOccupancy_[4];

  MonitorElement* meTimeRes_;
  MonitorElement* meEnergyRes_;
  MonitorElement* meTPullvsE_;
  MonitorElement* meTPullvsEta_;

  MonitorElement* meCluTimeRes_[2];
  MonitorElement* meCluTPullvsE_[2];
  MonitorElement* meCluTPullvsEta_[2];
  MonitorElement* meCluXRes_[2];
  MonitorElement* meCluYRes_[2];
  MonitorElement* meCluZRes_[2];
  MonitorElement* meCluXPull_[2];
  MonitorElement* meCluYPull_[2];
  MonitorElement* meCluYXLocal_[2];
  MonitorElement* meCluYXLocalSim_[2];
  MonitorElement* meCluXLocalErr_[2];
  MonitorElement* meCluYLocalErr_[2];

  // resolution using MtdSimLayerClusters as truth
  MonitorElement* meCluTimeRes_simLC_[2];
  MonitorElement* meCluTPullvsE_simLC_[2];
  MonitorElement* meCluTPullvsEta_simLC_[2];
  MonitorElement* meCluXRes_simLC_[2];
  MonitorElement* meCluYRes_simLC_[2];
  MonitorElement* meCluZRes_simLC_[2];
  MonitorElement* meCluXPull_simLC_[2];
  MonitorElement* meCluYPull_simLC_[2];
  MonitorElement* meCluYXLocalSim_simLC_[2];

  // --- UncalibratedRecHits histograms

  static constexpr int nBinsTot_ = 20;
  static constexpr float binWidthTot_ = 1.3;  // in MIP units

  MonitorElement* meTimeResTot_[2][nBinsTot_];

  static constexpr int nBinsEta_ = 26;
  static constexpr float binWidthEta_ = 0.05;
  static constexpr float etaMin_ = 1.65;

  MonitorElement* meTimeResEta_[2][nBinsEta_];
};

bool EtlLocalRecoValidation::isSameCluster(const FTLCluster& clu1, const FTLCluster& clu2) {
  return clu1.id() == clu2.id() && clu1.size() == clu2.size() && clu1.x() == clu2.x() && clu1.y() == clu2.y() &&
         clu1.time() == clu2.time();
}

// ------------ constructor and destructor --------------
EtlLocalRecoValidation::EtlLocalRecoValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy2Dis_(iConfig.getParameter<double>("hitMinimumEnergy2Dis")),
      optionalPlots_(iConfig.getParameter<bool>("optionalPlots")),
      uncalibRecHitsPlots_(iConfig.getParameter<bool>("UncalibRecHitsPlots")),
      hitMinAmplitude_(iConfig.getParameter<double>("HitMinimumAmplitude")),
      mtdgeoToken_(esConsumes<MTDGeometry, MTDDigiGeometryRecord>()),
      mtdtopoToken_(esConsumes<MTDTopology, MTDTopologyRcd>()),
      cpeToken_(esConsumes<MTDClusterParameterEstimator, MTDCPERecord>(edm::ESInputTag("", "MTDCPEBase"))) {
  etlRecHitsToken_ = consumes<FTLRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsTag"));
  if (uncalibRecHitsPlots_)
    etlUncalibRecHitsToken_ =
        consumes<FTLUncalibratedRecHitCollection>(iConfig.getParameter<edm::InputTag>("uncalibRecHitsTag"));
  etlSimHitsToken_ = consumes<CrossingFrame<PSimHit>>(iConfig.getParameter<edm::InputTag>("simHitsTag"));
  etlRecCluToken_ = consumes<FTLClusterCollection>(iConfig.getParameter<edm::InputTag>("recCluTag"));
  mtdTrackingHitToken_ = consumes<MTDTrackingDetSetVector>(iConfig.getParameter<edm::InputTag>("trkHitTag"));
  r2sAssociationMapToken_ = consumes<MtdRecoClusterToSimLayerClusterAssociationMap>(
      iConfig.getParameter<edm::InputTag>("r2sAssociationMapTag"));
}

EtlLocalRecoValidation::~EtlLocalRecoValidation() {}

// ------------ method called for each event  ------------
void EtlLocalRecoValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace geant_units::operators;
  using namespace mtd;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();

  MTDGeomUtil geomUtil;
  geomUtil.setGeometry(geom);

  auto const& cpe = iSetup.getData(cpeToken_);

  auto etlRecHitsHandle = makeValid(iEvent.getHandle(etlRecHitsToken_));
  auto etlSimHitsHandle = makeValid(iEvent.getHandle(etlSimHitsToken_));
  auto etlRecCluHandle = makeValid(iEvent.getHandle(etlRecCluToken_));
  auto mtdTrkHitHandle = makeValid(iEvent.getHandle(mtdTrackingHitToken_));
  const auto& r2sAssociationMap = iEvent.get(r2sAssociationMapToken_);
  MixCollection<PSimHit> etlSimHits(etlSimHitsHandle.product());

#ifdef EDM_ML_DEBUG
  for (const auto& hits : *mtdTrkHitHandle) {
    if (MTDDetId(hits.id()).mtdSubDetector() == MTDDetId::MTDType::ETL) {
      LogDebug("EtlLocalRecoValidation") << "MTD cluster DetId " << hits.id() << " # cluster " << hits.size();
      for (const auto& hit : hits) {
        LogDebug("EtlLocalRecoValidation")
            << "MTD_TRH: " << hit.localPosition().x() << "," << hit.localPosition().y() << " : "
            << hit.localPositionError().xx() << "," << hit.localPositionError().yy() << " : " << hit.time() << " : "
            << hit.timeError();
      }
    }
  }
#endif

  // --- Loop over the ETL SIM hits
  std::unordered_map<mtd_digitizer::MTDCellId, MTDHit> m_etlSimHits[4];
  for (auto const& simHit : etlSimHits) {
    // --- Use only hits compatible with the in-time bunch-crossing
    if (simHit.tof() < 0 || simHit.tof() > 25.)
      continue;

    ETLDetId id = simHit.detUnitId();

    int idet = -1;

    if ((id.zside() == -1) && (id.nDisc() == 1)) {
      idet = 0;
    } else if ((id.zside() == -1) && (id.nDisc() == 2)) {
      idet = 1;
    } else if ((id.zside() == 1) && (id.nDisc() == 1)) {
      idet = 2;
    } else if ((id.zside() == 1) && (id.nDisc() == 2)) {
      idet = 3;
    } else {
      edm::LogWarning("EtlLocalRecoValidation") << "Unknown ETL DetId configuration: " << id;
      continue;
    }

    const auto& position = simHit.localPosition();

    LocalPoint simscaled(convertMmToCm(position.x()), convertMmToCm(position.y()), convertMmToCm(position.z()));
    std::pair<uint8_t, uint8_t> pixel = geomUtil.pixelInModule(id, simscaled);

    mtd_digitizer::MTDCellId pixelId(id.rawId(), pixel.first, pixel.second);
    auto simHitIt = m_etlSimHits[idet].emplace(pixelId, MTDHit()).first;

    // --- Accumulate the energy (in MeV) of SIM hits in the same detector cell
    (simHitIt->second).energy += convertUnitsTo(0.001_MeV, simHit.energyLoss());

    // --- Get the time of the first SIM hit in the cell
    if ((simHitIt->second).time == 0 || simHit.tof() < (simHitIt->second).time) {
      (simHitIt->second).time = simHit.tof();

      auto hit_pos = simHit.localPosition();
      (simHitIt->second).x = hit_pos.x();
      (simHitIt->second).y = hit_pos.y();
      (simHitIt->second).z = hit_pos.z();
    }

  }  // simHit loop

  // --- Loop over the ELT RECO hits
  unsigned int n_reco_etl[4] = {0, 0, 0, 0};
  for (const auto& recHit : *etlRecHitsHandle) {
    double weight = 1.0;
    ETLDetId detId = recHit.id();
    DetId geoId = detId.geographicalId();
    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("EtlLocalRecoValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                     << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    Local3DPoint local_point(topo.localX(recHit.row()), topo.localY(recHit.column()), 0.);
    const auto& global_point = thedet->toGlobal(local_point);

    int idet = 999;

    if (detId.discSide() == 1) {
      weight = -weight;
    }
    if ((detId.zside() == -1) && (detId.nDisc() == 1)) {
      idet = 0;
    } else if ((detId.zside() == -1) && (detId.nDisc() == 2)) {
      idet = 1;
    } else if ((detId.zside() == 1) && (detId.nDisc() == 1)) {
      idet = 2;
    } else if ((detId.zside() == 1) && (detId.nDisc() == 2)) {
      idet = 3;
    } else {
      edm::LogWarning("EtlLocalRecoValidation") << "Unknown ETL DetId configuration: " << detId;
      continue;
    }

    // --- Fill the histograms

    meHitTime_[idet]->Fill(recHit.time());
    meHitTimeError_[idet]->Fill(recHit.timeError());

    if ((idet == 0) || (idet == 1)) {
      meHitXlocal_[0]->Fill(local_point.x());
      meHitYlocal_[0]->Fill(local_point.y());
    }
    if ((idet == 2) || (idet == 3)) {
      meHitXlocal_[1]->Fill(local_point.x());
      meHitYlocal_[1]->Fill(local_point.y());
    }

    if (optionalPlots_) {
      meOccupancy_[idet]->Fill(global_point.x(), global_point.y(), weight);
      if ((idet == 0) || (idet == 1)) {
        meLocalOccupancy_[0]->Fill(local_point.x(), local_point.y());
      }
      if ((idet == 2) || (idet == 3)) {
        meLocalOccupancy_[1]->Fill(local_point.x(), local_point.y());
      }
    }
    meHitX_[idet]->Fill(global_point.x());
    meHitY_[idet]->Fill(global_point.y());
    meHitZ_[idet]->Fill(global_point.z());
    meHitPhi_[idet]->Fill(global_point.phi());
    meHitEta_[idet]->Fill(global_point.eta());
    meHitTvsPhi_[idet]->Fill(global_point.phi(), recHit.time());
    meHitTvsEta_[idet]->Fill(global_point.eta(), recHit.time());

    // Resolution histograms
    std::pair<uint8_t, uint8_t> pixel = geomUtil.pixelInModule(detId, local_point);
    mtd_digitizer::MTDCellId pixelId(detId.rawId(), pixel.first, pixel.second);

    if (m_etlSimHits[idet].count(pixelId) == 1) {
      if (m_etlSimHits[idet][pixelId].energy > hitMinEnergy2Dis_) {
        float time_res = recHit.time() - m_etlSimHits[idet][pixelId].time;

        meTimeRes_->Fill(time_res);

        meTPullvsEta_->Fill(std::abs(global_point.eta()), time_res / recHit.timeError());
        meTPullvsE_->Fill(m_etlSimHits[idet][pixelId].energy, time_res / recHit.timeError());
      }
    }

    n_reco_etl[idet]++;
  }  // recHit loop

  for (int i = 0; i < 4; i++) {
    meNhits_[i]->Fill(std::log10(n_reco_etl[i]));
  }

  size_t index(0);

  // --- Loop over the ETL RECO clusters ---
  for (const auto& DetSetClu : *etlRecCluHandle) {
    for (const auto& cluster : DetSetClu) {
      double weight = 1.0;
      ETLDetId cluId = cluster.id();
      DetId detIdObject(cluId);
      const auto& genericDet = geom->idToDetUnit(detIdObject);
      if (genericDet == nullptr) {
        throw cms::Exception("EtlLocalRecoValidation")
            << "GeographicalID: " << std::hex << cluId << " is invalid!" << std::dec << std::endl;
      }

      MTDClusterParameterEstimator::ReturnType tuple = cpe.getParameters(cluster, *genericDet);

      // --- Cluster position in the module reference frame
      LocalPoint local_point(std::get<0>(tuple));
      const auto& global_point = genericDet->toGlobal(local_point);

      int idet = 999;

      if (cluId.discSide() == 1) {
        weight = -weight;
      }
      if ((cluId.zside() == -1) && (cluId.nDisc() == 1)) {
        idet = 0;
      } else if ((cluId.zside() == -1) && (cluId.nDisc() == 2)) {
        idet = 1;
      } else if ((cluId.zside() == 1) && (cluId.nDisc() == 1)) {
        idet = 2;
      } else if ((cluId.zside() == 1) && (cluId.nDisc() == 2)) {
        idet = 3;
      } else {
        edm::LogWarning("EtlLocalRecoValidation") << "Unknown ETL DetId configuration: " << cluId;
        continue;
      }

      index++;
      LogDebug("EtlLocalRecoValidation") << "Cluster # " << index << " DetId " << cluId.rawId() << " idet " << idet;

      meCluTime_[idet]->Fill(cluster.time());
      meCluTimeError_[idet]->Fill(cluster.timeError());
      meCluPhi_[idet]->Fill(global_point.phi());
      meCluEta_[idet]->Fill(global_point.eta());
      meCluHits_[idet]->Fill(cluster.size());
      if (optionalPlots_) {
        meCluOccupancy_[idet]->Fill(global_point.x(), global_point.y(), weight);
      }

      // --- Get the SIM hits associated to the cluster and calculate
      //     the cluster SIM energy, time and position

      double cluEneSIM = 0.;
      double cluTimeSIM = 0.;
      double cluLocXSIM = 0.;
      double cluLocYSIM = 0.;
      double cluLocZSIM = 0.;

      if (optionalPlots_) {
        for (int ihit = 0; ihit < cluster.size(); ++ihit) {
          int hit_row = cluster.minHitRow() + cluster.hitOffset()[ihit * 2];
          int hit_col = cluster.minHitCol() + cluster.hitOffset()[ihit * 2 + 1];

          // Match the RECO hit to the corresponding SIM hit
          for (const auto& recHit : *etlRecHitsHandle) {
            ETLDetId detId(recHit.id().rawId());

            DetId geoId = detId.geographicalId();
            const MTDGeomDet* thedet = geom->idToDet(geoId);
            const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
            const RectangularMTDTopology& topo =
                static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

            Local3DPoint local_point(topo.localX(recHit.row()), topo.localY(recHit.column()), 0.);

            std::pair<uint8_t, uint8_t> pixel = geomUtil.pixelInModule(detId, local_point);
            mtd_digitizer::MTDCellId pixelId(detId.rawId(), pixel.first, pixel.second);

            if (m_etlSimHits[idet].count(pixelId) == 0)
              continue;

            // Check the hit position
            if (detId.zside() != cluId.zside() || detId.mtdRR() != cluId.mtdRR() || detId.module() != cluId.module() ||
                recHit.row() != hit_row || recHit.column() != hit_col)
              continue;

            // Check the hit time
            if (recHit.time() != cluster.hitTIME()[ihit])
              continue;

            // SIM hit's position in the module reference frame
            Local3DPoint local_point_sim(convertMmToCm(m_etlSimHits[idet][pixelId].x),
                                         convertMmToCm(m_etlSimHits[idet][pixelId].y),
                                         convertMmToCm(m_etlSimHits[idet][pixelId].z));

            // Calculate the SIM cluster's position in the module reference frame
            cluLocXSIM += local_point_sim.x() * m_etlSimHits[idet][pixelId].energy;
            cluLocYSIM += local_point_sim.y() * m_etlSimHits[idet][pixelId].energy;
            cluLocZSIM += local_point_sim.z() * m_etlSimHits[idet][pixelId].energy;

            // Calculate the SIM cluster energy and time
            cluEneSIM += m_etlSimHits[idet][pixelId].energy;
            cluTimeSIM += m_etlSimHits[idet][pixelId].time * m_etlSimHits[idet][pixelId].energy;

            break;

          }  // recHit loop

        }  // ihit loop
      }

      // Find the MTDTrackingRecHit corresponding to the cluster
      MTDTrackingRecHit* comp(nullptr);
      bool matchClu = false;
      const auto& trkHits = (*mtdTrkHitHandle)[detIdObject];
      for (const auto& trkHit : trkHits) {
        if (isSameCluster(trkHit.mtdCluster(), cluster)) {
          comp = trkHit.clone();
          matchClu = true;
          break;
        }
      }
      if (!matchClu) {
        edm::LogWarning("BtlLocalRecoValidation")
            << "No valid TrackingRecHit corresponding to cluster, detId = " << detIdObject.rawId();
      }

      // --- Fill the cluster resolution histograms
      int iside = (cluId.zside() == -1 ? 0 : 1);
      if (optionalPlots_) {
        if (cluTimeSIM > 0. && cluEneSIM > 0.) {
          cluTimeSIM /= cluEneSIM;

          Local3DPoint cluLocalPosSIM(cluLocXSIM / cluEneSIM, cluLocYSIM / cluEneSIM, cluLocZSIM / cluEneSIM);
          const auto& cluGlobalPosSIM = genericDet->toGlobal(cluLocalPosSIM);

          float time_res = cluster.time() - cluTimeSIM;
          float x_res = global_point.x() - cluGlobalPosSIM.x();
          float y_res = global_point.y() - cluGlobalPosSIM.y();
          float z_res = global_point.z() - cluGlobalPosSIM.z();

          meCluTimeRes_[iside]->Fill(time_res);
          meCluXRes_[iside]->Fill(x_res);
          meCluYRes_[iside]->Fill(y_res);
          meCluZRes_[iside]->Fill(z_res);

          meCluTPullvsEta_[iside]->Fill(cluGlobalPosSIM.eta(), time_res / cluster.timeError());
          meCluTPullvsE_[iside]->Fill(cluEneSIM, time_res / cluster.timeError());

          if (matchClu && comp != nullptr) {
            meCluXPull_[iside]->Fill(x_res / std::sqrt(comp->globalPositionError().cxx()));
            meCluYPull_[iside]->Fill(y_res / std::sqrt(comp->globalPositionError().cyy()));
            meCluXLocalErr_[iside]->Fill(std::sqrt(comp->localPositionError().xx()));
            meCluYLocalErr_[iside]->Fill(std::sqrt(comp->localPositionError().yy()));
          }
          meCluYXLocal_[iside]->Fill(local_point.x(), local_point.y());
          meCluYXLocalSim_[iside]->Fill(cluLocalPosSIM.x(), cluLocalPosSIM.y());

        }  // if ( cluTimeSIM > 0. &&  cluEneSIM > 0. )
      }

      // --- Fill the cluster resolution histograms using MtdSimLayerClusters as mtd truth
      edm::Ref<edmNew::DetSetVector<FTLCluster>, FTLCluster> clusterRef = edmNew::makeRefTo(etlRecCluHandle, &cluster);
      auto itp = r2sAssociationMap.equal_range(clusterRef);
      if (itp.first != itp.second) {
        std::vector<MtdSimLayerClusterRef> simClustersRefs =
            (*itp.first).second;  // the range of itp.first, itp.second should be always 1
        for (unsigned int i = 0; i < simClustersRefs.size(); i++) {
          auto simClusterRef = simClustersRefs[i];

          float simClusEnergy = convertUnitsTo(0.001_MeV, (*simClusterRef).simLCEnergy());  // GeV --> MeV
          float simClusTime = (*simClusterRef).simLCTime();
          LocalPoint simClusLocalPos = (*simClusterRef).simLCPos();
          const auto& simClusGlobalPos = genericDet->toGlobal(simClusLocalPos);

          float time_res = cluster.time() - simClusTime;
          float x_res = global_point.x() - simClusGlobalPos.x();
          float y_res = global_point.y() - simClusGlobalPos.y();
          float z_res = global_point.z() - simClusGlobalPos.z();

          meCluTimeRes_simLC_[iside]->Fill(time_res);
          meCluXRes_simLC_[iside]->Fill(x_res);
          meCluYRes_simLC_[iside]->Fill(y_res);
          meCluZRes_simLC_[iside]->Fill(z_res);

          meCluTPullvsEta_simLC_[iside]->Fill(simClusGlobalPos.eta(), time_res / cluster.timeError());
          meCluTPullvsE_simLC_[iside]->Fill(simClusEnergy, time_res / cluster.timeError());

          if (matchClu && comp != nullptr) {
            meCluXPull_simLC_[iside]->Fill(x_res / std::sqrt(comp->globalPositionError().cxx()));
            meCluYPull_simLC_[iside]->Fill(y_res / std::sqrt(comp->globalPositionError().cyy()));
          }
          if (optionalPlots_) {
            meCluYXLocalSim_simLC_[iside]->Fill(simClusLocalPos.x(), simClusLocalPos.y());
          }

        }  // loop over MtdSimLayerClusters
      }

    }  // cluster loop

  }  // DetSetClu loop

  // --- Loop over the ETL Uncalibrated RECO hits
  if (optionalPlots_) {
    if (uncalibRecHitsPlots_) {
      auto etlUncalibRecHitsHandle = makeValid(iEvent.getHandle(etlUncalibRecHitsToken_));

      for (const auto& uRecHit : *etlUncalibRecHitsHandle) {
        ETLDetId detId = uRecHit.id();
        int idet = detId.zside() + detId.nDisc();

        DetId geoId = detId.geographicalId();
        const MTDGeomDet* thedet = geom->idToDet(geoId);
        const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
        const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

        Local3DPoint local_point(topo.localX(uRecHit.row()), topo.localY(uRecHit.column()), 0.);
        const auto& global_point = thedet->toGlobal(local_point);

        std::pair<uint8_t, uint8_t> pixel = geomUtil.pixelInModule(detId, local_point);
        mtd_digitizer::MTDCellId pixelId(detId.rawId(), pixel.first, pixel.second);

        // --- Skip UncalibratedRecHits not matched to SimHits
        if (m_etlSimHits[idet].count(pixelId) == 0)
          continue;

        if (thedet == nullptr)
          throw cms::Exception("EtlLocalRecoValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                         << detId.rawId() << ") is invalid!" << std::dec << std::endl;

        // --- Fill the histograms

        if (uRecHit.amplitude().first < hitMinAmplitude_)
          continue;

        float time_res = uRecHit.time().first - m_etlSimHits[idet][pixelId].time;

        int iside = (detId.zside() == -1 ? 0 : 1);

        // amplitude histograms

        int totBin = (int)(uRecHit.amplitude().first / binWidthTot_);
        if (totBin > nBinsTot_ - 1)
          totBin = nBinsTot_ - 1;

        meTimeResTot_[iside][totBin]->Fill(time_res);

        // eta histograms

        int etaBin = (int)((fabs(global_point.eta()) - etaMin_) / binWidthEta_);
        if (etaBin < 0)
          etaBin = 0;
        else if (etaBin > nBinsEta_ - 1)
          etaBin = nBinsEta_ - 1;

        meTimeResEta_[iside][etaBin]->Fill(time_res);

      }  // uRecHit loop
    }
  }
}

// ------------ method for histogram booking ------------
void EtlLocalRecoValidation::bookHistograms(DQMStore::IBooker& ibook,
                                            edm::Run const& run,
                                            edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNhits_[0] = ibook.book1D("EtlNhitsZnegD1",
                             "Number of ETL RECO hits (-Z, Single(topo1D)/First(topo2D) disk);log_10(N_{RECO})",
                             100,
                             0.,
                             5.25);
  meNhits_[1] =
      ibook.book1D("EtlNhitsZnegD2", "Number of ETL RECO hits (-Z, Second disk);log_10(N_{RECO})", 100, 0., 5.25);
  meNhits_[2] = ibook.book1D("EtlNhitsZposD1",
                             "Number of ETL RECO hits (+Z, Single(topo1D)/First(topo2D) disk);log_10(N_{RECO})",
                             100,
                             0.,
                             5.25);
  meNhits_[3] =
      ibook.book1D("EtlNhitsZposD2", "Number of ETL RECO hits (+Z, Second disk);log_10(N_{RECO})", 100, 0., 5.25);
  meHitTime_[0] = ibook.book1D(
      "EtlHitTimeZnegD1", "ETL RECO hits ToA (-Z, Single(topo1D)/First(topo2D) disk);ToA_{RECO} [ns]", 100, 0., 25.);
  meHitTime_[1] = ibook.book1D("EtlHitTimeZnegD2", "ETL RECO hits ToA (-Z, Second disk);ToA_{RECO} [ns]", 100, 0., 25.);
  meHitTime_[2] = ibook.book1D(
      "EtlHitTimeZposD1", "ETL RECO hits ToA (+Z, Single(topo1D)/First(topo2D) disk);ToA_{RECO} [ns]", 100, 0., 25.);
  meHitTime_[3] = ibook.book1D("EtlHitTimeZposD2", "ETL RECO hits ToA (+Z, Second disk);ToA_{RECO} [ns]", 100, 0., 25.);
  meHitTimeError_[0] =
      ibook.book1D("EtlHitTimeErrorZnegD1",
                   "ETL RECO hits ToA error (-Z, Single(topo1D)/First(topo2D) disk);#sigma^{ToA}_{RECO} [ns]",
                   50,
                   0.,
                   0.1);
  meHitTimeError_[1] = ibook.book1D(
      "EtlHitTimeErrorZnegD2", "ETL RECO hits ToA error(-Z, Second disk);#sigma^{ToA}_{RECO} [ns]", 50, 0., 0.1);
  meHitTimeError_[2] =
      ibook.book1D("EtlHitTimeErrorZposD1",
                   "ETL RECO hits ToA error (+Z, Single(topo1D)/First(topo2D) disk);#sigma^{ToA}_{RECO} [ns]",
                   50,
                   0.,
                   0.1);
  meHitTimeError_[3] = ibook.book1D(
      "EtlHitTimeErrorZposD2", "ETL RECO hits ToA error(+Z, Second disk);#sigma^{ToA}_{RECO} [ns]", 50, 0., 0.1);

  if (optionalPlots_) {
    meOccupancy_[0] =
        ibook.book2D("EtlOccupancyZnegD1",
                     "ETL RECO hits occupancy (-Z, Single(topo1D)/First(topo2D) disk);X_{RECO} [cm];Y_{RECO} [cm]",
                     135,
                     -135.,
                     135.,
                     135,
                     -135.,
                     135.);
    meOccupancy_[1] = ibook.book2D("EtlOccupancyZnegD2",
                                   "ETL RECO hits occupancy (-Z, Second disk);X_{RECO} [cm];Y_{RECO} [cm]",
                                   135,
                                   -135.,
                                   135.,
                                   135,
                                   -135.,
                                   135.);
    meOccupancy_[2] =
        ibook.book2D("EtlOccupancyZposD1",
                     "ETL RECO hits occupancy (+Z, Single(topo1D)/First(topo2D) disk);X_{RECO} [cm];Y_{RECO} [cm]",
                     135,
                     -135.,
                     135.,
                     135,
                     -135.,
                     135.);
    meOccupancy_[3] = ibook.book2D("EtlOccupancyZposD2",
                                   "ETL RECO hits occupancy (+Z, Second disk);X_{RECO} [cm];Y_{RECO} [cm]",
                                   135,
                                   -135.,
                                   135.,
                                   135,
                                   -135.,
                                   135.);
    meLocalOccupancy_[0] = ibook.book2D("EtlLocalOccupancyZneg",
                                        "ETL RECO hits local occupancy (-Z);X_{RECO} [cm];Y_{RECO} [cm]",
                                        100,
                                        -2.2,
                                        2.2,
                                        50,
                                        -1.1,
                                        1.1);
    meLocalOccupancy_[1] = ibook.book2D("EtlLocalOccupancyZpos",
                                        "ETL RECO hits local occupancy (+Z);X_{RECO} [cm];Y_{RECO} [cm]",
                                        100,
                                        -2.2,
                                        2.2,
                                        50,
                                        -1.1,
                                        1.1);
  }
  meHitXlocal_[0] = ibook.book1D("EtlHitXlocalZneg", "ETL RECO local X (-Z);X_{RECO}^{LOC} [cm]", 100, -2.2, 2.2);
  meHitXlocal_[1] = ibook.book1D("EtlHitXlocalZpos", "ETL RECO local X (+Z);X_{RECO}^{LOC} [cm]", 100, -2.2, 2.2);
  meHitYlocal_[0] = ibook.book1D("EtlHitYlocalZneg", "ETL RECO local Y (-Z);Y_{RECO}^{LOC} [cm]", 50, -1.1, 1.1);
  meHitYlocal_[1] = ibook.book1D("EtlHitYlocalZpos", "ETL RECO local Y (-Z);Y_{RECO}^{LOC} [cm]", 50, -1.1, 1.1);
  meHitX_[0] = ibook.book1D(
      "EtlHitXZnegD1", "ETL RECO hits X (-Z, Single(topo1D)/First(topo2D) Disk);X_{RECO} [cm]", 100, -130., 130.);
  meHitX_[1] = ibook.book1D("EtlHitXZnegD2", "ETL RECO hits X (-Z, Second Disk);X_{RECO} [cm]", 100, -130., 130.);
  meHitX_[2] = ibook.book1D(
      "EtlHitXZposD1", "ETL RECO hits X (+Z, Single(topo1D)/First(topo2D) Disk);X_{RECO} [cm]", 100, -130., 130.);
  meHitX_[3] = ibook.book1D("EtlHitXZposD2", "ETL RECO hits X (+Z, Second Disk);X_{RECO} [cm]", 100, -130., 130.);
  meHitY_[0] = ibook.book1D(
      "EtlHitYZnegD1", "ETL RECO hits Y (-Z, Single(topo1D)/First(topo2D) Disk);Y_{RECO} [cm]", 100, -130., 130.);
  meHitY_[1] = ibook.book1D("EtlHitYZnegD2", "ETL RECO hits Y (-Z, Second Disk);Y_{RECO} [cm]", 100, -130., 130.);
  meHitY_[2] = ibook.book1D(
      "EtlHitYZposD1", "ETL RECO hits Y (+Z, Single(topo1D)/First(topo2D) Disk);Y_{RECO} [cm]", 100, -130., 130.);
  meHitY_[3] = ibook.book1D("EtlHitYZposD2", "ETL RECO hits Y (+Z, Second Disk);Y_{RECO} [cm]", 100, -130., 130.);
  meHitZ_[0] = ibook.book1D(
      "EtlHitZZnegD1", "ETL RECO hits Z (-Z, Single(topo1D)/First(topo2D) Disk);Z_{RECO} [cm]", 100, -302., -298.);
  meHitZ_[1] = ibook.book1D("EtlHitZZnegD2", "ETL RECO hits Z (-Z, Second Disk);Z_{RECO} [cm]", 100, -304., -300.);
  meHitZ_[2] = ibook.book1D(
      "EtlHitZZposD1", "ETL RECO hits Z (+Z, Single(topo1D)/First(topo2D) Disk);Z_{RECO} [cm]", 100, 298., 302.);
  meHitZ_[3] = ibook.book1D("EtlHitZZposD2", "ETL RECO hits Z (+Z, Second Disk);Z_{RECO} [cm]", 100, 300., 304.);
  meHitPhi_[0] = ibook.book1D(
      "EtlHitPhiZnegD1", "ETL RECO hits #phi (-Z, Single(topo1D)/First(topo2D) Disk);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meHitPhi_[1] =
      ibook.book1D("EtlHitPhiZnegD2", "ETL RECO hits #phi (-Z, Second Disk);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meHitPhi_[2] = ibook.book1D(
      "EtlHitPhiZposD1", "ETL RECO hits #phi (+Z, Single(topo1D)/First(topo2D) Disk);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meHitPhi_[3] =
      ibook.book1D("EtlHitPhiZposD2", "ETL RECO hits #phi (+Z, Second Disk);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meHitEta_[0] = ibook.book1D(
      "EtlHitEtaZnegD1", "ETL RECO hits #eta (-Z, Single(topo1D)/First(topo2D) Disk);#eta_{RECO}", 100, -3.2, -1.56);
  meHitEta_[1] = ibook.book1D("EtlHitEtaZnegD2", "ETL RECO hits #eta (-Z, Second Disk);#eta_{RECO}", 100, -3.2, -1.56);
  meHitEta_[2] = ibook.book1D(
      "EtlHitEtaZposD1", "ETL RECO hits #eta (+Z, Single(topo1D)/First(topo2D) Disk);#eta_{RECO}", 100, 1.56, 3.2);
  meHitEta_[3] = ibook.book1D("EtlHitEtaZposD2", "ETL RECO hits #eta (+Z, Second Disk);#eta_{RECO}", 100, 1.56, 3.2);

  meTimeRes_ = ibook.book1D("EtlTimeRes", "ETL time resolution;T_{RECO}-T_{SIM}", 100, -0.5, 0.5);
  meHitTvsPhi_[0] = ibook.bookProfile(
      "EtlHitTvsPhiZnegD1",
      "ETL RECO time vs #phi (-Z, Single(topo1D)/First(topo2D) Disk);#phi_{RECO} [rad];ToA_{RECO} [ns]",
      50,
      -3.2,
      3.2,
      0.,
      100.);
  meHitTvsPhi_[1] = ibook.bookProfile("EtlHitTvsPhiZnegD2",
                                      "ETL RECO time vs #phi (-Z, Second Disk);#phi_{RECO} [rad];ToA_{RECO} [ns]",
                                      50,
                                      -3.2,
                                      3.2,
                                      0.,
                                      100.);
  meHitTvsPhi_[2] = ibook.bookProfile(
      "EtlHitTvsPhiZposD1",
      "ETL RECO time vs #phi (+Z, Single(topo1D)/First(topo2D) Disk);#phi_{RECO} [rad];ToA_{RECO} [ns]",
      50,
      -3.2,
      3.2,
      0.,
      100.);
  meHitTvsPhi_[3] = ibook.bookProfile("EtlHitTvsPhiZposD2",
                                      "ETL RECO time vs #phi (+Z, Second Disk);#phi_{RECO} [rad];ToA_{RECO} [ns]",
                                      50,
                                      -3.2,
                                      3.2,
                                      0.,
                                      100.);
  meHitTvsEta_[0] =
      ibook.bookProfile("EtlHitTvsEtaZnegD1",
                        "ETL RECO time vs #eta (-Z, Single(topo1D)/First(topo2D) Disk);#eta_{RECO};ToA_{RECO} [ns]",
                        50,
                        -3.2,
                        -1.56,
                        0.,
                        100.);
  meHitTvsEta_[1] = ibook.bookProfile("EtlHitTvsEtaZnegD2",
                                      "ETL RECO time vs #eta (-Z, Second Disk);#eta_{RECO};ToA_{RECO} [ns]",
                                      50,
                                      -3.2,
                                      -1.56,
                                      0.,
                                      100.);
  meHitTvsEta_[2] =
      ibook.bookProfile("EtlHitTvsEtaZposD1",
                        "ETL RECO time vs #eta (+Z, Single(topo1D)/First(topo2D) Disk);#eta_{RECO};ToA_{RECO} [ns]",
                        50,
                        1.56,
                        3.2,
                        0.,
                        100.);
  meHitTvsEta_[3] = ibook.bookProfile("EtlHitTvsEtaZposD2",
                                      "ETL RECO time vs #eta (+Z, Second Disk);#eta_{RECO};ToA_{RECO} [ns]",
                                      50,
                                      1.56,
                                      3.2,
                                      0.,
                                      100.);
  meTPullvsE_ = ibook.bookProfile(
      "EtlTPullvsE", "ETL time pull vs E;E_{SIM} [MeV];(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}", 20, 0., 2., -5., 5., "S");
  meTPullvsEta_ = ibook.bookProfile("EtlTPullvsEta",
                                    "ETL time pull vs #eta;|#eta_{RECO}|;(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
                                    26,
                                    1.65,
                                    3.0,
                                    -5.,
                                    5.,
                                    "S");
  meCluTime_[0] =
      ibook.book1D("EtlCluTimeZnegD1", "ETL cluster ToA (-Z, Single(topo1D)/First(topo2D) Disk);ToA [ns]", 250, 0, 25);
  meCluTime_[1] = ibook.book1D("EtlCluTimeZnegD2", "ETL cluster ToA (-Z, Second Disk);ToA [ns]", 250, 0, 25);
  meCluTime_[2] =
      ibook.book1D("EtlCluTimeZposD1", "ETL cluster ToA (+Z, Single(topo1D)/First(topo2D) Disk);ToA [ns]", 250, 0, 25);
  meCluTime_[3] = ibook.book1D("EtlCluTimeZposD2", "ETL cluster ToA (+Z, Second Disk);ToA [ns]", 250, 0, 25);
  meCluTimeError_[0] = ibook.book1D("EtlCluTimeErrosZnegD1",
                                    "ETL cluster time error (-Z, Single(topo1D)/First(topo2D) Disk);#sigma_{t} [ns]",
                                    100,
                                    0,
                                    0.1);
  meCluTimeError_[1] =
      ibook.book1D("EtlCluTimeErrorZnegD2", "ETL cluster time error (-Z, Second Disk);#sigma_{t} [ns]", 100, 0, 0.1);
  meCluTimeError_[2] = ibook.book1D("EtlCluTimeErrorZposD1",
                                    "ETL cluster time error (+Z, Single(topo1D)/First(topo2D) Disk);#sigma_{t} [ns]",
                                    100,
                                    0,
                                    0.1);
  meCluTimeError_[3] =
      ibook.book1D("EtlCluTimeErrorZposD2", "ETL cluster time error (+Z, Second Disk);#sigma_{t} [ns]", 100, 0, 0.1);
  meCluPhi_[0] = ibook.book1D(
      "EtlCluPhiZnegD1", "ETL cluster #phi (-Z, Single(topo1D)/First(topo2D) Disk);#phi_{RECO} [rad]", 126, -3.2, 3.2);
  meCluPhi_[1] =
      ibook.book1D("EtlCluPhiZnegD2", "ETL cluster #phi (-Z, Second Disk);#phi_{RECO} [rad]", 126, -3.2, 3.2);
  meCluPhi_[2] = ibook.book1D(
      "EtlCluPhiZposD1", "ETL cluster #phi (+Z, Single(topo1D)/First(topo2D) Disk);#phi_{RECO} [rad]", 126, -3.2, 3.2);
  meCluPhi_[3] =
      ibook.book1D("EtlCluPhiZposD2", "ETL cluster #phi (+Z, Second Disk);#phi_{RECO} [rad]", 126, -3.2, 3.2);
  meCluEta_[0] = ibook.book1D(
      "EtlCluEtaZnegD1", "ETL cluster #eta (-Z, Single(topo1D)/First(topo2D) Disk);#eta_{RECO}", 100, -3.2, -1.4);
  meCluEta_[1] = ibook.book1D("EtlCluEtaZnegD2", "ETL cluster #eta (-Z, Second Disk);#eta_{RECO}", 100, -3.2, -1.4);
  meCluEta_[2] = ibook.book1D(
      "EtlCluEtaZposD1", "ETL cluster #eta (+Z, Single(topo1D)/First(topo2D) Disk);#eta_{RECO}", 100, 1.4, 3.2);
  meCluEta_[3] = ibook.book1D("EtlCluEtaZposD2", "ETL cluster #eta (+Z, Second Disk);#eta_{RECO}", 100, 1.4, 3.2);
  meCluHits_[0] = ibook.book1D(
      "EtlCluHitNumberZnegD1", "ETL hits per cluster (-Z, Single(topo1D)/First(topo2D) Disk);Cluster size", 5, 0, 5);
  meCluHits_[1] = ibook.book1D("EtlCluHitNumberZnegD2", "ETL hits per cluster (-Z, Second Disk);Cluster size", 5, 0, 5);
  meCluHits_[2] = ibook.book1D(
      "EtlCluHitNumberZposD1", "ETL hits per cluster (+Z, Single(topo1D)/First(topo2D) Disk);Cluster size", 5, 0, 5);
  meCluHits_[3] = ibook.book1D("EtlCluHitNumberZposD2", "ETL hits per cluster (+Z, Second Disk);Cluster size", 5, 0, 5);

  if (optionalPlots_) {
    meCluTimeRes_[0] =
        ibook.book1D("EtlCluTimeResZneg", "ETL cluster time resolution (-Z);T_{RECO}-T_{SIM} [ns]", 100, -0.5, 0.5);
    meCluTimeRes_[1] =
        ibook.book1D("EtlCluTimeResZpos", "ETL cluster time resolution (+Z);T_{RECO}-T_{SIM} [MeV]", 100, -0.5, 0.5);

    meCluTPullvsE_[0] =
        ibook.bookProfile("EtlCluTPullvsEZneg",
                          "ETL cluster time pull vs E (-Z);E_{SIM} [MeV];(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
                          25,
                          0.,
                          0.5,
                          -5.,
                          5.,
                          "S");
    meCluTPullvsE_[1] =
        ibook.bookProfile("EtlCluTPullvsEZpos",
                          "ETL cluster time pull vs E (+Z);E_{SIM} [MeV];(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
                          25,
                          0.,
                          0.5,
                          -5.,
                          5.,
                          "S");
    meCluTPullvsEta_[0] =
        ibook.bookProfile("EtlCluTPullvsEtaZneg",
                          "ETL cluster time pull vs #eta (-Z);|#eta_{RECO}|;(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
                          30,
                          -3.,
                          -1.65,
                          -5.,
                          5.,
                          "S");
    meCluTPullvsEta_[1] =
        ibook.bookProfile("EtlCluTPullvsEtaZpos",
                          "ETL cluster time pull vs #eta (+Z);|#eta_{RECO}|;(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
                          30,
                          1.65,
                          3.,
                          -5.,
                          5.,
                          "S");
    meCluXRes_[0] =
        ibook.book1D("EtlCluXResZneg", "ETL cluster X resolution (-Z);X_{RECO}-X_{SIM} [cm]", 100, -0.1, 0.1);
    meCluXRes_[1] =
        ibook.book1D("EtlCluXResZpos", "ETL cluster X resolution (+Z);X_{RECO}-X_{SIM} [cm]", 100, -0.1, 0.1);
    meCluYRes_[0] =
        ibook.book1D("EtlCluYResZneg", "ETL cluster Y resolution (-Z);Y_{RECO}-Y_{SIM} [cm]", 100, -0.1, 0.1);
    meCluYRes_[1] =
        ibook.book1D("EtlCluYResZpos", "ETL cluster Y resolution (+Z);Y_{RECO}-Y_{SIM} [cm]", 100, -0.1, 0.1);
    meCluZRes_[0] =
        ibook.book1D("EtlCluZResZneg", "ETL cluster Z resolution (-Z);Z_{RECO}-Z_{SIM} [cm]", 100, -0.003, 0.003);
    meCluZRes_[1] =
        ibook.book1D("EtlCluZResZpos", "ETL cluster Z resolution (+Z);Z_{RECO}-Z_{SIM} [cm]", 100, -0.003, 0.003);
    meCluXPull_[0] =
        ibook.book1D("EtlCluXPullZneg", "ETL cluster X pull (-Z);X_{RECO}-X_{SIM}/sigmaX_[RECO] [cm]", 100, -5., 5.);
    meCluXPull_[1] =
        ibook.book1D("EtlCluXPullZpos", "ETL cluster X pull (+Z);X_{RECO}-X_{SIM}/sigmaX_[RECO] [cm]", 100, -5., 5.);
    meCluYPull_[0] =
        ibook.book1D("EtlCluYPullZneg", "ETL cluster Y pull (-Z);Y_{RECO}-Y_{SIM}/sigmaY_[RECO] [cm]", 100, -5., 5.);
    meCluYPull_[1] =
        ibook.book1D("EtlCluYPullZpos", "ETL cluster Y pull (+Z);Y_{RECO}-Y_{SIM}/sigmaY_[RECO] [cm]", 100, -5., 5.);
    meCluXLocalErr_[0] =
        ibook.book1D("EtlCluXLocalErrNeg", "ETL cluster X local error (-Z);sigmaX_{RECO,loc} [cm]", 50, 0., 0.2);
    meCluXLocalErr_[1] =
        ibook.book1D("EtlCluXLocalErrPos", "ETL cluster X local error (+Z);sigmaX_{RECO,loc} [cm]", 50, 0., 0.2);
    meCluYLocalErr_[0] =
        ibook.book1D("EtlCluYLocalErrNeg", "ETL cluster Y local error (-Z);sigmaY_{RECO,loc} [cm]", 50., 0., 0.2);
    meCluYLocalErr_[1] =
        ibook.book1D("EtlCluYLocalErrPos", "ETL cluster Y local error (+Z);sigmaY_{RECO,loc} [cm]", 50, 0., 0.2);

    meCluOccupancy_[0] =
        ibook.book2D("EtlCluOccupancyZnegD1",
                     "ETL cluster X vs Y (-Z, Single(topo1D)/First(topo2D) Disk);X_{RECO} [cm]; Y_{RECO} [cm]",
                     100,
                     -150.,
                     150.,
                     100,
                     -150,
                     150);
    meCluOccupancy_[1] = ibook.book2D("EtlCluOccupancyZnegD2",
                                      "ETL cluster X vs Y (-Z, Second Disk);X_{RECO} [cm]; Y_{RECO} [cm]",
                                      100,
                                      -150.,
                                      150.,
                                      100,
                                      -150,
                                      150);
    meCluOccupancy_[2] =
        ibook.book2D("EtlCluOccupancyZposD1",
                     "ETL cluster X vs Y (+Z, Single(topo1D)/First(topo2D) Disk);X_{RECO} [cm]; Y_{RECO} [cm]",
                     100,
                     -150.,
                     150.,
                     100,
                     -150,
                     150);
    meCluOccupancy_[3] = ibook.book2D("EtlCluOccupancyZposD2",
                                      "ETL cluster X vs Y (+Z, Second Disk);X_{RECO} [cm]; Y_{RECO} [cm]",
                                      100,
                                      -150.,
                                      150.,
                                      100,
                                      -150,
                                      150);
    meCluYXLocal_[0] = ibook.book2D("EtlCluYXLocalZneg",
                                    "ETL cluster local Y vs X (-Z);X^{local}_{RECO} [cm];Y^{local}_{RECO} [cm]",
                                    100,
                                    -2.2,
                                    2.2,
                                    100,
                                    -1.1,
                                    1.1);
    meCluYXLocal_[1] = ibook.book2D("EtlCluYXLocalZpos",
                                    "ETL cluster local Y vs X (+Z);X^{local}_{RECO} [cm];Y^{local}_{RECO} [cm]",
                                    100,
                                    -2.2,
                                    2.2,
                                    100,
                                    -1.1,
                                    1.1);
    meCluYXLocalSim_[0] = ibook.book2D("EtlCluYXLocalSimZneg",
                                       "ETL cluster local Y vs X (-Z);X^{local}_{SIM} [cm];Y^{local}_{SIM} [cm]",
                                       200,
                                       -2.2,
                                       2.2,
                                       200,
                                       -1.1,
                                       1.1);
    meCluYXLocalSim_[1] = ibook.book2D("EtlCluYXLocalSimZpos",
                                       "ETL cluster local Y vs X (+Z);X^{local}_{SIM} [cm];Y^{local}_{SIM} [cm]",
                                       200,
                                       -2.2,
                                       2.2,
                                       200,
                                       -1.1,
                                       1.1);
    meCluYXLocalSim_simLC_[0] =
        ibook.book2D("EtlCluYXLocalSimZneg_simLC",
                     "ETL cluster local Y vs X (-Z, MtdSimLayerClusters);X^{local}_{SIM} [cm];Y^{local}_{SIM} [cm]",
                     200,
                     -2.2,
                     2.2,
                     200,
                     -1.1,
                     1.1);
    meCluYXLocalSim_simLC_[1] =
        ibook.book2D("EtlCluYXLocalSimZpos_simLC",
                     "ETL cluster local Y vs X (+Z, MtdSimLayerClusters);X^{local}_{SIM} [cm];Y^{local}_{SIM} [cm]",
                     200,
                     -2.2,
                     2.2,
                     200,
                     -1.1,
                     1.1);
  }

  // resolution plots using MtdSimLayerClusters as truth
  meCluTimeRes_simLC_[0] = ibook.book1D("EtlCluTimeResZneg_simLC",
                                        "ETL cluster time resolution (MtdSimLayerClusters, -Z);T_{RECO}-T_{SIM} [ns]",
                                        100,
                                        -0.5,
                                        0.5);
  meCluTimeRes_simLC_[1] = ibook.book1D("EtlCluTimeResZpos_simLC",
                                        "ETL cluster time resolution (MtdSimLayerClusters, +Z);T_{RECO}-T_{SIM} [MeV]",
                                        100,
                                        -0.5,
                                        0.5);

  meCluTPullvsE_simLC_[0] = ibook.bookProfile(
      "EtlCluTPullvsEZneg_simLC",
      "ETL cluster time pull vs E (MtdSimLayerClusters, -Z);E_{SIM} [MeV];(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
      25,
      0.,
      0.5,
      -5.,
      5.,
      "S");
  meCluTPullvsE_simLC_[1] = ibook.bookProfile(
      "EtlCluTPullvsEZpos_simLC",
      "ETL cluster time pull vs E (MtdSimLayerClusters, +Z);E_{SIM} [MeV];(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
      25,
      0.,
      0.5,
      -5.,
      5.,
      "S");
  meCluTPullvsEta_simLC_[0] = ibook.bookProfile(
      "EtlCluTPullvsEtaZneg_simLC",
      "ETL cluster time pull vs #eta (MtdSimLayerClusters, -Z);|#eta_{RECO}|;(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
      30,
      -3.,
      -1.65,
      -5.,
      5.,
      "S");
  meCluTPullvsEta_simLC_[1] = ibook.bookProfile(
      "EtlCluTPullvsEtaZpos_simLC",
      "ETL cluster time pull vs #eta (MtdSimLayerClusters, +Z);|#eta_{RECO}|;(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
      30,
      1.65,
      3.,
      -5.,
      5.,
      "S");
  meCluXRes_simLC_[0] = ibook.book1D("EtlCluXResZneg_simLC",
                                     "ETL cluster X resolution (MtdSimLayerClusters, -Z);X_{RECO}-X_{SIM} [cm]",
                                     100,
                                     -0.1,
                                     0.1);
  meCluXRes_simLC_[1] = ibook.book1D("EtlCluXResZpos_simLC",
                                     "ETL cluster X resolution (MtdSimLayerClusters, +Z);X_{RECO}-X_{SIM} [cm]",
                                     100,
                                     -0.1,
                                     0.1);
  meCluYRes_simLC_[0] = ibook.book1D("EtlCluYResZneg_simLC",
                                     "ETL cluster Y resolution (MtdSimLayerClusters, -Z);Y_{RECO}-Y_{SIM} [cm]",
                                     100,
                                     -0.1,
                                     0.1);
  meCluYRes_simLC_[1] = ibook.book1D("EtlCluYResZpos_simLC",
                                     "ETL cluster Y resolution (MtdSimLayerClusters, +Z);Y_{RECO}-Y_{SIM} [cm]",
                                     100,
                                     -0.1,
                                     0.1);
  meCluZRes_simLC_[0] = ibook.book1D("EtlCluZResZneg_simLC",
                                     "ETL cluster Z resolution (MtdSimLayerClusters, -Z);Z_{RECO}-Z_{SIM} [cm]",
                                     100,
                                     -0.003,
                                     0.003);
  meCluZRes_simLC_[1] = ibook.book1D("EtlCluZResZpos_simLC",
                                     "ETL cluster Z resolution (MtdSimLayerClusters, +Z);Z_{RECO}-Z_{SIM} [cm]",
                                     100,
                                     -0.003,
                                     0.003);
  meCluXPull_simLC_[0] =
      ibook.book1D("EtlCluXPullZneg_simLC",
                   "ETL cluster X pull (MtdSimLayerClusters, -Z);X_{RECO}-X_{SIM}/sigmaX_[RECO] [cm]",
                   100,
                   -5.,
                   5.);
  meCluXPull_simLC_[1] =
      ibook.book1D("EtlCluXPullZpos_simLC",
                   "ETL cluster X pull (MtdSimLayerClusters, +Z);X_{RECO}-X_{SIM}/sigmaX_[RECO] [cm]",
                   100,
                   -5.,
                   5.);
  meCluYPull_simLC_[0] =
      ibook.book1D("EtlCluYPullZneg_simLC",
                   "ETL cluster Y pull (MtdSimLayerClusters, -Z);Y_{RECO}-Y_{SIM}/sigmaY_[RECO] [cm]",
                   100,
                   -5.,
                   5.);
  meCluYPull_simLC_[1] =
      ibook.book1D("EtlCluYPullZpos_simLC",
                   "ETL cluster Y pull (MtdSimLayerClusters, +Z);Y_{RECO}-Y_{SIM}/sigmaY_[RECO] [cm]",
                   100,
                   -5.,
                   5.);

  // --- UncalibratedRecHits histograms

  if (optionalPlots_) {
    if (uncalibRecHitsPlots_) {
      const std::string det_name[2] = {"ETL-", "ETL+"};
      for (unsigned int iside = 0; iside < 2; ++iside) {
        for (unsigned int ihistoTot = 0; ihistoTot < nBinsTot_; ++ihistoTot) {
          std::string hname = Form("TimeResTot_%d_%d", iside, ihistoTot);
          std::string htitle =
              Form("%s time resolution (Tot bin = %d);T_{RECO} - T_{SIM} [ns]", det_name[iside].data(), ihistoTot);
          meTimeResTot_[iside][ihistoTot] = ibook.book1D(hname, htitle, 200, -0.5, 0.5);

        }  // ihistoTot loop

        for (unsigned int ihistoEta = 0; ihistoEta < nBinsEta_; ++ihistoEta) {
          std::string hname = Form("TimeResEta_%d_%d", iside, ihistoEta);
          std::string htitle =
              Form("%s time resolution (|#eta| bin = %d);T_{RECO} - T_{SIM} [ns]", det_name[iside].data(), ihistoEta);
          meTimeResEta_[iside][ihistoEta] = ibook.book1D(hname, htitle, 200, -0.5, 0.5);

        }  // ihistoEta loop
      }
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void EtlLocalRecoValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ETL/LocalReco");
  desc.add<edm::InputTag>("recHitsTag", edm::InputTag("mtdRecHits", "FTLEndcap"));
  desc.add<edm::InputTag>("uncalibRecHitsTag", edm::InputTag("mtdUncalibratedRecHits", "FTLEndcap"));
  desc.add<edm::InputTag>("simHitsTag", edm::InputTag("mix", "g4SimHitsFastTimerHitsEndcap"));
  desc.add<edm::InputTag>("recCluTag", edm::InputTag("mtdClusters", "FTLEndcap"));
  desc.add<edm::InputTag>("trkHitTag", edm::InputTag("mtdTrackingRecHits"));
  desc.add<edm::InputTag>("r2sAssociationMapTag", edm::InputTag("mtdRecoClusterToSimLayerClusterAssociation"));
  desc.add<double>("hitMinimumEnergy2Dis", 0.001);  // [MeV]
  desc.add<bool>("optionalPlots", false);
  desc.add<bool>("UncalibRecHitsPlots", false);
  desc.add<double>("HitMinimumAmplitude", 0.33);  // [MIP] old, now amplitude for recHit is time_over_threshold in ETL

  descriptions.add("etlLocalRecoValid", desc);
}

DEFINE_FWK_MODULE(EtlLocalRecoValidation);
