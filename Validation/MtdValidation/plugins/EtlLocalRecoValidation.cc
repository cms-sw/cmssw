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

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

struct MTDHit {
  float energy;
  float time;
  float x_local;
  float y_local;
  float z_local;
};

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
  const float hitMinEnergy1Dis_;
  const float hitMinEnergy2Dis_;
  const bool optionalPlots_;
  const bool uncalibRecHitsPlots_;
  const double hitMinAmplitude_;

  edm::EDGetTokenT<FTLRecHitCollection> etlRecHitsToken_;
  edm::EDGetTokenT<FTLUncalibratedRecHitCollection> etlUncalibRecHitsToken_;
  edm::EDGetTokenT<CrossingFrame<PSimHit> > etlSimHitsToken_;
  edm::EDGetTokenT<FTLClusterCollection> etlRecCluToken_;
  edm::EDGetTokenT<MTDTrackingDetSetVector> mtdTrackingHitToken_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

  // --- histograms declaration

  MonitorElement* meNhits_[4];
  MonitorElement* meHitEnergy_[4];
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

  MonitorElement* meHitTvsE_[4];
  MonitorElement* meHitEvsPhi_[4];
  MonitorElement* meHitEvsEta_[4];
  MonitorElement* meHitTvsPhi_[4];
  MonitorElement* meHitTvsEta_[4];

  MonitorElement* meCluTime_[4];
  MonitorElement* meCluTimeError_[4];
  MonitorElement* meCluEnergy_[4];
  MonitorElement* meCluPhi_[4];
  MonitorElement* meCluEta_[4];
  MonitorElement* meCluHits_[4];
  MonitorElement* meCluOccupancy_[4];

  MonitorElement* meTimeRes_;
  MonitorElement* meEnergyRes_;
  MonitorElement* meTPullvsE_;
  MonitorElement* meTPullvsEta_;

  MonitorElement* meCluTimeRes_[2];
  MonitorElement* meCluEnergyRes_[2];
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

  MonitorElement* meUnmatchedCluEnergy_[2];

  // --- UncalibratedRecHits histograms

  static constexpr int nBinsQ_ = 20;
  static constexpr float binWidthQ_ = 1.3;  // in MIP units

  MonitorElement* meTimeResQ_[2][nBinsQ_];

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
      hitMinEnergy1Dis_(iConfig.getParameter<double>("hitMinimumEnergy1Dis")),
      hitMinEnergy2Dis_(iConfig.getParameter<double>("hitMinimumEnergy2Dis")),
      optionalPlots_(iConfig.getParameter<bool>("optionalPlots")),
      uncalibRecHitsPlots_(iConfig.getParameter<bool>("UncalibRecHitsPlots")),
      hitMinAmplitude_(iConfig.getParameter<double>("HitMinimumAmplitude")) {
  etlRecHitsToken_ = consumes<FTLRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsTag"));
  if (uncalibRecHitsPlots_)
    etlUncalibRecHitsToken_ =
        consumes<FTLUncalibratedRecHitCollection>(iConfig.getParameter<edm::InputTag>("uncalibRecHitsTag"));
  etlSimHitsToken_ = consumes<CrossingFrame<PSimHit> >(iConfig.getParameter<edm::InputTag>("simHitsTag"));
  etlRecCluToken_ = consumes<FTLClusterCollection>(iConfig.getParameter<edm::InputTag>("recCluTag"));
  mtdTrackingHitToken_ = consumes<MTDTrackingDetSetVector>(iConfig.getParameter<edm::InputTag>("trkHitTag"));

  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
}

EtlLocalRecoValidation::~EtlLocalRecoValidation() {}

// ------------ method called for each event  ------------
void EtlLocalRecoValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace geant_units::operators;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  bool topo1Dis = false;
  bool topo2Dis = false;
  if (topology->getMTDTopologyMode() <= static_cast<int>(MTDTopologyMode::Mode::barphiflat)) {
    topo1Dis = true;
  }
  if (topology->getMTDTopologyMode() > static_cast<int>(MTDTopologyMode::Mode::barphiflat)) {
    topo2Dis = true;
  }

  auto etlRecHitsHandle = makeValid(iEvent.getHandle(etlRecHitsToken_));
  auto etlSimHitsHandle = makeValid(iEvent.getHandle(etlSimHitsToken_));
  auto etlRecCluHandle = makeValid(iEvent.getHandle(etlRecCluToken_));
  auto mtdTrkHitHandle = makeValid(iEvent.getHandle(mtdTrackingHitToken_));
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
  std::unordered_map<uint32_t, MTDHit> m_etlSimHits[4];
  for (auto const& simHit : etlSimHits) {
    // --- Use only hits compatible with the in-time bunch-crossing
    if (simHit.tof() < 0 || simHit.tof() > 25.)
      continue;

    ETLDetId id = simHit.detUnitId();

    int idet = -1;

    if ((id.zside() == -1) && (id.nDisc() == 1))
      idet = 0;
    else if ((id.zside() == -1) && (id.nDisc() == 2))
      idet = 1;
    else if ((id.zside() == 1) && (id.nDisc() == 1))
      idet = 2;
    else if ((id.zside() == 1) && (id.nDisc() == 2))
      idet = 3;
    else
      continue;

    auto simHitIt = m_etlSimHits[idet].emplace(id.rawId(), MTDHit()).first;

    // --- Accumulate the energy (in MeV) of SIM hits in the same detector cell
    (simHitIt->second).energy += convertUnitsTo(0.001_MeV, simHit.energyLoss());

    // --- Get the time of the first SIM hit in the cell
    if ((simHitIt->second).time == 0 || simHit.tof() < (simHitIt->second).time) {
      (simHitIt->second).time = simHit.tof();

      auto hit_pos = simHit.entryPoint();
      (simHitIt->second).x_local = hit_pos.x();
      (simHitIt->second).y_local = hit_pos.y();
      (simHitIt->second).z_local = hit_pos.z();
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

    if (topo1Dis) {
      if (detId.zside() == -1) {
        idet = 0;
      } else if (detId.zside() == 1) {
        idet = 2;
      } else {
        continue;
      }
    }

    if (topo2Dis) {
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
        continue;
      }
    }

    // --- Fill the histograms

    meHitEnergy_[idet]->Fill(recHit.energy());
    meHitTime_[idet]->Fill(recHit.time());
    meHitTimeError_[idet]->Fill(recHit.timeError());

    meOccupancy_[idet]->Fill(global_point.x(), global_point.y(), weight);

    if (optionalPlots_) {
      if ((idet == 0) || (idet == 1)) {
        meLocalOccupancy_[0]->Fill(local_point.x(), local_point.y());
        meHitXlocal_[0]->Fill(local_point.x());
        meHitYlocal_[0]->Fill(local_point.y());
      }
      if ((idet == 2) || (idet == 3)) {
        meLocalOccupancy_[1]->Fill(local_point.x(), local_point.y());
        meHitXlocal_[1]->Fill(local_point.x());
        meHitYlocal_[1]->Fill(local_point.y());
      }
    }
    meHitX_[idet]->Fill(global_point.x());
    meHitY_[idet]->Fill(global_point.y());
    meHitZ_[idet]->Fill(global_point.z());
    meHitPhi_[idet]->Fill(global_point.phi());
    meHitEta_[idet]->Fill(global_point.eta());
    meHitTvsE_[idet]->Fill(recHit.energy(), recHit.time());
    meHitEvsPhi_[idet]->Fill(global_point.phi(), recHit.energy());
    meHitEvsEta_[idet]->Fill(global_point.eta(), recHit.energy());
    meHitTvsPhi_[idet]->Fill(global_point.phi(), recHit.time());
    meHitTvsEta_[idet]->Fill(global_point.eta(), recHit.time());

    // Resolution histograms
    if (m_etlSimHits[idet].count(detId.rawId()) == 1) {
      if ((topo1Dis && m_etlSimHits[idet][detId.rawId()].energy > hitMinEnergy1Dis_) ||
          (topo2Dis && m_etlSimHits[idet][detId.rawId()].energy > hitMinEnergy2Dis_)) {
        float time_res = recHit.time() - m_etlSimHits[idet][detId.rawId()].time;
        float energy_res = recHit.energy() - m_etlSimHits[idet][detId.rawId()].energy;

        meTimeRes_->Fill(time_res);
        meEnergyRes_->Fill(energy_res);

        meTPullvsEta_->Fill(std::abs(global_point.eta()), time_res / recHit.timeError());
        meTPullvsE_->Fill(m_etlSimHits[idet][detId.rawId()].energy, time_res / recHit.timeError());
      }
    }

    n_reco_etl[idet]++;
  }  // recHit loop

  if (topo1Dis) {
    meNhits_[0]->Fill(n_reco_etl[0]);
    meNhits_[2]->Fill(n_reco_etl[2]);
  }

  if (topo2Dis) {
    for (int i = 0; i < 4; i++) {
      meNhits_[i]->Fill(n_reco_etl[i]);
    }
  }

  // --- Loop over the ETL RECO clusters ---
  for (const auto& DetSetClu : *etlRecCluHandle) {
    for (const auto& cluster : DetSetClu) {
      double weight = 1.0;
      if (topo1Dis) {
        if (cluster.energy() < hitMinEnergy1Dis_)
          continue;
      }
      if (topo2Dis) {
        if (cluster.energy() < hitMinEnergy2Dis_)
          continue;
      }
      ETLDetId cluId = cluster.id();
      DetId detIdObject(cluId);
      const auto& genericDet = geom->idToDetUnit(detIdObject);
      if (genericDet == nullptr) {
        throw cms::Exception("EtlLocalRecoValidation")
            << "GeographicalID: " << std::hex << cluId << " is invalid!" << std::dec << std::endl;
      }
      const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(genericDet->topology());
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

      Local3DPoint local_point(topo.localX(cluster.x()), topo.localY(cluster.y()), 0.);
      const auto& global_point = genericDet->toGlobal(local_point);

      int idet = 999;

      if (topo1Dis) {
        if (cluId.zside() == -1) {
          idet = 0;
        } else if (cluId.zside() == 1) {
          idet = 2;
        } else {
          continue;
        }
      }

      if (topo2Dis) {
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
          continue;
        }
      }

      meCluEnergy_[idet]->Fill(cluster.energy());
      meCluTime_[idet]->Fill(cluster.time());
      meCluTimeError_[idet]->Fill(cluster.timeError());
      meCluPhi_[idet]->Fill(global_point.phi());
      meCluEta_[idet]->Fill(global_point.eta());
      meCluOccupancy_[idet]->Fill(global_point.x(), global_point.y(), weight);
      meCluHits_[idet]->Fill(cluster.size());

      // --- Get the SIM hits associated to the cluster and calculate
      //     the cluster SIM energy, time and position

      double cluEneSIM = 0.;
      double cluTimeSIM = 0.;
      double cluLocXSIM = 0.;
      double cluLocYSIM = 0.;
      double cluLocZSIM = 0.;

      for (int ihit = 0; ihit < cluster.size(); ++ihit) {
        int hit_row = cluster.minHitRow() + cluster.hitOffset()[ihit * 2];
        int hit_col = cluster.minHitCol() + cluster.hitOffset()[ihit * 2 + 1];

        // Match the RECO hit to the corresponding SIM hit
        for (const auto& recHit : *etlRecHitsHandle) {
          ETLDetId hitId(recHit.id().rawId());

          if (m_etlSimHits[idet].count(hitId.rawId()) == 0)
            continue;

          // Check the hit position
          if (hitId.zside() != cluId.zside() || hitId.mtdRR() != cluId.mtdRR() || hitId.module() != cluId.module() ||
              recHit.row() != hit_row || recHit.column() != hit_col)
            continue;

          // Check the hit energy and time
          if (recHit.energy() != cluster.hitENERGY()[ihit] || recHit.time() != cluster.hitTIME()[ihit])
            continue;

          // SIM hit's position in the module reference frame
          Local3DPoint local_point_sim(convertMmToCm(m_etlSimHits[idet][recHit.id().rawId()].x_local),
                                       convertMmToCm(m_etlSimHits[idet][recHit.id().rawId()].y_local),
                                       convertMmToCm(m_etlSimHits[idet][recHit.id().rawId()].z_local));

          // Calculate the SIM cluster's position in the module reference frame
          cluLocXSIM += local_point_sim.x() * m_etlSimHits[idet][recHit.id().rawId()].energy;
          cluLocYSIM += local_point_sim.y() * m_etlSimHits[idet][recHit.id().rawId()].energy;
          cluLocZSIM += local_point_sim.z() * m_etlSimHits[idet][recHit.id().rawId()].energy;

          // Calculate the SIM cluster energy and time
          cluEneSIM += m_etlSimHits[idet][recHit.id().rawId()].energy;
          cluTimeSIM += m_etlSimHits[idet][recHit.id().rawId()].time * m_etlSimHits[idet][recHit.id().rawId()].energy;

          break;

        }  // recHit loop

      }  // ihit loop

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
      if (cluTimeSIM > 0. && cluEneSIM > 0.) {
        cluTimeSIM /= cluEneSIM;

        Local3DPoint cluLocalPosSIM(cluLocXSIM / cluEneSIM, cluLocYSIM / cluEneSIM, cluLocZSIM / cluEneSIM);
        const auto& cluGlobalPosSIM = genericDet->toGlobal(cluLocalPosSIM);

        float time_res = cluster.time() - cluTimeSIM;
        float energy_res = cluster.energy() - cluEneSIM;
        float x_res = global_point.x() - cluGlobalPosSIM.x();
        float y_res = global_point.y() - cluGlobalPosSIM.y();
        float z_res = global_point.z() - cluGlobalPosSIM.z();

        meCluTimeRes_[iside]->Fill(time_res);
        meCluEnergyRes_[iside]->Fill(energy_res);
        meCluXRes_[iside]->Fill(x_res);
        meCluYRes_[iside]->Fill(y_res);
        meCluZRes_[iside]->Fill(z_res);

        meCluTPullvsEta_[iside]->Fill(cluGlobalPosSIM.eta(), time_res / cluster.timeError());
        meCluTPullvsE_[iside]->Fill(cluEneSIM, time_res / cluster.timeError());

        if (optionalPlots_) {
          if (matchClu && comp != nullptr) {
            meCluXPull_[iside]->Fill(x_res / std::sqrt(comp->globalPositionError().cxx()));
            meCluYPull_[iside]->Fill(y_res / std::sqrt(comp->globalPositionError().cyy()));
            meCluXLocalErr_[iside]->Fill(std::sqrt(comp->localPositionError().xx()));
            meCluYLocalErr_[iside]->Fill(std::sqrt(comp->localPositionError().yy()));
          }
          meCluYXLocal_[iside]->Fill(local_point.x(), local_point.y());
          meCluYXLocalSim_[iside]->Fill(cluLocalPosSIM.x(), cluLocalPosSIM.y());
        }

      }  // if ( cluTimeSIM > 0. &&  cluEneSIM > 0. )
      else {
        meUnmatchedCluEnergy_[iside]->Fill(std::log10(cluster.energy()));
      }

    }  // cluster loop

  }  // DetSetClu loop

  // --- Loop over the ETL Uncalibrated RECO hits
  if (uncalibRecHitsPlots_) {
    auto etlUncalibRecHitsHandle = makeValid(iEvent.getHandle(etlUncalibRecHitsToken_));

    for (const auto& uRecHit : *etlUncalibRecHitsHandle) {
      ETLDetId detId = uRecHit.id();

      int idet = detId.zside() + detId.nDisc();

      // --- Skip UncalibratedRecHits not matched to SimHits
      if (m_etlSimHits[idet].count(detId.rawId()) != 1)
        continue;

      DetId geoId = detId.geographicalId();
      const MTDGeomDet* thedet = geom->idToDet(geoId);
      if (thedet == nullptr)
        throw cms::Exception("EtlLocalRecoValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                       << detId.rawId() << ") is invalid!" << std::dec << std::endl;

      const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

      Local3DPoint local_point(topo.localX(uRecHit.row()), topo.localY(uRecHit.column()), 0.);
      const auto& global_point = thedet->toGlobal(local_point);

      // --- Fill the histograms

      if (uRecHit.amplitude().first < hitMinAmplitude_)
        continue;

      float time_res = uRecHit.time().first - m_etlSimHits[idet][detId.rawId()].time;

      int iside = (detId.zside() == -1 ? 0 : 1);

      // amplitude histograms

      int qBin = (int)(uRecHit.amplitude().first / binWidthQ_);
      if (qBin > nBinsQ_ - 1)
        qBin = nBinsQ_ - 1;

      meTimeResQ_[iside][qBin]->Fill(time_res);

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

// ------------ method for histogram booking ------------
void EtlLocalRecoValidation::bookHistograms(DQMStore::IBooker& ibook,
                                            edm::Run const& run,
                                            edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNhits_[0] = ibook.book1D(
      "EtlNhitsZnegD1", "Number of ETL RECO hits (-Z, Single(topo1D)/First(topo2D) disk);N_{RECO}", 100, 0., 5000.);
  meNhits_[1] = ibook.book1D("EtlNhitsZnegD2", "Number of ETL RECO hits (-Z, Second disk);N_{RECO}", 100, 0., 5000.);
  meNhits_[2] = ibook.book1D(
      "EtlNhitsZposD1", "Number of ETL RECO hits (+Z, Single(topo1D)/First(topo2D) disk);N_{RECO}", 100, 0., 5000.);
  meNhits_[3] = ibook.book1D("EtlNhitsZposD2", "Number of ETL RECO hits (+Z, Second disk);N_{RECO}", 100, 0., 5000.);
  meHitEnergy_[0] = ibook.book1D(
      "EtlHitEnergyZnegD1", "ETL RECO hits energy (-Z, Single(topo1D)/First(topo2D) disk);E_{RECO} [MeV]", 100, 0., 3.);
  meHitEnergy_[1] =
      ibook.book1D("EtlHitEnergyZnegD2", "ETL RECO hits energy (-Z, Second disk);E_{RECO} [MeV]", 100, 0., 3.);
  meHitEnergy_[2] = ibook.book1D(
      "EtlHitEnergyZposD1", "ETL RECO hits energy (+Z, Single(topo1D)/First(topo2D) disk);E_{RECO} [MeV]", 100, 0., 3.);
  meHitEnergy_[3] =
      ibook.book1D("EtlHitEnergyZposD2", "ETL RECO hits energy (+Z, Second disk);E_{RECO} [MeV]", 100, 0., 3.);
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
  if (optionalPlots_) {
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
    meHitXlocal_[0] = ibook.book1D("EtlHitXlocalZneg", "ETL RECO local X (-Z);X_{RECO}^{LOC} [cm]", 100, -2.2, 2.2);
    meHitXlocal_[1] = ibook.book1D("EtlHitXlocalZpos", "ETL RECO local X (+Z);X_{RECO}^{LOC} [cm]", 100, -2.2, 2.2);
    meHitYlocal_[0] = ibook.book1D("EtlHitYlocalZneg", "ETL RECO local Y (-Z);Y_{RECO}^{LOC} [cm]", 50, -1.1, 1.1);
    meHitYlocal_[1] = ibook.book1D("EtlHitYlocalZpos", "ETL RECO local Y (-Z);Y_{RECO}^{LOC} [cm]", 50, -1.1, 1.1);
  }
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
  meEnergyRes_ = ibook.book1D("EtlEnergyRes", "ETL energy resolution;E_{RECO}-E_{SIM}", 100, -0.5, 0.5);
  meHitTvsE_[0] = ibook.bookProfile(
      "EtlHitTvsEZnegD1",
      "ETL RECO time vs energy (-Z, Single(topo1D)/First(topo2D) Disk);E_{RECO} [MeV];ToA_{RECO} [ns]",
      50,
      0.,
      2.,
      0.,
      100.);
  meHitTvsE_[1] = ibook.bookProfile("EtlHitTvsEZnegD2",
                                    "ETL RECO time vs energy (-Z, Second Disk);E_{RECO} [MeV];ToA_{RECO} [ns]",
                                    50,
                                    0.,
                                    2.,
                                    0.,
                                    100.);
  meHitTvsE_[2] = ibook.bookProfile(
      "EtlHitTvsEZposD1",
      "ETL RECO time vs energy (+Z, Single(topo1D)/First(topo2D) Disk);E_{RECO} [MeV];ToA_{RECO} [ns]",
      50,
      0.,
      2.,
      0.,
      100.);
  meHitTvsE_[3] = ibook.bookProfile("EtlHitTvsEZposD2",
                                    "ETL RECO time vs energy (+Z, Second Disk);E_{RECO} [MeV];ToA_{RECO} [ns]",
                                    50,
                                    0.,
                                    2.,
                                    0.,
                                    100.);
  meHitEvsPhi_[0] = ibook.bookProfile(
      "EtlHitEvsPhiZnegD1",
      "ETL RECO energy vs #phi (-Z, Single(topo1D)/First(topo2D) Disk);#phi_{RECO} [rad];E_{RECO} [MeV]",
      50,
      -3.2,
      3.2,
      0.,
      100.);
  meHitEvsPhi_[1] = ibook.bookProfile("EtlHitEvsPhiZnegD2",
                                      "ETL RECO energy vs #phi (-Z, Second Disk);#phi_{RECO} [rad];E_{RECO} [MeV]",
                                      50,
                                      -3.2,
                                      3.2,
                                      0.,
                                      100.);
  meHitEvsPhi_[2] = ibook.bookProfile(
      "EtlHitEvsPhiZposD1",
      "ETL RECO energy vs #phi (+Z, Single(topo1D)/First(topo2D) Disk);#phi_{RECO} [rad];E_{RECO} [MeV]",
      50,
      -3.2,
      3.2,
      0.,
      100.);
  meHitEvsPhi_[3] = ibook.bookProfile("EtlHitEvsPhiZposD2",
                                      "ETL RECO energy vs #phi (+Z, Second Disk);#phi_{RECO} [rad];E_{RECO} [MeV]",
                                      50,
                                      -3.2,
                                      3.2,
                                      0.,
                                      100.);
  meHitEvsEta_[0] =
      ibook.bookProfile("EtlHitEvsEtaZnegD1",
                        "ETL RECO energy vs #eta (-Z, Single(topo1D)/First(topo2D) Disk);#eta_{RECO};E_{RECO} [MeV]",
                        50,
                        -3.2,
                        -1.56,
                        0.,
                        100.);
  meHitEvsEta_[1] = ibook.bookProfile("EtlHitEvsEtaZnegD2",
                                      "ETL RECO energy vs #eta (-Z, Second Disk);#eta_{RECO};E_{RECO} [MeV]",
                                      50,
                                      -3.2,
                                      -1.56,
                                      0.,
                                      100.);
  meHitEvsEta_[2] =
      ibook.bookProfile("EtlHitEvsEtaZposD1",
                        "ETL RECO energy vs #eta (+Z, Single(topo1D)/First(topo2D) Disk);#eta_{RECO};E_{RECO} [MeV]",
                        50,
                        1.56,
                        3.2,
                        0.,
                        100.);
  meHitEvsEta_[3] = ibook.bookProfile("EtlHitEvsEtaZposD2",
                                      "ETL RECO energy vs #eta (+Z, Second Disk);#eta_{RECO};E_{RECO} [MeV]",
                                      50,
                                      1.56,
                                      3.2,
                                      0.,
                                      100.);
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
  meCluEnergy_[0] = ibook.book1D(
      "EtlCluEnergyZnegD1", "ETL cluster energy (-Z, Single(topo1D)/First(topo2D) Disk);E_{RECO} [MeV]", 100, 0, 10);
  meCluEnergy_[1] =
      ibook.book1D("EtlCluEnergyZnegD2", "ETL cluster energy (-Z, Second Disk);E_{RECO} [MeV]", 100, 0, 10);
  meCluEnergy_[2] = ibook.book1D(
      "EtlCluEnergyZposD1", "ETL cluster energy (+Z, Single(topo1D)/First(topo2D) Disk);E_{RECO} [MeV]", 100, 0, 10);
  meCluEnergy_[3] =
      ibook.book1D("EtlCluEnergyZposD2", "ETL cluster energy (+Z, Second Disk);E_{RECO} [MeV]", 100, 0, 10);
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
      "EtlCluHitNumberZnegD1", "ETL hits per cluster (-Z, Single(topo1D)/First(topo2D) Disk);Cluster size", 10, 0, 10);
  meCluHits_[1] =
      ibook.book1D("EtlCluHitNumberZnegD2", "ETL hits per cluster (-Z, Second Disk);Cluster size", 10, 0, 10);
  meCluHits_[2] = ibook.book1D(
      "EtlCluHitNumberZposD1", "ETL hits per cluster (+Z, Single(topo1D)/First(topo2D) Disk);Cluster size", 10, 0, 10);
  meCluHits_[3] =
      ibook.book1D("EtlCluHitNumberZposD2", "ETL hits per cluster (+Z, Second Disk);Cluster size", 10, 0, 10);
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

  meCluTimeRes_[0] =
      ibook.book1D("EtlCluTimeResZneg", "ETL cluster time resolution (-Z);T_{RECO}-T_{SIM} [ns]", 100, -0.5, 0.5);
  meCluTimeRes_[1] =
      ibook.book1D("EtlCluTimeResZpos", "ETL cluster time resolution (+Z);T_{RECO}-T_{SIM} [MeV]", 100, -0.5, 0.5);
  meCluEnergyRes_[0] =
      ibook.book1D("EtlCluEnergyResZneg", "ETL cluster energy resolution (-Z);E_{RECO}-E_{SIM}", 100, -0.5, 0.5);
  meCluEnergyRes_[1] =
      ibook.book1D("EtlCluEnergyResZpos", "ETL cluster energy resolution (+Z);E_{RECO}-E_{SIM}", 100, -0.5, 0.5);

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
  meCluXRes_[0] = ibook.book1D("EtlCluXResZneg", "ETL cluster X resolution (-Z);X_{RECO}-X_{SIM} [cm]", 100, -0.1, 0.1);
  meCluXRes_[1] = ibook.book1D("EtlCluXResZpos", "ETL cluster X resolution (+Z);X_{RECO}-X_{SIM} [cm]", 100, -0.1, 0.1);
  meCluYRes_[0] = ibook.book1D("EtlCluYResZneg", "ETL cluster Y resolution (-Z);Y_{RECO}-Y_{SIM} [cm]", 100, -0.1, 0.1);
  meCluYRes_[1] = ibook.book1D("EtlCluYResZpos", "ETL cluster Y resolution (+Z);Y_{RECO}-Y_{SIM} [cm]", 100, -0.1, 0.1);
  meCluZRes_[0] =
      ibook.book1D("EtlCluZResZneg", "ETL cluster Z resolution (-Z);Z_{RECO}-Z_{SIM} [cm]", 100, -0.003, 0.003);
  meCluZRes_[1] =
      ibook.book1D("EtlCluZResZpos", "ETL cluster Z resolution (+Z);Z_{RECO}-Z_{SIM} [cm]", 100, -0.003, 0.003);
  if (optionalPlots_) {
    meCluXPull_[0] =
        ibook.book1D("EtlCluXPullZneg", "ETL cluster X pull (-Z);X_{RECO}-X_{SIM}/sigmaX_[RECO] [cm]", 100, -5., 5.);
    meCluXPull_[1] =
        ibook.book1D("EtlCluXPullZpos", "ETL cluster X pull (+Z);X_{RECO}-X_{SIM}/sigmaX_[RECO] [cm]", 100, -5., 5.);
    meCluYPull_[0] =
        ibook.book1D("EtlCluYPullZneg", "ETL cluster Y pull (-Z);Y_{RECO}-Y_{SIM}/sigmaY_[RECO] [cm]", 100, -5., 5.);
    meCluYPull_[1] =
        ibook.book1D("EtlCluYPullZpos", "ETL cluster Y pull (+Z);Y_{RECO}-Y_{SIM}/sigmaY_[RECO] [cm]", 100, -5., 5.);
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
    meCluXLocalErr_[0] =
        ibook.book1D("EtlCluXLocalErrNeg", "ETL cluster X local error (-Z);sigmaX_{RECO,loc} [cm]", 50, 0., 0.2);
    meCluXLocalErr_[1] =
        ibook.book1D("EtlCluXLocalErrPos", "ETL cluster X local error (+Z);sigmaX_{RECO,loc} [cm]", 50, 0., 0.2);
    meCluYLocalErr_[0] =
        ibook.book1D("EtlCluYLocalErrNeg", "ETL cluster Y local error (-Z);sigmaY_{RECO,loc} [cm]", 50., 0., 0.2);
    meCluYLocalErr_[1] =
        ibook.book1D("EtlCluYLocalErrPos", "ETL cluster Y local error (+Z);sigmaY_{RECO,loc} [cm]", 50, 0., 0.2);
  }
  meUnmatchedCluEnergy_[0] = ibook.book1D(
      "EtlUnmatchedCluEnergyNeg", "ETL unmatched cluster log10(energy) (-Z);log10(E_{RECO} [MeV])", 5, -3, 2);
  meUnmatchedCluEnergy_[1] = ibook.book1D(
      "EtlUnmatchedCluEnergyPos", "ETL unmatched cluster log10(energy) (+Z);log10(E_{RECO} [MeV])", 5, -3, 2);

  // --- UncalibratedRecHits histograms

  if (uncalibRecHitsPlots_) {
    const std::string det_name[2] = {"ETL-", "ETL+"};
    for (unsigned int iside = 0; iside < 2; ++iside) {
      for (unsigned int ihistoQ = 0; ihistoQ < nBinsQ_; ++ihistoQ) {
        std::string hname = Form("TimeResQ_%d_%d", iside, ihistoQ);
        std::string htitle =
            Form("%s time resolution (Q bin = %d);T_{RECO} - T_{SIM} [ns]", det_name[iside].data(), ihistoQ);
        meTimeResQ_[iside][ihistoQ] = ibook.book1D(hname, htitle, 200, -0.5, 0.5);

      }  // ihistoQ loop

      for (unsigned int ihistoEta = 0; ihistoEta < nBinsEta_; ++ihistoEta) {
        std::string hname = Form("TimeResEta_%d_%d", iside, ihistoEta);
        std::string htitle =
            Form("%s time resolution (|#eta| bin = %d);T_{RECO} - T_{SIM} [ns]", det_name[iside].data(), ihistoEta);
        meTimeResEta_[iside][ihistoEta] = ibook.book1D(hname, htitle, 200, -0.5, 0.5);

      }  // ihistoEta loop
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
  desc.add<double>("hitMinimumEnergy1Dis", 1.);     // [MeV]
  desc.add<double>("hitMinimumEnergy2Dis", 0.001);  // [MeV]
  desc.add<bool>("optionalPlots", false);
  desc.add<bool>("UncalibRecHitsPlots", false);
  desc.add<double>("HitMinimumAmplitude", 0.33);  // [MIP]

  descriptions.add("etlLocalRecoValid", desc);
}

DEFINE_FWK_MODULE(EtlLocalRecoValidation);
