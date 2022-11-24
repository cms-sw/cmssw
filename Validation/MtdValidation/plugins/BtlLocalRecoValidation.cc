// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlLocalRecoValidation
//
/**\class BtlLocalRecoValidation BtlLocalRecoValidation.cc Validation/MtdValidation/plugins/BtlLocalRecoValidation.cc

 Description: BTL RECO hits and clusters validation

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
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include "RecoLocalFastTime/Records/interface/MTDCPERecord.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDClusterParameterEstimator.h"

#include "MTDHit.h"

class BtlLocalRecoValidation : public DQMEDAnalyzer {
public:
  explicit BtlLocalRecoValidation(const edm::ParameterSet&);
  ~BtlLocalRecoValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  bool isSameCluster(const FTLCluster&, const FTLCluster&);

  // ------------ member data ------------

  const std::string folder_;
  const double hitMinEnergy_;
  const bool optionalPlots_;
  const bool uncalibRecHitsPlots_;
  const double hitMinAmplitude_;

  edm::EDGetTokenT<FTLRecHitCollection> btlRecHitsToken_;
  edm::EDGetTokenT<FTLUncalibratedRecHitCollection> btlUncalibRecHitsToken_;
  edm::EDGetTokenT<CrossingFrame<PSimHit> > btlSimHitsToken_;
  edm::EDGetTokenT<FTLClusterCollection> btlRecCluToken_;
  edm::EDGetTokenT<MTDTrackingDetSetVector> mtdTrackingHitToken_;

  const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  const edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
  const edm::ESGetToken<MTDClusterParameterEstimator, MTDCPERecord> cpeToken_;

  // --- histograms declaration

  MonitorElement* meNevents_;

  MonitorElement* meNhits_;

  MonitorElement* meHitEnergy_;
  MonitorElement* meHitLogEnergy_;
  MonitorElement* meHitTime_;
  MonitorElement* meHitTimeError_;

  MonitorElement* meOccupancy_;

  //local position monitoring
  MonitorElement* meLocalOccupancy_;
  MonitorElement* meHitXlocal_;
  MonitorElement* meHitYlocal_;
  MonitorElement* meHitZlocal_;

  MonitorElement* meHitZ_;
  MonitorElement* meHitPhi_;
  MonitorElement* meHitEta_;

  MonitorElement* meHitTvsE_;
  MonitorElement* meHitEvsPhi_;
  MonitorElement* meHitEvsEta_;
  MonitorElement* meHitEvsZ_;
  MonitorElement* meHitTvsPhi_;
  MonitorElement* meHitTvsEta_;
  MonitorElement* meHitTvsZ_;
  MonitorElement* meHitLongPos_;
  MonitorElement* meHitLongPosErr_;

  MonitorElement* meTimeRes_;
  MonitorElement* meEnergyRes_;

  MonitorElement* meLongPosPull_;
  MonitorElement* meLongPosPullvsE_;
  MonitorElement* meLongPosPullvsEta_;

  MonitorElement* meTPullvsE_;
  MonitorElement* meTPullvsEta_;

  MonitorElement* meNclusters_;

  MonitorElement* meCluTime_;
  MonitorElement* meCluTimeError_;
  MonitorElement* meCluEnergy_;
  MonitorElement* meCluPhi_;
  MonitorElement* meCluEta_;
  MonitorElement* meCluHits_;
  MonitorElement* meCluZvsPhi_;
  MonitorElement* meCluEnergyvsEta_;
  MonitorElement* meCluHitsvsEta_;

  MonitorElement* meCluTimeRes_;
  MonitorElement* meCluEnergyRes_;
  MonitorElement* meCluTResvsE_;
  MonitorElement* meCluTResvsEta_;
  MonitorElement* meCluTPullvsE_;
  MonitorElement* meCluTPullvsEta_;
  MonitorElement* meCluRhoRes_;
  MonitorElement* meCluPhiRes_;
  MonitorElement* meCluLocalXRes_;
  MonitorElement* meCluLocalYRes_;
  MonitorElement* meCluZRes_;
  MonitorElement* meCluLocalXPull_;
  MonitorElement* meCluLocalYPull_;
  MonitorElement* meCluZPull_;
  MonitorElement* meCluYXLocal_;
  MonitorElement* meCluYXLocalSim_;
  MonitorElement* meCluXLocalErr_;
  MonitorElement* meCluYLocalErr_;

  MonitorElement* meUnmatchedCluEnergy_;

  // --- UncalibratedRecHits histograms

  static constexpr int nBinsQ_ = 20;
  static constexpr float binWidthQ_ = 30.;
  static constexpr int nBinsQEta_ = 3;
  static constexpr float binsQEta_[nBinsQEta_ + 1] = {0., 0.65, 1.15, 1.55};

  MonitorElement* meTimeResQ_[nBinsQ_];
  MonitorElement* meTimeResQvsEta_[nBinsQ_][nBinsQEta_];

  static constexpr int nBinsEta_ = 31;
  static constexpr float binWidthEta_ = 0.05;
  static constexpr int nBinsEtaQ_ = 7;
  static constexpr float binsEtaQ_[nBinsEtaQ_ + 1] = {0., 30., 60., 90., 120., 150., 360., 600.};

  MonitorElement* meTimeResEta_[nBinsEta_];
  MonitorElement* meTimeResEtavsQ_[nBinsEta_][nBinsEtaQ_];
};

bool BtlLocalRecoValidation::isSameCluster(const FTLCluster& clu1, const FTLCluster& clu2) {
  return clu1.id() == clu2.id() && clu1.size() == clu2.size() && clu1.x() == clu2.x() && clu1.y() == clu2.y() &&
         clu1.time() == clu2.time();
}

// ------------ constructor and destructor --------------
BtlLocalRecoValidation::BtlLocalRecoValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy_(iConfig.getParameter<double>("HitMinimumEnergy")),
      optionalPlots_(iConfig.getParameter<bool>("optionalPlots")),
      uncalibRecHitsPlots_(iConfig.getParameter<bool>("UncalibRecHitsPlots")),
      hitMinAmplitude_(iConfig.getParameter<double>("HitMinimumAmplitude")),
      mtdgeoToken_(esConsumes<MTDGeometry, MTDDigiGeometryRecord>()),
      mtdtopoToken_(esConsumes<MTDTopology, MTDTopologyRcd>()),
      cpeToken_(esConsumes<MTDClusterParameterEstimator, MTDCPERecord>(edm::ESInputTag("", "MTDCPEBase"))) {
  btlRecHitsToken_ = consumes<FTLRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsTag"));
  if (uncalibRecHitsPlots_)
    btlUncalibRecHitsToken_ =
        consumes<FTLUncalibratedRecHitCollection>(iConfig.getParameter<edm::InputTag>("uncalibRecHitsTag"));
  btlSimHitsToken_ = consumes<CrossingFrame<PSimHit> >(iConfig.getParameter<edm::InputTag>("simHitsTag"));
  btlRecCluToken_ = consumes<FTLClusterCollection>(iConfig.getParameter<edm::InputTag>("recCluTag"));
  mtdTrackingHitToken_ = consumes<MTDTrackingDetSetVector>(iConfig.getParameter<edm::InputTag>("trkHitTag"));
}

BtlLocalRecoValidation::~BtlLocalRecoValidation() {}

// ------------ method called for each event  ------------
void BtlLocalRecoValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace geant_units::operators;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  auto const& cpe = iSetup.getData(cpeToken_);

  auto btlRecHitsHandle = makeValid(iEvent.getHandle(btlRecHitsToken_));
  auto btlSimHitsHandle = makeValid(iEvent.getHandle(btlSimHitsToken_));
  auto btlRecCluHandle = makeValid(iEvent.getHandle(btlRecCluToken_));
  auto mtdTrkHitHandle = makeValid(iEvent.getHandle(mtdTrackingHitToken_));
  MixCollection<PSimHit> btlSimHits(btlSimHitsHandle.product());

#ifdef EDM_ML_DEBUG
  for (const auto& hits : *mtdTrkHitHandle) {
    if (MTDDetId(hits.id()).mtdSubDetector() == MTDDetId::MTDType::BTL) {
      LogDebug("BtlLocalRecoValidation") << "MTD cluster DetId " << hits.id() << " # cluster " << hits.size();
      for (const auto& hit : hits) {
        LogDebug("BtlLocalRecoValidation")
            << "MTD_TRH: " << hit.localPosition().x() << "," << hit.localPosition().y() << " : "
            << hit.localPositionError().xx() << "," << hit.localPositionError().yy() << " : " << hit.time() << " : "
            << hit.timeError();
      }
    }
  }
#endif

  // --- Loop over the BTL SIM hits
  std::unordered_map<uint32_t, MTDHit> m_btlSimHits;
  for (auto const& simHit : btlSimHits) {
    // --- Use only hits compatible with the in-time bunch-crossing
    if (simHit.tof() < 0 || simHit.tof() > 25.)
      continue;

    DetId id = simHit.detUnitId();

    auto simHitIt = m_btlSimHits.emplace(id.rawId(), MTDHit()).first;

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

  // --- Loop over the BTL RECO hits
  unsigned int n_reco_btl = 0;
  for (const auto& recHit : *btlRecHitsHandle) {
    BTLDetId detId = recHit.id();
    DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("BtlLocalRecoValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                     << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    Local3DPoint local_point(0., 0., 0.);
    local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
    const auto& global_point = thedet->toGlobal(local_point);

    meHitEnergy_->Fill(recHit.energy());
    meHitLogEnergy_->Fill(log10(recHit.energy()));
    meHitTime_->Fill(recHit.time());
    meHitTimeError_->Fill(recHit.timeError());
    meHitLongPos_->Fill(recHit.position());
    meHitLongPosErr_->Fill(recHit.positionError());

    meOccupancy_->Fill(global_point.z(), global_point.phi());

    if (optionalPlots_) {
      meLocalOccupancy_->Fill(local_point.x() + recHit.position(), local_point.y());
    }
    meHitXlocal_->Fill(local_point.x());
    meHitYlocal_->Fill(local_point.y());
    meHitZlocal_->Fill(local_point.z());
    meHitZ_->Fill(global_point.z());
    meHitPhi_->Fill(global_point.phi());
    meHitEta_->Fill(global_point.eta());

    meHitTvsE_->Fill(recHit.energy(), recHit.time());
    meHitEvsPhi_->Fill(global_point.phi(), recHit.energy());
    meHitEvsEta_->Fill(global_point.eta(), recHit.energy());
    meHitEvsZ_->Fill(global_point.z(), recHit.energy());
    meHitTvsPhi_->Fill(global_point.phi(), recHit.time());
    meHitTvsEta_->Fill(global_point.eta(), recHit.time());
    meHitTvsZ_->Fill(global_point.z(), recHit.time());

    // Resolution histograms
    if (m_btlSimHits.count(detId.rawId()) == 1 && m_btlSimHits[detId.rawId()].energy > hitMinEnergy_) {
      float longpos_res = recHit.position() - convertMmToCm(m_btlSimHits[detId.rawId()].x);
      float time_res = recHit.time() - m_btlSimHits[detId.rawId()].time;
      float energy_res = recHit.energy() - m_btlSimHits[detId.rawId()].energy;

      Local3DPoint local_point_sim(convertMmToCm(m_btlSimHits[detId.rawId()].x),
                                   convertMmToCm(m_btlSimHits[detId.rawId()].y),
                                   convertMmToCm(m_btlSimHits[detId.rawId()].z));
      local_point_sim =
          topo.pixelToModuleLocalPoint(local_point_sim, detId.row(topo.nrows()), detId.column(topo.nrows()));
      const auto& global_point_sim = thedet->toGlobal(local_point_sim);

      meTimeRes_->Fill(time_res);
      meEnergyRes_->Fill(energy_res);

      meLongPosPull_->Fill(longpos_res / recHit.positionError());
      meLongPosPullvsEta_->Fill(std::abs(global_point_sim.eta()), longpos_res / recHit.positionError());
      meLongPosPullvsE_->Fill(m_btlSimHits[detId.rawId()].energy, longpos_res / recHit.positionError());

      meTPullvsEta_->Fill(std::abs(global_point_sim.eta()), time_res / recHit.timeError());
      meTPullvsE_->Fill(m_btlSimHits[detId.rawId()].energy, time_res / recHit.timeError());
    }

    n_reco_btl++;

  }  // recHit loop

  if (n_reco_btl > 0)
    meNhits_->Fill(log10(n_reco_btl));

  // --- Loop over the BTL RECO clusters ---
  unsigned int n_clus_btl(0);
  for (const auto& DetSetClu : *btlRecCluHandle) {
    for (const auto& cluster : DetSetClu) {
      if (cluster.energy() < hitMinEnergy_)
        continue;
      BTLDetId cluId = cluster.id();
      DetId detIdObject(cluId);
      const auto& genericDet = geom->idToDetUnit(detIdObject);
      if (genericDet == nullptr) {
        throw cms::Exception("BtlLocalRecoValidation")
            << "GeographicalID: " << std::hex << cluId << " is invalid!" << std::dec << std::endl;
      }
      n_clus_btl++;

      const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(genericDet->topology());
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

      MTDClusterParameterEstimator::ReturnType tuple = cpe.getParameters(cluster, *genericDet);

      // --- Cluster position in the module reference frame
      LocalPoint local_point(std::get<0>(tuple));
      const auto& global_point = genericDet->toGlobal(local_point);

      meCluEnergy_->Fill(cluster.energy());
      meCluTime_->Fill(cluster.time());
      meCluTimeError_->Fill(cluster.timeError());
      meCluPhi_->Fill(global_point.phi());
      meCluEta_->Fill(global_point.eta());
      meCluZvsPhi_->Fill(global_point.z(), global_point.phi());
      meCluHits_->Fill(cluster.size());

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
        for (const auto& recHit : *btlRecHitsHandle) {
          BTLDetId hitId(recHit.id().rawId());

          if (m_btlSimHits.count(hitId.rawId()) == 0)
            continue;

          // Check the hit position
          if (hitId.mtdSide() != cluId.mtdSide() || hitId.mtdRR() != cluId.mtdRR() || recHit.row() != hit_row ||
              recHit.column() != hit_col)
            continue;

          // Check the hit energy and time
          if (recHit.energy() != cluster.hitENERGY()[ihit] || recHit.time() != cluster.hitTIME()[ihit])
            continue;

          // SIM hit's position in the module reference frame
          Local3DPoint local_point_sim(convertMmToCm(m_btlSimHits[recHit.id().rawId()].x),
                                       convertMmToCm(m_btlSimHits[recHit.id().rawId()].y),
                                       convertMmToCm(m_btlSimHits[recHit.id().rawId()].z));
          local_point_sim =
              topo.pixelToModuleLocalPoint(local_point_sim, hitId.row(topo.nrows()), hitId.column(topo.nrows()));

          // Calculate the SIM cluster's position in the module reference frame
          cluLocXSIM += local_point_sim.x() * m_btlSimHits[recHit.id().rawId()].energy;
          cluLocYSIM += local_point_sim.y() * m_btlSimHits[recHit.id().rawId()].energy;
          cluLocZSIM += local_point_sim.z() * m_btlSimHits[recHit.id().rawId()].energy;

          // Calculate the SIM cluster energy and time
          cluEneSIM += m_btlSimHits[recHit.id().rawId()].energy;
          cluTimeSIM += m_btlSimHits[recHit.id().rawId()].time * m_btlSimHits[recHit.id().rawId()].energy;

          break;

        }  // recHit loop

      }  // ihit loop

      // Find the MTDTrackingRecHit corresponding to the cluster
      const MTDTrackingRecHit* comp(nullptr);
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
      if (cluTimeSIM > 0. && cluEneSIM > 0.) {
        cluTimeSIM /= cluEneSIM;

        Local3DPoint cluLocalPosSIM(cluLocXSIM / cluEneSIM, cluLocYSIM / cluEneSIM, cluLocZSIM / cluEneSIM);
        const auto& cluGlobalPosSIM = genericDet->toGlobal(cluLocalPosSIM);

        float time_res = cluster.time() - cluTimeSIM;
        float energy_res = cluster.energy() - cluEneSIM;
        meCluTimeRes_->Fill(time_res);
        meCluEnergyRes_->Fill(energy_res);

        float rho_res = global_point.perp() - cluGlobalPosSIM.perp();
        float phi_res = global_point.phi() - cluGlobalPosSIM.phi();

        meCluRhoRes_->Fill(rho_res);
        meCluPhiRes_->Fill(phi_res);

        float xlocal_res = local_point.x() - cluLocalPosSIM.x();
        float ylocal_res = local_point.y() - cluLocalPosSIM.y();
        float z_res = global_point.z() - cluGlobalPosSIM.z();

        meCluZRes_->Fill(z_res);

        if (matchClu && comp != nullptr) {
          meCluLocalXRes_->Fill(xlocal_res);
          meCluLocalYRes_->Fill(ylocal_res);
          meCluLocalXPull_->Fill(xlocal_res / std::sqrt(comp->localPositionError().xx()));
          meCluLocalYPull_->Fill(ylocal_res / std::sqrt(comp->localPositionError().yy()));
          meCluZPull_->Fill(z_res / std::sqrt(comp->globalPositionError().czz()));
          meCluXLocalErr_->Fill(std::sqrt(comp->localPositionError().xx()));
          meCluYLocalErr_->Fill(std::sqrt(comp->localPositionError().yy()));
        }

        if (optionalPlots_) {
          meCluYXLocal_->Fill(local_point.x(), local_point.y());
          meCluYXLocalSim_->Fill(cluLocalPosSIM.x(), cluLocalPosSIM.y());
        }

        meCluEnergyvsEta_->Fill(std::abs(cluGlobalPosSIM.eta()), cluster.energy());
        meCluHitsvsEta_->Fill(std::abs(cluGlobalPosSIM.eta()), cluster.size());

        meCluTResvsEta_->Fill(std::abs(cluGlobalPosSIM.eta()), time_res);
        meCluTResvsE_->Fill(cluEneSIM, time_res);

        meCluTPullvsEta_->Fill(std::abs(cluGlobalPosSIM.eta()), time_res / cluster.timeError());
        meCluTPullvsE_->Fill(cluEneSIM, time_res / cluster.timeError());

      }  // if ( cluTimeSIM > 0. &&  cluEneSIM > 0. )
      else {
        meUnmatchedCluEnergy_->Fill(std::log10(cluster.energy()));
      }

    }  // cluster loop

  }  // DetSetClu loop

  if (n_clus_btl > 0)
    meNclusters_->Fill(log10(n_clus_btl));

  // --- This is to count the number of processed events, needed in the harvesting step
  meNevents_->Fill(0.5);

  // --- Loop over the BTL Uncalibrated RECO hits
  if (uncalibRecHitsPlots_) {
    auto btlUncalibRecHitsHandle = makeValid(iEvent.getHandle(btlUncalibRecHitsToken_));

    for (const auto& uRecHit : *btlUncalibRecHitsHandle) {
      BTLDetId detId = uRecHit.id();

      // --- Skip UncalibratedRecHits not matched to SimHits
      if (m_btlSimHits.count(detId.rawId()) != 1)
        continue;

      DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
      const MTDGeomDet* thedet = geom->idToDet(geoId);
      if (thedet == nullptr)
        throw cms::Exception("BtlLocalRecoValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                       << detId.rawId() << ") is invalid!" << std::dec << std::endl;
      const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

      Local3DPoint local_point(0., 0., 0.);
      local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
      const auto& global_point = thedet->toGlobal(local_point);

      // --- Combine the information from the left and right BTL cell sides

      float nHits = 0.;
      float hit_amplitude = 0.;
      float hit_time = 0.;

      // left side:
      if (uRecHit.amplitude().first > 0.) {
        hit_amplitude += uRecHit.amplitude().first;
        hit_time += uRecHit.time().first;
        nHits += 1.;
      }
      // right side:
      if (uRecHit.amplitude().second > 0.) {
        hit_amplitude += uRecHit.amplitude().second;
        hit_time += uRecHit.time().second;
        nHits += 1.;
      }

      hit_amplitude /= nHits;
      hit_time /= nHits;

      // --- Fill the histograms

      if (hit_amplitude < hitMinAmplitude_)
        continue;

      float time_res = hit_time - m_btlSimHits[detId.rawId()].time;

      // amplitude histograms

      int qBin = (int)(hit_amplitude / binWidthQ_);
      if (qBin > nBinsQ_ - 1)
        qBin = nBinsQ_ - 1;

      meTimeResQ_[qBin]->Fill(time_res);

      int etaBin = 0;
      for (int ibin = 1; ibin < nBinsQEta_; ++ibin)
        if (fabs(global_point.eta()) >= binsQEta_[ibin] && fabs(global_point.eta()) < binsQEta_[ibin + 1])
          etaBin = ibin;

      meTimeResQvsEta_[qBin][etaBin]->Fill(time_res);

      // eta histograms

      etaBin = (int)(fabs(global_point.eta()) / binWidthEta_);
      if (etaBin > nBinsEta_ - 1)
        etaBin = nBinsEta_ - 1;

      meTimeResEta_[etaBin]->Fill(time_res);

      qBin = 0;
      for (int ibin = 1; ibin < nBinsEtaQ_; ++ibin)
        if (hit_amplitude >= binsEtaQ_[ibin] && hit_amplitude < binsEtaQ_[ibin + 1])
          qBin = ibin;

      meTimeResEtavsQ_[etaBin][qBin]->Fill(time_res);

    }  // uRecHit loop
  }
}

// ------------ method for histogram booking ------------
void BtlLocalRecoValidation::bookHistograms(DQMStore::IBooker& ibook,
                                            edm::Run const& run,
                                            edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNevents_ = ibook.book1D("BtlNevents", "Number of events", 1, 0., 1.);

  meNhits_ = ibook.book1D("BtlNhits", "Number of BTL RECO hits;log_{10}(N_{RECO})", 100, 0., 5.25);

  meHitEnergy_ = ibook.book1D("BtlHitEnergy", "BTL RECO hits energy;E_{RECO} [MeV]", 100, 0., 20.);
  meHitLogEnergy_ = ibook.book1D("BtlHitLogEnergy", "BTL RECO hits energy;log_{10}(E_{RECO} [MeV])", 16, -0.1, 1.5);
  meHitTime_ = ibook.book1D("BtlHitTime", "BTL RECO hits ToA;ToA_{RECO} [ns]", 100, 0., 25.);
  meHitTimeError_ = ibook.book1D("BtlHitTimeError", "BTL RECO hits ToA error;#sigma^{ToA}_{RECO} [ns]", 50, 0., 0.1);
  meOccupancy_ = ibook.book2D(
      "BtlOccupancy", "BTL RECO hits occupancy;Z_{RECO} [cm]; #phi_{RECO} [rad]", 65, -260., 260., 126, -3.2, 3.2);
  if (optionalPlots_) {
    meLocalOccupancy_ = ibook.book2D(
        "BtlLocalOccupancy", "BTL RECO hits local occupancy;X_{RECO} [cm]; Y_{RECO} [cm]", 100, 10., 10., 60, -3., 3.);
  }
  meHitXlocal_ = ibook.book1D("BtlHitXlocal", "BTL RECO local X;X_{RECO}^{LOC} [cm]", 100, -10., 10.);
  meHitYlocal_ = ibook.book1D("BtlHitYlocal", "BTL RECO local Y;Y_{RECO}^{LOC} [cm]", 60, -3, 3);
  meHitZlocal_ = ibook.book1D("BtlHitZlocal", "BTL RECO local z;z_{RECO}^{LOC} [cm]", 8, -0.4, 0.4);
  meHitZ_ = ibook.book1D("BtlHitZ", "BTL RECO hits Z;Z_{RECO} [cm]", 100, -260., 260.);
  meHitPhi_ = ibook.book1D("BtlHitPhi", "BTL RECO hits #phi;#phi_{RECO} [rad]", 126, -3.2, 3.2);
  meHitEta_ = ibook.book1D("BtlHitEta", "BTL RECO hits #eta;#eta_{RECO}", 100, -1.55, 1.55);
  meHitTvsE_ =
      ibook.bookProfile("BtlHitTvsE", "BTL RECO ToA vs energy;E_{RECO} [MeV];ToA_{RECO} [ns]", 50, 0., 20., 0., 100.);
  meHitEvsPhi_ = ibook.bookProfile(
      "BtlHitEvsPhi", "BTL RECO energy vs #phi;#phi_{RECO} [rad];E_{RECO} [MeV]", 50, -3.2, 3.2, 0., 100.);
  meHitEvsEta_ = ibook.bookProfile(
      "BtlHitEvsEta", "BTL RECO energy vs #eta;#eta_{RECO};E_{RECO} [MeV]", 50, -1.55, 1.55, 0., 100.);
  meHitEvsZ_ =
      ibook.bookProfile("BtlHitEvsZ", "BTL RECO energy vs Z;Z_{RECO} [cm];E_{RECO} [MeV]", 50, -260., 260., 0., 100.);
  meHitTvsPhi_ = ibook.bookProfile(
      "BtlHitTvsPhi", "BTL RECO ToA vs #phi;#phi_{RECO} [rad];ToA_{RECO} [ns]", 50, -3.2, 3.2, 0., 100.);
  meHitTvsEta_ =
      ibook.bookProfile("BtlHitTvsEta", "BTL RECO ToA vs #eta;#eta_{RECO};ToA_{RECO} [ns]", 50, -1.6, 1.6, 0., 100.);
  meHitTvsZ_ =
      ibook.bookProfile("BtlHitTvsZ", "BTL RECO ToA vs Z;Z_{RECO} [cm];ToA_{RECO} [ns]", 50, -260., 260., 0., 100.);
  meHitLongPos_ = ibook.book1D("BtlLongPos", "BTL RECO hits longitudinal position;long. pos._{RECO}", 100, -10, 10);
  meHitLongPosErr_ =
      ibook.book1D("BtlLongPosErr", "BTL RECO hits longitudinal position error; long. pos. error_{RECO}", 100, -1, 1);
  meTimeRes_ = ibook.book1D("BtlTimeRes", "BTL time resolution;T_{RECO}-T_{SIM}", 100, -0.5, 0.5);
  meEnergyRes_ = ibook.book1D("BtlEnergyRes", "BTL energy resolution;E_{RECO}-E_{SIM}", 100, -0.5, 0.5);
  meLongPosPull_ = ibook.book1D("BtlLongPosPull",
                                "BTL longitudinal position pull;X^{loc}_{RECO}-X^{loc}_{SIM}/#sigma_{xloc_{RECO}}",
                                100,
                                -5.,
                                5.);
  meLongPosPullvsE_ = ibook.bookProfile(
      "BtlLongposPullvsE",
      "BTL longitudinal position pull vs E;E_{SIM} [MeV];X^{loc}_{RECO}-X^{loc}_{SIM}/#sigma_{xloc_{RECO}}",
      20,
      0.,
      20.,
      -5.,
      5.,
      "S");
  meLongPosPullvsEta_ = ibook.bookProfile(
      "BtlLongposPullvsEta",
      "BTL longitudinal position pull vs #eta;|#eta_{RECO}|;X^{loc}_{RECO}-X^{loc}_{SIM}/#sigma_{xloc_{RECO}}",
      32,
      0,
      1.55,
      -5.,
      5.,
      "S");
  meTPullvsE_ = ibook.bookProfile(
      "BtlTPullvsE", "BTL time pull vs E;E_{SIM} [MeV];(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}", 20, 0., 20., -5., 5., "S");
  meTPullvsEta_ = ibook.bookProfile("BtlTPullvsEta",
                                    "BTL time pull vs #eta;|#eta_{RECO}|;(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
                                    30,
                                    0,
                                    1.55,
                                    -5.,
                                    5.,
                                    "S");

  meNclusters_ = ibook.book1D("BtlNclusters", "Number of BTL RECO clusters;log_{10}(N_{RECO})", 100, 0., 5.25);
  meCluTime_ = ibook.book1D("BtlCluTime", "BTL cluster time ToA;ToA [ns]", 250, 0, 25);
  meCluTimeError_ = ibook.book1D("BtlCluTimeError", "BTL cluster time error;#sigma_{t} [ns]", 100, 0, 0.1);
  meCluEnergy_ = ibook.book1D("BtlCluEnergy", "BTL cluster energy;E_{RECO} [MeV]", 100, 0, 20);
  meCluPhi_ = ibook.book1D("BtlCluPhi", "BTL cluster #phi;#phi_{RECO} [rad]", 144, -3.2, 3.2);
  meCluEta_ = ibook.book1D("BtlCluEta", "BTL cluster #eta;#eta_{RECO}", 100, -1.55, 1.55);
  meCluHits_ = ibook.book1D("BtlCluHitNumber", "BTL hits per cluster; Cluster size", 10, 0, 10);
  meCluZvsPhi_ = ibook.book2D(
      "BtlOccupancy", "BTL cluster Z vs #phi;Z_{RECO} [cm]; #phi_{RECO} [rad]", 144, -260., 260., 50, -3.2, 3.2);
  meCluEnergyvsEta_ = ibook.bookProfile(
      "BtlCluEnergyVsEta", "BTL cluster energy vs #eta; |#eta_{RECO}|; E_{RECO} [cm]", 30, 0., 1.55, 0., 20., "S");
  meCluHitsvsEta_ = ibook.bookProfile(
      "BtlCluHitsVsEta", "BTL hits per cluster vs #eta; |#eta_{RECO}|;Cluster size", 30, 0., 1.55, 0., 10., "S");

  meCluTimeRes_ = ibook.book1D("BtlCluTimeRes", "BTL cluster time resolution;T_{RECO}-T_{SIM} [ns]", 100, -0.5, 0.5);
  meCluEnergyRes_ =
      ibook.book1D("BtlCluEnergyRes", "BTL cluster energy resolution;E_{RECO}-E_{SIM} [MeV]", 100, -0.5, 0.5);
  meCluTResvsE_ = ibook.bookProfile("BtlCluTResvsE",
                                    "BTL cluster time resolution vs E;E_{SIM} [MeV];(T_{RECO}-T_{SIM}) [ns]",
                                    20,
                                    0.,
                                    20.,
                                    -0.5,
                                    0.5,
                                    "S");
  meCluTResvsEta_ = ibook.bookProfile("BtlCluTResvsEta",
                                      "BTL cluster time resolution vs #eta;|#eta_{RECO}|;(T_{RECO}-T_{SIM}) [ns]",
                                      30,
                                      0,
                                      1.55,
                                      -0.5,
                                      0.5,
                                      "S");
  meCluTPullvsE_ = ibook.bookProfile("BtlCluTPullvsE",
                                     "BTL cluster time pull vs E;E_{SIM} [MeV];(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
                                     20,
                                     0.,
                                     20.,
                                     -5.,
                                     5.,
                                     "S");
  meCluTPullvsEta_ =
      ibook.bookProfile("BtlCluTPullvsEta",
                        "BTL cluster time pull vs #eta;|#eta_{RECO}|;(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
                        30,
                        0,
                        1.55,
                        -5.,
                        5.,
                        "S");
  meCluRhoRes_ =
      ibook.book1D("BtlCluRhoRes", "BTL cluster #rho resolution;#rho_{RECO}-#rho_{SIM} [cm]", 100, -0.5, 0.5);
  meCluPhiRes_ =
      ibook.book1D("BtlCluPhiRes", "BTL cluster #phi resolution;#phi_{RECO}-#phi_{SIM} [rad]", 100, -0.03, 0.03);
  meCluZRes_ = ibook.book1D("BtlCluZRes", "BTL cluster Z resolution;Z_{RECO}-Z_{SIM} [cm]", 100, -0.2, 0.2);
  meCluLocalXRes_ =
      ibook.book1D("BtlCluLocalXRes", "BTL cluster local X resolution;X_{RECO}-X_{SIM} [cm]", 100, -3.1, 3.1);
  meCluLocalYRes_ =
      ibook.book1D("BtlCluLocalYRes", "BTL cluster local Y resolution;Y_{RECO}-Y_{SIM} [cm]", 100, -0.2, 0.2);
  meCluLocalXPull_ =
      ibook.book1D("BtlCluLocalXPull", "BTL cluster local X pull;X_{RECO}-X_{SIM}/sigmaX_[RECO]", 100, -5., 5.);
  meCluLocalYPull_ =
      ibook.book1D("BtlCluLocalYPull", "BTL cluster local Y pull;Y_{RECO}-Y_{SIM}/sigmaY_[RECO]", 100, -5., 5.);
  meCluZPull_ = ibook.book1D("BtlCluZPull", "BTL cluster Z pull;Z_{RECO}-Z_{SIM}/sigmaZ_[RECO]", 100, -5., 5.);
  meCluXLocalErr_ = ibook.book1D("BtlCluXLocalErr", "BTL cluster X local error;sigmaX_{RECO,loc} [cm]", 20, 0., 2.);
  meCluYLocalErr_ = ibook.book1D("BtlCluYLocalErr", "BTL cluster Y local error;sigmaY_{RECO,loc} [cm]", 20, 0., 0.4);
  if (optionalPlots_) {
    meCluYXLocal_ = ibook.book2D("BtlCluYXLocal",
                                 "BTL cluster local Y vs X;X^{local}_{RECO} [cm];Y^{local}_{RECO} [cm]",
                                 200,
                                 -9.5,
                                 9.5,
                                 200,
                                 -2.8,
                                 2.8);
    meCluYXLocalSim_ = ibook.book2D("BtlCluYXLocalSim",
                                    "BTL cluster local Y vs X;X^{local}_{SIM} [cm];Y^{local}_{SIM} [cm]",
                                    200,
                                    -9.5,
                                    9.5,
                                    200,
                                    -2.8,
                                    2.8);
  }
  meUnmatchedCluEnergy_ =
      ibook.book1D("BtlUnmatchedCluEnergy", "BTL unmatched cluster log10(energy);log10(E_{RECO} [MeV])", 5, -3, 2);

  // --- UncalibratedRecHits histograms

  if (uncalibRecHitsPlots_) {
    for (unsigned int ihistoQ = 0; ihistoQ < nBinsQ_; ++ihistoQ) {
      std::string hname = Form("TimeResQ_%d", ihistoQ);
      std::string htitle = Form("BTL time resolution (Q bin = %d);T_{RECO} - T_{SIM} [ns]", ihistoQ);
      meTimeResQ_[ihistoQ] = ibook.book1D(hname, htitle, 200, -0.3, 0.7);

      for (unsigned int ihistoEta = 0; ihistoEta < nBinsQEta_; ++ihistoEta) {
        hname = Form("TimeResQvsEta_%d_%d", ihistoQ, ihistoEta);
        htitle = Form("BTL time resolution (Q bin = %d, |#eta| bin = %d);T_{RECO} - T_{SIM} [ns]", ihistoQ, ihistoEta);
        meTimeResQvsEta_[ihistoQ][ihistoEta] = ibook.book1D(hname, htitle, 200, -0.3, 0.7);

      }  // ihistoEta loop

    }  // ihistoQ loop

    for (unsigned int ihistoEta = 0; ihistoEta < nBinsEta_; ++ihistoEta) {
      std::string hname = Form("TimeResEta_%d", ihistoEta);
      std::string htitle = Form("BTL time resolution (|#eta| bin = %d);T_{RECO} - T_{SIM} [ns]", ihistoEta);
      meTimeResEta_[ihistoEta] = ibook.book1D(hname, htitle, 200, -0.3, 0.7);

      for (unsigned int ihistoQ = 0; ihistoQ < nBinsEtaQ_; ++ihistoQ) {
        hname = Form("TimeResEtavsQ_%d_%d", ihistoEta, ihistoQ);
        htitle = Form("BTL time resolution (|#eta| bin = %d, Q bin = %d);T_{RECO} - T_{SIM} [ns]", ihistoEta, ihistoQ);
        meTimeResEtavsQ_[ihistoEta][ihistoQ] = ibook.book1D(hname, htitle, 200, -0.3, 0.7);

      }  // ihistoQ loop

    }  // ihistoEta loop
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlLocalRecoValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/LocalReco");
  desc.add<edm::InputTag>("recHitsTag", edm::InputTag("mtdRecHits", "FTLBarrel"));
  desc.add<edm::InputTag>("uncalibRecHitsTag", edm::InputTag("mtdUncalibratedRecHits", "FTLBarrel"));
  desc.add<edm::InputTag>("simHitsTag", edm::InputTag("mix", "g4SimHitsFastTimerHitsBarrel"));
  desc.add<edm::InputTag>("recCluTag", edm::InputTag("mtdClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("trkHitTag", edm::InputTag("mtdTrackingRecHits"));
  desc.add<double>("HitMinimumEnergy", 1.);  // [MeV]
  desc.add<bool>("optionalPlots", false);
  desc.add<bool>("UncalibRecHitsPlots", false);
  desc.add<double>("HitMinimumAmplitude", 30.);  // [pC]

  descriptions.add("btlLocalRecoValid", desc);
}

DEFINE_FWK_MODULE(BtlLocalRecoValidation);
