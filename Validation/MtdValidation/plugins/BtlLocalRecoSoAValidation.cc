// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      BtlLocalRecoSoAValidation
//
/**\class BtlLocalRecoSoAValidation BtlLocalRecoSoAValidation.cc Validation/MtdValidation/plugins/BtlLocalRecoSoAValidation.cc

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
#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/BTLRecHitHostCollection.h"

#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociationMap.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"

#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include "RecoLocalFastTime/Records/interface/MTDCPERecord.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDClusterParameterEstimator.h"

#include "MTDHit.h"

class BtlLocalRecoSoAValidation : public DQMEDAnalyzer {
public:
  explicit BtlLocalRecoSoAValidation(const edm::ParameterSet&);
  ~BtlLocalRecoSoAValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  //bool isSameCluster(const FTLCluster&, const FTLCluster&);

  // ------------ member data ------------

  const std::string folder_;
  const double hitMinEnergy_;
  const bool optionalPlots_;
  const bool uncalibRecHitsPlots_;
  const double hitMinAmplitude_;

  edm::EDGetTokenT<btlrechit::BTLBaseRecHitHostCollection> btlBaseRecHitsSoAToken_;
  edm::EDGetTokenT<btlrechit::BTLRecHitHostCollection> btlRecHitsSoAToken_;
  edm::EDGetTokenT<CrossingFrame<PSimHit>> btlSimHitsToken_;

  const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  const edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

  // --- histograms declaration

  MonitorElement* meNevents_;

  MonitorElement* meNhits_;

  static constexpr int nRU_ = BTLDetId::kRUPerRod;

  MonitorElement* meHitEnergy_;
  MonitorElement* meHitEnergyRUSlice_[nRU_];
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

  MonitorElement* meTimeRes_;
  MonitorElement* meTimeResVsE_;
  MonitorElement* meEnergyRes_;
  MonitorElement* meEnergyRelResVsE_;

  MonitorElement* meLongPosPull_;
  MonitorElement* meLongPosPullvsE_;
  MonitorElement* meLongPosPullvsEta_;

  MonitorElement* meTPullvsE_;
  MonitorElement* meTPullvsEta_;
  MonitorElement* meUnmatchedRecHit_;

  // --- BaseratedRecHits histograms

  MonitorElement* meUncEneMinusVsX_;
  MonitorElement* meUncEnePlusVsX_;
  MonitorElement* meUncTimeMinusVsX_;
  MonitorElement* meUncTimePlusVsX_;

  static constexpr int nBinsQ_ = 30;
  static constexpr float binWidthQ_ = 1142.5;  // [npe]
  static constexpr int nBinsQEta_ = 3;
  static constexpr float binsQEta_[nBinsQEta_ + 1] = {0., 0.65, 1.15, 1.55};

  MonitorElement* meTimeResQ_[nBinsQ_];
  MonitorElement* meTimeResQvsEta_[nBinsQ_][nBinsQEta_];

  static constexpr int nBinsEta_ = 31;
  static constexpr float binWidthEta_ = 0.05;
  static constexpr int nBinsEtaQ_ = 6;
  static constexpr float binsEtaQ_[nBinsEtaQ_ + 1] = {0., 2., 4., 6., 8., 12., 15.};

  MonitorElement* meTimeResEta_[nBinsEta_];
  MonitorElement* meTimeResEtavsQ_[nBinsEta_][nBinsEtaQ_];
};

// ------------ constructor and destructor --------------
BtlLocalRecoSoAValidation::BtlLocalRecoSoAValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy_(iConfig.getParameter<double>("HitMinimumEnergy")),
      optionalPlots_(iConfig.getParameter<bool>("optionalPlots")),
      uncalibRecHitsPlots_(iConfig.getParameter<bool>("BaseRecHitsPlots")),
      hitMinAmplitude_(iConfig.getParameter<double>("HitMinimumAmplitude")),
      mtdgeoToken_(esConsumes<MTDGeometry, MTDDigiGeometryRecord>()),
      mtdtopoToken_(esConsumes<MTDTopology, MTDTopologyRcd>()) {
  btlRecHitsSoAToken_ =
      consumes<btlrechit::BTLRecHitHostCollection>(iConfig.getParameter<edm::InputTag>("recHitsSoATag"));
  btlBaseRecHitsSoAToken_ =
      consumes<btlrechit::BTLBaseRecHitHostCollection>(iConfig.getParameter<edm::InputTag>("uncalibRecHitsSoATag"));
  btlSimHitsToken_ = consumes<CrossingFrame<PSimHit>>(iConfig.getParameter<edm::InputTag>("simHitsTag"));
}

BtlLocalRecoSoAValidation::~BtlLocalRecoSoAValidation() {}

// ------------ method called for each event  ------------
void BtlLocalRecoSoAValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace geant_units::operators;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  auto btlRecHitsSoAHandle = makeValid(iEvent.getHandle(btlRecHitsSoAToken_));
  auto btlSimHitsHandle = makeValid(iEvent.getHandle(btlSimHitsToken_));
  MixCollection<PSimHit> btlSimHits(btlSimHitsHandle.product());

  // --- Loop over the BTL SIM hits
  std::unordered_map<uint32_t, MTDHit> m_btlSimHits;
  for (auto const& simHit : btlSimHits) {
    // --- Use only hits compatible with the in-time bunch-crossing
    if (simHit.tof() < 0 || simHit.tof() > 25.)
      continue;

    DetId id = simHit.detUnitId();

    auto simHitIt = m_btlSimHits.emplace(id.rawId(), MTDHit()).first;

    // --- Accumulate the energy (in GeV) of SIM hits in the same detector cell
    (simHitIt->second).energy += simHit.energyLoss();  // [GeV]

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
  unsigned int n_reco_btl_nosimhit = 0;
  //for (const auto& recHit : *btlRecHitsHandle) {
  for (int i = 0; i < btlRecHitsSoAHandle->view().metadata().size(); i++) {
    auto recHit = btlRecHitsSoAHandle->view()[i];
    LogTrace("BtlLocalRecoSoAValidation") << "@RH detid " << recHit.detId().rawId() << " r/c/X/dX " << recHit.row()
                                          << " " << recHit.position() << " " << recHit.position_error() << " E,T,dT "
                                          << recHit.energy() << " " << recHit.time1() << " " << recHit.time1_error();

    BTLDetId detId = recHit.detId();
    DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("BtlLocalRecoSoAValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                        << detId.rawId() << ") is invalid!" << std::dec << std::endl;
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    Local3DPoint local_point(0., 0., 0.);
    local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
    const auto& global_point = thedet->toGlobal(local_point);

    meHitEnergy_->Fill(recHit.energy());
    const int iRU = (detId.rawId() & BTLDetId::kBTLNewFormat ? detId.runit() : detId.runit() - 1);
    meHitEnergyRUSlice_[iRU]->Fill(recHit.energy());
    meHitLogEnergy_->Fill(log10(recHit.energy()));
    meHitTime_->Fill(recHit.time1());
    meHitTimeError_->Fill(recHit.time1_error());
    meHitLongPos_->Fill(recHit.position());

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

    meHitTvsE_->Fill(recHit.energy(), recHit.time1());
    meHitEvsPhi_->Fill(global_point.phi(), recHit.energy());
    meHitEvsEta_->Fill(global_point.eta(), recHit.energy());
    meHitEvsZ_->Fill(global_point.z(), recHit.energy());
    meHitTvsPhi_->Fill(global_point.phi(), recHit.time1());
    meHitTvsEta_->Fill(global_point.eta(), recHit.time1());
    meHitTvsZ_->Fill(global_point.z(), recHit.time1());

    // Resolution histograms
    LogDebug("BtlLocalRecoSoAValidation")
        << "RecoHit DetId= " << detId.rawId() << " sim hits in id= " << m_btlSimHits.count(detId.rawId());
    if (m_btlSimHits.count(detId.rawId()) == 1 && m_btlSimHits[detId.rawId()].energy > hitMinEnergy_) {
      float longpos_res = recHit.position() - convertMmToCm(m_btlSimHits[detId.rawId()].x);
      float time_res = recHit.time1() - m_btlSimHits[detId.rawId()].time;
      float energy_res = recHit.energy() - m_btlSimHits[detId.rawId()].energy;

      Local3DPoint local_point_sim(convertMmToCm(m_btlSimHits[detId.rawId()].x),
                                   convertMmToCm(m_btlSimHits[detId.rawId()].y),
                                   convertMmToCm(m_btlSimHits[detId.rawId()].z));
      local_point_sim =
          topo.pixelToModuleLocalPoint(local_point_sim, detId.row(topo.nrows()), detId.column(topo.nrows()));
      const auto& global_point_sim = thedet->toGlobal(local_point_sim);

      meTimeRes_->Fill(time_res);
      meTimeResVsE_->Fill(recHit.energy(), time_res);
      meEnergyRes_->Fill(energy_res);
      meEnergyRelResVsE_->Fill(recHit.energy(), energy_res / recHit.energy());

      meLongPosPull_->Fill(longpos_res / recHit.position_error());
      meLongPosPullvsEta_->Fill(std::abs(global_point_sim.eta()), longpos_res / recHit.position_error());
      meLongPosPullvsE_->Fill(m_btlSimHits[detId.rawId()].energy, longpos_res / recHit.position_error());

      meTPullvsEta_->Fill(std::abs(global_point_sim.eta()), time_res / recHit.time1_error());
      meTPullvsE_->Fill(m_btlSimHits[detId.rawId()].energy, time_res / recHit.time1_error());
    } else if (m_btlSimHits.count(detId.rawId()) == 0) {
      n_reco_btl_nosimhit++;
      LogDebug("BtlLocalRecoSoAValidation")
          << "BTL rec hit with no corresponding sim hit in crystal, DetId= " << detId.rawId()
          << " geoId= " << geoId.rawId() << " ene= " << recHit.energy() << " time= " << recHit.time1();
    }

    n_reco_btl++;

  }  // recHit loop

  if (n_reco_btl > 0) {
    meNhits_->Fill(std::log10(n_reco_btl));
  }
  if (n_reco_btl_nosimhit == 0) {
    meUnmatchedRecHit_->Fill(-1.5);
  } else {
    meUnmatchedRecHit_->Fill(std::log10(n_reco_btl_nosimhit));
  }

  // --- Loop over the BTL Baserated RECO hits
  if (optionalPlots_) {
    auto btlBaseRecHitsSoAHandle = makeValid(iEvent.getHandle(btlBaseRecHitsSoAToken_));
    for (int i = 0; i < btlBaseRecHitsSoAHandle->view().metadata().size(); i++) {
      //for (const auto& uRecHit : *btlBaseRecHitsHandle) {
      auto uRecHit = btlBaseRecHitsSoAHandle->view()[i];
      BTLDetId detId = uRecHit.detId();

      LogTrace("BtlLocalRecoSoAValidation")
          << "@URH detid " << detId.rawId() << " A " << uRecHit.ampPlus() << " " << uRecHit.ampMinus() << " T "
          << uRecHit.time1Plus() << " " << uRecHit.time1Minus();

      // --- Skip BaseratedRecHits not matched to SimHits
      if (m_btlSimHits.count(detId.rawId()) != 1)
        continue;

      // --- Combine the information from the left and right BTL cell sides

      float nHits = 0.;
      float hit_amplitude = 0.;
      float hit_time = 0.;

      // left side:
      if (uRecHit.ampPlus() > 0.) {
        hit_amplitude += uRecHit.ampPlus();
        hit_time += uRecHit.time1Plus();
        nHits += 1.;
      }
      // right side:
      if (uRecHit.ampMinus() > 0.) {
        hit_amplitude += uRecHit.ampMinus();
        hit_time += uRecHit.time1Minus();
        nHits += 1.;
      }

      hit_amplitude /= nHits;
      hit_time /= nHits;

      LogDebug("BtlLocalRecoSoAValidation") << "#unc " << nHits << " A/T " << hit_amplitude << " " << hit_time;
      if (nHits == 0.) {
        edm::LogWarning("BtlLocalRecoSoAValidation") << "Empty uncalibrated hit in DetId " << detId;
        continue;
      }

      if (hit_amplitude < hitMinAmplitude_)
        continue;

      if (uncalibRecHitsPlots_) {
        DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
        const MTDGeomDet* thedet = geom->idToDet(geoId);
        if (thedet == nullptr)
          throw cms::Exception("BtlLocalRecoSoAValidation")
              << "GeographicalID: " << std::hex << geoId.rawId() << " (" << detId.rawId() << ") is invalid!" << std::dec
              << std::endl;
        const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
        const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

        Local3DPoint local_point(0., 0., 0.);
        local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
        const auto& global_point = thedet->toGlobal(local_point);

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
      }
    }  // uRecHit loop}
  }
}

// ------------ method for histogram booking ------------
void BtlLocalRecoSoAValidation::bookHistograms(DQMStore::IBooker& ibook,
                                               edm::Run const& run,
                                               edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // --- histograms booking

  meNevents_ = ibook.book1D("BtlNevents", "Number of events", 1, 0., 1.);

  meNhits_ = ibook.book1D("BtlNhits", "Number of BTL RECO hits;log_{10}(N_{RECO})", 100, 0., 5.25);

  meHitEnergy_ = ibook.book1D("BtlHitEnergy", "BTL RECO hits energy;E_{RECO} [GeV]", 100, 0., 0.020);
  for (unsigned int ihistoRU = 0; ihistoRU < nRU_; ++ihistoRU) {
    std::string name_Energy = "BtlHitEnergyRUSlice" + std::to_string(ihistoRU);
    std::string title_Energy = "BTL RECO hits energy (RU " + std::to_string(ihistoRU) + ");E_{RECO} [GeV])";
    meHitEnergyRUSlice_[ihistoRU] = ibook.book1D(name_Energy, title_Energy, 100, 0., 0.020);
  }
  meHitLogEnergy_ = ibook.book1D("BtlHitLogEnergy", "BTL RECO hits energy;log_{10}(E_{RECO} [GeV])", 16, -3.1, -1.5);
  meHitTime_ = ibook.book1D("BtlHitTime", "BTL RECO hits ToA;ToA_{RECO} [ns]", 100, 0., 25.);
  meHitTimeError_ = ibook.book1D("BtlHitTimeError", "BTL RECO hits ToA error;#sigma^{ToA}_{RECO} [ns]", 50, 0., 0.1);
  meOccupancy_ = ibook.book2D(
      "BtlOccupancy", "BTL RECO hits occupancy;Z_{RECO} [cm]; #phi_{RECO} [rad]", 65, -260., 260., 126, -3.2, 3.2);
  if (optionalPlots_) {
    meLocalOccupancy_ = ibook.book2D("BtlLocalOccupancy",
                                     "BTL RECO hits local occupancy;X^{loc}_{RECO} [cm]; Y^{loc}_{RECO} [cm]",
                                     100,
                                     10.,
                                     10.,
                                     60,
                                     -3.,
                                     3.);
  }
  meHitXlocal_ = ibook.book1D("BtlHitXlocal", "BTL RECO hits local X;X_{RECO}^{loc} [cm]", 100, -10., 10.);
  meHitYlocal_ = ibook.book1D("BtlHitYlocal", "BTL RECO hits local Y;Y_{RECO}^{loc} [cm]", 60, -3, 3);
  meHitZlocal_ = ibook.book1D("BtlHitZlocal", "BTL RECO hits local Z;Z_{RECO}^{loc} [cm]", 8, -0.4, 0.4);
  meHitZ_ = ibook.book1D("BtlHitZ", "BTL RECO hits Z;Z_{RECO} [cm]", 100, -260., 260.);
  meHitPhi_ = ibook.book1D("BtlHitPhi", "BTL RECO hits #phi;#phi_{RECO} [rad]", 126, -3.2, 3.2);
  meHitEta_ = ibook.book1D("BtlHitEta", "BTL RECO hits #eta;#eta_{RECO}", 100, -1.55, 1.55);
  meHitTvsE_ = ibook.bookProfile(
      "BtlHitTvsE", "BTL RECO hits ToA vs energy;E_{RECO} [GeV];ToA_{RECO} [ns]", 50, 0., 0.02, 0., 100.);
  meHitEvsPhi_ = ibook.bookProfile(
      "BtlHitEvsPhi", "BTL RECO hits energy vs #phi;#phi_{RECO} [rad];E_{RECO} [GeV]", 50, -3.2, 3.2, 0., 0.1);
  meHitEvsEta_ = ibook.bookProfile(
      "BtlHitEvsEta", "BTL RECO hits energy vs #eta;#eta_{RECO};E_{RECO} [GeV]", 50, -1.55, 1.55, 0., 0.1);
  meHitEvsZ_ = ibook.bookProfile(
      "BtlHitEvsZ", "BTL RECO hits energy vs Z;Z_{RECO} [cm];E_{RECO} [GeV]", 50, -260., 260., 0., 0.1);
  meHitTvsPhi_ = ibook.bookProfile(
      "BtlHitTvsPhi", "BTL RECO hits ToA vs #phi;#phi_{RECO} [rad];ToA_{RECO} [ns]", 50, -3.2, 3.2, 0., 100.);
  meHitTvsEta_ = ibook.bookProfile(
      "BtlHitTvsEta", "BTL RECO hits ToA vs #eta;#eta_{RECO};ToA_{RECO} [ns]", 50, -1.6, 1.6, 0., 100.);
  meHitTvsZ_ = ibook.bookProfile(
      "BtlHitTvsZ", "BTL RECO hits ToA vs Z;Z_{RECO} [cm];ToA_{RECO} [ns]", 50, -260., 260., 0., 100.);
  meHitLongPos_ = ibook.book1D("BtlLongPos", "BTL RECO hits longitudinal position; L_{RECO} [cm]", 50, -5, 5);
  meTimeRes_ = ibook.book1D("BtlTimeRes", "BTL RECO hits time resolution;T_{RECO}-T_{SIM} [ns]", 100, -0.5, 0.5);
  meTimeResVsE_ = ibook.bookProfile("BtlTimeResvsE",
                                    "BTL RECO hits time resolution vs energy;E_{RECO} [GeV];T_{RECO}-T_{SIM} [ns]",
                                    50,
                                    0.,
                                    0.020,
                                    -0.5,
                                    0.5,
                                    "S");
  meEnergyRes_ =
      ibook.book1D("BtlEnergyRes", "BTL RECO hits energy resolution;E_{RECO}-E_{SIM} [GeV]", 100, -0.0005, 0.0005);
  meEnergyRelResVsE_ =
      ibook.bookProfile("BtlEnergyRelResvsE",
                        "BTL relative energy resolution vs hit energy;E_{RECO} [GeV];E_{RECO}-E_{SIM} [GeV]",
                        50,
                        0.,
                        0.020,
                        -0.00015,
                        0.00015,
                        "S");
  meLongPosPull_ = ibook.book1D(
      "BtlLongPosPull", "BTL longitudinal position pull;(L_{RECO}-L_{SIM})/#sigma_{L_{RECO}}", 100, -5., 5.);
  meLongPosPullvsE_ =
      ibook.bookProfile("BtlLongposPullvsE",
                        "BTL longitudinal position pull vs E;E_{SIM} [GeV];(L_{RECO}-L_{SIM})/#sigma_{L_{RECO}}",
                        20,
                        0.,
                        0.020,
                        -5.,
                        5.,
                        "S");
  meLongPosPullvsEta_ =
      ibook.bookProfile("BtlLongposPullvsEta",
                        "BTL longitudinal position pull vs #eta;|#eta_{RECO}|;(L_{RECO}-L_{SIM})/#sigma_{L_{RECO}}",
                        32,
                        0,
                        1.55,
                        -5.,
                        5.,
                        "S");
  meTPullvsE_ = ibook.bookProfile("BtlTPullvsE",
                                  "BTL time pull vs E;E_{SIM} [GeV];(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
                                  20,
                                  0.,
                                  0.020,
                                  -5.,
                                  5.,
                                  "S");
  meTPullvsEta_ = ibook.bookProfile("BtlTPullvsEta",
                                    "BTL time pull vs #eta;|#eta_{RECO}|;(T_{RECO}-T_{SIM})/#sigma_{T_{RECO}}",
                                    30,
                                    0,
                                    1.55,
                                    -5.,
                                    5.,
                                    "S");

  meUnmatchedRecHit_ = ibook.book1D(
      "UnmatchedRecHit", "log10(#BTL crystals with rechits but no simhit);log10(#BTL rechits)", 80, -2., 6.);

  // --- BaseratedRecHits histograms

  if (optionalPlots_) {
    meUncEneMinusVsX_ =
        ibook.bookProfile("BTLUncEneMinusVsX",
                          "BTL uncalibrated left hit energy - average vs X;X [cm];#Delta(E_{minus}) [GeV]",
                          20,
                          -0.005,
                          0.005,
                          -20.,
                          20.,
                          "S");
    meUncEnePlusVsX_ =
        ibook.bookProfile("BTLUncEnePlusVsX",
                          "BTL uncalibrated right hit energy - average vs X;X [cm];#Delta(E_{plus}) [GeV]",
                          20,
                          -0.005,
                          0.005,
                          -20.,
                          20.,
                          "S");

    meUncTimeMinusVsX_ = ibook.bookProfile("BTLUncTimeMinusVsX",
                                           "BTL uncalibrated left hit time - average vs X;X [cm];#Delta(T_{plus}) [ns]",
                                           20,
                                           -5.,
                                           5.,
                                           -25.,
                                           25.,
                                           "S");
    meUncTimePlusVsX_ =
        ibook.bookProfile("BTLUncTimePlusVsX",
                          "BTL uncalibrated right hit time - average vs X;X [cm];#Delta(T_{minus}) [ns]",
                          20,
                          -5.,
                          5.,
                          -25.,
                          25.,
                          "S");
    if (uncalibRecHitsPlots_) {
      for (unsigned int ihistoQ = 0; ihistoQ < nBinsQ_; ++ihistoQ) {
        std::string hname = Form("TimeResQ_%d", ihistoQ);
        std::string htitle = Form("BTL time resolution (E bin = %d);T_{RECO} - T_{SIM} [ns]", ihistoQ);
        meTimeResQ_[ihistoQ] = ibook.book1D(hname, htitle, 200, -0.3, 0.7);

        for (unsigned int ihistoEta = 0; ihistoEta < nBinsQEta_; ++ihistoEta) {
          hname = Form("TimeResQvsEta_%d_%d", ihistoQ, ihistoEta);
          htitle =
              Form("BTL time resolution (E bin = %d, |#eta| bin = %d);T_{RECO} - T_{SIM} [ns]", ihistoQ, ihistoEta);
          meTimeResQvsEta_[ihistoQ][ihistoEta] = ibook.book1D(hname, htitle, 200, -0.3, 0.7);

        }  // ihistoEta loop

      }  // ihistoQ loop

      for (unsigned int ihistoEta = 0; ihistoEta < nBinsEta_; ++ihistoEta) {
        std::string hname = Form("TimeResEta_%d", ihistoEta);
        std::string htitle = Form("BTL time resolution (|#eta| bin = %d);T_{RECO} - T_{SIM} [ns]", ihistoEta);
        meTimeResEta_[ihistoEta] = ibook.book1D(hname, htitle, 200, -0.3, 0.7);

        for (unsigned int ihistoQ = 0; ihistoQ < nBinsEtaQ_; ++ihistoQ) {
          hname = Form("TimeResEtavsQ_%d_%d", ihistoEta, ihistoQ);
          htitle =
              Form("BTL time resolution (|#eta| bin = %d, E bin = %d);T_{RECO} - T_{SIM} [ns]", ihistoEta, ihistoQ);
          meTimeResEtavsQ_[ihistoEta][ihistoQ] = ibook.book1D(hname, htitle, 200, -0.3, 0.7);

        }  // ihistoQ loop

      }  // ihistoEta loop
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BtlLocalRecoSoAValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/LocalRecoSoA");
  desc.add<edm::InputTag>("recHitsSoATag", edm::InputTag("btlRecHitsSoA"));
  desc.add<edm::InputTag>("uncalibRecHitsSoATag", edm::InputTag("btlBaseRecHitsSoA"));
  desc.add<edm::InputTag>("simHitsTag", edm::InputTag("mix", "g4SimHitsFastTimerHitsBarrel"));
  desc.add<double>("HitMinimumEnergy", 0.001);  // [GeV]
  desc.add<bool>("optionalPlots", false);
  desc.add<bool>("BaseRecHitsPlots", false);
  desc.add<double>("HitMinimumAmplitude", 2285.);  // [npe]

  descriptions.add("btlLocalRecoSoAValid", desc);
}

DEFINE_FWK_MODULE(BtlLocalRecoSoAValidation);
