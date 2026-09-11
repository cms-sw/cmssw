// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      EtlLocalRecoSoAValidation
//
/**\class EtlLocalRecoSoAValidation EtlLocalRecoSoAValidation.cc Validation/MtdValidation/plugins/EtlLocalRecoSoAValidation.cc

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
#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLRecHitHostCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTypes.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomUtil.h"

#include "RecoLocalFastTime/Records/interface/MTDCPERecord.h"

#include "MTDHit.h"

class EtlLocalRecoSoAValidation : public DQMEDAnalyzer {
public:
  explicit EtlLocalRecoSoAValidation(const edm::ParameterSet&);
  ~EtlLocalRecoSoAValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;
  const float hitMinEnergy2Dis_;
  const bool optionalPlots_;
  const bool uncalibRecHitsPlots_;
  const double hitMinAmplitude_;

  edm::EDGetTokenT<etlrechit::ETLBaseRecHitHostCollection> etlBaseRecHitsSoAToken_;
  edm::EDGetTokenT<etlrechit::ETLRecHitHostCollection> etlRecHitsSoAToken_;
  edm::EDGetTokenT<CrossingFrame<PSimHit>> etlSimHitsToken_;
  edm::EDGetTokenT<MTDTrackingDetSetVector> mtdTrackingHitToken_;

  const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  const edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

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

  MonitorElement* meTimeRes_;
  MonitorElement* meEnergyRes_;
  MonitorElement* meTPullvsE_;
  MonitorElement* meTPullvsEta_;

  // --- UncalibratedRecHits histograms

  static constexpr int nBinsTot_ = 20;
  static constexpr float binWidthTot_ = 1.3;  // in MIP units

  MonitorElement* meTimeResTot_[2][nBinsTot_];

  static constexpr int nBinsEta_ = 26;
  static constexpr float binWidthEta_ = 0.05;
  static constexpr float etaMin_ = 1.65;

  MonitorElement* meTimeResEta_[2][nBinsEta_];
};

// ------------ constructor and destructor --------------
EtlLocalRecoSoAValidation::EtlLocalRecoSoAValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy2Dis_(iConfig.getParameter<double>("hitMinimumEnergy2Dis")),
      optionalPlots_(iConfig.getParameter<bool>("optionalPlots")),
      uncalibRecHitsPlots_(iConfig.getParameter<bool>("BaseRecHitsPlots")),
      hitMinAmplitude_(iConfig.getParameter<double>("HitMinimumAmplitude")),
      mtdgeoToken_(esConsumes<MTDGeometry, MTDDigiGeometryRecord>()),
      mtdtopoToken_(esConsumes<MTDTopology, MTDTopologyRcd>()) {
  etlRecHitsSoAToken_ =
      consumes<etlrechit::ETLRecHitHostCollection>(iConfig.getParameter<edm::InputTag>("recHitsSoATag"));
  etlBaseRecHitsSoAToken_ =
      consumes<etlrechit::ETLBaseRecHitHostCollection>(iConfig.getParameter<edm::InputTag>("uncalibRecHitsSoATag"));
  etlSimHitsToken_ = consumes<CrossingFrame<PSimHit>>(iConfig.getParameter<edm::InputTag>("simHitsTag"));
  mtdTrackingHitToken_ = consumes<MTDTrackingDetSetVector>(iConfig.getParameter<edm::InputTag>("trkHitTag"));
}

EtlLocalRecoSoAValidation::~EtlLocalRecoSoAValidation() {}

// ------------ method called for each event  ------------
void EtlLocalRecoSoAValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace geant_units::operators;
  using namespace mtd;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();

  MTDGeomUtil geomUtil;
  geomUtil.setGeometry(geom);

  auto etlRecHitsSoAHandle = makeValid(iEvent.getHandle(etlRecHitsSoAToken_));
  auto etlBaseRecHitsSoAHandle = makeValid(iEvent.getHandle(etlBaseRecHitsSoAToken_));
  auto etlSimHitsHandle = makeValid(iEvent.getHandle(etlSimHitsToken_));
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

      (simHitIt->second).thetaAtEntry = simHit.thetaAtEntry();
    }

  }  // simHit loop

  // --- Loop over the ELT RECO hits
  unsigned int n_reco_etl[4] = {0, 0, 0, 0};
  for (int i = 0; i < etlRecHitsSoAHandle->view().metadata().size(); i++) {
    auto recHit = etlRecHitsSoAHandle->view()[i];
    LogTrace("EtlLocalRecoSoAValidation")
        << "@RH detid " << recHit.detId().rawId() << " r/c " << recHit.row() << "/" << recHit.column()
        << " ToA,dToA,ToT " << recHit.toa() << ", " << recHit.toa_error() << ", " << recHit.tot();
    double weight = 1.0;
    ETLDetId detId = recHit.detId();
    DetId geoId = detId.geographicalId();
    const MTDGeomDet* thedet = geom->idToDet(geoId);
    if (thedet == nullptr)
      throw cms::Exception("EtlLocalRecoSoAValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
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

    meHitTime_[idet]->Fill(recHit.toa());
    meHitTimeError_[idet]->Fill(recHit.toa_error());

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
    meHitTvsPhi_[idet]->Fill(global_point.phi(), recHit.toa());
    meHitTvsEta_[idet]->Fill(global_point.eta(), recHit.toa());

    // Resolution histograms
    std::pair<uint8_t, uint8_t> pixel = geomUtil.pixelInModule(detId, local_point);
    mtd_digitizer::MTDCellId pixelId(detId.rawId(), pixel.first, pixel.second);

    if (m_etlSimHits[idet].count(pixelId) == 1) {
      if (m_etlSimHits[idet][pixelId].energy > hitMinEnergy2Dis_) {
        float time_res = recHit.toa() - m_etlSimHits[idet][pixelId].time;

        meTimeRes_->Fill(time_res);

        meTPullvsEta_->Fill(std::abs(global_point.eta()), time_res / recHit.toa_error());
        meTPullvsE_->Fill(m_etlSimHits[idet][pixelId].energy, time_res / recHit.toa_error());
      }
    }

    n_reco_etl[idet]++;
  }  // recHit loop

  for (int i = 0; i < 4; i++) {
    meNhits_[i]->Fill(std::log10(n_reco_etl[i]));
  }

  // --- Loop over the ETL Uncalibrated RECO hits
  if (optionalPlots_) {
    auto etlBaseRecHitsSoAHandle = makeValid(iEvent.getHandle(etlBaseRecHitsSoAToken_));

    for (int i = 0; i < etlBaseRecHitsSoAHandle->view().metadata().size(); i++) {
      auto uRecHit = etlBaseRecHitsSoAHandle->view()[i];
      ETLDetId detId = uRecHit.detId();
      int idet = detId.zside() + detId.nDisc();

      LogTrace("EtlLocalRecoSoAValidation")
          << "@URH detid " << detId.rawId() << " T " << uRecHit.toa() << " " << uRecHit.tot();

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

      if (uRecHit.tot() < hitMinAmplitude_)
        continue;

      float time_res = uRecHit.toa() - m_etlSimHits[idet][pixelId].time;

      int iside = (detId.zside() == -1 ? 0 : 1);

      // amplitude histograms

      int totBin = (int)(uRecHit.tot() / binWidthTot_);
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
    }
  }
}

// ------------ method for histogram booking ------------
void EtlLocalRecoSoAValidation::bookHistograms(DQMStore::IBooker& ibook,
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

  // --- UncalibratedRecHits histograms

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

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void EtlLocalRecoSoAValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ETL/LocalRecoSoA");
  desc.add<edm::InputTag>("recHitsSoATag", edm::InputTag("etlRecHitsSoA"));
  desc.add<edm::InputTag>("uncalibRecHitsSoATag", edm::InputTag("etlBaseRecHitsSoA"));
  desc.add<edm::InputTag>("simHitsTag", edm::InputTag("mix", "g4SimHitsFastTimerHitsEndcap"));
  desc.add<edm::InputTag>("trkHitTag", edm::InputTag("mtdTrackingRecHits"));
  desc.add<double>("hitMinimumEnergy2Dis", 0.001);  // [MeV]
  desc.add<bool>("optionalPlots", false);
  desc.add<bool>("BaseRecHitsPlots", false);
  desc.add<double>("HitMinimumAmplitude", 0.33);  // [MIP] old, now amplitude for recHit is time_over_threshold in ETL

  descriptions.add("etlLocalRecoSoAValid", desc);
}

DEFINE_FWK_MODULE(EtlLocalRecoSoAValidation);
