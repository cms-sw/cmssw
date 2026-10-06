/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"
#include "CondFormats/DataRecord/interface/PPSDirectSimulationDataRcd.h"
#include "CondFormats/DataRecord/interface/PPSPixelTopologyRcd.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"
#include "CondFormats/PPSObjects/interface/PPSDirectSimulationData.h"
#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"
#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimPPS/DirectSimProducer/interface/DirectSimulator.h"

#include <CLHEP/Random/RandFlat.h>
#include <CLHEP/Random/RandGauss.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>
#include <TF1.h>
#include <TMatrixD.h>
#include <TVectorD.h>

//----------------------------------------------------------------------------------------------------

class PPSDirectProtonSimulation : public edm::stream::EDProducer<> {
public:
  explicit PPSDirectProtonSimulation(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void processProton(const HepMC::GenVertex* in_vtx,
                     const HepMC::GenParticle* in_trk,
                     const CTPPSGeometry& geometry,
                     const PPSPixelTopology& ppt,
                     CLHEP::HepRandomEngine* rndEngine,
                     std::vector<CTPPSLocalTrackLite>& out_tracks,
                     edm::DetSetVector<TotemRPRecHit>& out_strip_hits,
                     edm::DetSetVector<CTPPSPixelRecHit>& out_pixel_hits,
                     edm::DetSetVector<CTPPSDiamondRecHit>& out_diamond_hits,
                     std::map<int, edm::DetSetVector<TotemRPRecHit>>& out_strip_hits_per_particle,
                     std::map<int, edm::DetSetVector<CTPPSPixelRecHit>>& out_pixel_hits_per_particle,
                     std::map<int, edm::DetSetVector<CTPPSDiamondRecHit>>& out_diamond_hits_per_particle) const;

  static constexpr double inv_sqrt_12_ = 1. / std::sqrt(12.);

  // ------------ config file parameters ------------

  DirectSimulator direct_simulator_;

  // conditions
  edm::ESGetToken<LHCInfo, LHCInfoRcd> tokenLHCInfo_;
  edm::ESGetToken<CTPPSBeamParameters, CTPPSBeamParametersRcd> tokenBeamParameters_;
  edm::ESGetToken<PPSPixelTopology, PPSPixelTopologyRcd> pixelTopologyToken_;
  edm::ESGetToken<LHCInterpolatedOpticalFunctionsSetCollection, CTPPSInterpolatedOpticsRcd> tokenOpticalFunctions_;
  edm::ESGetToken<CTPPSGeometry, VeryForwardMisalignedGeometryRecord> tokenGeometry_;
  edm::ESGetToken<PPSDirectSimulationData, PPSDirectSimulationDataRcd> tokenDirectSimuData_;

  edm::ESWatcher<LHCInfoRcd> lhcInfoRcdWatcher_;
  edm::ESWatcher<CTPPSBeamParametersRcd> beamParametersRcdWatcher_;
  edm::ESWatcher<CTPPSInterpolatedOpticsRcd> opticalFunctionsRcdWatcher_;
  edm::ESWatcher<VeryForwardMisalignedGeometryRecord> geometryRcdWatcher_;
  edm::ESWatcher<PPSDirectSimulationDataRcd> directSimuDataRcdWatcher_;

  // input
  edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;

  // flags what output to be produced
  bool produceScoringPlaneHits_;
  bool produceRecHits_;

  // efficiency flags
  bool useTrackingEfficiencyPerPlane_;
  bool useTimingEfficiencyPerPlane_;

  // efficiency maps
  std::map<unsigned int, std::unique_ptr<TH2F>> efficiencyMapsPerPlane_;

  // other parameters
  bool roundToPitch_;
  bool checkIsHit_;

  double pitchStrips_;              ///< strip pitch in mm
  double insensitiveMarginStrips_;  ///< size of insensitive margin at sensor's edge facing the beam, in mm

  const double pitchPixelsHor_;
  const double pitchPixelsVer_;
  const LocalError pixel_local_error_;

  unsigned int verbosity_;

  std::unique_ptr<TF1> timeResolutionDiamonds45_, timeResolutionDiamonds56_;

  // ------------ internal parameters ------------

  /// internal variable: v position of strip 0, in mm
  double stripZeroPosition_;
};

//----------------------------------------------------------------------------------------------------

PPSDirectProtonSimulation::PPSDirectProtonSimulation(const edm::ParameterSet& iConfig)
    : direct_simulator_(iConfig),
      tokenLHCInfo_(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("lhcInfoLabel")})),
      tokenBeamParameters_(esConsumes()),
      pixelTopologyToken_(esConsumes()),
      tokenOpticalFunctions_(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("opticsLabel")})),
      tokenGeometry_(esConsumes()),
      tokenDirectSimuData_(esConsumes()),
      hepMCToken_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("hepMCTag"))),
      produceScoringPlaneHits_(iConfig.getParameter<bool>("produceScoringPlaneHits")),
      produceRecHits_(iConfig.getParameter<bool>("produceRecHits")),
      useTrackingEfficiencyPerPlane_(iConfig.getParameter<bool>("useTrackingEfficiencyPerPlane")),
      useTimingEfficiencyPerPlane_(iConfig.getParameter<bool>("useTimingEfficiencyPerPlane")),
      roundToPitch_(iConfig.getParameter<bool>("roundToPitch")),
      checkIsHit_(iConfig.getParameter<bool>("checkIsHit")),
      pitchStrips_(iConfig.getParameter<double>("pitchStrips")),
      insensitiveMarginStrips_(iConfig.getParameter<double>("insensitiveMarginStrips")),
      pitchPixelsHor_(iConfig.getParameter<double>("pitchPixelsHor")),
      pitchPixelsVer_(iConfig.getParameter<double>("pitchPixelsVer")),
      pixel_local_error_(pitchPixelsHor_ * inv_sqrt_12_, 0., pitchPixelsVer_ * inv_sqrt_12_),
      verbosity_(iConfig.getUntrackedParameter<unsigned int>("verbosity", 0)) {
  if (produceScoringPlaneHits_)
    produces<std::vector<CTPPSLocalTrackLite>>();

  if (produceRecHits_) {
    produces<edm::DetSetVector<TotemRPRecHit>>();
    produces<edm::DetSetVector<CTPPSDiamondRecHit>>();
    produces<edm::DetSetVector<CTPPSPixelRecHit>>();

    produces<std::map<int, edm::DetSetVector<TotemRPRecHit>>>();
    produces<std::map<int, edm::DetSetVector<CTPPSDiamondRecHit>>>();
    produces<std::map<int, edm::DetSetVector<CTPPSPixelRecHit>>>();
  }

  // check user input
  if (iConfig.getParameter<bool>("useTrackingEfficiencyPerRP") && useTrackingEfficiencyPerPlane_)
    throw cms::Exception("PPS")
        << "useTrackingEfficiencyPerRP and useTrackingEfficiencyPerPlane should not be simultaneously set true.";

  if (iConfig.getParameter<bool>("useTimingEfficiencyPerRP") && useTimingEfficiencyPerPlane_)
    throw cms::Exception("PPS")
        << "useTimingEfficiencyPerRP and useTimingEfficiencyPerPlane should not be simultaneously set true.";

  // v position of strip 0
  stripZeroPosition_ = RPTopology::last_strip_to_border_dist_ + (RPTopology::no_of_strips_ - 1) * RPTopology::pitch_ -
                       RPTopology::y_width_ / 2.;
}

//----------------------------------------------------------------------------------------------------

void PPSDirectProtonSimulation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<unsigned int>("verbosity", 0);

  desc.add<std::string>("lhcInfoLabel", "")->setComment("label of the LHCInfo record");
  desc.add<std::string>("opticsLabel", "")->setComment("label of the optics records");

  desc.add<edm::InputTag>("hepMCTag", edm::InputTag("generator", "unsmeared"));

  desc.add<bool>("produceScoringPlaneHits", true);
  desc.add<bool>("produceRecHits", true);

  desc.add<bool>("useEmpiricalApertures", true);

  desc.add<bool>("useTrackingEfficiencyPerRP", false);
  desc.add<bool>("useTimingEfficiencyPerRP", false);
  desc.add<bool>("useTrackingEfficiencyPerPlane", false);
  desc.add<bool>("useTimingEfficiencyPerPlane", false);

  desc.add<bool>("produceHitsRelativeToBeam", true);
  desc.add<bool>("roundToPitch", true);
  desc.add<bool>("checkIsHit", true);
  desc.add<double>("pitchStrips", 66.e-3);              // in mm
  desc.add<double>("insensitiveMarginStrips", 34.e-3);  // in mm

  desc.add<double>("pitchPixelsHor", 100.e-3);
  desc.add<double>("pitchPixelsVer", 150.e-3);

  descriptions.add("ppsDirectProtonSimulation", desc);
}

//----------------------------------------------------------------------------------------------------

void PPSDirectProtonSimulation::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get conditions
  if (lhcInfoRcdWatcher_.check(iSetup))
    direct_simulator_.setLHCInfo(iSetup.getData(tokenLHCInfo_));
  if (beamParametersRcdWatcher_.check(iSetup))
    direct_simulator_.setBeamParameters(iSetup.getData(tokenBeamParameters_));
  if (opticalFunctionsRcdWatcher_.check(iSetup))
    direct_simulator_.setOpticalFunctions(iSetup.getData(tokenOpticalFunctions_));
  if (geometryRcdWatcher_.check(iSetup))
    direct_simulator_.setGeometry(iSetup.getData(tokenGeometry_));
  if (directSimuDataRcdWatcher_.check(iSetup)) {
    auto const& directSimuData = iSetup.getData(tokenDirectSimuData_);
    direct_simulator_.setDirectSimulationData(directSimuData);
    timeResolutionDiamonds45_ =
        std::make_unique<TF1>(TF1("timeResolutionDiamonds45", directSimuData.getTimeResolutionDiamonds45().c_str()));
    timeResolutionDiamonds56_ =
        std::make_unique<TF1>(TF1("timeResolutionDiamonds56", directSimuData.getTimeResolutionDiamonds56().c_str()));

    if (useTrackingEfficiencyPerPlane_ || useTimingEfficiencyPerPlane_)  // load the efficiency maps
      efficiencyMapsPerPlane_ = directSimuData.loadEffeciencyHistogramsPerPlane();
  }

  // prepare outputs
  auto pStripRecHits = std::make_unique<edm::DetSetVector<TotemRPRecHit>>();
  auto pDiamondRecHits = std::make_unique<edm::DetSetVector<CTPPSDiamondRecHit>>();
  auto pPixelRecHits = std::make_unique<edm::DetSetVector<CTPPSPixelRecHit>>();

  auto pStripRecHitsPerParticle = std::make_unique<std::map<int, edm::DetSetVector<TotemRPRecHit>>>();
  auto pDiamondRecHitsPerParticle = std::make_unique<std::map<int, edm::DetSetVector<CTPPSDiamondRecHit>>>();
  auto pPixelRecHitsPerParticle = std::make_unique<std::map<int, edm::DetSetVector<CTPPSPixelRecHit>>>();

  std::unique_ptr<std::vector<CTPPSLocalTrackLite>> pTracks(new std::vector<CTPPSLocalTrackLite>());

  // get random engine
  edm::Service<edm::RandomNumberGenerator> rng;
  auto* engine = &rng->getEngine(iEvent.streamID());
  const auto& ppt = iSetup.getData(pixelTopologyToken_);
  const auto& geometry = iSetup.getData(tokenGeometry_);

  // loop over event vertices
  const auto& evt = iEvent.get(hepMCToken_).GetEvent();
  for (auto it_vtx = evt->vertices_begin(); it_vtx != evt->vertices_end(); ++it_vtx) {
    const auto& vtx = *it_vtx;

    // loop over outgoing particles
    for (auto it_part = vtx->particles_out_const_begin(); it_part != vtx->particles_out_const_end(); ++it_part) {
      const auto& part = *it_part;

      // accept only stable protons
      if (part->pdg_id() != 2212)
        continue;
      if (part->status() != 1 && part->status() < 83)
        continue;

      processProton(vtx,
                    part,
                    geometry,
                    ppt,
                    engine,
                    *pTracks,
                    *pStripRecHits,
                    *pPixelRecHits,
                    *pDiamondRecHits,
                    *pStripRecHitsPerParticle,
                    *pPixelRecHitsPerParticle,
                    *pDiamondRecHitsPerParticle);
    }
  }

  if (produceScoringPlaneHits_)
    iEvent.put(std::move(pTracks));

  if (produceRecHits_) {
    iEvent.put(std::move(pStripRecHits));
    iEvent.put(std::move(pPixelRecHits));
    iEvent.put(std::move(pDiamondRecHits));

    iEvent.put(std::move(pStripRecHitsPerParticle));
    iEvent.put(std::move(pPixelRecHitsPerParticle));
    iEvent.put(std::move(pDiamondRecHitsPerParticle));
  }
}

//----------------------------------------------------------------------------------------------------

void PPSDirectProtonSimulation::processProton(
    const HepMC::GenVertex* in_vtx,
    const HepMC::GenParticle* in_trk,
    const CTPPSGeometry& geometry,
    const PPSPixelTopology& ppt,
    CLHEP::HepRandomEngine* rndEngine,
    std::vector<CTPPSLocalTrackLite>& out_tracks,
    edm::DetSetVector<TotemRPRecHit>& out_strip_hits,
    edm::DetSetVector<CTPPSPixelRecHit>& out_pixel_hits,
    edm::DetSetVector<CTPPSDiamondRecHit>& out_diamond_hits,
    std::map<int, edm::DetSetVector<TotemRPRecHit>>& out_strip_hits_per_particle,
    std::map<int, edm::DetSetVector<CTPPSPixelRecHit>>& out_pixel_hits_per_particle,
    std::map<int, edm::DetSetVector<CTPPSDiamondRecHit>>& out_diamond_hits_per_particle) const {
  /// xi is positive for diffractive protons, thus proton momentum p = (1-xi) * p_nom
  /// horizontal component of proton momentum: p_x = th_x * (1-xi) * p_nom

  std::stringstream ssLog;

  std::map<CTPPSDetId, DirectSimulator::Parameters> simulated_parameters;
  // transport the proton into each pot/scoring plane
  // vectors in CMS convention
  if (!direct_simulator_(
          {{in_vtx->position().x(), in_vtx->position().y(), in_vtx->position().z(), in_vtx->position().t()}},
          {{in_trk->momentum().x(), in_trk->momentum().y(), in_trk->momentum().z(), in_trk->momentum().t()}},
          simulated_parameters))
    return;
  if (produceScoringPlaneHits_)  // save scoring plane hit
    direct_simulator_.produceLiteTracks(simulated_parameters, out_tracks);

  if (produceRecHits_)  // stop if rec hits are not to be produced
    for (const auto& [rpId, scoring_parameters] : simulated_parameters) {
      // determine RP type
      const bool isTrackingRP =
                     rpId.subdetId() == CTPPSDetId::sdTrackingStrip || rpId.subdetId() == CTPPSDetId::sdTrackingPixel,
                 isTimingRP = rpId.subdetId() == CTPPSDetId::sdTimingDiamond;
      for (const auto& detIdInt : geometry.sensorsInRP(rpId)) {  // loop over all sensors in the RP
        const CTPPSDetId detId(detIdInt);

        // determine the track impact point (in global coordinates)
        // !! this assumes that local axes (1, 0, 0) and (0, 1, 0) describe the sensor surface
        const auto& detector_origin_global = geometry.localToGlobal(detId, CTPPSGeometry::Vector(0, 0, 0));
        const auto& detector_axis1_global =
            geometry.localToGlobal(detId, CTPPSGeometry::Vector(1, 0, 0)) - detector_origin_global;
        const auto& detector_axis2_global =
            geometry.localToGlobal(detId, CTPPSGeometry::Vector(0, 1, 0)) - detector_origin_global;

        const auto z_sign = -in_trk->momentum().z() / std::fabs(in_trk->momentum().z());

        TMatrixD A(3, 3);
        A(0, 0) = scoring_parameters.ax;
        A(0, 1) = -detector_axis1_global.x();
        A(0, 2) = -detector_axis2_global.x();
        A(1, 0) = scoring_parameters.ay;
        A(1, 1) = -detector_axis1_global.y();
        A(1, 2) = -detector_axis2_global.y();
        A(2, 0) = z_sign;
        A(2, 1) = -detector_axis1_global.z();
        A(2, 2) = -detector_axis2_global.z();

        TVectorD B(3);
        B(0) = detector_origin_global.x() - scoring_parameters.bx;
        B(1) = detector_origin_global.y() - scoring_parameters.by;
        B(2) = detector_origin_global.z() - scoring_parameters.z;

        const auto P = A.Invert() * B;

        const CTPPSGeometry::Vector hit_position_global(
            scoring_parameters.ax * P(0) + scoring_parameters.bx,
            scoring_parameters.ay * P(0) + scoring_parameters.by,
            z_sign * P(0) + scoring_parameters.z);                                     // hit in global coordinates
        auto hit_position_local = geometry.globalToLocal(detId, hit_position_global);  // hit in local coordinates

        if (verbosity_)
          ssLog << std::endl
                << "    de z = " << P(0) << " mm, "
                << "p1 = " << P(1) << " mm, "
                << "p2 = " << P(2) << " mm\n"
                << "    h_glo: "
                << "x = " << hit_position_global.x() << " mm, "
                << "y = " << hit_position_global.y() << " mm, "
                << "z = " << hit_position_global.z() << " mm\n"
                << "    h_loc: "
                << "c1 = " << hit_position_local.x() << " mm, "
                << "c2 = " << hit_position_local.y() << " mm, "
                << "c3 = " << hit_position_local.z() << " mm" << std::endl;

        if (((useTimingEfficiencyPerPlane_ && isTimingRP) || (useTrackingEfficiencyPerPlane_ && isTrackingRP)) &&
            efficiencyMapsPerPlane_.count(detId)) {  // apply per-plane efficiency
          const auto& efficiencies = efficiencyMapsPerPlane_.at(detId);
          if (const auto r_variable = CLHEP::RandFlat::shoot(rndEngine, 0., 1.),
              efficiency =
                  efficiencies->GetBinContent(efficiencies->FindBin(hit_position_global.x(), hit_position_global.y()));
              r_variable > efficiency) {
            if (verbosity_)
              ssLog << "    stop due to per-plane efficiency" << std::endl;
            continue;
          }
        }

        if (detId.subdetId() == CTPPSDetId::sdTrackingStrip) {  // strips
          auto u_coord = hit_position_local.x(), v_coord = hit_position_local.y();
          if (verbosity_ > 5)
            ssLog << "            u=" << u_coord << ", v=" << v_coord;

          if (checkIsHit_ &&
              !RPTopology::IsHit(u_coord, v_coord, insensitiveMarginStrips_)) {  // is it within detector?
            if (verbosity_ > 5)
              ssLog << " | no hit" << std::endl;
            continue;
          }

          if (roundToPitch_) {  // round the measurement
            double m = stripZeroPosition_ - v_coord;
            auto strip = static_cast<int>(floor(m / pitchStrips_ + 0.5));
            v_coord = stripZeroPosition_ - pitchStrips_ * strip;

            if (verbosity_ > 5)
              ssLog << " | strip=" << strip;
          }

          const auto sigma = pitchStrips_ * inv_sqrt_12_;

          if (verbosity_ > 5)
            ssLog << " | m=" << v_coord << ", sigma=" << sigma << std::endl;

          edm::DetSet<TotemRPRecHit>& hits = out_strip_hits.find_or_insert(detId);
          hits.emplace_back(v_coord, sigma);

          edm::DetSet<TotemRPRecHit>& hits_per_particle =
              out_strip_hits_per_particle[in_trk->barcode()].find_or_insert(detId);
          hits_per_particle.emplace_back(v_coord, sigma);
        } else if (detId.subdetId() == CTPPSDetId::sdTimingDiamond) {  // diamonds
          // check acceptance
          const auto* dg = geometry.sensor(detIdInt);
          const auto& diamondDimensions = dg->getDiamondDimensions();
          const auto x_half_width = diamondDimensions.xHalfWidth, y_half_width = diamondDimensions.yHalfWidth,
                     z_half_width = diamondDimensions.zHalfWidth;

          if (hit_position_local.x() < -x_half_width || hit_position_local.x() > +x_half_width ||
              hit_position_local.y() < -y_half_width || hit_position_local.y() > +y_half_width)
            continue;

          // timing information
          // calculate effective RP arrival time
          // effective time mimics the timing calibration -> effective times are distributed about 0
          // units:
          //    vertex: all components in mm
          //    c_light: in mm/ns
          //    time_eff: in ns
          const double time_eff = (in_vtx->position().t() + z_sign * in_vtx->position().z()) / CLHEP::c_light;
          const double time_resolution = CTPPSDiamondDetId(detIdInt).arm() == 0
                                             ? timeResolutionDiamonds45_->Eval(hit_position_global.x())
                                             : timeResolutionDiamonds56_->Eval(hit_position_global.x());

          const double t0 = time_eff + CLHEP::RandGauss::shoot(rndEngine, 0., time_resolution);
          const double tot = 1.23456;
          const double ch_t_precis = time_resolution;
          const int time_slice = 0;
          const bool multiHit = false;

          // build rec hit
          const auto diamond_rechit = CTPPSDiamondRecHit(detector_origin_global.x(),
                                                         2. * x_half_width,
                                                         detector_origin_global.y(),
                                                         2. * y_half_width,
                                                         detector_origin_global.z(),
                                                         2. * z_half_width,
                                                         t0,
                                                         tot,
                                                         ch_t_precis,
                                                         time_slice,
                                                         {},
                                                         multiHit);

          auto& hits = out_diamond_hits.find_or_insert(detId);
          hits.push_back(diamond_rechit);

          auto& hits_per_particle = out_diamond_hits_per_particle[in_trk->barcode()].find_or_insert(detId);
          hits_per_particle.push_back(diamond_rechit);
        } else if (detId.subdetId() == CTPPSDetId::sdTrackingPixel) {  // pixels
          if (verbosity_)
            ssLog << "    pixel plane " << CTPPSPixelDetId(detIdInt).plane() << ": local hit "
                  << "x = " << hit_position_local.x() << " mm, "
                  << "y = " << hit_position_local.y() << " mm, "
                  << "z = " << hit_position_local.z() << " mm" << std::endl;

          if (const bool module3By2 = geometry.sensor(detIdInt)->sensorType() != DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2;
              checkIsHit_ && !ppt.isPixelHit(hit_position_local.x(), hit_position_local.y(), module3By2))
            continue;

          if (roundToPitch_) {
            hit_position_local.SetX(pitchPixelsHor_ * floor(hit_position_local.x() / pitchPixelsHor_ + 0.5));
            hit_position_local.SetY(pitchPixelsVer_ * floor(hit_position_local.y() / pitchPixelsVer_ + 0.5));
          }

          if (verbosity_ > 5)
            ssLog << "            hit accepted: "
                  << "m1 = " << hit_position_local.x() << " mm, "
                  << "m2 = " << hit_position_local.y() << " mm" << std::endl;

          const auto pixel_local_point =
              LocalPoint(hit_position_local.x(), hit_position_local.y(), hit_position_local.z());

          edm::DetSet<CTPPSPixelRecHit>& hits = out_pixel_hits.find_or_insert(detId);
          hits.emplace_back(pixel_local_point, pixel_local_error_);

          edm::DetSet<CTPPSPixelRecHit>& hits_per_particle =
              out_pixel_hits_per_particle[in_trk->barcode()].find_or_insert(detId);
          hits_per_particle.emplace_back(pixel_local_point, pixel_local_error_);
        }
      }
    }

  if (verbosity_)
    edm::LogInfo("PPS") << ssLog.str();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PPSDirectProtonSimulation);
