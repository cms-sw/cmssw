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

#include "FWCore/Utilities/interface/ESInputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CondFormats/DataRecord/interface/PPSDirectSimulationDataRcd.h"
#include "CondFormats/PPSObjects/interface/PPSDirectSimulationData.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "Geometry/VeryForwardGeometry/interface/CTPPSPixelTopology.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <unordered_map>

#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TF1.h"
#include "TF2.h"
#include "TFile.h"
#include "CLHEP/Random/RandFlat.h"

//----------------------------------------------------------------------------------------------------

class CTPPSDirectProtonSimulation : public edm::stream::EDProducer<> {
public:
  explicit CTPPSDirectProtonSimulation(const edm::ParameterSet &);
  ~CTPPSDirectProtonSimulation() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  void processProton(const HepMC::GenVertex *in_vtx,
                     const HepMC::GenParticle *in_trk,
                     const CTPPSGeometry &geometry,
                     const LHCInfo &lhcInfo,
                     const CTPPSBeamParameters &beamParameters,
                     const LHCInterpolatedOpticalFunctionsSetCollection &opticalFunctions,
                     CLHEP::HepRandomEngine *rndEngine,

                     std::vector<CTPPSLocalTrackLite> &out_tracks,

                     edm::DetSetVector<TotemRPRecHit> &out_strip_hits,
                     edm::DetSetVector<CTPPSPixelRecHit> &out_pixel_hits,
                     edm::DetSetVector<CTPPSDiamondRecHit> &out_diamond_hits,

                     std::map<int, edm::DetSetVector<TotemRPRecHit>> &out_strip_hits_per_particle,
                     std::map<int, edm::DetSetVector<CTPPSPixelRecHit>> &out_pixel_hits_per_particle,
                     std::map<int, edm::DetSetVector<CTPPSDiamondRecHit>> &out_diamond_hits_per_particle) const;

  // ------------ config file parameters ------------

  // conditions
  edm::ESGetToken<LHCInfo, LHCInfoRcd> tokenLHCInfo_;
  edm::ESGetToken<CTPPSBeamParameters, CTPPSBeamParametersRcd> tokenBeamParameters_;
  edm::ESGetToken<LHCInterpolatedOpticalFunctionsSetCollection, CTPPSInterpolatedOpticsRcd> tokenOpticalFunctions_;
  edm::ESGetToken<CTPPSGeometry, VeryForwardMisalignedGeometryRecord> tokenGeometry_;
  edm::ESGetToken<PPSDirectSimulationData, PPSDirectSimulationDataRcd> tokenDirectSimuData_;

  edm::ESWatcher<PPSDirectSimulationDataRcd> directSimuDataRcdWatcher_;

  // input
  edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;

  // flags what output to be produced
  bool produceScoringPlaneHits_;
  bool produceRecHits_;

  // settings of LHC aperture limitations (high xi)
  bool useEmpiricalApertures_;
  std::unique_ptr<TF2> empiricalAperture45_;
  std::unique_ptr<TF2> empiricalAperture56_;

  // timing-RP efficiency
  bool useTimingRPEfficiency_;
  std::unique_ptr<TH2F> effTimeMap45_;
  std::unique_ptr<TH2F> effTimeMap56_;

  bool produceHitsRelativeToBeam_;
  bool roundToPitch_;
  bool checkIsHit_;

  double pitchStrips_;              ///< strip pitch in mm
  double insensitiveMarginStrips_;  ///< size of insensitive margin at sensor's edge facing the beam, in mm

  double pitchPixelsHor_;
  double pitchPixelsVer_;

  unsigned int verbosity_;

  std::unique_ptr<TF1> timeResolutionDiamonds45_, timeResolutionDiamonds56_;

  // ------------ internal parameters ------------

  /// internal variable: v position of strip 0, in mm
  double stripZeroPosition_;
};

//----------------------------------------------------------------------------------------------------

CTPPSDirectProtonSimulation::CTPPSDirectProtonSimulation(const edm::ParameterSet &iConfig)
    : tokenLHCInfo_(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("lhcInfoLabel")})),
      tokenBeamParameters_(esConsumes()),
      tokenOpticalFunctions_(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("opticsLabel")})),
      tokenGeometry_(esConsumes()),
      tokenDirectSimuData_(esConsumes()),

      hepMCToken_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("hepMCTag"))),

      produceScoringPlaneHits_(iConfig.getParameter<bool>("produceScoringPlaneHits")),
      produceRecHits_(iConfig.getParameter<bool>("produceRecHits")),

      useEmpiricalApertures_(iConfig.getParameter<bool>("useEmpiricalApertures")),
      useTimingRPEfficiency_(iConfig.getParameter<bool>("useTimingRPEfficiency")),

      produceHitsRelativeToBeam_(iConfig.getParameter<bool>("produceHitsRelativeToBeam")),
      roundToPitch_(iConfig.getParameter<bool>("roundToPitch")),
      checkIsHit_(iConfig.getParameter<bool>("checkIsHit")),

      pitchStrips_(iConfig.getParameter<double>("pitchStrips")),
      insensitiveMarginStrips_(iConfig.getParameter<double>("insensitiveMarginStrips")),

      pitchPixelsHor_(iConfig.getParameter<double>("pitchPixelsHor")),
      pitchPixelsVer_(iConfig.getParameter<double>("pitchPixelsVer")),

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

  // v position of strip 0
  stripZeroPosition_ = RPTopology::last_strip_to_border_dist_ + (RPTopology::no_of_strips_ - 1) * RPTopology::pitch_ -
                       RPTopology::y_width_ / 2.;
}

//----------------------------------------------------------------------------------------------------

void CTPPSDirectProtonSimulation::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<unsigned int>("verbosity", 0);

  desc.add<std::string>("lhcInfoLabel", "")->setComment("label of the LHCInfo record");
  desc.add<std::string>("opticsLabel", "")->setComment("label of the optics records");

  desc.add<edm::InputTag>("hepMCTag", edm::InputTag("generator", "unsmeared"));

  desc.add<bool>("produceScoringPlaneHits", true);
  desc.add<bool>("produceRecHits", true);

  desc.add<bool>("useEmpiricalApertures", true);
  desc.add<bool>("useTimingRPEfficiency", false);
  desc.add<bool>("produceHitsRelativeToBeam", true);
  desc.add<bool>("roundToPitch", true);
  desc.add<bool>("checkIsHit", true);
  desc.add<double>("pitchStrips", 66.e-3);              // in mm
  desc.add<double>("insensitiveMarginStrips", 34.e-3);  // in mm

  desc.add<double>("pitchPixelsHor", 100.e-3);
  desc.add<double>("pitchPixelsVer", 150.e-3);

  descriptions.add("ctppsDirectProtonSimulation", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSDirectProtonSimulation::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get input
  edm::Handle<edm::HepMCProduct> hepmc_prod;
  iEvent.getByToken(hepMCToken_, hepmc_prod);

  // get conditions
  auto const &lhcInfo = iSetup.getData(tokenLHCInfo_);
  auto const &beamParameters = iSetup.getData(tokenBeamParameters_);
  auto const &opticalFunctions = iSetup.getData(tokenOpticalFunctions_);
  auto const &geometry = iSetup.getData(tokenGeometry_);
  auto const &directSimuData = iSetup.getData(tokenDirectSimuData_);

  if (directSimuDataRcdWatcher_.check(iSetup)) {
    timeResolutionDiamonds45_ =
        std::make_unique<TF1>(TF1("timeResolutionDiamonds45", directSimuData.getTimeResolutionDiamonds45().c_str()));
    timeResolutionDiamonds56_ =
        std::make_unique<TF1>(TF1("timeResolutionDiamonds56", directSimuData.getTimeResolutionDiamonds56().c_str()));

    empiricalAperture45_ =
        std::make_unique<TF2>(TF2("empiricalAperture45", directSimuData.getEmpiricalAperture45().c_str()));
    empiricalAperture56_ =
        std::make_unique<TF2>(TF2("empiricalAperture56", directSimuData.getEmpiricalAperture56().c_str()));

    if (useTimingRPEfficiency_) {
      edm::FileInPath fip(directSimuData.getEffTimePath().c_str());
      std::unique_ptr<TFile> f_in(TFile::Open(fip.fullPath().c_str()));
      effTimeMap45_ = std::unique_ptr<TH2F>((TH2F *)f_in->Get(directSimuData.getEffTimeObject45().c_str()));
      effTimeMap56_ = std::unique_ptr<TH2F>((TH2F *)f_in->Get(directSimuData.getEffTimeObject56().c_str()));
      effTimeMap45_->SetDirectory(nullptr);
      effTimeMap56_->SetDirectory(nullptr);
      f_in->Close();
    }
  }

  // prepare outputs
  std::unique_ptr<edm::DetSetVector<TotemRPRecHit>> pStripRecHits(new edm::DetSetVector<TotemRPRecHit>());
  std::unique_ptr<edm::DetSetVector<CTPPSDiamondRecHit>> pDiamondRecHits(new edm::DetSetVector<CTPPSDiamondRecHit>());
  std::unique_ptr<edm::DetSetVector<CTPPSPixelRecHit>> pPixelRecHits(new edm::DetSetVector<CTPPSPixelRecHit>());

  auto pStripRecHitsPerParticle = std::make_unique<std::map<int, edm::DetSetVector<TotemRPRecHit>>>();
  auto pDiamondRecHitsPerParticle = std::make_unique<std::map<int, edm::DetSetVector<CTPPSDiamondRecHit>>>();
  auto pPixelRecHitsPerParticle = std::make_unique<std::map<int, edm::DetSetVector<CTPPSPixelRecHit>>>();

  std::unique_ptr<std::vector<CTPPSLocalTrackLite>> pTracks(new std::vector<CTPPSLocalTrackLite>());

  // get random engine
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine *engine = &rng->getEngine(iEvent.streamID());

  // loop over event vertices
  auto evt = hepmc_prod->GetEvent();
  for (auto it_vtx = evt->vertices_begin(); it_vtx != evt->vertices_end(); ++it_vtx) {
    auto vtx = *(it_vtx);

    // loop over outgoing particles
    for (auto it_part = vtx->particles_out_const_begin(); it_part != vtx->particles_out_const_end(); ++it_part) {
      auto part = *(it_part);

      // accept only stable protons
      if (part->pdg_id() != 2212)
        continue;

      if (part->status() != 1 && part->status() < 83)
        continue;

      processProton(vtx,
                    part,
                    geometry,
                    lhcInfo,
                    beamParameters,
                    opticalFunctions,
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

void CTPPSDirectProtonSimulation::processProton(
    const HepMC::GenVertex *in_vtx,
    const HepMC::GenParticle *in_trk,
    const CTPPSGeometry &geometry,
    const LHCInfo &lhcInfo,
    const CTPPSBeamParameters &beamParameters,
    const LHCInterpolatedOpticalFunctionsSetCollection &opticalFunctions,
    CLHEP::HepRandomEngine *rndEngine,
    std::vector<CTPPSLocalTrackLite> &out_tracks,
    edm::DetSetVector<TotemRPRecHit> &out_strip_hits,
    edm::DetSetVector<CTPPSPixelRecHit> &out_pixel_hits,
    edm::DetSetVector<CTPPSDiamondRecHit> &out_diamond_hits,
    std::map<int, edm::DetSetVector<TotemRPRecHit>> &out_strip_hits_per_particle,
    std::map<int, edm::DetSetVector<CTPPSPixelRecHit>> &out_pixel_hits_per_particle,
    std::map<int, edm::DetSetVector<CTPPSDiamondRecHit>> &out_diamond_hits_per_particle) const {
  /// xi is positive for diffractive protons, thus proton momentum p = (1-xi) * p_nom
  /// horizontal component of proton momentum: p_x = th_x * (1-xi) * p_nom

  std::stringstream ssLog;

  // vectors in CMS convention
  const HepMC::FourVector &vtx_cms = in_vtx->position();  // in mm
  const HepMC::FourVector &mom_cms = in_trk->momentum();

  // transformation to LHC/TOTEM convention
  HepMC::FourVector vtx_lhc(-vtx_cms.x(), vtx_cms.y(), -vtx_cms.z(), vtx_cms.t());
  HepMC::FourVector mom_lhc(-mom_cms.x(), mom_cms.y(), -mom_cms.z(), mom_cms.t());

  // determine the LHC arm and related parameters
  unsigned int arm = 3;
  double z_sign;
  double beamMomentum = 0.;
  double xangle = 0.;
  const std::unique_ptr<TF2> *empiricalAperture;
  if (mom_lhc.z() < 0)  // sector 45
  {
    arm = 0;
    z_sign = -1;
    beamMomentum = beamParameters.getBeamMom45();
    xangle = beamParameters.getHalfXangleX45();
    empiricalAperture = &empiricalAperture45_;
  } else {  // sector 56
    arm = 1;
    z_sign = +1;
    beamMomentum = beamParameters.getBeamMom56();
    xangle = beamParameters.getHalfXangleX56();
    empiricalAperture = &empiricalAperture56_;
  }

  // calculate effective RP arrival time
  // effective time mimics the timing calibration -> effective times are distributed about 0
  // units:
  //    vertex: all components in mm
  //    c_light: in mm/ns
  //    time_eff: in ns
  const double time_eff = (vtx_lhc.t() - z_sign * vtx_lhc.z()) / CLHEP::c_light;

  // calculate kinematics for optics parametrisation
  const double p = mom_lhc.rho();
  const double xi = 1. - p / beamMomentum;
  const double th_x_phys = mom_lhc.x() / p;
  const double th_y_phys = mom_lhc.y() / p;
  const double vtx_lhc_eff_x = vtx_lhc.x() - vtx_lhc.z() * (mom_lhc.x() / mom_lhc.z() + xangle);
  const double vtx_lhc_eff_y = vtx_lhc.y() - vtx_lhc.z() * (mom_lhc.y() / mom_lhc.z());

  if (verbosity_) {
    ssLog << "simu: xi = " << xi << ", th_x_phys = " << th_x_phys << ", th_y_phys = " << th_y_phys
          << ", vtx_lhc_eff_x = " << vtx_lhc_eff_x << ", vtx_lhc_eff_y = " << vtx_lhc_eff_y << std::endl;
  }

  // check empirical aperture
  if (useEmpiricalApertures_) {
    const auto &xangle = lhcInfo.crossingAngle();
    (*empiricalAperture)->SetParameter("xi", xi);
    (*empiricalAperture)->SetParameter("xangle", xangle);
    const double th_x_th = (*empiricalAperture)->EvalPar(nullptr);

    if (th_x_th > th_x_phys) {
      if (verbosity_) {
        ssLog << "stop because of empirical appertures";
        edm::LogInfo("PPS") << ssLog.str();
      }

      return;
    }
  }

  // transport the proton into each pot/scoring plane
  for (const auto &ofp : opticalFunctions) {
    CTPPSDetId rpId(ofp.first);
    const unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    // first check the arm
    if (rpId.arm() != arm)
      continue;

    if (verbosity_)
      ssLog << "  RP " << rpDecId << std::endl;

    // transport proton
    LHCInterpolatedOpticalFunctionsSet::Kinematics k_in = {
        vtx_lhc_eff_x * 1E-1, th_x_phys, vtx_lhc_eff_y * 1E-1, th_y_phys, xi};  // conversions: mm -> cm
    LHCInterpolatedOpticalFunctionsSet::Kinematics k_out;
    ofp.second.transport(k_in, k_out, true);

    double b_x = k_out.x * 1E1, b_y = k_out.y * 1E1;  // conversions: cm -> mm
    double a_x = k_out.th_x, a_y = k_out.th_y;

    // if needed, subtract beam position and angle
    if (produceHitsRelativeToBeam_) {
      // determine beam position
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_be_in = {0., 0., 0., 0., 0.};
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_be_out;
      ofp.second.transport(k_be_in, k_be_out, true);

      a_x -= k_be_out.th_x;
      a_y -= k_be_out.th_y;
      b_x -= k_be_out.x * 1E1;
      b_y -= k_be_out.y * 1E1;  // conversions: cm -> mm
    }

    const double z_scoringPlane = ofp.second.getScoringPlaneZ() * 1E1;  // conversion: cm --> mm

    if (verbosity_) {
      ssLog << "    proton transported: a_x = " << a_x << " rad, a_y = " << a_y << " rad, b_x = " << b_x
            << " mm, b_y = " << b_y << " mm, z = " << z_scoringPlane << " mm" << std::endl;
    }

    // save scoring plane hit
    if (produceScoringPlaneHits_)
      out_tracks.emplace_back(
          rpId.rawId(), b_x, 0., b_y, 0., 0., 0., 0., 0., 0., CTPPSpixelLocalTrackReconstructionInfo::invalid, 0, 0., 0.);

    // stop if rec hits are not to be produced
    if (!produceRecHits_)
      continue;

    // loop over all sensors in the RP
    for (const auto &detIdInt : geometry.sensorsInRP(rpId)) {
      CTPPSDetId detId(detIdInt);

      // determine the track impact point (in global coordinates)
      // !! this assumes that local axes (1, 0, 0) and (0, 1, 0) describe the sensor surface
      const auto &gl_o = geometry.localToGlobal(detId, CTPPSGeometry::Vector(0, 0, 0));
      const auto &gl_a1 = geometry.localToGlobal(detId, CTPPSGeometry::Vector(1, 0, 0)) - gl_o;
      const auto &gl_a2 = geometry.localToGlobal(detId, CTPPSGeometry::Vector(0, 1, 0)) - gl_o;

      double gl_o_z = gl_o.z();
      if (detId.subdetId() == CTPPSDetId::sdTimingDiamond)
        gl_o_z = -gl_o_z;  // fix bug in diamond geometry

      TMatrixD A(3, 3);
      TVectorD B(3);
      A(0, 0) = a_x;
      A(0, 1) = -gl_a1.x();
      A(0, 2) = -gl_a2.x();
      B(0) = gl_o.x() - b_x;
      A(1, 0) = a_y;
      A(1, 1) = -gl_a1.y();
      A(1, 2) = -gl_a2.y();
      B(1) = gl_o.y() - b_y;
      A(2, 0) = z_sign;
      A(2, 1) = -gl_a1.z();
      A(2, 2) = -gl_a2.z();
      B(2) = gl_o_z - z_scoringPlane;
      TMatrixD Ai(3, 3);
      Ai = A.Invert();
      TVectorD P(3);
      P = Ai * B;

      double ze = P(0);
      const CTPPSGeometry::Vector h_glo(a_x * ze + b_x, a_y * ze + b_y, z_sign * ze + z_scoringPlane);

      // hit in local coordinates
      auto h_loc = geometry.globalToLocal(detId, h_glo);

      if (verbosity_) {
        ssLog << std::endl
              << "    de z = " << P(0) << " mm, p1 = " << P(1) << " mm, p2 = " << P(2) << " mm" << std::endl
              << "    h_glo: x = " << h_glo.x() << " mm, y = " << h_glo.y() << " mm, z = " << h_glo.z() << " mm"
              << std::endl
              << "    h_loc: c1 = " << h_loc.x() << " mm, c2 = " << h_loc.y() << " mm, c3 = " << h_loc.z() << " mm"
              << std::endl;
      }

      // strips
      if (detId.subdetId() == CTPPSDetId::sdTrackingStrip) {
        double u = h_loc.x();
        double v = h_loc.y();

        if (verbosity_ > 5)
          ssLog << "            u=" << u << ", v=" << v;

        // is it within detector?
        if (checkIsHit_ && !RPTopology::IsHit(u, v, insensitiveMarginStrips_)) {
          if (verbosity_ > 5)
            ssLog << " | no hit" << std::endl;
          continue;
        }

        // round the measurement
        if (roundToPitch_) {
          double m = stripZeroPosition_ - v;
          signed int strip = (int)floor(m / pitchStrips_ + 0.5);

          v = stripZeroPosition_ - pitchStrips_ * strip;

          if (verbosity_ > 5)
            ssLog << " | strip=" << strip;
        }

        double sigma = pitchStrips_ / sqrt(12.);

        if (verbosity_ > 5)
          ssLog << " | m=" << v << ", sigma=" << sigma << std::endl;

        edm::DetSet<TotemRPRecHit> &hits = out_strip_hits.find_or_insert(detId);
        hits.push_back(TotemRPRecHit(v, sigma));

        edm::DetSet<TotemRPRecHit> &hits_per_particle =
            out_strip_hits_per_particle[in_trk->barcode()].find_or_insert(detId);
        hits_per_particle.push_back(TotemRPRecHit(v, sigma));
      }

      // diamonds
      if (detId.subdetId() == CTPPSDetId::sdTimingDiamond) {
        CTPPSDiamondDetId diamondDetId(detIdInt);

        //efficiency
        if (useTimingRPEfficiency_) {
          TH2F *effMap = (diamondDetId.arm() == 0) ? effTimeMap45_.get() : effTimeMap56_.get();
          if (CLHEP::RandFlat::shoot(rndEngine, 0., 1.) > effMap->GetBinContent(effMap->FindBin(h_glo.x(), h_glo.y())))
            continue;
        }

        // check acceptance
        const auto *dg = geometry.sensor(detIdInt);
        const auto &diamondDimensions = dg->getDiamondDimensions();
        const auto x_half_width = diamondDimensions.xHalfWidth;
        const auto y_half_width = diamondDimensions.yHalfWidth;
        const auto z_half_width = diamondDimensions.zHalfWidth;

        if (h_loc.x() < -x_half_width || h_loc.x() > +x_half_width || h_loc.y() < -y_half_width ||
            h_loc.y() > +y_half_width)
          continue;

        // timing information
        const double time_resolution = (diamondDetId.arm() == 0) ? timeResolutionDiamonds45_->Eval(h_glo.x())
                                                                 : timeResolutionDiamonds56_->Eval(h_glo.x());

        const double t0 = time_eff + CLHEP::RandGauss::shoot(rndEngine, 0., time_resolution);
        const double tot = 1.23456;
        const double ch_t_precis = time_resolution;
        const int time_slice = 0;

        // build rec hit
        const bool multiHit = false;

        CTPPSDiamondRecHit rc(gl_o.x(),
                              2. * x_half_width,
                              gl_o.y(),
                              2. * y_half_width,
                              gl_o_z,
                              2. * z_half_width,
                              t0,
                              tot,
                              ch_t_precis,
                              time_slice,
                              HPTDCErrorFlags(),
                              multiHit);

        edm::DetSet<CTPPSDiamondRecHit> &hits = out_diamond_hits.find_or_insert(detId);
        hits.push_back(rc);

        edm::DetSet<CTPPSDiamondRecHit> &hits_per_particle =
            out_diamond_hits_per_particle[in_trk->barcode()].find_or_insert(detId);
        hits_per_particle.push_back(rc);
      }

      // pixels
      if (detId.subdetId() == CTPPSDetId::sdTrackingPixel) {
        if (verbosity_) {
          CTPPSPixelDetId pixelDetId(detIdInt);
          ssLog << "    pixel plane " << pixelDetId.plane() << ": local hit x = " << h_loc.x()
                << " mm, y = " << h_loc.y() << " mm, z = " << h_loc.z() << " mm" << std::endl;
        }

        bool module3By2 = (geometry.sensor(detIdInt)->sensorType() != DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2);
        if (checkIsHit_ && !CTPPSPixelTopology::isPixelHit(h_loc.x(), h_loc.y(), module3By2))
          continue;

        if (roundToPitch_) {
          h_loc.SetX(pitchPixelsHor_ * floor(h_loc.x() / pitchPixelsHor_ + 0.5));
          h_loc.SetY(pitchPixelsVer_ * floor(h_loc.y() / pitchPixelsVer_ + 0.5));
        }

        if (verbosity_ > 5)
          ssLog << "            hit accepted: m1 = " << h_loc.x() << " mm, m2 = " << h_loc.y() << " mm" << std::endl;

        const double sigmaHor = pitchPixelsHor_ / sqrt(12.);
        const double sigmaVer = pitchPixelsVer_ / sqrt(12.);

        const LocalPoint lp(h_loc.x(), h_loc.y(), h_loc.z());
        const LocalError le(sigmaHor, 0., sigmaVer);

        edm::DetSet<CTPPSPixelRecHit> &hits = out_pixel_hits.find_or_insert(detId);
        hits.push_back(CTPPSPixelRecHit(lp, le));

        edm::DetSet<CTPPSPixelRecHit> &hits_per_particle =
            out_pixel_hits_per_particle[in_trk->barcode()].find_or_insert(detId);
        hits_per_particle.push_back(CTPPSPixelRecHit(lp, le));
      }
    }
  }

  if (verbosity_)
    edm::LogInfo("PPS") << ssLog.str();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSDirectProtonSimulation);