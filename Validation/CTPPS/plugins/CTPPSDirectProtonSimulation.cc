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

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"

#include <unordered_map>

#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"

//----------------------------------------------------------------------------------------------------

class CTPPSDirectProtonSimulation : public edm::stream::EDProducer<> {
public:
  explicit CTPPSDirectProtonSimulation(const edm::ParameterSet&);
  ~CTPPSDirectProtonSimulation() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void processProton(const HepMC::GenVertex* in_vtx,
                     const HepMC::GenParticle* in_trk,
                     const CTPPSGeometry& geometry,
                     const CTPPSBeamParameters& beamParameters,
                     const LHCInterpolatedOpticalFunctionsSetCollection& opticalFunctions,
                     std::vector<CTPPSLocalTrackLite>& out_tracks,
                     edm::DetSetVector<TotemRPRecHit>& out_strip_hits,
                     edm::DetSetVector<CTPPSPixelRecHit>& out_pixel_hits,
                     edm::DetSetVector<CTPPSDiamondRecHit>& out_diamond_hits) const;

  static bool isPixelHit(float xLocalCoordinate, float yLocalCoordinate, bool is3x2 = true) {
    float tmpXlocalCoordinate = xLocalCoordinate + (79 * 0.1 + 0.2);
    float tmpYlocalCoordinate = yLocalCoordinate + (0.15 * 51 + 0.3 * 2 + 0.15 * 25);

    if (tmpXlocalCoordinate < 0)
      return false;
    if (tmpYlocalCoordinate < 0)
      return false;
    int xModuleSize = 0.1 * 79 + 0.2 * 2 + 0.1 * 79;  // mm - 100 um pitch direction
    int yModuleSize;                                  // mm - 150 um pitch direction
    if (is3x2)
      yModuleSize = 0.15 * 51 + 0.3 * 2 + 0.15 * 50 + 0.3 * 2 + 0.15 * 51;
    else
      yModuleSize = 0.15 * 51 + 0.3 * 2 + 0.15 * 51;
    if (tmpXlocalCoordinate > xModuleSize)
      return false;
    if (tmpYlocalCoordinate > yModuleSize)
      return false;
    return true;
  }

  // ------------ config file parameters ------------

  /// input tag
  edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;

  /// flags what output to be produced
  bool produceScoringPlaneHits_;
  bool produceRecHits_;

  /// simulation parameters
  bool checkApertures_;

  bool useEmpiricalApertures_;
  double empiricalAperture45_xi0_, empiricalAperture45_a_;
  double empiricalAperture56_xi0_, empiricalAperture56_a_;

  bool produceHitsRelativeToBeam_;
  bool roundToPitch_;
  bool checkIsHit_;

  double pitchStrips_;              ///< strip pitch in mm
  double insensitiveMarginStrips_;  ///< size of insensitive margin at sensor's edge facing the beam, in mm

  double pitchPixelsHor_;
  double pitchPixelsVer_;

  unsigned int verbosity_;

  // ------------ internal parameters ------------

  /// internal variable: v position of strip 0, in mm
  double stripZeroPosition_;
};

//----------------------------------------------------------------------------------------------------

CTPPSDirectProtonSimulation::CTPPSDirectProtonSimulation(const edm::ParameterSet& iConfig)
    : hepMCToken_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("hepMCTag"))),

      produceScoringPlaneHits_(iConfig.getParameter<bool>("produceScoringPlaneHits")),
      produceRecHits_(iConfig.getParameter<bool>("produceRecHits")),

      useEmpiricalApertures_(iConfig.getParameter<bool>("useEmpiricalApertures")),
      empiricalAperture45_xi0_(iConfig.getParameter<double>("empiricalAperture45_xi0")),
      empiricalAperture45_a_(iConfig.getParameter<double>("empiricalAperture45_a")),
      empiricalAperture56_xi0_(iConfig.getParameter<double>("empiricalAperture56_xi0")),
      empiricalAperture56_a_(iConfig.getParameter<double>("empiricalAperture56_a")),

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
  }

  // v position of strip 0
  stripZeroPosition_ = RPTopology::last_strip_to_border_dist_ + (RPTopology::no_of_strips_ - 1) * RPTopology::pitch_ -
                       RPTopology::y_width_ / 2.;
}

//----------------------------------------------------------------------------------------------------

void CTPPSDirectProtonSimulation::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get input
  edm::Handle<edm::HepMCProduct> hepmc_prod;
  iEvent.getByToken(hepMCToken_, hepmc_prod);

  // get conditions
  edm::ESHandle<CTPPSBeamParameters> hBeamParameters;
  iSetup.get<CTPPSBeamParametersRcd>().get(hBeamParameters);

  edm::ESHandle<LHCInterpolatedOpticalFunctionsSetCollection> hOpticalFunctions;
  iSetup.get<CTPPSInterpolatedOpticsRcd>().get(hOpticalFunctions);

  edm::ESHandle<CTPPSGeometry> geometry;
  iSetup.get<VeryForwardMisalignedGeometryRecord>().get(geometry);

  // prepare outputs
  std::unique_ptr<edm::DetSetVector<TotemRPRecHit>> pStripRecHits(new edm::DetSetVector<TotemRPRecHit>());
  std::unique_ptr<edm::DetSetVector<CTPPSDiamondRecHit>> pDiamondRecHits(new edm::DetSetVector<CTPPSDiamondRecHit>());
  std::unique_ptr<edm::DetSetVector<CTPPSPixelRecHit>> pPixelRecHits(new edm::DetSetVector<CTPPSPixelRecHit>());

  std::unique_ptr<std::vector<CTPPSLocalTrackLite>> pTracks(new std::vector<CTPPSLocalTrackLite>());

  // loop over event vertices
  auto evt = new HepMC::GenEvent(*hepmc_prod->GetEvent());
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
                    *geometry,
                    *hBeamParameters,
                    *hOpticalFunctions,
                    *pTracks,
                    *pStripRecHits,
                    *pPixelRecHits,
                    *pDiamondRecHits);
    }
  }

  if (produceScoringPlaneHits_)
    iEvent.put(std::move(pTracks));

  if (produceRecHits_) {
    iEvent.put(std::move(pStripRecHits));
    iEvent.put(std::move(pPixelRecHits));
    iEvent.put(std::move(pDiamondRecHits));
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSDirectProtonSimulation::processProton(const HepMC::GenVertex* in_vtx,
                                                const HepMC::GenParticle* in_trk,
                                                const CTPPSGeometry& geometry,
                                                const CTPPSBeamParameters& beamParameters,
                                                const LHCInterpolatedOpticalFunctionsSetCollection& opticalFunctions,
                                                std::vector<CTPPSLocalTrackLite>& out_tracks,
                                                edm::DetSetVector<TotemRPRecHit>& out_strip_hits,
                                                edm::DetSetVector<CTPPSPixelRecHit>& out_pixel_hits,
                                                edm::DetSetVector<CTPPSDiamondRecHit>& out_diamond_hits) const {
  /// xi is positive for diffractive protons, thus proton momentum p = (1-xi) * p_nom
  /// horizontal component of proton momentum: p_x = th_x * (1-xi) * p_nom

  std::stringstream ssLog;

  // vectors in CMS convention
  const HepMC::FourVector& vtx_cms = in_vtx->position();  // in mm
  const HepMC::FourVector& mom_cms = in_trk->momentum();

  // transformation to LHC/TOTEM convention
  HepMC::FourVector vtx_lhc(-vtx_cms.x(), vtx_cms.y(), -vtx_cms.z(), vtx_cms.t());
  HepMC::FourVector mom_lhc(-mom_cms.x(), mom_cms.y(), -mom_cms.z(), mom_cms.t());

  // determine the LHC arm and related parameters
  unsigned int arm = 3;
  double z_sign;
  double beamMomentum = 0.;
  double xangle = 0.;
  double empiricalAperture_xi0, empiricalAperture_a;

  if (mom_lhc.z() < 0)  // sector 45
  {
    arm = 0;
    z_sign = -1;
    beamMomentum = beamParameters.getBeamMom45();
    xangle = beamParameters.getHalfXangleX45();
    empiricalAperture_xi0 = empiricalAperture45_xi0_;
    empiricalAperture_a = empiricalAperture45_a_;
  } else {  // sector 56
    arm = 1;
    z_sign = +1;
    beamMomentum = beamParameters.getBeamMom56();
    xangle = beamParameters.getHalfXangleX56();
    empiricalAperture_xi0 = empiricalAperture56_xi0_;
    empiricalAperture_a = empiricalAperture56_a_;
  }

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
    const double xi_th = empiricalAperture_xi0 + th_x_phys * empiricalAperture_a;
    if (xi > xi_th) {
      if (verbosity_) {
        ssLog << "stop because of empirical appertures";
        edm::LogInfo("CTPPSDirectProtonSimulation") << ssLog.str();
      }

      return;
    }
  }

  // transport the proton into each pot/scoring plane
  for (const auto& ofp : opticalFunctions) {
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
    for (const auto& detIdInt : geometry.getSensorsInRP(rpId)) {
      CTPPSDetId detId(detIdInt);

      // determine the track impact point (in global coordinates)
      // !! this assumes that local axes (1, 0, 0) and (0, 1, 0) describe the sensor surface
      CLHEP::Hep3Vector gl_o = geometry.localToGlobal(detId, CLHEP::Hep3Vector(0, 0, 0));
      CLHEP::Hep3Vector gl_a1 = geometry.localToGlobal(detId, CLHEP::Hep3Vector(1, 0, 0)) - gl_o;
      CLHEP::Hep3Vector gl_a2 = geometry.localToGlobal(detId, CLHEP::Hep3Vector(0, 1, 0)) - gl_o;

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
      B(2) = gl_o.z() - z_scoringPlane;
      TMatrixD Ai(3, 3);
      Ai = A.Invert();
      TVectorD P(3);
      P = Ai * B;

      double ze = P(0);
      CLHEP::Hep3Vector h_glo(a_x * ze + b_x, a_y * ze + b_y, z_sign * ze + z_scoringPlane);

      // hit in local coordinates
      CLHEP::Hep3Vector h_loc = geometry.globalToLocal(detId, h_glo);

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

        edm::DetSet<TotemRPRecHit>& hits = out_strip_hits.find_or_insert(detId);
        hits.push_back(TotemRPRecHit(v, sigma));
      }

      // diamonds
      if (detId.subdetId() == CTPPSDetId::sdTimingDiamond) {
        throw cms::Exception("CTPPSDirectProtonSimulation") << "Diamonds are not yet supported.";
      }

      // pixels
      if (detId.subdetId() == CTPPSDetId::sdTrackingPixel) {
        if (verbosity_) {
          CTPPSPixelDetId pixelDetId(detIdInt);
          ssLog << "    pixel plane " << pixelDetId.plane() << ": local hit x = " << h_loc.x()
                << " mm, y = " << h_loc.y() << " mm, z = " << h_loc.z() << " mm" << std::endl;
        }

        if (checkIsHit_ && !isPixelHit(h_loc.x(), h_loc.y()))
          continue;

        if (roundToPitch_) {
          h_loc.setX(pitchPixelsHor_ * floor(h_loc.x() / pitchPixelsHor_ + 0.5));
          h_loc.setY(pitchPixelsVer_ * floor(h_loc.y() / pitchPixelsVer_ + 0.5));
        }

        if (verbosity_ > 5)
          ssLog << "            hit accepted: m1 = " << h_loc.x() << " mm, m2 = " << h_loc.y() << " mm" << std::endl;

        const double sigmaHor = pitchPixelsHor_ / sqrt(12.);
        const double sigmaVer = pitchPixelsVer_ / sqrt(12.);

        const LocalPoint lp(h_loc.x(), h_loc.y(), h_loc.z());
        const LocalError le(sigmaHor, 0., sigmaVer);

        edm::DetSet<CTPPSPixelRecHit>& hits = out_pixel_hits.find_or_insert(detId);
        hits.push_back(CTPPSPixelRecHit(lp, le));
      }
    }
  }

  if (verbosity_)
    edm::LogInfo("CTPPSDirectProtonSimulation") << ssLog.str();
}

//----------------------------------------------------------------------------------------------------

void CTPPSDirectProtonSimulation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<unsigned int>("verbosity", 0);
  desc.add<edm::InputTag>("hepMCTag", edm::InputTag("generator", "unsmeared"));
  desc.add<bool>("produceScoringPlaneHits", true);
  desc.add<bool>("produceRecHits", true);
  desc.add<bool>("useEmpiricalApertures", false);
  desc.add<double>("empiricalAperture45_xi0", 0.);
  desc.add<double>("empiricalAperture45_a", 0.);
  desc.add<double>("empiricalAperture56_xi0", 0.);
  desc.add<double>("empiricalAperture56_a", 0.);
  desc.add<bool>("produceHitsRelativeToBeam", false);
  desc.add<bool>("roundToPitch", true);
  desc.add<bool>("checkIsHit", true);
  desc.add<double>("pitchStrips", 66.e-3);              // in mm
  desc.add<double>("insensitiveMarginStrips", 34.e-3);  // in mm
  desc.add<double>("pitchPixelsHor", 100.e-3)->setComment("x in local coordinates, in mm");
  desc.add<double>("pitchPixelsVer", 150.e-3)->setComment("y in local coordinates, in mm");

  descriptions.add("ctppsDirectProtonSimulation", desc);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSDirectProtonSimulation);
