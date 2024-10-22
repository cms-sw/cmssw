#include "SimTransport/PPSProtonTransport/interface/OpticalFunctionsTransport.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

OpticalFunctionsTransport::OpticalFunctionsTransport(const edm::ParameterSet& iConfig, edm::ConsumesCollector iC)
    : BaseProtonTransport(iConfig),
      lhcInfoToken_(iC.esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoLabel")))),
      beamParametersToken_(iC.esConsumes()),
      opticsToken_(iC.esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("opticsLabel")))),
      beamspotToken_(iC.esConsumes()),
      useEmpiricalApertures_(iConfig.getParameter<bool>("useEmpiricalApertures")),
      empiricalAperture45_xi0_int_(iConfig.getParameter<double>("empiricalAperture45_xi0_int")),
      empiricalAperture45_xi0_slp_(iConfig.getParameter<double>("empiricalAperture45_xi0_slp")),
      empiricalAperture45_a_int_(iConfig.getParameter<double>("empiricalAperture45_a_int")),
      empiricalAperture45_a_slp_(iConfig.getParameter<double>("empiricalAperture45_a_slp")),
      empiricalAperture56_xi0_int_(iConfig.getParameter<double>("empiricalAperture56_xi0_int")),
      empiricalAperture56_xi0_slp_(iConfig.getParameter<double>("empiricalAperture56_xi0_slp")),
      empiricalAperture56_a_int_(iConfig.getParameter<double>("empiricalAperture56_a_int")),
      empiricalAperture56_a_slp_(iConfig.getParameter<double>("empiricalAperture56_a_slp")) {
  MODE = TransportMode::OPTICALFUNCTIONS;
}
OpticalFunctionsTransport::~OpticalFunctionsTransport() {}

void OpticalFunctionsTransport::process(const HepMC::GenEvent* evt,
                                        const edm::EventSetup& iSetup,
                                        CLHEP::HepRandomEngine* engine) {
  clear();

  lhcInfo_ = &iSetup.getData(lhcInfoToken_);
  beamParameters_ = &iSetup.getData(beamParametersToken_);
  opticalFunctions_ = &iSetup.getData(opticsToken_);
  beamspot_ = &iSetup.getData(beamspotToken_);

  // Choose the optical function corresponding to the first station ono each side (it is in lhc ref. frame)
  optFunctionId45_ = 0;
  optFunctionId56_ = 0;
  for (const auto& ofp : (*opticalFunctions_)) {
    if (ofp.second.getScoringPlaneZ() < 0) {
      if (optFunctionId45_ == 0)
        optFunctionId45_ = ofp.first;
      if (opticalFunctions_->at(optFunctionId45_).getScoringPlaneZ() < ofp.second.getScoringPlaneZ())
        optFunctionId45_ = ofp.first;
    }
    if (ofp.second.getScoringPlaneZ() > 0) {
      if (optFunctionId56_ == 0)
        optFunctionId56_ = ofp.first;
      if (opticalFunctions_->at(optFunctionId56_).getScoringPlaneZ() > ofp.second.getScoringPlaneZ())
        optFunctionId56_ = ofp.first;
    }
  }
  //
  engine_ = engine;  // the engine needs to be updated for each event

  for (HepMC::GenEvent::particle_const_iterator eventParticle = evt->particles_begin();
       eventParticle != evt->particles_end();
       ++eventParticle) {
    if (!((*eventParticle)->status() == 1 && (*eventParticle)->pdg_id() == 2212))
      continue;

    if (!(fabs((*eventParticle)->momentum().eta()) > etaCut_ && fabs((*eventParticle)->momentum().pz()) > momentumCut_))
      continue;  // discard protons outside kinematic acceptance

    unsigned int line = (*eventParticle)->barcode();
    HepMC::GenParticle* gpart = (*eventParticle);
    if (gpart->pdg_id() != 2212)
      continue;  // only transport stable protons
    if (gpart->status() != 1)
      continue;
    if (m_beamPart.find(line) != m_beamPart.end())  // assures this protons has not been already propagated
      continue;

    transportProton(gpart);
  }
}

bool OpticalFunctionsTransport::transportProton(const HepMC::GenParticle* in_trk) {
  const HepMC::FourVector& vtx_cms = in_trk->production_vertex()->position();  // in mm
  const HepMC::FourVector& mom_cms = in_trk->momentum();

  // transformation to LHC/TOTEM convention
  HepMC::FourVector vtx_lhc(-vtx_cms.x(), vtx_cms.y(), -vtx_cms.z(), vtx_cms.t());
  HepMC::FourVector mom_lhc(-mom_cms.x(), mom_cms.y(), -mom_cms.z(), mom_cms.t());

  // determine the LHC arm and related parameters
  double urad = 1.e-6;
  double beamMomentum = 0.;
  double xangle = 0.;
  double empiricalAperture_xi0_int, empiricalAperture_xi0_slp;
  double empiricalAperture_a_int, empiricalAperture_a_slp;
  unsigned int optFunctionId;
  // get the beam position at the IP in mm and in the LHC ref. frame
  double vtxXoffset;
  double vtxYoffset;
  if (useBeamPositionFromLHCInfo_) {
    vtxXoffset = -beamParameters_->getVtxOffsetX45() * cm_to_mm;
    vtxYoffset = beamParameters_->getVtxOffsetY45() * cm_to_mm;
  } else {
    vtxXoffset = -beamspot_->x() * cm_to_mm;
    vtxYoffset = beamspot_->y() * cm_to_mm;
  }

  if (mom_lhc.z() < 0)  // sector 45
  {
    optFunctionId = optFunctionId45_;
    beamMomentum = beamParameters_->getBeamMom45();
    xangle = beamParameters_->getHalfXangleX45();
    empiricalAperture_xi0_int = empiricalAperture45_xi0_int_;
    empiricalAperture_xi0_slp = empiricalAperture45_xi0_slp_;
    empiricalAperture_a_int = empiricalAperture45_a_int_;
    empiricalAperture_a_slp = empiricalAperture45_a_slp_;
  } else {  // sector 56
    optFunctionId = optFunctionId56_;
    beamMomentum = beamParameters_->getBeamMom56();
    xangle = beamParameters_->getHalfXangleX56();
    empiricalAperture_xi0_int = empiricalAperture56_xi0_int_;
    empiricalAperture_xi0_slp = empiricalAperture56_xi0_slp_;
    empiricalAperture_a_int = empiricalAperture56_a_int_;
    empiricalAperture_a_slp = empiricalAperture56_a_slp_;
  }
  if (xangle > 1.0)
    xangle *= urad;
  // calculate kinematics for optics parametrisation, avoid the aproximation for small angles xangle -> tan(xangle)
  const double p = mom_lhc.rho();
  const double xi = 1. - p / beamMomentum;
  const double th_x_phys = mom_lhc.x() / abs(mom_lhc.z()) - tan(xangle);  //"-" in the LHC ref. frame
  const double th_y_phys = mom_lhc.y() / abs(mom_lhc.z());
  const double vtx_lhc_eff_x = vtx_lhc.x() - vtx_lhc.z() * (mom_lhc.x() / mom_lhc.z() + tan(xangle)) - (vtxXoffset);
  const double vtx_lhc_eff_y = vtx_lhc.y() - vtx_lhc.z() * (mom_lhc.y() / mom_lhc.z()) - (vtxYoffset);

  if (verbosity_) {
    LogDebug("OpticalFunctionsTransport")
        << "simu: xi = " << xi << ", th_x_phys = " << th_x_phys << ", th_y_phys = " << th_y_phys
        << ", vtx_lhc_eff_x = " << vtx_lhc_eff_x << ", vtx_lhc_eff_y = " << vtx_lhc_eff_y;
  }

  // check empirical aperture
  if (useEmpiricalApertures_) {
    const auto& xangle =
        (lhcInfo_->crossingAngle() > 1.0) ? lhcInfo_->crossingAngle() * urad : lhcInfo_->crossingAngle();
    const double xi_th = (empiricalAperture_xi0_int + xangle * empiricalAperture_xi0_slp) +
                         (empiricalAperture_a_int + xangle * empiricalAperture_a_slp) * th_x_phys;

    if (xi > xi_th) {
      if (verbosity_) {
        LogDebug("OpticalFunctionsTransport") << "stop because of empirical appertures";
      }
      return false;
    }
  }

  // transport the proton into  pot/scoring plane
  auto ofp = opticalFunctions_->at(optFunctionId);
  CTPPSDetId rpId(optFunctionId);
  const unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

  if (verbosity_)
    LogDebug("OpticalFunctionsTransport") << "  RP " << rpDecId << std::endl;

  // transport proton
  LHCInterpolatedOpticalFunctionsSet::Kinematics k_in = {
      vtx_lhc_eff_x * cm_to_mm, th_x_phys, vtx_lhc_eff_y * cm_to_mm, th_y_phys, xi};  // conversions: mm -> cm

  LHCInterpolatedOpticalFunctionsSet::Kinematics k_out;
  ofp.transport(k_in, k_out, true);

  // Original code uses mm, but CMS uses cm, so keep it in cm
  double b_x = k_out.x * cm_to_mm, b_y = k_out.y * cm_to_mm;  // conversions: cm -> mm
  double a_x = k_out.th_x, a_y = k_out.th_y;

  // if needed, subtract beam position and angle
  if (produceHitsRelativeToBeam_) {
    // determine beam position
    LHCInterpolatedOpticalFunctionsSet::Kinematics k_be_in = {0., -tan(xangle), 0., 0., 0.};
    LHCInterpolatedOpticalFunctionsSet::Kinematics k_be_out;
    ofp.transport(k_be_in, k_be_out, true);

    a_x -= k_be_out.th_x;
    a_y -= k_be_out.th_y;
    b_x -= k_be_out.x * cm_to_mm;
    b_y -= k_be_out.y * cm_to_mm;  // conversions: cm -> mm
  }

  const double z_scoringPlane = ofp.getScoringPlaneZ() * cm_to_mm;  // conversion: cm --> mm

  if (verbosity_) {
    LogDebug("OpticalFunctionsTransport")
        << "    proton transported: a_x = " << a_x << " rad, a_y = " << a_y << " rad, b_x = " << b_x
        << " mm, b_y = " << b_y << " mm, z = " << z_scoringPlane << " mm";
  }
  //
  // Project the track back to the starting of PPS region in mm
  b_x -= (abs(z_scoringPlane) - ((z_scoringPlane < 0) ? fPPSRegionStart_45_ : fPPSRegionStart_56_) * 1e3) * a_x;
  b_y -= (abs(z_scoringPlane) - ((z_scoringPlane < 0) ? fPPSRegionStart_45_ : fPPSRegionStart_56_) * 1e3) * a_y;

  unsigned int line = in_trk->barcode();
  double px = -p * a_x;
  double py = p * a_y;
  double pz = std::copysign(sqrt(p * p - px * px - py * py), mom_cms.z());
  double e = sqrt(px * px + py * py + pz * pz + ProtonMassSQ);
  TLorentzVector p_out(px, py, pz, e);
  m_beamPart[line] = p_out;
  m_xAtTrPoint[line] = -b_x;
  m_yAtTrPoint[line] = b_y;
  return true;
}
