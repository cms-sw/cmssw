#include "SimTransport/PPSProtonTransport/interface/OpticalFunctionsTransport.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

OpticalFunctionsTransport::OpticalFunctionsTransport(const edm::ParameterSet& iConfig)
    : BaseProtonTransport(iConfig),
      lhcInfoLabel_(iConfig.getParameter<std::string>("lhcInfoLabel")),
      opticsLabel_(iConfig.getParameter<std::string>("opticsLabel")),
      useEmpiricalApertures_(iConfig.getParameter<bool>("useEmpiricalApertures")),
      empiricalAperture45_xi0_int_(iConfig.getParameter<double>("empiricalAperture45_xi0_int")),
      empiricalAperture45_xi0_slp_(iConfig.getParameter<double>("empiricalAperture45_xi0_slp")),
      empiricalAperture45_a_int_(iConfig.getParameter<double>("empiricalAperture45_a_int")),
      empiricalAperture45_a_slp_(iConfig.getParameter<double>("empiricalAperture45_a_slp")),
      empiricalAperture56_xi0_int_(iConfig.getParameter<double>("empiricalAperture56_xi0_int")),
      empiricalAperture56_xi0_slp_(iConfig.getParameter<double>("empiricalAperture56_xi0_slp")),
      empiricalAperture56_a_int_(iConfig.getParameter<double>("empiricalAperture56_a_int")),
      empiricalAperture56_a_slp_(iConfig.getParameter<double>("empiricalAperture56_a_slp")),
      produceHitsRelativeToBeam_(iConfig.getParameter<bool>("produceHitsRelativeToBeam")) {
  MODE = TransportMode::OPTICALFUNCTIONS;
}
void OpticalFunctionsTransport::process(const HepMC::GenEvent* evt,
                                        const edm::EventSetup& iSetup,
                                        CLHEP::HepRandomEngine* _engine) {
  this->clear();

  iSetup.get<LHCInfoRcd>().get(lhcInfoLabel_, lhcInfo_);
  iSetup.get<CTPPSBeamParametersRcd>().get(beamParameters_);
  iSetup.get<CTPPSInterpolatedOpticsRcd>().get(opticsLabel_, opticalFunctions_);

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
  engine_ = _engine;  // the engine needs to be updated for each event

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
  unsigned int optFunctionId_;

  if (mom_lhc.z() < 0)  // sector 45
  {
    optFunctionId_ = optFunctionId45_;
    beamMomentum = beamParameters_->getBeamMom45();
    xangle = beamParameters_->getHalfXangleX45();
    empiricalAperture_xi0_int = empiricalAperture45_xi0_int_;
    empiricalAperture_xi0_slp = empiricalAperture45_xi0_slp_;
    empiricalAperture_a_int = empiricalAperture45_a_int_;
    empiricalAperture_a_slp = empiricalAperture45_a_slp_;
  } else {  // sector 56
    optFunctionId_ = optFunctionId56_;
    beamMomentum = beamParameters_->getBeamMom56();
    xangle = beamParameters_->getHalfXangleX56();
    empiricalAperture_xi0_int = empiricalAperture56_xi0_int_;
    empiricalAperture_xi0_slp = empiricalAperture56_xi0_slp_;
    empiricalAperture_a_int = empiricalAperture56_a_int_;
    empiricalAperture_a_slp = empiricalAperture56_a_slp_;
  }
  if (xangle > 1.0)
    xangle *= urad;
  // calculate kinematics for optics parametrisation
  const double p = mom_lhc.rho();
  const double xi = 1. - p / beamMomentum;
  const double th_x_phys = mom_lhc.x() / p;
  const double th_y_phys = mom_lhc.y() / p;
  const double vtx_lhc_eff_x = vtx_lhc.x() - vtx_lhc.z() * (mom_lhc.x() / mom_lhc.z() + xangle);
  const double vtx_lhc_eff_y = vtx_lhc.y() - vtx_lhc.z() * (mom_lhc.y() / mom_lhc.z());

  if (verbosity_) {
    LogDebug("OpticalFunctionsTransport")
        << "simu: xi = " << xi << ", th_x_phys = " << th_x_phys << ", th_y_phys = " << th_y_phys
        << ", vtx_lhc_eff_x = " << vtx_lhc_eff_x << ", vtx_lhc_eff_y = " << vtx_lhc_eff_y << std::endl;
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
  auto ofp = opticalFunctions_->at(optFunctionId_);
  CTPPSDetId rpId(optFunctionId_);
  const unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

  if (verbosity_)
    LogDebug("OpticalFunctionsTransport") << "  RP " << rpDecId << std::endl;

  // transport proton
  LHCInterpolatedOpticalFunctionsSet::Kinematics k_in = {
      vtx_lhc_eff_x * 1E-1, th_x_phys, vtx_lhc_eff_y * 1E-1, th_y_phys, xi};  // conversions: mm -> cm

  LHCInterpolatedOpticalFunctionsSet::Kinematics k_out;
  ofp.transport(k_in, k_out, true);

  // Original code uses mm, but CMS uses cm, so keep it in cm
  double b_x = k_out.x * 1E1, b_y = k_out.y * 1E1;  // conversions: cm -> mm
  double a_x = k_out.th_x, a_y = k_out.th_y;

  // if needed, subtract beam position and angle
  if (produceHitsRelativeToBeam_) {
    // determine beam position
    LHCInterpolatedOpticalFunctionsSet::Kinematics k_be_in = {0., 0., 0., 0., 0.};
    LHCInterpolatedOpticalFunctionsSet::Kinematics k_be_out;
    ofp.transport(k_be_in, k_be_out, true);

    a_x -= k_be_out.th_x;
    a_y -= k_be_out.th_y;
    b_x -= k_be_out.x * 1E1;
    b_y -= k_be_out.y * 1E1;  // conversions: cm -> mm
  }

  const double z_scoringPlane = ofp.getScoringPlaneZ() * 1E1;  // conversion: cm --> mm

  if (verbosity_) {
    LogDebug("OpticalFunctionsTransport")
        << "    proton transported: a_x = " << a_x << " rad, a_y = " << a_y << " rad, b_x = " << b_x
        << " mm, b_y = " << b_y << " mm, z = " << z_scoringPlane << " mm" << std::endl;
  }

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
