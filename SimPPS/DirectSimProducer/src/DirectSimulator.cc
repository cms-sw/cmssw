/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#include <CLHEP/Random/RandFlat.h>
#include <CLHEP/Random/RandGauss.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>
#include <HepMC/SimpleVector.h>

#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"
#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/PPSObjects/interface/PPSDirectSimulationData.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "SimPPS/DirectSimProducer/interface/DirectSimulator.h"

DirectSimulator::DirectSimulator(const edm::ParameterSet& iConfig)
    : useEmpiricalApertures_(iConfig.getParameter<bool>("useEmpiricalApertures")),
      produceHitsRelativeToBeam_(iConfig.getParameter<bool>("produceHitsRelativeToBeam")),
      verbosity_(iConfig.getUntrackedParameter<unsigned int>("verbosity", 0)) {}

void DirectSimulator::setLHCInfo(const LHCInfo& lhc_info) { crossing_angle_ = lhc_info.crossingAngle(); }

void DirectSimulator::setBeamParameters(const CTPPSBeamParameters& beam_parameters) {
  beamline_parameters_ = {
      {{.beam_momentum = beam_parameters.getBeamMom45(), .half_crossing_angle_x = beam_parameters.getHalfXangleX45()},
       {.beam_momentum = beam_parameters.getBeamMom56(), .half_crossing_angle_x = beam_parameters.getHalfXangleX56()}}};
}

void DirectSimulator::setDirectSimulationData(const PPSDirectSimulationData& data) {
  empiricalAperture45_ = std::make_unique<TF2>("empiricalAperture45", data.getEmpiricalAperture45().c_str());
  empiricalAperture56_ = std::make_unique<TF2>("empiricalAperture56", data.getEmpiricalAperture56().c_str());
  if (useTrackingEfficiencyPerRP_)  // load the efficiency maps
    efficiencyMapsPerRP_ = data.loadEffeciencyHistogramsPerRP();
}

bool DirectSimulator::operator()(const std::array<double, 4>& vtx_cms /*mm*/,
                                 const std::array<double, 4>& mom_cms,
                                 std::map<CTPPSDetId, DirectSimulator::Parameters>& out_params) const {
  out_params.clear();
  if (!optical_functions_)
    throw cms::Exception("DirectSimulator")
        << "Optical functions collection was not provided to the direct simulator algorithm.";

  // transformation to LHC/TOTEM convention
  const auto vtx_lhc = HepMC::FourVector(-vtx_cms.at(0), vtx_cms.at(1), -vtx_cms.at(2), vtx_cms.at(3)),
             mom_lhc = HepMC::FourVector(-mom_cms.at(0), mom_cms.at(1), -mom_cms.at(2), mom_cms.at(3));

  // determine the LHC arm
  const auto arm = mom_lhc.z() < 0 ? 0u   // sector 45
                                   : 1u;  // sector 56

  // calculate kinematics for optics parametrisation
  const double p = mom_lhc.rho(),  /// horizontal component of proton momentum: p_x = th_x * (1-xi) * p_nom
      xi = 1. - p / beamline_parameters_.at(arm).beam_momentum;  /// xi is positive for diffractive protons,
                                                                 /// thus proton momentum p = (1-xi) * p_nom
  const double th_x_phys = mom_lhc.x() / p, th_y_phys = mom_lhc.y() / p;
  const double vtx_lhc_eff_x = vtx_lhc.x() - vtx_lhc.z() * (mom_lhc.x() / mom_lhc.z() +
                                                            beamline_parameters_.at(arm).half_crossing_angle_x),
               vtx_lhc_eff_y = vtx_lhc.y() - vtx_lhc.z() * (mom_lhc.y() / mom_lhc.z());

  std::stringstream ssLog;
  if (verbosity_)
    ssLog << "simu: xi = " << xi << ", "
          << "th_x_phys = " << th_x_phys << ", "
          << "th_y_phys = " << th_y_phys << ", "
          << "vtx_lhc_eff_x = " << vtx_lhc_eff_x << ", "
          << "vtx_lhc_eff_y = " << vtx_lhc_eff_y << std::endl;

  if (useEmpiricalApertures_) {  // check empirical aperture
    auto& empirical_aperture = arm == 0 ? empiricalAperture45_ : empiricalAperture56_;
    empirical_aperture->SetParameter("xi", xi);
    empirical_aperture->SetParameter("xangle", crossing_angle_);
    if (const double th_x_th = empirical_aperture->EvalPar(nullptr); th_x_th > th_x_phys) {
      if (verbosity_) {
        ssLog << "stop because of empirical apertures";
        edm::LogInfo("DirectSimulator") << ssLog.str();
      }
      return false;
    }
  }

  // transport the proton into each pot/scoring plane
  for (const auto& [detid, optical_function] : *optical_functions_) {
    const CTPPSDetId rpId(detid);
    if (rpId.arm() != arm)  // first check the arm
      continue;

    if (verbosity_)
      ssLog << "  RP " << rpId.arm() * 100 + rpId.station() * 10 + rpId.rp() << std::endl;

    // transport proton
    LHCInterpolatedOpticalFunctionsSet::Kinematics k_in = {vtx_lhc_eff_x * 1.e-1,  // conversions: mm -> cm
                                                           th_x_phys,
                                                           vtx_lhc_eff_y * 1.e-1,  // conversions: mm -> cm
                                                           th_y_phys,
                                                           xi},
                                                   k_out;
    optical_function.transport(k_in, k_out, true);

    Parameters scoring_parameters{
        .ax = k_out.th_x,
        .ay = k_out.th_y,
        .bx = k_out.x * 1.e1,                            // conversion: cm -> mm
        .by = k_out.y * 1.e1,                            // conversion: cm -> mm
        .z = optical_function.getScoringPlaneZ() * 1.e1  // conversion: cm -> mm
    };

    if (produceHitsRelativeToBeam_) {  // if needed, subtract beam position and angle
      LHCInterpolatedOpticalFunctionsSet::Kinematics k_be_in = {0., 0., 0., 0., 0.}, k_be_out;
      optical_function.transport(k_be_in, k_be_out, true);  // determine beam position
      scoring_parameters.ax -= k_be_out.th_x;
      scoring_parameters.ay -= k_be_out.th_y;
      scoring_parameters.bx -= k_be_out.x * 1.e1;  // conversion: cm -> mm
      scoring_parameters.by -= k_be_out.y * 1.e1;  // conversion: cm -> mm
    }

    if (verbosity_)
      ssLog << "    proton transported: "
            << "a_x = " << scoring_parameters.ax << " rad, "
            << "a_y = " << scoring_parameters.ay << " rad, "
            << "b_x = " << scoring_parameters.bx << " mm, "
            << "b_y = " << scoring_parameters.by << " mm, "
            << "z = " << scoring_parameters.z << " mm" << std::endl;

    // RP type
    const bool isTrackingRP =
                   rpId.subdetId() == CTPPSDetId::sdTrackingStrip || rpId.subdetId() == CTPPSDetId::sdTrackingPixel,
               isTimingRP = rpId.subdetId() == CTPPSDetId::sdTimingDiamond;

    // apply per-RP efficiency
    if ((useTimingEfficiencyPerRP_ && isTimingRP) || (useTrackingEfficiencyPerRP_ && isTrackingRP)) {
      if (const auto it = efficiencyMapsPerRP_.find(rpId); it != efficiencyMapsPerRP_.end()) {
        const double r = CLHEP::RandFlat::shoot(random_engine_, 0., 1.);
        auto* effMap = it->second.get();
        const double eff = effMap->GetBinContent(effMap->FindBin(scoring_parameters.bx, scoring_parameters.by));
        if (r > eff) {
          if (verbosity_)
            ssLog << "    stop due to per-RP efficiency" << std::endl;
          continue;
        }
      }
    }
    out_params[rpId] = scoring_parameters;
  }

  if (verbosity_)
    edm::LogInfo("DirectSimulator") << ssLog.str();
  return true;
}

void DirectSimulator::produceLiteTracks(const std::map<CTPPSDetId, Parameters>& simulated_parameters,
                                        std::vector<CTPPSLocalTrackLite>& tracks) const {
  for (const auto& [detid, scoring_parameters] : simulated_parameters)
    tracks.emplace_back(detid.rawId(),
                        scoring_parameters.bx,
                        0.,
                        scoring_parameters.by,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.,
                        CTPPSpixelLocalTrackReconstructionInfo::invalid,
                        0,
                        0.,
                        0.);
}
