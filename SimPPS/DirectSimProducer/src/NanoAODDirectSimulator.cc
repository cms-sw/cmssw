/****************************************************************************
 * Authors:
 *   Laurent Forthomme
 ****************************************************************************/

#include "CondFormats/PPSObjects/interface/LHCOpticalFunctionsSetCollection.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/DirectSimProducer/interface/NanoAODDirectSimulator.h"

NanoAODDirectSimulator::NanoAODDirectSimulator() {}

void NanoAODDirectSimulator::initialise() {
  auto params = edm::ParameterSet{};
  params.addParameter<bool>("useEmpiricalApertures", false);
  params.addParameter<bool>("produceHitsRelativeToBeam", false);
  params.addUntrackedParameter<unsigned int>("verbosity", verbosity_);
  simulator_ = std::make_unique<DirectSimulator>(params);

  simulator_->setBeamParameters(beam_parameters_);
  simulator_->setLHCInfo(lhc_info_);

  buildInterpolatedOpticalFunctions();
  simulator_->setOpticalFunctions(interpolated_optical_functions_);
}

void NanoAODDirectSimulator::addScoringPlane(unsigned int detid, double zpos, const std::string& dir_name) {
  scoring_planes_.emplace_back(ScoringPlaneInfo{.rp_id = detid, .z_position = zpos, .directory_name = dir_name});
}

void NanoAODDirectSimulator::addCrossingAngleOpticalFunctions(double crossing_angle, const std::string& filename) {
  optical_functions_files_[crossing_angle] = filename;
}

void NanoAODDirectSimulator::setBeamEnergy(double beam_energy) {
  lhc_info_.setEnergy(beam_energy);
  beam_parameters_.setBeamMom45(beam_energy);  //FIXME symmetric for now
  beam_parameters_.setBeamMom56(beam_energy);
}

void NanoAODDirectSimulator::setBetaStar(double beta_ast) {
  lhc_info_.setBetaStar(beta_ast);
  beam_parameters_.setBetaStarX45(beta_ast);  //FIXME symmetric for now
  beam_parameters_.setBetaStarX56(beta_ast);
}

void NanoAODDirectSimulator::setCrossingAngle(double crossing_angle) {
  lhc_info_.setCrossingAngle(crossing_angle);
  beam_parameters_.setHalfXangleX45(0.5 * crossing_angle);  //FIXME symmetric for now
  beam_parameters_.setHalfXangleX56(0.5 * crossing_angle);
}

std::vector<CTPPSLocalTrackLite> NanoAODDirectSimulator::computeProton(const TLorentzVector& vtx_cms /*mm*/,
                                                                       const TLorentzVector& mom_cms) const {
  std::vector<CTPPSLocalTrackLite> out_tracks;
  std::map<CTPPSDetId, DirectSimulator::Parameters> simulated_parameters;
  if (!(*simulator_)({{vtx_cms.X(), vtx_cms.Y(), vtx_cms.Z(), vtx_cms.T()}},
                     {{mom_cms.Px(), mom_cms.Py(), mom_cms.Pz(), mom_cms.E()}},
                     simulated_parameters)) {
    edm::LogWarning("NanoAODDirectSimulator") << "Failed to propagate a proton to the pots." << std::endl;
    return out_tracks;
  }
  simulator_->produceLiteTracks(simulated_parameters, out_tracks);
  return out_tracks;
}

/// Helper to gain access to protected members of the LHCOpticalFunctionsSet object
struct LHCOpticalFunctionsSetBuilder : LHCOpticalFunctionsSet {
  explicit LHCOpticalFunctionsSetBuilder(double z,
                                         const std::vector<double>& xi_values,
                                         const std::vector<std::vector<double>>& fcn_values) {
    m_z = z, m_xi_values = xi_values, m_fcn_values = fcn_values;
  }
};

void NanoAODDirectSimulator::buildInterpolatedOpticalFunctions() {
  const auto cms_data_path = std::filesystem::path(getenv("CMSSW_DATA_PATH"));
  LHCOpticalFunctionsSetCollection optical_functions;
  for (const auto& [xangle, filename] : optical_functions_files_) {
    auto& crossing_angle_functions = optical_functions[xangle];
    for (const auto& [rp_id, z_position, directory_name] : scoring_planes_)
      crossing_angle_functions[rp_id] = LHCOpticalFunctionsSet(
          cms_data_path / "data-CalibPPS-ESProducers/V01-05-00" / filename, directory_name, z_position);
  }
  if (optical_functions.size() == 1)  // trivial case: pick this set
    //TODO: introduce some checks on the crossing angle value
    for (const auto& [rpId, rp_optical_functions] : optical_functions.begin()->second) {
      interpolated_optical_functions_[rpId] = LHCInterpolatedOpticalFunctionsSet(rp_optical_functions);
      interpolated_optical_functions_[rpId].initializeSplines();
    }
  else {  // have to extrapolate from the two surrounding crossing angle values
    const auto first_crossing_angle = optical_functions.begin()->first,
               last_crossing_angle = optical_functions.rbegin()->first;
    if (lhc_info_.crossingAngle() < first_crossing_angle || lhc_info_.crossingAngle() > last_crossing_angle)
      throw cms::Exception("NanoAODDirectSimulator")
          << "Crossing angle " << lhc_info_.crossingAngle() << " is not within acceptable range ["
          << first_crossing_angle << ", " << last_crossing_angle << "].";
    auto it_low = optical_functions.begin(), it_high = it_low;
    for (; it_low != optical_functions.end(); ++it_low) {
      if (it_high = std::next(it_low); it_high == optical_functions.end())
        break;
      if (it_low->first <= lhc_info_.crossingAngle() && it_high->first >= lhc_info_.crossingAngle())
        break;
    }
    const auto& [xangle_low, scoring_planes_low] = *it_low;
    const auto& [xangle_high, scoring_planes_high] = *it_high;

    for (const auto& [rpId, optical_functions_low] : scoring_planes_low) {  // do the interpolation RP by RP
      if (!scoring_planes_high.count(rpId))
        throw cms::Exception("NanoAODDirectSimulator") << "RP mismatch between scoring planes 1 and 2.";

      const size_t num_xi_vals = optical_functions_low.getXiValues().size();
      const auto& optical_functions_high = scoring_planes_high.at(rpId);
      if (optical_functions_high.getXiValues().size() != num_xi_vals)
        throw cms::Exception("NanoAODDirectSimulator") << "Size mismatch between scoring planes 1 and 2.";

      std::vector<double> xi_values(num_xi_vals);
      std::vector<std::vector<double>> fcn_values(LHCInterpolatedOpticalFunctionsSet::nFunctions);
      for (size_t fi = 0; fi < optical_functions_low.getFcnValues().size(); ++fi) {
        fcn_values[fi].resize(num_xi_vals);
        for (size_t pi = 0; pi < num_xi_vals; ++pi) {  // linear extrapolation of each xi-dependent function
          xi_values[pi] = optical_functions_low.getXiValues()[pi];
          if (const auto xi_control = optical_functions_high.getXiValues()[pi];
              std::fabs(xi_values[pi] - xi_control) > 1e-6)
            throw cms::Exception("NanoAODDirectSimulator") << "xi mismatch between scoring planes 1 and 2.";
          const auto v1 = optical_functions_low.getFcnValues()[fi][pi],
                     v2 = optical_functions_high.getFcnValues()[fi][pi];
          fcn_values[fi][pi] = v1 + (v2 - v1) / (xangle_high - xangle_low) * (lhc_info_.crossingAngle() - xangle_low);
        }
      }
      interpolated_optical_functions_[rpId] = LHCInterpolatedOpticalFunctionsSet(
          LHCOpticalFunctionsSetBuilder(optical_functions_low.getScoringPlaneZ(), xi_values, fcn_values));
      interpolated_optical_functions_[rpId].initializeSplines();
    }
  }
  //TODO: ideal case would be to retrieve these values from conditions database
  /*const auto optical_functions_source =
      edm::eventsetup::SourcePluginFactory::get()->create("ctppsOpticalFunctionsESSource");*/
}
