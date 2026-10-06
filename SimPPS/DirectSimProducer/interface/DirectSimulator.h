/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#ifndef SimPPS_DirectSimProducer_DirectSimulator_h
#define SimPPS_DirectSimProducer_DirectSimulator_h

#include <map>

#include <TF2.h>
#include <TH2F.h>

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

// forward definitions
class CTPPSBeamParameters;
class CTPPSGeometry;
class CTPPSLocalTrackLite;
class LHCInfo;
class LHCInterpolatedOpticalFunctionsSetCollection;
class PPSDirectSimulationData;
namespace edm {
  class ParameterSet;
}  // namespace edm
namespace CLHEP {
  class HepRandomEngine;
}  // namespace CLHEP

class DirectSimulator {
public:
  explicit DirectSimulator(const edm::ParameterSet& iConfig);

  void setGeometry(const CTPPSGeometry& geometry) { geometry_ = &geometry; }
  void setLHCInfo(const LHCInfo&);
  void setBeamParameters(const CTPPSBeamParameters&);
  void setOpticalFunctions(const LHCInterpolatedOpticalFunctionsSetCollection& optical_functions) {
    optical_functions_ = &optical_functions;
  }
  void setRandomEngine(CLHEP::HepRandomEngine& random_engine) { random_engine_ = &random_engine; }
  void setDirectSimulationData(const PPSDirectSimulationData&);

  void setCrossingAngle(double crossing_angle) { crossing_angle_ = crossing_angle; }

  struct Parameters {
    double ax{0.}, ay{0.}, bx{0.}, by{0.};
    double z{0.};
  };
  /// Compute the per-pot parameters for a given particle with a given vertex
  /// \param[in] vtx_cms 4-vector (x, y, z, t) in mm/s for the particle vertex
  /// \param[in] mom_cms 4-momentum (px, py, pz, E) in GeV for the particle
  /// \param[out] out_params collection of parameters for each pot position encountered
  /// \return a boolean stating whether the particle was properly propagated to the pots location
  bool operator()(const std::array<double, 4>& vtx_cms /*mm*/,
                  const std::array<double, 4>& mom_cms,
                  std::map<CTPPSDetId, Parameters>& out_params) const;
  /// Produce track objects from each pot computed parameters
  void produceLiteTracks(const std::map<CTPPSDetId, Parameters>&, std::vector<CTPPSLocalTrackLite>&) const;

private:
  const CTPPSBeamParameters* beam_parameters_{nullptr};
  const LHCInterpolatedOpticalFunctionsSetCollection* optical_functions_{nullptr};
  const CTPPSGeometry* geometry_{nullptr};
  CLHEP::HepRandomEngine* random_engine_{nullptr};

  double crossing_angle_{0.};
  struct SectorBeamlineParameters {
    double beam_momentum{0.}, half_crossing_angle_x{0.};
  };
  std::array<SectorBeamlineParameters, 2> beamline_parameters_;

  // settings of LHC aperture limitations (high xi)
  bool useEmpiricalApertures_;
  std::unique_ptr<TF2> empiricalAperture45_;
  std::unique_ptr<TF2> empiricalAperture56_;

  // efficiency flags
  bool useTrackingEfficiencyPerRP_;
  bool useTimingEfficiencyPerRP_;

  // efficiency maps
  std::map<unsigned int, std::unique_ptr<TH2F>> efficiencyMapsPerRP_;

  // other parameters
  bool produceHitsRelativeToBeam_;

  unsigned int verbosity_;
};

#endif
