/****************************************************************************
 * Authors:
 *   Laurent Forthomme
 ****************************************************************************/

#ifndef SimPPS_DirectSimProducer_NanoAODDirectSimulator_h

#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "SimPPS/DirectSimProducer/interface/DirectSimulator.h"

#include <TLorentzVector.h>

class NanoAODDirectSimulator {
public:
  NanoAODDirectSimulator();

  void addCrossingAngleOpticalFunctions(double crossing_angle,
                                        const std::string& filename);  ///< Add a new crossing angle value
  void addScoringPlane(unsigned int, double, const std::string&);      ///< Add a RP position with its optical functions
  void setBeamEnergy(double);                                          ///< Set the current beam energy, in GeV
  void setBetaStar(double);                                            ///< Set the current beta*, in m
  void setCrossingAngle(double);                                       ///< Set the current crossing angle, in urad
  void initialise();

  /// Compute the per-pot parameters for a given particle with a given vertex
  /// \param[in] vtx_cms 4-vector (x, y, z, t) in mm/s for the particle vertex
  /// \param[in] mom_cms 4-momentum (px, py, pz, E) in GeV for the particle
  /// \return a collection of tracks information for each pot location
  std::vector<CTPPSLocalTrackLite> computeProton(const TLorentzVector& vtx_cms /*mm*/,
                                                 const TLorentzVector& mom_cms) const;

private:
  void buildInterpolatedOpticalFunctions();

  std::unique_ptr<DirectSimulator> simulator_;

  std::map<double, std::string> optical_functions_files_;  ///< location of per-xangle optical functions
  struct ScoringPlaneInfo {
    unsigned int rp_id{0};
    double z_position{0.};
    std::string directory_name;
  };
  std::vector<ScoringPlaneInfo> scoring_planes_;  ///< list of RP stations to be simulated
  LHCInterpolatedOpticalFunctionsSetCollection interpolated_optical_functions_;  ///< optical functions
  LHCInfo lhc_info_;                                                             ///< LHC optics metadata
};

#endif
