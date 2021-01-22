#ifndef _SimTracker_SiPhase2Digitizer_Pixel3DDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_Pixel3DDigitizerAlgorithm_h

//-------------------------------------------------------------
// class Pixel3DDigitizerAlgorithm
//
// Specialization of the tracker digitizer for the 3D pixel
// sensors placed at PXB-Layer1 (and possibly Layer2), and
// at PXF-Disk1 and Disk2
//
// Authors: Jordi Duarte-Campderros (CERN/IFCA)
//          Clara Lasaosa Garcia (IFCA)
//--------------------------------------------------------------

#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"

// Data formats
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

// system
#include <functional>

class Pixel3DDigitizerAlgorithm : public Phase2TrackerDigitizerAlgorithm {
public:
  Pixel3DDigitizerAlgorithm(const edm::ParameterSet& conf);
  ~Pixel3DDigitizerAlgorithm() override;

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es) override;
  bool select_hit(const PSimHit& hit, double tCorr, double& sigScale) const override;
  std::vector<DigitizerUtility::SignalPoint> drift(
      const PSimHit& hit,
      const Phase2TrackerGeomDetUnit* pixdet,
      const GlobalVector& bfield,
      const std::vector<DigitizerUtility::EnergyDepositUnit>& ionization_points) const override;
  // overload drift
  std::vector<DigitizerUtility::SignalPoint> drift(
      const PSimHit& hit,
      const Phase2TrackerGeomDetUnit* pixdet,
      const GlobalVector& bfield,
      const std::vector<DigitizerUtility::EnergyDepositUnit>& ionization_points,
      bool diffusion_activated) const;

  // New diffusion function: check implementation
  std::vector<DigitizerUtility::EnergyDepositUnit> diffusion(const LocalPoint& pos,
                                                             const float& ncarriers,
                                                             const std::function<LocalVector(float, float)>& u_drift,
                                                             const std::pair<float, float> pitches,
                                                             const float& thickness) const;
  // Specific for 3D-pixel
  void induce_signal(const PSimHit& hit,
                     const size_t hitIndex,
                     const uint32_t tofBin,
                     const Phase2TrackerGeomDetUnit* pixdet,
                     const std::vector<DigitizerUtility::SignalPoint>& collection_points) override;

private:
  // Radius of Column np and ohmic
  const float np_column_radius_;
  const float ohm_column_radius_;
  // Gap of np column
  const float np_column_gap_;

  // Check if a carrier is inside the column: The point should
  // be described in the pixel cell frame
  const bool is_inside_n_column_(const LocalPoint& p, const float& sensor_thickness) const;
  const bool is_inside_ohmic_column_(const LocalPoint& p, const std::pair<float, float>& pitch) const;
};
#endif
