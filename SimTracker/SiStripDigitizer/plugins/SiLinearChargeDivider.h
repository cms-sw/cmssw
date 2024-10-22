#ifndef Tracker_SiLinearChargeDivider_H
#define Tracker_SiLinearChargeDivider_H

#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SiChargeDivider.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

namespace CLHEP {
  class HepRandomEngine;
}

/**
 * Concrete implementation of SiChargeDivider. 
 * It divides the charge on the line connecting entry and exit point of the SimTrack in the Silicon.
 * Effects that are considered here are:
 * - fluctuations of the charge deposit fronm segment to segment
 * - pulse shape ( in peak of deconvolution mode)
 */
class SiLinearChargeDivider : public SiChargeDivider {
public:
  // constructor
  SiLinearChargeDivider(const edm::ParameterSet& conf);

  // destructor
  ~SiLinearChargeDivider() override;

  // main method: divide the charge (from the PSimHit) into several energy deposits in the bulk
  SiChargeDivider::ionization_type divide(
      const PSimHit*, const LocalVector&, double, const StripGeomDetUnit& det, CLHEP::HepRandomEngine*) override;

  // set the ParticleDataTable (used to fluctuate the charge properly)
  void setParticleDataTable(const ParticleDataTable* pdt) override { theParticleDataTable = pdt; }

private:
  // configuration data
  const bool peakMode;
  const bool fluctuateCharge;
  const int chargedivisionsPerStrip;
  const double deltaCut;
  const double cosmicShift;
  const ParticleDataTable* theParticleDataTable;
  double pulseResolution;
  unsigned int pulset0Idx;
  std::vector<double> pulseValues;

  // Geant4 engine used by fluctuateEloss()
  std::unique_ptr<SiG4UniversalFluctuation> fluctuate;
  // utility: drifts the charge to the surface to estimate the number of relevant strips
  inline float driftXPos(const Local3DPoint& pos, const LocalVector& drift, double thickness) {
    return pos.x() + (thickness / 2. - pos.z()) * drift.x() / drift.z();
  }
  // fluctuate the Eloss
  void fluctuateEloss(double const particleMass,
                      float momentum,
                      float eloss,
                      float length,
                      int NumberOfSegmentation,
                      float elossVector[],
                      CLHEP::HepRandomEngine*);
  // time response (from the pulse shape)
  float TimeResponse(const PSimHit* hit, const StripGeomDetUnit& det);
  void readPulseShape(const std::string& pulseShapeFileName);
};

#endif
