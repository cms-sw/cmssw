#ifndef Tracker_SiChargeDivider_H
#define Tracker_SiChargeDivider_H

#include "EnergyDepositUnit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <vector>
/**
 * Base class for the division of a Geant energy deposit in smaller elementary charges inside the silicon
 */
class SiChargeDivider{
 public:
  typedef std::vector< EnergyDepositUnit > ionization_type;
  virtual ~SiChargeDivider() { }
  virtual ionization_type divide(const PSimHit*, const LocalVector&, double, const StripGeomDetUnit& det ) = 0;
  virtual void setParticleDataTable(const ParticleDataTable * pdt) = 0;
};


#endif
