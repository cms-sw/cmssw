#ifndef _TRACKER_SiHitDigitizer_H_
#define _TRACKER_SiHitDigitizer_H_
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SiChargeCollectionDrifter.h"
#include "SiChargeDivider.h"
#include "SiInduceChargeOnStrips.h"
#include "SiPileUpSignals.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <map>
#include <memory>

class TrackerTopology;

namespace CLHEP {
  class HepRandomEngine;
}

class SiStripDetType;
/**
* Digitizes the response for a single SimHit.
*/
class SiHitDigitizer {
 public:

  SiHitDigitizer(const edm::ParameterSet& conf,CLHEP::HepRandomEngine&);

  ~SiHitDigitizer();

  void setChargeDivider(SiChargeDivider* cd) {
    theSiChargeDivider.reset(cd);
  }

  void setChargeCollectionDrifter(SiChargeCollectionDrifter* cd) {
    theSiChargeCollectionDrifter.reset(cd);
  }

  void setInduceChargeOnStrips(SiInduceChargeOnStrips* cd) {
    theSiInduceChargeOnStrips.reset(cd);
  }
  
  void setParticleDataTable(const ParticleDataTable * pdt) { 
    theSiChargeDivider->setParticleDataTable(pdt); 
  }

  void processHit(const PSimHit*, const StripGeomDetUnit&, GlobalVector,float,
		  std::vector<double>&, size_t&, size_t&,
		  const TrackerTopology *tTopo);
  
 private:
  const double depletionVoltage;
  const double chargeMobility;
  std::unique_ptr<SiChargeDivider> theSiChargeDivider;
  std::unique_ptr<SiChargeCollectionDrifter> theSiChargeCollectionDrifter;
  std::unique_ptr<const SiInduceChargeOnStrips> theSiInduceChargeOnStrips;

  typedef GloballyPositioned<double> Frame;
  
  LocalVector DriftDirection(const StripGeomDetUnit* _detp, GlobalVector _bfield, float langle) {
    LocalVector Bfield=Frame(_detp->surface().position(),_detp->surface().rotation()).toLocal(_bfield);
    return LocalVector(-langle * Bfield.y(),langle * Bfield.x(),1.);
  }

};

#endif
