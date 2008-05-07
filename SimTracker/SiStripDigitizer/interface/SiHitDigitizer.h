#ifndef _TRACKER_SiHitDigitizer_H_
#define _TRACKER_SiHitDigitizer_H_
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/SiStripDigitizer/interface/SiChargeCollectionDrifter.h"
#include "SimTracker/SiStripDigitizer/interface/SiChargeDivider.h"
#include "SimTracker/SiStripDigitizer/interface/SiInduceChargeOnStrips.h"
#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <map>

namespace CLHEP {
  class HepRandomEngine;
}

class SiStripDetType;
/**
* Digitizes the response for a single SimHit.
*/
class SiHitDigitizer{
 public:

  SiHitDigitizer(const edm::ParameterSet& conf,CLHEP::HepRandomEngine&);

  ~SiHitDigitizer();

  void setChargeDivider(SiChargeDivider* cd){
    if (theSiChargeDivider) delete theSiChargeDivider;
    theSiChargeDivider = cd;
  }
  void setChargeCollectionDrifter(SiChargeCollectionDrifter* cd){
    if (theSiChargeCollectionDrifter) delete theSiChargeCollectionDrifter;
    theSiChargeCollectionDrifter = cd;
  }
  void setInduceChargeOnStrips(SiInduceChargeOnStrips* cd){
    if (theSiInduceChargeOnStrips) delete theSiInduceChargeOnStrips;
    theSiInduceChargeOnStrips = cd;
  }
  
  void setParticleDataTable(const ParticleDataTable * pdt);

  void processHit(const PSimHit&, const StripGeomDetUnit&, GlobalVector,float,
		  std::vector<double>&, unsigned int&, unsigned int&);
  
 private:
  SiChargeDivider* theSiChargeDivider;
  SiChargeCollectionDrifter* theSiChargeCollectionDrifter;
  SiInduceChargeOnStrips* theSiInduceChargeOnStrips;

  edm::ParameterSet conf_;
  CLHEP::HepRandomEngine& rndEngine;
  double depletionVoltage;
  double appliedVoltage;
  double chargeMobility;
  double temperature;
  bool noDiffusion;
  double chargeDistributionRMS;
  double gevperelectron;
  LocalVector DriftDirection(const StripGeomDetUnit*,GlobalVector,float);
  typedef GloballyPositioned<double>      Frame;

};

#endif
