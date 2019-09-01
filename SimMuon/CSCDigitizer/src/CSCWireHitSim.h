#ifndef MU_END_WIRE_HIT_SIM_H
#define MU_END_WIRE_HIT_SIM_H

/** \class CSCWireHitSim
 * Class used to simulate hit on wire in Endcap Muon CSC. <BR>
 * \author Rick Wilkinson
 * \author Tim Cox
 */

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"
#include <vector>

class CSCDriftSim;
class CSCLayer;
class CSCG3Hit;
class CSCGasCollisions;
class CSCLayerGeometry;

namespace CLHEP {
  class HepRandomEngine;
}

class CSCWireHitSim {
public:
  explicit CSCWireHitSim(CSCDriftSim *driftSim, const edm::ParameterSet &p);
  ~CSCWireHitSim();

  // makes wire hits from the given g3hits
  std::vector<CSCDetectorHit> &simulate(const CSCLayer *layer,
                                        const edm::PSimHitContainer &simHits,
                                        CLHEP::HepRandomEngine *);

  void setParticleDataTable(const ParticleDataTable *pdt);

private:
  // Helper functions
  std::vector<Local3DPoint> getIonizationClusters(const PSimHit &hit, const CSCLayer *, CLHEP::HepRandomEngine *);
  CSCDetectorHit driftElectronsToWire();

  // member data
  CSCDriftSim *theDriftSim;
  CSCGasCollisions *theGasIonizer;
  std::vector<CSCDetectorHit> theNewWireHits;
};

#endif
