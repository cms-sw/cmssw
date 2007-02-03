#ifndef MU_END_WIRE_HIT_SIM_H
#define MU_END_WIRE_HIT_SIM_H

/** \class CSCWireHitSim
 * Class used to simulate hit on wire in Endcap Muon CSC. <BR>
 * \author Rick Wilkinson
 * \author Tim Cox
 */

#include <vector>
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

class CSCDriftSim;
class CSCLayer;
class CSCG3Hit;
class CSCGasCollisions;
class CSCLayerGeometry;


class CSCWireHitSim
{
public:
  explicit CSCWireHitSim(CSCDriftSim* driftSim);
  ~CSCWireHitSim();

  // makes wire hits from the given g3hits
  std::vector<CSCDetectorHit> & simulate(const CSCLayer * layer, 
				    const edm::PSimHitContainer & simHits);
 
  void setParticleDataTable(const ParticleDataTable * pdt);

private:
  // Helper functions
  std::vector<Local3DPoint> getIonizationClusters(const PSimHit & hit, 
                                             const CSCLayer *);
  CSCDetectorHit driftElectronsToWire();

  // member data
  CSCDriftSim*  theDriftSim;
  CSCGasCollisions* theGasIonizer;
  std::vector<CSCDetectorHit> theNewWireHits;
};

#endif
