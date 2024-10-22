#ifndef MU_END_DRIFT_SIM_H
#define MU_END_DRIFT_SIM_H

/** \class CSCDriftSim
 * Simulate drift of electrons through CSC gas.
 *
 * \author Rick Wilkinson
 *
 * Someday this class will be subclassed into fast and slow sims,
 * Now it's just slow, according to the rest of the ORCA developers...
 *
 * Last mod: <BR>
 * 30-Jun-00 ptc Doxygenate. <BR>
 *               Bug-fix in ctor: first bin of integral was 'nan'. <BR>
 *               Bug-trap in avalancheCharge(): very rarely could attempt access
 * outside std::vector. <BR> <p> 01-08/00 vin  use binary search in
 * avalancheCharge()<br>
 */

#include <vector>
class CSCLayer;
class CSCDetectorHit;
class PSimHit;
class MagneticField;
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

namespace CLHEP {
  class HepRandomEngine;
}

class CSCDriftSim {
public:
  CSCDriftSim();

  ~CSCDriftSim();

  /** takes a point,
   and creates a signal on the wire
  */
  CSCDetectorHit getWireHit(const Local3DPoint &ionClusterPosition,
                            const CSCLayer *,
                            int wire,
                            const PSimHit &simHit,
                            CLHEP::HepRandomEngine *);

  void setMagneticField(const MagneticField *field) { theMagneticField = field; }

private:
  // helper functions
  double avgPathLengthLowB();
  double pathSigmaLowB();
  double avgDriftTimeLowB();
  double driftTimeSigmaLowB();
  double avgPathLengthHighB();
  double pathSigmaHighB();
  double avgDriftTimeHighB();
  double driftTimeSigmaHighB();
  double avgDrift() const;
  double driftSigma() const;
  double avalancheCharge(CLHEP::HepRandomEngine *);
  double gasGain(const CSCDetId &id) const;

  // local magnetic field
  float bz;
  // distances from wire
  double ycell, zcell;
  // dN/dE for the avalanche sim.
  std::vector<double> dNdEIntegral;
  const double STEP_SIZE;

  const double ELECTRON_DIFFUSION_COEFF;

  const MagneticField *theMagneticField;
};

#endif
