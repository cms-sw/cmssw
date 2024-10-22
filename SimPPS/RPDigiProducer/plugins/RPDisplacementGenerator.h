#ifndef SimPPS_RPDigiProducer_RP_DISPLACEMENT_GENERATOR_H
#define SimPPS_RPDigiProducer_RP_DISPLACEMENT_GENERATOR_H

#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include <Math/Rotation3D.h>
#include <map>

class PSimHit;

/**
 * \ingroup TotemDigiProduction
 * \brief This class introduces displacements of RP.
 * It actually shifts and rotates PSimHit positions. It doesn't test whether the displaced
 * hit is still on the detector's surface. This check takes place later in the process. It
 * is done via edge effectivity.
 *
 * PSimHit points are given in the "local Det frame" (PSimHit.h)
 */
class CTPPSRPAlignmentCorrectionsData;
class CTPPSGeometry;

class RPDisplacementGenerator {
public:
  using RotationMatrix = ROOT::Math::Rotation3D;
  using Translation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;

  RPDisplacementGenerator(bool iIsOn,
                          RPDetId,
                          const CTPPSRPAlignmentCorrectionsData* alignments,
                          const CTPPSGeometry& geom);

  /// returns displaced PSimHit
  PSimHit displace(const PSimHit&);

  static uint32_t rawToDecId(uint32_t raw);

private:
  /// ID of the detector
  RPDetId detId_;

  /// displacement
  Translation shift_;
  RotationMatrix rotation_;

  /// set to false to bypass displacements
  bool isOn_;

  /// displaces a point
  Local3DPoint displacePoint(const Local3DPoint&);
};

#endif
