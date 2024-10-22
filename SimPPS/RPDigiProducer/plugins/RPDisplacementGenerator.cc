#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimPPS/RPDigiProducer/plugins/RPDisplacementGenerator.h"

#include <Math/RotationZYX.h>
#include <Math/Rotation3D.h>

using namespace std;
using namespace edm;

RPDisplacementGenerator::RPDisplacementGenerator(bool iIsOn,
                                                 RPDetId _detId,
                                                 const CTPPSRPAlignmentCorrectionsData *alignments,
                                                 const CTPPSGeometry &geom)
    : detId_(_detId) {
  isOn_ = iIsOn;

  unsigned int decId = rawToDecId(detId_);

  math::XYZVectorD S_m;
  RotationMatrix R_m;

  if (alignments) {
    const CTPPSRPAlignmentCorrectionData &ac = alignments->getFullSensorCorrection(decId);
    S_m = ac.getTranslation();
    R_m = ac.getRotationMatrix();
  } else
    isOn_ = false;

  // transform shift and rotation to the local coordinate frame
  const DetGeomDesc *g = geom.sensor(detId_);
  const RotationMatrix &R_l = g->rotation();
  rotation_ = R_l.Inverse() * R_m.Inverse() * R_l;
  shift_ = R_l.Inverse() * R_m.Inverse() * S_m;

  LogDebug("RPDisplacementGenerator").log([&](auto &log) {
    log << " det id = " << decId << ", isOn = " << isOn_ << "\n";
    if (isOn_) {
      log << " shift = " << shift_ << "\n";
      log << " rotation = " << rotation_ << "\n";
    }
  });
}

Local3DPoint RPDisplacementGenerator::displacePoint(const Local3DPoint &p) {
  /// input is in mm, shifts are in mm too

  Translation v(p.x(), p.y(), p.z());
  v = rotation_ * v - shift_;

  return Local3DPoint(v.x(), v.y(), v.z());
}

PSimHit RPDisplacementGenerator::displace(const PSimHit &input) {
  if (!isOn_)
    return input;

  const Local3DPoint &ep = input.entryPoint(), &xp = input.exitPoint();
  const Local3DPoint &dep = displacePoint(ep), &dxp = displacePoint(xp);

  LogDebug("RPDisplacementGenerator::displace\n") << " entry point: " << ep << " -> " << dep << "\n"
                                                  << " exit point : " << xp << " -> " << dxp << "\n";

  return PSimHit(dep,
                 dxp,
                 input.pabs(),
                 input.tof(),
                 input.energyLoss(),
                 input.particleType(),
                 input.detUnitId(),
                 input.trackId(),
                 input.thetaAtEntry(),
                 input.phiAtEntry(),
                 input.processType());
}

uint32_t RPDisplacementGenerator::rawToDecId(uint32_t raw) {
  return ((raw >> CTPPSDetId::startArmBit) & CTPPSDetId::maskArm) * 1000 +
         ((raw >> CTPPSDetId::startStationBit) & CTPPSDetId::maskStation) * 100 +
         ((raw >> CTPPSDetId::startRPBit) & CTPPSDetId::maskRP) * 10 +
         ((raw >> TotemRPDetId::startPlaneBit) & TotemRPDetId::maskPlane);
}
