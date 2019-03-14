#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <Math/RotationZYX.h>
#include <Math/Rotation3D.h>

#include "SimPPS/RPDigiProducer/interface/RPDisplacementGenerator.h"

using namespace std;
using namespace edm;

RPDisplacementGenerator::RPDisplacementGenerator(const edm::ParameterSet &ps, RPDetId _detId, const edm::EventSetup &iSetup) : detId(_detId)
{
  isOn = ps.getParameter<bool>("RPDisplacementOn");

  // read the alignment correction
  ESHandle<CTPPSRPAlignmentCorrectionsData> alignments;
  if( auto rec = iSetup.tryToGet<VeryForwardMisalignedGeometryRecord>()) {
      iSetup.get<VeryForwardMisalignedGeometryRecord>().get(alignments);
  } 

  unsigned int decId = rawToDecId(detId);

  math::XYZVectorD S_m;
  RotationMatrix R_m;

  if (alignments.isValid()) {
    const CTPPSRPAlignmentCorrectionData& ac = alignments->getFullSensorCorrection(decId);
    S_m = ac.getTranslation();
    R_m = ac.getRotationMatrix();
  } else
    isOn = false;

  // transform shift and rotation to the local coordinate frame
  ESHandle<CTPPSGeometry> geom;
  iSetup.get<VeryForwardRealGeometryRecord>().get(geom);
  const DetGeomDesc *g = geom->getSensor(detId);
  const DDRotationMatrix& R_l = g->rotation();
  rotation = R_l.Inverse() * R_m.Inverse() * R_l;
  shift = R_l.Inverse() * R_m.Inverse() * S_m;

#ifdef DEBUG
  edm::LogDebug("RPDisplacementGenerator")<<" det id = " << decId << ", isOn = " << isOn << "\n";
  if (isOn) {
    edm::LogDebug("RPDisplacementGenerator") << " shift = " << shift << "\n";
    edm::LogDebug("RPDisplacementGenerator") << " rotation = " << rotation << "\n";
  }
#endif
}

Local3DPoint RPDisplacementGenerator::DisplacePoint(const Local3DPoint &p)
{
  /// input is in mm, shifts are in mm too

  DDTranslation v(p.x(), p.y(), p.z());
  v = rotation * v - shift;

  return Local3DPoint(v.x(), v.y(), v.z());
}

PSimHit RPDisplacementGenerator::Displace(const PSimHit &input)
{
  if (!isOn)
    return input;

  const Local3DPoint &ep = input.entryPoint(), &xp = input.exitPoint();
  const Local3DPoint &dep = DisplacePoint(ep), &dxp = DisplacePoint(xp);

#ifdef DEBUG
  edm::LogDebug("RPDisplacementGenerator::Displace\n")
       << " entry point: " << ep << " -> " << dep << "\n";
       << " exit point : " << xp << " -> " << dxp << "\n";
#endif


  return PSimHit(dep, dxp, input.pabs(), input.tof(), input.energyLoss(), input.particleType(),
    input.detUnitId(), input.trackId(), input.thetaAtEntry(), input.phiAtEntry(), input.processType());
}

uint32_t RPDisplacementGenerator::rawToDecId(uint32_t raw)
{
      return ((raw >> CTPPSDetId::startArmBit) & CTPPSDetId::maskArm) * 1000
             + ((raw >> CTPPSDetId::startStationBit) & CTPPSDetId::maskStation) * 100
             + ((raw >> CTPPSDetId::startRPBit) & CTPPSDetId::maskRP) * 10
             + ((raw >> TotemRPDetId::startPlaneBit) & TotemRPDetId::maskPlane);
    
}
