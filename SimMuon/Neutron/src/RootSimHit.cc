#include "SimMuon/Neutron/src/RootSimHit.h"
ClassImp(RootSimHit)


RootSimHit::RootSimHit(const PSimHit & hit)
: theEntryX(hit.entryPoint().x()),
  theEntryY(hit.entryPoint().y()),
  theEntryZ(hit.entryPoint().z()),
  theExitX(hit.exitPoint().x()),
  theExitY(hit.exitPoint().y()),
  theExitZ(hit.exitPoint().z()),
  thePabs(hit.pabs()),
  theEnergyLoss(hit.energyLoss()),
  theThetaAtEntry(hit.thetaAtEntry()), thePhiAtEntry(hit.phiAtEntry()),
  theTof(hit.tof()),
  theParticleType(hit.particleType()), theProcessType(hit.processType()),
  theDetUnitId( hit.detUnitId()), theTrackId( hit.trackId())
{
}


PSimHit RootSimHit::get() const
{
 return PSimHit(Local3DPoint(theEntryX, theEntryY, theEntryZ),
                Local3DPoint(theExitX, theExitY, theExitZ),
                thePabs, theTof,
                theEnergyLoss, theParticleType, theDetUnitId, theTrackId,
                theThetaAtEntry,  thePhiAtEntry, theProcessType);
}



