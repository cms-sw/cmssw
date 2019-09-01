#ifndef UpdatablePSimHit_H
#define UpdatablePSimHit_H

/** \class UpdatablePSimHit
 * extension of PSimHit;
 * the exit point and the energy loss can be modified;
 * maybe not the final solution
 */

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

class UpdatablePSimHit : public PSimHit {
public:
  UpdatablePSimHit() : PSimHit() {}
  UpdatablePSimHit(const Local3DPoint& entry,
                   const Local3DPoint& exit,
                   float pabs,
                   float tof,
                   float eloss,
                   int particleType,
                   unsigned int detId,
                   unsigned int trackId,
                   float theta,
                   float phi,
                   unsigned short processType = 0)
      : PSimHit(entry, exit, pabs, tof, eloss, particleType, detId, trackId, theta, phi, processType) {}
  ~UpdatablePSimHit(){};
  void updateExitPoint(const Local3DPoint& exit) { theSegment = exit - theEntryPoint; }
  void setExitPoint(const Local3DPoint& exit) { updateExitPoint(exit); }
  void setEntryPoint(const Local3DPoint& entry) {
    theSegment = theSegment + theEntryPoint - entry;
    theEntryPoint = entry;
  }
  void addEnergyLoss(float eloss) { theEnergyLoss += eloss; }
  void setEnergyLoss(float eloss) { theEnergyLoss = eloss; }
  void setTrackId(unsigned int k) { theTrackId = k; }
};

#endif
