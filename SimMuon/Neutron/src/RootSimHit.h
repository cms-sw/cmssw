#ifndef SimMuon_Neutron_RootSimHit_h
#define SimMuon_Neutron_RootSimHit_h

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <TObject.h>

class RootSimHit : public TObject
{
public:
  RootSimHit() {}
  RootSimHit(const PSimHit & hit);
  PSimHit get() const;
private:
  // properties
//  Local3DPoint    theEntryPoint;    // position at entry
//  Local3DPoint   theExitPoint;     // exitPos 
  float           theEntryX;
  float           theEntryY;
  float           theEntryZ;
  float           theExitX;
  float           theExitY;
  float           theExitZ;
  float           thePabs;          // momentum
  float           theEnergyLoss;    // Energy loss
  float           theThetaAtEntry;
  float           thePhiAtEntry;

  float           theTof;           // Time Of Flight
  short           theParticleType;
  unsigned short  theProcessType;   // ID of the process which created the track
                                    // which created the PSimHit

  // association
  unsigned int    theDetUnitId;
  unsigned int    theTrackId;
  ClassDef(RootSimHit, 1)
};

#endif

