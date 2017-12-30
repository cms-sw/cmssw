#ifndef SimG4Core_G4TrackToParticleID_H
#define SimG4Core_G4TrackToParticleID_H

class G4Track;

/**
 * Converts G4Track to particle ID. For PDG Particles it is the obvious number; 
 * for alpha, triton and deuteron the CMS convention is used
 */

class G4TrackToParticleID
{
public:
  static int  particleID(const G4Track *);
  static bool isGammaElectronPositron(const G4Track *);
  static bool isMuon(const G4Track *);
  static bool isStableHadron(const G4Track *);
};

#endif
