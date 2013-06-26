#ifndef SimG4Core_G4TrackToParticleID_H
#define SimG4Core_G4TrackToParticleID_H

class G4Track;

/**
 * Converts G4Track to particle ID. For PDG Particles it is the obvious number; for alpha, triton and deuteron 
 * the CMS convention is used
 */

class G4TrackToParticleID
{
public:
    G4TrackToParticleID();
    ~G4TrackToParticleID();
    int particleID(const G4Track *);
private:
};

#endif
