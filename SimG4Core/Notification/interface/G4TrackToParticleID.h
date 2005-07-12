#ifndef SimG4Core_G4TrackToParticleID_H
#define SimG4Core_G4TrackToParticleID_H

#include <map>
#include <string>

class G4Track;

/**
 * Converts G4Track to particle ID. For PDG Particles it is the obvious number; for alpha, triton and deuteron 
 * the CMS convention is used
 */

class G4TrackToParticleID
{
public:
    typedef std::map<std::string, int> MapType;
    G4TrackToParticleID();
    ~G4TrackToParticleID();
    int particleID(const G4Track *);
private:
    MapType theInternalMap;
};

#endif
