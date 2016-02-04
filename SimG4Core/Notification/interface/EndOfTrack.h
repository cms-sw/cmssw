#ifndef SimG4Core_EndOfTrack_H
#define SimG4Core_EndOfTrack_H

#include "G4Track.hh"

class EndOfTrack
{
public:
    EndOfTrack(const G4Track * tTrack) : aTrack(tTrack) {}
    const G4Track * operator()() const { return aTrack; }
private:
    const G4Track * aTrack;
};

#endif
