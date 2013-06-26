#ifndef SimG4Core_BeginOfTrack_H
#define SimG4Core_BeginOfTrack_H

#include "G4Track.hh"

class BeginOfTrack
{
public:
    BeginOfTrack(const G4Track * tTrack) : aTrack(tTrack) {}
    const G4Track * operator()() const { return aTrack; }
private:
    const G4Track * aTrack;
};

#endif
