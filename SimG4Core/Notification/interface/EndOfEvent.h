#ifndef SimG4Core_EndOfEvent_H
#define SimG4Core_EndOfEvent_H

#include "G4Event.hh"

class EndOfEvent
{
public:
    EndOfEvent(const G4Event * tEvent) : anEvent(tEvent) {}
    const G4Event * operator()() const { return anEvent; }
private:
    const G4Event * anEvent;
};

#endif
