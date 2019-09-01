#ifndef SimG4Core_BeginOfEvent_H
#define SimG4Core_BeginOfEvent_H

#include "G4Event.hh"

class BeginOfEvent {
public:
  BeginOfEvent(const G4Event* tEvent) : anEvent(tEvent) {}
  const G4Event* operator()() const { return anEvent; }

private:
  const G4Event* anEvent;
};

#endif
