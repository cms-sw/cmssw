#ifndef SimG4Core_EndOfRun_H
#define SimG4Core_EndOfRun_H

#include "G4Run.hh"

class EndOfRun {
public:
  EndOfRun(const G4Run* tRun) : aRun(tRun) {}
  const G4Run* operator()() const { return aRun; }

private:
  const G4Run* aRun;
};

#endif
