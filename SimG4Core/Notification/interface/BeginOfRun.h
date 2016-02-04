#ifndef SimG4Core_BeginOfRun_H
#define SimG4Core_BeginOfRun_H

#include "G4Run.hh"

class BeginOfRun
{
public:
    BeginOfRun(const G4Run * tRun) : aRun(tRun) {}
    const G4Run * operator()() const { return aRun; }
private:
    const G4Run * aRun;
};

#endif
