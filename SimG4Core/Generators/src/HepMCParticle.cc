#include "SimG4Core/Generators/interface/HepMCParticle.h"

HepMCParticle::HepMCParticle() {}

HepMCParticle::HepMCParticle(G4PrimaryParticle* pp, int status) : 
    theParticle(pp),status_code(status) {}

HepMCParticle::~HepMCParticle() {}

const HepMCParticle & HepMCParticle::operator=(const HepMCParticle &right)
{ return *this; }

int HepMCParticle::operator==(const HepMCParticle &right) const { return false; }
int HepMCParticle::operator!=(const HepMCParticle &right) const { return true;}

G4PrimaryParticle * HepMCParticle::getTheParticle() { return theParticle; }
void HepMCParticle::done() { status_code= -1; }
const int HepMCParticle::getStatus() { return status_code;}
