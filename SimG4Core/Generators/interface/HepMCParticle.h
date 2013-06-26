#ifndef SimG4Core_HepMCParticle_h
#define SimG4Core_HepMCParticle_h

#include "G4PrimaryParticle.hh"

class HepMCParticle 
{
public:
    HepMCParticle();
    HepMCParticle(G4PrimaryParticle * pp, int status); 
    ~HepMCParticle();
    const HepMCParticle & operator=(const HepMCParticle &right);
    int operator==(const HepMCParticle &right) const;
    int operator!=(const HepMCParticle &right) const;
private:
    G4PrimaryParticle * theParticle;
    /// status code of the entry
    /// set to 0 after generating links of G4PrimaryParticle object
    int status_code;
public:
    G4PrimaryParticle * getTheParticle();
    void done();
    const int getStatus();
};

#endif
