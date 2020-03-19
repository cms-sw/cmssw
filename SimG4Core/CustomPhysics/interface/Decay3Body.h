#ifndef SimG4Core_CustomPhysics_DECAY3BODY_H
#define SimG4Core_CustomPhysics_DECAY3BODY_H

#include "G4LorentzVector.hh"

class Decay3Body {
public:
  Decay3Body();
  ~Decay3Body();

  void doDecay(const G4LorentzVector& mother,
               G4LorentzVector& daughter1,
               G4LorentzVector& daughter2,
               G4LorentzVector& daughter3);

private:
  inline double sqr(double a);
};

#endif
