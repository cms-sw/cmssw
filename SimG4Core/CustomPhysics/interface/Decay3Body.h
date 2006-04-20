#ifndef SimG4Core_DECAY3BODY_H
#define SimG4Core_DECAY3BODY_H

#include "CLHEP/Vector/LorentzVector.h"

class Decay3Body 
{
public:
    Decay3Body();
    ~Decay3Body();
    void doDecay(const HepLorentzVector & mother,
		 HepLorentzVector & daughter1,
		 HepLorentzVector & daughter2,
		 HepLorentzVector & daughter3);
private:
    inline double sqr(double a);
};

#endif
