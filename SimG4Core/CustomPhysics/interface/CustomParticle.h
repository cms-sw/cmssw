#ifndef SimG4Core_CustomParticle_H
#define SimG4Core_CustomParticle_H

#include "G4ParticleDefinition.hh"

#include <string>

class CustomParticleFactory;

class CustomParticle : public G4ParticleDefinition
{
    friend class CustomParticleFactory;
private:
    CustomParticle(
       const std::string &     	aName,        double            mass,
       double            	width,        double            charge,   
       int			iSpin,	      int               iParity,    
       int			iConjugation, int               iIsospin,   
       int               	iIsospin3,    int               gParity,
       const std::string &     	pType,        int               lepton,      
       int               	baryon,       int               encoding,
       bool              	stable,       double            lifetime,
       G4DecayTable *	decaytable
   );
public:
   virtual ~CustomParticle() {}
};

#endif
