#ifndef SimG4Core_GFlash_ParametrisedPhysics_H
#define SimG4Core_GFlash_ParametrisedPhysics_H

#include "G4VPhysicsConstructor.hh"

// Joanna Weng 08.2005
// Physics process for Gflash parameterisation

class ParametrisedPhysics : public G4VPhysicsConstructor
{
	public:
	ParametrisedPhysics(std::string name);
	virtual ~ParametrisedPhysics();
	
	protected:
	virtual void ConstructParticle();
	virtual void ConstructProcess();
	void addParametrisation(); 
};

#endif

