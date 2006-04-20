#include "SimG4Core/CustomPhysics/interface/CustomParticle.h"

CustomParticle::CustomParticle(
    const std::string &		aName,        double            mass,
    double			width,        double            charge,   
    int				iSpin,        int               iParity,    
    int				iConjugation, int               iIsospin,   
    int				iIsospin3,    int               gParity,
    const std::string &		pType,        int               lepton,      
    int				baryon,       int               encoding,
    bool			stable,       double            lifetime,
    G4DecayTable *		decaytable)
    : G4ParticleDefinition(aName,mass,width,charge,iSpin,iParity,
			   iConjugation,iIsospin,iIsospin3,gParity,pType,
			   lepton,baryon,encoding,stable,lifetime,decaytable)
{}
