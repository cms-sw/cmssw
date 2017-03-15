#ifndef SimG4Core_Application_CMSEmParticleList_h
#define SimG4Core_Application_CMSEmParticleList_h 1

// V.Ivanchenko 6 March 2017
// List of Geant4 basic particle names used in SIM step 

#include "globals.hh"

static const G4int nEmParticles = 39;
static const G4String EmPartNames[nEmParticles] = 
{ 
        "gamma",            "e-",           "e+",           "mu+",        "mu-",
          "pi+",           "pi-",        "kaon+",         "kaon-",     "proton",
  "anti_proton",         "alpha",          "He3",    "GenericIon",         "B+",
           "B-",            "D+",           "D-",           "Ds+",        "Ds-",
     "anti_He3",    "anti_alpha","anti_deuteron","anti_lambda_c+","anti_omega-",
"anti_sigma_c+","anti_sigma_c++",  "anti_sigma+",   "anti_sigma-","anti_triton",
     "sigma_c+",     "sigma_c++",       "sigma+",        "sigma-",       "tau+",
         "tau-",        "triton",        "xi_c+",           "xi-"
};


#endif

