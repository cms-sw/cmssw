
#include "G4SIMP.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"

#include "G4PhaseSpaceDecayChannel.hh"
#include "G4DecayTable.hh"


G4SIMP* G4SIMP::theInstance = 0;

G4SIMP* G4SIMP::Definition(double mass)
{
  if (theInstance !=0) return theInstance;
  const G4String name = "chi";
  // search in particle table]
  G4ParticleTable* pTable = G4ParticleTable::GetParticleTable();
  G4ParticleDefinition* anInstance = pTable->FindParticle(name);
  if (anInstance ==0)
  {
  // create particle
  //
  //    Arguments for constructor are as follows
  //               name             mass          width         charge
  //             2*spin           parity  C-conjugation
  //          2*Isospin       2*Isospin3       G-parity   
  //               type    lepton number  baryon number   PDG encoding
  //             stable         lifetime    decay table
  //             shortlived      subType    anti_encoding
  
    anInstance = new G4ParticleDefinition(
                 name,            mass,             0,         0.0,
                    1,              +1,             0,
                    0,              -1,             0,             
               "simp",               0,            +1,      9000006,
                 true,            -1.0,          NULL,
                false,       "nucleon");
 

  }
  theInstance = reinterpret_cast<G4SIMP*>(anInstance);
  return theInstance;
}

G4SIMP*  G4SIMP::SIMPDefinition(double mass)
{ 
  return Definition(mass);
}

G4SIMP*  G4SIMP::SIMP()
{ 
  return Definition(1*GeV); // will use correct mass if instance exists
}


