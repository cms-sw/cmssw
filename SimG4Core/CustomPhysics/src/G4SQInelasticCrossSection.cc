
#include "G4SystemOfUnits.hh"
#include "G4DynamicParticle.hh"
#include "G4NistManager.hh"

#include "SimG4Core/CustomPhysics/interface/G4SQ.h"
#include "SimG4Core/CustomPhysics/interface/G4AntiSQ.h"
#include "SimG4Core/CustomPhysics/interface/G4SQInelasticCrossSection.h"

G4SQInelasticCrossSection::G4SQInelasticCrossSection(double mass)
  : G4VCrossSectionDataSet("SQ-neutron")
{
  nist = G4NistManager::Instance();
  theSQ = G4SQ::SQ(mass);
  theAntiSQ = G4AntiSQ::AntiSQ(mass);
}


G4SQInelasticCrossSection::~G4SQInelasticCrossSection()
{}


G4bool G4SQInelasticCrossSection::IsElementApplicable(
                             const G4DynamicParticle* aPart, 
                             G4int Z, const G4Material*)
{
  return ((0 < Z) && 
          (aPart->GetDefinition() == theSQ || 
           aPart->GetDefinition() == theAntiSQ)
         );
}


G4double G4SQInelasticCrossSection::GetElementCrossSection(
                             const G4DynamicParticle* aPart, 
                             G4int Z, const G4Material*)
{
  // return zero for particle instead of antiparticle
  // sexaquark interaction with matter expected really tiny
  if (aPart->GetDefinition() != theAntiSQ) return 0;

  // zero crosssection for particle at rest
  if(aPart->GetKineticEnergy() <= 0.0) { return 0.0; }

  //I don't want to interact on hydrogen
  if(Z <= 1){return 0.0;}

  // get the atomic weight (to estimate nr neutrons)
  G4double A = nist->GetAtomicMassAmu(Z);

  // put the X section low for the antiS to get a flat interaction rate,
  // but also make it scale with the number of neutrons in the material
  // because we are going to interact on neutrons, not on protons
  return (100.*(A-(G4double)Z)/(G4double)Z)*millibarn;

}
