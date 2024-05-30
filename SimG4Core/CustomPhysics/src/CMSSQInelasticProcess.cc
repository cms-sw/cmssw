
#include "SimG4Core/CustomPhysics/interface/CMSSQInelasticProcess.h"
#include "SimG4Core/CustomPhysics/interface/CMSSQInelasticCrossSection.h"
#include "SimG4Core/CustomPhysics/interface/CMSSQ.h"

#include "G4Types.hh"
#include "G4SystemOfUnits.hh"
#include "G4HadProjectile.hh"
#include "G4ElementVector.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4Element.hh"
#include "G4ParticleChange.hh"
#include "G4NucleiProperties.hh"
#include "G4Nucleus.hh"

#include "G4HadronicException.hh"
#include "G4HadronicProcessStore.hh"
#include "G4HadronicInteraction.hh"

#include "G4ParticleDefinition.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


CMSSQInelasticProcess::CMSSQInelasticProcess(double mass, const G4String& processName)
    : G4HadronicProcess(processName, fHadronic) {
  AddDataSet(new CMSSQInelasticCrossSection(mass));
  theParticle = CMSSQ::SQ(mass);
}


