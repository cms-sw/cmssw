#include "EMPhysics1CMS.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4EmStandardPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4DataQuestionaire.hh"

EMPhysics1CMS::EMPhysics1CMS(G4LogicalVolumeToDDLogicalPartMap& map, 
			     const HepPDT::ParticleDataTable * table_,
			     sim::ChordFinderSetter *chordFinderSetter_, 
			     const edm::ParameterSet & p) : PhysicsList(map, table_, chordFinderSetter_, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "EMPhysicsStandard \n Flags for EM Physics "
			      << emPhys;

  if (emPhys) {
    // EM Physics
    RegisterPhysics(new G4EmStandardPhysics(ver));

    // Synchroton Radiation & GN Physics
    RegisterPhysics(new G4EmExtraPhysics(ver));
  }

  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));
}

