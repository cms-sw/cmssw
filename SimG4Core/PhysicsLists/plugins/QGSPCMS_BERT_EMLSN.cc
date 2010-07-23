#include "QGSPCMS_BERT_EMLSN.hh"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics92.h"
#include "SimG4Core/PhysicsLists/interface/CMSGlauberGribovXS.h"
#include "SimG4Core/PhysicsLists/interface/CMSNeutronXS.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "globals.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleWithCuts.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4ParticleTypes.hh"
#include "G4ParticleTable.hh"

#include "G4Material.hh"
#include "G4MaterialTable.hh"
#include "G4ios.hh"
#include <iomanip>   

#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics_option1.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"

#include "G4DataQuestionaire.hh"
#include "HadronPhysicsQGSP_BERT.hh"


QGSPCMS_BERT_EMLSN::QGSPCMS_BERT_EMLSN(G4LogicalVolumeToDDLogicalPartMap& map,
				       const HepPDT::ParticleDataTable * table_,
				       const edm::ParameterSet & p) : PhysicsList(map, table_, p) {

  G4DataQuestionaire it(photon);

  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  double charge= p.getUntrackedParameter<double>("MonopoleCharge",1.0);
  std::string region = p.getParameter<std::string>("Region");
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
                              << "QGSP_BERT_EMLSN 3.3 with Flags for EM Physics "
                              << emPhys << " and for Hadronic Physics "
                              << hadPhys << " and special region " << region
                              << "\n";

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics92("emstandard_eml",table_,ver,region,charge));
    
    // Synchroton Radiation & GN Physics
    RegisterPhysics( new G4EmExtraPhysics("extra EM"));
  }

  // Decays
  RegisterPhysics( new G4DecayPhysics("decay",ver) );

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics( new G4HadronElasticPhysics("elastic",ver,false));

    // Hadron Physics
    G4bool quasiElastic=true;
    RegisterPhysics( new HadronPhysicsQGSP_BERT("hadron",quasiElastic));

    // Stopping Physics
    RegisterPhysics( new G4QStoppingPhysics("stopping"));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics("ion"));
  
    // Neutron tracking cut
    RegisterPhysics( new G4NeutronTrackingCut("Neutron tracking cut", ver));

    // Neutron x-section
    RegisterPhysics( new CMSNeutronXS("neutronXS",ver));

    // Glauber Griboc Cross Section
    RegisterPhysics( new CMSGlauberGribovXS(ver));
  }

}


