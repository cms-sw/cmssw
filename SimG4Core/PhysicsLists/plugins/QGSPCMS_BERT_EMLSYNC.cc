#include "QGSPCMS_BERT_EMLSYNC.hh"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsSync.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"

#include "G4DataQuestionaire.hh"
#include "HadronPhysicsQGSP_BERT.hh"

#include <string>

QGSPCMS_BERT_EMLSYNC::QGSPCMS_BERT_EMLSYNC(G4LogicalVolumeToDDLogicalPartMap& map,
					   const edm::ParameterSet & p) : PhysicsList(map, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  bool srType  = p.getParameter<bool>("SRType");
  std::string region = p.getParameter<std::string>("Region");
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "QGSP_BERT_EMLSYNC 3.3 with Flags for EM Physics "
			      << emPhys << " and for Hadronic Physics "
			      << hadPhys << " S.R. Type " << srType
			      << " and special region " << region << "\n";

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysicsSync("standard EM EMLSYNC",ver,srType,region));

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
    //RegisterPhysics( new HadronPhysicsQGSP_BERT("hadron"));
  
    // Stopping Physics
    RegisterPhysics( new G4QStoppingPhysics("stopping"));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics("ion"));

    // Neutron tracking cut
    RegisterPhysics( new G4NeutronTrackingCut("Neutron tracking cut", ver));
  }
}
