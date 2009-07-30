#include "QGSPCMS_BERT_EMLGG.hh"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics92.h"
#include "SimG4Core/PhysicsLists/interface/CMSGlauberGribovXS.h"
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

QGSPCMS_BERT_EMLGG::QGSPCMS_BERT_EMLGG(G4LogicalVolumeToDDLogicalPartMap& map,
				       const edm::ParameterSet & p) : PhysicsList(map, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  std::string region = p.getParameter<std::string>("Region");
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "QGSP_BERT_EMLGG 3.3 with Flags for EM Physics "
			      << emPhys << " and for Hadronic Physics "
			      << hadPhys << " and special region " << region
			      << "\n";

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics92("standard EM EML",ver,region));

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

    // Glauber Griboc Cross Section
    RegisterPhysics( new CMSGlauberGribovXS(ver));
  }
}
