#include "CMSModel.hh"
#include "SimG4Core/PhysicsLists/interface/HadronPhysicsCMS.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh" 

#include "G4DataQuestionaire.hh"

CMSModel::CMSModel(G4LogicalVolumeToDDLogicalPartMap& map, 
		   const HepPDT::ParticleDataTable * table_, 
		   sim::FieldBuilder *fieldBuilder_, 
		   const edm::ParameterSet & p) : PhysicsList(map, table_, fieldBuilder_, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  std::string model = p.getUntrackedParameter<std::string>("Model","QGSP");
  bool quasiElastic = p.getUntrackedParameter<bool>("QuasiElastic",true);
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << model << " with Flags for QuasiElastic "
			      << quasiElastic << " and Verbosity Flag " 
			      << ver << "\n";
  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));

  // Hadron Elastic scattering
  RegisterPhysics(new G4HadronElasticPhysics(ver)); 

  // Hadron Physics
  RegisterPhysics(new HadronPhysicsCMS(model, quasiElastic));

  // Stopping Physics
  RegisterPhysics(new G4StoppingPhysics(ver));

  // Ion Physics
  RegisterPhysics(new G4IonPhysics(ver));
}

