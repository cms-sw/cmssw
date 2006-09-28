//#define G4v7
#include "SimG4Core/CustomPhysics/interface/CustomPhysics.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "SimG4Core/CustomPhysics/src/DefaultHadronPhysicsQGSP.hh"
 
#ifdef G4v7
#include "SimG4Core/Packaging/src/G4DataQuestionaire.hh"
#include "SimG4Core/Packaging/src/GeneralPhysics.hh"
#include "SimG4Core/Packaging/src/EMPhysics.hh"
#include "SimG4Core/Packaging/src/MuonPhysics.hh"
#include "SimG4Core/Packaging/src/IonPhysics.hh"
#else
#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics71.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh" 
#include "G4DataQuestionaire.hh"
#endif
 
CustomPhysics::CustomPhysics(const edm::ParameterSet & p) : PhysicsList(p)
{
    G4DataQuestionaire it(photon);
#ifdef G4v7
    std::cout << "You are using the simulation engine: QGSP 2.8" << std::endl;
   
    RegisterPhysics(new GeneralPhysics("general"));
    RegisterPhysics(new EMPhysics("EM"));
    RegisterPhysics(new MuonPhysics("muon"));
    RegisterPhysics(new HadronPhysicsQGSP("hadron"));
    RegisterPhysics(new IonPhysics("ion"));
    RegisterPhysics(new CustomPhysicsList("custom",p));
#else
    std::cout << "You are using the simulation engine: QGSP_EMV 3.1" << std::endl;

    // EM Physics
    RegisterPhysics(new G4EmStandardPhysics71("standard EM v71"));
    // Synchroton Radiation & GN Physics
    RegisterPhysics(new G4EmExtraPhysics("extra EM"));
    // Decays
    RegisterPhysics(new G4DecayPhysics("decay"));
    // Hadron Elastic scattering
    RegisterPhysics(new G4HadronElasticPhysics("elastic")); 
    // Hadron Physics
    RegisterPhysics(new HadronPhysicsQGSP("hadron"));
    // Stopping Physics
    RegisterPhysics(new G4QStoppingPhysics("stopping"));
    // Ion Physics
    RegisterPhysics(new G4IonPhysics("ion"));
#endif
}
