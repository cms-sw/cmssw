#include "SimG4Core/CustomPhysics/interface/CustomPhysics.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "SimG4Core/CustomPhysics/src/DefaultHadronPhysicsQGSP.hh"
 
#include "SimG4Core/Packaging/src/G4DataQuestionaire.hh"
#include "SimG4Core/Packaging/src/GeneralPhysics.hh"
#include "SimG4Core/Packaging/src/EMPhysics.hh"
#include "SimG4Core/Packaging/src/MuonPhysics.hh"
#include "SimG4Core/Packaging/src/IonPhysics.hh"
 
CustomPhysics::CustomPhysics(const edm::ParameterSet & p) : PhysicsList(p)
{
    G4DataQuestionaire it(photon);
    std::cout << "You are using the simulation engine: QGSP 2.8" << std::endl;
   
    RegisterPhysics(new GeneralPhysics("general"));
    RegisterPhysics(new EMPhysics("EM"));
    RegisterPhysics(new MuonPhysics("muon"));
    RegisterPhysics(new HadronPhysicsQGSP("hadron"));
    RegisterPhysics(new IonPhysics("ion"));
    RegisterPhysics(new CustomPhysicsList("custom",p));
}
