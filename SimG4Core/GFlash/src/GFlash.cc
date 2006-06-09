#include "SimG4Core/GFlash/interface/GFlash.h"
#include "SimG4Core/GFlash/interface/CaloModel.h"
#include "SimG4Core/GFlash/interface/ParametrisedPhysics.h"
#include "SimG4Core/GFlash/src/DefaultHadronPhysicsQGSP.hh"

#include "SimG4Core/Packaging/src/G4DataQuestionaire.hh"
#include "SimG4Core/Packaging/src/GeneralPhysics.hh"
#include "SimG4Core/Packaging/src/EMPhysics.hh"
#include "SimG4Core/Packaging/src/MuonPhysics.hh"
#include "SimG4Core/Packaging/src/IonPhysics.hh"

GFlash::GFlash(const edm::ParameterSet & p) : PhysicsList(p)
{
    G4DataQuestionaire it(photon);
    std::cout << "You are using the simulation engine: QGSP 2.8 + CMS GFLASH" << std::endl;

    if (caloModel==0) caloModel = new CaloModel(p);
    RegisterPhysics(new GeneralPhysics("general"));
    RegisterPhysics(new EMPhysics("EM"));
    RegisterPhysics(new MuonPhysics("muon"));
    RegisterPhysics(new HadronPhysicsQGSP("hadron"));
    RegisterPhysics(new IonPhysics("ion"));
    RegisterPhysics(new ParametrisedPhysics("parametrised"));
}

GFlash::~GFlash() { if (caloModel!=0) delete caloModel; }

