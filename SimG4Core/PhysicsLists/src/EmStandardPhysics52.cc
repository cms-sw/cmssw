#include "SimG4Core/PhysicsLists/interface/EmStandardPhysics52.hh"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4LossTableManager.hh"

#include "G4ComptonScattering52.hh"
#include "G4GammaConversion52.hh"
#include "G4PhotoElectricEffect52.hh"

#include "G4MultipleScattering52.hh"
#include "G4MultipleScattering.hh"

#include "G4eIonisation52.hh"
#include "G4eBremsstrahlung52.hh"
#include "G4eplusAnnihilation52.hh"

#include "G4MuIonisation52.hh"
#include "G4MuBremsstrahlung52.hh"
#include "G4MuPairProduction52.hh"

#include "G4hIonisation52.hh"
#include "G4ionIonisation.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4MuonPlus.hh"
#include "G4MuonMinus.hh"
#include "G4PionPlus.hh"
#include "G4PionMinus.hh"
#include "G4KaonPlus.hh"
#include "G4KaonMinus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"
#include "G4Deuteron.hh"
#include "G4Triton.hh"
#include "G4He3.hh"
#include "G4Alpha.hh"
#include "G4GenericIon.hh"


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EmStandardPhysics52::EmStandardPhysics52(const G4String& name, G4int ver)
   :  G4VPhysicsConstructor(name), verbose(ver)
{
  G4LossTableManager::Instance();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EmStandardPhysics52::~EmStandardPhysics52()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EmStandardPhysics52::ConstructParticle()
{
// gamma
  G4Gamma::Gamma();

// leptons
  G4Electron::Electron();
  G4Positron::Positron();
  G4MuonPlus::MuonPlus();
  G4MuonMinus::MuonMinus();

// mesons
  G4PionPlus::PionPlusDefinition();
  G4PionMinus::PionMinusDefinition();
  G4KaonPlus::KaonPlusDefinition();
  G4KaonMinus::KaonMinusDefinition();

// barions
  G4Proton::Proton();
  G4AntiProton::AntiProton();

// ions
  G4Deuteron::Deuteron();
  G4Triton::Triton();
  G4He3::He3();
  G4Alpha::Alpha();
  G4GenericIon::GenericIonDefinition();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EmStandardPhysics52::ConstructProcess()
{
  // Add EM processes realised on base of Geant4 version 5.2

  theParticleIterator->reset();
  while( (*theParticleIterator)() ){
    G4ParticleDefinition* particle = theParticleIterator->value();
    G4ProcessManager* pmanager = particle->GetProcessManager();
    G4String particleName = particle->GetParticleName();
     
    if (particleName == "gamma") {
      // gamma         
      pmanager->AddDiscreteProcess(new G4PhotoElectricEffect52);
      pmanager->AddDiscreteProcess(new G4ComptonScattering52);
      pmanager->AddDiscreteProcess(new G4GammaConversion52);
      
    } else if (particleName == "e-") {
      //electron
      if(verbose > 1)
        G4cout << "### EmStandard52 instantiates eIoni and msc52 for "
               << particleName << G4endl;
      pmanager->AddProcess(new G4MultipleScattering52, -1, 1, 1);
      pmanager->AddProcess(new G4eIonisation52,        -1, 2, 2);
      pmanager->AddProcess(new G4eBremsstrahlung52,    -1,-1, 3);
	    
    } else if (particleName == "e+") {
      //positron
      if(verbose > 1)
        G4cout << "### EmStandard52 instantiates eIoni and msc52 for "
               << particleName << G4endl;
      pmanager->AddProcess(new G4MultipleScattering52, -1, 1, 1);
      pmanager->AddProcess(new G4eIonisation52,        -1, 2, 2);
      pmanager->AddProcess(new G4eBremsstrahlung52,    -1,-1, 3);
      pmanager->AddProcess(new G4eplusAnnihilation52,   0,-1, 4);
      
    } else if( particleName == "mu+" || 
               particleName == "mu-"    ) {
      //muon  
      if(verbose > 1)
        G4cout << "### EmStandard52 instantiates muIoni and msc52 for "
               << particleName << G4endl;
      pmanager->AddProcess(new G4MultipleScattering52,-1, 1, 1);
      pmanager->AddProcess(new G4MuIonisation52,      -1, 2, 2);
      pmanager->AddProcess(new G4MuBremsstrahlung52,  -1,-1, 3);
      pmanager->AddProcess(new G4MuPairProduction52,  -1,-1, 4);       
     
    } else if( particleName == "alpha" ||
               particleName == "He3" ||
	       particleName == "GenericIon" ) {
 
      if(verbose > 1)
        G4cout << "### EmStandard52 instantiates ionIoni and msc52 for "
               << particleName << G4endl;
      pmanager->AddProcess(new G4MultipleScattering52,-1,1,1);
      pmanager->AddProcess(new G4hIonisation52,      -1,2,2);

    } else if ((!particle->IsShortLived()) &&
	       (particle->GetPDGCharge() != 0.0) && 
	       (particle->GetParticleName() != "chargedgeantino")) {
      //all others charged particles except geantino
      if(verbose > 1)
        G4cout << "### EmStandard52 instantiates hIoni and msc52 for "
               << particleName << G4endl;
      pmanager->AddProcess(new G4MultipleScattering52,-1,1,1);
      pmanager->AddProcess(new G4hIonisation52,     -1,2,2);
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

