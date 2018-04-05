#include "SimG4Core/PhysicsLists/interface/CMSThermalNeutrons.h"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4HadronicProcess.hh"

#include "G4ParticleHPThermalScattering.hh"
#include "G4ParticleHPThermalScatteringData.hh"

#include "G4BuilderType.hh"

#include "G4SystemOfUnits.hh"

CMSThermalNeutrons::CMSThermalNeutrons(G4int ver) :
  G4VHadronPhysics("CMSThermalNeutrons"), verbose(ver) {
}

CMSThermalNeutrons::~CMSThermalNeutrons() {}

void CMSThermalNeutrons::ConstructProcess() {
 
  if(verbose > 0) {
    G4cout << "### " << GetPhysicsName() << " Construct Processes " << G4endl;
  }
  G4Neutron* part = G4Neutron::Neutron();
  G4HadronicProcess* hpel = FindElasticProcess(part);
  if(!hpel) {
    G4cout << "### " << GetPhysicsName() 
	   << " WARNING: Fail to add thermal neutron scattering" << G4endl;
    return;
  }

  G4int ni = (hpel->GetHadronicInteractionList()).size();
  if(ni < 1) {
    G4cout << "### " << GetPhysicsName() 
	   << " WARNING: Fail to add thermal neutron scattering - Nint= " 
	   << ni << G4endl;
    return;
  }
  (hpel->GetHadronicInteractionList())[ni-1]->SetMinEnergy(4*CLHEP::eV);

  hpel->RegisterMe(new G4ParticleHPThermalScattering());
  hpel->AddDataSet(new G4ParticleHPThermalScatteringData());
  
}
