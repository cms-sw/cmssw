#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "SimG4Core/Physics/interface/G4Monopole.hh"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4StepLimiter.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4mplIonisation.hh"
#include "G4hhIonisation.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

CMSMonopolePhysics::CMSMonopolePhysics(const HepPDT::ParticleData *particle, 
				       G4double charge, G4int ver) :
  verbose(ver), ok(false), magCharge((int)(charge)) {
  if (particle) {
    monopoleMass = (particle->mass())*CLHEP::GeV;
    elCharge     = (int)(particle->charge());
    pdgEncoding  = particle->pid();
    ok           = true;
  }
}

CMSMonopolePhysics::~CMSMonopolePhysics() {}

void CMSMonopolePhysics::ConstructParticle() {
  
  if (ok) {
    G4Monopole::MonopoleDefinition(monopoleMass, magCharge, elCharge, pdgEncoding);
    edm::LogInfo("PhysicsList") << "Create G4Monopole of mass " 
				<< monopoleMass/CLHEP::GeV
				<< " GeV, magnetic charge " << magCharge 
				<< ", electric charge " << elCharge
				<< " and PDG encoding " << pdgEncoding;
  }
}

void CMSMonopolePhysics::ConstructProcess() {
  // Add standard EM Processes

  G4Monopole* mpl = G4Monopole::Monopole();
  G4ProcessManager* pmanager = mpl->GetProcessManager();
  G4String particleName      = mpl->GetParticleName();
  if (verbose > 1)
    G4cout << "### CMSMonopolePhysics instantiates for " << particleName 
	   << " at " << mpl << " Process manager " << pmanager << G4endl;

  // defined monopole parameters and binning

  G4double magn = mpl->MagneticCharge();
  G4double emin = mpl->GetPDGMass()/20000.;
  if(emin < keV) emin = keV;
  G4double emax = 10.*TeV;
  G4int nbin = G4int(std::log10(emin/eV));
  emin       = std::pow(10.,G4double(nbin))*eV;
  if (verbose > 1)
    G4cout << "### Magnetic charge " << magn << " and electric charge " 
	   << mpl->GetPDGCharge() << "\n   # of bins in dE/dx table = " << nbin
	   << " in the range " << emin << ":" << emax << G4endl;

  if(magn != 0.0) {
    G4mplIonisation* mplioni = new G4mplIonisation(magn);
    mplioni->SetDEDXBinning(nbin);
    mplioni->SetMinKinEnergy(emin);
    mplioni->SetMaxKinEnergy(emax);
    pmanager->AddProcess(mplioni, -1, 1, 1);
  }
  if(mpl->GetPDGCharge() != 0.0) {
    G4hhIonisation* hhioni = new G4hhIonisation();
    hhioni->SetDEDXBinning(nbin);
    hhioni->SetMinKinEnergy(emin);
    hhioni->SetMaxKinEnergy(emax);
    pmanager->AddProcess(hhioni,  -1, 2, 2);
  }
}
