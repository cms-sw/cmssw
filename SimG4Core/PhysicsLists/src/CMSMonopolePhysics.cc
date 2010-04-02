#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "SimG4Core/Physics/interface/G4Monopole.hh"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4StepLimiter.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "CMSG4mplIonisation.hh"
#include "G4hhIonisation.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

CMSMonopolePhysics::CMSMonopolePhysics(const HepPDT::ParticleDataTable * pdt,
				       G4double chg, G4int ver) : verbose(ver),
								  magn(0) {
  
  if (pdt) {
    int ii=0;
    for (HepPDT::ParticleDataTable::const_iterator p=pdt->begin(); 
	 p != pdt->end(); ++p,++ii) {
      HepPDT::ParticleData particle = (p->second);
      std::string particleName = (particle.name()).substr(0,8);
      if (strcmp(particleName.c_str(),"Monopole") == 0) {
	names.push_back(particle.name());
	masses.push_back((particle.mass())*CLHEP::GeV);
	elCharges.push_back((int)(particle.charge()));
	pdgEncodings.push_back(particle.pid());
	monopoles.push_back(0);
      }
    }
  }
  magCharge = (int)chg;
  if (verbose > 0) G4cout << "CMSMonopolePhysics has " << names.size()
			  << " monopole candidates" << G4endl;
}

CMSMonopolePhysics::~CMSMonopolePhysics() {}

void CMSMonopolePhysics::ConstructParticle() {
  
  for (unsigned int ii=0; ii<names.size(); ++ii) {
    if (!monopoles[ii]) {
      G4Monopole* mpl = new G4Monopole(names[ii], pdgEncodings[ii],
				       masses[ii], magCharge, elCharges[ii]);;
      magn = mpl->MagneticCharge();
      monopoles[ii] = mpl;
      if (verbose > 0) G4cout << "Create G4Monopole " << names[ii] 
			      << " of mass " << masses[ii]/CLHEP::GeV
			      << " GeV, magnetic charge " << magCharge 
			      << ", electric charge " << elCharges[ii]
			      << " and PDG encoding " << pdgEncodings[ii]
			      << " at " << monopoles[ii] << G4endl;
    }
  }
}

void CMSMonopolePhysics::ConstructProcess() {
  // Add standard EM Processes

  if (verbose > 0)
    G4cout << "### CMSMonopolePhysics ConstructProcess()" << G4endl;
  for (unsigned int ii=0; ii<monopoles.size(); ++ii) {
    if (monopoles[ii]) {
      G4ParticleDefinition* particle = monopoles[ii];
      G4ProcessManager* pmanager = particle->GetProcessManager();
      G4String particleName = particle->GetParticleName();
      if (verbose > 1)
	G4cout << "### CMSMonopolePhysics instantiates for " << particleName 
	       << " at " << particle << " Mass " 
	       << particle->GetPDGMass()/CLHEP::GeV << " GeV Mag " << magn
	       << " Process manager " << pmanager << G4endl;

      // defined monopole parameters and binning

      G4double emin = particle->GetPDGMass()/20000.;
      if(emin < CLHEP::keV) emin = CLHEP::keV;
      G4double emax = 10.*CLHEP::TeV;
      G4int nbin = G4int(std::log10(emin/CLHEP::eV));
      emin       = std::pow(10.,G4double(nbin))*CLHEP::eV;

      nbin = G4int(std::log10(emax/emin));
      if (nbin < 1) nbin = 1;
      nbin *= 10;

      if (verbose > 1)
	G4cout << "### Magnetic charge " << magn << " and electric charge " 
	       << particle->GetPDGCharge() <<"\n   # of bins in dE/dx table = "
	       << nbin << " in the range " << emin << ":" << emax << G4endl;

      if(magn != 0.0) {
	CMSG4mplIonisation* mplioni = new CMSG4mplIonisation(magn);
	mplioni->SetDEDXBinning(nbin);
	mplioni->SetMinKinEnergy(emin);
	mplioni->SetMaxKinEnergy(emax);
	pmanager->AddProcess(mplioni, -1, 1, 1);
      }
      if(particle->GetPDGCharge() != 0.0) {
	G4hhIonisation* hhioni = new G4hhIonisation();
	hhioni->SetDEDXBinning(nbin);
	hhioni->SetMinKinEnergy(emin);
	hhioni->SetMaxKinEnergy(emax);
	pmanager->AddProcess(hhioni,  -1, 2, 2);
      }
      if (verbose > 1) pmanager->DumpInfo();
    }
  }
}
