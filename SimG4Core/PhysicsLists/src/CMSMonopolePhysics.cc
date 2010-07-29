#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"

#include "G4StepLimiter.hh"
#include "G4Transportation.hh"
#include "G4MonopoleTransportation.hh"
#include "CMSG4mplIonisation.hh"
#include "G4mplIonisationWithDeltaModel.hh"
#include "G4hMultipleScattering.hh"
#include "G4hIonisation.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

CMSMonopolePhysics::CMSMonopolePhysics(const HepPDT::ParticleDataTable * pdt,
				       sim::FieldBuilder * fB_, G4double chg, 
				       G4int ver) : 
  G4VPhysicsConstructor("Monopole Physics"), fieldBuilder(fB_),  verbose(ver) {
  
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
	if (verbose > 0) G4cout << "CMSMonopolePhysics: Monopole[" << ii
				<< "] " << particleName << " Mass "
				<< particle.mass() << " GeV, Charge "
				<< particle.charge() << G4endl;
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
      G4Monopole* mpl = monopoles[ii];
      G4ProcessManager* pmanager = new G4ProcessManager(mpl);
      mpl->SetProcessManager(pmanager);

      G4String particleName = mpl->GetParticleName();
      G4double magn         = mpl->MagneticCharge();
      if (verbose > 1)
	G4cout << "### CMSMonopolePhysics instantiates for " << particleName 
	       << " at " << mpl << " Mass " << mpl->GetPDGMass()/CLHEP::GeV 
	       << " GeV Mag " << magn  << " Process manager " << pmanager 
	       << G4endl;

      // defined monopole parameters and binning

      G4double emin = mpl->GetPDGMass()/20000.;
      if(emin < CLHEP::keV) emin = CLHEP::keV;
      G4double emax = 10.*CLHEP::TeV;
      G4int nbin = G4int(std::log10(emin/CLHEP::eV));
      emin       = std::pow(10.,G4double(nbin))*CLHEP::eV;

      nbin = G4int(std::log10(emax/emin));
      if (nbin < 1) nbin = 1;
      nbin *= 10;

      G4int idx = 1;
      if (verbose > 1)
	G4cout << "### Magnetic charge " << magn << " and electric charge " 
	       << mpl->GetPDGCharge() <<"\n   # of bins in dE/dx table = "
	       << nbin << " in the range " << emin << ":" << emax << G4endl;
  
      if(magn == 0.0) {
	pmanager->AddProcess( new G4Transportation(verbose), -1, 0, 0);
      } else {
	pmanager->AddProcess( new G4MonopoleTransportation(mpl,fieldBuilder,verbose), -1, 0, 0);
      }

      if(mpl->GetPDGCharge() != 0.0) {
	//G4hMultipleScattering* hmsc = new G4hMultipleScattering();
	//pmanager->AddProcess(hmsc,  -1, idx, idx);
	//++idx;
	G4hIonisation* hhioni = new G4hIonisation();
	hhioni->SetDEDXBinning(nbin);
	hhioni->SetMinKinEnergy(emin);
	hhioni->SetMaxKinEnergy(emax);
	pmanager->AddProcess(hhioni,  -1, idx, idx);
	++idx;
      }
      if(magn != 0.0) {
	CMSG4mplIonisation* mplioni = new CMSG4mplIonisation(magn);
	mplioni->SetDEDXBinning(nbin);
	mplioni->SetMinKinEnergy(emin);
	mplioni->SetMaxKinEnergy(emax);
	G4mplIonisationWithDeltaModel* mod = 
	  new G4mplIonisationWithDeltaModel(magn,"PAI");
	mplioni->AddEmModel(0,mod,mod);
	pmanager->AddProcess(mplioni, -1, idx, idx);
	++idx;
      }
      pmanager->AddProcess( new G4StepLimiter(),  -1, -1, idx);
      if (verbose > 1) pmanager->DumpInfo();
    }
  }
}
