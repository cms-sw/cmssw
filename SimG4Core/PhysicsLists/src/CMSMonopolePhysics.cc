#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "SimG4Core/PhysicsLists/interface/MonopoleTransportation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"

#include "G4StepLimiter.hh"
#include "G4mplIonisation.hh"
#include "G4mplIonisationWithDeltaModel.hh"
#include "G4hMultipleScattering.hh"
#include "G4hIonisation.hh"
#include "G4hhIonisation.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

CMSMonopolePhysics::CMSMonopolePhysics(const HepPDT::ParticleDataTable * pdt,
				       const edm::ParameterSet & p) :
  G4VPhysicsConstructor("Monopole Physics")
{  
  verbose   = p.getUntrackedParameter<int>("Verbosity",0);
  magCharge = p.getUntrackedParameter<int>("MonopoleCharge",1);
  deltaRay  = p.getUntrackedParameter<bool>("MonopoleDeltaRay",true);
  multiSc   = p.getUntrackedParameter<bool>("MonopoleMultiScatter",false);
  transport = p.getUntrackedParameter<bool>("MonopoleTransport",true);
  double mass = p.getUntrackedParameter<double>("MonopoleMass",200);
  if (pdt && mass > 0.0) {
    int ii=0;
    for (HepPDT::ParticleDataTable::const_iterator p=pdt->begin(); 
	 p != pdt->end(); ++p,++ii) {
      HepPDT::ParticleData particle = (p->second);
      std::string particleName = (particle.name()).substr(0,8);
      if (strcmp(particleName.c_str(),"Monopole") == 0) {
	names.push_back(particle.name());
	masses.push_back(mass*CLHEP::GeV);
	elCharges.push_back((int)(particle.charge()));
	pdgEncodings.push_back(particle.pid());
	monopoles.push_back(nullptr);
	if (verbose > 0) G4cout << "CMSMonopolePhysics: Monopole[" << ii
				<< "] " << particleName << " Mass "
				<< particle.mass() << " GeV, Magnetic Charge "
				<< magCharge << ", Electric Charge "
				<< particle.charge() << G4endl;
      } else if(strcmp(particleName.c_str(),"AntiMono") == 0) {
	names.push_back(particle.name());
	masses.push_back(mass*CLHEP::GeV);
	elCharges.push_back((int)(particle.charge()));
	pdgEncodings.push_back(particle.pid());
	monopoles.push_back(nullptr);
	if (verbose > 0) G4cout << "CMSMonopolePhysics: Monopole[" << ii
				<< "] " << particleName << " Mass "
				<< particle.mass() << " GeV, Magnetic Charge "
				<< magCharge << ", Electric Charge "
				<< particle.charge() << G4endl; 
      }
    }
  }
  if (verbose > 0) G4cout << "CMSMonopolePhysics has " << names.size()
			  << " monopole candidates and delta Ray option " 
			  << deltaRay << G4endl;
}

CMSMonopolePhysics::~CMSMonopolePhysics() {}

void CMSMonopolePhysics::ConstructParticle() {
  
  for (unsigned int ii=0; ii<names.size(); ++ii) {
    // monopoles are created once in the master thread
    if (!monopoles[ii]) {
      G4int mc = (pdgEncodings[ii] >= 0 ) ? magCharge : -magCharge;
      Monopole* mpl = new Monopole(names[ii], pdgEncodings[ii], masses[ii],
                                   mc, elCharges[ii]);
      monopoles[ii] = mpl;
      if (verbose > 0) G4cout << "Create Monopole " << names[ii] 
			      << " of mass " << masses[ii]/CLHEP::GeV
			      << " GeV, magnetic charge " << mc 
			      << ", electric charge " << elCharges[ii]
			      << " and PDG encoding " << pdgEncodings[ii]
			      << " at " << monopoles[ii] << G4endl;
    }
  }
}

void CMSMonopolePhysics::ConstructProcess() {
  // Add standard EM Processes
  if (verbose > 0) {
    G4cout << "### CMSMonopolePhysics ConstructProcess()" << G4endl;
  }
  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

  for (unsigned int ii=0; ii<monopoles.size(); ++ii) {
    if (monopoles[ii]) {
      Monopole* mpl = monopoles[ii];
      G4ProcessManager *pmanager = mpl->GetProcessManager();
      if(!pmanager) {
        std::ostringstream o;
        o << "Monopole without a Process Manager";
        throw edm::Exception( edm::errors::Configuration, o.str().c_str());
        return;
      }

      G4double magn = mpl->MagneticCharge();
      G4double mass = mpl->GetPDGMass();
      if (verbose > 1) {
	G4cout << "### CMSMonopolePhysics instantiates for " 
	       << mpl->GetParticleName()
	       << " at " << mpl << " Mass " << mass/CLHEP::GeV 
	       << " GeV Mag " << magn  << " Process manager " << pmanager 
	       << G4endl;
      }
  
      if (magn != 0.0) {
        G4int idxt(0);
        pmanager->RemoveProcess(idxt);
        pmanager->AddProcess(new MonopoleTransportation(mpl,verbose),-1,0,0);
      }

      if (mpl->GetPDGCharge() != 0.0) {
	if (multiSc) {
	  G4hMultipleScattering* hmsc = new G4hMultipleScattering();
	  ph->RegisterProcess(hmsc, mpl);
	}
	G4hIonisation* hioni = new G4hIonisation();
	ph->RegisterProcess(hioni, mpl);
      }
      if(magn != 0.0) {
	G4mplIonisation* mplioni = new G4mplIonisation(magn);
	ph->RegisterProcess(mplioni, mpl);
      }
      pmanager->AddDiscreteProcess(new G4StepLimiter());
      if (verbose > 1) { pmanager->DumpInfo(); }
    }
  }
}
