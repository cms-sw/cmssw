///////////////////////////////////////////////////////////////////////////////
// File: CFCShowerLibrary.cc
// Description: Shower library for Combined Forward Calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CFCShowerLibrary.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4VPhysicalVolume.hh"
#include "G4NavigationHistory.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "Randomize.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#define DebugLog

CFCShowerLibrary::CFCShowerLibrary(edm::ParameterSet const & p,
				   std::vector<double> & gp) : gpar(gp) {
  

  edm::ParameterSet m_CFC = p.getParameter<edm::ParameterSet>("CFCSD");
  edm::FileInPath fp      = m_CFC.getParameter<edm::FileInPath>("FileName");
  std::string pTreeName   = fp.fullPath();
  if (pTreeName.find(".") == 0) pTreeName.erase(0,2);
  const char* nTree       = pTreeName.c_str();
  hfile                   = TFile::Open(nTree);

  if (!hfile->IsOpen()) { 
    edm::LogError("CFCShower") << "CFCShowerLibrary: opening " << nTree 
			       << " failed";
    throw cms::Exception("Unknown", "CFCShowerLibrary") 
      << "Opening of " << pTreeName << " fails\n";
  } else {
    edm::LogInfo("CFCShower") << "CFCShowerLibrary: opening " << nTree 
			      << " successfully"; 
  }

  emPDG = epPDG = gammaPDG = 0;
  pi0PDG = etaPDG = nuePDG = numuPDG = nutauPDG= 0;
  anuePDG= anumuPDG = anutauPDG = geantinoPDG = 0;
}

CFCShowerLibrary::~CFCShowerLibrary() {
  if (hfile)  hfile->Close();
}

void CFCShowerLibrary::initRun(G4ParticleTable * theParticleTable) {

  G4String parName;
  emPDG = theParticleTable->FindParticle(parName="e-")->GetPDGEncoding();
  epPDG = theParticleTable->FindParticle(parName="e+")->GetPDGEncoding();
  gammaPDG = theParticleTable->FindParticle(parName="gamma")->GetPDGEncoding();
  pi0PDG = theParticleTable->FindParticle(parName="pi0")->GetPDGEncoding();
  etaPDG = theParticleTable->FindParticle(parName="eta")->GetPDGEncoding();
  nuePDG = theParticleTable->FindParticle(parName="nu_e")->GetPDGEncoding();
  numuPDG = theParticleTable->FindParticle(parName="nu_mu")->GetPDGEncoding();
  nutauPDG= theParticleTable->FindParticle(parName="nu_tau")->GetPDGEncoding();
  anuePDG = theParticleTable->FindParticle(parName="anti_nu_e")->GetPDGEncoding();
  anumuPDG= theParticleTable->FindParticle(parName="anti_nu_mu")->GetPDGEncoding();
  anutauPDG= theParticleTable->FindParticle(parName="anti_nu_tau")->GetPDGEncoding();
  geantinoPDG= theParticleTable->FindParticle(parName="geantino")->GetPDGEncoding();
#ifdef DebugLog
  edm::LogInfo("CFCShower") << "CFCShowerLibrary: Particle codes for e- = " 
			    << emPDG << ", e+ = " << epPDG << ", gamma = " 
			    << gammaPDG << ", pi0 = " << pi0PDG << ", eta = " 
			    << etaPDG << ", geantino = " << geantinoPDG 
			    << "\n        nu_e = " << nuePDG << ", nu_mu = " 
			    << numuPDG << ", nu_tau = " << nutauPDG 
			    << ", anti_nu_e = " << anuePDG << ", anti_nu_mu = " 
			    << anumuPDG << ", anti_nu_tau = " << anutauPDG;
#endif
}


std::vector<CFCShowerLibrary::Hit> CFCShowerLibrary::getHits(G4Step *,
							     bool & ) {


  std::vector<CFCShowerLibrary::Hit> hit;

  return hit;
}
