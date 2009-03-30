///////////////////////////////////////////////////////////////////////////////
// File: CastorShowerLibrary.cc
// Description: Shower library for CASTOR calorimeter
//              Adapted from HFShowerLibrary
//
//  Wagner Carvalho
//
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Forward/interface/CastorShowerLibrary.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleTable.hh"
#include "Randomize.hh"
#include "CLHEP/Units/SystemOfUnits.h"

//#define DebugLog

CastorShowerLibrary::CastorShowerLibrary(std::string & name, edm::ParameterSet const & p) 
                                          : hf(0), evtInfo(0), emBranch(0), hadBranch(0),
                                            nMomBin(0), totEvents(0), evtPerBin(0) {
  
  initFile(p);
  
}

//=============================================================================================


CastorShowerLibrary::~CastorShowerLibrary() {
  if (hf)     hf->Close();
}


//=============================================================================================

void CastorShowerLibrary::initFile(edm::ParameterSet const & p) {
//////////////////////////////////////////////////////////
//
//  Init TFile and associated TBranch's of CASTOR Root file 
//  holding library events 
//
//////////////////////////////////////////////////////////

  //
  //  Read PSet for Castor shower library
  //

  edm::ParameterSet m_CS   = p.getParameter<edm::ParameterSet>("CastorShowerLibrary");
  edm::FileInPath fp       = m_CS.getParameter<edm::FileInPath>("FileName");
  std::string pTreeName    = fp.fullPath();
  std::string branchEvInfo = m_CS.getUntrackedParameter<std::string>("BranchEvt");
  std::string branchEM     = m_CS.getUntrackedParameter<std::string>("BranchEM");
  std::string branchHAD    = m_CS.getUntrackedParameter<std::string>("BranchHAD");
  verbose                  = m_CS.getUntrackedParameter<bool>("Verbosity",false);

  // Open TFile 
  if (pTreeName.find(".") == 0) pTreeName.erase(0,2);
  const char* nTree = pTreeName.c_str();
  hf                = TFile::Open(nTree);

  // Check that TFile has been successfully opened
  if (!hf->IsOpen()) { 
     edm::LogError("CastorShower") << "CastorShowerLibrary: opening " << nTree << " failed";
     throw cms::Exception("Unknown", "CastorShowerLibrary") 
                                   << "Opening of " << pTreeName << " fails\n";
  } else {
     edm::LogInfo("CastorShower")  << "CastorShowerLibrary: opening " << nTree << " successfully"; 
  }

  // Check for the TBranch holding EventInfo in "Events" TTree
  TTree* event = (TTree *) hf ->Get("CastorCherenkovPhotons");
  if (event) {
     evtInfo = (TBranchObject *) event->GetBranch(branchEvInfo.c_str());
     if (evtInfo) {
        loadEventInfo(evtInfo);
     } else {
        edm::LogError("CastorShower") << "CastorShowerLibrary: CastorShowerLibrayEventInfo"
				      << " Branch does not exist in Event";
        throw cms::Exception("Unknown", "CastorShowerLibrary") << "Event information absent\n";
     }
  } else {
     edm::LogError("CastorShower") << "CastorShowerLibrary: Events Tree does not exist";
     throw cms::Exception("Unknown", "CastorShowerLibrary") << "Events tree absent\n";
  }
  
  // Get EM and HAD Branchs
  emBranch         = (TBranchObject *) event->GetBranch(branchEM.c_str());
  if (verbose) emBranch->Print();
  hadBranch        = (TBranchObject *) event->GetBranch(branchHAD.c_str());
  if (verbose) hadBranch->Print();
  edm::LogInfo("CastorShower") << "CastorShowerLibrary: Branch " << branchEM 
			       << " has " << emBranch->GetEntries() 
			       << " entries and Branch " << branchHAD 
			       << " has " << hadBranch->GetEntries() 
			       << " entries";

}

//=============================================================================================

void CastorShowerLibrary::loadEventInfo(TBranchObject* branch) {
//////////////////////////////////////////////////////////
//
//  Get EventInfo from the "TBranch* branch" of Root file 
//  holding library events 
//
//  Based on HFShowerLibrary::loadEventInfo
//
//////////////////////////////////////////////////////////

  eventInfo = new CastorShowerLibraryInfo();
  branch->SetAddress(&eventInfo);
  branch->GetEntry(0);
  // Initialize shower library general parameters
  totEvents   = eventInfo->getNEv();
  nMomBin     = eventInfo->getNEnBins();
  evtPerBin   = eventInfo->getNEvPerBin();
  pmom        = eventInfo->getEnergies();
  // Convert from GeV to MeV
  for (unsigned int i=0; i<pmom.size(); i++) pmom[i] *= GeV;
  
  edm::LogInfo("CastorShower") << " CastorShowerLibrary::loadEventInfo : " 
			       << "\n \n Total number of events:  " << totEvents 
			       <<    "\n  Number of energy bins:  " << nMomBin 
			       <<    "\n   Number of events/bin:  " << evtPerBin << "\n";
  
  for (unsigned int i=0; i<nMomBin+1; i++)
     edm::LogInfo("CastorShower") << "CastorShowerLibrary: pmom[" << i << "] = "
			          << pmom[i]/GeV << " GeV";

}

//=============================================================================================

void CastorShowerLibrary::initParticleTable(G4ParticleTable * theParticleTable) {
////////////////////////////////////////////////////////
//
//  Set particle codes according to PDG encoding 
//
//  Based on HFShowerLibrary::initRun
//
////////////////////////////////////////////////////////

  G4String particleName;

  edm::LogInfo("CastorShower") << "CastorShowerLibrary::initParticleTable"
                               << " ***  Accessing PDGEncoding  ***" ;

  emPDG       = theParticleTable->FindParticle(particleName="e-")->GetPDGEncoding();
  epPDG       = theParticleTable->FindParticle(particleName="e+")->GetPDGEncoding();
  gammaPDG    = theParticleTable->FindParticle(particleName="gamma")->GetPDGEncoding();
  pi0PDG      = theParticleTable->FindParticle(particleName="pi0")->GetPDGEncoding();
  etaPDG      = theParticleTable->FindParticle(particleName="eta")->GetPDGEncoding();
  nuePDG      = theParticleTable->FindParticle(particleName="nu_e")->GetPDGEncoding();
  numuPDG     = theParticleTable->FindParticle(particleName="nu_mu")->GetPDGEncoding();
  nutauPDG    = theParticleTable->FindParticle(particleName="nu_tau")->GetPDGEncoding();
  anuePDG     = theParticleTable->FindParticle(particleName="anti_nu_e")->GetPDGEncoding();
  anumuPDG    = theParticleTable->FindParticle(particleName="anti_nu_mu")->GetPDGEncoding();
  anutauPDG   = theParticleTable->FindParticle(particleName="anti_nu_tau")->GetPDGEncoding();
  geantinoPDG = theParticleTable->FindParticle(particleName="geantino")->GetPDGEncoding();
  mumPDG      = theParticleTable->FindParticle(particleName="mu-")->GetPDGEncoding();
  mupPDG      = theParticleTable->FindParticle(particleName="mu+")->GetPDGEncoding();
  
//#ifdef DebugLog
  LogDebug("CastorShower") << "CastorShowerLibrary: Particle codes for e- = " << emPDG
		           << ", e+ = " << epPDG << ", gamma = " << gammaPDG 
		           << ", pi0 = " << pi0PDG << ", eta = " << etaPDG
		           << ", geantino = " << geantinoPDG << "\n        nu_e = "
		           << nuePDG << ", nu_mu = " << numuPDG << ", nu_tau = "
		           << nutauPDG << ", anti_nu_e = " << anuePDG
		           << ", anti_nu_mu = " << anumuPDG << ", anti_nu_tau = "
		           << anutauPDG << ", mu- = " << mumPDG << ", mu+ = "
		           << mupPDG;
//#endif

    edm::LogInfo("CastorShower") << "  *****   Successfully called:  " 
                                 << "CastorShowerLibrary::initParticleTable()   ***** " ;

}


//=============================================================================================

CastorShowerEvent CastorShowerLibrary::getShowerHits(G4Step * aStep, bool & ok) {

  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  // G4StepPoint * postStepPoint = aStep->GetPostStepPoint(); 
  G4Track *     track         = aStep->GetTrack();
  // Get Z-direction 
  const G4DynamicParticle *aParticle = track->GetDynamicParticle();
  G4ThreeVector               momDir = aParticle->GetMomentumDirection();
  //  double mom = aParticle->GetTotalMomentum();

  G4ThreeVector hitPoint = preStepPoint->GetPosition();   
  G4String      partType = track->GetDefinition()->GetParticleName();
  int           parCode  = track->GetDefinition()->GetPDGEncoding();

  CastorShowerEvent hit;
  hit.Clear();
  
  ok = false;
  if (parCode == pi0PDG   || parCode == etaPDG    || parCode == nuePDG  ||
      parCode == numuPDG  || parCode == nutauPDG  || parCode == anuePDG ||
      parCode == anumuPDG || parCode == anutauPDG || parCode == geantinoPDG) 
    return hit;
  ok = true;

  double pin    = preStepPoint->GetTotalEnergy();

  // Replace "interpolation/extrapolation" by new method "select" that just randomly 
  // selects a record from the appropriate energy bin and fills its content to  
  // "showerEvent" instance of class CastorShowerEvent
  
  if (parCode == emPDG || parCode == epPDG || parCode == gammaPDG ) {
    select(0, pin);
    // if (pin<pmom[nMomBin-1]) {
    //   interpolate(0, pin);
    // } else {
    //   extrapolate(0, pin);
    // }
  } else {
    select(1, pin);
    // if (pin<pmom[nMomBin-1]) {
    //   interpolate(1, pin);
    // } else {
    //   extrapolate(1, pin);
    // }
  }
    
  hit = (*showerEvent);
  return hit;

}

//=============================================================================================

void CastorShowerLibrary::getRecord(int type, int record) {
//////////////////////////////////////////////////////////////
//
//  Retrieve event # "record" from the library and stores it   
//  into  vector<CastorShowerHit> showerHit;
//
//  Based on HFShowerLibrary::getRecord
//
//  Modified 02/02/09 by W. Carvalho
//
//////////////////////////////////////////////////////////////

  edm::LogInfo("CastorShower") << "CastorShowerLibrary::getRecord: ";
  
  int nrc  = record-1;
  int nHit = 0;
  showerEvent = new CastorShowerEvent();
  if (type > 0) {
    hadBranch->SetAddress(&showerEvent);
    hadBranch->GetEntry(nrc);
  } else {
    emBranch->SetAddress(&showerEvent);
    emBranch->GetEntry(nrc);
  }
  nHit = showerEvent->getNhit();

#ifdef DebugLog

  LogDebug("CastorShower") << "CastorShowerLibrary::getRecord: Record " << record
		           << " of type " << type << " with " << nHit 
		           << " CastorShowerHits";

#endif

  edm::LogInfo("CastorShower") << "CastorShowerLibrary::getRecord: Record " << record
		               << " of type " << type << " with " << nHit 
		               << " CastorShowerHits";

}

//=======================================================================================

void CastorShowerLibrary::select(int type, double pin) {
////////////////////////////////////////////////////////
//
//  Selects an event from the library based on 
//
//    type:   0 --> em
//           >0 --> had
//    pin :  momentum
//
//  Created 30/01/09 by W. Carvalho
//
////////////////////////////////////////////////////////

  int irec;                         // to hold record number
  double r = G4UniformRand();       // to randomly select within an energy bin (r=[0,1])

  // Randomly select a record from the right energy bin in the library, 
  // based on track momentum (pin) 
   
  //   pin < pmom[MIN]
  if (pin<pmom[0]) {
     edm::LogWarning("CastorShower") << "CastorShowerLibrary: Warning, pin = " << pin 
                                     << " less than minimum pmom " << pmom[0] << " in library." 
                                     << " For the moment, selecting hit from the first bin" ;
     irec = int(evtPerBin*r) + 1;
  //   pin > pmom[MAX]
  } else if (pin>pmom[nMomBin]) {
    edm::LogWarning("CastorShower") << "CastorShowerLibrary: Warning, pin = " << pin 
                                    << " greater than maximum pmom " << pmom[nMomBin] << " in library." 
                                    << " For the moment, selecting hit from the last bin";
    irec = (nMomBin-1)*evtPerBin + int(evtPerBin*r) + 1;
  //   pmom[MIN] < pin < pmom[MAX]
  } else {
     for (unsigned int j=0; j<nMomBin; j++) {
        if (pin >= pmom[j] && pin < pmom[j+1]) {
	   irec = j*evtPerBin + int(evtPerBin*r) + 1 ;
	   if (irec<0) {
	      edm::LogWarning("CastorShower") << "CastorShowerLibrary:: Illegal irec = "
				              << irec << " now set to 0";
	      irec = 0;
	   } else if (irec > totEvents) {
	      edm::LogWarning("CastorShower") << "CastorShowerLibrary:: Illegal irec = "
				              << irec << " now set to "<< totEvents;
	      irec = totEvents;
	   }
        }
     }
  }

  edm::LogInfo("CastorShower") << "CastorShowerLibrary:: Select record " << irec 
                               << " of type " << type ; 
#ifdef DebugLog
  LogDebug("CastorShower") << "CastorShowerLibrary:: Select record " << irec; 
                           << " of type " << type ; 
#endif

  //  Retrieve record number "irec" from the library
  getRecord (type, irec);
  
}
