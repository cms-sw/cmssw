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
#include "G4PhysicalConstants.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

CastorShowerLibrary::CastorShowerLibrary(std::string & name, edm::ParameterSet const & p) 
                                          : hf(0), evtInfo(0), emBranch(0), hadBranch(0),
                                            nMomBin(0), totEvents(0), evtPerBin(0),
                                            nBinsE(0),nBinsEta(0),nBinsPhi(0),
                                            nEvtPerBinE(0),nEvtPerBinEta(0),nEvtPerBinPhi(0),
                                            SLenergies(),SLetas(),SLphis() {
  
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
        edm::LogError("CastorShower") << "CastorShowerLibrary: " << branchEvInfo.c_str()
                                      << " Branch does not exit in Event";
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

  totEvents   = eventInfo->Energy.getNEvts();
//  nMomBin     = eventInfo->Energy.getNBins();
//  evtPerBin   = eventInfo->Energy.getNEvtPerBin();
//  pmom        = eventInfo->Energy.getBin();
  nBinsE        = eventInfo->Energy.getNBins();
  nEvtPerBinE   = eventInfo->Energy.getNEvtPerBin();
  SLenergies    = eventInfo->Energy.getBin();
  nBinsEta      = eventInfo->Eta.getNBins();
  nEvtPerBinEta = eventInfo->Eta.getNEvtPerBin();
  SLetas        = eventInfo->Eta.getBin();
  nBinsPhi      = eventInfo->Phi.getNBins();
  nEvtPerBinPhi = eventInfo->Phi.getNEvtPerBin();
  SLphis        = eventInfo->Phi.getBin();
  
  // Convert from GeV to MeV
  for (unsigned int i=0; i<SLenergies.size(); i++) SLenergies[i] *= GeV;
  
  edm::LogInfo("CastorShower") << " CastorShowerLibrary::loadEventInfo : " 
			       << "\n \n Total number of events     :  " << totEvents 
			       <<    "\n   Number of bins  (E)       :  " << nBinsE
			       <<    "\n   Number of events/bin (E)  :  " << nEvtPerBinE
			       <<    "\n   Number of bins  (Eta)       :  " << nBinsEta
			       <<    "\n   Number of events/bin (Eta)  :  " << nEvtPerBinEta 
			       <<    "\n   Number of bins  (Phi)       :  " << nBinsPhi
			       <<    "\n   Number of events/bin (Phi)  :  " << nEvtPerBinPhi << "\n";
  
  for (unsigned int i=0; i<nBinsE; i++)
     edm::LogInfo("CastorShower") << "CastorShowerLibrary: SLenergies[" << i << "] = "
			          << SLenergies[i]/GeV << " GeV";
  for (unsigned int i=0; i<nBinsEta; i++)
     edm::LogInfo("CastorShower") << "CastorShowerLibrary: SLetas[" << i << "] = "
			          << SLetas[i];
  for (unsigned int i=0; i<nBinsPhi; i++)
     edm::LogInfo("CastorShower") << "CastorShowerLibrary: SLphis[" << i << "] = "
			          << SLphis[i] << " rad";
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
  //  double etain  = momDir.getEta();
  //double phiin  = momDir.getPhi();
  
  double zint = hitPoint.z();
  double R=sqrt(hitPoint.x()*hitPoint.x() + hitPoint.y()*hitPoint.y());
  double theta = atan2(R,std::abs(zint));
  double phiin = atan2(hitPoint.y(),hitPoint.x());
  double etain = -std::log(std::tan((pi-theta)*0.5));

  // Replace "interpolation/extrapolation" by new method "select" that just randomly 
  // selects a record from the appropriate energy bin and fills its content to  
  // "showerEvent" instance of class CastorShowerEvent
  
  if (parCode == emPDG || parCode == epPDG || parCode == gammaPDG ) {
    select(0, pin, etain, phiin);
    // if (pin<SLenergies[nBinsE-1]) {
    //   interpolate(0, pin);
    // } else {
    //   extrapolate(0, pin);
    // }
  } else {
    select(1, pin, etain, phiin);
    // if (pin<SLenergies[nBinsE-1]) {
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

#ifdef DebugLog
  LogDebug("CastorShower") << "CastorShowerLibrary::getRecord: ";
#endif  
  int nrc  = record;
  showerEvent = new CastorShowerEvent();
  if (type > 0) {
    hadBranch->SetAddress(&showerEvent);
    hadBranch->GetEntry(nrc);
  } else {
    emBranch->SetAddress(&showerEvent);
    emBranch->GetEntry(nrc);
  }

#ifdef DebugLog
  int nHit = showerEvent->getNhit();
  LogDebug("CastorShower") << "CastorShowerLibrary::getRecord: Record " << record
		           << " of type " << type << " with " << nHit 
		           << " CastorShowerHits";

#endif

}

//=======================================================================================

void CastorShowerLibrary::select(int type, double pin, double etain, double phiin) {
////////////////////////////////////////////////////////
//
//  Selects an event from the library based on 
//
//    type:   0 --> em
//           >0 --> had
//    pin  :  momentum
//    etain:  eta (if not given, disregard the eta binning
//    phiin:  phi (if not given, disregard the phi binning
//
//  Created 30/01/09 by W. Carvalho
//
////////////////////////////////////////////////////////

  int irec;                         // to hold record number
  double r = G4UniformRand();       // to randomly select within an energy bin (r=[0,1])

  // Randomly select a record from the right energy bin in the library, 
  // based on track momentum (pin) 
   
  //   pin < SLenergies[MIN]
/*
  if (pin<SLenergies[0]) {
     edm::LogWarning("CastorShower") << "CastorShowerLibrary: Warning, pin = " << pin 
                                     << " less than minimum SLenergies " << SLenergies[0] << " in library." 
                                     << " For the moment, selecting hit from the first bin" ;
     irec = int(nEvtPerBinE*r) + 1;
  //   pin > SLenergies[MAX]
  } else if (pin>SLenergies[nBinsE]) {

// This part needs rethinking because the last element of SLenergies is no longer the upper limit of the bins
    edm::LogWarning("CastorShower") << "CastorShowerLibrary: Warning, pin = " << pin 
                                    << " greater than maximum SLenergies " << SLenergies[nBinsE] << " in library." 
                                    << " For the moment, selecting hit from the last bin";
    irec = (nBinsE-1)*nEvtPerBinE + int(evtPerBin*r) + 1;
  //   SLenergies[MIN] < pin < SLenergies[MAX]
  } else {
     for (unsigned int j=0; j<nBinsE; j++) {
        if (pin >= SLenergies[j] && pin < SLenergies[j+1]) {
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
*/
  int ienergy = FindEnergyBin(pin);
  int ieta    = FindEtaBin(etain);
#ifdef DebugLog
  if (verbose) edm::LogInfo("CastorShower") << " ienergy = " << ienergy ;
  if (verbose) edm::LogInfo("CastorShower") << " ieta = " << ieta;
#endif

  int iphi;
  double phiMin = 0. ;
  double phiMax = M_PI/4 ;     // 45 * (pi/180)  rad 
  if(phiin < phiMin) phiin = phiin + M_PI ;
  if(phiin >= phiMin && phiin < phiMax) {
     iphi = FindPhiBin(phiin) ;
  } else {
     double remainder = fmod(phiin , M_PI/4) ;
     phiin = phiMin + remainder ;
     iphi = FindPhiBin(phiin) ;
  }
#ifdef DebugLog
  if (verbose) edm::LogInfo("CastorShower") << " iphi = " << iphi;
#endif
  if (ienergy==-1) ienergy=0;    // if pin < first bin, choose an event in the first one
  if (ieta==-1)    ieta=0; // idem for eta
  if (iphi!=-1) irec = int(nEvtPerBinE*ienergy+nEvtPerBinEta*ieta+nEvtPerBinPhi*(iphi+r));
  else irec = int(nEvtPerBinE*(ienergy+r));

#ifdef DebugLog
  edm::LogInfo("CastorShower") << "CastorShowerLibrary:: Select record " << irec 
                               << " of type " << type ; 
#endif

  //  Retrieve record number "irec" from the library
  getRecord (type, irec);
  
}
int CastorShowerLibrary::FindEnergyBin(double energy) {
  //
  // returns the integer index of the energy bin, taken from SLenergies vector
  // returns -1 if ouside valid range
  //
  if (energy >= SLenergies.back()) return SLenergies.size()-1;

  unsigned int i = 0;
  for(;i<SLenergies.size()-1;i++)
    if (energy >= SLenergies.at(i) && energy < SLenergies.at(i+1)) return (int)i;

  // now i points to the last but 1 bin
  if (energy>=SLenergies.at(i)) return (int)i;
  // energy outside bin range
  return -1;
}
int CastorShowerLibrary::FindEtaBin(double eta) {
  //
  // returns the integer index of the eta bin, taken from SLetas vector
  // returns -1 if ouside valid range
  //
  if (eta>=SLetas.back()) return SLetas.size()-1;
  unsigned int i = 0;
  for(;i<SLetas.size()-1;i++)
     if (eta >= SLetas.at(i) && eta < SLetas.at(i+1)) return (int)i;
  // now i points to the last but 1 bin
  if (eta>=SLetas.at(i)) return (int)i;
  // eta outside bin range
  return -1;
}
int CastorShowerLibrary::FindPhiBin(double phi) {
  //
  // returns the integer index of the phi bin, taken from SLphis vector
  // returns -1 if ouside valid range
  //
  // needs protection in case phi is outside range -pi,pi
  //
  if (phi>=SLphis.back()) return SLphis.size()-1;
  unsigned int i = 0;
  for(;i<SLphis.size()-1;i++)
     if (phi >= SLphis.at(i) && phi < SLphis.at(i+1)) return (int)i;
  // now i points to the last but 1 bin
  if (phi>=SLphis.at(i)) return (int)i;
  // phi outside bin range
  return -1;
}
