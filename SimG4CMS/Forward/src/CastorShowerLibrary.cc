///////////////////////////////////////////////////////////////////////////////
// File: CastorShowerLibrary.cc
// Description: Shower library for CASTOR calorimeter
//              Adapted from HFShowerLibrary
//
//  Wagner Carvalho
//
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Forward/interface/CastorShowerLibrary.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleTable.hh"
#include "Randomize.hh"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

//ROOT
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

//#define DebugLog

CastorShowerLibrary::CastorShowerLibrary(const std::string& name, edm::ParameterSet const& p)
    : hf(nullptr),
      emBranch(nullptr),
      hadBranch(nullptr),
      verbose(false),
      nMomBin(0),
      totEvents(0),
      evtPerBin(0),
      nBinsE(0),
      nBinsEta(0),
      nBinsPhi(0),
      nEvtPerBinE(0),
      nEvtPerBinEta(0),
      nEvtPerBinPhi(0),
      SLenergies(),
      SLetas(),
      SLphis() {
  initFile(p);
}

//=============================================================================================

CastorShowerLibrary::~CastorShowerLibrary() {
  if (hf)
    hf->Close();
}

//=============================================================================================

void CastorShowerLibrary::initFile(edm::ParameterSet const& p) {
  //////////////////////////////////////////////////////////
  //
  //  Init TFile and associated TBranch's of CASTOR Root file
  //  holding library events
  //
  //////////////////////////////////////////////////////////

  //
  //  Read PSet for Castor shower library
  //

  edm::ParameterSet m_CS = p.getParameter<edm::ParameterSet>("CastorShowerLibrary");
  edm::FileInPath fp = m_CS.getParameter<edm::FileInPath>("FileName");
  std::string pTreeName = fp.fullPath();
  std::string branchEvInfo = m_CS.getUntrackedParameter<std::string>("BranchEvt");
  std::string branchEM = m_CS.getUntrackedParameter<std::string>("BranchEM");
  std::string branchHAD = m_CS.getUntrackedParameter<std::string>("BranchHAD");
  verbose = m_CS.getUntrackedParameter<bool>("Verbosity", false);

  // Open TFile
  if (pTreeName.find('.') == 0)
    pTreeName.erase(0, 2);
  const char* nTree = pTreeName.c_str();
  hf = TFile::Open(nTree);

  // Check that TFile has been successfully opened
  if (!hf->IsOpen()) {
    edm::LogError("CastorShower") << "CastorShowerLibrary: opening " << nTree << " failed";
    throw cms::Exception("Unknown", "CastorShowerLibrary") << "Opening of " << pTreeName << " fails\n";
  } else {
    edm::LogVerbatim("CastorShower") << "CastorShowerLibrary: opening " << nTree << " successfully";
  }

  // Check for the TBranch holding EventVerbatim in "Events" TTree
  TTree* event = hf->Get<TTree>("CastorCherenkovPhotons");
  if (event) {
    auto evtInfo = event->GetBranch(branchEvInfo.c_str());
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
  emBranch = event->GetBranch(branchEM.c_str());
  if (verbose)
    emBranch->Print();
  hadBranch = event->GetBranch(branchHAD.c_str());
  if (verbose)
    hadBranch->Print();
  edm::LogVerbatim("CastorShower") << "CastorShowerLibrary: Branch " << branchEM << " has " << emBranch->GetEntries()
                                   << " entries and Branch " << branchHAD << " has " << hadBranch->GetEntries()
                                   << " entries";
}

//=============================================================================================

void CastorShowerLibrary::loadEventInfo(TBranch* branch) {
  //////////////////////////////////////////////////////////
  //
  //  Get EventInfo from the "TBranch* branch" of Root file
  //  holding library events
  //
  //  Based on HFShowerLibrary::loadEventInfo
  //
  //////////////////////////////////////////////////////////

  CastorShowerLibraryInfo tempInfo;
  auto* eventInfo = &tempInfo;
  branch->SetAddress(&eventInfo);
  branch->GetEntry(0);
  // Initialize shower library general parameters

  totEvents = eventInfo->Energy.getNEvts();
  nBinsE = eventInfo->Energy.getNBins();
  nEvtPerBinE = eventInfo->Energy.getNEvtPerBin();
  SLenergies = eventInfo->Energy.getBin();
  nBinsEta = eventInfo->Eta.getNBins();
  nEvtPerBinEta = eventInfo->Eta.getNEvtPerBin();
  SLetas = eventInfo->Eta.getBin();
  nBinsPhi = eventInfo->Phi.getNBins();
  nEvtPerBinPhi = eventInfo->Phi.getNEvtPerBin();
  SLphis = eventInfo->Phi.getBin();

  // Convert from GeV to MeV
  for (unsigned int i = 0; i < SLenergies.size(); i++) {
    SLenergies[i] *= CLHEP::GeV;
  }

  edm::LogVerbatim("CastorShower") << " CastorShowerLibrary::loadEventInfo : "
                                   << "\n \n Total number of events     :  " << totEvents
                                   << "\n   Number of bins  (E)       :  " << nBinsE
                                   << "\n   Number of events/bin (E)  :  " << nEvtPerBinE
                                   << "\n   Number of bins  (Eta)       :  " << nBinsEta
                                   << "\n   Number of events/bin (Eta)  :  " << nEvtPerBinEta
                                   << "\n   Number of bins  (Phi)       :  " << nBinsPhi
                                   << "\n   Number of events/bin (Phi)  :  " << nEvtPerBinPhi << "\n";

  std::stringstream ss1;
  ss1 << "CastorShowerLibrary: energies in GeV:\n";
  for (unsigned int i = 0; i < nBinsE; ++i) {
    if (i > 0 && i / 10 * 10 == i) {
      ss1 << "\n";
    }
    ss1 << " " << SLenergies[i] / CLHEP::GeV;
  }
  ss1 << "\nCastorShowerLibrary: etas:\n";
  for (unsigned int i = 0; i < nBinsEta; ++i) {
    if (i > 0 && i / 10 * 10 == i) {
      ss1 << "\n";
    }
    ss1 << " " << SLetas[i];
  }
  ss1 << "\nCastorShowerLibrary: phis:\n";
  for (unsigned int i = 0; i < nBinsPhi; ++i) {
    if (i > 0 && i / 10 * 10 == i) {
      ss1 << "\n";
    }
    ss1 << " " << SLphis[i];
  }
  edm::LogVerbatim("CastorShower") << ss1.str();
}

//=============================================================================================

CastorShowerEvent CastorShowerLibrary::getShowerHits(const G4Step* aStep, bool& ok) {
  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  G4Track* track = aStep->GetTrack();

  const G4ThreeVector& hitPoint = preStepPoint->GetPosition();

  CastorShowerEvent hit;
  hit.Clear();

  ok = false;
  bool isEM = G4TrackToParticleID::isGammaElectronPositron(track);
  if (!isEM && !G4TrackToParticleID::isStableHadronIon(track)) {
    return hit;
  }
  ok = true;

  double pin = preStepPoint->GetTotalEnergy();
  double zint = hitPoint.z();
  double R = sqrt(hitPoint.x() * hitPoint.x() + hitPoint.y() * hitPoint.y());
  double theta = atan2(R, std::abs(zint));
  double phiin = atan2(hitPoint.y(), hitPoint.x());
  double etain = -std::log(std::tan((CLHEP::pi - theta) * 0.5));

  // Replace "interpolation/extrapolation" by new method "select" that just randomly
  // selects a record from the appropriate energy bin and fills its content to
  // "showerEvent" instance of class CastorShowerEvent

  if (isEM) {
    hit = select(0, pin, etain, phiin);
  } else {
    hit = select(1, pin, etain, phiin);
  }

  return hit;
}

//=============================================================================================

CastorShowerEvent CastorShowerLibrary::getRecord(int type, int record) {
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
  int nrc = record;
  CastorShowerEvent retValue;
  CastorShowerEvent* showerEvent = &retValue;
  if (type > 0) {
    hadBranch->SetAddress(&showerEvent);
    hadBranch->GetEntry(nrc);
  } else {
    emBranch->SetAddress(&showerEvent);
    emBranch->GetEntry(nrc);
  }

#ifdef DebugLog
  int nHit = showerEvent->getNhit();
  LogDebug("CastorShower") << "CastorShowerLibrary::getRecord: Record " << record << " of type " << type << " with "
                           << nHit << " CastorShowerHits";

#endif
  return retValue;
}

//=======================================================================================

CastorShowerEvent CastorShowerLibrary::select(int type, double pin, double etain, double phiin) {
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

  int irec;                    // to hold record number
  double r = G4UniformRand();  // to randomly select within an energy bin (r=[0,1])

  // Randomly select a record from the right energy bin in the library,
  // based on track momentum (pin)

  int ienergy = FindEnergyBin(pin);
  int ieta = FindEtaBin(etain);
#ifdef DebugLog
  if (verbose)
    edm::LogVerbatim("CastorShower") << " ienergy = " << ienergy;
  if (verbose)
    edm::LogVerbatim("CastorShower") << " ieta = " << ieta;
#endif

  int iphi;
  const double phiMin = 0.;
  const double phiMax = M_PI / 4.;  // 45 * (pi/180)  rad
  if (phiin < phiMin)
    phiin = phiin + M_PI;
  if (phiin >= phiMin && phiin < phiMax) {
    iphi = FindPhiBin(phiin);
  } else {
    double remainder = fmod(phiin, phiMax);
    phiin = phiMin + remainder;
    iphi = FindPhiBin(phiin);
  }
#ifdef DebugLog
  if (verbose)
    edm::LogVerbatim("CastorShower") << " iphi = " << iphi;
#endif
  if (ienergy == -1)
    ienergy = 0;  // if pin < first bin, choose an event in the first one
  if (ieta == -1)
    ieta = 0;  // idem for eta
  if (iphi != -1)
    irec = int(nEvtPerBinE * ienergy + nEvtPerBinEta * ieta + nEvtPerBinPhi * (iphi + r));
  else
    irec = int(nEvtPerBinE * (ienergy + r));

#ifdef DebugLog
  edm::LogVerbatim("CastorShower") << "CastorShowerLibrary:: Select record " << irec << " of type " << type;
#endif

  //  Retrieve record number "irec" from the library
  return getRecord(type, irec);
}

//=======================================================================================

int CastorShowerLibrary::FindEnergyBin(double energy) {
  //
  // returns the integer index of the energy bin, taken from SLenergies vector
  // returns -1 if ouside valid range
  //
  if (energy >= SLenergies.back())
    return SLenergies.size() - 1;

  unsigned int i = 0;
  for (; i < SLenergies.size() - 1; i++)
    if (energy >= SLenergies.at(i) && energy < SLenergies.at(i + 1))
      return (int)i;

  // now i points to the last but 1 bin
  if (energy >= SLenergies.at(i))
    return (int)i;
  // energy outside bin range
  return -1;
}

//=======================================================================================

int CastorShowerLibrary::FindEtaBin(double eta) {
  //
  // returns the integer index of the eta bin, taken from SLetas vector
  // returns -1 if ouside valid range
  //
  if (eta >= SLetas.back())
    return SLetas.size() - 1;
  unsigned int i = 0;
  for (; i < SLetas.size() - 1; i++)
    if (eta >= SLetas.at(i) && eta < SLetas.at(i + 1))
      return (int)i;
  // now i points to the last but 1 bin
  if (eta >= SLetas.at(i))
    return (int)i;
  // eta outside bin range
  return -1;
}

//=======================================================================================

int CastorShowerLibrary::FindPhiBin(double phi) {
  //
  // returns the integer index of the phi bin, taken from SLphis vector
  // returns -1 if ouside valid range
  //
  // needs protection in case phi is outside range -pi,pi
  //
  if (phi >= SLphis.back())
    return SLphis.size() - 1;
  unsigned int i = 0;
  for (; i < SLphis.size() - 1; i++)
    if (phi >= SLphis.at(i) && phi < SLphis.at(i + 1))
      return (int)i;
  // now i points to the last but 1 bin
  if (phi >= SLphis.at(i))
    return (int)i;
  // phi outside bin range
  return -1;
}

//=======================================================================================
