#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "SimG4Core/SensitiveDetector/interface/FrameRotation.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include "SimG4CMS/Tracker/interface/TkAccumulatingSensitiveDetector.h"
#include "SimG4CMS/Tracker/interface/FakeFrameRotation.h"
#include "SimG4CMS/Tracker/interface/TrackerFrameRotation.h"
#include "SimG4CMS/Tracker/interface/TkSimHitPrinter.h"
#include "SimG4CMS/Tracker/interface/TrackerG4SimHitNumberingScheme.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4VProcess.hh"

#include "G4SystemOfUnits.hh"

#include <memory>

#include <iostream>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define FAKEFRAMEROTATION

static TrackerG4SimHitNumberingScheme& numberingScheme(const GeometricDet& det) {
  static thread_local TrackerG4SimHitNumberingScheme s_scheme(det);
  return s_scheme;
}

TkAccumulatingSensitiveDetector::TkAccumulatingSensitiveDetector(const std::string& name,
                                                                 const GeometricDet* pDD,
                                                                 const SensitiveDetectorCatalog& clg,
                                                                 edm::ParameterSet const& p,
                                                                 const SimTrackManager* manager)
    : SensitiveTkDetector(name, clg),
      pDD_(pDD),
      theManager(manager),
      rTracker(1200. * CLHEP::mm),
      zTracker(3000. * CLHEP::mm),
      mySimHit(nullptr),
      lastId(0),
      lastTrack(0),
      oldVolume(nullptr),
      px(0.0f),
      py(0.0f),
      pz(0.0f),
      eventno(0),
      pname("") {
  edm::ParameterSet m_TrackerSD = p.getParameter<edm::ParameterSet>("TrackerSD");
  allowZeroEnergyLoss = m_TrackerSD.getParameter<bool>("ZeroEnergyLoss");
  neverAccumulate = m_TrackerSD.getParameter<bool>("NeverAccumulate");
  printHits = m_TrackerSD.getParameter<bool>("PrintHits");
  theTofLimit = m_TrackerSD.getParameter<double>("ElectronicSigmaInNanoSeconds") * 3 * CLHEP::ns;  // 3 sigma
  energyCut =
      m_TrackerSD.getParameter<double>("EnergyThresholdForPersistencyInGeV") * CLHEP::GeV;  //default must be 0.5
  energyHistoryCut =
      m_TrackerSD.getParameter<double>("EnergyThresholdForHistoryInGeV") * CLHEP::GeV;  //default must be 0.05
  rTracker2 = rTracker * rTracker;

  // No Rotation given in input, automagically choose one based upon the name
  std::string rotType;
  theRotation = std::make_unique<TrackerFrameRotation>();
  rotType = "TrackerFrameRotation";

#ifdef FAKEFRAMEROTATION
  theRotation.reset(new FakeFrameRotation());
  rotType = "FakeFrameRotation";
#endif

  edm::LogVerbatim("TrackerSim") << " TkAccumulatingSensitiveDetector: "
                                 << " Criteria for Saving Tracker SimTracks: \n"
                                 << " History: " << energyHistoryCut << " MeV; Persistency: " << energyCut
                                 << " MeV;  TofLimit: " << theTofLimit << " ns"
                                 << "\n FrameRotation type " << rotType << " rTracker(cm)= " << rTracker / CLHEP::cm
                                 << " zTracker(cm)= " << zTracker / CLHEP::cm
                                 << " allowZeroEnergyLoss: " << allowZeroEnergyLoss
                                 << " neverAccumulate: " << neverAccumulate << " printHits: " << printHits;

  slaveLowTof = std::make_unique<TrackingSlaveSD>(name + "LowTof");
  slaveHighTof = std::make_unique<TrackingSlaveSD>(name + "HighTof");

  std::vector<std::string> temp;
  temp.push_back(slaveLowTof.get()->name());
  temp.push_back(slaveHighTof.get()->name());
  setNames(temp);

  theG4ProcTypeEnumerator = std::make_unique<G4ProcessTypeEnumerator>();
  theNumberingScheme = nullptr;
}

TkAccumulatingSensitiveDetector::~TkAccumulatingSensitiveDetector() {}

bool TkAccumulatingSensitiveDetector::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  LogDebug("TrackerSimDebug") << " Entering a new Step " << aStep->GetTotalEnergyDeposit() << " "
                              << aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName();

  if (aStep->GetTotalEnergyDeposit() > 0. || allowZeroEnergyLoss) {
    if (!mySimHit) {
      createHit(aStep);
    } else if (neverAccumulate || newHit(aStep)) {
      sendHit();
      createHit(aStep);
    } else {
      updateHit(aStep);
    }
    return true;
  }
  return false;
}

uint32_t TkAccumulatingSensitiveDetector::setDetUnitId(const G4Step* step) {
  return theNumberingScheme->g4ToNumberingScheme(step->GetPreStepPoint()->GetTouchable());
}

void TkAccumulatingSensitiveDetector::update(const BeginOfTrack* bot) {
  const G4Track* gTrack = (*bot)();

#ifdef DUMPPROCESSES
  if (gTrack->GetCreatorProcess()) {
    edm::LogVerbatim("TrackerSim") << " -> PROCESS CREATOR : " << gTrack->GetCreatorProcess()->GetProcessName();
  } else {
    edm::LogVerbatim("TrackerSim") << " -> No Creator process";
  }
#endif

  //
  //Position
  //
  const G4ThreeVector& pos = gTrack->GetPosition();
  LogDebug("TrackerSimDebug") << " update(..) of " << gTrack->GetDefinition()->GetParticleName()
                              << " trackID= " << gTrack->GetTrackID() << " E(MeV)= " << gTrack->GetKineticEnergy()
                              << " Ecut= " << energyCut << " R(mm)= " << pos.perp() << " Z(mm)= " << pos.z();

  //
  // Check if in Tracker Volume
  //
  if (pos.x() * pos.x() + pos.y() * pos.y() < rTracker2 && std::abs(pos.z()) < zTracker) {
    //
    // inside the Tracker
    //
    TrackInformation* info = nullptr;
    if (gTrack->GetKineticEnergy() > energyCut) {
      info = cmsTrackInformation(gTrack);
      info->setStoreTrack();
    }
    //
    // Save History?
    //
    if (gTrack->GetKineticEnergy() > energyHistoryCut) {
      if (nullptr == info) {
        info = cmsTrackInformation(gTrack);
      }
      info->putInHistory();
      LogDebug("TrackerSimDebug") << " Track inside the tracker selected for HISTORY"
                                  << " Track ID= " << gTrack->GetTrackID();
    }
  }
}

void TkAccumulatingSensitiveDetector::sendHit() {
  if (mySimHit == nullptr)
    return;
  if (printHits) {
    TkSimHitPrinter thePrinter("TkHitPositionOSCAR.dat");
    thePrinter.startNewSimHit(GetName(),
                              oldVolume->GetLogicalVolume()->GetName(),
                              mySimHit->detUnitId(),
                              mySimHit->trackId(),
                              lastTrack,
                              eventno);
    thePrinter.printLocal(mySimHit->entryPoint(), mySimHit->exitPoint());
    thePrinter.printGlobal(globalEntryPoint, globalExitPoint);
    thePrinter.printHitData(pname, mySimHit->pabs(), mySimHit->energyLoss(), mySimHit->timeOfFlight());
    thePrinter.printGlobalMomentum(px, py, pz);
    LogDebug("TrackerSimDebug") << " Storing PSimHit: " << mySimHit->detUnitId() << " " << mySimHit->trackId() << " "
                                << mySimHit->energyLoss() << " " << mySimHit->entryPoint() << " "
                                << mySimHit->exitPoint();
  }

  if (mySimHit->timeOfFlight() < theTofLimit) {
    slaveLowTof.get()->processHits(*mySimHit);  // implicit conversion (slicing) to PSimHit!!!
  } else {
    slaveHighTof.get()->processHits(*mySimHit);  // implicit conversion (slicing) to PSimHit!!!
  }
  //
  // clean up
  delete mySimHit;
  mySimHit = nullptr;
  lastTrack = 0;
  lastId = 0;
}

void TkAccumulatingSensitiveDetector::createHit(const G4Step* aStep) {
  // VI: previous hit should be already deleted
  //     in past here was a check if a hit is inside a sensitive detector,
  //     this is not needed, because call to senstive detector happens
  //     only inside the volume
  const G4Track* theTrack = aStep->GetTrack();
  Local3DPoint theExitPoint = theRotation.get()->transformPoint(LocalPostStepPosition(aStep));
  Local3DPoint theEntryPoint;
  //
  //  Check particle type - for gamma and neutral hadrons energy deposition
  //  should be local (VI)
  //
  if (0.0 == theTrack->GetDefinition()->GetPDGCharge()) {
    theEntryPoint = theExitPoint;
  } else {
    theEntryPoint = theRotation.get()->transformPoint(LocalPreStepPosition(aStep));
  }

  //
  //	This allows to send he skipEvent if it is outside!
  //
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  float thePabs = preStepPoint->GetMomentum().mag() / GeV;
  float theTof = preStepPoint->GetGlobalTime() / nanosecond;
  float theEnergyLoss = aStep->GetTotalEnergyDeposit() / GeV;
  int theParticleType = G4TrackToParticleID::particleID(theTrack);
  uint32_t theDetUnitId = setDetUnitId(aStep);
  int theTrackID = theTrack->GetTrackID();
  if (theDetUnitId == 0) {
    edm::LogWarning("TkAccumulatingSensitiveDetector::createHit") << " theDetUnitId is not valid for " << GetName();
    throw cms::Exception("TkAccumulatingSensitiveDetector::createHit")
        << "cannot get theDetUnitId for G4Track " << theTrackID;
  }

  // To whom assign the Hit?
  // First iteration: if the track is to be stored, use the current number;
  // otherwise, get to the mother
  unsigned int theTrackIDInsideTheSimHit = theTrackID;

  const TrackInformation* temp = cmsTrackInformation(theTrack);
  if (!temp->storeTrack()) {
    // Go to the mother!
    theTrackIDInsideTheSimHit = theTrack->GetParentID();
    LogDebug("TrackerSimDebug") << " TkAccumulatingSensitiveDetector::createHit(): setting the TrackID from "
                                << theTrackIDInsideTheSimHit << " to the mother one " << theTrackIDInsideTheSimHit
                                << " " << theEnergyLoss;
  } else {
    LogDebug("TrackerSimDebug") << " TkAccumulatingSensitiveDetector:createHit(): leaving the current TrackID "
                                << theTrackIDInsideTheSimHit;
  }

  const G4ThreeVector& gmd = preStepPoint->GetMomentumDirection();
  // convert it to local frame
  G4ThreeVector lmd =
      ((G4TouchableHistory*)(preStepPoint->GetTouchable()))->GetHistory()->GetTopTransform().TransformAxis(gmd);
  Local3DPoint lnmd = theRotation.get()->transformPoint(ConvertToLocal3DPoint(lmd));
  float theThetaAtEntry = lnmd.theta();
  float thePhiAtEntry = lnmd.phi();

  mySimHit = new UpdatablePSimHit(theEntryPoint,
                                  theExitPoint,
                                  thePabs,
                                  theTof,
                                  theEnergyLoss,
                                  theParticleType,
                                  theDetUnitId,
                                  theTrackIDInsideTheSimHit,
                                  theThetaAtEntry,
                                  thePhiAtEntry,
                                  theG4ProcTypeEnumerator.get()->processId(theTrack->GetCreatorProcess()));
  lastId = theDetUnitId;
  lastTrack = theTrackID;

  // only for debugging
  if (printHits) {
    // point on Geant4 unit (mm)
    globalEntryPoint = ConvertToLocal3DPoint(preStepPoint->GetPosition());
    globalExitPoint = ConvertToLocal3DPoint(aStep->GetPostStepPoint()->GetPosition());
    // in CMS unit (GeV)
    px = preStepPoint->GetMomentum().x() / CLHEP::GeV;
    py = preStepPoint->GetMomentum().y() / CLHEP::GeV;
    pz = preStepPoint->GetMomentum().z() / CLHEP::GeV;
    oldVolume = preStepPoint->GetPhysicalVolume();
    pname = theTrack->GetDefinition()->GetParticleName();
    LogDebug("TrackerSimDebug") << " Created PSimHit: " << pname << " " << mySimHit->detUnitId() << " "
                                << mySimHit->trackId() << " " << theTrackID
                                << " p= " << aStep->GetPreStepPoint()->GetMomentum().mag() << " "
                                << mySimHit->energyLoss() << " " << mySimHit->entryPoint() << " "
                                << mySimHit->exitPoint();
  }
}

void TkAccumulatingSensitiveDetector::updateHit(const G4Step* aStep) {
  // VI: in past here was a check if a hit is inside a sensitive detector,
  //     this is not needed, because call to senstive detector happens
  //     only inside the volume
  Local3DPoint theExitPoint = theRotation.get()->transformPoint(LocalPostStepPosition(aStep));
  float theEnergyLoss = aStep->GetTotalEnergyDeposit() / GeV;
  mySimHit->setExitPoint(theExitPoint);
  mySimHit->addEnergyLoss(theEnergyLoss);
  if (printHits) {
    globalExitPoint = ConvertToLocal3DPoint(aStep->GetPostStepPoint()->GetPosition());
    LogDebug("TrackerSimDebug") << " updateHit: for " << aStep->GetTrack()->GetDefinition()->GetParticleName()
                                << " trackID= " << aStep->GetTrack()->GetTrackID() << " deltaEloss= " << theEnergyLoss
                                << "\n Updated PSimHit: " << mySimHit->detUnitId() << " " << mySimHit->trackId() << " "
                                << mySimHit->energyLoss() << " " << mySimHit->entryPoint() << " "
                                << mySimHit->exitPoint();
  }
}

bool TkAccumulatingSensitiveDetector::newHit(const G4Step* aStep) {
  const G4Track* theTrack = aStep->GetTrack();

  // for neutral particles do not merge hits (V.I.)
  if (0.0 == theTrack->GetDefinition()->GetPDGCharge())
    return true;

  uint32_t theDetUnitId = setDetUnitId(aStep);
  int theTrackID = theTrack->GetTrackID();

  LogDebug("TrackerSimDebug") << "newHit: OLD(detID,trID) = (" << lastId << "," << lastTrack << "), NEW = ("
                              << theDetUnitId << "," << theTrackID << ") Step length(mm)= " << aStep->GetStepLength()
                              << " Edep= " << aStep->GetTotalEnergyDeposit()
                              << " p= " << aStep->GetPreStepPoint()->GetMomentum().mag();
  return ((theTrackID == lastTrack) && (lastId == theDetUnitId) && closeHit(aStep)) ? false : true;
}

bool TkAccumulatingSensitiveDetector::closeHit(const G4Step* aStep) {
  const float tolerance2 = 0.0025f;  // (0.5 mm)^2 are allowed between entry and exit
  Local3DPoint theEntryPoint = theRotation.get()->transformPoint(LocalPreStepPosition(aStep));
  LogDebug("TrackerSimDebug") << " closeHit: distance = " << (mySimHit->exitPoint() - theEntryPoint).mag();

  return ((mySimHit->exitPoint() - theEntryPoint).mag2() < tolerance2) ? true : false;
}

void TkAccumulatingSensitiveDetector::EndOfEvent(G4HCofThisEvent*) {
  LogDebug("TrackerSimDebug") << " Saving the last hit in a ROU " << GetName();
  if (mySimHit != nullptr)
    sendHit();
}

void TkAccumulatingSensitiveDetector::update(const BeginOfEvent* i) {
  clearHits();
  eventno = (*i)()->GetEventID();
  delete mySimHit;
  mySimHit = nullptr;
}

void TkAccumulatingSensitiveDetector::update(const BeginOfJob* i) { theNumberingScheme = &(numberingScheme(*pDD_)); }

void TkAccumulatingSensitiveDetector::clearHits() {
  slaveLowTof.get()->Initialize();
  slaveHighTof.get()->Initialize();
}

void TkAccumulatingSensitiveDetector::fillHits(edm::PSimHitContainer& cc, const std::string& hname) {
  if (slaveLowTof.get()->name() == hname) {
    cc = slaveLowTof.get()->hits();
  } else if (slaveHighTof.get()->name() == hname) {
    cc = slaveHighTof.get()->hits();
  }
}
