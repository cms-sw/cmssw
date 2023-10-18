#include "SimG4Core/Application/interface/ExceptionHandler.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4EventManager.hh"
#include "G4TrackingManager.hh"
#include "G4Track.hh"
#include "globals.hh"
#include <sstream>

ExceptionHandler::ExceptionHandler(double th, bool tr) : m_eth(th), m_trace(tr) {}

ExceptionHandler::~ExceptionHandler() {}

bool ExceptionHandler::Notify(const char* exceptionOrigin,
                              const char* exceptionCode,
                              G4ExceptionSeverity severity,
                              const char* description) {
  static const G4String es_banner = "\n-------- EEEE ------- G4Exception-START -------- EEEE -------\n";
  static const G4String ee_banner = "\n-------- EEEE -------- G4Exception-END --------- EEEE -------\n";
  static const G4String ws_banner = "\n-------- WWWW ------- G4Exception-START -------- WWWW -------\n";
  static const G4String we_banner = "\n-------- WWWW -------- G4Exception-END --------- WWWW -------\n";

  G4Track* track = G4EventManager::GetEventManager()->GetTrackingManager()->GetTrack();
  double ekin = m_eth;

  std::stringstream message;
  message << "*** G4Exception : " << exceptionCode << "\n"
          << "      issued by : " << exceptionOrigin << "\n"
          << description;

  // part of exception happens outside tracking loop
  if (nullptr != track) {
    ekin = track->GetKineticEnergy();
    message << "\n CMS info: "
            << "TrackID=" << track->GetTrackID() << " ParentID=" << track->GetParentID() << "  "
            << track->GetParticleDefinition()->GetParticleName() << "; Ekin(MeV)=" << ekin / CLHEP::MeV
            << "; time(ns)=" << track->GetGlobalTime() / CLHEP::ns << "; status=" << track->GetTrackStatus()
            << "\n   position(mm): " << track->GetPosition() << "; direction: " << track->GetMomentumDirection();
    const G4VPhysicalVolume* vol = track->GetVolume();
    if (nullptr != vol) {
      message << "\n   PhysicalVolume: " << vol->GetName() << ";";
      const G4LogicalVolume* lv = vol->GetLogicalVolume();
      if (nullptr != lv) {
        message << " material: " << lv->GetMaterial()->GetName();
      }
    }
    message << "\n   stepNumber=" << track->GetCurrentStepNumber()
            << "; stepLength(mm)=" << track->GetStepLength() / CLHEP::mm << "; weight=" << track->GetWeight();
    const G4VProcess* proc = track->GetCreatorProcess();
    if (nullptr != proc) {
      message << "; creatorProcess: " << proc->GetProcessName() << "; modelID=" << track->GetCreatorModelID();
    }
  }
  message << " \n";

  G4ExceptionSeverity localSeverity = severity;
  std::stringstream mescode;
  mescode << exceptionCode << "\n";
  G4String code;
  mescode >> code;

  // track is killed
  if (ekin < m_eth && (code == "GeomNav0003" || code == "GeomField0003")) {
    localSeverity = JustWarning;
    track->SetTrackStatus(fStopAndKill);
    ++m_number;
  }

  bool res = false;
  if (localSeverity == JustWarning) { 
      if (m_number < 20)
        edm::LogWarning("SimG4CoreApplication")
            << ws_banner << message.str() << "*** This is just a warning message. ***" << we_banner;
  } else {
    edm::LogWarning("SimG4CoreApplication") << es_banner << message.str() << ee_banner;
    throw cms::Exception("Geant4 fatal exception");
    res = m_trace;
  }
  return res;
}
