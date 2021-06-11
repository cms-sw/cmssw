#include "SimG4CMS/Forward/interface/Bcm1fSD.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"
#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4VProcess.hh"

#include <string>
#include <vector>
#include <iostream>

#include "CLHEP/Units/GlobalSystemOfUnits.h"

Bcm1fSD::Bcm1fSD(const std::string& name,
                 const edm::EventSetup& es,
                 const SensitiveDetectorCatalog& clg,
                 edm::ParameterSet const& p,
                 const SimTrackManager* manager)
    : TimingSD(name, clg, manager) {
  edm::ParameterSet m_TrackerSD = p.getParameter<edm::ParameterSet>("Bcm1fSD");
  energyCut = m_TrackerSD.getParameter<double>("EnergyThresholdForPersistencyInGeV") * GeV;  //default must be 0.5 (?)
  energyHistoryCut =
      m_TrackerSD.getParameter<double>("EnergyThresholdForHistoryInGeV") * GeV;  //default must be 0.05 (?)

  setCuts(energyCut, energyHistoryCut);
}

Bcm1fSD::~Bcm1fSD() {}

uint32_t Bcm1fSD::setDetUnitId(const G4Step* aStep) {
  uint32_t detId = 0;

  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = (touch) ? ((touch->GetHistoryDepth()) + 1) : 0;

  //Get name and copy numbers
  if (level > 1) {
    G4String sensorName = touch->GetVolume(0)->GetName();
    G4String diamondName = touch->GetVolume(1)->GetName();
    G4String detectorName = touch->GetVolume(2)->GetName();
    G4String volumeName = touch->GetVolume(3)->GetName();

    if (sensorName != "BCM1FSensor") {
      edm::LogWarning("ForwardSim") << "Bcm1fSD::setDetUnitId -w- Sensor name not BCM1FSensor ";
    }
    if (detectorName != "BCM1F") {
      edm::LogWarning("ForwardSim") << " Bcm1fSD::setDetUnitId -w- Detector name not BCM1F ";
    }
    int sensorNo = touch->GetReplicaNumber(0);
    int diamondNo = touch->GetReplicaNumber(1);
    int volumeNo = touch->GetReplicaNumber(3);

    // Detector ID definition
    // detId = XYYZ
    // X  = volume,  1: +Z, 2: -Z
    // YY = diamond, 01-12, 12: phi = 90 deg, numbering clockwise when looking from the IP
    // Z  = sensor,  1 or 2, clockwise when looking from the IP

    detId = 1000 * volumeNo + 10 * diamondNo + sensorNo;
  }
  return detId;
}

bool Bcm1fSD::checkHit(const G4Step*, BscG4Hit* hit) {
  // 50 micron are allowed between the exit
  // point of the current hit and the entry point of the new hit
  static const float tolerance2 = (float)(0.0025 * CLHEP::mm * CLHEP::mm);
  return ((hit->getExitLocalP() - getLocalEntryPoint()).mag2() < tolerance2);
}
