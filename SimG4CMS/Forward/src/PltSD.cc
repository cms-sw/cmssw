#include "SimG4CMS/Forward/interface/PltSD.h"
#include "SimG4CMS/Forward/interface/ForwardName.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4ThreeVector.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>

#include <iostream>

//#define EDM_ML_DEBUG

PltSD::PltSD(const std::string& name,
             const SensitiveDetectorCatalog& clg,
             edm::ParameterSet const& p,
             const SimTrackManager* manager)
    : TimingSD(name, clg, manager) {
  edm::ParameterSet m_TrackerSD = p.getParameter<edm::ParameterSet>("PltSD");
  energyCut =
      m_TrackerSD.getParameter<double>("EnergyThresholdForPersistencyInGeV") * CLHEP::GeV;  //default must be 0.5
  energyHistoryCut =
      m_TrackerSD.getParameter<double>("EnergyThresholdForHistoryInGeV") * CLHEP::GeV;  //default must be 0.05

  setCuts(energyCut, energyHistoryCut);
}

PltSD::~PltSD() {}

uint32_t PltSD::setDetUnitId(const G4Step* aStep) {
  unsigned int detId = 0;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("PltSD") << " DetID = " << detId;
#endif
  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = 0;
  if (touch)
    level = ((touch->GetHistoryDepth()) + 1);

  //Get name and copy numbers
  if (level > 1) {
    //some debugging with the names
    std::string sensorName = ForwardName::getName(touch->GetVolume(2)->GetName());
    std::string telName = ForwardName::getName(touch->GetVolume(3)->GetName());
    std::string volumeName = ForwardName::getName(touch->GetVolume(4)->GetName());
    if (sensorName != "PLTSensorPlane")
      edm::LogVerbatim("PltSD") << " PltSD::setDetUnitId -w- Sensor name not PLTSensorPlane ";
    if (telName != "Telescope")
      edm::LogVerbatim("PltSD") << " PltSD::setDetUnitId -w- Telescope name not Telescope ";
    if (volumeName != "PLT")
      edm::LogVerbatim("PltSD") << " PltSD::setDetUnitId -w- Volume name not PLT ";

    //Get the information about which telescope, plane, row/column was hit
    int columnNum = touch->GetReplicaNumber(0);
    int rowNum = touch->GetReplicaNumber(1);
    int sensorNum = touch->GetReplicaNumber(2);
    int telNum = touch->GetReplicaNumber(3);
    //temp stores the PLTBCM volume the hit occured in (i.e. was the hit on the + or -z side?)
    int temp = touch->GetReplicaNumber(5);
    //convert to the PLT hit id standard
    int pltNum;
    if (temp == 2)
      pltNum = 0;
    else
      pltNum = 1;

    //correct the telescope numbers on the -z side to have the same naming convention in phi as the +z side
    if (pltNum == 0) {
      if (telNum == 0) {
        telNum = 7;
      } else if (telNum == 1) {
        telNum = 6;
      } else if (telNum == 2) {
        telNum = 5;
      } else if (telNum == 3) {
        telNum = 4;
      } else if (telNum == 4) {
        telNum = 3;
      } else if (telNum == 5) {
        telNum = 2;
      } else if (telNum == 6) {
        telNum = 1;
      } else if (telNum == 7) {
        telNum = 0;
      }
    }
    //the PLT is divided into sets of telescopes on the + and -x sides
    int halfCarriageNum = -1;

    //If the telescope is on the -x side of the carriage, halfCarriageNum=0.  If on the +x side, it is = 1.
    if (telNum == 0 || telNum == 1 || telNum == 2 || telNum == 3)
      halfCarriageNum = 0;
    else
      halfCarriageNum = 1;
    //correct the telescope numbers of the +x half-carriage to range from 0 to 3
    if (halfCarriageNum == 1) {
      if (telNum == 4) {
        telNum = 0;
      } else if (telNum == 5) {
        telNum = 1;
      } else if (telNum == 6) {
        telNum = 2;
      } else if (telNum == 7) {
        telNum = 3;
      }
    }
    //Define unique detId for each pixel.  See https://twiki.cern.ch/twiki/bin/viewauth/CMS/PLTSimulationGuide for more information
    detId =
        10000000 * pltNum + 1000000 * halfCarriageNum + 100000 * telNum + 10000 * sensorNum + 100 * rowNum + columnNum;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("PltSD") << "Hit Recorded at "
                              << "plt:" << pltNum << " hc:" << halfCarriageNum << " tel:" << telNum
                              << " plane:" << sensorNum;
#endif
  }
  return detId;
}

bool PltSD::checkHit(const G4Step*, BscG4Hit* hit) {
  // 50 micron are allowed between the exit
  // point of the current hit and the entry point of the new hit
  static const float tolerance2 = (float)(0.0025 * CLHEP::mm * CLHEP::mm);
  return ((hit->getExitLocalP() - getLocalEntryPoint()).mag2() < tolerance2);
}
