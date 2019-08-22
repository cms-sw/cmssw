// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemT1Organization
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  R. Capra
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files
#include "SimG4CMS/Forward/interface/TotemT1Organization.h"
#include "SimG4CMS/Forward/interface/TotemNumberMerger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"
#include "globals.hh"

//
// constructors and destructor
//
TotemT1Organization ::TotemT1Organization()
    : _needUpdateUnitID(false),
      _needUpdateData(false),
      _currentUnitID(-1),
      _currentPlane(-1),
      _currentCSC(-1),
      _currentLayer(-1),
      _currentObjectType(Undefined) {
  edm::LogInfo("ForwardSim") << "Creating TotemT1Organization";
}

TotemT1Organization ::~TotemT1Organization() {}

//
// member functions
//

uint32_t TotemT1Organization ::getUnitID(const G4Step* aStep) const {
  return const_cast<TotemT1Organization*>(this)->getUnitID(aStep);
}

uint32_t TotemT1Organization ::getUnitID(const G4Step* aStep) {
  int currLAOT;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4VPhysicalVolume* physVol;
  int ii = 0;
  for (ii = 0; ii < touch->GetHistoryDepth(); ii++) {
    physVol = touch->GetVolume(ii);

#ifdef SCRIVI
    LogDebug("ForwardSim") << "physVol=" << physVol->GetName() << ", level=" << ii
                           << ", physVol->GetCopyNo()=" << physVol->GetCopyNo();
#endif

    if (physVol->GetName() == "TotemT1" && physVol->GetCopyNo() == 1)
      _currentDetectorPosition = 1;
    if (physVol->GetName() == "TotemT1" && physVol->GetCopyNo() == 2)
      _currentDetectorPosition = 2;
  }

  touch = aStep->GetPreStepPoint()->GetTouchable();
  physVol = touch->GetVolume(0);

  currLAOT = physVol->GetCopyNo();
  _currentObjectType = static_cast<ObjectType>(currLAOT % MaxObjectTypes);
  _currentLayer = currLAOT / MaxObjectTypes;
  _currentPlane = -1;
  _currentCSC = -1;

  if (touch->GetVolume(1)) {
    _currentCSC = touch->GetVolume(1)->GetCopyNo();
    if (touch->GetVolume(2))
      _currentPlane = touch->GetVolume(2)->GetCopyNo();
  }
#ifdef SCRIVI
  LogDebug("ForwardSim") << "CURRENT CSC " << _currentCSC << "\n"
                         << "CURRENT PLANE " << _currentPlane;
#endif
  _needUpdateUnitID = true;
  return getCurrentUnitID();
}

int TotemT1Organization ::getCurrentUnitID(void) const {
  _checkUnitIDUpdate();
#ifdef SCRIVI
  LogDebug("ForwardSim") << "getCurrentUnitID()=" << _currentUnitID;
  << endl;
#endif
  return _currentUnitID;
}

void TotemT1Organization ::setCurrentUnitID(int currentUnitID) {
#ifdef SCRIVI
  LogDebug("ForwardSim") << "_currentUnitID=" << currentUnitID;
#endif
  _currentUnitID = currentUnitID;
  _needUpdateData = true;
}

int TotemT1Organization ::getCurrentDetectorPosition(void) const {
  _checkDataUpdate();
#ifdef SCRIVI
  LogDebug("ForwardSim") << "getCurrentDetectorPosition()=" << _currentDetectorPosition;
#endif
  return _currentDetectorPosition;
}

void TotemT1Organization ::setCurrentDetectorPosition(int currentDetectorPosition) {
#ifdef SCRIVI
  LogDebug("ForwardSim") << "_currentDetectorPosition=" << currentDetectorPosition;
#endif
  _currentDetectorPosition = currentDetectorPosition;
  _needUpdateUnitID = true;
}

int TotemT1Organization ::getCurrentPlane(void) const {
  _checkDataUpdate();

#ifdef SCRIVI
  LogDebug("ForwardSim") << "getCurrentPlane()=" << _currentPlane;
#endif
  return _currentPlane;
}

void TotemT1Organization ::setCurrentPlane(int currentPlane) {
#ifdef SCRIVI
  LogDebug("ForwardSim") << "_currentPlane=" << currentPlane;
#endif
  _currentPlane = currentPlane;
  _needUpdateUnitID = true;
}

int TotemT1Organization ::getCurrentCSC(void) const {
  _checkDataUpdate();
#ifdef SCRIVI
  LogDebug("ForwardSim") << "getCurrentCSC()=" << _currentCSC;
#endif
  return _currentCSC;
}

void TotemT1Organization ::setCurrentCSC(int currentCSC) {
#ifdef SCRIVI
  LogDebug("ForwardSim") << "_currentCSC=" << currentCSC;
#endif
  _currentCSC = currentCSC;
  _needUpdateUnitID = true;
}

int TotemT1Organization ::getCurrentLayer(void) const {
  _checkDataUpdate();
#ifdef SCRIVI
  LogDebug("ForwardSim") << "getCurrentLayer()=" << _currentLayer;
#endif
  return _currentLayer;
}

void TotemT1Organization ::setCurrentLayer(int currentLayer) {
#ifdef SCRIVI
  LogDebug("ForwardSim") << "_currentLayer=" << currentLayer;
#endif
  _currentLayer = currentLayer;
  _needUpdateUnitID = true;
}

TotemT1Organization::ObjectType TotemT1Organization ::getCurrentObjectType(void) const {
  _checkDataUpdate();
#ifdef SCRIVI
  LogDebug("ForwardSim") << "getCurrentObjectType()=" << _currentObjectType;
#endif
  return _currentObjectType;
}

void TotemT1Organization ::setCurrentObjectType(ObjectType currentObjectType) {
#ifdef SCRIVI
  LogDebug("ForwardSim") << "_currentObjectType=" << currentObjectType;
#endif
  _currentObjectType = currentObjectType;
  _needUpdateUnitID = true;
}

int TotemT1Organization ::fromObjectTypeToInt(ObjectType objectType) {
  int result(static_cast<int>(objectType));
  if (result < 0 || result >= MaxObjectTypes) {
    result = 0;
    edm::LogInfo("ForwardSim") << "Invalid ObjectType value (" << objectType << "). Now is \"Undefined\"";
  }
  return result;
}

int TotemT1Organization ::fromObjectTypeToInt(ObjectType objectType, int layer) {
  return fromObjectTypeToInt(objectType) + layer * MaxObjectTypes;
}

//
// private member functions
//

void TotemT1Organization ::_checkUnitIDUpdate(void) const {
  if (_needUpdateUnitID) {
#ifdef SCRIVI
    LogDebug("ForwardSim") << "UnitID update needed.";
#endif
    const_cast<TotemT1Organization*>(this)->_FromDataToUnitID();
  } else {
#ifdef SCRIVI
    LogDebug("ForwardSim") << "UnitID update not needed.";
#endif
  }
}

void TotemT1Organization ::_checkDataUpdate(void) const {
  if (_needUpdateData) {
#ifdef SCRIVI
    LogDebug("ForwardSim") << "Data update needed.";
#endif
    const_cast<TotemT1Organization*>(this)->_FromUnitIDToData();
  } else {
#ifdef SCRIVI
    LogDebug("ForwardSim") << "Data update not needed.";
#endif
  }
}

void TotemT1Organization ::_FromUnitIDToData(void) {
  int currDP, currCSC, currOT, currPLA;
  unsigned long currPL, currLA;

  // currDP:  0..4 (5)
  // currPL:  0..infty
  // currCSC: 0..6 (7)
  // currLA:  0..infty
  // currOT:  0..MaxObjectTypes-1 (MaxObjectTypes)

  currDP = (_currentUnitID / 100000) % 5;  // 3;
  currCSC = (_currentUnitID / 5) % 7;
  currOT = (_currentUnitID / (5 * 7)) % MaxObjectTypes;
  currPLA = _currentUnitID / (5 * 7 * MaxObjectTypes);

  TotemNumberMerger splitter;
  splitter.Split(currPLA, currPL, currLA);

#ifdef SCRIVI
  LogDebug("ForwardSim") << "currDP=" << currDP << ", currPL=" << currPL << ", currCSC=" << currCSC
                         << ", currLA=" << currLA << ", currOT=" << currOT << ", currPLA=" << currPLA
                         << ", _currentUnitID=" << _currentUnitID;
#endif
  _currentPlane = currPL - 1;
  _currentCSC = currCSC - 1;
  _currentLayer = currLA - 1;
  _currentObjectType = static_cast<ObjectType>(currOT);

  switch (currDP) {
    case 0:
      _currentDetectorPosition = 0;
      break;
    case 1:
      _currentDetectorPosition = 1;
      break;
    case 2:
      _currentDetectorPosition = 2;
      break;
    case 3:
      _currentDetectorPosition = 3;
      break;
    case 4:
      _currentDetectorPosition = 4;
      break;
  }
  _needUpdateData = false;
}

void TotemT1Organization ::_FromDataToUnitID(void) {
  int currDP, currPL, currCSC, currLA, currOT;
#ifdef SCRIVI
  LogDebug("ForwardSim") << " CURRENT DETECTOR POSITION (0-3) " << _currentDetectorPosition;
#endif
  switch (_currentDetectorPosition) {
    case 0:
      currDP = 0;
      break;
    case 1:
      currDP = 1;
      break;
    case 2:
      currDP = 2;
      break;
    case 3:
      currDP = 3;
      break;
    case 4:
      currDP = 4;
      break;
    default:
      _currentDetectorPosition = 0;
      currDP = 0;
      edm::LogInfo("ForwardSim") << "Invalid _currentDetectorPosition value (" << _currentDetectorPosition
                                 << "). Now is \"Undefined\"";
  }

  if (_currentPlane < -1) {
    edm::LogInfo("ForwardSim") << "Invalid _currentPlane value (" << _currentPlane << "). Now is -1";
    _currentPlane = -1;
  }
  currPL = _currentPlane + 1;

  if (_currentCSC < -1 || _currentCSC > 5) {
    edm::LogInfo("ForwardSim") << "Invalid _currentCSC value (" << _currentCSC << "). Now is -1";
    _currentCSC = -1;
  }
  currCSC = _currentCSC + 1;

  if (_currentLayer < -1) {
    edm::LogInfo("ForwardSim") << "Invalid _currentLayer value (" << _currentLayer << "). Now is -1";
    _currentLayer = -1;
  }
  currLA = _currentLayer + 1;

  currOT = fromObjectTypeToInt(_currentObjectType);

  // currDP:  0..2 (3)
  // currPL:  0..infty
  // currCSC: 0..6 (7)
  // currLA:  0..infty
  // currOT:  0..MaxObjectTypes-1 (MaxObjectTypes)

  TotemNumberMerger merger;
  int currPLA(merger.Merge(currPL, currLA));

  _currentUnitID = currDP * 100000 + 5 * (currCSC + 7 * (currOT + MaxObjectTypes * (currPLA)));
#ifdef SCRIVI
  LogDebug("ForwardSim") << "currDP=" << currDP << ", currPL=" << currPL << ", currCSC=" << currCSC
                         << ", currLA=" << currLA << ", currOT=" << currOT << ", currPLA=" << currPLA
                         << ", _currentUnitID=" << _currentUnitID;
#endif

  _needUpdateUnitID = false;
}
