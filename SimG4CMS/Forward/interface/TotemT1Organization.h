#ifndef Forward_TotemT1Organization_h
#define Forward_TotemT1Organization_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemT1Organization
//
/**\class TotemT1Organization TotemT1Organization.h SimG4CMS/Forward/interface/TotemT1Organization.h
 
 Description: This class manages the UnitID that labels TotemT1 sensitive
              volumes
 
 Usage:
    Used in TotemSD to get unique ID of sensitive detector element
 
*/
//
// Original Author:  R. Capra
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files

#include "SimG4CMS/Forward/interface/TotemVDetectorOrganization.h"

class TotemT1Organization : public TotemVDetectorOrganization {
public:
  // ---------- public enums -------------------------------

  enum ObjectType {
    Undefined = 0,
    Upper = 1,
    Lower = 2,
    InternalUpper = 3,
    InternalLower = 4,
    Frame1 = 5,
    Frame2Left = 6,
    Frame2Right = 7,
    Frame3Left = 8,
    Frame3Right = 9,
    Frame4Left = 10,
    Frame4Right = 11,
    Frame5 = 12,
    Triangle6Left = 13,
    Triangle6Right = 14,
    MaxObjectTypes = 15
  };

public:
  // ---------- Constructor and destructor -----------------
  TotemT1Organization();
  ~TotemT1Organization() override;

  // ---------- member functions ---------------------------
  uint32_t getUnitID(const G4Step* aStep) const override;

  int getCurrentUnitID(void) const;
  void setCurrentUnitID(int currentUnitID);

  // ---------- Detector position --------------------------

  int getCurrentDetectorPosition(void) const;
  void setCurrentDetectorPosition(int currentDetectorPosition);

  // ---------- Plane: between 0 and (nPlanes-1) (or -1 for Undefined)
  int getCurrentPlane(void) const;
  void setCurrentPlane(int currentPlane);

  // ---------- CSC: between 0 and 5 (or -1 for Undefined)
  int getCurrentCSC(void) const;
  void setCurrentCSC(int currentCSC);

  // ---------- Layer: between 0 and (nLayers-1) (or -1 for Undefined)
  int getCurrentLayer(void) const;
  void setCurrentLayer(int currentLayer);

  // ---------- Object Type --------------------------------

  ObjectType getCurrentObjectType(void) const;
  void setCurrentObjectType(ObjectType currentObjectType);

  int fromObjectTypeToInt(ObjectType objectType);
  int fromObjectTypeToInt(ObjectType objectType, int layer);

  // ---------- Private methods ----------------------------

private:
  uint32_t getUnitID(const G4Step* aStep);

  void _checkUnitIDUpdate(void) const;
  void _checkDataUpdate(void) const;

  void _FromUnitIDToData(void);
  void _FromDataToUnitID(void);

  // ---------- Private data members -----------------------
  bool _needUpdateUnitID;
  bool _needUpdateData;
  int _currentUnitID;
  int _currentDetectorPosition;
  int _currentPlane;
  int _currentCSC;
  int _currentLayer;
  ObjectType _currentObjectType;
};
#endif
