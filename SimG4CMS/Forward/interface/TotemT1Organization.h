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
// $Id: TotemT1Organization.h,v 1.1 2006/05/17 16:18:57 sunanda Exp $
//
 
// system include files

// user include files

#include "SimG4CMS/Forward/interface/TotemVDetectorOrganization.h"
#include "globals.hh"

class TotemT1Organization : public TotemVDetectorOrganization {

public: 
  
  // ---------- public enums -------------------------------

  enum ObjectType  {
   Undefined=0,
   Upper=1,
   Lower=2,
   InternalUpper=3,
   InternalLower=4,
   Frame1=5,
   Frame2Left=6,
   Frame2Right=7,
   Frame3Left=8,
   Frame3Right=9,
   Frame4Left=10,
   Frame4Right=11,
   Frame5=12,
   Triangle6Left=13,
   Triangle6Right=14,
   MaxObjectTypes=15
  };

public:

  // ---------- Constructor and destructor -----------------
  TotemT1Organization();
  virtual          ~TotemT1Organization();

  // ---------- member functions ---------------------------
  uint32_t         GetUnitID(const G4Step* aStep);
  uint32_t         GetUnitID(const G4Step* aStep) const;
  
  int              GetCurrentUnitID(void) const;
  void             SetCurrentUnitID(int currentUnitID);

  // ---------- Detector position --------------------------

  int              GetCurrentDetectorPosition(void) const;
  void             SetCurrentDetectorPosition(int currentDetectorPosition);

  // ---------- Plane: between 0 and (nPlanes-1) (or -1 for Undefined)
  int              GetCurrentPlane(void) const;
  void             SetCurrentPlane(int currentPlane);

  // ---------- CSC: between 0 and 5 (or -1 for Undefined)
  int              GetCurrentCSC(void) const;
  void             SetCurrentCSC(int currentCSC);

  // ---------- Layer: between 0 and (nLayers-1) (or -1 for Undefined)
  int              GetCurrentLayer(void) const;
  void             SetCurrentLayer(int currentLayer);

  // ---------- Object Type --------------------------------

  ObjectType       GetCurrentObjectType(void) const;
  inline void      SetCurrentObjectType(ObjectType currentObjectType);

  int              FromObjectTypeToInt(ObjectType objectType);
  int              FromObjectTypeToInt(ObjectType objectType, int layer);

private:
  // ---------- Private methods ----------------------------
  void             _checkUnitIDUpdate(void) const;
  void             _checkDataUpdate(void) const;

  void             _FromUnitIDToData(void);
  void             _FromDataToUnitID(void);


private:
  // ---------- Private data members -----------------------
  bool             _needUpdateUnitID;
  bool             _needUpdateData;
  int              _currentUnitID;
  int              _currentDetectorPosition ;
  int              _currentPlane;
  int              _currentCSC;
  int              _currentLayer;
  ObjectType       _currentObjectType;

};
#endif
