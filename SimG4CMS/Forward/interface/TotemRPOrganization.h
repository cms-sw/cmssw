#ifndef Forward_TotemRPOrganization_h
#define Forward_TotemRPOrganization_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemRPOrganization
//
/**\class TotemRPOrganization TotemRPOrganization.h SimG4CMS/Forward/interface/TotemRPOrganization.h
 
 Description: This class manages the UnitID that labels TotemRP sensitive
              volumes
 
 Usage:
    Used in TotemSD to get unique ID of sensitive detector element
 
*/
//
// Original Author:  
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemRPOrganization.h,v 1.1 2006/05/17 16:18:57 sunanda Exp $
//
 
// system include files

// user include files

#include "SimG4CMS/Forward/interface/TotemVDetectorOrganization.h"
#include "globals.hh"

class TotemRPOrganization : public TotemVDetectorOrganization {

public: 

  // ---------- Constructor and destructor -----------------
  TotemRPOrganization();
  virtual          ~TotemRPOrganization();

  // ---------- member functions ---------------------------
  uint32_t         GetUnitID(const G4Step* aStep);
  uint32_t         GetUnitID(const G4Step* aStep) const;

private:

  // ---------- member data --------------------------------
  bool             _needUpdateUnitID;
  bool             _needUpdateData;

  int              _currentUnitID;
  int              _currentDetectorPosition ;
  int              _currentPlane;
  int              _currentCSC;
  int              _currentLayer;

};
#endif
