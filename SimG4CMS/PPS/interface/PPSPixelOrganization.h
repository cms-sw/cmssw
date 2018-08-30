#ifndef _PPS_TrackerOrganization_h
#define _PPS_TrackerOrganization_h 1
// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelOrganization
//
/**\class PPSPixelOrganization PPSPixelOrganization.h SimG4CMS/PPS/interface/PPSPixelOrganization.h
 
 Description: This class manages the UnitID that labels PPS sensitive
              volumes
 
 Usage:
    Used in PPSPixelSD to get unique ID of sensitive detector element
 
*/
//
// Original Author:  R. Capra
//         Created:  Tue May 16 10:14:34 CEST 2006
//
 
// system include files

// user include files

#include "SimG4CMS/PPS/interface/PPSVDetectorOrganization.h"
#include "globals.hh"

class PPSPixelOrganization : public PPSVDetectorOrganization {

public: 
  
  // ---------- public enums -------------------------------

//  enum ObjectType  {
/*
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
*/
//  };

public:

  // ---------- Constructor and destructor -----------------
  PPSPixelOrganization();
           ~PPSPixelOrganization() override;

  // ---------- member functions ---------------------------
  uint32_t         GetUnitID(const G4Step* aStep);
  uint32_t         GetUnitID(const G4Step* aStep) const override;
  


private:
  // ---------- Private data members -----------------------
  int              _currentUnitID;
//  int              _currentDetectorPosition ;
  int              _currentArm;
  int              _currentStation;
  int              _currentRP;
  int              _currentPlane;

};
#endif


