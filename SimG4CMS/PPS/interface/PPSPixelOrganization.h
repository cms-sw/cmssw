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

#include "SimG4CMS/PPS/interface/PPSVDetectorOrganization.h"
#include "globals.hh"

class PPSPixelOrganization : public PPSVDetectorOrganization {
public:
  // ---------- Constructor and destructor -----------------
  PPSPixelOrganization();
  ~PPSPixelOrganization() override = default;

  // ---------- member functions ---------------------------
  uint32_t unitID(const G4Step* aStep) override;

private:
  // ---------- Private data members -----------------------
  int currentUnitID_;
  int currentArm_;
  int currentStation_;
  int currentRP_;
  int currentPlane_;
};
#endif
