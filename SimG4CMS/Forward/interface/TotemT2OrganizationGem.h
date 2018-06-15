#ifndef Forward_TotemT2OrganizationGem_h
#define Forward_TotemT2OrganizationGem_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemT2OrganizationGem
//
/**\class TotemT2OrganizationGem TotemT2OrganizationGem.h SimG4CMS/Forward/interface/TotemT2OrganizationGem.h
 
 Description: This class manages the UnitID that labels TotemT2Gem sensitive
              volumes
 
 Usage:
    Used in TotemSD to get unique ID of sensitive detector element
 
*/
//
// Original Author:  
//         Created:  Tue May 16 10:14:34 CEST 2006
//
 
// system include files

// user include files
#include "SimG4CMS/Forward/interface/TotemVDetectorOrganization.h"

class TotemT2OrganizationGem : public TotemVDetectorOrganization {

public:

  // ---------- Constructor and destructor -----------------
  TotemT2OrganizationGem();
  ~TotemT2OrganizationGem() override;

  // ---------- member functions ---------------------------
  uint32_t         GetUnitID(const G4Step* aStep) const override;

};

#endif
