#ifndef Forward_TotemT2NumberingSchemeGem_h
#define Forward_TotemT2NumberingSchemeGem_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemT2NumberingSchemeGem
//
/**\class TotemT2NumberingSchemeGem TotemT2NumberingSchemeGem.h SimG4CMS/Forward/interface/TotemT2NumberingSchemeGem.h
 
 Description: This class manages the UnitID that labels TotemT2Gem sensitive
              volumes
 
 Usage:
    Used in TotemSD to get unique ID of sensitive detector element
 
*/
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemT2NumberingSchemeGem.h,v 1.1 2006/05/17 16:18:57 sunanda Exp $
//
 
// system include files

// user include files

#include "SimG4CMS/Forward/interface/TotemT2OrganizationGem.h"

class TotemT2NumberingSchemeGem : public TotemT2OrganizationGem {

public:

  // ---------- Constructor and destructor -----------------
  TotemT2NumberingSchemeGem(int i);
  ~TotemT2NumberingSchemeGem();
	 
  //  virtual uint32_t GetUnitID(const G4Step* aStep) const ;

};

#endif
