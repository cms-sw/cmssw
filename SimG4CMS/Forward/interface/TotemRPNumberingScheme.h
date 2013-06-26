#ifndef Forward_TotemRPNumberingScheme_h
#define Forward_TotemRPNumberingScheme_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemRPNumberingScheme
//
/**\class TotemRPNumberingScheme TotemRPNumberingScheme.h SimG4CMS/Forward/interface/TotemRPNumberingScheme.h
 
 Description: This class manages the UnitID that labels TotemRP sensitive
              volumes
 
 Usage:
    Used in TotemSD to get unique ID of sensitive detector element
 
*/
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemRPNumberingScheme.h,v 1.1 2006/05/17 16:18:57 sunanda Exp $
//
 
// system include files

// user include files

#include "SimG4CMS/Forward/interface/TotemRPOrganization.h"

class TotemRPNumberingScheme : public TotemRPOrganization {

public:

  // ---------- Constructor and destructor -----------------
  TotemRPNumberingScheme(int i);
  ~TotemRPNumberingScheme();
	 
  //  virtual uint32_t GetUnitID(const G4Step* aStep) const ;

};

#endif
