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
//
 
// system include files

// user include files

#include "SimG4CMS/Forward/interface/TotemRPOrganization.h"

class TotemRPNumberingScheme : public TotemRPOrganization {

public:

  // ---------- Constructor and destructor -----------------
  TotemRPNumberingScheme(int i);
  ~TotemRPNumberingScheme() override;
	 
};

#endif
