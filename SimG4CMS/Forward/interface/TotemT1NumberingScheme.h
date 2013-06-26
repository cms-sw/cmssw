#ifndef Forward_TotemT1NumberingScheme_h
#define Forward_TotemT1NumberingScheme_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemT1NumberingScheme
//
/**\class TotemT1NumberingScheme TotemT1NumberingScheme.h SimG4CMS/Forward/interface/TotemT1NumberingScheme.h
 
 Description: This class manages the UnitID that labels TotemT1 sensitive
              volumes
 
 Usage:
    Used in TotemSD to get unique ID of sensitive detector element
 
*/
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemT1NumberingScheme.h,v 1.1 2006/05/17 16:18:57 sunanda Exp $
//
 
// system include files

// user include files

#include "SimG4CMS/Forward/interface/TotemT1Organization.h"

class TotemT1NumberingScheme : public TotemT1Organization {

public:

  // ---------- Constructor and destructor -----------------
  TotemT1NumberingScheme(int i);
  ~TotemT1NumberingScheme();
	 
  //  virtual uint32_t GetUnitID(const G4Step* aStep) const ;

};

#endif
