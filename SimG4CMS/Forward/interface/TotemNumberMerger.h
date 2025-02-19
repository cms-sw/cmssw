#ifndef Forward_TotemNumberMerger_h
#define Forward_TotemNumberMerger_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemNumberMerger
//
/**\class TotemNumberMerger TotemNumberMerger.h SimG4CMS/Forward/interface/TotemNumberMerger.h
 
 Description: Takes two integers and merges them together in only an integer!
              It's also possible the opposite operation.
 
 Usage:
    Used in TotemOrganizations to get unique ID of sensitive detector element
 
*/
//
// Original Author:  R. Capra
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemNumberMerger.h,v 1.1 2006/05/17 16:18:57 sunanda Exp $
//
 
// system include files

// user include files

#include "globals.hh"

class TotemNumberMerger {

public: 

  // ---------- Constructor and destructor -----------------
  TotemNumberMerger();
  ~TotemNumberMerger();

  // ---------- The two operations -------------------------
  unsigned long    Merge(unsigned long value1,  unsigned long value2) const;
  unsigned long    Merge(unsigned long value1,  unsigned long value2, 
			 unsigned long value3)  const;
  unsigned long    Merge(unsigned long value1,  unsigned long value2,
			 unsigned long value3,  unsigned long value4) const;
  void             Split(unsigned long source,  unsigned long &value1, 
			 unsigned long &value2) const;
  void             Split(unsigned long source,  unsigned long &value1, 
			 unsigned long &value2, unsigned long &value3) const;
  void             Split(unsigned long source,  unsigned long &value1, 
			 unsigned long &value2, unsigned long &value3,
			 unsigned long &value4) const;
};
#endif

