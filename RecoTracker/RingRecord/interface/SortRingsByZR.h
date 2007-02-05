#ifndef RECOTRACKER_SORTRINGSBYZR_H
#define RECOTRACKER_SORTRINGSBYZR_H

//
// Package:         RecoTracker/RingRecord
// Class:           SortRingsByZR
// 
// Description:     sort rings by ZR
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Dec 20 17:31:01 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/08/29 14:48:15 $
// $Revision: 1.3 $
//

#include "RecoTracker/RingRecord/interface/Ring.h"

class SortRingsByZR {
 
 public:

  SortRingsByZR();
  ~SortRingsByZR();

  bool operator()( const Ring *a, const Ring *b) const {
    return RingsSortedInZR(a,b);
  }

  bool RingsSortedInZR(const Ring *a, const Ring *b) const;
  
 private:

};

#endif
