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
// $Date: 2007/02/05 19:10:03 $
// $Revision: 1.1 $
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
