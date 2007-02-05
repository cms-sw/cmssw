#ifndef RECOTRACKER_SORTLAYERSBYZR_H
#define RECOTRACKER_SORTLAYERSBYZR_H

//
// Package:         RecoTracker/LayerRecord
// Class:           SortLayersByZR
// 
// Description:     sort layers by ZR
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Dec 20 17:31:01 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/08/29 14:48:15 $
// $Revision: 1.3 $
//

#include "RecoTracker/RingRecord/interface/Ring.h"

#include <vector>

class SortLayersByZR {
 
 public:

  SortLayersByZR();
  ~SortLayersByZR();

  bool operator()(const std::vector<const Ring*> &layerA, const std::vector<const Ring*> &layerB) const {
    return LayersSortedInZR(layerA,layerB);
  }

  bool LayersSortedInZR(const std::vector<const Ring*> &layerA, const std::vector<const Ring*> &layerB) const;
  
 private:

};

#endif
