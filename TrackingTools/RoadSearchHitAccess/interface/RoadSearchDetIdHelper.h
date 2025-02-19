#ifndef RoadSearch_RoadSearchDetIdHelper_h
#define RoadSearch_RoadSearchDetIdHelper_h

//
// Package:         TrackingTools/RoadSearchDetIdHelper
// Class:           RoadSearchDetIdHelper
// 
// Description:     helper functions concerning DetIds
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sun Jan 28 19:06:20 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/02/16 00:48:20 $
// $Revision: 1.2 $
//

#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

class RoadSearchDetIdHelper {

 public:

  RoadSearchDetIdHelper();

  ~RoadSearchDetIdHelper();

  static std::string Print(const DetId id);

  static bool IsMatched(const DetId id);
  
  static bool IsGluedRPhi(const DetId id);

  static bool IsSingleRPhi(const DetId id);
  
  static bool IsStereo(const DetId id);

  static DetId ReturnRPhiId(const DetId id);

  static bool detIdsOnSameLayer(DetId id1, DetId id2);

};

#endif
