//
// Package:         RecoTracker/RoadMapRecord
// Class:           Roads
// 
// Description:     The Roads object holds the RoadSeeds
//                  and the RoadSets of all Roads through 
//                  the detector. A RoadSeed consists
//                  of the inner and outer SeedRing,
//                  a RoadSet consists of all Rings in
//                  in the Road.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Jan 12 21:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/01/14 22:00:00 $
// $Revision: 1.1 $
//

#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(Roads);
