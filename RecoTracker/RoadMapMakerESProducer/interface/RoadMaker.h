#ifndef RECOTRACKER_ROADMAKER_H
#define RECOTRACKER_ROADMAKER_H

//
// Package:         RecoTracker/RoadMapMakerESProducer
// Class:           RoadMaker
// 
// Description:     Creates a Roads object by combining all
//                  inner and outer SeedRings into RoadSeeds
//                  and determines all Rings of the RoadSet
//                  belonging to the RoadSeeds.     
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Jan 12 21:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/03/03 22:23:12 $
// $Revision: 1.3 $
//

#include <vector>
#include <string>
#include <map>
#include <utility>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/RoadMapRecord/interface/Roads.h"
#include "RecoTracker/RoadMapMakerESProducer/interface/Rings.h"

class RoadMaker {
 
 public:
  
  RoadMaker(const TrackerGeometry &tracker, unsigned int verbosity = 0);
  
  ~RoadMaker();

  void constructRoads();

  void collectInnerSeedRings(std::vector<Ring*>& set);
  void collectInnerTIBSeedRings(std::vector<Ring*>& set);
  void collectInnerTIDSeedRings(std::vector<Ring*>& set);
  void collectInnerTECSeedRings(std::vector<Ring*>& set);

  void collectOuterSeedRings(std::vector<Ring*>& set);
  void collectOuterTOBSeedRings(std::vector<Ring*>& set);
  void collectOuterTECSeedRings(std::vector<Ring*>& set);

  std::string printTrackerDetUnits(const TrackerGeometry &tracker);

  inline Roads* getRoads() { return roads_; }

  inline void dumpOldStyle(std::string ascii_filename = "geodump.dat") {
    rings_->dumpOldStyle(ascii_filename); }

 private:

  Rings *rings_;
  Roads *roads_;

  int verbosity_;
  
};

#endif
