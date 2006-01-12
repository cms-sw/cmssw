#ifndef RoadMaker_h
#define RoadMaker_h

/** \class RoadMaker
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Oct. 14, 2005  

 *
 ************************************************************/

#include <vector>
#include <string>
#include <map>
#include <utility>

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "RecoTracker/RoadMapRecord/interface/Roads.h"
#include "RecoTracker/RoadMapMakerESProducer/interface/Rings.h"

class RoadMaker {
 
 public:
  
  RoadMaker(const TrackingGeometry &tracker, unsigned int verbosity = 0);
  
  ~RoadMaker();

  void constructRoads();

  void collectInnerSeedRings(std::vector<Ring*>& set);
  void collectInnerTIBSeedRings(std::vector<Ring*>& set);
  void collectInnerTIDSeedRings(std::vector<Ring*>& set);
  void collectInnerTECSeedRings(std::vector<Ring*>& set);

  void collectOuterSeedRings(std::vector<Ring*>& set);
  void collectOuterTOBSeedRings(std::vector<Ring*>& set);
  void collectOuterTECSeedRings(std::vector<Ring*>& set);

  void printTrackerDetUnits(const TrackingGeometry &tracker);

  inline Roads* getRoads() { return roads_; }

  inline void dumpOldStyle(std::string ascii_filename = "geodump.dat") {
    rings_->dumpOldStyle(ascii_filename); }

 private:

  Rings *rings_;
  Roads *roads_;

  int verbosity_;
  
};

#endif
