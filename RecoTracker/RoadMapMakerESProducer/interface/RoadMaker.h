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
// $Author: hlliu $
// $Date: 2008/01/08 17:42:16 $
// $Revision: 1.8 $
//

#include <vector>
#include <string>
#include <map>
#include <utility>

#include "RecoTracker/RoadMapRecord/interface/Roads.h"
#include "RecoTracker/RingRecord/interface/Rings.h"

class RoadMaker {
 
 public:
  
  enum GeometryStructure {
    FullDetector,
    FullDetectorII,
    MTCC,
    TIF,
    TIFTOB,
    TIFTIB,
    TIFTIBTOB,
    TIFTOBTEC,
    P5
  };

  enum SeedingType {
    FourRingSeeds,
    TwoRingSeeds
  };

  RoadMaker(const Rings *rings,
	    GeometryStructure structure = FullDetector,
	    SeedingType seedingType = FourRingSeeds);
  
  ~RoadMaker();

  void constructRoads();

  void collectInnerSeedRings();
  void collectInnerTIBSeedRings();
  void collectInnerTIDSeedRings();
  void collectInnerTECSeedRings();
  void collectInnerTOBSeedRings();

  void collectInnerSeedRings1();
  void collectInnerTIBSeedRings1();
  void collectInnerTIDSeedRings1();
  void collectInnerTECSeedRings1();
  void collectInnerTOBSeedRings1();

  void collectInnerSeedRings2();
  void collectInnerTIBSeedRings2();
  void collectInnerTIDSeedRings2();
  void collectInnerTECSeedRings2();
  void collectInnerTOBSeedRings2();

  void collectOuterSeedRings();
  void collectOuterTIBSeedRings();
  void collectOuterTOBSeedRings();
  void collectOuterTECSeedRings();

  void collectOuterSeedRings1();
  void collectOuterTIBSeedRings1();
  void collectOuterTOBSeedRings1();
  void collectOuterTECSeedRings1();

  inline Roads* getRoads() { return roads_; }

  bool RingsOnSameLayer(const Ring *ring1, const Ring* ring2);
  bool RingsOnSameLayer(std::pair<const Ring *,const Ring *> seed1, 
			std::pair<const Ring *,const Ring *> seed2);
  bool RingInBarrel(const Ring *ring);
  std::vector<std::pair<double,double> > LinesThroughRingAndBS(const Ring *ring );
  std::vector<std::pair<double,double> > LinesThroughRings(const Ring *ring1,
							   const Ring *ring2);
  bool CompatibleWithLines(std::vector<std::pair<double,double> > lines, 
			   const Ring* ring);
  Roads::RoadSet RingsCompatibleWithSeed(Roads::RoadSeed seed);
  Roads::RoadSeed CloneSeed(Roads::RoadSeed seed);
  bool AddRoad(Roads::RoadSeed seed,
	       Roads::RoadSet set);
  std::pair<Roads::RoadSeed, Roads::RoadSet> AddInnerSeedRing(std::pair<Roads::RoadSeed, Roads::RoadSet> input);
  std::pair<Roads::RoadSeed, Roads::RoadSet> AddOuterSeedRing(std::pair<Roads::RoadSeed, Roads::RoadSet> input);
  bool SameRoadSet(Roads::RoadSet set1, Roads::RoadSet set2 );
  Roads::RoadSet SortRingsIntoLayers(std::vector<const Ring*> input);

 private:

  const Rings *rings_;

  Roads       *roads_;
  GeometryStructure  structure_;
  SeedingType        seedingType_;

  std::vector<const Ring*> innerSeedRings_;
  std::vector<const Ring*> innerSeedRings1_;
  std::vector<const Ring*> innerSeedRings2_;
  std::vector<const Ring*> outerSeedRings_;
  std::vector<const Ring*> outerSeedRings1_;

  float              zBS_;

};

#endif
