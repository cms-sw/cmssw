#ifndef RECOTRACKER_ROADS_H
#define RECOTRACKER_ROADS_H

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
// $Date: 2006/01/15 01:01:25 $
// $Revision: 1.2 $
//

#include <vector>
#include <map>
#include <utility>
#include <string>
#include <fstream>

#include "RecoTracker/RoadMapRecord/interface/Ring.h"

class Roads {
 
 public:
  
  typedef std::pair<Ring,Ring> RoadSeed;
  typedef std::vector<Ring> RoadSet;
  typedef std::multimap<RoadSeed,RoadSet> RoadMap;
  
  typedef RoadSet::iterator RoadSetIterator;
  typedef RoadSet::const_iterator RoadSetConstIterator;
  typedef std::pair<RoadSetIterator,RoadSetIterator> RoadSetIteratorRange;
  typedef std::pair<RoadSetConstIterator,RoadSetConstIterator> RoadSetConstIteratorRange;
  typedef RoadMap::iterator iterator;
  typedef RoadMap::const_iterator const_iterator;
  typedef std::pair<iterator,iterator> RoadMapRange;
  typedef std::pair<const_iterator,const_iterator> RoadMapConstRange;

  typedef std::vector<unsigned int> NumberOfLayersPerSubdetector;
  typedef std::vector<unsigned int>::iterator NumberOfLayersPerSubdetectorIterator;
  typedef std::vector<unsigned int>::const_iterator NumberOfLayersPerSubdetectorConstIterator;

  enum type {
    RPhi,
    ZPhi
  };

  Roads();
  Roads(std::string ascii_file, unsigned int verbosity = 0);

  ~Roads();

  inline void insert(RoadSeed *seed, RoadSet *set) { roadMap_.insert(make_pair(*seed,*set)); }

  inline iterator begin() { return roadMap_.begin(); }
  inline iterator end()   { return roadMap_.end();   }

  inline const_iterator begin() const { return roadMap_.begin(); }
  inline const_iterator end()   const { return roadMap_.end();   }

  void dump(std::string ascii_filename = "roads.dat") const;
  
  void dumpHeader(std::ofstream &stream) const;

  void readInFromAsciiFile(std::string ascii_file, unsigned int verbosity);

  const RoadSeed* getRoadSeed(DetId InnerSeedRing, DetId OuterSeedRing, 
			      double InnerSeedRingPhi = 999999., double OuterSeedRingPhi = 999999.) const;

  inline RoadMapConstRange getRoadSet(const RoadSeed *const seed) const { return roadMap_.equal_range(*seed); }

  const type getRoadType(const RoadSeed *const seed) const;

  inline NumberOfLayersPerSubdetector getNumberOfLayersPerSubdetector() const { return numberOfLayers_; }

  inline void setNumberOfLayersPerSubdetector(NumberOfLayersPerSubdetector input) { numberOfLayers_ = input; }

  const Ring::type getRingType(DetId id) const;

 private:

  RoadMap roadMap_;

  NumberOfLayersPerSubdetector numberOfLayers_;

};

#endif
