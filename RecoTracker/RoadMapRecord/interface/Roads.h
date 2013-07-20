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
// $Date: 2007/04/17 21:56:53 $
// $Revision: 1.6 $
//

#include <vector>
#include <map>
#include <utility>
#include <string>
#include <fstream>

#include "RecoTracker/RingRecord/interface/Ring.h"
#include "RecoTracker/RingRecord/interface/Rings.h"

#include "RecoTracker/RoadMapRecord/interface/RoadMapSorting.h"

class Roads {
 
 public:
  
  typedef std::pair<std::vector<const Ring*>, std::vector<const Ring*> > RoadSeed;
  typedef std::vector<std::vector<const Ring*> > RoadSet;
  typedef std::multimap<RoadSeed,RoadSet,RoadMapSorting> RoadMap;
  
  typedef RoadMap::iterator iterator;
  typedef RoadMap::const_iterator const_iterator;

  enum type {
    RPhi,
    ZPhi
  };

  Roads();
  Roads(std::string ascii_file, const Rings *rings);

  ~Roads();

  inline void insert(RoadSeed *seed, RoadSet *set) { roadMap_.insert(make_pair(*seed,*set)); }
  inline void insert(RoadSeed seed, RoadSet set) { roadMap_.insert(make_pair(seed,set)); }

  inline iterator begin() { return roadMap_.begin(); }
  inline iterator end()   { return roadMap_.end();   }

  inline const_iterator begin() const { return roadMap_.begin(); }
  inline const_iterator end()   const { return roadMap_.end();   }

  inline RoadMap::size_type size() const { return roadMap_.size(); }

  void dump(std::string ascii_filename = "roads.dat") const;
  
  void dumpHeader(std::ofstream &stream) const;

  void readInFromAsciiFile(std::string ascii_file);

  const RoadSeed* getRoadSeed(DetId InnerSeedRing, 
			      DetId OuterSeedRing, 
			      double InnerSeedRingPhi = 999999., 
			      double OuterSeedRingPhi = 999999.,
			      double dphi_scalefactor=1.5) const;

  const RoadSeed* getRoadSeed(std::vector<DetId> seedRingDetIds,
			      std::vector<double> seedRingHitsPhi,
			      double dphi_scalefactor=1.5) const;

  inline const_iterator getRoadSet(const RoadSeed *seed) const { return roadMap_.find(*seed); }

  const type getRoadType(const RoadSeed *const seed) const;

  const Ring::type getRingType(DetId id) const;

  inline void erase(iterator entry) { roadMap_.erase(entry); }

  inline const Ring* getRing(DetId id, double phi = 999999., double z = 999999.) const {
    return rings_->getRing(id,phi,z);
  }

 private:

  const Rings *rings_;
  RoadMap roadMap_;

};

#endif
