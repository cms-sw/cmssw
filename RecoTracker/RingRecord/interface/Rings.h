#ifndef RECOTRACKER_RINGS_H
#define RECOTRACKER_RINGS_H

//
// Package:         RecoTracker/RingRecord
// Class:           Rings
// 
// Description:     The Rings object holds all Rings of
//                  the tracker mapped in z of their centers
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Tue Oct  3 22:14:25 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/02/05 19:10:03 $
// $Revision: 1.1 $
//

#include <vector>
#include <map>
#include <utility>
#include <string>
#include <fstream>

#include "RecoTracker/RingRecord/interface/Ring.h"

class Rings {
 
 public:
  
  typedef std::multimap<double,Ring> RingMap;
  typedef RingMap::iterator iterator;
  typedef RingMap::const_iterator const_iterator;

  Rings();
  Rings(std::string ascii_file);

  ~Rings();

  inline void insert(double z, Ring &ring) { ringMap_.insert(std::make_pair(z,ring)); }

  inline iterator begin() { return ringMap_.begin(); }
  inline iterator end()   { return ringMap_.end();   }
  inline const_iterator begin() const { return ringMap_.begin(); }
  inline const_iterator end()   const { return ringMap_.end();   }
  inline iterator lower_bound(double z) { return ringMap_.lower_bound(z); }
  inline iterator upper_bound(double z) { return ringMap_.upper_bound(z); }
  inline const_iterator lower_bound(double z) const { return ringMap_.lower_bound(z); }
  inline const_iterator upper_bound(double z) const { return ringMap_.upper_bound(z); }

  void dump(std::string ascii_filename = "rings.dat") const;
  void dumpHeader(std::ofstream &stream) const;

  void readInFromAsciiFile(std::string ascii_file);

  const Ring* getRing(DetId id, double phi = 999999., double z = 999999.) const;
  const Ring* getRing(unsigned int ringIndex, double z = 999999.) const;
  const Ring* getTIBRing(unsigned int layer,
			 unsigned int fw_bw,
			 unsigned int ext_int,
			 unsigned int detector) const;
  const Ring* getTOBRing(unsigned int layer,
			 unsigned int rod_fw_bw,
			 unsigned int detector) const;
  const Ring* getTIDRing(unsigned int fw_bw,
			 unsigned int wheel,
			 unsigned int ring) const;
  const Ring* getTECRing(unsigned int fw_bw,
			 unsigned int wheel,
			 unsigned int ring) const;
  const Ring* getPXBRing(unsigned int layer,
			 unsigned int detector) const;
  const Ring* getPXFRing(unsigned int fw_bw,
			 unsigned int disk,
			 unsigned int panel,
			 unsigned int module) const;

 private:

  RingMap ringMap_;

};

#endif
