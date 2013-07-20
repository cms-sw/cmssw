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
// $Date: 2007/03/30 02:49:37 $
// $Revision: 1.2 $
//

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "RecoTracker/RingRecord/interface/Rings.h"

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


Rings::Rings() {
  //
  // default constructor
  //
}

Rings::Rings(std::string ascii_filename) {
  //
  // constructor reading in ascii file
  //

  readInFromAsciiFile(ascii_filename);

}

Rings::~Rings() { 
  //
  // default destructor
  //
}

void Rings::readInFromAsciiFile(std::string ascii_filename) {
  // 
  // read in all rings stored in ascii file
  //

  std::ifstream input(ascii_filename.c_str());
  std::istringstream stream;
  std::string line;
  unsigned int index, type;
  float rmin,rmax,zmin,zmax;
  unsigned int ndetid = 0;
  double phi;
  unsigned int detid;
  
  while ( std::getline(input,line) ) {
    if ( !std::isspace(line[0]) && !(line[0] == 35) ) {

      // ring
      stream.str(line);
      stream.clear();
      stream >> index >> rmin >> rmax >> zmin >> zmax >> type;
      Ring ring(index,rmin,rmax,zmin,zmax,type);
      std::getline(input,line);
      while (std::isspace(line[0]) || (line[0] == 35) ) {
	std::getline(input,line);
      }
      stream.str(line);
      stream.clear();
      stream >> ndetid;
      for (unsigned int i = 0; i < ndetid; ++i ) {
	std::getline(input,line);
	while (std::isspace(line[0]) || (line[0] == 35) ) {
	  std::getline(input,line);
	}
	stream.str(line);
	stream.clear();
	stream >> phi >> detid;
	ring.addId(phi,DetId(detid));
      }
      double center_z = zmin + ((zmax-zmin)/2);
      ringMap_.insert(std::make_pair(center_z,ring));
    }
  }

  edm::LogInfo("RoadSearch") << "Read in: " << ringMap_.size() << " Rings from file: " << ascii_filename;

}

void Rings::dump(std::string ascii_filename) const {
  //
  // dump all rings in ascii file
  //

  std::ofstream stream(ascii_filename.c_str());

  dumpHeader(stream);

  for ( const_iterator ring = ringMap_.begin(); ring != ringMap_.end(); ++ring ) {

    stream << ring->second.dump();

  }

}

void Rings::dumpHeader(std::ofstream &stream) const {
  //
  // header with general information about content of rings ascii file
  //

  stream << "#" << std::endl;
  stream << "# Rings for the RoadSearch tracking algorithm" << std::endl;
  stream << "# Ascii Dump" << std::endl;
  stream << "# " << std::endl;
  stream << "# Content:" << std::endl;
  stream << "# " << std::endl;
  stream << "# a dump of all Rings:" << std::endl;
  stream << "#" << std::endl;
  stream << "# Ring   : index, rmin, rmax, zmin, zmax, std::multimap<phi,DetId>: Ring of DetUnits mapped in phi" << std::endl;
  stream << "# " << std::endl;
  stream << "# Ascii-Format:" << std::endl;
  stream << "# " << std::endl;
  stream << "# Ring:" << std::endl;
  stream << "#" << std::endl;
  stream << "#       ### Ring: <index> ###" << std::endl;
  stream << "#       <index> <rmin> <rmax> <zmin> <zmax>" << std::endl;
  stream << "#       <number of DetId's in std::vector<DetId> >" << std::endl;
  stream << "#       <phi of DetUnit described by DetId> <DetId::rawId()>" << std::endl;
  stream << "#       <phi of DetUnit described by DetId> <DetId::rawId()>" << std::endl;
  stream << "#            ..." << std::endl;
  stream << "#" << std::endl;
  stream << "#" << std::endl;
  
}

const Ring* Rings::getRing(DetId id, double phi,double z) const {
  //
  // loop over rings to discover ring which contains DetId id
  // loop is restricted to window in z

  // calculate window around given z (if z == 999999. set window to maximum)
  // window is += 1.5 times the longest sensor in z (TOB: ~20 cm)
  double z_min = -999999.;
  double z_max = 999999.;
  double delta_z = 1.5 * 20.;
  if ( z != 999999. ) {
    z_min = z - delta_z;
    z_max = z + delta_z;
  }

  // loop over rings
  for ( const_iterator ring = ringMap_.lower_bound(z_min); ring != ringMap_.upper_bound(z_max); ++ring ) {
    if ( ring->second.containsDetId(id,phi) ) {
      return &(ring->second);
    }
  }
      
  return 0;
}

const Ring* Rings::getRing(unsigned int ringIndex, double z) const {
  //
  // loop over rings to discover ring which has RingIndex ringIndex
  // loop is restricted to window in z

  // calculate window around given z (if z == 999999. set window to maximum)
  // window is += 1.5 times the longest sensor in z (TOB: ~20 cm)
  double z_min = -999999.;
  double z_max = 999999.;
  double delta_z = 1.5 * 20.;
  if ( z != 999999. ) {
    z_min = z - delta_z;
    z_max = z + delta_z;
  }

  for ( const_iterator ring = ringMap_.lower_bound(z_min); ring != ringMap_.upper_bound(z_max); ++ring ) {
    if ( ring->second.getindex() == ringIndex ) {
      return &(ring->second);
    }
  }
      
  return 0;
}

const Ring* Rings::getTIBRing(unsigned int layer,
			      unsigned int fw_bw,
			      unsigned int ext_int,
			      unsigned int detector) const {
  
  // construct DetID from info using else the first of all entities and return Ring

  const Ring* ring = 0;

  // first try stereo = 0, then stereo = 2, then fail
  TIBDetId id(layer,fw_bw,ext_int,1,detector,0);
  ring = getRing(DetId(id.rawId()));
  if ( ring == 0 ) {
    TIBDetId id(layer,fw_bw,ext_int,1,detector,2);
    ring = getRing(DetId(id.rawId()));
  }

  if ( ring == 0 ) {
    edm::LogError("RoadSearch") << "TIB Ring for layer: " << layer
				<< " fw_bw: " << fw_bw
				<< " ext_int: " << ext_int
				<< " detector: " << detector 
				<< " with rawId: " << id.rawId()
				<< " could not be found.";
  }

  return ring;
}

const Ring* Rings::getTIDRing(unsigned int fw_bw,
			      unsigned int wheel,
			      unsigned int ring) const {
  
  // construct DetID from info using else the first of all entities and return Ring
  
  const Ring* int_ring = 0;
  
  // first try stereo = 0, then stereo = 2, then fail
  TIDDetId id(fw_bw,wheel,ring,1,1,0);
  int_ring = getRing(DetId(id.rawId()));
  if ( int_ring == 0 ) {
    TIDDetId id(fw_bw,wheel,ring,1,1,2);
    int_ring = getRing(DetId(id.rawId()));
  }
  
  if ( int_ring == 0 ) {
    edm::LogError("RoadSearch") << "TID Ring for fw_bw: " << fw_bw
				<< " wheel: " << wheel
				<< " ring: " << ring
				<< " with rawId: " << id.rawId()
				<< " could not be found.";
  }
  
  return int_ring;
}

const Ring* Rings::getTECRing(unsigned int fw_bw,
			      unsigned int wheel,
			      unsigned int ring) const {
  
  // try to construct first detid from fw_bw, wheel, ring
  // set petal and module to 1 (zero in c-array terms)
  // check for combination if petal_fw_bw is valid, otherwise set to 0 is valid
  // if not, increase them to get a valid id

  int petal_fw_bw = 1;
  int petal       = 1;
  int module      = 1;

  const Ring* int_ring = 0;
  
  // first try stereo = 0, then stereo = 2, then fail
  TECDetId id(fw_bw,wheel,petal_fw_bw,petal,ring,module,0);
  int_ring = getRing(DetId(id.rawId()));
  if ( int_ring == 0 ) {
    TECDetId id(fw_bw,wheel,petal_fw_bw,petal,ring,module,2);
    int_ring = getRing(DetId(id.rawId()));
  }
  
  if ( int_ring == 0 ) {
    edm::LogError("RoadSearch") << "TEC Ring for fw_bw: " << fw_bw
				<< " wheel: " << wheel
				<< " ring: " << ring
				<< " with rawId: " << id.rawId()
				<< " could not be found.";
  }

  return int_ring;
}

const Ring* Rings::getTOBRing(unsigned int layer,
			      unsigned int rod_fw_bw,
			      unsigned int detector) const {
    
  // construct DetID from info using else the first of all entities and return Ring
  const Ring* ring = 0;
  
  // first try stereo = 0, then stereo = 2, then fail
  TOBDetId id(layer,rod_fw_bw,1,detector,0);
  ring = getRing(DetId(id.rawId()));
  if ( ring == 0 ) {
    TOBDetId id(layer,rod_fw_bw,1,detector,2);
    ring = getRing(DetId(id.rawId()));
  }
  
  if ( ring == 0 ) {
    edm::LogError("RoadSearch") << "TOB Ring for layer: " << layer
				<< " rod_fw_bw: " << rod_fw_bw
				<< " detector: " << detector 
				<< " with rawId: " << id.rawId()
				<< " could not be found.";
  }
  
  return ring;
}

const Ring* Rings::getPXBRing(unsigned int layer,
			unsigned int detector) const {

  // construct DetID from info using else the first of all entities and return Ring
  unsigned int ladder = 0;

  PXBDetId id(layer,ladder,detector);

  return getRing(DetId(id.rawId()));
}


const Ring* Rings::getPXFRing(unsigned int fw_bw,
			unsigned int disk,
			unsigned int panel,
			unsigned int module) const {
  
  // construct DetID from info using else the first of all entities and return Ring
  unsigned int detector = 0;
  
  PXFDetId id(fw_bw+1,disk+1,detector+1,panel+1,module+1);
  
  return getRing(DetId(id.rawId()));
}

