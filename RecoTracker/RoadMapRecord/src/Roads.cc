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
// $Date: 2006/08/29 14:48:14 $
// $Revision: 1.4 $
//

#include <iostream>
#include <sstream>

#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

Roads::Roads() : numberOfLayers_(0) {

}

Roads::Roads(std::string ascii_filename, unsigned int verbosity) : numberOfLayers_(0) {

  readInFromAsciiFile(ascii_filename,verbosity);

}

Roads::~Roads() { 

}

void Roads::readInFromAsciiFile(std::string ascii_filename, unsigned int verbosity) {

  unsigned int counter = 0;

  std::ifstream input(ascii_filename.c_str());
  std::istringstream stream;
  std::string line;
  unsigned int index, type;
  float rmin,rmax,zmin,zmax;
  unsigned int ndetid = 0;
  double phi;
  unsigned int detid;
  unsigned int nringsinroadset;
  
  // number of subdetector components
  unsigned int comp;
  std::getline(input,line);
  while (std::isspace(line[0]) || (line[0] == 35) ) {
    std::getline(input,line);
  }
  stream.str(line);
  stream.clear();
  stream >> comp;

  std::getline(input,line);
  while (std::isspace(line[0]) || (line[0] == 35) ) {
    std::getline(input,line);
  }
  stream.str(line);
  stream.clear();
  unsigned int temp;
  for (unsigned int i = 0; i < comp; ++i ) {
    stream >> temp;
    numberOfLayers_.push_back(temp);
  }

  while ( std::getline(input,line) ) {
    if ( !std::isspace(line[0]) && !(line[0] == 35) ) {

      // inner seed ring
      stream.str(line);
      stream.clear();
      stream >> index >> rmin >> rmax >> zmin >> zmax >> type;
      Ring innerSeed(index,rmin,rmax,zmin,zmax,type);
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
	innerSeed.addId(phi,DetId(detid));
      }

      // outer seed ring
      std::getline(input,line);
      while (std::isspace(line[0]) || (line[0] == 35) ) {
	std::getline(input,line);
      }
      stream.str(line);
      stream.clear();
      stream >> index >> rmin >> rmax >> zmin >> zmax >> type;
      Ring outerSeed(index,rmin,rmax,zmin,zmax,type);
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
	outerSeed.addId(phi,DetId(detid));
      }

      // RoadSeed
      RoadSeed seed(innerSeed,outerSeed);

      // RoadSet
      RoadSet set;

      // number of rings in set
      std::getline(input,line);
      while (std::isspace(line[0]) || (line[0] == 35) ) {
	std::getline(input,line);
      }
      stream.str(line);
      stream.clear();
      stream >> nringsinroadset;
      for ( unsigned int i = 0; i < nringsinroadset; ++i ) {
	std::getline(input,line);
	while (std::isspace(line[0]) || (line[0] == 35) ) {
	  std::getline(input,line);
	}
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
	set.push_back(ring);
      }

      // add seed and set to map
      roadMap_.insert(make_pair(seed,set));
      ++counter;
    }
  }

  edm::LogInfo("RoadSearch") << "Read in: " << counter << " RoadSets from file: " << ascii_filename;

}

void Roads::dump(std::string ascii_filename) const {

  std::ofstream stream(ascii_filename.c_str());

  dumpHeader(stream);

    stream << "### Road subdetector information ###" << std::endl;
    unsigned int totalNumberOfLayersPerSubdetector = 0;
    for ( NumberOfLayersPerSubdetectorConstIterator component = numberOfLayers_.begin(); component != numberOfLayers_.end(); ++component) {
      totalNumberOfLayersPerSubdetector += *component;
    }
    stream << totalNumberOfLayersPerSubdetector << std::endl;
    for ( NumberOfLayersPerSubdetectorConstIterator component = numberOfLayers_.begin(); component != numberOfLayers_.end(); ++component) {
      stream << *component << " ";
    }
    stream << std::endl;

  for ( const_iterator roaditerator = roadMap_.begin(); roaditerator != roadMap_.end(); ++roaditerator ) {

    stream << "### RoadMap Entry ###" << std::endl;

    RoadSeed seed = (*roaditerator).first;
    RoadSet  set  = (*roaditerator).second;

    stream << "### RoadSeed First Ring ###" << std::endl;
    stream << seed.first.dump();
    stream << "### RoadSeed Second Ring ###" << std::endl;
    stream << seed.second.dump();

    stream << "### RoadSet ###" << std::endl;
    stream << set.size() << std::endl;

    for ( RoadSet::const_iterator setiterator = set.begin(); setiterator != set.end(); ++setiterator ) {
      stream << (*setiterator).dump();
    }
  }
}

void Roads::dumpHeader(std::ofstream &stream) const {

  stream << "#" << std::endl;
  stream << "# Roads for the RoadSearch tracking algorithm" << std::endl;
  stream << "# Ascii Dump" << std::endl;
  stream << "# " << std::endl;
  stream << "# Content:" << std::endl;
  stream << "# " << std::endl;
  stream << "# a dump of the RoadMap structure:" << std::endl;
  stream << "#" << std::endl;
  stream << "# Road subdetector information: number of subdetectors, number of layers per subdetector" << std::endl;
  stream << "# Ring: index, rmin, rmax, zmin, zmax, std::vector<DetId>: Ring of DetUnits in phi" << std::endl;
  stream << "# RoadSeed: std::pair<Ring,Ring>: inner and outer Ring Seed for the Road" << std::endl;
  stream << "# RoadSet : std::vector<DetId>: all Rings belonging to a road" << std::endl;
  stream << "# RoadMap: std::multimap<RoadSeed,RoadSet>: main container for the Roads" << std::endl;
  stream << "# " << std::endl;
  stream << "# Ascii-Format:" << std::endl;
  stream << "# " << std::endl;
  stream << "# Road subdetector information:" << std::endl;
  stream << "#" << std::endl;
  stream << "#       ### Road subdetector information ###" << std::endl;
  stream << "#       <number of subdetectors n>" << std::endl;
  stream << "#       <number of layers of subdetector 1> <number of layers of subdetector 2> ... <number of layers of subdetector n>" << std::endl;
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
  stream << "# RoadMap:" << std::endl;
  stream << "#" << std::endl;
  stream << "#       ### RoadMap Entry ###" << std::endl;
  stream << "#       ### RoadSeed First Ring ###" << std::endl;
  stream << "#       < std::pair<Ring,Ring>.first>" << std::endl;
  stream << "#       ### RoadSeed First Ring ###" << std::endl;
  stream << "#       < std::pair<Ring,Ring>.second>" << std::endl;
  stream << "#       ### RoadSet ###" << std::endl;
  stream << "#       <number of Rings in RoadSet>" << std::endl;
  stream << "#       <Ring>" << std::endl;
  stream << "#       <Ring>" << std::endl;
  stream << "#        ..." << std::endl;
  stream << "# " << std::endl;
  stream << "# " << std::endl;
  
}

const Roads::RoadSeed* Roads::getRoadSeed(DetId InnerSeedRing, DetId OuterSeedRing, double InnerSeedRingPhi,double OuterSeedRingPhi, double dphi_scalefactor) const {

  // loop over seed Ring pairs

  // determine ringtype for inner seed ring detid
  Ring::type innerSeedRingType = getRingType(InnerSeedRing);
  Ring::type outerSeedRingType = getRingType(OuterSeedRing);

  for ( const_iterator road = roadMap_.begin(); road != roadMap_.end(); ++road ) {
    if ( road->first.first.getType() == innerSeedRingType &&
	 road->first.second.getType() == outerSeedRingType ) {
      if ( road->first.first.containsDetId(InnerSeedRing,InnerSeedRingPhi,dphi_scalefactor) &&
	   road->first.second.containsDetId(OuterSeedRing,OuterSeedRingPhi,dphi_scalefactor) ) {
	return &(road->first);
      }
    }
  }
      
  edm::LogError("RoadSearch") << "RoadSeed could not be found for inner SeedRing type: " << innerSeedRingType << " DetId: " << InnerSeedRing.rawId() 
			      << " at " << InnerSeedRingPhi
			      << " and outer SeedRing type : " << outerSeedRingType << " DetID: " << OuterSeedRing.rawId() 
			      << " at " << OuterSeedRingPhi;
  return 0;
}

const Roads::type Roads::getRoadType(const RoadSeed *const seed) const {

  if ( (seed->second.getType() == Ring::PXBRing) ||
       (seed->second.getType() == Ring::TIBRing) ||
       (seed->second.getType() == Ring::TOBRing) ) {
    return Roads::RPhi;
  } else {
    return Roads::ZPhi;
  }

}


const Ring::type Roads::getRingType(DetId id) const {

  Ring::type type = Ring::Unspecified;

  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    type = Ring::TIBRing;
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    type = Ring::TOBRing;
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
    type = Ring::TIDRing;
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
    type = Ring::TECRing;
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel ) {
    type = Ring::PXBRing;
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelEndcap ) {
    type = Ring::PXFRing;
  }

  return type;

}
