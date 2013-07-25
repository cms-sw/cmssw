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
// $Date: 2007/02/05 19:22:46 $
// $Revision: 1.6 $
//

#include <iostream>
#include <sstream>

#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "TrackingTools/RoadSearchHitAccess/interface/RoadSearchDetIdHelper.h"

Roads::Roads() {

}

Roads::Roads(std::string ascii_filename, const Rings *rings) : rings_(rings) {

  readInFromAsciiFile(ascii_filename);

}

Roads::~Roads() { 

}

void Roads::readInFromAsciiFile(std::string ascii_filename) {

  // input file
  std::ifstream input(ascii_filename.c_str());

  // variable declaration
  unsigned int counter         = 0;
  std::istringstream stream;
  std::string line;
  unsigned int nroads          = 0;
  unsigned int nrings          = 0;
  unsigned int nlayers         = 0;
  unsigned int index           = 0;

  // read in number of roads
  std::getline(input,line);
  while (std::isspace(line[0]) || (line[0] == 35) ) {
    std::getline(input,line);
  }
  stream.str(line);
  stream.clear();
  stream >> nroads;

  for (unsigned int road = 0; 
       road < nroads;
       ++road ) {
    // read in number of inner seed rings
    std::getline(input,line);
    while (std::isspace(line[0]) || (line[0] == 35) ) {
      std::getline(input,line);
    }
    std::vector<const Ring*> innerSeedRings;
    stream.str(line);
    stream.clear();
    stream >> nrings;
    for ( unsigned int i = 0;
	  i < nrings;
	  ++i ) {

      // read in ring indices for inner seed rings
      std::getline(input,line);
      while (std::isspace(line[0]) || (line[0] == 35) ) {
	std::getline(input,line);
      }
      stream.str(line);
      stream.clear();
      stream >> index;
      innerSeedRings.push_back(rings_->getRing(index));
    }

    // read in number of outer seed rings
    std::getline(input,line);
    while (std::isspace(line[0]) || (line[0] == 35) ) {
      std::getline(input,line);
    }
    std::vector<const Ring*> outerSeedRings;
    stream.str(line);
    stream.clear();
    stream >> nrings;
    for ( unsigned int i = 0;
	  i < nrings;
	  ++i ) {

      // read in ring indices for outer seed rings
      std::getline(input,line);
      while (std::isspace(line[0]) || (line[0] == 35) ) {
	std::getline(input,line);
      }
      stream.str(line);
      stream.clear();
      stream >> index;
      outerSeedRings.push_back(rings_->getRing(index));
    }

    // RoadSeed
    RoadSeed seed(innerSeedRings,outerSeedRings);

    // RoadSet
    RoadSet set;

    // number of layers in road set
    std::getline(input,line);
    while (std::isspace(line[0]) || (line[0] == 35) ) {
      std::getline(input,line);
    }
    stream.str(line);
    stream.clear();
    stream >> nlayers;

    for ( unsigned int i = 0;
	  i < nlayers;
	  ++i ) {

      std::vector<const Ring*> layer;

      // number of rings in layer
      std::getline(input,line);
      while (std::isspace(line[0]) || (line[0] == 35) ) {
	std::getline(input,line);
      }
      stream.str(line);
      stream.clear();
      stream >> nrings;
      for ( unsigned int j = 0; j < nrings; ++j ) {
	std::getline(input,line);
	while (std::isspace(line[0]) || (line[0] == 35) ) {
	  std::getline(input,line);
	}
	stream.str(line);
	stream.clear();
	stream >> index;
	layer.push_back(rings_->getRing(index));
      }
      set.push_back(layer);
    }
      

    // add seed and set to map
    roadMap_.insert(make_pair(seed,set));
    ++counter;
  }
  
  edm::LogInfo("RoadSearch") << "Read in: " << counter << " RoadSets from file: " << ascii_filename;

}

void Roads::dump(std::string ascii_filename) const {

  std::ofstream stream(ascii_filename.c_str());
  
  dumpHeader(stream);

  stream << "### Road information ###" << std::endl;
  stream << roadMap_.size() << std::endl;

  unsigned int counter = 0;

  for ( const_iterator roaditerator = roadMap_.begin(); 
	roaditerator != roadMap_.end(); 
	++roaditerator ) {

    ++counter;

    stream << "### RoadMap Entry " << counter << " ###" << std::endl;

    RoadSeed seed = (*roaditerator).first;
    RoadSet  set  = (*roaditerator).second;

    stream << "### RoadSeed First Ring ###" << std::endl;
    stream << seed.first.size() << std::endl;
    for (std::vector<const Ring*>::const_iterator ring = seed.first.begin();
	 ring != seed.first.end();
	 ++ring ) {
      stream << (*ring)->getindex() << std::endl;
    }
    stream << "### RoadSeed Second Ring ###" << std::endl;
    stream << seed.second.size() << std::endl;
    for (std::vector<const Ring*>::const_iterator ring = seed.second.begin();
	 ring != seed.second.end();
	 ++ring ) {
      stream << (*ring)->getindex() << std::endl;
    }

    stream << "### RoadSet ###" << std::endl;
    stream << set.size() << std::endl;
    for ( RoadSet::const_iterator layer = set.begin(); layer != set.end(); ++layer ) {
      stream << "### Layer ###" << std::endl;
      stream << layer->size() << std::endl;
      for ( std::vector<const Ring*>::const_iterator ring = layer->begin();
	    ring != layer->end();
	    ++ring ) {
	stream << (*ring)->getindex() << std::endl;
      }
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
  stream << "# Road Information: <number of roads>" << std::endl;
  stream << "# Ring: index, rmin, rmax, zmin, zmax, std::vector<DetId>: Ring of DetUnits in phi taken from ring service" << std::endl;
  stream << "# RoadSeed: std::pair<std::vector<const Ring*>,std::vector<const Ring*> >: inner and outer Ring Seed for the Road" << std::endl;
  stream << "# RoadSet : std::vector<std::vectro<const Ring*> >: all Rings belonging to a road structured in layers" << std::endl;
  stream << "# RoadMap: std::multimap<RoadSeed,RoadSet>: main container for the Roads" << std::endl;
  stream << "# " << std::endl;
  stream << "# Ascii-Format:" << std::endl;
  stream << "# " << std::endl;
  stream << "# Road Information:" << std::endl;
  stream << "#       <number of roads>" << std::endl;
  stream << "#" << std::endl;
  stream << "# RoadMap for each road:" << std::endl;
  stream << "#" << std::endl;
  stream << "#       ### RoadMap Entry ###" << std::endl;
  stream << "#       ### RoadSeed First Ring ###" << std::endl;
  stream << "#       <number of inner seed rings>" << std::endl;
  stream << "#       <index>" << std::endl;
  stream << "#       <index>" << std::endl;
  stream << "#       ..." << std::endl;
  stream << "#       ### RoadSeed Second Ring ###" << std::endl;
  stream << "#       <number of outer seed rings>" << std::endl;
  stream << "#       <index>" << std::endl;
  stream << "#       <index>" << std::endl;
  stream << "#       ..." << std::endl;
  stream << "#       ### RoadSet ###" << std::endl;
  stream << "#       <number of Layers in RoadSet>" << std::endl;
  stream << "#       ### Layer ###" << std::endl;
  stream << "#       <number of rings in layer>" << std::endl;
  stream << "#       <index>" << std::endl;
  stream << "#       <index>" << std::endl;
  stream << "#        ..." << std::endl;
  stream << "#       ### Layer ###" << std::endl;
  stream << "#        ..." << std::endl;
  stream << "#" << std::endl;
  stream << "#" << std::endl;
  
}

const Roads::RoadSeed* Roads::getRoadSeed(DetId InnerSeedRing, 
					  DetId OuterSeedRing, 
					  double InnerSeedRingPhi,
					  double OuterSeedRingPhi,
					  double dphi_scalefactor) const {

  // loop over seed Ring pairs

  // determine ringtype for inner seed ring detid
  Ring::type innerSeedRingType = getRingType(InnerSeedRing);
  Ring::type outerSeedRingType = getRingType(OuterSeedRing);

  for ( const_iterator road = roadMap_.begin(); road != roadMap_.end(); ++road ) {
    for ( std::vector<const Ring*>::const_iterator innerRing = road->first.first.begin();
	  innerRing != road->first.first.end();
	  ++innerRing ) {
      if ( (*innerRing)->getType() == innerSeedRingType ) {
	for ( std::vector<const Ring*>::const_iterator outerRing = road->first.second.begin();
	      outerRing != road->first.second.end();
	      ++outerRing ) {
	  if ( (*outerRing)->getType() == outerSeedRingType ) {
	    if ( (*innerRing)->containsDetId(InnerSeedRing,InnerSeedRingPhi,dphi_scalefactor) &&
		 (*outerRing)->containsDetId(OuterSeedRing,OuterSeedRingPhi,dphi_scalefactor) ) {
	      return &(road->first);
	    }
	  }
	}
      }
    }
  }
      
  edm::LogError("RoadSearch") << "RoadSeed could not be found for inner SeedRing type: " << innerSeedRingType << " DetId: " << InnerSeedRing.rawId() 
			      << " at " << InnerSeedRingPhi
			      << " and outer SeedRing type : " << outerSeedRingType << " DetID: " << OuterSeedRing.rawId() 
			      << " at " << OuterSeedRingPhi;
  return 0;
}

const Roads::RoadSeed* Roads::getRoadSeed(std::vector<DetId> seedRingDetIds,
					  std::vector<double> seedRingHitsPhi,
					  double dphi_scalefactor) const {
  //
  // loop over roads and return first road which contains all seedRingDetIds
  //

  for ( const_iterator road = roadMap_.begin(); road != roadMap_.end(); ++road ) {
    unsigned int found = 0;
    for ( unsigned int detIdCounter = 0;
	  detIdCounter < seedRingDetIds.size();
	  ++detIdCounter ) {
      DetId      id   = RoadSearchDetIdHelper::ReturnRPhiId(seedRingDetIds[detIdCounter]);
      double     phi  = seedRingHitsPhi[detIdCounter];
      Ring::type type = getRingType(id);

      bool foundInInnerRing = false;
      for ( std::vector<const Ring*>::const_iterator innerRing = road->first.first.begin();
	    innerRing != road->first.first.end();
	    ++innerRing ) {
	if ( (*innerRing)->getType() == type ) {
	  if ( (*innerRing)->containsDetId(id,phi,dphi_scalefactor) ) {
	    ++found;
	    foundInInnerRing = true;
	  }
	}
      }

      if ( !foundInInnerRing ) {
	for ( std::vector<const Ring*>::const_iterator outerRing = road->first.second.begin();
	      outerRing != road->first.second.end();
	      ++outerRing ) {
	  if ( (*outerRing)->getType() == type ) {
	    if ( (*outerRing)->containsDetId(id,phi,dphi_scalefactor) ) {
	      ++found;
	    }
	  }
	}
      }

      if ( found == seedRingDetIds.size() ) {
	      return &(road->first);
      }
    }
  }

  std::ostringstream ost;
  
  ost << "RoadSeed could not be found for following hits:\n";
  for ( unsigned int detIdCounter = 0;
	detIdCounter < seedRingDetIds.size();
	++detIdCounter ) {
    ost << "Hit DetId: " << seedRingDetIds[detIdCounter].rawId() << " phi: " << seedRingHitsPhi[detIdCounter] << "\n";
  }
  
  edm::LogError("RoadSearch") << ost.str();
  
  return 0;
}

const Roads::type Roads::getRoadType(const RoadSeed *const seed) const {
  //
  // check if one of the outer rings is in TOB, then mark as RPhi
  // problematic for transition region
  bool TOBRing = false;
  for ( std::vector<const Ring*>::const_iterator ring = seed->second.begin();
	ring != seed->second.end();
	++ring) {
    if ( (*ring)->getType() == Ring::TOBRing) {
      TOBRing = true;
    }
  }
  if ( TOBRing ) {
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
