#ifndef RECOTRACKER_RINGS_H
#define RECOTRACKER_RINGS_H

//
// Package:         RecoTracker/RoadMapMakerESProducer
// Class:           Rings
// 
// Description:     The Rings object creates and povides
//                  all Rings in the detector.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Jan 12 21:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/03/03 22:23:12 $
// $Revision: 1.3 $
//

#include <vector>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/RoadMapRecord/interface/Ring.h"

#include "DataFormats/DetId/interface/DetId.h"

class Rings {
 
 public:

  typedef std::vector<Ring>::iterator       iterator;
  typedef std::vector<Ring>::const_iterator const_iterator;
  
  Rings(const TrackerGeometry &tracker, unsigned int verbosity = 0);
  
  ~Rings();

  void constructTrackerRings(const TrackerGeometry &tracker);

  void constructTrackerTIBRings(const TrackerGeometry &tracker);
  void constructTrackerTOBRings(const TrackerGeometry &tracker);
  void constructTrackerTIDRings(const TrackerGeometry &tracker);
  void constructTrackerTECRings(const TrackerGeometry &tracker);
  void constructTrackerPXBRings(const TrackerGeometry &tracker);
  void constructTrackerPXFRings(const TrackerGeometry &tracker);

  Ring constructTrackerTIBRing(const TrackerGeometry &tracker,
			       unsigned int layer,
			       unsigned int fw_bw,
			       unsigned int ext_int,
			       unsigned int detector);
  
  DetId constructTrackerTIBDetId(unsigned int layer,
				 unsigned int fw_bw,
				 unsigned int ext_int,
				 unsigned int string,
				 unsigned int detector,
				 unsigned int stereo);

  Ring constructTrackerTOBRing(const TrackerGeometry &tracker,
			       unsigned int layer,
			       unsigned int rod_fw_bw,
			       unsigned int detector);
  
  DetId constructTrackerTOBDetId(unsigned int layer,
				 unsigned int rod_fw_bw,
				 unsigned int rod,
				 unsigned int detector,
				 unsigned int stereo);

  Ring constructTrackerTIDRing(const TrackerGeometry &tracker,
			       unsigned int fw_bw,
			       unsigned int wheel,
			       unsigned int ring);

  DetId constructTrackerTIDDetId(unsigned int fw_bw,
				 unsigned int wheel,
				 unsigned int ring,
				 unsigned int detector_fw_bw,
				 unsigned int detector,
				 unsigned int stereo);

  Ring constructTrackerTECRing(const TrackerGeometry &tracker,
			       unsigned int fw_bw,
			       unsigned int wheel,
			       unsigned int ring);

  DetId constructTrackerTECDetId(unsigned int fw_bw,
				 unsigned int wheel,
				 unsigned int petal_fw_bw,
				 unsigned int petal,
				 unsigned int ring,
				 unsigned int module,
				 unsigned int stereo);

  Ring constructTrackerPXBRing(const TrackerGeometry &tracker,
			       unsigned int layer,
			       unsigned int detector);

  DetId constructTrackerPXBDetId(unsigned int layer,
				 unsigned int ladder,
				 unsigned int detector);

  Ring constructTrackerPXFRing(const TrackerGeometry &tracker,
			       unsigned int fw_bw,
			       unsigned int disk,
			       unsigned int detector);

  DetId constructTrackerPXFDetId(unsigned int fw_bw,
				 unsigned int disk,
				 unsigned int blade,
				 unsigned int detector);

  Ring* getTrackerRing(DetId id);

  Ring* getTrackerTIBRing(unsigned int layer,
			  unsigned int fw_bw,
			  unsigned int ext_int,
			  unsigned int detector);
  
  Ring* getTrackerTOBRing(unsigned int layer,
			  unsigned int rod_fw_bw,
			  unsigned int detector);

  Ring* getTrackerTIDRing(unsigned int fw_bw,
			  unsigned int wheel,
			  unsigned int ring);

  Ring* getTrackerTECRing(unsigned int fw_bw,
			  unsigned int wheel,
			  unsigned int ring);

  Ring* getTrackerPXBRing(unsigned int layer,
			  unsigned int detector);

  Ring* getTrackerPXFRing(unsigned int fw_bw,
			  unsigned int disk,
			  unsigned int detector);

  inline std::vector<Ring>* getRings() { return &rings_; }

  void fixIndexNumberingScheme();

  void setVerbosity(unsigned int input) { verbosity_ = input; }

  double determineExtensions(const TrackerGeometry &tracker, 
			     DetId id, 
			     float &rmin, float &rmax, 
			     float &zmin, float& zmax, Ring::type type);

  std::vector<unsigned int> dumpOldStyle(std::string ascii_filename = "geodump.dat", bool writeFile = true);

  inline std::vector<unsigned int> getNumberOfLayersPerSubdetector() { return dumpOldStyle("",false); }

  std::string dumpOldStyleTIB(unsigned int &nLayer);
  std::string dumpOldStyleTOB(unsigned int &nLayer);
  std::string dumpOldStyleTID(unsigned int &nLayer);
  std::string dumpOldStyleTEC(unsigned int &nLayer);
  std::string dumpOldStylePXB(unsigned int &nLayer);
  std::string dumpOldStylePXF(unsigned int &nLayer);

  void fillTECGeometryArray(const TrackerGeometry &tracker);

 private:
  
  int verbosity_;

  std::vector<Ring> rings_;

  int tec_[2][9][2][8][7][20][3];

};

#endif
