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
// $Author: tmoulik $
// $Date: 2006/07/25 20:22:51 $
// $Revision: 1.5 $
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
			       unsigned int panel,
			       unsigned int module);

  DetId constructTrackerPXFDetId(unsigned int fw_bw,
				 unsigned int disk,
				 unsigned int blade,
				 unsigned int panel,
				 unsigned int module);

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
			  unsigned int panel,
			  unsigned int module);

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
  void fillTIBGeometryArray(const TrackerGeometry &tracker);
  void fillTIDGeometryArray(const TrackerGeometry &tracker);
  void fillTOBGeometryArray(const TrackerGeometry &tracker);
  void fillPXBGeometryArray(const TrackerGeometry &tracker);
  void fillPXFGeometryArray(const TrackerGeometry &tracker);

 private:
  
  int verbosity_;

  std::vector<Ring> rings_;
  
  int tec_[2][9][2][8][7][20][3]; // tec[fw_bw][wheel][ring][petal][petal_fw_bw][module][stereo]
  int tib_[4][2][2][3][3];        // tib[layer][str_fw_bw][str_int_ext][module][stereo]
  int tid_[2][3][3][2][20][3];    // tid[side][wheel][ring][module_fw_bw][module][stereo]
  int tob_[6][2][74][6][3];       // tob[layer][rod_fw_bw][rod][module][stereo]
  int pxb_[3][8][44];             // pxb[layer][ladder][module]
  int pxf_[2][2][24][2][4];       // pxf[side][disk][blade][panel][module]

};

#endif
