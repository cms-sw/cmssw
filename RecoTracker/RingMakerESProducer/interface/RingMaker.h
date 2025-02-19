#ifndef RECOTRACKER_RINGMAKER_H
#define RECOTRACKER_RINGMAKER_H

//
// Package:         RecoTracker/RingMakerESProducer
// Class:           RingMaker
// 
// Description:     The RingMaker object creates and povides
//                  all Rings in the detector.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Tue Oct  3 23:51:34 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/30 02:49:35 $
// $Revision: 1.2 $
//

#include <vector>
#include <string>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/RingRecord/interface/Rings.h"

#include "DataFormats/DetId/interface/DetId.h"

class RingMaker {
 
 public:

  RingMaker(const TrackerGeometry *tracker,
	    std::string configuration);
  
  ~RingMaker();

  void constructRings();

  void constructTIBRings(unsigned int &index);
  void constructTOBRings(unsigned int &index);
  void constructTIDRings(unsigned int &index);
  void constructTECRings(unsigned int &index);
  void constructPXBRings(unsigned int &index);
  void constructPXFRings(unsigned int &index);

  Ring constructTIBRing(unsigned int layer,
			unsigned int fw_bw,
			unsigned int ext_int,
			unsigned int detector);
  
  DetId constructTIBDetId(unsigned int layer,
			  unsigned int fw_bw,
			  unsigned int ext_int,
			  unsigned int string,
			  unsigned int detector,
			  unsigned int stereo);

  Ring constructTOBRing(unsigned int layer,
			unsigned int rod_fw_bw,
			unsigned int detector);
  
  DetId constructTOBDetId(unsigned int layer,
			  unsigned int rod_fw_bw,
			  unsigned int rod,
			  unsigned int detector,
			  unsigned int stereo);

  Ring constructTIDRing(unsigned int fw_bw,
			unsigned int wheel,
			unsigned int ring);

  DetId constructTIDDetId(unsigned int fw_bw,
			  unsigned int wheel,
			  unsigned int ring,
			  unsigned int detector_fw_bw,
			  unsigned int detector,
			  unsigned int stereo);

  Ring constructTECRing(unsigned int fw_bw,
			unsigned int wheel,
			unsigned int ring);

  DetId constructTECDetId(unsigned int fw_bw,
			  unsigned int wheel,
			  unsigned int petal_fw_bw,
			  unsigned int petal,
			  unsigned int ring,
			  unsigned int module,
			  unsigned int stereo);

  Ring constructPXBRing(unsigned int layer,
			unsigned int module);

  DetId constructPXBDetId(unsigned int layer,
			  unsigned int ladder,
			  unsigned int module);

  Ring constructPXFRing(unsigned int fw_bw,
			unsigned int disk,
			unsigned int panel,
			unsigned int module);

  DetId constructPXFDetId(unsigned int fw_bw,
			  unsigned int disk,
			  unsigned int blade,
			  unsigned int panel,
			  unsigned int module);

  double determineExtensions(DetId id, 
			     float &rmin, float &rmax, 
			     float &zmin, float& zmax, 
			     Ring::type type);

  void fillTECGeometryArray();
  void fillTIBGeometryArray();
  void fillTIDGeometryArray();
  void fillTOBGeometryArray();
  void fillPXBGeometryArray();
  void fillPXFGeometryArray();

  inline Rings* getRings() { return rings_;}
 
  bool dumpDetIdsIntoFile(std::string fileName);
  std::string dumpDetIds();

 private:
  
  const TrackerGeometry *tracker_;

  Rings *rings_;
  
  int tib_[5][3][3][57][4][3];    // tib[layer][str_fw_bw][str_int_ext][str][module][stereo]
  int tob_[7][3][75][7][3];       // tob[layer][rod_fw_bw][rod][module][stereo]
  int tid_[3][4][4][3][21][3];    // tid[side][wheel][ring][module_fw_bw][module][stereo]
  int tec2_[3][10][8];             // tec2[side][wheel][ring]
  int tec_[3][10][3][9][8][21][3]; // tec[side][wheel][petal_fw_bw][petal][ring][module][stereo]
  int pxb_[3][44][8];             // pxb[layer][ladder][module]
  int pxf_[2][2][24][2][4];       // pxf[side][disk][blade][panel][module]
  int pxf2_[2][2][2][4];          // pxf[side][disk][panel][module]

  std::string configuration_;

};

#endif
