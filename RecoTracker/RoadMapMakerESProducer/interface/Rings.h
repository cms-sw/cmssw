#ifndef Rings_h
#define Rings_h

/** \class Rings
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Oct. 14, 2005  

 all indices are given in C style and transformed to fortran style
 for final generation of the DetId's

 *
 ************************************************************/

#include <vector>

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "RecoTracker/RoadMapRecord/interface/Ring.h"

#include "DataFormats/DetId/interface/DetId.h"

class Rings {
 
 public:

  typedef std::vector<Ring>::iterator       iterator;
  typedef std::vector<Ring>::const_iterator const_iterator;
  
  Rings(const TrackingGeometry &tracker, unsigned int verbosity = 0);
  
  ~Rings();

  void constructTrackerRings(const TrackingGeometry &tracker);

  void constructTrackerTIBRings(const TrackingGeometry &tracker);
  void constructTrackerTOBRings(const TrackingGeometry &tracker);
  void constructTrackerTIDRings(const TrackingGeometry &tracker);
  void constructTrackerTECRings(const TrackingGeometry &tracker);
  void constructTrackerPXBRings(const TrackingGeometry &tracker);
  void constructTrackerPXFRings(const TrackingGeometry &tracker);

  Ring constructTrackerTIBRing(const TrackingGeometry &tracker,
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

  Ring constructTrackerTOBRing(const TrackingGeometry &tracker,
			       unsigned int layer,
			       unsigned int rod_fw_bw,
			       unsigned int detector);
  
  DetId constructTrackerTOBDetId(unsigned int layer,
				 unsigned int rod_fw_bw,
				 unsigned int rod,
				 unsigned int detector,
				 unsigned int stereo);

  Ring constructTrackerTIDRing(const TrackingGeometry &tracker,
			       unsigned int fw_bw,
			       unsigned int wheel,
			       unsigned int ring);

  DetId constructTrackerTIDDetId(unsigned int fw_bw,
				 unsigned int wheel,
				 unsigned int ring,
				 unsigned int detector_fw_bw,
				 unsigned int detector,
				 unsigned int stereo);

  Ring constructTrackerTECRing(const TrackingGeometry &tracker,
			       unsigned int fw_bw,
			       unsigned int wheel,
			       unsigned int ring);

  DetId constructTrackerTECDetId(unsigned int fw_bw,
				 unsigned int wheel,
				 unsigned int petal_fw_bw,
				 unsigned int petal,
				 unsigned int ring,
				 unsigned int detector_fw_bw,
				 unsigned int detector,
				 unsigned int stereo);

  Ring constructTrackerPXBRing(const TrackingGeometry &tracker,
			       unsigned int layer,
			       unsigned int detector);

  DetId constructTrackerPXBDetId(unsigned int layer,
				 unsigned int ladder,
				 unsigned int detector);

  Ring constructTrackerPXFRing(const TrackingGeometry &tracker,
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

  double determineExtensions(const TrackingGeometry &tracker, 
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

 private:
  
  int verbosity_;

  std::vector<Ring> rings_;

};

#endif
