#ifndef DetLayers_DetLayer_h
#define DetLayers_DetLayer_h

/** \class DetLayer
 *  The DetLayer is the detector abstraction used for track reconstruction.
 *  It inherits from GeometricSearchDet the interface for accessing 
 *  components and compatible components. 
 *  It extends the interface by providing navigation capability 
 *  from one layer to another. 
 *  The Navigation links are managed by the NavigationSchool
 *
 */

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

#include "TrackingTools/DetLayers/interface/NavigableLayer.h"
#include "TrackingTools/DetLayers/interface/NavigationDirection.h"

#include <vector>

class DetLayer : public GeometricSearchDet {  
 public:

  typedef GeomDetEnumerators::SubDetector SubDetector;
  typedef GeomDetEnumerators::Location Location;

  DetLayer(bool doHaveGroup, bool ibar) : GeometricSearchDet(doHaveGroup), theSeqNum(-1), iAmBarrel(ibar) {}

  ~DetLayer() override;

  // a detLayer can be either barrel or forward
  bool isBarrel() const { return iAmBarrel;}
  bool isForward() const { return !isBarrel();}

  // sequential number to be used in "maps"
  int seqNum() const { return theSeqNum;}
  void setSeqNum(int sq) { theSeqNum=sq;}

  // Extension of the interface 

  /// The type of detector (PixelBarrel, PixelEndcap, TIB, TOB, TID, TEC, CSC, DT, RPCBarrel, RPCEndcap)
  virtual SubDetector subDetector() const = 0;

  /// Which part of the detector (barrel, endcap)
  virtual Location location() const = 0;


  
 private:
  int theSeqNum;
  bool iAmBarrel;
};

#endif 
