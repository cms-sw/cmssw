#ifndef DetLayers_DetLayer_h
#define DetLayers_DetLayer_h

/** \class DetLayer
 *  The DetLayer is the detector abstraction used for track reconstruction.
 *  It inherits from GeometricSearchDet the interface for accessing 
 *  components and compatible components. 
 *  It extends the interface by providing navigation capability 
 *  from one layer to another. 
 *  The Navigation links must be created in a 
 *  NavigationSchool and activated with a NavigationSetter before they 
 *  can be used.
 *
 */

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/NavigableLayer.h"
#include "TrackingTools/DetLayers/interface/NavigationDirection.h"

#include <vector>

class DetLayer : public  virtual GeometricSearchDet {  
 public:

  typedef GeomDetEnumerators::SubDetector SubDetector;
  typedef GeomDetEnumerators::Location Location;

  DetLayer(bool ibar) : theNavigableLayer(0), theSeqNum(-1), iAmBarrel(ibar) {}

  virtual ~DetLayer();

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

  /// Return the NavigableLayer associated with this DetLayer
  NavigableLayer* navigableLayer() const { return theNavigableLayer;}

  /// Set the NavigableLayer associated with this DetLayer
  void setNavigableLayer( NavigableLayer* nlp);

  /// Return the next (closest) layer(s) that can be reached in the specified
  /// NavigationDirection
  template<typename... Args>
  std::vector<const DetLayer*> 
  nextLayers(Args && ...args) const {
    return theNavigableLayer
      ? theNavigableLayer->nextLayers(std::forward<Args>(args)...)
      : std::vector<const DetLayer*>();
  }
  
  /// Returns all layers compatible 
  template<typename... Args>
  std::vector<const DetLayer*> 
  compatibleLayers(Args && ...args) const {
    return theNavigableLayer
      ? theNavigableLayer->compatibleLayers(std::forward<Args>(args)...)
      : std::vector<const DetLayer*>();
  }
  
  
 private:
  NavigableLayer* theNavigableLayer;
  int theSeqNum;
  bool iAmBarrel;
};

#endif 
