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
 *  $Date: 2007/08/21 20:48:47 $
 *  $Revision: 1.11 $
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

  DetLayer() : theNavigableLayer(0){};

  virtual ~DetLayer();

  // Extension of the interface 

  /// The type of detector (PixelBarrel, PixelEndcap, TIB, TOB, TID, TEC, CSC, DT, RPCBarrel, RPCEndcap)
  virtual SubDetector subDetector() const = 0;

  /// Which part of the detector (barrel, endcap)
  virtual Location location() const = 0;

  /// Return the NavigableLayer associated with this DetLayer
  NavigableLayer* navigableLayer() const { return theNavigableLayer;}

  /// Set the NavigableLayer associated with this DetLayer
  virtual void setNavigableLayer( NavigableLayer* nlp);

  /// Return the next (closest) layer(s) that can be reached in the specified
  /// NavigationDirection
  virtual std::vector<const DetLayer*> 
  nextLayers( NavigationDirection direction) const;

  /// Return the next (closest) layer(s) compatible with the specified
  /// FreeTrajectoryState and PropagationDirection
  virtual std::vector<const DetLayer*> 
  nextLayers( const FreeTrajectoryState& fts, 
	      PropagationDirection timeDirection) const;

  /// Return all layers that can be reached from this one along the
  /// specified PropagationDirection 
  std::vector<const DetLayer*> 
  compatibleLayers( NavigationDirection direction ) const;

  /// Returns all layers compatible with the specified FreeTrajectoryState
  /// and PropagationDirection  
  std::vector<const DetLayer*> 
  compatibleLayers(const FreeTrajectoryState& fts, 
		   PropagationDirection timeDirection) const;


  /// Returns all layers compatible with the specified FreeTrajectoryState
  /// and PropagationDirection. the counter is used to count how many levels the recursive call was depth;
  /// counter = -1 is returned in case of error.
  std::vector<const DetLayer*> 
  compatibleLayers(const FreeTrajectoryState& fts, 
		   PropagationDirection timeDirection,
		   int& counter) const;

  
 private:
  NavigableLayer* theNavigableLayer;

};

#endif 
