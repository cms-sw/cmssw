#ifndef TkDetLayers_TIDLayer_h
#define TkDetLayers_TIDLayer_h


#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "TIDRing.h"


/** A concrete implementation for TID layer 
 *  built out of TIDRings
 */

#pragma GCC visibility push(hidden)
class TIDLayer GCC11_FINAL : public RingedForwardLayer, public GeometricSearchDetWithGroups {
 public:
  TIDLayer(std::vector<const TIDRing*>& rings);
  ~TIDLayer();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const {return theComps;}

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const;

  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::TID;}


 private:
  // private methods for the implementation of groupedCompatibleDets()
  virtual BoundDisk* computeDisk( const std::vector<const TIDRing*>& rings) const;

  virtual std::vector<int> ringIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
							  const Propagator& prop ) const;

 protected:  
  //  bool isCompatible( const TrajectoryStateOnSurface& ms,
  //	     const MeasurementEstimator& est) const;

  int findClosest( const GlobalPoint[3] ) const;
  
  int findNextIndex( const GlobalPoint[3] , int ) const;
  
  bool overlapInR( const TrajectoryStateOnSurface& tsos, int i, double ymax) const;
  
  
  float computeWindowSize( const GeomDet* det, 
  			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;
  
  static void
  orderAndMergeLevels(const TrajectoryStateOnSurface& tsos,
		      const Propagator& prop,
		      const std::vector<std::vector<DetGroup> > & groups,
		      const std::vector<int> & indices,
		      std::vector<DetGroup> & result );


 protected:
  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeomDet*> theBasicComps;
  
};


#pragma GCC visibility pop
#endif 
