#ifndef TkDetLayers_Phase2OTEndcapLayer_h
#define TkDetLayers_Phase2OTEndcapLayer_h

#define NOTECRINGS 15

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Phase2OTEndcapRing.h"
#include<array>


/** A concrete implementation for Phase 2 OT EC layer 
 *  built out of Phase2OTEndcapRings
 */

#pragma GCC visibility push(hidden)
class Phase2OTEndcapLayer GCC11_FINAL : public RingedForwardLayer, public GeometricSearchDetWithGroups {
 public:
  Phase2OTEndcapLayer(std::vector<const Phase2OTEndcapRing*>& rings);
  ~Phase2OTEndcapLayer();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const;

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const;

  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::P2OTEC];}


 private:
  // private methods for the implementation of groupedCompatibleDets()
  BoundDisk* computeDisk( const std::vector<const Phase2OTEndcapRing*>& rings) const;

  std::array<int,3> ringIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
						   const Propagator& prop ) const;

  //  bool isCompatible( const TrajectoryStateOnSurface& ms,
  //	     const MeasurementEstimator& est) const;

  std::array<int,3> findThreeClosest( const GlobalPoint[NOTECRINGS] ) const;
  
  bool overlapInR( const TrajectoryStateOnSurface& tsos, int i, double ymax) const;
  
  
  float computeWindowSize( const GeomDet* det, 
  			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;
  
  void fillRingPars(int i);

 private:
  std::vector<GeomDet const*> theBasicComps;
  const Phase2OTEndcapRing* theComps[NOTECRINGS];
  struct RingPar { float theRingR, thetaRingMin, thetaRingMax;};
  RingPar ringPars[NOTECRINGS];

};


#pragma GCC visibility pop
#endif 
