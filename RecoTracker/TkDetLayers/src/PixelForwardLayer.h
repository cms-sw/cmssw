#ifndef TkDetLayers_PixelForwardLayer_h
#define TkDetLayers_PixelForwardLayer_h


#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "PixelBlade.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"


/** A concrete implementation for PixelForward layer 
 *  built out of ForwardPixelBlade
 */

#pragma GCC visibility push(hidden)
class PixelForwardLayer GCC11_FINAL : public ForwardDetLayer {
 public:
  PixelForwardLayer(std::vector<const PixelBlade*>& blades);
  ~PixelForwardLayer();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const __attribute__ ((cold)) {return theComps;}
  
  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const __attribute__ ((hot));

  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::PixelEndcap];}
  

 private:  
  // methods for groupedCompatibleDets implementation
  static int computeHelicity(const GeometricSearchDet* firstBlade,const GeometricSearchDet* secondBlade);

  struct SubTurbineCrossings {
    SubTurbineCrossings(): isValid(false){}
    SubTurbineCrossings( int ci, int ni, float nd) : 
      isValid(true),closestIndex(ci), nextIndex(ni), nextDistance(nd) {}
    
    bool  isValid;
    int   closestIndex;
    int   nextIndex;
    float nextDistance;
  };
  
  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubTurbineCrossings& crossings,
			float window, 
			std::vector<DetGroup>& result) const __attribute__ ((hot));
  
  SubTurbineCrossings 
    computeCrossings( const TrajectoryStateOnSurface& startingState,
		      PropagationDirection propDir) const __attribute__ ((hot));

  static  float computeWindowSize( const GeomDet* det, 
				   const TrajectoryStateOnSurface& tsos, 
				   const MeasurementEstimator& est);
  
 private:
  typedef PeriodicBinFinderInPhi<float>   BinFinderType;
  BinFinderType    theBinFinder;

  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeomDet*> theBasicComps;
};


#pragma GCC visibility pop
#endif 
