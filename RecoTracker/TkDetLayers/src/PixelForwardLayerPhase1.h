#ifndef TkDetLayers_PixelForwardLayerPhase1_h
#define TkDetLayers_PixelForwardLayerPhase1_h


#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "PixelBlade.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
#include "PixelForwardLayer.h"

/** A concrete implementation for PixelForward layer 
 *  built out of ForwardPixelBlade
 */

#pragma GCC visibility push(hidden)
class PixelForwardLayerPhase1 GCC11_FINAL : public ForwardDetLayer {

 public:
  PixelForwardLayerPhase1(std::vector<const PixelBlade*>& blades);
  ~PixelForwardLayerPhase1();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const __attribute__ ((cold)) {return theComps;}
  
  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const __attribute__ ((hot));
  
  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::PixelEndcap;}
  

 private:  

  // methods for groupedCompatibleDets implementation
  int computeHelicity(const GeometricSearchDet* firstBlade,const GeometricSearchDet* secondBlade) const;

  struct SubTurbineCrossings {
    SubTurbineCrossings(): isValid(false){};
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
			std::vector<DetGroup>& result,
			bool innerDisk) const;
  
  SubTurbineCrossings 
    computeCrossings( const TrajectoryStateOnSurface& startingState,
		      PropagationDirection propDir, bool innerDisk) const;
  
  float computeWindowSize( const GeomDet* det, 
                           const TrajectoryStateOnSurface& tsos, 
                           const MeasurementEstimator& est) const;
  
 private:
  // need separate objects for inner and outer disk
  // or a smarter bin finder class
  typedef PeriodicBinFinderInPhi<double>   BinFinderType;

  BinFinderType    theBinFinder_inner;
  BinFinderType    theBinFinder_outer;
  unsigned int     _num_innerpanels;
  unsigned int     _num_outerpanels;

  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeomDet*> theBasicComps;

};

#pragma GCC visibility pop

#endif 
