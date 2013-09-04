#ifndef TkDetLayers_PixelForwardLayer_h
#define TkDetLayers_PixelForwardLayer_h


#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "PixelBlade.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"


/** A concrete implementation for PixelForward layer 
 *  built out of ForwardPixelBlade
 */

#pragma GCC visibility push(hidden)
class PixelForwardLayer GCC11_FINAL : public ForwardDetLayer, public GeometricSearchDetWithGroups {
 public:
  PixelForwardLayer(std::vector<const PixelBlade*>& blades);
  ~PixelForwardLayer();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const {return theComps;}
  
  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const;

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
		      PropagationDirection propDir,bool innerDisk) const;

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;
  
 private:
  typedef PeriodicBinFinderInPhi<double>   BinFinderType;
  // need separate objects for inner and outer disk
  // or a smarter bin finder class
  BinFinderType    theBinFinder_inner;
  BinFinderType    theBinFinder_outer;
  unsigned int     _num_innerpanels;
  unsigned int     _num_outerpanels;

  std::vector<float> theBinFinder_byR;
  std::vector<unsigned int> theBinFinder_byR_index;
  std::vector<unsigned int> theBinFinder_byR_nextindex;
  bool useR;
  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeomDet*> theBasicComps;
};


#pragma GCC visibility pop
#endif 
