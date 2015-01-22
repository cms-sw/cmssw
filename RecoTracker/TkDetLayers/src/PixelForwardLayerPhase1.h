#ifndef TkDetLayers_PixelForwardLayerPhase1_h
#define TkDetLayers_PixelForwardLayerPhase1_h


#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "Phase1PixelBlade.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"


/** A concrete implementation for PixelForward layer
 *  built out of ForwardPhase1PixelBlade
 */

#pragma GCC visibility push(hidden)

class PixelForwardLayerPhase1 GCC11_FINAL : public ForwardDetLayer {
 public:
  PixelForwardLayerPhase1(std::vector<const Phase1PixelBlade*>& blades);
  ~PixelForwardLayerPhase1();

  // GeometricSearchDet interface

  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const __attribute__ ((cold)) {return theComps;}

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const __attribute__ ((hot));

  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::P1PXEC];}


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
			std::vector<DetGroup>& result,
			bool innerDisk) const __attribute__ ((hot));

  SubTurbineCrossings
    computeCrossings( const TrajectoryStateOnSurface& startingState,
		      PropagationDirection propDir,bool innerDisk) const __attribute__ ((hot));

  static float computeWindowSize( const GeomDet* det,
			   const TrajectoryStateOnSurface& tsos,
			   const MeasurementEstimator& est);

 private:
  typedef PeriodicBinFinderInPhi<float>   BinFinderType;
  // need separate objects for inner and outer disk
  // or a smarter bin finder class
  BinFinderType    theBinFinder_inner;
  BinFinderType    theBinFinder_outer;
  unsigned int     _num_innerpanels;
  unsigned int     _num_outerpanels;

  std::vector<float> theBinFinder_byR;
  std::vector<unsigned int> theBinFinder_byR_index;
  std::vector<unsigned int> theBinFinder_byR_nextindex;
  // bool useR;
  std::vector<const GeometricSearchDet*> theComps;
  std::vector<const GeomDet*> theBasicComps;
};


#pragma GCC visibility pop
#endif
