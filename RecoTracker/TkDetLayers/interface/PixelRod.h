#ifndef TkDetLayers_PixelRod_h
#define TkDetLayers_PixelRod_h


#include "TrackingTools/DetLayers/interface/DetRodOneR.h"
#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInZ.h"

/** A concrete implementation for PixelRod
 */

class PixelRod : public DetRodOneR{
 public:
    typedef PeriodicBinFinderInZ<float>   BinFinderType;

  PixelRod(std::vector<const GeomDet*>& theDets);
  ~PixelRod();
  
  // GeometricSearchDet interface

  virtual const std::vector<const GeometricSearchDet*>& components() const;
  
  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  virtual std::vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;

  virtual std::vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;


  virtual bool hasGroups() const {return false;}

 private:
  BinFinderType theBinFinder;
      
  
};


#endif 
