#ifndef TkDetLayers_PixelBlade_h
#define TkDetLayers_PixelBlade_h

#include "RecoTracker/TkDetLayers/interface/BoundDiskSector.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

/** A concrete implementation for PixelBlade
 */

class PixelBlade : public GeometricSearchDet{
 public:

  PixelBlade(vector<const GeomDet*>& frontDets,
	     vector<const GeomDet*>& backDets  );

  ~PixelBlade(){};
  
  // GeometricSearchDet interface
  virtual const BoundSurface& surface() const {return *theDiskSector;}

  virtual vector<const GeomDet*> basicComponents() const {return theDets;}

  virtual vector<const GeometricSearchDet*> components() const;

    
  virtual pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
	      const MeasurementEstimator&) const;

  virtual vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;

  virtual vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;

  

  virtual bool hasGroups() const {return true;}

  //Extension of the interface
  virtual const BoundDiskSector& specificSurface() const {return *theDiskSector;}


 private:
  vector<const GeomDet*> theDets;
  vector<const GeomDet*> theFrontDets;
  vector<const GeomDet*> theBackDets;
  
  ReferenceCountingPointer<BoundDiskSector> theDiskSector;
  ReferenceCountingPointer<BoundDiskSector> theFrontDiskSector;
  ReferenceCountingPointer<BoundDiskSector> theBackDiskSector;

      
  
};


#endif 
