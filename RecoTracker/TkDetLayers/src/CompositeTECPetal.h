#ifndef TkDetLayers_CompositeTECPetal_h
#define TkDetLayers_CompositeTECPetal_h


#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TECPetal.h"
#include "TECWedge.h"
#include "SubLayerCrossings.h"

#include "FWCore/Utilities/interface/Visibility.h"


/** A concrete implementation for TEC petals
 */

#pragma GCC visibility push(hidden)
class CompositeTECPetal GCC11_FINAL : public TECPetal{
 public:
  struct WedgePar { float theR, thetaMin, thetaMax;};

  CompositeTECPetal(std::vector<const TECWedge*>& innerWedges,
		    std::vector<const TECWedge*>& outerWedges);
  
  ~CompositeTECPetal();
  
  // GeometricSearchDet interface  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const {return theComps;}
  
  virtual std::pair<bool, TrajectoryStateOnSurface>
    compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		const MeasurementEstimator&) const;
  
  virtual void
    groupedCompatibleDetsV( const TrajectoryStateOnSurface& startingState,
			    const Propagator& prop,
			    const MeasurementEstimator& est,
			    std::vector<DetGroup> & result) const;
  
  
 private:
  
  
  // private methods for the implementation of groupedCompatibleDets()
  SubLayerCrossings computeCrossings(const TrajectoryStateOnSurface& tsos,
				     PropagationDirection propDir) const dso_internal;
  
  
  
  bool addClosest( const TrajectoryStateOnSurface& tsos,
		   const Propagator& prop,
		   const MeasurementEstimator& est,
		   const SubLayerCrossing& crossing,
		   std::vector<DetGroup>& result) const dso_internal;
  
  void searchNeighbors( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			float window, 
			std::vector<DetGroup>& result,
			bool checkClosest) const dso_internal;
  
  
  static
    float computeWindowSize( const GeomDet* det, 
			     const TrajectoryStateOnSurface& tsos, 
			     const MeasurementEstimator& est) dso_internal;
  
  int findBin( float R,int layer) const dso_internal;
  
  WedgePar const &  findPar(int index,int diskSectorType) const  dso_internal {
    return (diskSectorType == 0) ? theFrontPars[index] : theBackPars[index];
  }
  
  const std::vector<const TECWedge*>& subLayer( int ind) const  dso_internal {
    return (ind==0 ? theFrontComps : theBackComps);
  }

  
 private:
  std::vector<const GeomDet*> theBasicComps;
  std::vector<const GeometricSearchDet*> theComps;

  std::vector<const TECWedge*> theFrontComps;
  std::vector<const TECWedge*> theBackComps;
  
  std::vector<float> theFrontBoundaries;
  std::vector<float> theBackBoundaries;
  std::vector<WedgePar> theFrontPars;
  std::vector<WedgePar> theBackPars;
  
  ReferenceCountingPointer<BoundDiskSector> theFrontSector;
  ReferenceCountingPointer<BoundDiskSector> theBackSector;  
  
};


#pragma GCC visibility pop
#endif 
