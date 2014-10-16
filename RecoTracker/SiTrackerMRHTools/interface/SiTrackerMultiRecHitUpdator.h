/** \class SiTrackerMultiRecHitUpdator
  *  Builds a SiTrackerMultiRecHit out of a vector of TrackingRecHit
  *  or updates an existing SiTrackerMultiRecHit given a tsos.
  *
  *  \author tropiano, genta
  *  \review in May 2014 by brondolin 
  */

#ifndef SiTrackerMultiRecHitUpdator_h
#define SiTrackerMultiRecHitUpdator_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetWithData.h"

#include <vector>

class SiTrackerMultiRecHit;
class TrajectoryStateOnSurface;
class TrackingRecHit;	
class TransientTrackingRecHitBuilder;
class LocalError;
class TrackingRecHitPropagator;

class SiTrackerMultiRecHitUpdator{

public:

  typedef std::pair<LocalPoint,LocalError>  LocalParameters;
  SiTrackerMultiRecHitUpdator(const TransientTrackingRecHitBuilder* builder,
			      const TrackingRecHitPropagator* hitpropagator,
			      const float Chi2Cut1D,
			      const float Chi2Cut2D,
			      const std::vector<double>& anAnnealingProgram, bool debug);
  virtual ~SiTrackerMultiRecHitUpdator(){};
  
  //calls the update method in order to build a SiTrackerMultiRecHit 
  virtual TransientTrackingRecHit::RecHitPointer buildMultiRecHit(const std::vector<const TrackingRecHit*>& rhv, 
								  const TrajectoryStateOnSurface& tsos,
								  MeasurementDetWithData& measDet,
								  float annealing=1.) const;
  
  //updates an existing SiTrackerMultiRecHit
  //in case a different kind of rechit is passed it returns clone(tsos)
  virtual TransientTrackingRecHit::RecHitPointer update( TransientTrackingRecHit::ConstRecHitPointer original,  
							  const TrajectoryStateOnSurface& tsos,
							  MeasurementDetWithData& measDet,
							  double annealing=1.) const;
  
  //returns a SiTrackerMultiRecHit out of the transient components	
  TransientTrackingRecHit::RecHitPointer update( TransientTrackingRecHit::ConstRecHitContainer& tcomponents,  
					         const TrajectoryStateOnSurface& tsos,
						 MeasurementDetWithData& measDet, 
						 double annealing=1. ) const;

  //computes weights or the cut-off value (depending on CutWeight variable)
  double ComputeWeight(const TrajectoryStateOnSurface& tsos, const TransientTrackingRecHit& aRecHit, 
		       bool CutWeight, double annealing=1.) const;
  template <unsigned int N> double ComputeWeight(const TrajectoryStateOnSurface& tsos,
                                                 const TransientTrackingRecHit& aRecHit, 
						 bool CutWeight, double annealing=1.) const; 

  const std::vector<double>&  annealingProgram() const {return theAnnealingProgram;}
  const std::vector<double>& getAnnealingProgram() const {return theAnnealingProgram;}

  const TransientTrackingRecHitBuilder* getBuilder() const {return theBuilder;}
  
private:
  //computes parameters for 1 dim or 2 dim hits 
  LocalParameters calcParameters(const TrajectoryStateOnSurface& tsos, 
				 std::vector<std::pair<const TrackingRecHit*, float> >& aHitMap) const;
  template <unsigned int N> LocalParameters calcParameters(const TrajectoryStateOnSurface& tsos,
                                 std::vector<std::pair<const TrackingRecHit*, float> >& aHitMap) const;
  bool TIDorTEChit(const TrackingRecHit* const& hit) const;
 
  const TransientTrackingRecHitBuilder* theBuilder;
  const TrackingRecHitPropagator* theHitPropagator;
  double theChi2Cut1D;
  double theChi2Cut2D;
  const std::vector<double> theAnnealingProgram;
  TkClonerImpl theHitCloner;
  bool debug_;

};
#endif
