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
			      const float Chi2Cut,
			      const std::vector<double>& anAnnealingProgram, bool debug);
  virtual ~SiTrackerMultiRecHitUpdator(){};
  
  //calls the update method in order to build a SiTrackerMultiRecHit 
  virtual TransientTrackingRecHit::RecHitPointer  buildMultiRecHit(const std::vector<const TrackingRecHit*>& rhv, 
								   TrajectoryStateOnSurface tsos,
								   float annealing=1.) const;
  
  //updates an existing SiTrackerMultiRecHit
  //in case a different kind of rechit is passed it returns clone(tsos)
  virtual TransientTrackingRecHit::RecHitPointer  update( TransientTrackingRecHit::ConstRecHitPointer original,  
							  TrajectoryStateOnSurface tsos,
							  double annealing=1.) const;
  
  //returns a SiTrackerMultiRecHit out of the transient components	
  TransientTrackingRecHit::RecHitPointer update( TransientTrackingRecHit::ConstRecHitContainer& tcomponents,  
					         TrajectoryStateOnSurface tsos,
						 double annealing=1.) const;

  //computes weights or the cut-off value (depending on CutWeight variable)
  double ComputeWeight(const TrajectoryStateOnSurface& tsos, const TransientTrackingRecHit& aRecHit, 
		       bool CutWeight, double annealing=1.) const;
  template <unsigned int N> double ComputeWeight(const TrajectoryStateOnSurface& tsos,
                                                 const TransientTrackingRecHit& aRecHit, 
						 bool CutWeight, double annealing=1.) const; 

  //computes parameters for 2 dim hits 
  std::pair<AlgebraicVector2,AlgebraicSymMatrix22> ComputeParameters2dim(const TrajectoryStateOnSurface& tsos, 
									 const TransientTrackingRecHit& aRecHit) const;
  template <unsigned int N> std::pair<AlgebraicVector2,AlgebraicSymMatrix22> ComputeParameters2dim (const TrajectoryStateOnSurface& tsos, 
												    const TransientTrackingRecHit& aRecHit ) const; 

  const std::vector<double>&  annealingProgram() const {return theAnnealingProgram;}
  const std::vector<double>& getAnnealingProgram() const {return theAnnealingProgram;}

  
private:
  //Obsolete methods
  //LocalError calcParametersError(TransientTrackingRecHit::ConstRecHitContainer& map) const;
  //LocalPoint calcParameters(TransientTrackingRecHit::ConstRecHitContainer& map, const LocalError& er) const;

  LocalParameters calcParameters(const TrajectoryStateOnSurface& tsos, 
				 std::vector<std::pair<const TrackingRecHit*, float> >& aHitMap) const;
  
  const TransientTrackingRecHitBuilder* theBuilder;
  const TrackingRecHitPropagator* theHitPropagator;
  double theChi2Cut;
  const std::vector<double> theAnnealingProgram;
  TkClonerImpl theHitCloner;
  bool debug_;

};
#endif
