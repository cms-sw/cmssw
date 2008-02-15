#ifndef SiTrackerMultiRecHitUpdator_h
#define SiTrackerMultiRecHitUpdator_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"


#include <vector>

class SiTrackerMultiRecHit;
class TrajectoryStateOnSurface;
class TrackingRecHit;	
class TransientTrackingRecHitBuilder;
class LocalError;
class TrackingRecHitPropagator;
/*
builds a TSiTrackerMultiRecHitout of a vector of TrackingRecHit ans a tsos
or updates an existing TSiTrackerMultiRecHitout given a tsos.
*/
class SiTrackerMultiRecHitUpdator{
	public:
  SiTrackerMultiRecHitUpdator(const TransientTrackingRecHitBuilder* builder,
				    const TrackingRecHitPropagator* hitpropagator,
				    const std::vector<double>& anAnnealingProgram);
	virtual ~SiTrackerMultiRecHitUpdator(){};

	const std::vector<double>&  annealingProgram() const {return theAnnealingProgram;}

	//builds a TSiTrackerMultiRecHit 
		virtual TransientTrackingRecHit::RecHitPointer  buildMultiRecHit(const std::vector<const TrackingRecHit*>& rhv, 
										 TrajectoryStateOnSurface tsos,
								 	         double annealing=1.) const;
	
	//updates an existing TSiTrackerMultiRecHit
	//in case a diffrenet king of rechit is passed it returns  clone(tsos)
	virtual TransientTrackingRecHit::RecHitPointer  update( TransientTrackingRecHit::ConstRecHitPointer original,  
								TrajectoryStateOnSurface tsos,
								double annealing=1.) const;

	//returns a TSiTrackerMultiRecHit out of the transient components	
        virtual TransientTrackingRecHit::RecHitPointer  update( TransientTrackingRecHit::ConstRecHitContainer& tcomponents,          
                                                                TrajectoryStateOnSurface tsos,
								double annealing=1.) const;

	const std::vector<double>& getAnnealingProgram() const {return theAnnealingProgram;}
	
	private:
	LocalError calcParametersError(TransientTrackingRecHit::ConstRecHitContainer& map) const;
	LocalPoint calcParameters(TransientTrackingRecHit::ConstRecHitContainer& map, const LocalError& er) const;

	const TransientTrackingRecHitBuilder* theBuilder;
	const TrackingRecHitPropagator* theHitPropagator;
	double theChi2Cut;
	const std::vector<double> theAnnealingProgram;

};
#endif
