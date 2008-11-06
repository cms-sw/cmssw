#ifndef SiTrackerMultiRecHitUpdatorMTF_h
#define SiTrackerMultiRecHitUpdatorMTF_h
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrajectoryMeasurement.h"

#include <vector>

class SiTrackerMultiRecHit;
class TrajectoryStateOnSurface;
class TrackingRecHit;	
class TransientTrackingRecHitBuilder;
class LocalError;
class TrackingRecHitPropagator;

typedef TrajectoryStateOnSurface TSOS;


/*
builds a TSiTrackerMultiRecHit out of a vector of TrackingRecHit and a vector of tsos
or updates an existing TSiTrackerMultiRecHitout given a vector of tsos.
*/
class SiTrackerMultiRecHitUpdatorMTF{
 public:
    typedef std::pair<LocalPoint,LocalError>  LocalParameters;

  SiTrackerMultiRecHitUpdatorMTF(const TransientTrackingRecHitBuilder* builder,
				 const TrackingRecHitPropagator* hitpropagator,
				 const float Chi2Cut,
				 const std::vector<double>& anAnnealingProgram);
  virtual ~SiTrackerMultiRecHitUpdatorMTF(){};
  
  //  const std::vector<double>&  annealingProgram() const {return theAnnealingProgram;}
  
  //builds a TSiTrackerMultiRecHit 
  virtual TransientTrackingRecHit::RecHitPointer  buildMultiRecHit(TrajectoryStateOnSurface& tsos,
								   TransientTrackingRecHit::ConstRecHitContainer& hits,
								   MultiTrajectoryMeasurement* mtm,
								   float annealing=1.) const;
  
  //updates an existing TSiTrackerMultiRecHit
  //in case a diffrenet king of rechit is passed it returns  clone(tsos)
  virtual double update(TransientTrackingRecHit::ConstRecHitPointer original,  
		       TrajectoryStateOnSurface tsos,
		       float annealing=1.) const;
  
  //returns a TSiTrackerMultiRecHit out of the transient components	
  virtual TransientTrackingRecHit::RecHitPointer update(double rowsum, 
							MultiTrajectoryMeasurement* mtm, 
							double c, 
							std::vector< std::pair<TransientTrackingRecHit::RecHitPointer, double> >& mymap, 
							std::vector<TransientTrackingRecHit::RecHitPointer>& updatedcomponents, 
							float annealing=1.) const;

  
  virtual double updaterow(TransientTrackingRecHit::ConstRecHitContainer& tcomponents,
			  TrajectoryStateOnSurface& tsos,
			  std::vector<TransientTrackingRecHit::RecHitPointer>& updatedcomponents,
			  float annealing) const;
  
  virtual double updatecolumn(TransientTrackingRecHit::ConstRecHitPointer trechit, 
			     MultiTrajectoryMeasurement* multi,
			     float annealing) const;
  
  virtual double calculatecut(TransientTrackingRecHit::ConstRecHitContainer& trechit, 
			     TrajectoryStateOnSurface& vtsos, 
			     std::vector<TransientTrackingRecHit::RecHitPointer>& updatedcomponents,
			     float annealing) const;
  
  virtual std::vector< std::pair<TransientTrackingRecHit::RecHitPointer, double> > mapmaker(TransientTrackingRecHit::ConstRecHitContainer& tcomponents,
											   TrajectoryStateOnSurface& tsos,
											   float annealing) const;
  
  

  virtual std::vector<TransientTrackingRecHit::RecHitPointer> updatecomponents(TransientTrackingRecHit::ConstRecHitContainer& tcomponents,
									       TrajectoryStateOnSurface& tsos,
									       float annealing) const;
  
  

  


	const std::vector<double>& getAnnealingProgram() const {return theAnnealingProgram;}
	
	private:
	LocalError calcParametersError(TransientTrackingRecHit::ConstRecHitContainer& map) const;
	LocalPoint calcParameters(TransientTrackingRecHit::ConstRecHitContainer& map, const LocalError& er) const;
	LocalParameters calcParameters(TransientTrackingRecHit::ConstRecHitContainer& map) const;

	const TransientTrackingRecHitBuilder* theBuilder;
	const TrackingRecHitPropagator* theHitPropagator;
	const float theChi2Cut;
	const std::vector<double> theAnnealingProgram;

};
#endif
