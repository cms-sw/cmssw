#ifndef KinematicConstrainedVertexUpdator_H
#define KinematicConstrainedVertexUpdator_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"
#include "RecoVertex/KinematicFit/interface/VertexKinematicConstraint.h"

/**
 * Class caching the math part for
 * KinematicConstrainedVertexFitter
 */

class KinematicConstrainedVertexUpdator
{
public:

/**
 * Default constructor and destructor
 */
  KinematicConstrainedVertexUpdator();
  
 ~KinematicConstrainedVertexUpdator();
 
/**
 * Method updating the states. Takes a vector of full parameters:
 * (x,y,z,particle_1,...,particle_n), corresponding linearization 
 * point: vector of states and GlobalPoint, 
 * and constraint to be applied during the vertex fit.
 * Returns refitted vector of 7n+3 parameters and corresponding
 * covariance matrix, where n - number of tracks.
 */ 
 std::pair<std::pair<std::vector<KinematicState>, AlgebraicMatrix >, RefCountedKinematicVertex > 
  update(const AlgebraicVector& inState, const AlgebraicMatrix& inCov, const std::vector<KinematicState> &lStates, 
                                   const GlobalPoint& lPoint,MultiTrackKinematicConstraint * cs)const;
 
private:
				       
 KinematicVertexFactory * vFactory;
 VertexKinematicConstraint * vConstraint;			       	
};
#endif
