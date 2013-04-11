#ifndef MultiTrackKinematicConstraint_H
#define MultiTrackKinematicConstraint_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"


/**
 * Pure abstract class implementing constraint application 
 * on multiple tracks (back to back, collinearity etc.)
 * To be used by KinematicConstraindeVertexFitter only
 * Class caches the information about calculation of
 * of constraint equation derivatives and values at given
 * linearization point. Point should be of 7*n+3 dimensions
 * Where n - number of particles. 7 - parametrization for 
 * particles is (x,y,z,p_x,p_y,p_z,m), for vertex (x_v,y_v,z_v)
 * Fitter usually takes current parameters as the first step point
 * and the change it to the result of the first iteration.
 *
 * Kirill Prokofiev, October 2003
 */

class MultiTrackKinematicConstraint
{
public:

/**
 *  Default constructor and destructor
 */
 
MultiTrackKinematicConstraint() {}

virtual ~MultiTrackKinematicConstraint() {}

/**
 * Methods making vector of values
 * and derivative matrices with
 * respect to vertex position and
 * particle parameters.
 * Input paramters are put into one vector: 
 * (Vertex position, particle_parameters_1,..., particle_parameters_n)
 */
virtual AlgebraicVector  value(const std::vector<KinematicState>&,
                               const GlobalPoint& ) const = 0; 

virtual AlgebraicMatrix parametersDerivative(const std::vector<KinematicState>&,
                                             const GlobalPoint& ) const = 0;


virtual AlgebraicMatrix positionDerivative(const std::vector<KinematicState>&,
                                           const GlobalPoint& ) const = 0;
				    
virtual int numberOfEquations() const = 0;

virtual MultiTrackKinematicConstraint * clone() const = 0;

};


#endif
