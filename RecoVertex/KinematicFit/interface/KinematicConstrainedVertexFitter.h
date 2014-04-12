#ifndef KinematicConstrainedVertexFitter_H
#define KinematicConstrainedVertexFitter_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexUpdator.h"
#include "RecoVertex/KinematicFit/interface/VertexKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/ConstrainedTreeBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
 * Class fitting the veretx out of set of tracks via usual LMS
 * with Lagrange multipliers.
 * Additional constraints can be applyed to the tracks during the vertex fit
 * (solves non-factorizabele cases). Since the vertex constraint is included by default, do not add a separate
 * VertexKinematicConstraint!
 * Example: Vertex fit with collinear tracks..
 */

class KinematicConstrainedVertexFitter{

public:

/**
 * Default constructor using LMSLinearizationPointFinder
 */
 KinematicConstrainedVertexFitter();
  
/**
 * Constructor with user-provided LinearizationPointFinder
 */  
 KinematicConstrainedVertexFitter(const LinearizationPointFinder& fnd);
  
 ~KinematicConstrainedVertexFitter();

  /**
   * Configuration through PSet: number of iterations(maxDistance) and
   * stopping condition (maxNbrOfIterations)
   */

 void setParameters(const edm::ParameterSet& pSet);

/**
 * Without additional constraint, this will perform a simple
 * vertex fit using LMS with Lagrange multipliers method.
 */  
 RefCountedKinematicTree fit(const std::vector<RefCountedKinematicParticle> &part) {
   return fit(part, 0, 0);
 }

/**
 * LMS with Lagrange multipliers fit of vertex constraint and user-specified constraint.
 */  
 RefCountedKinematicTree fit(const std::vector<RefCountedKinematicParticle> &part,
                            MultiTrackKinematicConstraint * cs) {
   return fit(part, cs, 0);
 };
    
/**
 * LMS with Lagrange multipliers fit of vertex constraint, user-specified constraint and user-specified starting point.
 */  
 RefCountedKinematicTree fit(const std::vector<RefCountedKinematicParticle> &part,
                           MultiTrackKinematicConstraint * cs,
                           GlobalPoint * pt);

//return the number of iterations
 int getNit() const;
//return the value of the constraint equation
 float getCSum() const;

private:

 void defaultParameters();

 float theMaxDelta; //maximum (delta parameter)^2/(sigma parameter)^2 per iteration for convergence
 int theMaxStep; 				       
 float theMaxReducedChiSq; //max of initial (after 2 iterations) chisq/dof value
 float theMinChiSqImprovement; //minimum required improvement in chisq to avoid fit termination for cases exceeding theMaxReducedChiSq
 LinearizationPointFinder * finder;				       
 KinematicConstrainedVertexUpdator * updator;
 VertexKinematicConstraint * vCons;
 ConstrainedTreeBuilder * tBuilder;
 int iterations;
 float csum;
};

#endif
