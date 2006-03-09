#ifndef KinematicConstrainedVertexFitter_H
#define KinematicConstrainedVertexFitter_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexUpdator.h"
#include "RecoVertex/KinematicFit/interface/VertexKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/ConstrainedTreeBuilder.h"

/**
 * Class fitting the veretx out
 * of set of tracks via usual LMS
 * with Lagrange multipliers.
 * Additional constraints can be applyed 
 * to the tracks during the vertex fit
 * (solves non-factorizabele cases)
 * Example: Vertex fit with collinear tracks..
 *
 * Kirill Prokofiev September 2003.
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
 * If multitrack constraint is empty,
 * the output will be just a simple
 * vertex fit performed by LMS with
 * Lagrange multipliers method.
 */  
 RefCountedKinematicTree fit(vector<RefCountedKinematicParticle> part, 
                           MultiTrackKinematicConstraint * cs) const;

private:

//method to deal with simple configurable parameters:
//number of iterations and stopping condition
 void readParameters();
 float theMaxDiff;
 int theMaxStep; 				       
 LinearizationPointFinder * finder;				       
 KinematicConstrainedVertexUpdator * updator;
 VertexKinematicConstraint * vCons;
 ConstrainedTreeBuilder * tBuilder;
};

#endif
