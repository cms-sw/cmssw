#ifndef KinematicParticleVertexFitter_H
#define KinematicParticleVertexFitter_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/KinematicFit/interface/InputSort.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
 * Class creating a kinematic particle out
 * of set of daughter particles. Daughter
 * particles are supposed to come from a 
 * common vertex. Arbitrary VerexFitter
 * can be used to fit the common vertex 
 * and refit the daughter particles with the
 * knowledge of vertex. The Kinematic Vertex is also
 * created and the resulting KinematicParticle points
 * on it.
 * 
 * Kirill Prokofiev, December 2002
 */

class KinematicParticleVertexFitter
{

public:
 
/**
 * Constructor with LMSLinearizationPointFinder used as default.
 *
 */ 
 KinematicParticleVertexFitter();
 
 KinematicParticleVertexFitter(const edm::ParameterSet &pSet);
 
/*
 * Constructor with the LinearizationPointFinder
 * Linearization point finder should have an 
 * ability to find point out of set of FreeTrajectoryStates
 * LMSLinearizationPointFinder is used as default.
 */
//  KinematicParticleVertexFitter(const LinearizationPointFinder& finder);
 
 ~KinematicParticleVertexFitter();
 
/**
 * Fit method taking set of particles, fitting them to the
 * common vertex and creating tree out of them.
 * Input particles can belong to kinmaticTrees.
 * In such a case it should be TOP particle of
 * corresponding tree.
 */  
 RefCountedKinematicTree  fit(const std::vector<RefCountedKinematicParticle> & particles) const;
 
private:

  edm::ParameterSet defaultParameters() const;
  void setup(const edm::ParameterSet &pSet);

  VertexFitter<6> * fitter;
  LinearizationPointFinder * pointFinder; 
  VertexTrackFactory<6> * vFactory;
};
#endif
