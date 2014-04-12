#ifndef KinematicParticleFitter_H
#define KinematicParticleFitter_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFit/interface/ParentParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/ChildUpdator.h"

/**
 * Class making kinematic fit of the particle inside the 
 * KinematicTree. The daughter states of the tree get 
 * automathically refitted according to the changes 
 * done to mother state. Mechanism is split in 2 parts:
 * ParentParticleFitter to fit the mother particle and 
 * ChildUpdator to update the states of daughter particles.
 * Child updator is currently not implemented.
 * Fitter is designed to use any user provided algorithm
 * for state refit. 
 */

class KinematicParticleFitter
{
public:

/**
 * Default constructor using LMS with Lagrange
 * multipliers for particle refit.
 */
  KinematicParticleFitter();

/**
 * Constructor allowing use of any
 * fitter-updator pair implemented
 */  
  KinematicParticleFitter(const ParentParticleFitter& fitter, const ChildUpdator& updator);
  
  ~KinematicParticleFitter();

/**
 * Method applying the constraint to
 * the _TOP_ particle inside the
 * Kinematic Tree. Tree containing the
 * refitted state is returned. The 
 * initial state of the particle and
 * constraint applyed are stored in
 * particle's corresponding data memebers
 * In case of failure, an empty vector is returned.
 */ 

 std::vector<RefCountedKinematicTree> fit(KinematicConstraint * cs , 
                   const std::vector<RefCountedKinematicTree> & trees)const;  


/**
 * Method refitting a top particle of the single tree.
 * for backup compatibility and constraints not allowing
 * multiple track refits.
 * In case of failure, an invalid tree is returned.
 */
 RefCountedKinematicTree fit(KinematicConstraint * cs , 
                    RefCountedKinematicTree tree)const;
private:
 
 ParentParticleFitter * parentFitter;
 ChildUpdator * cUpdator;
};

#endif
