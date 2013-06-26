#ifndef InputSort_H
#define InputSort_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"

/**
 * Helper class checking the
 * input of Kinematic Vertex Fitters
 * If some of particles provided have 
 * trees after them, makes sure that
 * only top tree particles are used in the fit.
 */

class InputSort{

public:

 InputSort(){}
 ~InputSort(){}
 
 std::pair<std::vector<RefCountedKinematicParticle>, std::vector<FreeTrajectoryState> > 
                          sort(const std::vector<RefCountedKinematicParticle> &particles) const;

 std::vector<RefCountedKinematicParticle> sort(const std::vector<RefCountedKinematicTree> &trees) const;
 
private:

};
#endif
