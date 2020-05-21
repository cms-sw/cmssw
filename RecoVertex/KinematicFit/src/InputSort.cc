#include "RecoVertex/KinematicFit/interface/InputSort.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicTree.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

std::pair<std::vector<RefCountedKinematicParticle>, std::vector<FreeTrajectoryState> > InputSort::sort(
    const std::vector<RefCountedKinematicParticle> &particles) const {
  if (particles.empty())
    throw VertexException("Sorting particles for vertex fitter::number of particles = 0");
  std::vector<RefCountedKinematicParticle> sortedParticles;
  std::vector<FreeTrajectoryState> sortedStates;

  //checking that only top particles of the tree are passed by user
  //correcting them for the top ones otherwise
  for (const auto &particle : particles) {
    if (particle->correspondingTree() != nullptr) {
      sortedParticles.push_back(particle->correspondingTree()->topParticle());
      sortedStates.push_back(particle->correspondingTree()->topParticle()->currentState().freeTrajectoryState());
    } else {
      sortedParticles.push_back(particle);
      sortedStates.push_back(particle->currentState().freeTrajectoryState());
    }
  }
  return std::pair<std::vector<RefCountedKinematicParticle>, std::vector<FreeTrajectoryState> >(sortedParticles,
                                                                                                sortedStates);
}

std::vector<RefCountedKinematicParticle> InputSort::sort(const std::vector<RefCountedKinematicTree> &trees) const {
  if (trees.empty())
    throw VertexException("Input Sort::Zero vector of trees passed");
  std::vector<RefCountedKinematicParticle> res;
  for (const auto &tree : trees) {
    tree->movePointerToTheTop();
    res.push_back(tree->currentParticle());
  }
  return res;
}
