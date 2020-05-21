#include "RecoVertex/KinematicFit/interface/ConstrainedTreeBuilderT.h"
#include "DataFormats/CLHEP/interface/Migration.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

RefCountedKinematicTree ConstrainedTreeBuilderT::buildRealTree(
    const RefCountedKinematicParticle virtualParticle,
    const RefCountedKinematicVertex vtx,
    const std::vector<RefCountedKinematicParticle>& particles) const {
  //making a resulting tree:
  RefCountedKinematicTree resTree = ReferenceCountingPointer<KinematicTree>(new KinematicTree());

  //fake production vertex:
  RefCountedKinematicVertex fVertex = vFactory.vertex();
  resTree->addParticle(fVertex, vtx, virtualParticle);

  //adding final state
  for (const auto& particle : particles) {
    if (particle->previousParticle()->correspondingTree() != nullptr) {
      KinematicTree* tree = particle->previousParticle()->correspondingTree();
      tree->movePointerToTheTop();
      tree->replaceCurrentParticle(particle);
      RefCountedKinematicVertex cdVertex = resTree->currentDecayVertex();
      resTree->addTree(cdVertex, tree);
    } else {
      RefCountedKinematicVertex ffVertex = vFactory.vertex();
      resTree->addParticle(vtx, ffVertex, particle);
    }
  }
  return resTree;
}
