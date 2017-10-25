#include "RecoVertex/KinematicFit/interface/ConstrainedTreeBuilderT.h"
#include "DataFormats/CLHEP/interface/Migration.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"




RefCountedKinematicTree ConstrainedTreeBuilderT::buildRealTree(const RefCountedKinematicParticle virtualParticle,
							   const RefCountedKinematicVertex vtx, const std::vector<RefCountedKinematicParticle> & particles) const
{

//making a resulting tree:
 RefCountedKinematicTree resTree = ReferenceCountingPointer<KinematicTree>(new KinematicTree());

//fake production vertex:
 RefCountedKinematicVertex fVertex = vFactory.vertex();
 resTree->addParticle(fVertex, vtx, virtualParticle);

//adding final state
 for(std::vector<RefCountedKinematicParticle>::const_iterator il = particles.begin(); il != particles.end(); il++)
 {
  if((*il)->previousParticle()->correspondingTree() != nullptr)
  {
   KinematicTree * tree = (*il)->previousParticle()->correspondingTree();
   tree->movePointerToTheTop();
   tree->replaceCurrentParticle(*il);
   RefCountedKinematicVertex cdVertex = resTree->currentDecayVertex();
   resTree->addTree(cdVertex, tree);
  }else{
   RefCountedKinematicVertex ffVertex = vFactory.vertex();
   resTree->addParticle(vtx,ffVertex,*il);
  }
 }
 return resTree;
}

