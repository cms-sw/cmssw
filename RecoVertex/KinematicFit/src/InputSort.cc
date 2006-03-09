#include "RecoVertex/KinematicFit/interface/InputSort.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicTree.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

pair<vector<RefCountedKinematicParticle>, vector<FreeTrajectoryState> > 
                    InputSort::sort(vector<RefCountedKinematicParticle> particles) const
{
 if(particles.size()==0) throw VertexException("Sorting particles for vertex fitter::number of particles = 0");
 vector<RefCountedKinematicParticle> sortedParticles;
 vector<FreeTrajectoryState> sortedStates;

//checking that only top particles of the tree are passed by user
//correcting them for the top ones otherwise 
 for(vector<RefCountedKinematicParticle>::const_iterator i = particles.begin(); i != particles.end(); i++)
 {
  if((*i)->correspondingTree() != 0)
  {
   sortedParticles.push_back((*i)->correspondingTree()->topParticle());
   sortedStates.push_back((*i)->correspondingTree()->topParticle()->currentState().freeTrajectoryState());
  }else{
   sortedParticles.push_back(*i);
   sortedStates.push_back((*i)->currentState().freeTrajectoryState());
  }
 }
 return pair<vector<RefCountedKinematicParticle>, vector<FreeTrajectoryState> >(sortedParticles, sortedStates);
}

vector<RefCountedKinematicParticle> InputSort::sort(vector<RefCountedKinematicTree> trees) const
{
 if(trees.size() ==0) throw VertexException("Input Sort::Zero vector of trees passed"); 
 vector<RefCountedKinematicParticle> res;
 for(vector<RefCountedKinematicTree>::const_iterator i = trees.begin(); i!=trees.end(); i++)
 {
  (*i)->movePointerToTheTop();
  res.push_back((*i)->currentParticle());
 }
 return res;
}
