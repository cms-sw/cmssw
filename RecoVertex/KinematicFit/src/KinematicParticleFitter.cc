#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/LagrangeParentParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/LagrangeChildUpdator.h"

KinematicParticleFitter::KinematicParticleFitter()
{
 parentFitter = new LagrangeParentParticleFitter();
 cUpdator = new LagrangeChildUpdator();
}

KinematicParticleFitter::KinematicParticleFitter(const ParentParticleFitter& fitter, const ChildUpdator& updator)
{
 parentFitter = fitter.clone();
 cUpdator = updator.clone();
}
  
KinematicParticleFitter::~KinematicParticleFitter()
{
 delete parentFitter;
 delete cUpdator;
}

std::vector<RefCountedKinematicTree> KinematicParticleFitter::fit(KinematicConstraint * cs , 
                                            const std::vector<RefCountedKinematicTree> &trees)const
{
 std::vector<RefCountedKinematicTree> tr = parentFitter->fit(trees,cs);
 // In case of problem in fit:
 if (tr.empty()) return tr;
 tr = cUpdator->update(tr);
 return tr;
} 

RefCountedKinematicTree KinematicParticleFitter::fit(KinematicConstraint * cs , 
                                            RefCountedKinematicTree tree)const
{
 std::vector<RefCountedKinematicTree> trees;
 trees.push_back(tree);
 std::vector<RefCountedKinematicTree> tr = parentFitter->fit(trees,cs);
 // In case of problem in fit:
 if (tr.empty()) return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
 tr = cUpdator->update(tr);
 return tr.front();
}
