#include "RecoVertex/KinematicFit/interface/CombinedKinematicConstraintT.h"

#include "RecoVertex/KinematicFit/interface/VertexKinematicConstraintT.h"
#include "RecoVertex/KinematicFit/interface/ColinearityKinematicConstraintT.h"

#include "RecoVertex/KinematicFit/interface/KinematicConstraintedVertexFitterT.h"


#include<iostream>

int main() {

  typedef ColinearityKinematicConstraintT<colinearityKinematic::PhiTheta> ColinearityConstraint;
 
  typedef CombinedKinematicConstraintT<std::tuple<ColinearityConstraint,VertexKinematicConstraintT>, 2> CKC;
  CKC ckc(std::make_tuple(ColinearityConstraint(),VertexKinematicConstraintT()));

  std::cout << CKC::nTrk << " " , CKC::nDim << " " << ckc.numberOfEquations() << std::endl;

  KinematicConstraintedVertexFitterT<CKC::nTrk,CKC::nDim> kinefit(0);


  return 0;
}


