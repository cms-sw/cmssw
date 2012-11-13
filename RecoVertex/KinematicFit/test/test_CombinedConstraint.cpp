#include "RecoVertex/KinematicFit/interface/CombinedKinematicConstraintT.h"

#include "RecoVertex/KinematicFit/interface/VertexKinematicConstraintT.h"
#include "RecoVertex/KinematicFit/interface/ColinearityKinematicConstraintT.h"

#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitterT.h"


#include<iostream>

int main() {


  typedef ColinearityKinematicConstraintT<colinearityKinematic::PhiTheta> ColinearityConstraint;
  { ColinearityConstraint cc; std::cout << " cc " << cc.numberOfEquations() << std::endl;}

 
  typedef CombinedKinematicConstraintT<std::tuple<ColinearityConstraint,VertexKinematicConstraintT>, 2> CKC;
  CKC ckc(std::make_tuple(ColinearityConstraint(),VertexKinematicConstraintT()));

  std::cout << CKC::nTrk << " " << CKC::nDim 
            << " " << ckc.numberOfEquations() 
            << std::endl;

  std::vector<KinematicState> states(2);
  const GlobalPoint point;
  const GlobalVector mf;
  ckc.init(states,point,mf);

  CKC::valueType v = ckc.value();
  CKC::parametersDerivativeType pad =  ckc.parametersDerivative();
  CKC::positionDerivativeType pod = ckc.positionDerivative();

  std::cout << v(0) << " " << pad(0,0) << " " << pod(0,0) << std::endl;

  KinematicConstrainedVertexFitterT<CKC::nTrk,CKC::nDim> kinefit(0);


  return 0;
}


