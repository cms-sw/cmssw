#include "RecoVertex/KinematicFit/interface/MomentumKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


MomentumKinematicConstraint::MomentumKinematicConstraint(const AlgebraicVector& momentum,
                                                         const AlgebraicVector& dev)
{
 if((momentum.num_row() != 3) || (dev.num_row() != 3))
     throw VertexException("MomentumKinemticConstraint::Momentum or Deviation vector passed is not 3-dimensional");
 mm = momentum;
 AlgebraicVector dev_l(7,0);
 dev_l(4) = dev(1) * dev(1);
 dev_l(5) = dev(2) * dev(2);
 dev_l(6) = dev(3) * dev(3);
 dd = dev_l;
}

std::pair<AlgebraicVector,AlgebraicVector> MomentumKinematicConstraint::value(const AlgebraicVector& exPoint) const
{
 if(exPoint.num_row() ==0 ) throw VertexException("MomentumKinematicConstraint::value requested for zero Linearization point");

//security check for extended cartesian parametrization 
 int inSize = exPoint.num_row(); 
 if((inSize%7) !=0) throw VertexException("MomentumKinematicConstraint::linearization point has a wrong dimension");
 int nStates = inSize/7;
 if(nStates != 1) throw VertexException("MomentumKinematicConstraint::Multistate refit is not foreseen for this constraint"); 
 AlgebraicVector pr = exPoint;
 AlgebraicVector vl(3,0);
 vl(1) = pr(4) - mm(1);
 vl(2) = pr(5) - mm(2);
 vl(3) = pr(6) - mm(3);
 return std::pair<AlgebraicVector,AlgebraicVector>(vl,pr); 
}

std::pair<AlgebraicMatrix, AlgebraicVector> MomentumKinematicConstraint::derivative(const AlgebraicVector& exPoint) const
{
 if(exPoint.num_row() ==0 ) throw VertexException("MomentumKinematicConstraint::derivative requested for zero Linearization point");

//security check for extended cartesian parametrization 
 int inSize = exPoint.num_row(); 
 if((inSize%7) !=0) throw VertexException("MomentumKinematicConstraint::linearization point has a wrong dimension");
 int nStates = inSize/7;
 if(nStates != 1) throw VertexException("MomentumKinematicConstraint::Multistate refit is not foreseen for this constraint"); 

 AlgebraicVector pr = exPoint;
 AlgebraicMatrix dr(3,7,0);
 dr(1,4) = 1.;
 dr(2,5) = 1.;
 dr(3,6) = 1.;
 return std::pair<AlgebraicMatrix,AlgebraicVector>(dr,pr);
}

std::pair<AlgebraicVector, AlgebraicVector> MomentumKinematicConstraint::value(const std::vector<RefCountedKinematicParticle> &par) const
{
 int nStates = par.size(); 
 if(nStates == 0) throw VertexException("MomentumKinematicConstraint::Empty vector of particles passed");
 if(nStates != 1) throw VertexException("MomentumKinematicConstraint::Multistate refit is not foreseen for this constraint");  
 AlgebraicVector point = asHepVector<7>(par.front()->currentState().kinematicParameters().vector());
 AlgebraicVector vl(3,0);
 vl(1) = point(4) - mm(1);
 vl(2) = point(5) - mm(2);
 vl(3) = point(6) - mm(3);
 return std::pair<AlgebraicVector,AlgebraicVector>(vl,point);
}

std::pair<AlgebraicMatrix, AlgebraicVector> MomentumKinematicConstraint::derivative(const std::vector<RefCountedKinematicParticle> &par) const
{
 int nStates = par.size();
 if(nStates == 0) throw VertexException("MomentumKinematicConstraint::Empty vector of particles passed");
 if(nStates != 1) throw VertexException("MomentumKinematicConstraint::Multistate refit is not foreseen for this constraint");
 AlgebraicVector point = asHepVector<7>(par.front()->currentState().kinematicParameters().vector());
 AlgebraicMatrix dr(3,7,0);
 dr(1,4) = 1.;
 dr(2,5) = 1.;
 dr(3,6) = 1.;
 return std::pair<AlgebraicMatrix,AlgebraicVector>(dr,point);
}

AlgebraicVector MomentumKinematicConstraint::deviations(int nStates) const
{
 if(nStates == 0) throw VertexException("MomentumKinematicConstraint::Empty vector of particles passed");
 if(nStates != 1) throw VertexException("MomentumKinematicConstraint::Multistate refit is not foreseen for this constraint");
 AlgebraicVector res = dd;
 return res;
}

int MomentumKinematicConstraint::numberOfEquations() const
{return 3;}

