#include "RecoVertex/KinematicFit/interface/FourMomentumKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


FourMomentumKinematicConstraint::FourMomentumKinematicConstraint(const AlgebraicVector& momentum,
                                                                 const AlgebraicVector& deviation)
{
 if((momentum.num_row() != 4)||(deviation.num_row() != 4)) 
  throw VertexException("FourMomentumKinematicConstraint::vector of wrong size passed as 4-Momentum or Deviations");
 mm = momentum; 
 AlgebraicVector deviation_l(7,0);
 deviation_l(6) = momentum(3);
 deviation_l(5) = momentum(2);
 deviation_l(4) = momentum(1);
  
 double mass_sq = momentum(4)*momentum(4) - momentum(3)*momentum(3)
                -momentum(2)*momentum(2) - momentum(1)*momentum(1); 

 if(mass_sq == 0.)throw VertexException("FourMomentumKinematicConstraint::the constraint vector passed corresponds to 0 mass");
//deviation for mass calculated from deviations
//of momentum
 deviation_l(7) = momentum(4)*momentum(4)*deviation(4)/mass_sq 
                + momentum(3)*momentum(3)*deviation(3)/mass_sq
		+ momentum(2)*momentum(2)*deviation(2)/mass_sq
		+ momentum(1)*momentum(1)*deviation(1)/mass_sq;
//mass sigma because of the energy 
 
 dd = deviation_l;
}

std::pair<AlgebraicVector,AlgebraicVector> FourMomentumKinematicConstraint::value(const AlgebraicVector& exPoint) const
{
 if(exPoint.num_row() ==0 ) throw VertexException("FourMomentumKinematicConstraint::value requested for zero Linearization point");

//security check for extended cartesian parametrization 
 int inSize = exPoint.num_row(); 
 if((inSize%7) !=0) throw VertexException("FourMomentumKinematicConstraint::linearization point has a wrong dimension");
 int nStates = inSize/7;
 if(nStates != 1) throw VertexException("FourMomentumKinematicConstraint::Multi particle refit is not supported in this version");
 AlgebraicVector pr = exPoint;
 AlgebraicVector vl(4,0);
 vl(1) += (pr(4) - mm(1));
 vl(2) += (pr(5) - mm(2));
 vl(3) += (pr(6) - mm(3));
 vl(7) += (sqrt(pr(4)*pr(4)+pr(5)*pr(5)+pr(6)*pr(6)+pr(7)*pr(7)) - mm(4));
 
 return std::pair<AlgebraicVector,AlgebraicVector>(vl,pr);
}
 
std::pair<AlgebraicMatrix, AlgebraicVector> FourMomentumKinematicConstraint::derivative(const AlgebraicVector& exPoint) const
{
 if(exPoint.num_row() ==0) throw VertexException("FourMomentumKinematicConstraint::value requested for zero Linearization point");

//security check for extended cartesian parametrization 
 int inSize = exPoint.num_row(); 
 if((inSize%7) !=0) throw VertexException("FourMomentumKinematicConstraint::linearization point has a wrong dimension");
 int nStates = inSize/7;
 if(nStates != 1) throw VertexException("FourMomentumKinematicConstraint::Multi particle refit is not supported in this version");
 AlgebraicVector pr = exPoint;
 AlgebraicMatrix dr(4,7,0);

 dr(1,4) = 1.;
 dr(2,5) = 1.;
 dr(3,6) = 1.;
 dr(4,7) = pr(7)/ sqrt(pr(4)*pr(4)+pr(5)*pr(5)+pr(6)*pr(6)+pr(7)*pr(7));
  
 return std::pair<AlgebraicMatrix,AlgebraicVector>(dr,pr);
}

std::pair<AlgebraicVector, AlgebraicVector> FourMomentumKinematicConstraint::value(const std::vector<RefCountedKinematicParticle> &par) const
{
 int nStates = par.size();
 if(nStates == 0) throw VertexException("FourMomentumKinematicConstraint::Empty vector of particles passed");
 if(nStates != 1) throw VertexException("FourMomentumKinematicConstraint::Multi particle refit is not supported in this version");
 AlgebraicVector pr = asHepVector<7>(par.front()->currentState().kinematicParameters().vector());
 AlgebraicVector vl(4,0);
 
 vl(1) = pr(4) - mm(1);
 vl(2) = pr(5) - mm(2);
 vl(3) = pr(6) - mm(3);
 vl(7) = sqrt(pr(4)*pr(4)+pr(5)*pr(5)+pr(6)*pr(6)+pr(7)*pr(7)) - mm(4);
 
 return std::pair<AlgebraicVector,AlgebraicVector>(vl,pr);
}

std::pair<AlgebraicMatrix, AlgebraicVector> FourMomentumKinematicConstraint::derivative(const std::vector<RefCountedKinematicParticle> &par) const
{
 int nStates = par.size();
 if(nStates == 0) throw VertexException("FourMomentumKinematicConstraint::Empty vector of particles passed");
 if(nStates != 1) throw VertexException("FourMomentumKinematicConstraint::Multi particle refit is not supported in this version");
 AlgebraicMatrix dr(4,7,0);
 
 AlgebraicVector pr = asHepVector<7>(par.front()->currentState().kinematicParameters().vector());
 dr(1,4) = 1.;
 dr(2,5) = 1.;
 dr(3,6) = 1.;
 dr(4,7) = - pr(7)/sqrt(pr(4)*pr(4)+pr(5)*pr(5)+pr(6)*pr(6)+pr(7)*pr(7));
 
 return std::pair<AlgebraicMatrix,AlgebraicVector>(dr,pr);
}

AlgebraicVector FourMomentumKinematicConstraint::deviations(int nStates) const
{
 if(nStates == 0) throw VertexException("FourMomentumKinematicConstraint::Empty vector of particles passed");
 if(nStates != 1) throw VertexException("FourMomentumKinematicConstraint::Multi particle refit is not supported in this version");
 AlgebraicVector res = dd; 
 return res;
}

int FourMomentumKinematicConstraint::numberOfEquations() const
{return 4;}

