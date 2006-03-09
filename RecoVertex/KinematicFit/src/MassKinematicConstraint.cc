#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

MassKinematicConstraint::MassKinematicConstraint(const ParticleMass& m, const float sigma)
{
 mass = m;
 AlgebraicVector deviation_l(7,0);
 deviation_l(7) = sigma * sigma; 
 dd = deviation_l;
}

pair<AlgebraicVector, AlgebraicVector> MassKinematicConstraint::value(const AlgebraicVector& exPoint) const
{

//we have only one equation and only one track, means the constraint value 
//for track parameters is just a single number
 if(exPoint.num_row() ==0 ) throw VertexException("MomentumKinematicConstraint::value requested for zero Linearization point");

//security check for extended cartesian parametrization 
 int inSize = exPoint.num_row(); 
 if((inSize%7) !=0) throw VertexException("MomentumKinematicConstraint::linearization point has a wrong dimension");
 int nStates = inSize/7;
 if(nStates !=1) throw VertexException("MassKinematicConstraint::multiple state refit is not supported in this version");
 AlgebraicVector vl(1,0);
 AlgebraicVector point = exPoint;
 vl(1) = point(7) - mass; 
 return pair<AlgebraicVector,AlgebraicVector>(vl,point);
}

pair<AlgebraicMatrix, AlgebraicVector> MassKinematicConstraint::derivative(const AlgebraicVector& exPoint) const
{
 if(exPoint.num_row() == 0) throw VertexException("MomentumKinematicConstraint::deriavtive requested for zero Linearization point");

//security check for extended cartesian parametrization 
 int inSize = exPoint.num_row(); 
 if((inSize%7) !=0) throw VertexException("MomentumKinematicConstraint::linearization point has a wrong dimension");
 int nStates = inSize/7;
 if(nStates !=1) throw VertexException("MassKinematicConstraint::multiple state refit is not supported in this version");
 AlgebraicMatrix dr(1,7,0);
 dr(1,7) = 1; 
 AlgebraicVector point = exPoint;
 return pair<AlgebraicMatrix,AlgebraicVector>(dr,point); 
}

pair<AlgebraicVector, AlgebraicVector> MassKinematicConstraint::value(const vector<RefCountedKinematicParticle> par) const
{
 int nStates = par.size();
 if(nStates == 0) throw VertexException("MassKinematicConstraint::empty vector of particles passed");
 if(nStates !=1) throw VertexException("MassKinematicConstraint::multiple state refit is not supported in this version");
 
 AlgebraicVector point = par.front()->currentState().kinematicParameters().vector();  
 AlgebraicVector vl(1,0);
 vl(1) = point(7) - mass;
 return pair<AlgebraicVector,AlgebraicVector>(vl,point);
} 

pair<AlgebraicMatrix, AlgebraicVector> MassKinematicConstraint::derivative(const vector<RefCountedKinematicParticle> par) const
{
 int nStates = par.size();
 if(nStates == 0) throw VertexException("MassKinematicConstraint::empty vector of particles passed");
 if(nStates !=1) throw VertexException("MassKinematicConstraint::multiple state refit is not supported in this version");

 AlgebraicVector point = par.front()->currentState().kinematicParameters().vector();
 AlgebraicMatrix dr(1,7,0);
 dr(1,7) = 1;
 return pair<AlgebraicMatrix,AlgebraicVector>(dr,point);
}

AlgebraicVector MassKinematicConstraint::deviations(int nStates) const
{
 if(nStates == 0) throw VertexException("MassKinematicConstraint::empty vector of particles passed");
 if(nStates !=1) throw VertexException("MassKinematicConstraint::multiple state refit is not supported in this version");
 AlgebraicVector res = dd;
 return res;
}

int MassKinematicConstraint::numberOfEquations() const 
{return 1;}







