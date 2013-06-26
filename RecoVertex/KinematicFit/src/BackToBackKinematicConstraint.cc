#include "RecoVertex/KinematicFit/interface/BackToBackKinematicConstraint.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "DataFormats/CLHEP/interface/Migration.h"

std::pair<AlgebraicVector, AlgebraicVector> BackToBackKinematicConstraint::value(const AlgebraicVector& exPoint) const
{
//security check for extended cartesian parametrization 
 int inSize = exPoint.num_row(); 
 if(inSize != 14) throw VertexException("BackToBackKinematicConstraint::linearization point has a wrong dimension");
 AlgebraicVector pr = exPoint;
 
//number of states should be always equal to 2 for such a constraint
 AlgebraicVector vl(3,0);
 vl(1) = pr(4)+pr(11);
 vl(2) = pr(5)+pr(12);
 vl(3) = pr(6)+pr(13);
 return std::pair<AlgebraicVector, AlgebraicVector>(vl,pr); 
}

std::pair<AlgebraicMatrix, AlgebraicVector> BackToBackKinematicConstraint::derivative(const AlgebraicVector& exPoint) const
{
//security check for extended cartesian parametrization 
 int inSize = exPoint.num_row(); 
 if(inSize != 14) throw VertexException("BackToBackKinematicConstraint::linearization point has a wrong dimension");
 AlgebraicVector pr = exPoint;
 
//number of states should be always equal to 2 for such a constraint 
 AlgebraicMatrix dr(3,14,0);
 dr(1,4) = 1.;
 dr(1,11) = 1.;
 dr(2,5) = 1;
 dr(2,12) = 1;
 dr(3,6) = 1;
 dr(3,13) = 1;
 return std::pair<AlgebraicMatrix, AlgebraicVector>(dr,pr); 
}

std::pair<AlgebraicVector, AlgebraicVector> BackToBackKinematicConstraint::value(const std::vector<RefCountedKinematicParticle> &par) const
{
 int nStates = par.size();
 if(nStates != 2) throw VertexException("BackToBackKinematicConstraint::number of tracks is not equal to 2");
 AlgebraicVector point(14,0);
 int co = 0;
 for(std::vector<RefCountedKinematicParticle>::const_iterator i = par.begin(); i!=par.end(); i++)
 {
  AlgebraicVector7 cPar = (*i)->currentState().kinematicParameters().vector();
  for(int j = 1; j<8; j++){point((co-1)*7+j) = cPar(j-1);}
  co++;
 }
 AlgebraicVector vl(3,0);
 AlgebraicVector st1 = asHepVector<7>(par[0]->currentState().kinematicParameters().vector());
 AlgebraicVector st2 = asHepVector<7>(par[1]->currentState().kinematicParameters().vector());
 vl(1) = st1(4)+st2(4);
 vl(2) = st1(5)+st2(5);
 vl(3) = st1(6)+st2(6);
 
 return std::pair<AlgebraicVector, AlgebraicVector>(vl,point); 
}

std::pair<AlgebraicMatrix, AlgebraicVector> BackToBackKinematicConstraint::derivative(const std::vector<RefCountedKinematicParticle> &par) const
{
 int nStates = par.size();
 if(nStates != 2) throw VertexException("BackToBackKinematicConstraint::number of tracks is not equal to 2"); 
 AlgebraicVector point(14,0);
 int co = 0;
 for(std::vector<RefCountedKinematicParticle>::const_iterator i = par.begin(); i!=par.end(); i++)
 {
  AlgebraicVector7 cPar = (*i)->currentState().kinematicParameters().vector();
  for(int j = 1; j<8; j++){point((co-1)*7+j) = cPar(j-1);}
  co++;
 }
 AlgebraicMatrix dr(3,14,0);

 return std::pair<AlgebraicMatrix, AlgebraicVector>(dr,point);
}

AlgebraicVector BackToBackKinematicConstraint::deviations(int nStates) const
{
 AlgebraicVector dd(7*nStates,0);
 return dd;
}

int BackToBackKinematicConstraint::numberOfEquations() const
{return 3;}

KinematicConstraint * BackToBackKinematicConstraint::clone() const
{return new BackToBackKinematicConstraint(*this);}
