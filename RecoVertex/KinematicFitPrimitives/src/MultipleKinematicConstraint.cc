#include "RecoVertex/KinematicFitPrimitives/interface/MultipleKinematicConstraint.h"

void MultipleKinematicConstraint::addConstraint(KinematicConstraint * newConst) const
{
 if(newConst == 0)throw VertexException("MultipleKinematicConstraint::zero constraint pointer passed");
 cts.push_back(newConst);
 em = false;
}
  
std::pair<AlgebraicVector,AlgebraicVector> MultipleKinematicConstraint::value(const AlgebraicVector& exPoint) const
{
 if(cts.size() == 0)throw VertexException("MultipleKinematicConstraint::value requested for empty constraint");
 if(exPoint.num_row() ==0 ) throw VertexException("MultipleKinematicConstraint::value requested for zero Linearization point");

//looking for total number of states, represented by this  point. 
//Since this version only works with extended  cartesian parametrization, 
//we check whether the dimensionof he point is n*7
 AlgebraicVector expansion = exPoint;
 int inSize = exPoint.num_row(); 
 if((inSize%7) !=0) throw VertexException("MultipleKinematicConstraint::linearization point has a wrong dimension");
 
 int total = 0;
 for(std::vector<KinematicConstraint *>::const_iterator i = cts.begin(); i != cts.end(); i++)
 {total += (*i)->numberOfEquations();}
 AlgebraicVector vl(total,0);
 
 int cr_size = 0;
 for(std::vector<KinematicConstraint *>::const_iterator i = cts.begin(); i != cts.end(); i++)
 {
  AlgebraicVector vlc = (*i)->value(expansion).first;
  int sz = vlc.num_row();
  for(int j = 1; j < sz+1; j++)
  {vl(cr_size+j) = vlc(j);}
  cr_size += sz;
 }
 return std::pair<AlgebraicVector, AlgebraicVector>(vl,expansion); 
}
 
std::pair<AlgebraicMatrix, AlgebraicVector> MultipleKinematicConstraint::derivative(const AlgebraicVector& exPoint) const
{
 if(cts.size() == 0) throw VertexException("MultipleKinematicConstraint::derivative requested for empty constraint");
 if(exPoint.num_row() ==0 ) throw VertexException("MultipleKinematicConstraint::value requested for zero Linearization point");

//security check for extended cartesian parametrization 
 AlgebraicVector expansion = exPoint;
 int inSize = exPoint.num_row(); 
 if((inSize%7) !=0) throw VertexException("MultipleKinematicConstraint::linearization point has a wrong dimension");


 AlgebraicVector par = exPoint;
 int total = 0;
 for(std::vector<KinematicConstraint *>::const_iterator i = cts.begin(); i != cts.end(); i++)
 {total += (*i)->numberOfEquations();}
 
//Full derivative matrix: (numberOfConstraints x 7*numberOfStates) 
 AlgebraicMatrix dr(total,inSize);
 
 int  cr_size = 0;
 for(std::vector<KinematicConstraint *>::const_iterator i = cts.begin(); i != cts.end(); i++)
 {
//matrix should be (nx7*NumberOfStates)
  AlgebraicMatrix lConst  = (*i)->derivative(par).first;
  dr.sub(cr_size+1,1,lConst);
  cr_size += (*i)->numberOfEquations();  
 }
 return std::pair<AlgebraicMatrix, AlgebraicVector>(dr,par);  
}

int MultipleKinematicConstraint::numberOfEquations() const
{
 int ne = 0;
 if(cts.size() == 0) throw VertexException("MultipleKinematicConstraint::number of equations requested for empty constraint");
 for(std::vector<KinematicConstraint *>::const_iterator i = cts.begin(); i != cts.end(); i++)
 {ne += (*i)->numberOfEquations();}
 return ne;
}

std::pair<AlgebraicVector, AlgebraicVector> MultipleKinematicConstraint::value(const std::vector<RefCountedKinematicParticle> &par) const
{
 if(cts.size() == 0) throw VertexException("MultipleKinematicConstraint::derivative requested for empty constraint");
 int nStates = par.size();
 AlgebraicVector param(7*nStates,0);
 int count = 1;
 for(std::vector<RefCountedKinematicParticle>::const_iterator i = par.begin(); i!=par.end(); i++)
 {
  for(int j = 1; j<8; j++){param((count -1)*7+j) = (*i)->currentState().kinematicParameters().vector()(j-1);}
  count++;
 } 

//looking for total number of equations
 int total = 0;
 for(std::vector<KinematicConstraint *>::const_iterator i = cts.begin(); i != cts.end(); i++)
 {total += (*i)->numberOfEquations();}
 AlgebraicVector vl(total,0);
 
 int cr_size = 0;
 for(std::vector<KinematicConstraint *>::const_iterator i = cts.begin(); i != cts.end(); i++)
 {
  AlgebraicVector vlc = (*i)->value(par).first;
  int sz = vlc.num_row();
  for(int j = 1; j <= sz; j++)
  {vl(cr_size+j) = vlc(j);}
  cr_size += sz;
 }
 return std::pair<AlgebraicVector, AlgebraicVector>(vl,param); 
}

std::pair<AlgebraicMatrix, AlgebraicVector> MultipleKinematicConstraint::derivative(const std::vector<RefCountedKinematicParticle> &par) const
{
 if(cts.size() == 0) throw VertexException("MultipleKinematicConstraint::derivative requested for empty constraint");
 int nStates = par.size();
 AlgebraicVector param(7*nStates,0);
 
 int count = 1;
 for(std::vector<RefCountedKinematicParticle>::const_iterator i = par.begin(); i!=par.end(); i++)
 {
  for(int j = 1; j<8; j++){param((count -1)*7+j) = (*i)->currentState().kinematicParameters().vector()(j-1);}
  count++;
 } 
 int total = 0;
 for(std::vector<KinematicConstraint *>::const_iterator i = cts.begin(); i != cts.end(); i++)
 {total += (*i)->numberOfEquations();}
 AlgebraicMatrix dr(total,7*nStates);
 
 int  cr_size = 0;
 for(std::vector<KinematicConstraint *>::const_iterator i = cts.begin(); i != cts.end(); i++)
 {

//matrix should be (TotalNumberOfEquations x 7* TotalNumberOfStates)
//Derivative matrix for given constraint
  AlgebraicMatrix lConst  = (*i)->derivative(param).first;
  
//putting it into the appropriate line  
  dr.sub(cr_size+1,1,lConst);  
  cr_size += (*i)->numberOfEquations(); 
 }
 return std::pair<AlgebraicMatrix, AlgebraicVector>(dr,param); 
}

AlgebraicVector MultipleKinematicConstraint::deviations(int nStates) const
{
 AlgebraicVector dev(nStates*7,0);
 if(cts.size() == 0) throw VertexException("MultipleKinematicConstraint::deviations requested for empty constraint");
 for(std::vector<KinematicConstraint *>::const_iterator i = cts.begin(); i != cts.end(); i++)
 {
  AlgebraicVector dev_loc =(*i)->deviations(nStates);
  for(int j = 1; j < nStates*7+1; j++){dev(j) = dev(j) + dev_loc(j);} 
 }
 return dev;
}

