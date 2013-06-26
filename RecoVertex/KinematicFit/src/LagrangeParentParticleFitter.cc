#include "RecoVertex/KinematicFit/interface/LagrangeParentParticleFitter.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"
#include "RecoVertex/KinematicFit/interface/InputSort.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

LagrangeParentParticleFitter::LagrangeParentParticleFitter()
{defaultParameters();}

/*
RefCountedKinematicTree LagrangeParentParticleFitter::fit(RefCountedKinematicTree tree, KinematicConstraint * cs) const
{
//fitting top paticle only
 tree->movePointerToTheTop();
 RefCountedKinematicParticle particle = tree->currentParticle(); 
 RefCountedKinematicVertex inVertex = tree->currentDecayVertex();

//parameters and covariance matrix of the particle to fit 
 AlgebraicVector par = particle->currentState().kinematicParameters().vector();
 AlgebraicSymMatrix cov = particle->currentState().kinematicParametersError().matrix();

//chi2 and ndf
 float chi = 0.;
 float ndf = 0.;
 
//constraint
//here it is defined at 7-point
//taken just at current state parameters
 AlgebraicVector vl;
 AlgebraicMatrix dr;
 AlgebraicVector dev;
 
//initial expansion point 
 AlgebraicVector exPoint = particle->currentState().kinematicParameters().vector();
 
//refitted parameters and covariance matrix:
//simple and symmetric 
 AlgebraicVector refPar;
 AlgebraicSymMatrix refCovS; 
 
 int nstep = 0;
 double df = 0.;
 do{ 
   chi = particle->chiSquared();
   ndf = particle->degreesOfFreedom();
   df =  0.;
   
//derivative and value at expansion point  
   vl = cs->value(exPoint).first;
   dr = cs->derivative(exPoint).first;
   dev = cs->deviations();  
   
//residual between expansion and current parameters
//0 at the first step   
   AlgebraicVector delta_alpha = par - exPoint;  
  
//parameters needed for refit
// v_d = (D * V_alpha & * D.T)^(-1)   
   AlgebraicMatrix drt = dr.T();
   AlgebraicMatrix v_d = dr * cov * drt;
   int ifail = 0;
   v_d.invert(ifail);
   if(ifail != 0) throw VertexException("ParentParticleFitter::error inverting covariance matrix");
  
//lagrangian multipliers  
//lambda = V_d * (D * delta_alpha + d)
   AlgebraicVector lambda = v_d * (dr*delta_alpha + vl);
  
//refitted parameters
   refPar = par - (cov * drt * lambda);  
  
//refitted covariance: simple and SymMatrix  
   refCovS = cov;
   AlgebraicMatrix sPart = drt * v_d * dr;
   AlgebraicMatrix covF = cov * sPart * cov; 
   AlgebraicSymMatrix sCovF(7,0);
  
   for(int i = 1; i<8; i++)
   {
     for(int j = 1; j<8; j++)
     {if(i<=j) sCovF(i,j) = covF(i,j);}
   }
   refCovS -= sCovF; 
  
   for(int i = 1; i < 8; i++)
   {refCovS(i,i) += dev(i);}
  
//chiSquared  
   chi +=  (lambda.T() * (dr*delta_alpha + vl))(1);
   ndf +=  cs->numberOfEquations();
   
//new expansionPoint
   exPoint = refPar;
   AlgebraicVector vlp = cs->value(exPoint).first;
   for(int i = 1; i< vl.num_row();i++)
   {df += std::abs(vlp(i));}
   nstep++; 
  }while(df>theMaxDiff && nstep<theMaxStep);
  
//!!!!!!!!!!!!!!!here the math part is finished!!!!!!!!!!!!!!!!!!!!!! 
//creating an output KinematicParticle 
//and putting it on its place in the tree
  KinematicParameters param(refPar);
  KinematicParametersError er(refCovS);
  KinematicState kState(param,er,particle->initialState().particleCharge());
  RefCountedKinematicParticle refParticle  = particle->refittedParticle(kState,chi,ndf,cs->clone());							 
  tree->replaceCurrentParticle(refParticle);
  
//replacing the  vertex with its refitted version
  GlobalPoint nvPos(param.vector()(1), param.vector()(2), param.vector()(3)); 
  AlgebraicSymMatrix nvMatrix = er.matrix().sub(1,3);
  GlobalError nvError(asSMatrix<3>(nvMatrix));
  VertexState vState(nvPos, nvError, 1.0);
  KinematicVertexFactory vFactory;
  RefCountedKinematicVertex nVertex = vFactory.vertex(vState, inVertex,chi,ndf);
  tree->replaceCurrentVertex(nVertex);
   
  return tree;
}
*/

std::vector<RefCountedKinematicTree>  LagrangeParentParticleFitter::fit(const std::vector<RefCountedKinematicTree> &trees, 
                                                                           KinematicConstraint * cs)const					  
{
 
 InputSort iSort;
 std::vector<RefCountedKinematicParticle> prt = iSort.sort(trees);
 int nStates = prt.size(); 

//making full initial parameters and covariance
 AlgebraicVector part(7*nStates,0);
 AlgebraicSymMatrix cov(7*nStates,0);
 
 AlgebraicVector chi_in(nStates,0);
 AlgebraicVector ndf_in(nStates,0);
 int l_c=0;
 for(std::vector<RefCountedKinematicParticle>::const_iterator i = prt.begin(); i != prt.end(); i++)
 {
  AlgebraicVector7 lp = (*i)->currentState().kinematicParameters().vector();
  for(int j = 1; j != 8; j++){part(7*l_c + j) = lp(j-1);}
  AlgebraicSymMatrix lc= asHepMatrix<7>((*i)->currentState().kinematicParametersError().matrix());
  cov.sub(7*l_c+1,lc);
  chi_in(l_c+1) = (*i)->chiSquared();
  ndf_in(l_c+1) = (*i)->degreesOfFreedom();
  l_c++;
 }
//refitted parameters and covariance matrix:
//simple and symmetric 
 AlgebraicVector refPar;
 AlgebraicSymMatrix refCovS; 
 
//constraint values, derivatives and deviations:  
 AlgebraicVector vl;
 AlgebraicMatrix dr;
 AlgebraicVector dev;
 int nstep = 0;
 double df = 0.; 
 AlgebraicVector exPoint = part;
 
//this piece of code should later be refactored:
// The algorithm is the same as above, but the
// number of refitted particles is >1. Smart way of
// refactoring should be chosen for it.
 AlgebraicVector chi;
 AlgebraicVector ndf;
// cout<<"Starting the main loop"<<endl;
 do{
  df = 0.;
  chi = chi_in;
  ndf = ndf_in;
//  cout<<"Iterational linearization point: "<<exPoint<<endl;
  vl = cs->value(exPoint).first;
  dr = cs->derivative(exPoint).first;
  dev = cs->deviations(nStates);

//  cout<<"The value : "<<vl<<endl;
//  cout<<"The derivative: "<<dr<<endl;
//  cout<<"deviations: "<<dev<<endl;
//  cout<<"covariance "<<cov<<endl;

//residual between expansion and current parameters
//0 at the first step   
  AlgebraicVector delta_alpha = part - exPoint;  

//parameters needed for refit
// v_d = (D * V_alpha & * D.T)^(-1)   
   AlgebraicMatrix drt = dr.T();
   AlgebraicMatrix v_d = dr * cov * drt;
   int ifail = 0;
   v_d.invert(ifail);
   if(ifail != 0) {
     LogDebug("KinematicConstrainedVertexFitter")
	<< "Fit failed: unable to invert covariance matrix\n";
     return std::vector<RefCountedKinematicTree>();
   }
  
//lagrangian multipliers  
//lambda = V_d * (D * delta_alpha + d)
   AlgebraicVector lambda = v_d * (dr*delta_alpha + vl);
  
//refitted parameters
   refPar = part - (cov * drt * lambda);  
  
//refitted covariance: simple and SymMatrix  
   refCovS = cov;
   AlgebraicMatrix sPart = drt * v_d * dr;
   AlgebraicMatrix covF = cov * sPart * cov; 

//total covariance symmatrix    
  AlgebraicSymMatrix sCovF(nStates*7,0);
  for(int i = 1; i< nStates*7 +1; ++i)
  {
   for(int j = 1; j< nStates*7 +1; j++)
   {if(i<=j) sCovF(i,j) = covF(i,j);}
  }
  
  refCovS -= sCovF; 
  
//  cout<<"Fitter: refitted covariance "<<refCovS<<endl;
  for(int i = 1; i < nStates+1; i++)
  {for(int j = 1; j<8; j++){refCovS((i-1)+j,(i-1)+j) += dev(j);}} 
  
//chiSquared  
  for(int k =1; k<nStates+1; k++)
  {
   chi(k) +=  (lambda.T() * (dr*delta_alpha + vl))(1);
   ndf(k) +=  cs->numberOfEquations();
  }  
//new expansionPoint
  exPoint = refPar;
  AlgebraicVector vlp = cs->value(exPoint).first;
  for(int i = 1; i< vl.num_row();i++)
  {df += std::abs(vlp(i));}
  nstep++; 
 }while(df>theMaxDiff && nstep<theMaxStep);
//here math and iterative part is finished, starting an output production
//creating an output KinematicParticle and putting it on its place in the tree
 
//vector of refitted particles and trees 
 std::vector<RefCountedKinematicParticle> refPart;
 std::vector<RefCountedKinematicTree> refTrees = trees;
 
 int j=1;
 std::vector<RefCountedKinematicTree>::const_iterator tr = refTrees.begin();
 for(std::vector<RefCountedKinematicParticle>::const_iterator i = prt.begin(); i!= prt.end(); i++)
 {
  AlgebraicVector7 lRefPar;
  for(int k = 1; k<8 ; k++)
  {lRefPar(k-1) = refPar((j-1)*7+k);}
  AlgebraicSymMatrix77 lRefCovS = asSMatrix<7>(refCovS.sub((j-1)*7 +1,(j-1)*7+7));
  
//new refitted parameters and covariance  
  KinematicParameters param(lRefPar);
  KinematicParametersError er(lRefCovS); 
  KinematicState kState(param,er,(*i)->initialState().particleCharge(), (**i).magneticField());
  RefCountedKinematicParticle refParticle  = (*i)->refittedParticle(kState,chi(j),ndf(j),cs->clone());
  
//replacing particle itself  
  (*tr)->findParticle(*i);
  RefCountedKinematicVertex inVertex =  (*tr)->currentDecayVertex();
  (*tr)->replaceCurrentParticle(refParticle);
  
//replacing the  vertex with its refitted version
  GlobalPoint nvPos(param.position()); 
  AlgebraicSymMatrix nvMatrix = asHepMatrix<7>(er.matrix()).sub(1,3);
  GlobalError nvError(asSMatrix<3>(nvMatrix));
  VertexState vState(nvPos, nvError, 1.0);
  KinematicVertexFactory vFactory;
  RefCountedKinematicVertex nVertex = vFactory.vertex(vState,inVertex,chi(j),ndf(j));
  (*tr)->replaceCurrentVertex(nVertex);  
  tr++;
  j++;
 }
 return refTrees; 
}     

void LagrangeParentParticleFitter::setParameters(const edm::ParameterSet& pSet)
{
  theMaxDiff = pSet.getParameter<double>("maxDistance");
  theMaxStep = pSet.getParameter<int>("maxNbrOfIterations");;
}

void LagrangeParentParticleFitter::defaultParameters()
{
  theMaxDiff = 0.001;
  theMaxStep = 100;
}
