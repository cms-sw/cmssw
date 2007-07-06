#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/InputSort.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"
// #include "Utilities/UI/interface/SimpleConfigurable.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"



KinematicConstrainedVertexFitter::KinematicConstrainedVertexFitter()
{
 finder = new DefaultLinearizationPointFinder();
 vCons = new VertexKinematicConstraint();
 updator = new KinematicConstrainedVertexUpdator();
 tBuilder = new ConstrainedTreeBuilder;
 readParameters();
}

KinematicConstrainedVertexFitter:: KinematicConstrainedVertexFitter(const LinearizationPointFinder& fnd)
{
 
 finder = fnd.clone();
 vCons = new VertexKinematicConstraint();
 updator = new KinematicConstrainedVertexUpdator();
 tBuilder = new ConstrainedTreeBuilder;
 readParameters();
}

KinematicConstrainedVertexFitter::~KinematicConstrainedVertexFitter()
{
 delete finder;
 delete vCons;
 delete updator;
 delete tBuilder;
}

void KinematicConstrainedVertexFitter::readParameters()
{
//FIXME
//  static SimpleConfigurable<double>
//    maxEquationValueConfigurable(0.01,"KinematicConstrainedVertexFitterVertexFitter:maximumValue");
//  theMaxDiff = maxEquationValueConfigurable.value();
// 
//  static SimpleConfigurable<int>
//    maxStepConfigurable(10,"KinematicConstrainedVertexFitter:maximumNumberOfIterations");
//  theMaxStep = maxStepConfigurable.value();
  theMaxDiff = 0.01;
  theMaxStep = 10;
}

RefCountedKinematicTree KinematicConstrainedVertexFitter::fit(vector<RefCountedKinematicParticle> part, 
                                                             MultiTrackKinematicConstraint * cs)const
{
 if(part.size()<2) throw VertexException("KinematicConstrainedVertexFitter::input states are less than 2");
  
//sorting out the input particles
 InputSort iSort;
 pair<vector<RefCountedKinematicParticle>, vector<FreeTrajectoryState> > input = iSort.sort(part);
 const vector<RefCountedKinematicParticle> & prt  = input.first;
 const vector<FreeTrajectoryState> & fStates = input.second;

// linearization point: 
 GlobalPoint linPoint  = finder->getLinearizationPoint(fStates);

//initial parameters:  
 int vSize = prt.size();
 AlgebraicVector inPar(3 + 7*vSize,0);

//final parameters 
 AlgebraicVector finPar(3 + 7*vSize,0);

//initial covariance 
 AlgebraicMatrix inCov(3 + 7*vSize,3 + 7*vSize,0);

//making initial vector of parameters and initial particle-related covariance
 int nSt = 0;   
 vector<KinematicState> inStates;
 for(vector<RefCountedKinematicParticle>::const_iterator i = prt.begin(); i!=prt.end(); i++)
 {
  AlgebraicVector prPar = (*i)->stateAtPoint(linPoint).kinematicParameters().vector();
  for(int j = 1; j<8; j++){inPar(3 + 7*nSt + j) = prPar(j);}
  AlgebraicSymMatrix l_cov  = (*i)->stateAtPoint(linPoint).kinematicParametersError().matrix();
  inCov.sub(4 + 7*nSt,4 + 7*nSt ,l_cov); 
  inStates.push_back((*i)->stateAtPoint(linPoint));
  ++nSt;
 }

//initial vertex error matrix components (huge error method)
//and vertex related initial vector components
 double in_er = 100.;
 inCov(1,1) = in_er;
 inCov(2,2) = in_er;
 inCov(3,3) = in_er;
 
 inPar(1) = linPoint.x();
 inPar(2) = linPoint.y();
 inPar(3) = linPoint.z(); 
 
//constraint equations value and number of iterations 
 double eq;   
 int nit = 0; 
 
 vector<KinematicState> lStates = inStates;
 GlobalPoint lPoint  = linPoint;
 RefCountedKinematicVertex rVtx;
 AlgebraicMatrix refCCov;

//iterarions over the updator: each time updated parameters 
//are taken as new linearization point 
 do{ 
  eq = 0.;
  pair< pair< vector<KinematicState>, AlgebraicMatrix >,RefCountedKinematicVertex> lRes = 
                                      updator->update(inPar,inCov,lStates,lPoint,cs);
  lStates = lRes.first.first;
  rVtx = lRes.second;
  lPoint = rVtx->position(); 
  AlgebraicVector vValue = vCons->value(lStates, lPoint);
  for(int i = 1; i<vValue.num_row();++i)
  {eq += abs(vValue(i));}
  if(cs !=0)
  {
   AlgebraicVector cVal = cs->value(lStates, lPoint);
   for(int i = 1; i<cVal.num_row();++i)
   {eq += abs(cVal(i));}
  }
  refCCov = lRes.first.second;
  nit++;
 }while(nit<theMaxStep && eq>theMaxDiff);

// cout<<"number of relinearizations "<<nit<<endl;
// cout<<"value obtained: "<<eq<<endl;

//making refitted particles out of refitted states.
//none of the operations above violates the order of particles 
 if(prt.size() != lStates.size()) throw VertexException("KinematicConstrainedVertexFitter::updator failure");
 vector<RefCountedKinematicParticle>::const_iterator i;
 vector<KinematicState>::const_iterator j;
 vector<RefCountedKinematicParticle> rParticles;
 for(i = prt.begin(),j = lStates.begin(); i != prt.end(),j != lStates.end(); ++i,++j)
 {
  rParticles.push_back((*i)->refittedParticle((*j),rVtx->chiSquared(),rVtx->degreesOfFreedom()));
 }   
 
// cout<<"Constrained Vertex Fitter covariance: "<<refCCov <<endl;
 return  tBuilder->buildTree(rParticles,rVtx,refCCov); 
}




