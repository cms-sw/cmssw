#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/InputSort.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



KinematicConstrainedVertexFitter::KinematicConstrainedVertexFitter()
{
 finder = new DefaultLinearizationPointFinder();
 vCons = new VertexKinematicConstraint();
 updator = new KinematicConstrainedVertexUpdator();
 tBuilder = new ConstrainedTreeBuilder;
 defaultParameters();
 iterations = -1;
 csum = -1000.0;
}

KinematicConstrainedVertexFitter::KinematicConstrainedVertexFitter(const LinearizationPointFinder& fnd)
{
 finder = fnd.clone();
 vCons = new VertexKinematicConstraint();
 updator = new KinematicConstrainedVertexUpdator();
 tBuilder = new ConstrainedTreeBuilder;
 defaultParameters();
 iterations = -1;
 csum = -1000.0;
}

KinematicConstrainedVertexFitter::~KinematicConstrainedVertexFitter()
{
 delete finder;
 delete vCons;
 delete updator;
 delete tBuilder;
}

void KinematicConstrainedVertexFitter::setParameters(const edm::ParameterSet& pSet)
{
  theMaxDelta = pSet.getParameter<double>("maxDelta");
  theMaxStep = pSet.getParameter<int>("maxNbrOfIterations");
  theMaxReducedChiSq = pSet.getParameter<double>("maxReducedChiSq");
  theMinChiSqImprovement = pSet.getParameter<double>("minChiSqImprovement");
}

void KinematicConstrainedVertexFitter::defaultParameters()
{
  theMaxDelta = 0.01;
  theMaxStep = 1000;
  theMaxReducedChiSq = 225.;
  theMinChiSqImprovement = 50.;
}

RefCountedKinematicTree KinematicConstrainedVertexFitter::fit(const std::vector<RefCountedKinematicParticle> &part,
                                                             MultiTrackKinematicConstraint * cs,
                                                             GlobalPoint * pt)
{
 if(part.size()<2) throw VertexException("KinematicConstrainedVertexFitter::input states are less than 2");

//sorting out the input particles
 InputSort iSort;
 std::pair<std::vector<RefCountedKinematicParticle>, std::vector<FreeTrajectoryState> > input = iSort.sort(part);
 const std::vector<RefCountedKinematicParticle> & particles  = input.first;
 const std::vector<FreeTrajectoryState> & fStates = input.second;

// linearization point
// (only compute it using the linearization point finder if no point was passed to the fit function):
 GlobalPoint linPoint;
 if (pt!=0) {
   linPoint  = *pt;
 }
 else {
   linPoint = finder->getLinearizationPoint(fStates);
 }

//initial parameters:
 int vSize = particles.size();
 AlgebraicVector inPar(3 + 7*vSize,0);

//final parameters
 AlgebraicVector finPar(3 + 7*vSize,0);

//initial covariance
 AlgebraicMatrix inCov(3 + 7*vSize,3 + 7*vSize,0);

//making initial vector of parameters and initial particle-related covariance
 int nSt = 0;
 std::vector<KinematicState> inStates;
 for(std::vector<RefCountedKinematicParticle>::const_iterator i = particles.begin(); i!=particles.end(); i++)
 {
  KinematicState state = (*i)->stateAtPoint(linPoint);
  if (!state.isValid()) {
      LogDebug("KinematicConstrainedVertexFitter")
       << "State is invalid at point: "<<linPoint<<std::endl;
      return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
  }
  AlgebraicVector prPar = asHepVector<7>(state.kinematicParameters().vector());
  for(int j = 1; j<8; j++){inPar(3 + 7*nSt + j) = prPar(j);}
  AlgebraicSymMatrix l_cov  = asHepMatrix<7>(state.kinematicParametersError().matrix());
  inCov.sub(4 + 7*nSt,4 + 7*nSt ,l_cov);
  inStates.push_back(state);
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
 iterations = 0;
 csum = 0.0;

 std::vector<KinematicState> lStates = inStates;
 GlobalPoint lPoint  = linPoint;
 RefCountedKinematicVertex rVtx;
 AlgebraicMatrix refCCov;

 double chisq = 1e6;
 bool convergence = false;
//iterarions over the updator: each time updated parameters
//are taken as new linearization point
 do{
  eq = 0.;
  std::pair< std::pair< std::vector<KinematicState>, AlgebraicMatrix >,RefCountedKinematicVertex> lRes =
                                      updator->update(inPar,inCov,lStates,lPoint,cs);
 
  const std::vector<KinematicState> &newStates = lRes.first.first;

  if (particles.size() != newStates.size()) {
    LogDebug("KinematicConstrainedVertexFitter")
        << "updator failure\n";
    return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
  }

                                      
  rVtx = lRes.second;                                      
                    
  double newchisq = rVtx->chiSquared();
  if ( nit>2 && newchisq > theMaxReducedChiSq*rVtx->degreesOfFreedom() && (newchisq-chisq) > (-theMinChiSqImprovement) ) {
    LogDebug("KinematicConstrainedVertexFitter")
           << "bad chisq and insufficient improvement, bailing\n";
    return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
  }
  chisq = newchisq;
  

  const GlobalPoint &newPoint = rVtx->position();
  
  double maxDelta = 0.0;
  
  double deltapos[3];
  deltapos[0] = newPoint.x() - lPoint.x();
  deltapos[1] = newPoint.y() - lPoint.y();
  deltapos[2] = newPoint.z() - lPoint.z();
  for (int i=0; i<3; ++i) {
    double delta = deltapos[i]*deltapos[i]/rVtx->error().matrix_new()(i,i);
    if (delta>maxDelta) maxDelta = delta;
  }
  
  for (std::vector<KinematicState>::const_iterator itold = lStates.begin(), itnew = newStates.begin();
       itnew!=newStates.end(); ++itold,++itnew) {
    for (int i=0; i<7; ++i) {
      double deltapar = itnew->kinematicParameters()(i) - itold->kinematicParameters()(i);
      double delta = deltapar*deltapar/itnew->kinematicParametersError().matrix()(i,i);
      if (delta>maxDelta) maxDelta = delta;
    }
  }
  
  lStates = newStates;
  lPoint = newPoint;

  refCCov = lRes.first.second;
  nit++;
  convergence = maxDelta<theMaxDelta || (nit==theMaxStep && maxDelta<4.0*theMaxDelta);

 }while(nit<theMaxStep && !convergence);

 if (!convergence) {
   return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
 } 

  // std::cout << "old full cov matrix" << std::endl;
  // std::cout << refCCov << std::endl;


// cout<<"number of relinearizations "<<nit<<endl;
// cout<<"value obtained: "<<eq<<endl;
  iterations = nit;
  csum = eq;

  return  tBuilder->buildTree(particles, lStates, rVtx, refCCov);

}

int KinematicConstrainedVertexFitter::getNit() const {
    return iterations;
}

float KinematicConstrainedVertexFitter::getCSum() const {
    return csum;
}
