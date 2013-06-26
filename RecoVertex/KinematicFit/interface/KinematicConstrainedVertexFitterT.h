#ifndef KinematicConstrainedVertexFitterT_H
#define KinematicConstrainedVertexFitterT_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraintT.h"
#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexUpdatorT.h"
#include "RecoVertex/KinematicFit/interface/VertexKinematicConstraintT.h"

#include "RecoVertex/KinematicFit/interface/ConstrainedTreeBuilderT.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
 * Class fitting the veretx out of set of tracks via usual LMS
 * with Lagrange multipliers.
 * Additional constraints can be applyed to the tracks during the vertex fit
 * (solves non-factorizabele cases). Since the vertex constraint is included by default, do not add a separate
 * VertexKinematicConstraint!
 * Example: Vertex fit with collinear tracks..
 */

template < int nTrk, int nConstraint> class KinematicConstrainedVertexFitterT{

public:
  
  /**
   * Default constructor using LMSLinearizationPointFinder
   */
  explicit KinematicConstrainedVertexFitterT(const MagneticField* ifield);
  
  /**
   * Constructor with user-provided LinearizationPointFinder
   */  
  KinematicConstrainedVertexFitterT(const MagneticField* ifield, const LinearizationPointFinder& fnd);
  
  ~KinematicConstrainedVertexFitterT();
  
  /**
   * Configuration through PSet: number of iterations(maxDistance) and
   * stopping condition (maxNbrOfIterations)
   */
  
  void setParameters(const edm::ParameterSet& pSet);
  
  /**
   * Without additional constraint, this will perform a simple
   * vertex fit using LMS with Lagrange multipliers method (by definition valid only if nConstraint=0)
   */  
  RefCountedKinematicTree fit(const std::vector<RefCountedKinematicParticle> &part) {
    return fit(part, 0, 0);
  }
  
  /**
   * LMS with Lagrange multipliers fit of vertex constraint and user-specified constraint.
   */  
  RefCountedKinematicTree fit(const std::vector<RefCountedKinematicParticle> &part,
			      MultiTrackKinematicConstraintT< nTrk, nConstraint> * cs) {
    return fit(part, cs, 0);
  };
  
  /**
   * LMS with Lagrange multipliers fit of vertex constraint, user-specified constraint and user-specified starting point.
   */  
  RefCountedKinematicTree fit(const std::vector<RefCountedKinematicParticle> &part,
			      MultiTrackKinematicConstraintT< nTrk, nConstraint> * cs,
			      GlobalPoint * pt);
  
  //return the number of iterations
 int getNit() const;
  //return the value of the constraint equation
  float getCSum() const;
  
private:
  
  void defaultParameters();
  
  const MagneticField* field;
  LinearizationPointFinder * finder;				       
  KinematicConstrainedVertexUpdatorT<nTrk,nConstraint> * updator;
  VertexKinematicConstraintT * vCons;
  ConstrainedTreeBuilderT * tBuilder;

  float theMaxDelta; //maximum (delta parameter)^2/(sigma parameter)^2 per iteration for convergence
  int theMaxStep; 				       
  float theMaxReducedChiSq; //max of initial (after 2 iterations) chisq/dof value
  float theMinChiSqImprovement; //minimum required improvement in chisq to avoid fit termination for cases exceeding theMaxReducedChiSq


  int iterations;
  float csum;
};


#include "RecoVertex/KinematicFit/interface/InputSort.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



template < int nTrk, int nConstraint> 
KinematicConstrainedVertexFitterT< nTrk, nConstraint>::KinematicConstrainedVertexFitterT(const MagneticField* ifield) : 
  field(ifield)
{
  finder = new DefaultLinearizationPointFinder();
  vCons = new VertexKinematicConstraintT();
  updator = new KinematicConstrainedVertexUpdatorT<nTrk,nConstraint>();
  tBuilder = new ConstrainedTreeBuilderT;
  defaultParameters();
  iterations = -1;
  csum = -1000.0;
}

template < int nTrk, int nConstraint> 
KinematicConstrainedVertexFitterT< nTrk, nConstraint>::KinematicConstrainedVertexFitterT(const MagneticField* ifield, const LinearizationPointFinder& fnd) : 
  field(ifield)
{
  finder = fnd.clone();
  vCons = new VertexKinematicConstraintT();
  updator = new KinematicConstrainedVertexUpdatorT<nTrk,nConstraint>();
  tBuilder = new ConstrainedTreeBuilderT;
  defaultParameters();
  iterations = -1;
  csum = -1000.0;
}

template < int nTrk, int nConstraint> 
KinematicConstrainedVertexFitterT< nTrk, nConstraint>::~KinematicConstrainedVertexFitterT()
{
  delete finder;
  delete vCons;
  delete updator;
  delete tBuilder;
}

template < int nTrk, int nConstraint> 
void KinematicConstrainedVertexFitterT< nTrk, nConstraint>::setParameters(const edm::ParameterSet& pSet)
{
  theMaxDelta = pSet.getParameter<double>("maxDelta");
  theMaxStep = pSet.getParameter<int>("maxNbrOfIterations");
  theMaxReducedChiSq = pSet.getParameter<double>("maxReducedChiSq");
  theMinChiSqImprovement = pSet.getParameter<double>("minChiSqImprovement");
}

template < int nTrk, int nConstraint> 
void KinematicConstrainedVertexFitterT< nTrk, nConstraint>::defaultParameters()
{
  theMaxDelta = 0.01;
  theMaxStep = 1000;
  theMaxReducedChiSq = 225.;
  theMinChiSqImprovement = 50.;
 
}

template < int nTrk, int nConstraint> 
RefCountedKinematicTree 
KinematicConstrainedVertexFitterT< nTrk, nConstraint>::fit(const std::vector<RefCountedKinematicParticle> &part,
							   MultiTrackKinematicConstraintT< nTrk, nConstraint> * cs,
							   GlobalPoint * pt)
{
   assert( nConstraint==0 || cs!=0);
   if(part.size()!=nTrk) throw VertexException("KinematicConstrainedVertexFitterT::input states are not nTrk");
  
  //sorting out the input particles
  InputSort iSort;
  std::pair<std::vector<RefCountedKinematicParticle>, std::vector<FreeTrajectoryState> > input = iSort.sort(part);
  const std::vector<RefCountedKinematicParticle> & particles  = input.first;
  const std::vector<FreeTrajectoryState> & fStates = input.second;
  
  // linearization point:
  GlobalPoint linPoint  = (pt!=0) ? *pt :  finder->getLinearizationPoint(fStates);
  
  //initial parameters:
  ROOT::Math::SVector<double,3+7*nTrk> inPar; //3+ 7*ntracks
  ROOT::Math::SVector<double,3+7*nTrk> finPar; //3+ 7*ntracks
  
  ROOT::Math::SMatrix<double, 3+7*nTrk,3+7*nTrk ,ROOT::Math::MatRepSym<double,3+7*nTrk> > inCov;
  
  //making initial vector of parameters and initial particle-related covariance
  int nSt = 0;
  std::vector<KinematicState> lStates(nTrk);
  for(std::vector<RefCountedKinematicParticle>::const_iterator i = particles.begin(); i!=particles.end(); i++)
    {
      lStates[nSt] = (*i)->stateAtPoint(linPoint);
      KinematicState const & state = lStates[nSt];
      if (!state.isValid()) {
	LogDebug("KinematicConstrainedVertexFitter")
	  << "State is invalid at point: "<<linPoint<<std::endl;
	return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
      }
      inPar.Place_at(state.kinematicParameters().vector(),3+7*nSt);
      inCov.Place_at(state.kinematicParametersError().matrix(),3 + 7*nSt,3 + 7*nSt);
      ++nSt;
    }
  
  //initial vertex error matrix components (huge error method)
  //and vertex related initial vector components
  double in_er = 100.;
  inCov(0,0) = in_er;
  inCov(1,1) = in_er;
  inCov(2,2) = in_er;
  
  inPar(0) = linPoint.x();
  inPar(1) = linPoint.y();
  inPar(2) = linPoint.z();
  
  //constraint equations value and number of iterations
  double eq;
  int nit = 0;
  iterations = 0;
  csum = 0.0;
  
  GlobalPoint lPoint  = linPoint;
  RefCountedKinematicVertex rVtx;
  ROOT::Math::SMatrix<double, 3+7*nTrk,3+7*nTrk ,ROOT::Math::MatRepSym<double,3+7*nTrk> > refCCov;
  
  double chisq = 1e6;
  bool convergence = false;
  
  //iterarions over the updator: each time updated parameters
  //are taken as new linearization point
  do{
    eq = 0.;
    refCCov = inCov;
    std::vector<KinematicState> oldStates = lStates;
    GlobalVector mf = field->inInverseGeV(lPoint);
    rVtx = updator->update(inPar,refCCov,lStates,lPoint,mf,cs);
    if (particles.size() != lStates.size() || rVtx == 0) {
      LogDebug("KinematicConstrainedVertexFitter")
	<< "updator failure\n";
      return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
    }
    
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
    
    for (std::vector<KinematicState>::const_iterator itold = oldStates.begin(), itnew = lStates.begin();
	 itnew!=lStates.end(); ++itold,++itnew) {
      for (int i=0; i<7; ++i) {
	double deltapar = itnew->kinematicParameters()(i) - itold->kinematicParameters()(i);
	double delta = deltapar*deltapar/itnew->kinematicParametersError().matrix()(i,i);
	if (delta>maxDelta) maxDelta = delta;
      }
    }
    
    lPoint = newPoint;
    
    nit++;
    convergence = maxDelta<theMaxDelta || (nit==theMaxStep && maxDelta<4.0*theMaxDelta);
    
  }while(nit<theMaxStep && !convergence);

  if (!convergence) {
    return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
  } 
  
  // std::cout << "new full cov matrix" << std::endl;
  // std::cout << refCCov << std::endl;  
  
  iterations = nit;
  csum = eq;
  
  return  tBuilder->buildTree<nTrk>(particles, lStates, rVtx, refCCov);
  
}

template < int nTrk, int nConstraint> 
int KinematicConstrainedVertexFitterT< nTrk, nConstraint>::getNit() const {
  return iterations;
}

template < int nTrk, int nConstraint> 
float KinematicConstrainedVertexFitterT< nTrk, nConstraint>::getCSum() const {
  return csum;
}

#endif
