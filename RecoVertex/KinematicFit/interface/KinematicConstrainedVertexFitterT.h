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
  RefCountedKinematicTree fit(std::vector<RefCountedKinematicParticle> part) {
    return fit(part, 0, 0);
  }
  
  /**
   * LMS with Lagrange multipliers fit of vertex constraint and user-specified constraint.
   */  
  RefCountedKinematicTree fit(std::vector<RefCountedKinematicParticle> part,
			      MultiTrackKinematicConstraintT< nTrk, nConstraint> * cs) {
    return fit(part, cs, 0);
  };
  
  /**
   * LMS with Lagrange multipliers fit of vertex constraint, user-specified constraint and user-specified starting point.
   */  
  RefCountedKinematicTree fit(std::vector<RefCountedKinematicParticle> part,
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
  float theMaxDiff;
  int theMaxStep; 				       
  float theMaxInitial;//max of initial value

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
  theMaxDiff = pSet.getParameter<double>("maxDistance");
  theMaxStep = pSet.getParameter<int>("maxNbrOfIterations");
  theMaxInitial = pSet.getParameter<double>("maxOfInitialValue");
}

template < int nTrk, int nConstraint> 
void KinematicConstrainedVertexFitterT< nTrk, nConstraint>::defaultParameters()
{
  theMaxDiff = 0.0001;
  theMaxStep = 1000;
  theMaxInitial = 9999.; //dummy value
}

template < int nTrk, int nConstraint> 
RefCountedKinematicTree 
KinematicConstrainedVertexFitterT< nTrk, nConstraint>::fit(std::vector<RefCountedKinematicParticle> part,
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
  GlobalPoint linPoint  = finder->getLinearizationPoint(fStates);
  if (pt!=0) linPoint  = *pt;
  
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
  GlobalVector mf = field->inInverseGeV(lPoint);
  RefCountedKinematicVertex rVtx;
  ROOT::Math::SMatrix<double, 3+7*nTrk,3+7*nTrk ,ROOT::Math::MatRepSym<double,3+7*nTrk> > refCCov;
  
  //iterarions over the updator: each time updated parameters
  //are taken as new linearization point
  do{
    eq = 0.;
    refCCov = inCov;
    rVtx = updator->update(inPar,refCCov,lStates,lPoint,mf,cs);
    if (particles.size() != lStates.size()) {
      LogDebug("KinematicConstrainedVertexFitter")
	<< "updator failure\n";
      return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
    }
    lPoint = rVtx->position();
    GlobalVector mf = field->inInverseGeV(lPoint);
    //std::cout << "n3" << lPoint<< std::endl;
    //std::cout << lStates<<std::endl;
    vCons->init(lStates, lPoint,mf);
    ROOT::Math::SVector<double,4> vValue = vCons->value(); //guess
    //std::cout << "n3vv " << vValue << " ";
    for(int i = 0; i<4;++i)
      eq += std::abs(vValue(i));
    
    if(nConstraint!=0) {
      cs->init(lStates, lPoint,mf);
      ROOT::Math::SVector<double,nConstraint> cVal = cs->value(); //guess
      // std::cout << cVal << " ";
      for(int i = 0; i<nConstraint;++i)
	eq += std::abs(cVal(i));
    }
    // std::cout << eq << std::endl;
    if (nit == 0) {
      if (eq>theMaxInitial) return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
    }
    if( isnan(eq) ){
      LogDebug("KinematicConstrainedVertexFitter")
	<< "catched NaN.\n";
      return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
    }
    
    nit++;
  }while(nit<theMaxStep && eq>theMaxDiff);
  
  if (eq>theMaxDiff) {
    return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
  } 
  
  // cout<<"number of relinearizations "<<nit<<endl;
  // cout<<"value obtained: "<<eq<<endl;
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
