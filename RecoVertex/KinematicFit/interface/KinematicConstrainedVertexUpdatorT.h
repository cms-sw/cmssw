#ifndef KinematicConstrainedVertexUpdatorT_H
#define KinematicConstrainedVertexUpdatorT_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraintT.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"
#include "RecoVertex/KinematicFit/interface/VertexKinematicConstraintT.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include<cassert>
#include<iostream>
// the usual stupid counter
namespace KineDebug3 {
  struct Count {
    int n;
    Count() : n(0){}
    ~Count() {
    }

  };
  inline void count() {
    static Count c;
    ++c.n;
  }

}

/**
 * Class caching the math part for
 * KinematicConstrainedVertexFitter
 */

template < int nTrk, int nConstraint> class KinematicConstrainedVertexUpdatorT
{
public:
  
  /**
   * Default constructor and destructor
   */
  KinematicConstrainedVertexUpdatorT();
  
  ~KinematicConstrainedVertexUpdatorT();
  
  /**
   * Method updating the states. Takes a vector of full parameters:
   * (x,y,z,particle_1,...,particle_n), corresponding linearization 
   * point: vector of states and GlobalPoint, 
   * and constraint to be applied during the vertex fit.
   * Returns refitted vector of 7n+3 parameters and corresponding
   * covariance matrix, where n - number of tracks.
   */ 
  RefCountedKinematicVertex  
  update(const ROOT::Math::SVector<double, 3+7*nTrk> & inState, 
	 ROOT::Math::SMatrix<double, 3+7*nTrk,3+7*nTrk  ,ROOT::Math::MatRepSym<double,3+7*nTrk> >& inCov, std::vector<KinematicState> & lStates, 
	 const GlobalPoint& lPoint, GlobalVector const & fieldValue, MultiTrackKinematicConstraintT< nTrk, nConstraint > * cs);
  
private:
  
  KinematicVertexFactory vFactory;
  VertexKinematicConstraintT vConstraint;			       	
  ROOT::Math::SVector<double,3+7*nTrk> delta_alpha;
  ROOT::Math::SMatrix<double,nConstraint+4,3+7*nTrk> g;
  ROOT::Math::SVector<double,nConstraint+4> val;
  // ROOT::Math::SMatrix<double,3+7*nTrk,3+7*nTrk,ROOT::Math::MatRepSym<double,3+7*nTrk> > in_cov_sym;
  // ROOT::Math::SMatrix<double,3+7*nTrk>  rCov;
  ROOT::Math::SVector<double, 3+7*nTrk> finPar;
  ROOT::Math::SVector<double, nConstraint+4> lambda;
  // ROOT::Math::SMatrix<double,3+7*nTrk,3+7*nTrk,ROOT::Math::MatRepSym<double,3+7*nTrk> > r_cov_sym;
  ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> > pCov; 
  ROOT::Math::SMatrix<double,7,7,ROOT::Math::MatRepSym<double,7> > nCovariance;
  ROOT::Math::SMatrix<double,nConstraint+4,nConstraint+4,ROOT::Math::MatRepSym<double,nConstraint+4> > v_g_sym;   
  
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"

template < int nTrk, int nConstraint> 
KinematicConstrainedVertexUpdatorT< nTrk, nConstraint >::KinematicConstrainedVertexUpdatorT()
{}

template < int nTrk, int nConstraint> 
KinematicConstrainedVertexUpdatorT< nTrk, nConstraint >::~KinematicConstrainedVertexUpdatorT()
{}

template < int nTrk, int nConstraint> 
RefCountedKinematicVertex 
KinematicConstrainedVertexUpdatorT< nTrk, nConstraint >::update(const ROOT::Math::SVector<double,3+7*nTrk>& inPar,
								ROOT::Math::SMatrix<double, 3+7*nTrk,3+7*nTrk ,ROOT::Math::MatRepSym<double,3+7*nTrk> >& inCov,
								std::vector<KinematicState> & lStates,
								const GlobalPoint& lPoint,
								GlobalVector const & fieldValue, 
								MultiTrackKinematicConstraintT< nTrk, nConstraint > * cs)
{
  KineDebug3::count();

  int vSize = lStates.size();

  assert( nConstraint==0 || cs!=0);
  assert(vSize == nConstraint);

  const MagneticField* field=lStates.front().magneticField();
  
  delta_alpha=inPar;
  delta_alpha(0)-=lPoint.x();
  delta_alpha(1)-=lPoint.y();
  delta_alpha(2)-=lPoint.z();
  int cst=3;
  for(std::vector<KinematicState>::const_iterator i = lStates.begin(); i != lStates.end(); i++) 
    for ( int j=0; j<7; j++) {
      delta_alpha(cst)-=i->kinematicParameters()(j);
      cst++;
    }
  
  // cout<<"delta_alpha"<<delta_alpha<<endl;
  //resulting matrix of derivatives and vector of values.
  //their size  depends of number of tracks to analyze and number of
  //additional constraints to apply 
  
  if( nConstraint !=0) {
    cs->init(lStates, lPoint, fieldValue);
    val.Place_at(cs->value(),0);
    g.Place_at(cs->positionDerivative(),0,0);
    g.Place_at(cs->parametersDerivative(),0,3);
  }

  vConstraint.init(lStates, lPoint, fieldValue);
  val.Place_at(vConstraint.value(),nConstraint);
  g.Place_at(vConstraint.positionDerivative(),nConstraint, 0);
  g.Place_at(vConstraint.parametersDerivative(),nConstraint, 3);
  
  

  
  //debug code   
  v_g_sym = ROOT::Math::Similarity(g,inCov);
  
  // bool ifl1 = v_g_sym.Invert();
  bool ifl1 = invertPosDefMatrix(v_g_sym);
  if(!ifl1) {
    edm::LogWarning("KinematicConstrainedVertexUpdatorFailed")<< "invert failed\n"
							      << v_g_sym;
    LogDebug("KinematicConstrainedVertexFitter3")
      << "Fit failed: unable to invert SYM gain matrix\n";
    return  RefCountedKinematicVertex();	
  }
  
  // delta alpha is now valid!
  //full math case now!
  val += g*delta_alpha;
  lambda = v_g_sym *val;
  
  //final parameters  
  finPar = inPar -  inCov * (ROOT::Math::Transpose(g) * lambda);
  
  //refitted covariance 
  ROOT::Math::SMatrix<double,3+7*nTrk,3+7*nTrk,ROOT::Math::MatRepSym<double,3+7*nTrk> > prod = ROOT::Math::SimilarityT(g,v_g_sym);
  ROOT::Math::SMatrix<double,3+7*nTrk,3+7*nTrk,ROOT::Math::MatRepSym<double,3+7*nTrk> > prod1;
  ROOT::Math::AssignSym::Evaluate(prod1, inCov * prod * inCov); 
  // ROOT::Math::AssignSym::Evaluate(prod, prod1 * inCov); 
  inCov -= prod1;
  
  pCov = inCov.template Sub< ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> > >(0,0);
  
  // chi2
  double chi  = ROOT::Math::Dot(lambda,val); //??
  
  //this is ndf without significant prior
  //vertex so -3 factor exists here 
  float ndf = 2*vSize - 3;
  ndf += nConstraint;
 
  
  //making resulting vertex 
  GlobalPoint vPos (finPar(0),finPar(1),finPar(2)); 
  VertexState st(vPos,GlobalError(pCov));
  RefCountedKinematicVertex rVtx = vFactory.vertex(st,chi,ndf);
  
  //making refitted states of Kinematic Particles
  AlgebraicVector7 newPar; 
  int i_int = 0;
  for(std::vector<KinematicState>::iterator i_st=lStates.begin(); i_st != lStates.end(); i_st++)
    {
      for(int i =0; i<7; i++)
	{newPar(i) = finPar(3 + i_int*7 + i);}
      
   nCovariance = inCov.template Sub<ROOT::Math::SMatrix<double, 7,7,ROOT::Math::MatRepSym<double,7> > >(3 + i_int*7, 3 + i_int*7);
   TrackCharge chl = i_st->particleCharge();
   KinematicParameters nrPar(newPar);
   KinematicParametersError nrEr(nCovariance);
   KinematicState newState(nrPar,nrEr,chl, field);
   (*i_st) = newState;
   i_int++;
  }
  return rVtx;	
}

#endif
