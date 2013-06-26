#ifndef ConstrainedTreeBuilderT_H
#define ConstrainedTreeBuilderT_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/VirtualKinematicParticleFactory.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"

/**
 * Class constructing te final output tree  for the constrained vertex fitter. 
 * To be used by corresponding fitter only. Tree builders are scheduled for 
 * generalization: They should be inherited from the single generic class
 * in the next version of the library.
 */

class ConstrainedTreeBuilderT
{

public:

  ConstrainedTreeBuilderT(){}
 
  ~ConstrainedTreeBuilderT(){}

/**
 * Method constructing tree out of set of refitted states, vertex, and
 * full covariance matrix.
 */

  template<int nTrk>
  RefCountedKinematicTree 
  buildTree(const std::vector<RefCountedKinematicParticle> & initialParticles, 
	    const std::vector<KinematicState> & finalStates,
	    const RefCountedKinematicVertex vtx, 
	    const ROOT::Math::SMatrix<double,3+7*nTrk,3+7*nTrk,ROOT::Math::MatRepSym<double,3+7*nTrk> >& fCov) const;

private:

  RefCountedKinematicTree buildRealTree(const RefCountedKinematicParticle virtualParticle, 
	const RefCountedKinematicVertex vtx, const std::vector<RefCountedKinematicParticle> & particles) const;

  /**
   * Metod to reconstruct the full covariance matrix of the resulting particle.					      
   */
  template<int nTrk>
  static AlgebraicSymMatrix77 
  covarianceMatrix(const std::vector<RefCountedKinematicParticle> &rPart, 
		   const AlgebraicVector7& newPar,
		   const ROOT::Math::SMatrix<double,3+7*nTrk,3+7*nTrk,ROOT::Math::MatRepSym<double,3+7*nTrk> >& fitCov);
				       
 VirtualKinematicParticleFactory pFactory;				       
 KinematicVertexFactory vFactory;
};


#include "DataFormats/CLHEP/interface/Migration.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

template<int nTrk>
RefCountedKinematicTree 
ConstrainedTreeBuilderT::buildTree(const std::vector<RefCountedKinematicParticle> & initialParticles, 
				   const std::vector<KinematicState> & finalStates,
				   const RefCountedKinematicVertex vertex, 
				   const ROOT::Math::SMatrix<double,3+7*nTrk,3+7*nTrk,ROOT::Math::MatRepSym<double,3+7*nTrk> >& fullCov) const
{
  if (!vertex->vertexIsValid()) {
       LogDebug("RecoVertex/ConstrainedTreeBuilder")
	<< "Vertex is invalid\n";
       return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
  }
  AlgebraicVector3 vtx;
  vtx(0) = vertex->position().x();
  vtx(1) = vertex->position().y();
  vtx(2) = vertex->position().z();
  AlgebraicMatrix33 vertexCov = fullCov.template Sub<ROOT::Math::SMatrix<double, 3> >(0,0);

// cout << fullCov<<endl;
//  cout << "RecoVertex/ConstrainedTreeBuilder"<<vtx<<endl;

 double ent = 0.;
 int charge = 0;
  AlgebraicVector7 virtualPartPar;
  virtualPartPar(0) = vertex->position().x();
  virtualPartPar(1) = vertex->position().y();
  virtualPartPar(2) = vertex->position().z();

//making refitted particles out of refitted states.
//none of the operations above violates the order of particles

  ROOT::Math::SMatrix<double,7,7,ROOT::Math::MatRepStd<double,7,7> >  aMatrix;
  ROOT::Math::SMatrix<double,7,7,ROOT::Math::MatRepStd<double,7,7> >  aMatrixT;
  aMatrix(3,3) =  aMatrixT(3,3) = 1;
  aMatrix(4,4) =  aMatrixT(4,4)  = 1;
  aMatrix(5,5) =  aMatrixT(5,5)  = 1;
  aMatrix(6,6) =  aMatrixT(6,6)  = 1;
  ROOT::Math::SMatrix<double,7,3,ROOT::Math::MatRepStd<double,7,3> > bMatrix;
  ROOT::Math::SMatrix<double,3,7,ROOT::Math::MatRepStd<double,3,7> > bMatrixT;
  bMatrix(0,0) =  bMatrixT(0,0) = 1;
  bMatrix(1,1) =  bMatrixT(1,1) = 1;
  bMatrix(2,2) =  bMatrixT(2,2) = 1;
  AlgebraicSymMatrix77 trackParCov;
  ROOT::Math::SMatrix<double,3,7,ROOT::Math::MatRepStd<double,3,7> > vtxTrackCov;
  AlgebraicSymMatrix77 nCovariance;
  // AlgebraicSymMatrix77 tmp;

  std::vector<RefCountedKinematicParticle>::const_iterator i = initialParticles.begin();
  std::vector<KinematicState>::const_iterator iStates = finalStates.begin();
  std::vector<RefCountedKinematicParticle> rParticles;
  int n=0;
  // assert(initialParticles.size()==nTrk);
  for( ; i != initialParticles.end(), iStates != finalStates.end(); ++i,++iStates)
  {
    AlgebraicVector7 p = iStates->kinematicParameters().vector();
    double a = - iStates->particleCharge() *
	iStates->magneticField()->inInverseGeV(iStates->globalPosition()).z();

    aMatrix(4,0) = aMatrixT(0,4) = -a;
    aMatrix(3,1) = aMatrixT(1,3) =  a;
    bMatrix(4,0) = bMatrixT(0,4) =  a;
    bMatrix(3,1) = bMatrixT(1,3) = -a;

    AlgebraicVector7 par = aMatrix*p + bMatrix * vtx;

    trackParCov = fullCov.template Sub<AlgebraicSymMatrix77>(3+n*7,3+n*7);
    vtxTrackCov = fullCov.template Sub<ROOT::Math::SMatrix<double, 3, 7> >(0,3+n*7);
    ROOT::Math::AssignSym::Evaluate(nCovariance,
				    aMatrix * trackParCov * aMatrixT
				    + aMatrix * ROOT::Math::Transpose(vtxTrackCov) * bMatrixT
				    + bMatrix * vtxTrackCov * aMatrixT 
				    + bMatrix * vertexCov * bMatrixT
				    );
    /*
    ROOT::Math::AssignSym::Evaluate(tmp, aMatrix * ROOT::Math::Transpose(vtxTrackCov) * bMatrixT);
    nCovariance+=tmp;
    ROOT::Math::AssignSym::Evaluate(tmp, bMatrix * vtxTrackCov * aMatrixT);
    nCovariance+=tmp;
    ROOT::Math::AssignSym::Evaluate(tmp, bMatrix * vertexCov * bMatrixT);
    nCovariance+=tmp;
    */

    KinematicState stateAtVertex(KinematicParameters(par),
				 KinematicParametersError(nCovariance),
				 iStates->particleCharge(), iStates->magneticField());
    rParticles.push_back((*i)->refittedParticle(stateAtVertex, vertex->chiSquared(), vertex->degreesOfFreedom()));
    
    virtualPartPar(3) += par(3);
    virtualPartPar(4) += par(4);
    virtualPartPar(5) += par(5);
    ent += sqrt(par(6)*par(6) +  par(3)*par(3)+par(4)*par(4)+par(5)*par(5) );
    charge += iStates->particleCharge();

    ++n;

  }

 //total reconstructed mass
  double differ = ent*ent - (virtualPartPar(3)*virtualPartPar(3) + virtualPartPar(5)*virtualPartPar(5) + virtualPartPar(4)*virtualPartPar(4));
  if(differ>0.) {
    virtualPartPar(6) = sqrt(differ);
  } else {
   LogDebug("ConstrainedTreeBuilder")
	<< "Fit failed: Current precision does not allow to calculate the mass\n";
   return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
  }

 // covariance matrix:

  AlgebraicSymMatrix77 cov = this->covarianceMatrix<nTrk>(rParticles,virtualPartPar,fullCov);

  KinematicState nState(KinematicParameters(virtualPartPar),
			KinematicParametersError(cov),
			charge, initialParticles[0]->magneticField());

 //newborn kinematic particle
  float chi2 = vertex->chiSquared();
  float ndf = vertex->degreesOfFreedom();
  KinematicParticle * zp = 0;
  RefCountedKinematicParticle virtualParticle = pFactory.particle(nState,chi2,ndf,zp);

  return buildRealTree(virtualParticle, vertex, rParticles);
}

template<int nTrk>
AlgebraicSymMatrix77 
ConstrainedTreeBuilderT::covarianceMatrix(const std::vector<RefCountedKinematicParticle> &rPart, 
					  const AlgebraicVector7& newPar,
					  const ROOT::Math::SMatrix<double,3+7*nTrk,3+7*nTrk,ROOT::Math::MatRepSym<double,3+7*nTrk> >& fitCov)
 {

   typedef ROOT::Math::SMatrix<double,3+7*nTrk,3+7*nTrk,ROOT::Math::MatRepSym<double,3+7*nTrk> > FitCov;
  //constructing the full matrix using the simple fact
  //that we have never broken the order of tracks
  // during our fit.
  
  int size = nTrk;  

  if ( int(rPart.size())!=size) throw "error in ConstrainedTreeBuilderT ";

  //global propagation to the vertex position
  //Jacobian is done for all the parameters together
  ROOT::Math::SMatrix<double,3+7*nTrk> jac;
  jac(0,0) = 1;
  jac(1,1) = 1;
  jac(2,2) = 1;
  ROOT::Math::SMatrix<double,3,7> upper;
  ROOT::Math::SMatrix<double,7>  diagonal;
  for(  int i_int=0; i_int!=size; ++i_int) {
    RefCountedKinematicParticle const & i = rPart[i_int];   
    //vertex position related components of the matrix
    double a_i = - (i)->currentState().particleCharge() *
      (i)->magneticField()->inInverseGeV((i)->currentState().globalPosition()).z();
    upper(0,0) = 1;
    upper(1,1) = 1;
    upper(2,2) = 1;
    upper(1,3) = -a_i;
    upper(0,4) = a_i;
    jac.Place_at(upper,0,3+i_int*7);
    
    diagonal(3,3) = 1;
    diagonal(4,4) = 1;
    diagonal(5,5) = 1;
    diagonal(6,6) = 1;
    diagonal(1,3) = a_i;
    diagonal(0,4) = -a_i;
    jac.Place_at(diagonal,3+i_int*7,3+i_int*7);
  }
  
  // jacobian is constructed in such a way, that
  // right operation for transformation will be
  // fitCov.similarityT(jac)
  // WARNING: normal similarity operation is
  // not valid in this case
  //now making reduced matrix:
  // int vSize = rPart.size();
  FitCov const & fit_cov_sym = fitCov;
  /*
    for(int i = 0; i<7*vSize+3; ++i)
    {
    for(int j = 0; j<7*vSize+3; ++j)
    {if(i<=j) fit_cov_sym(i,j) = fitCov(i,j);}
    }
  */
  
  ROOT::Math::SMatrix<double,4*nTrk+3,4*nTrk+3, ROOT::Math::MatRepSym<double,4*nTrk+3> > reduced;
  FitCov transform=ROOT::Math::SimilarityT(jac,fit_cov_sym); // similarityT???
  
  //jacobian to add matrix components
  ROOT::Math::SMatrix<double,7,4*nTrk+3> jac_t;
  jac_t(0,0) = 1.;
  jac_t(1,1) = 1.;
  jac_t(2,2) = 1.;
  

  
  double energy_global = sqrt(newPar(3)*newPar(3)+newPar(4)*newPar(4) + newPar(5)*newPar(5)+newPar(6)*newPar(6));
  for(int il_int = 0; il_int!=size; ++il_int) {
    RefCountedKinematicParticle const & rs = rPart[il_int];
    //jacobian components:
    int off1=3; int off2=il_int*4+3;
    jac_t(off1+0,off2+0) = 1.;
    jac_t(off1+1,off2+1) = 1.;
    jac_t(off1+2,off2+2) = 1.;
    
    //non-trival elements: mass correlations:
    AlgebraicVector7 l_Par = (rs)->currentState().kinematicParameters().vector();
    double energy_local  = sqrt(l_Par(6)*l_Par(6) + l_Par(3)*l_Par(3) + l_Par(4)*l_Par(4) + l_Par(5)*l_Par(5));
    jac_t(off1+3,off2+3) = energy_global*l_Par(6)/(newPar(6)*energy_local);
    jac_t(off1+3,off2+0) = ((energy_global*l_Par(3)/energy_local) - newPar(3))/newPar(6);
    jac_t(off1+3,off2+1) = ((energy_global*l_Par(4)/energy_local) - newPar(4))/newPar(6);
    jac_t(off1+3,off2+2) = ((energy_global*l_Par(5)/energy_local) - newPar(5))/newPar(6);
  }

  for(int i = 0; i<7;++i)
    for(int j =0; j<7; ++j)
      reduced(i,j)  = transform(i+3, j+3);

  for(int i = 1; i<size; i++) {
    //non-trival elements: mass correlations:
    //upper row and right column

    int off1=0;
    int off2=3+4*i;
    for(int l1 = 0; l1<3;++l1) 
      for(int l2 = 0; l2<4;++l2)
	reduced(off1+l1,off2+l2) = transform(3+l1,6+7*i +l2);

    //diagonal elements
    off1=off2=3+4*i;
    for(int l1 = 0; l1<4;++l1)
      for(int l2 = 0; l2<4;++l2)
	reduced(off1+l1,off2+l2) = transform(6+7*i+l1, 6+7*i+l2);

     //off diagonal elements
   for(int j = 1; j<size; j++) {
      off1 =  3+4*(i-1); off2=3+4*j;
      for(int l1 = 0; l1<4;++l1)
	for(int l2 = 0; l2<4;++l2)
	  reduced(off1+l1,off2+l2)  = transform(6+7*(i-1)+l1,6+7*j+l2);
    }

  }
    
  AlgebraicSymMatrix77 ret;
  ROOT::Math::AssignSym::Evaluate(ret, jac_t*reduced*ROOT::Math::Transpose(jac_t));
  return ret;
}







#endif
