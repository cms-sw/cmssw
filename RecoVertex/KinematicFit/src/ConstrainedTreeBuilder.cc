#include "RecoVertex/KinematicFit/interface/ConstrainedTreeBuilder.h"
#include "DataFormats/CLHEP/interface/Migration.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ConstrainedTreeBuilder::ConstrainedTreeBuilder()
{
 pFactory = new VirtualKinematicParticleFactory();
 vFactory = new KinematicVertexFactory();
}

ConstrainedTreeBuilder::~ConstrainedTreeBuilder()
{
 delete pFactory;
 delete vFactory;
}


RefCountedKinematicTree ConstrainedTreeBuilder::buildTree(const std::vector<RefCountedKinematicParticle> & initialParticles,
                         const std::vector<KinematicState> & finalStates,
			 const RefCountedKinematicVertex vertex, const AlgebraicMatrix& fullCov) const
{
  if (!vertex->vertexIsValid()) {
	  LogDebug("ConstrainedTreeBuilder")
	<< "Vertex is invalid\n";
       return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
  }
  AlgebraicVector3 vtx;
  vtx(0) = vertex->position().x();
  vtx(1) = vertex->position().y();
  vtx(2) = vertex->position().z();
  AlgebraicMatrix33 vertexCov = asSMatrix<3,3>(fullCov.sub(1,3,1,3));

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
  aMatrix(3,3) = 1;
  aMatrix(4,4) = 1;
  aMatrix(5,5) = 1;
  aMatrix(6,6) = 1;
  ROOT::Math::SMatrix<double,7,3,ROOT::Math::MatRepStd<double,7,3> > bMatrix;
  bMatrix(0,0) = 1;
  bMatrix(1,1) = 1;
  bMatrix(2,2) = 1;
  AlgebraicMatrix77 trackParCov;
  ROOT::Math::SMatrix<double,3,7,ROOT::Math::MatRepStd<double,3,7> > vtxTrackCov;
  AlgebraicMatrix77 nCovariance;

  std::vector<RefCountedKinematicParticle>::const_iterator i = initialParticles.begin();
  std::vector<KinematicState>::const_iterator iStates = finalStates.begin();
  std::vector<RefCountedKinematicParticle> rParticles;
  int n=0;
  for( ; i != initialParticles.end(), iStates != finalStates.end(); ++i,++iStates)
  {
    AlgebraicVector7 p = iStates->kinematicParameters().vector();
    double a = - iStates->particleCharge() *
	iStates->magneticField()->inInverseGeV(iStates->globalPosition()).z();

    aMatrix(4,0) = -a;
    aMatrix(3,1) = a;
    bMatrix(4,0) = a;
    bMatrix(3,1) = -a;

    AlgebraicVector7 par = aMatrix*p + bMatrix * vtx;

    trackParCov = asSMatrix<7,7>(fullCov.sub(4+n*7,10+n*7,4+n*7,10+n*7));
    vtxTrackCov = asSMatrix<3,7>(fullCov.sub(1,3,4+n*7,10+n*7));;
    nCovariance = aMatrix * trackParCov * ROOT::Math::Transpose(aMatrix) +
	aMatrix * ROOT::Math::Transpose(vtxTrackCov) * ROOT::Math::Transpose(bMatrix) +
	bMatrix * vtxTrackCov * ROOT::Math::Transpose(aMatrix)+
	bMatrix * vertexCov * ROOT::Math::Transpose(bMatrix);

    KinematicState stateAtVertex(KinematicParameters(par),
	KinematicParametersError(AlgebraicSymMatrix77(nCovariance.LowerBlock())),
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

  AlgebraicMatrix77 cov = asSMatrix<7,7>(covarianceMatrix(rParticles,virtualPartPar,fullCov));

  KinematicState nState(KinematicParameters(virtualPartPar),
	KinematicParametersError(AlgebraicSymMatrix77(cov.LowerBlock())),
	charge, initialParticles[0]->magneticField());

 //newborn kinematic particle
  float chi2 = vertex->chiSquared();
  float ndf = vertex->degreesOfFreedom();
  KinematicParticle * zp = 0;
  RefCountedKinematicParticle virtualParticle = pFactory->particle(nState,chi2,ndf,zp);

  return buildTree(virtualParticle, vertex, rParticles);
}


RefCountedKinematicTree ConstrainedTreeBuilder::buildTree(const RefCountedKinematicParticle virtualParticle,
	const RefCountedKinematicVertex vtx, const std::vector<RefCountedKinematicParticle> & particles) const
{

//making a resulting tree:
 RefCountedKinematicTree resTree = ReferenceCountingPointer<KinematicTree>(new KinematicTree());

//fake production vertex:
 RefCountedKinematicVertex fVertex = vFactory->vertex();
 resTree->addParticle(fVertex, vtx, virtualParticle);

//adding final state
 for(std::vector<RefCountedKinematicParticle>::const_iterator il = particles.begin(); il != particles.end(); il++)
 {
  if((*il)->previousParticle()->correspondingTree() != 0)
  {
   KinematicTree * tree = (*il)->previousParticle()->correspondingTree();
   tree->movePointerToTheTop();
   tree->replaceCurrentParticle(*il);
   RefCountedKinematicVertex cdVertex = resTree->currentDecayVertex();
   resTree->addTree(cdVertex, tree);
  }else{
   RefCountedKinematicVertex ffVertex = vFactory->vertex();
   resTree->addParticle(vtx,ffVertex,*il);
  }
 }
 return resTree;
}

AlgebraicMatrix ConstrainedTreeBuilder::covarianceMatrix(const std::vector<RefCountedKinematicParticle> &rPart,
                             const AlgebraicVector7& newPar, const AlgebraicMatrix& fitCov)const
{
//constructing the full matrix using the simple fact
//that we have never broken the order of tracks
// during our fit.

 int size = rPart.size();
//global propagation to the vertex position
//Jacobian is done for all the parameters together
 AlgebraicMatrix jac(3+7*size,3+7*size,0);
 jac(1,1) = 1;
 jac(2,2) = 1;
 jac(3,3) = 1;
 int i_int=0;
 for(std::vector<RefCountedKinematicParticle>::const_iterator i = rPart.begin(); i != rPart.end(); i++)
 {

//vertex position related components of the matrix
  double a_i = - (*i)->currentState().particleCharge() * (*i)->magneticField()->inInverseGeV((*i)->currentState().globalPosition()).z();

  AlgebraicMatrix upper(3,7,0);
  AlgebraicMatrix diagonal(7,7,0);
  upper(1,1) = 1;
  upper(2,2) = 1;
  upper(3,3) = 1;
  upper(2,4) = -a_i;
  upper(1,5) = a_i;
  jac.sub(1,4+i_int*7,upper);

  diagonal(4,4) = 1;
  diagonal(5,5) = 1;
  diagonal(6,6) = 1;
  diagonal(7,7) = 1;
  diagonal(2,4) = a_i;
  diagonal(1,5) = -a_i;
  jac.sub(4+i_int*7,4+i_int*7,diagonal);
  i_int++;
 }

// jacobian is constructed in such a way, that
// right operation for transformation will be
// fitCov.similarityT(jac)
// WARNING: normal similarity operation is
// not valid in this case
//now making reduced matrix:
 int vSize = rPart.size();
 AlgebraicSymMatrix fit_cov_sym(7*vSize+3,0);
 for(int i = 1; i<7*vSize+4; ++i)
 {
  for(int j = 1; j<7*vSize+4; ++j)
  {if(i<=j) fit_cov_sym(i,j) = fitCov(i,j);}
 }


 AlgebraicMatrix reduced(3+4*size,3+4*size,0);
 AlgebraicMatrix transform = fit_cov_sym.similarityT(jac);

//jacobian to add matrix components
 AlgebraicMatrix jac_t(7,7+4*(size-1));
 jac_t(1,1) = 1.;
 jac_t(2,2) = 1.;
 jac_t(3,3) = 1.;

//debug code:
//CLHEP bugs: avoiding the
// HepMatrix::sub method use
 AlgebraicMatrix reduced_tm(7,7,0);
 for(int i = 1; i<8;++i)
 {
  for(int j =1; j<8; ++j)
  {reduced_tm(i,j)  = transform(i+3, j+3);}
 }

//left top corner
// reduced.sub(1,1,transform.sub(4,10,4,10));

//debug code:
//CLHEP bugs: avoiding the
// HepMatrix::sub method use
 reduced.sub(1,1,reduced_tm);

// cout<<"reduced matrix"<<reduced<<endl;
 int il_int = 0;
  double energy_global = sqrt(newPar(3)*newPar(3)+newPar(4)*newPar(4) + newPar(5)*newPar(5)+newPar(6)*newPar(6));
 for(std::vector<RefCountedKinematicParticle>::const_iterator rs = rPart.begin();
                                                       rs!=rPart.end();rs++)
 {
//jacobian components:
  AlgebraicMatrix jc_el(4,4,0);
  jc_el(1,1) = 1.;
  jc_el(2,2) = 1.;
  jc_el(3,3) = 1.;

//non-trival elements: mass correlations:
  AlgebraicVector7 l_Par = (*rs)->currentState().kinematicParameters().vector();
  double energy_local  = sqrt(l_Par(6)*l_Par(6) + l_Par(3)*l_Par(3) + l_Par(4)*l_Par(4) + l_Par(5)*l_Par(5));
  jc_el(4,4) = energy_global*l_Par(6)/(newPar(6)*energy_local);
  jc_el(4,1) = ((energy_global*l_Par(3)/energy_local) - newPar(3))/newPar(6);
  jc_el(4,2) = ((energy_global*l_Par(4)/energy_local) - newPar(4))/newPar(6);
  jc_el(4,3) = ((energy_global*l_Par(5)/energy_local) - newPar(5))/newPar(6);
  jac_t.sub(4,il_int*4+4,jc_el);
  il_int++;
 }
// cout<<"jac_t"<<jac_t<<endl;
//debug code
//CLHEP bugs workaround
// cout<<"Transform matrix"<< transform<<endl;

 for(int i = 1; i<size; i++)
 {

//non-trival elements: mass correlations:
//upper row and right column
  AlgebraicMatrix transform_sub1(3,4,0);
  AlgebraicMatrix transform_sub2(4,3,0);
  for(int l1 = 1; l1<4;++l1)
  {
   for(int l2 = 1; l2<5;++l2)
   {transform_sub1(l1,l2) = transform(3+l1,6+7*i +l2);}
  }

  for(int l1 = 1; l1<5;++l1)
  {
   for(int l2 = 1; l2<4;++l2)
   {transform_sub2(l1,l2) = transform(6+7*i+l1,3+l2);}
  }

  AlgebraicMatrix transform_sub3(4,4,0);
  for(int l1 = 1; l1<5;++l1)
  {
   for(int l2 = 1; l2<5;++l2)
   {transform_sub3(l1,l2) = transform(6+7*i+l1, 6+7*i+l2); }
  }

  reduced.sub(1,4+4*i,transform_sub1);
  reduced.sub(4+4*i,1,transform_sub2);

//diagonal elements
   reduced.sub(4+4*i,4+4*i,transform_sub3);

//off diagonal elements
  for(int j = 1; j<size; j++)
  {
   AlgebraicMatrix transform_sub4(4,4,0);
   AlgebraicMatrix transform_sub5(4,4,0);
  for(int l1 = 1; l1<5;++l1)
  {
   for(int l2 = 1; l2<5;++l2)
   {
    transform_sub4(l1,l2) = transform(6+7*(i-1)+l1,6+7*j+l2);
    transform_sub5(l1,l2) = transform(6+7*j+l1, 6+7*(i-1)+l2);
   }
  }
   reduced.sub(4+4*(i-1),4+4*j,transform_sub4);
   reduced.sub(4+4*j,4+4*(i-1),transform_sub5);
  }
 }

 return jac_t*reduced*jac_t.T();
}
