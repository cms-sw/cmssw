#include "RecoVertex/KinematicFit/interface/ConstrainedTreeBuilder.h"
#include "TrackingTools/TrajectoryState/interface/FakeField.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

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
  
RefCountedKinematicTree ConstrainedTreeBuilder::buildTree(vector<RefCountedKinematicParticle> prt,
                                 RefCountedKinematicVertex vtx,const AlgebraicMatrix& fCov) const
{

//now making a new virtual particle out of refitted states
 AlgebraicVector par(7,0);
 par(1) = vtx->position().x();
 par(2) = vtx->position().y();
 par(3) = vtx->position().z();
 double ent = 0.;
 int charge = 0;
 for(vector<RefCountedKinematicParticle>::iterator i = prt.begin(); i != prt.end(); i++)
 {
  charge += (*i)->currentState().particleCharge();
//  GlobalVector rMom = (*i)->currentState().globalMomentum();
//  ParticleMass mm = (*i)->currentState().mass();
//temporary hack
  
  GlobalVector rMom = (*i)->stateAtPoint(vtx->position()).globalMomentum();
  ParticleMass mm = (*i)->stateAtPoint(vtx->position()).mass();
  par(4) += rMom.x();
  par(5) += rMom.y();
  par(6) += rMom.z();
  ent += sqrt(mm*mm +rMom.x()*rMom.x() + rMom.y()*rMom.y() + rMom.z()*rMom.z());
 }
 
//total reconstructed mass
 double differ = ent*ent - (par(6)*par(6) + par(5)*par(5) + par(4)*par(4));
 if(differ>0.)
 {
  par(7) = sqrt(differ);
 }else{
  cout<<"warning! current precision does not allow to calculate the mass"<<endl;
  par(7) = 0.;
  throw VertexException("warning! current precision does not allow to calculate the mass");
  
 }
 
//now making covariance matrix: 

// cout<<"fCov"<<fCov<<endl;
 AlgebraicMatrix cov = momentumPart(prt,par,fCov);
// cout<<"Momentum part"<<cov<<endl;
 
//covariance sym matrix 
 AlgebraicSymMatrix sCov(7,0);
 for(int i = 1; i<8; i++)
 {
  for(int j = 1; j<8; j++)
  {if(i<=j) sCov(i,j) = cov(i,j);}
 }  
 KinematicParameters nP(par);
 KinematicParametersError nE(sCov);
 
 KinematicState nState(nP,nE,charge);
 
//new born kinematic particle 
 float chi2 = vtx->chiSquared();
 float ndf = vtx->degreesOfFreedom();
 KinematicParticle * zp = 0;
 RefCountedKinematicParticle pPart = ReferenceCountingPointer<KinematicParticle>(zp);
 RefCountedKinematicParticle nPart = pFactory->particle(nState,chi2,ndf,zp);
 
//making a resulting tree: 
 RefCountedKinematicTree resTree = ReferenceCountingPointer<KinematicTree>(new KinematicTree());
 
//fake production vertex: 
 RefCountedKinematicVertex fVertex = vFactory->vertex();
 resTree->addParticle(fVertex,vtx,nPart);
 
//adding final state 
 for(vector<RefCountedKinematicParticle>::const_iterator il = prt.begin(); il != prt.end(); il++)
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
 
AlgebraicMatrix ConstrainedTreeBuilder::momentumPart(vector<RefCountedKinematicParticle> rPart, 
                             const AlgebraicVector& newPar, const AlgebraicMatrix& fitCov)const
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
 for(vector<RefCountedKinematicParticle>::iterator i = rPart.begin(); i != rPart.end(); i++)
 {
 
//vertex position related components of the matrix 
  TrackCharge ch = (*i)->currentState().particleCharge();
  double field = TrackingTools::FakeField::Field::inInverseGeV((*i)->currentState().globalPosition()).z();
  double a_i = -0.29979246*ch*field;
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
 for(vector<RefCountedKinematicParticle>::const_iterator rs = rPart.begin();
                                                       rs!=rPart.end();rs++)
 {
//jacobian components: 
  AlgebraicMatrix jc_el(4,4,0);
  jc_el(1,1) = 1.;
  jc_el(2,2) = 1.;
  jc_el(3,3) = 1.;

//non-trival elements: mass correlations: 
  AlgebraicVector l_Par = (*rs)->currentState().kinematicParameters().vector();
  double energy_local  = sqrt(l_Par(7)*l_Par(7) + l_Par(4)*l_Par(4) + l_Par(5)*l_Par(5) + l_Par(6)*l_Par(6));

  double energy_global = sqrt(newPar(7)*newPar(7)+newPar(6)*newPar(6) + newPar(5)*newPar(5)+newPar(4)*newPar(4));
  
  jc_el(4,4) = energy_global*l_Par(7)/(newPar(7)*energy_local);
  jc_el(4,1) = ((energy_global*l_Par(4)/energy_local) - newPar(4))/newPar(7);
  jc_el(4,2) = ((energy_global*l_Par(5)/energy_local) - newPar(5))/newPar(7);
  jc_el(4,3) = ((energy_global*l_Par(6)/energy_local) - newPar(6))/newPar(7);

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
