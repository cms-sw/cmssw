#include "RecoVertex/KinematicFit/interface/FinalTreeBuilder.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicRefittedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicStatePropagator.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
//#include "Vertex/KinematicFitPrimitives/interface/KinematicLinearizedTrackState.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

FinalTreeBuilder::FinalTreeBuilder()
{
 kvFactory = new KinematicVertexFactory();
 KinematicStatePropagator * ksp = 0;
 pFactory = new VirtualKinematicParticleFactory(ksp);
}
 
FinalTreeBuilder::~FinalTreeBuilder()
{ 
 delete kvFactory;
 delete pFactory;
}

RefCountedKinematicTree FinalTreeBuilder::buildTree(const CachingVertex<6>& vtx, 
                             vector<RefCountedKinematicParticle> input) const
{
//creating resulting kinematic particle
 AlgebraicVector7 par;
 AlgebraicMatrix cov(7,7,0);
 par(0) = vtx.position().x();
 par(1) = vtx.position().y();
 par(2) = vtx.position().z();
 double en = 0.;
 int ch = 0; 
 
//new particle momentum calculation and refitted kinematic states 
 vector<KinematicRefittedTrackState *> rStates;
 vector<RefCountedVertexTrack> refTracks = vtx.tracks();
 for(vector<RefCountedVertexTrack>::const_iterator i = refTracks.begin();i !=refTracks.end();++i)
 {
  KinematicRefittedTrackState * rs = dynamic_cast<KinematicRefittedTrackState *>(&(*((*i)->refittedState())));
  AlgebraicVector4 f_mom = rs->kinematicMomentumVector();
  par(3) += f_mom(0);
  par(4) += f_mom(1);
  par(5) += f_mom(2);
  en += sqrt(f_mom(1)*f_mom(1)+f_mom(2)*f_mom(2)+f_mom(3)*f_mom(3) + f_mom(0)*f_mom(0));  
  ch += (*i)->linearizedTrack()->charge();
  rStates.push_back(rs);
 } 

//math precision check (numerical stability) 
 double differ = en*en - (par(3)*par(3)+par(4)*par(4)+par(5)*par(5));
 if(differ>0.)
 {
  par(6) = sqrt(differ); 
 }else{ 
   LogDebug("FinalTreeBuilder")
	<< "Fit failed: Current precision does not allow to calculate the mass\n";
   return ReferenceCountingPointer<KinematicTree>(new KinematicTree());
 }

// covariance matrix calculation: momentum-momentum components part (4x4)
// and position-momentum components part:
 AlgebraicMatrix m_all = momentumPart(rStates,vtx,par);
 
//position-position components part (3x3) 
 AlgebraicMatrix x_X = vtx.error().matrix();

//making new matrix itself 
 cov.sub(1,1,m_all);

//covariance sym matrix 
 AlgebraicSymMatrix77 sCov;
 for(int i = 1; i<8; i++)
 {
  for(int j = 1; j<8; j++)
  {
   if(i<=j) sCov(i-1,j-1) = cov(i,j);
  }
 } 
 
//valid decay vertex for our resulting particle 
 RefCountedKinematicVertex dVrt = kvFactory->vertex(vtx);
 
//invalid production vertex for our resulting particle 
 RefCountedKinematicVertex pVrt = kvFactory->vertex(); 
 
//new born kinematic particle 
 KinematicParameters kPar(par);
 KinematicParametersError kEr(sCov);  
 const MagneticField* field=input.front()->magneticField();
 KinematicState nState(kPar, kEr, ch, field);
 
//invalid previous particle and empty constraint:
 KinematicParticle * zp = 0;
 RefCountedKinematicParticle pPart = ReferenceCountingPointer<KinematicParticle>(zp);
 
 float vChi = vtx.totalChiSquared();
 float vNdf = vtx.degreesOfFreedom();
 RefCountedKinematicParticle nPart = pFactory->particle(nState, vChi, vNdf, pPart);
 
//adding top particle to the tree 
 RefCountedKinematicTree resTree = ReferenceCountingPointer<KinematicTree>(new KinematicTree());
 resTree->addParticle(pVrt,dVrt,nPart);
 
//making refitted kinematic particles and putting them to the tree
 vector<RefCountedKinematicParticle> rrP;

 vector<RefCountedKinematicParticle>::const_iterator j;
 vector<RefCountedVertexTrack>::const_iterator i;
 for(j=input.begin(), i=refTracks.begin(); j !=input.end(), i !=refTracks.end();++j, ++i)
 {
  RefCountedLinearizedTrackState lT = (*i)->linearizedTrack();
  KinematicRefittedTrackState * rS= dynamic_cast<KinematicRefittedTrackState *>(&(*((*i)->refittedState())));

//   RefCountedRefittedTrackState rS = (*i)->refittedState();
  AlgebraicVector7 lPar = rS->kinematicParameters();
  KinematicParameters lkPar(lPar);
  AlgebraicSymMatrix77 lCov = rS->kinematicParametersCovariance();
  KinematicParametersError lkCov(lCov);
  TrackCharge lch = lT->charge();
  KinematicState nState(lkPar,lkCov,lch, field);
  RefCountedKinematicParticle nPart = (*j)->refittedParticle(nState,vChi,vNdf);
  rrP.push_back(nPart);
  if((*j)->correspondingTree() != 0)
  {
  
//here are the particles having trees after them 
   KinematicTree * tree = (*j)->correspondingTree();
   tree->movePointerToTheTop();
   tree->replaceCurrentParticle(nPart);
   RefCountedKinematicVertex cdVertex = resTree->currentDecayVertex();
   resTree->addTree(cdVertex, tree);   
  }else{  
  
//here are just particles fitted to this tree  
   RefCountedKinematicVertex nV = kvFactory->vertex();
   resTree->addParticle(dVrt,nV,nPart);
  }
 } 
 return resTree;
}

//method returning the full covariance matrix
//of new born kinematic particle
AlgebraicMatrix FinalTreeBuilder::momentumPart(vector<KinematicRefittedTrackState *> rStates,
                                               const CachingVertex<6>& vtx, const AlgebraicVector7& par)const
{ 
 vector<RefCountedVertexTrack> refTracks  = vtx.tracks();
 int size = rStates.size();
 AlgebraicMatrix cov(7+4*(size-1),7+4*(size-1));
 AlgebraicMatrix jac(7,7+4*(size-1));
 jac(1,1) = 1.;
 jac(2,2) = 1.;
 jac(3,3) = 1.;
 vector<KinematicRefittedTrackState *>::const_iterator rs;
 vector<RefCountedVertexTrack>::const_iterator rt_i;
 int i_int = 0;
 for(rs = rStates.begin(), rt_i = refTracks.begin(); rs != rStates.end(), rt_i != refTracks.end(); rs++, rt_i++)
 {
  AlgebraicMatrix jc_el(4,4,0);
  jc_el(1,1) = 1.;
  jc_el(2,2) = 1.;
  jc_el(3,3) = 1.;

//non-trival elements: mass correlations: 
  AlgebraicVector7 l_Par = (*rs)->kinematicParameters();
  double energy_local  = sqrt(l_Par(3)*l_Par(3) + l_Par(4)*l_Par(4) + l_Par(5)*l_Par(5) + l_Par(6)*l_Par(6));

  double energy_global = sqrt(par(3)*par(3)+par(6)*par(6) + par(5)*par(5)+par(4)*par(4));
  
  jc_el(4,4) = energy_global*l_Par(6)/(par(6)*energy_local);
  
  jc_el(4,1) = ((energy_global*l_Par(3)/energy_local) - par(3))/par(6);
  jc_el(4,2) = ((energy_global*l_Par(4)/energy_local) - par(4))/par(6);
  jc_el(4,3) = ((energy_global*l_Par(5)/energy_local) - par(5))/par(6);
  
  jac.sub(4,i_int*4+4,jc_el);
  
//top left corner elements  
  if(i_int == 0)
  { 
   cov.sub(1,1,asHepMatrix<7>((*rs)->kinematicParametersCovariance()) );
  }else{
//4-momentum corellatons: diagonal elements of the matrix
   AlgebraicMatrix m_m_cov = asHepMatrix<7>((*rs)->kinematicParametersCovariance()).sub(4,7);

//position momentum and momentum position corellations
   AlgebraicMatrix xpcov = asHepMatrix<7>((*rs)->kinematicParametersCovariance());  
   AlgebraicMatrix p_x_cov(4,3);
   AlgebraicMatrix x_p_cov(3,4);
   
   for(int l1 = 1; l1<5; ++l1)
   {
    for(int l2 = 1; l2<4; ++l2)
    {p_x_cov(l1,l2) = xpcov(3+l1,l2);}
   }
  
   for(int l1 = 1; l1<4; ++l1)
   {
    for(int l2 = 1; l2<5; ++l2)
    {x_p_cov(l1,l2) = xpcov(l1,3+l2);}
   }
  
    
//   AlgebraicMatrix  p_x_cov = xpcov.sub(4,7,1,3);
//   AlgebraicMatrix  x_p_cov = xpcov.sub(1,3,4,7);
  
//here the clhep must be worken around
//  cout<<"p_x_cov"<< p_x_cov<<endl;
//  cout<<"x_p_cov"<< x_p_cov<<endl; 
   
//putting everything to the joint covariance matrix:
//diagonal momentum-momentum elements:   
   cov.sub(i_int*4 + 4, i_int*4 + 4,m_m_cov);
   
//position momentum elements:   
   cov.sub(1,i_int*4 + 4,x_p_cov);
   cov.sub(i_int*4 + 4,1,p_x_cov);
    
//off diagonal elements: corellations
// track momentum - track momentum
  }
   int j_int = 0;
   for(vector<RefCountedVertexTrack>::const_iterator rt_j = refTracks.begin(); rt_j != refTracks.end(); rt_j++)
   {
    if(i_int < j_int)
    {
     AlgebraicMatrix i_k_cov_m = asHepMatrix<4,4>(vtx.tkToTkCovariance((*rt_i),(*rt_j)));
//     cout<<"i_k_cov_m"<<i_k_cov_m <<endl;
     cov.sub(i_int*4 + 4, j_int*4 + 4,i_k_cov_m);
     cov.sub(j_int*4 + 4, i_int*4 + 4,i_k_cov_m);
    } 
    j_int++;
   }
  i_int++;
 }
// cout<<"jac"<<jac<<endl;
// cout<<"cov"<<cov<<endl;
 
 return jac*cov*jac.T();
}
