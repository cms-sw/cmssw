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
                             const std::vector<RefCountedKinematicParticle> &input) const
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
 std::vector<KinematicRefittedTrackState *> rStates;
 std::vector<RefCountedVertexTrack> refTracks = vtx.tracks();
 for(std::vector<RefCountedVertexTrack>::const_iterator i = refTracks.begin();i !=refTracks.end();++i)
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
 AlgebraicMatrix m_all = momentumPart(vtx,par);

//position-position components part (3x3)
 // AlgebraicMatrix x_X = vtx.error().matrix();

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
 std::vector<RefCountedKinematicParticle> rrP;

 std::vector<RefCountedKinematicParticle>::const_iterator j;
 std::vector<RefCountedVertexTrack>::const_iterator i;
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
AlgebraicMatrix FinalTreeBuilder::momentumPart(const CachingVertex<6>& vtx,
	const AlgebraicVector7& par) const
{
 std::vector<RefCountedVertexTrack> refTracks  = vtx.tracks();
 int size = refTracks.size();
 AlgebraicMatrix cov(7+4*(size-1),7+4*(size-1));
 AlgebraicMatrix jac(7,7+4*(size-1));
 jac(1,1) = 1.;
 jac(2,2) = 1.;
 jac(3,3) = 1.;

 double energy_total = sqrt(par(3)*par(3)+par(6)*par(6) + par(5)*par(5)+par(4)*par(4));

 std::vector<KinematicRefittedTrackState *>::const_iterator rs;
 std::vector<RefCountedVertexTrack>::const_iterator rt_i;
 int i_int = 0;
 for(rt_i = refTracks.begin(); rt_i != refTracks.end(); rt_i++)
 {
  double a;
  AlgebraicVector6 param = (**rt_i).refittedState()->parameters(); // rho, theta, phi,tr_im, z_im, mass
  double rho = param[0];
  double theta = param[1];
  double phi = param[2];
  double mass = param[5];

  if ((**rt_i).linearizedTrack()->charge()!=0) {
      a = -(**rt_i).refittedState()->freeTrajectoryState().parameters().magneticFieldInInverseGeV(vtx.position()).z()
      		* (**rt_i).refittedState()->freeTrajectoryState().parameters ().charge();
    if (a==0.) throw cms::Exception("FinalTreeBuilder", "Field is 0");
  } else {
    a = 1;
  }

  AlgebraicMatrix jc_el(4,4,0);
  jc_el(1,1) = -a*cos(phi)/(rho*rho); //dpx/d rho
  jc_el(2,1) = -a*sin(phi)/(rho*rho); //dpy/d rho
  jc_el(3,1) = -a/(rho*rho*tan(theta)); //dpz/d rho

  jc_el(3,2) = -a/(rho*sin(theta)*sin(theta)); //dpz/d theta

  jc_el(1,3) = -a*sin(phi)/rho; //dpx/d phi
  jc_el(2,3) = a*cos(phi)/rho; //dpy/d

//non-trival elements: mass correlations:
  double energy_local  = sqrt(a*a/(rho*rho)*(1+1/(tan(theta)*tan(theta)))  + mass*mass);

  jc_el(4,4) = energy_total*mass/(par(6)*energy_local); // dm/dm

  jc_el(4,1) = (-(energy_total/energy_local*a*a/(rho*rho*rho*sin(theta)*sin(theta)) )
  		+ par(3)*a/(rho*rho)*cos(phi) + par(4)*a/(rho*rho)*sin(phi)
		+ par(5)*a/(rho*rho*tan(theta)) )/par(6);	//dm / drho

  jc_el(4,2) = (-(energy_total/energy_local*a*a/(rho*rho*sin(theta)*sin(theta)*tan(theta)) )
		+ par(5)*a/(rho*sin(theta)*sin(theta)) )/par(6);//dm d theta

  jc_el(4,3) = ( par(3)*sin(phi) - par(4)*cos(phi) )*a/(rho*par(6));	//dm/dphi

  jac.sub(4,i_int*4+4,jc_el);

//top left corner elements
  if(i_int == 0) {
   cov.sub(1,1,asHepMatrix<7>((**rt_i).fullCovariance()));
  } else {
//4-momentum corellatons: diagonal elements of the matrix
   AlgebraicMatrix fullCovMatrix(asHepMatrix<7>((**rt_i).fullCovariance()));
   AlgebraicMatrix m_m_cov = fullCovMatrix.sub(4,7,4,7);
   AlgebraicMatrix x_p_cov = fullCovMatrix.sub(1,3,4,7);
   AlgebraicMatrix p_x_cov = fullCovMatrix.sub(4,7,1,3);

// cout << "Full covariance: \n"<< (**rt_i).fullCovariance()<<endl;
// cout << "Full m_m_cov: "<< m_m_cov<<endl;
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
   for(std::vector<RefCountedVertexTrack>::const_iterator rt_j = refTracks.begin(); rt_j != refTracks.end(); rt_j++)
   {
    if(i_int < j_int)
    {
     AlgebraicMatrix i_k_cov_m = asHepMatrix<4,4>(vtx.tkToTkCovariance((*rt_i),(*rt_j)));
//     cout<<"i_k_cov_m"<<i_k_cov_m <<endl;
     cov.sub(i_int*4 + 4, j_int*4 + 4,i_k_cov_m);
     cov.sub(j_int*4 + 4, i_int*4 + 4,i_k_cov_m.T());
    }
    j_int++;
   }
  i_int++;
 }
// cout<<"jac"<<jac<<endl;
// cout<<"cov"<<cov<<endl;
//  cout << "final result new"<<jac*cov*jac.T()<<endl;

 return jac*cov*jac.T();
}

