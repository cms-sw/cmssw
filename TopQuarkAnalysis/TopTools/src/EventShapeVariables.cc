#include "TopQuarkAnalysis/TopTools/interface/EventShapeVariables.h"

using namespace std;

EventShapeVariables::EventShapeVariables():
  nStep_(100), sph_(-999.), apl_(-999.),cir_(-999.), iso_(-999.)
{
}

double 
EventShapeVariables::sphericity(const std::vector<TVector3>& p)
{
  // Description: S=1.5*(Q1+Q2) where 0<=Q1<=Q2<=Q3 are the eigenvalues of
  //              the momemtum tensor S_ab = sum{p_j[a]*p_j[b]}/sum{p_j**2}
  //              normalized to 1. Returns: S=1 for spherical, S=3/4 for
  //              plane and S=0 for linear events.
  double sph1;
  double sph2;
  TVectorD eVal(3);
  TMatrixDSym pT=momentumTensor(p);
  if(pT.IsSymmetric()){
    if(pT.NonZeros()!=0) 
      pT.EigenVectors(eVal);
  }
  sph1=eVal(0);
  sph2=eVal(1);
  if(eVal(2)<sph1) sph1=eVal(2);
  else if(eVal(2)<sph2) sph2=eVal(2);
  sph_=1.5*(sph1+sph2);
  return sph_;
}

double 
EventShapeVariables::aplanarity(const std::vector<TVector3>& p)
{
  // Description: A=1.5*Q1 where 0<=Q1<=Q2<=Q3 are the eigenvalues of the 
  //              momemtum tensor S_ab = sum{p_j[a]*p_j[b]}/sum{p_j**2}
  //              normalized to 1. Returns: A=0.5 for spherical and A=0 
  //              for plane and linear events.
  double apl;
  TVectorD eVal(3);
  TMatrixDSym pT=momentumTensor(p);
  if(pT.IsSymmetric()){
    if(pT.NonZeros()!=0)
      pT.EigenVectors(eVal);
  }
  apl=eVal(0);
  if( eVal(1)<apl ) apl=eVal(1);
  else if( eVal(1)<apl ) apl=eVal(2);
  apl_=1.5*apl;
  return apl_;
}

TMatrixDSym 
EventShapeVariables::momentumTensor(const std::vector<TVector3>& p)
{
  double sumP2=0;
  TMatrixDSym pT(3);
  pT.Zero();
  if(p.size()<2){
    // error: too small number of 
    // momentum vectors -> return 
    // empty matrix...
    return pT;
  }
  for(int i=0; i<(int)p.size(); ++i){
    sumP2 += p[i]*p[i];
    for(int m=0; m<3; ++m){
      for(int n=0; n<=m; ++n){
	pT(m,n)+=(p[i])[m]*(p[i])[n];
	if(n!=m) pT(n,m)=pT(m,n);
      }
    }
  }
  pT*=1/sumP2;
  return pT;
}

double 
EventShapeVariables::circularity(const std::vector<TVector3>& p)
{
  // Description: Returns: C=1 for spherical and C=0 
  //              linear events in R-Phi.
  const double del=2*TMath::Pi()/nStep_;
  double phi=0;
  for(int i=0; i<nStep_; ++i){
    phi+=del;
    double sum=0;
    for(int j=0; j<(int)p.size(); ++j){
      sum+=TMath::Abs(TMath::Cos(phi)*p[j].X()+TMath::Sin(phi)*p[j].Y());
    }
    if( cir_<0 || sum<cir_ ) cir_=sum;
  }
  return cir_;
}

double 
EventShapeVariables::isotropy(const std::vector<TVector3>& p)
{
  // Description: Returns: C=1 for spherical and C=0
  //              linear events in R-Phi.
  const double del=2*TMath::Pi()/nStep_;
  double phi = 0;
  double eIn =-1.;
  double eOut=-1.;
  for(int i=0; i<nStep_; ++i){
    phi+=del;
    double sum=0;
    for(int j=0; j<(int)p.size(); ++j){
      // sum over inner product of unit vector and momenta
      sum+=TMath::Abs(TMath::Cos(phi)*p[j].X()+TMath::Sin(phi)*p[j].Y());
    }
    if( eOut<0. || sum<eOut ) eOut=sum;
    if( eIn <0. || sum>eIn  ) eIn =sum;
  }
  iso_=(eIn-eOut)/eIn;
  return iso_;
}
