// -------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DcxFittedHel.cc,v 1.18 2004/08/06 05:58:21 bartoldu Exp $
//
// Description:
//	Class Implementation for |DcxFittedHel|
//
// Environment:
//	Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//	S. Wagner
//
// Copyright Information:
//	Copyright (C) 1995	SLAC
//
//------------------------------------------------------------------------
//babar #include "BaBar/BaBar.hh"
//babar #include <math.h>

//babar #include "DcxReco/Dcxmatinv.hh"
//babar #include "DcxReco/DcxFittedHel.hh"
//babar #include "DcxReco/DcxHit.hh"
//babar #include "DcxReco/Dcxprobab.hh"
#include <cmath>
#include "RecoTracker/RoadSearchHelixMaker/interface/Dcxmatinv.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxFittedHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHit.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/Dcxprobab.hh"

using std::cout;
using std::endl;
using std::ostream;

void DcxFittedHel::basics() 
{nhits=0; itofit=0; fittime=0.0;
prob=0.0; chisq=1000000.0; fail=1300; quality=0; origin=-1; usedonhel=0;
bailout=1; chidofbail=1000.0; niter=10;
} // endof basics 

//babar void DcxFittedHel::basics(const HepAList<DcxHit> &e) {
void DcxFittedHel::basics(const std::vector<DcxHit*> &e) {
  basics();
  nhits=e.size();
  listohits=e;
  origin=OriginIncluded();
} // endof basics

//constructors
DcxFittedHel::DcxFittedHel(){basics();}

//points+guess
//babar DcxFittedHel::DcxFittedHel(HepAList<DcxHit> &ListOHits, DcxHel& hel, double Sfac)
DcxFittedHel::DcxFittedHel(std::vector<DcxHit*> &ListOHits, DcxHel& hel, double Sfac)
{ 
//  float tstart=clock();
  basics(ListOHits);
  sfac=Sfac;
  *this=hel;
  fail=IterateFit();
//  float tstop=clock();
//  fittime=tstop-tstart;
}//endof DcxFittedHel

//destructor
DcxFittedHel::~DcxFittedHel( ){ }//endof ~DcxFittedHel

//operators
DcxFittedHel& DcxFittedHel::operator=(const DcxHel& rhs){
  copy(rhs);
  return *this;
} //endof DcxFittedHel::operator=

DcxFittedHel& DcxFittedHel::operator=(const DcxFittedHel& rhs){
 copy(rhs);
 fail=rhs.Fail();
 chisq=rhs.Chisq();
 rcs=rhs.Rcs();
 prob=rhs.Prob();
 fittime=rhs.Fittime();
 nhits=rhs.Nhits();
 itofit=rhs.Itofit();
 quality=rhs.Quality();
 origin=rhs.Origin();
 listohits=rhs.ListOHits();
 sfac=rhs.Sfac();               
 usedonhel=rhs.GetUsedOnHel();
 bailout=1; chidofbail=1000.0; niter=10;
 return *this;
}//endof DcxFittedHel::operator=

//babar DcxFittedHel& DcxFittedHel::Grow(const DcxFittedHel& rhs, HepAList<DcxHit> &ListOAdds){
DcxFittedHel& DcxFittedHel::Grow(const DcxFittedHel& rhs, std::vector<DcxHit*> &ListOAdds){
 copy(rhs);
 fail=rhs.Fail();
 chisq=rhs.Chisq();
// rcs=rhs.Rcs();
// prob=rhs.Prob();
 fittime=0.0;
 nhits=rhs.Nhits();
 itofit=0;
 quality=rhs.Quality();
 origin=rhs.Origin();
 listohits=rhs.ListOHits();
 sfac=rhs.Sfac();               
 usedonhel=rhs.GetUsedOnHel();
 bailout=1; chidofbail=1000.0; niter=10;
 int kkk=0; while (ListOAdds[kkk]){ListOAdds[kkk]->SetUsedOnHel(0); kkk++;}
 kkk=0; while (listohits[kkk]){listohits[kkk]->SetUsedOnHel(1); kkk++;}
 double spull; DcxHel temp=rhs;
 kkk=0; while (ListOAdds[kkk]){
   if (ListOAdds[kkk]->GetUsedOnHel() == 0){
     spull=ListOAdds[kkk]->pull(temp)/sfac; chisq+=spull*spull; 
//babar     listohits.append(ListOAdds[kkk]); nhits++; 
     listohits.push_back(ListOAdds[kkk]); nhits++; 
   }
   kkk++;
 }
 int ndof=nhits-nfree;
 prob=Dcxprobab(ndof,chisq);
 rcs=chisq/ndof;
 return *this;
}//endof DcxFittedHel::Grow

//accessors
float DcxFittedHel::Residual(int i){
 float pull=listohits[i]->pull(*this);
 float E=listohits[i]->e();
 return pull*E;
}//endof Residual

float DcxFittedHel::Pull(int i){
 float pull=listohits[i]->pull(*this);
 return pull;
}//endof Pulls

int DcxFittedHel::Fail(float Probmin)const {
  if(fail) {return fail;}
  if(prob<Probmin) {return 1303;}
// now done in DoFit  if(fabs(omega)>omegmax) {return 1306;}
  return 0;
} // endof Fail

//utilities&workers

void DcxFittedHel::VaryRes() {
  int kbl=0; while (listohits[kbl]){listohits[kbl]->SetConstErr(0); kbl++;}
}

int DcxFittedHel::ReFit(){
 fail=IterateFit();
 return fail;
}//endof ReFit

int DcxFittedHel::IterateFit(){
  int ftemp=1301;		// not enough hits
  if(nfree>=nhits) {return ftemp;}
  ftemp=0;
  if(niter>=1) {
    float prevchisq=0.0;
    for (int i=0; i< niter; i++) {
      itofit=i+1;
      ftemp=DoFit();
//      if (nfree == 5){
//      cout << " iteration number= " << i  << " chisq= " << chisq;
//      cout << " nhits= " << nhits << " " << " fail= " << ftemp << endl;
//      }
//      print();
//      cout << "  " << endl;
      if(ftemp!=0) {break;}
      if(fabs(chisq-prevchisq)<0.01*chisq) {break;}
      prevchisq=chisq;
    }//endof iter loop
  }else{
    float prevchisq=0.0;
    chisq=1000000.0;
    int iter=0;
    while(fabs(chisq-prevchisq)>0.01) {
      iter++;
      prevchisq=chisq;
      ftemp=DoFit();
      if(ftemp!=0) break;
      if(iter>=1000) break;
    }//endof (fabs(chisq-oldchisq).gt.0.01)
  }//endof (niter>=1)
  int ndof=nhits-nfree;
  prob=Dcxprobab(ndof,chisq);
  rcs=chisq/ndof;
  return ftemp;
}//endof IterateFit

int DcxFittedHel::DoFit(){
 int ftemp=1301;
// if(nfree>nhits) {return Fail;}
 if(nfree>=nhits) {return ftemp;}
 double m_2pi=2.0*M_PI;
 ftemp=0;
 //pointloop
 int norder=nfree;
 double A[10][10], B[10], D[10], det; int ii, jj;
 for (ii=0; ii<norder; ii++){
  D[ii]=0.0; B[ii]=0.0; for (jj=0; jj<norder; jj++){A[ii][jj]=0.0;}
 }
 chisq=0.0;
 for (int i=0; i< nhits; i++){
  std::vector<float> derivs=listohits[i]->derivatives(*this);
//  cout << derivs;
  if (sfac != 1.0){
   for(int ipar=0; ipar<derivs.size(); ipar++) {derivs[ipar]/=sfac;
//    cout << " " << derivs[ipar];
   }
//   cout << endl;
  }
  chisq+=derivs[0]*derivs[0];
  //outer parameter loop
  for(int ipar=0; ipar<norder; ipar++){
   D[ipar]+=derivs[0]*derivs[ipar+1];
   //inner parameter loop
   for(int jpar=0; jpar<norder; jpar++){
    A[ipar][jpar]+=derivs[ipar+1]*derivs[jpar+1];
   }//endof inner parameter loop
  }//endof outer parameter loop
 }//pointloop
// cout << " D A " << endl;
// for (ii=0; ii<norder; ii++){
//  cout << D[ii] << " "; for (jj=0;jj<norder;jj++){cout << A[ii][jj] << " ";}
//  cout << endl;
// }
//invert A
  int ierr;
  if(bailout) {
    ftemp=1308;			// bailout
    int ndof=nhits-nfree;
    if(ndof>0) {
      float chiperdof=chisq/ndof;
      if(chiperdof>chidofbail) {return ftemp;}
//      cout << " got here; chiperdof = " << chiperdof << endl;
    } // (ndof>0)
  } // (bailout)
  ftemp=0;
  ierr = Dcxmatinv(&A[0][0],&norder,&det);
//  cout << " ierr = " << ierr << endl;
  if(ierr==0) {
   for(ii=0;ii<norder;ii++){for(jj=0;jj<norder;jj++){B[ii]+=A[ii][jj]*D[jj];}}
//   cout << " B A " << endl;
//   for (ii=0; ii<norder; ii++){
//    cout << B[ii] << " "; for(jj=0;jj<norder;jj++){cout << A[ii][jj] << " ";}
//    cout << endl;
//   }
   int bump=-1;
   if(qd0)    {bump++; d0-=B[bump];}
   if(qphi0)  {bump++; phi0-=B[bump]; 
               if (phi0 > M_PI){phi0-=m_2pi;} 
               if (phi0 < -M_PI){phi0+=m_2pi;}
               cphi0=cos(phi0); sphi0=sin(phi0);
              }
   if(qomega) {bump++; omega-=B[bump];
               ominfl=1; if(fabs(omega)<omin){ominfl=0;}
              }
   if(qz0)    {bump++; z0-=B[bump];}
   if(qtanl)  {bump++; tanl-=B[bump];}
   if(qt0)    {bump++; t0-=B[bump];}
   x0=X0(); y0=Y0(); xc=Xc(); yc=Yc();
   if ( fabs(d0) > 80.0 )ftemp=1305; // No longer in Dch
   if ( fabs(omega) > 1.0 )ftemp=1306; // Too tight (r < 1 cm)
  }else{
//   Fail=Fail+ierr;
   ftemp=ierr;
  }
  return ftemp;
}//endof DoFit

//is origin included in fit ?
int DcxFittedHel::OriginIncluded() {
  for(int i=0; listohits[i]; i++) {
    int type=listohits[i]->type();
    if(2==type) {		// origin "hit" ?
      //move to end, move fit point, return hit number
//cms??      listohits.swap(i,nhits-1);
      return nhits-1;
    } // (2==type)
  } // (int i=0; listohits[i]; i++)
  return -1;
}//endof OriginIncluded

int DcxFittedHel::FitPrint(){
 cout << " fail= " << fail << endl;
 cout << " iterations to fit= " << itofit << endl;
 cout << " nhits= " << nhits << endl;
 cout << " sfac= " << sfac << endl;
 cout << " chisq= " << chisq << endl;
 cout << " rcs= " << rcs << endl;
 cout << " prob= " << prob << endl;
 cout << " fittime= " << fittime << endl;
 return 0;
}//endof FitPrint

int DcxFittedHel::FitPrint(DcxHel &hel, ostream &o){
// FitPrint();
// cout << " difd0= " << d0-hel.D0() << endl;
 double m_2pi=2.0*M_PI;
 double difphi0=phi0-hel.Phi0();
 if (difphi0>M_PI)difphi0-=m_2pi; if (difphi0<-M_PI)difphi0+=m_2pi; 
// cout << " difphi0= " << difphi0 << endl;
// cout << " difomega= " << omega-hel.Omega() << endl;
// cout << " difz0= " << z0-hel.Z0() << endl;
// cout << " diftanl= " << tanl-hel.Tanl() << endl;
 o << "*&*" << " ";
// o << itofit << " " << fail << " " << chisq << " " << rcs << " "; 
 o << nhits << " " << fail << " " << chisq << " " << rcs << " "; 
// o << prob << " " << fittime << " " << d0-hel.D0() << " ";
 o << prob << " " << hel.Tanl() << " " << d0-hel.D0() << " ";
 o << difphi0 << " " << omega-hel.Omega() << " "; 
 o << z0-hel.Z0() << " " << tanl-hel.Tanl() << endl;
 return 0;
}//endof FitPrint

//Find layer number of |hitno|
int DcxFittedHel::Layer(int hitno)const {
  if(hitno>=nhits) {return 0;}
//babar  const HepAList<DcxHit> &temp=(HepAList<DcxHit>&)listohits;
  const std::vector<DcxHit*> &temp=(std::vector<DcxHit*>&)listohits;
  int layer=temp[hitno]->Layer();
  return layer;
} // endof Layer

//Find superlayer numbber of |hitno|
int DcxFittedHel::SuperLayer(int hitno)const {
    if(hitno>=nhits) {return 0;}
    if(hitno<0) {return 0;}
//babar    const HepAList<DcxHit> &temp=(HepAList<DcxHit>&)listohits;
    const std::vector<DcxHit*> &temp=(std::vector<DcxHit*>&)listohits;
    return temp[hitno]->SuperLayer();
  } // endof SuperLayer

