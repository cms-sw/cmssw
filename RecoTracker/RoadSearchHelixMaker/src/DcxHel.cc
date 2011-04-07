// -------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DcxHel.cc,v 1.5 2009/05/27 07:17:25 fabiocos Exp $
//
// Description:
//	Class Implementation for |DcxHel|
//
// Environment:
//	Software developed for the BaBar Detector at the SLAC B-Factory.
//	Attempted port to CMSSW
//
// Author List:
//	S. Wagner
//
//------------------------------------------------------------------------
//babar #include "BaBar/BaBar.hh"
//babar #include "BaBar/Constants.hh"
//babar #include "BbrGeom/BbrAngle.hh"
//babar #include <math.h>
//babar #include "DcxReco/DcxHel.hh"
//babar #include "DcxReco/DcxHit.hh"
#include <cmath>
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHit.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using std::cout;
using std::endl;

double DcxHel::half_pi=1.570796327;
double DcxHel::pi     =3.141592654;
double DcxHel::twoPi  =6.283185307;
//babar double DcxHel::ktopt  =0.0045;
double DcxHel::ktopt  =0.0120; //cms

//constructors

DcxHel::DcxHel( ){ }

DcxHel::DcxHel
(double D0, double Phi0, double Omega, double Z0, double Tanl,
 double T0, int Code, int Mode, double X, double Y){ 
  omin=0.000005; omega=Omega; ominfl=1; if(fabs(omega)<omin){ominfl=0;}
  phi0=Phi0; d0=D0;
  if (phi0 > DcxHel::pi){phi0-=DcxHel::twoPi;} 
  if (phi0 < -DcxHel::pi){phi0+=DcxHel::twoPi;}
  z0=Z0; tanl=Tanl;
  xref=X; yref=Y; t0=T0;
  cphi0=cos(phi0); sphi0=sin(phi0);
  x0=X0(); y0=Y0(); xc=Xc(); yc=Yc();
  code=Code;
  decode(code,qd0,qphi0,qomega,qz0,qtanl,qt0,nfree);
  mode=Mode; turnflag=0;
}//endof DcxHel

DcxHel::DcxHel
(float D0, float Phi0, float Omega, float Z0, float Tanl,
 float T0, int Code, int Mode, float X, float Y){ 
  omin=0.000005; omega=Omega; ominfl=1; if(fabs(omega)<omin){ominfl=0;}
  phi0=Phi0; d0=D0;
  if (phi0 > DcxHel::pi){phi0-=DcxHel::twoPi;} 
  if (phi0 < -DcxHel::pi){phi0+=DcxHel::twoPi;}
  z0=Z0; tanl=Tanl;
  xref=X; yref=Y; t0=T0;
  cphi0=cos(phi0); sphi0=sin(phi0);
  x0=X0(); y0=Y0(); xc=Xc(); yc=Yc();
  code=Code;
  decode(code,qd0,qphi0,qomega,qz0,qtanl,qt0,nfree);
  mode=Mode; turnflag=0;
}//endof DcxHel

DcxHel::~DcxHel( ){ }

//accessors

double DcxHel::Xc()const{
  if(ominfl){
    return (X0()-sphi0/omega);
  }else{
    return 999999999.9;
  }//(ominfl)
}//endof Xc

double DcxHel::Yc()const{
  if(ominfl){
    return (Y0()+cphi0/omega);
  }else{
    return 999999999.9;
  }//(ominfl)
}//endof Yc

double DcxHel::X0()const{
  return (xref-sphi0*d0);
}//endof X0

double DcxHel::Y0()const{
  return (yref+cphi0*d0);
}//endof Y0

double DcxHel::Xh(double l)const{
  if(ominfl){
    double phit=phi0+omega*l;
    return (xc+sin(phit)/omega);
  }else{
    return (x0+cphi0*l-0.5*l*l*omega*sphi0);
  }//ominfl
}//endof Xh

double DcxHel::Yh(double l)const{
  if(ominfl){
    double phit=phi0+omega*l;
    return (yc-cos(phit)/omega);
  }else{
    return (y0+sphi0*l+0.5*l*l*omega*cphi0);
  }//ominfl
}//endof Yh

double DcxHel::Zh(double l)const{
  return (z0+tanl*l);
}//endof Zh

double DcxHel::Pt(double l)const{
  if(ominfl){return DcxHel::ktopt/fabs(omega);}
  else{return 1000000.0;}//ominfl
}//endof Px

double DcxHel::Px(double l)const{
  if(ominfl){double phit=phi0+omega*l; return DcxHel::ktopt*cos(phit)/fabs(omega);}
  else{return 1000.0*cphi0;}//ominfl
}//endof Px

double DcxHel::Py(double l)const{
  if(ominfl){double phit=phi0+omega*l; return DcxHel::ktopt*sin(phit)/fabs(omega);}
  else{return 1000.0*sphi0;}//ominfl
}//endof Py

double DcxHel::Pz(double l)const{
  if(ominfl){return DcxHel::ktopt*tanl/fabs(omega);}
  else{return 1000.0*tanl;}//ominfl
}//endof Pz

double DcxHel::Ptot(double l)const{
  if(ominfl){return DcxHel::ktopt*sqrt(1.0+tanl*tanl)/fabs(omega);}
  else{return 1000.0*sqrt(1.0+tanl*tanl);}//ominfl
}//endof Ptot

double DcxHel::Lmax()const{
  double lmax=250.0;
  if(ominfl){
    double rmax=1.0/fabs(omega);
    double dmax=fabs(d0)+2.0*rmax;
    if (dmax>80.0)lmax=DcxHel::pi*rmax;
  }
  return lmax;
}//endof Lmax

//controls
//control fitting mode
void DcxHel::SetMode(int n){mode=n;}
void DcxHel::SetRef(double x, double y){xref=x; yref=y;}
//control free variables
void DcxHel::SetOmega(int Qomega){
  nfree=nfree+deltaq(qomega,Qomega);
  code=code+deltaq(qomega,Qomega)*100;
  qomega=Qomega;
}
void DcxHel::SetPhi0(int Qphi0){
  nfree=nfree+deltaq(qphi0,Qphi0);
  code=code+deltaq(qphi0,Qphi0)*10;
  qphi0=Qphi0;
}
void DcxHel::SetD0(int Qd0){
  nfree=nfree+deltaq(qd0,Qd0);
  code=code+deltaq(qd0,Qd0);
  qd0=Qd0;
}
void DcxHel::SetTanl(int Qtanl){
  nfree=nfree+deltaq(qtanl,Qtanl);
  code=code+deltaq(qtanl,Qtanl)*10000;
  qtanl=Qtanl;
}
void DcxHel::SetZ0(int Qz0){
  nfree=nfree+deltaq(qz0,Qz0);
  code=code+deltaq(qz0,Qz0)*1000;
  qz0=Qz0;
}
void DcxHel::SetT0(int Qt0) {
  nfree=nfree+deltaq(qt0, Qt0);
  code=code+deltaq(qt0, Qt0)*100000;
  qt0=Qt0;
}

//operators
DcxHel& DcxHel::operator=(const DcxHel& rhs){
  copy(rhs);
  return *this;
}

//decode free parameter code
void 
DcxHel::decode(const int kode,int& i1,int& i2,
               int& i3,int& i4,int& i5,int& i6,int& n)
{
  int temp=kode;
  temp=temp/1000000; temp=kode-1000000*temp;
  i6=temp/100000;    temp=temp-100000*i6;
  i5=temp/10000;     temp=temp-10000*i5;
  i4=temp/1000;      temp=temp-1000*i4;
  i3=temp/100;       temp=temp-100*i3;
  i2=temp/10;        i1=temp-10*i2;
  n=0;
  if(i6==1){n++;}else{i6=0;};
  if(i5==1){n++;}else{i5=0;};
  if(i4==1){n++;}else{i4=0;};
  if(i3==1){n++;}else{i3=0;};
  if(i2==1){n++;}else{i2=0;};
  if(i1==1){n++;}else{i1=0;};
}//endof decode

//copy |DcxHel| to |DcxHel|
void 
DcxHel::copy(const DcxHel& rhs)
{
  omega=rhs.Omega(); phi0=rhs.Phi0(); d0=rhs.D0(); t0=rhs.T0();
  tanl=rhs.Tanl(); z0=rhs.Z0();
  cphi0=rhs.CosPhi0(); sphi0=rhs.SinPhi0();
  x0=rhs.X0(); y0=rhs.Y0(); xc=rhs.Xc(); yc=rhs.Yc();
  xref=rhs.Xref(); yref=rhs.Yref();
  qomega=rhs.Qomega(); qphi0=rhs.Qphi0(); qd0=rhs.Qd0(); qt0=rhs.Qt0();
  qtanl=rhs.Qtanl(); qz0=rhs.Qz0();
  mode=rhs.Mode(); nfree=rhs.Nfree();
  code=rhs.Code(); ominfl=rhs.Ominfl(); omin=rhs.Omin();
  turnflag=rhs.GetTurnFlag();
}//endof copy

double 
DcxHel::Doca( double wx, double wy, double wz,
              double xi, double yi, double zi )
{
  // describe wire
//  edm::LogInfo("RoadSearch") << " In Doca, xi = " << xi << " yi = " << yi << " zi = " << zi ;
  CLHEP::Hep3Vector ivec(xi,yi,zi); 
  wvec=CLHEP::Hep3Vector(wx,wy,wz);
//  edm::LogInfo("RoadSearch") << " In Doca, wx = " << wx << " wy = " << wy << " wz = " << wz ;
  //  calculate len to doca
  double zd,xd=xi,yd=yi;
//  edm::LogInfo("RoadSearch") << " In Doca, start xd = " << xd << " yd = " << yd ;
  double lnew,t1,t2,dphi,dlen=1000.0; len=0.0; int itry=2;
  // int segflg=0; if ((code==111)&&(z0==0.0)&&(tanl==0.0))segflg=1;
  // int superseg=0; if ((code==11111)&&(xref!=0.0)&&(yref!=0.0))superseg=1;
  double circut, circum=10000.; 
  if (ominfl){circum=DcxHel::twoPi/fabs(omega);} circut=0.50*circum;
  while(itry){
    if (ominfl){ 
      t1=-xc+xd; t2=yc-yd; phi=atan2(t1,t2); 
      if (omega<0.0)phi+=DcxHel::pi; if (phi>DcxHel::pi)phi-=DcxHel::twoPi; dphi=phi-phi0;
      if (omega < 0.0){
	if (dphi > 0.0){dphi-=DcxHel::twoPi;}
	if (dphi < -DcxHel::twoPi){dphi+=DcxHel::twoPi;}
      }else{
	if (dphi < 0.0){dphi+=DcxHel::twoPi;}
	if (dphi > DcxHel::twoPi){dphi-=DcxHel::twoPi;}
      }
      lnew=dphi/omega; 
      //   if ((lnew>circut)&&(segflg))lnew-=circum; 
      //   if ((lnew>circut)&&(superseg))lnew-=circum; 
      if ((lnew>circut)&&(turnflag))lnew-=circum; 
      zh=Zh(lnew); 
      xd=xi+(zh-zi)*wx/wz; yd=yi+(zh-zi)*wy/wz; zd=zh;
//      edm::LogInfo("RoadSearch") << " In Doca, xd = " << xd << " yd = " << yd << " zh = " << zh;
//      edm::LogInfo("RoadSearch") << " lnew = " << lnew ;
      dlen=fabs(lnew-len); len=lnew; 
      //   if (segflg)break; 
      if (fabs(zh) > 250.0)break;
      if ( (0.0==wx) && (0.0==wy) )break; if (dlen < 0.000001)break; itry--;
    }else{len=(xi-xref)*cphi0+(yi-yref)*sphi0; zh=z0+tanl*len; phi=phi0; break;}
  }
  //  CLHEP::Hep3Vector Dvec(xd,yd,zd);
  xh=Xh(len); yh=Yh(len); CLHEP::Hep3Vector hvec(xh,yh,zh);
//  edm::LogInfo("RoadSearch") << " In Doca, xh = " << xh << " yh = " << yh << " zh = " << zh ;
  double lamb=atan(tanl); cosl=cos(lamb); sinl=sin(lamb);
  tx=cosl*cos(phi); ty=cosl*sin(phi); tz=sinl; 
  tvec=CLHEP::Hep3Vector(tx,ty,tz); 
  CLHEP::Hep3Vector vvec=wvec.cross(tvec); 
  vhat=vvec.unit(); vx=vhat.x(); vy=vhat.y(); vz=vhat.z();
//  edm::LogInfo("RoadSearch") << " In Doca, vx = " << vx << " vy = " << vy << " vz = " << vz ;
  dvec=ivec-hvec; double doca=dvec*vhat;
//  edm::LogInfo("RoadSearch") << " doca = " << doca ;
  double f1=dvec*tvec; double f2=wvec*tvec; double f3=dvec*wvec;
  f0=(f1-f2*f3)/(1.0-f2*f2); 
  if (doca>0.0){samb=-1;}else{samb=+1;}
  double wirephi=atan2(yd,xd); 
  //babar eang=BbrAngle(phi-wirephi);
  eang=normalize(phi-wirephi);
  if (fabs(eang)<half_pi){wamb=samb;}else{wamb=-samb;}
  if (fabs(zh) > 250.0)doca=1000.0; 
  return doca;
}//endof Doca

std::vector<float>
DcxHel::derivatives(const DcxHit& hit) 
{
  double doca=Doca(hit);
  std::vector<float> temp(nfree+1);
  temp[0]=doca;
  double fac=1.0;
  if((mode==0)&&(doca<0.0)) {fac=-fac;}
  if(mode==0) {temp[0]=fabs(temp[0]);}
  int bump=0;
  if(qd0){double dddd0=-vx*sphi0+vy*cphi0; bump++; temp[bump]=dddd0*fac;}
  if(qphi0){
    //           double dddp0=-(yh-y0)*vx+(xh-x0)*vy;
    double dddp0=-(yh-y0+f0*ty)*vx+(xh-x0+f0*tx)*vy;
    dddp0=dddp0*(1.0+d0*omega);
    bump++; temp[bump]=dddp0*fac;
  }
  if(qomega){double dddom;
    if (ominfl){ 
      dddom=((len*cos(phi)-xh+x0)*vx+(len*sin(phi)-yh+y0)*vy)/omega;
      dddom+=f0*len*cosl*(-sin(phi)*vx+cos(phi)*vy);
    }
    else{dddom=0.5*len*len*(-vx*sphi0+vy*cphi0);}
    bump++; temp[bump]=dddom*fac;
  }
  if(qz0){double dddz0=vz; bump++; temp[bump]=dddz0*fac;}
  if(qtanl){double ddds=vz*len; bump++; temp[bump]=ddds*fac;}
  if(qt0){bump++; temp[bump]=-hit.v();}
  return temp;
}//endof derivatives

void DcxHel::print()const {
  edm::LogInfo("RoadSearch") << "  " 
   << " d0 = " << d0
   << " phi0 = " << phi0
   << " omega = " << omega
   << " z0 = " << z0
   << " tanl = " << tanl
   << " t0 = " << t0 
   << " code = " << code 
   << " mode = " << mode
   << " ominfl = " << ominfl
   << " nfree = " << nfree 
   << " x0 = " << x0
   << " y0 = " << y0
   << " xc = " << xc
   << " yc = " << yc
   << " xref = " << xref
   << " yref = " << yref 
   << "  " ;
}//endof print

void DcxHel::flip(){ 
  if (ominfl){
    if ( (fabs(d0)+2.0/fabs(omega)) > 80.0)return;
    double lturn=DcxHel::twoPi/fabs(omega); double zturn=Zh(lturn);
//    edm::LogInfo("RoadSearch") << "z0 " << z0 << " zturn " << zturn ;
    if (fabs(zturn) < fabs(z0)){
      z0=zturn; tanl=-tanl; omega=-omega; d0=-d0; 
      phi0=phi0-DcxHel::pi; if (phi0<-DcxHel::pi){phi0+=DcxHel::twoPi;} 
      cphi0=cos(phi0); sphi0=sin(phi0); x0=X0(); y0=Y0();
    } 
  }
}//endof flip

double DcxHel::normalize(double angle) {
  if (angle < - DcxHel::pi) {
    angle += DcxHel::twoPi;
    if (angle < - DcxHel::pi) angle = fmod(angle+ DcxHel::pi, DcxHel::twoPi) + DcxHel::pi; 
  }
  else if (angle > DcxHel::pi) {
    angle -= DcxHel::twoPi;
    if (angle > DcxHel::pi) angle = fmod(angle+DcxHel::pi, DcxHel::twoPi) - DcxHel::pi;
  }
  return angle;
}
