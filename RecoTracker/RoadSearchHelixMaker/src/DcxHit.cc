//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DcxHit.cc,v 1.4 2006/04/10 22:06:41 stevew Exp $
//
// Description:
//	Class Implementation for |DcxHit|: drift chamber hit that can compute
//      derivatives and plot itself.
//
// Environment:
//	Software developed for the BaBar Detector at the SLAC B-Factory.
//	Attempted port to CMSSW
//
// Author List:
//	A. Snyder, S. Wagner
//
// Copyright Information:
//	Copyright (C) 1995	SLAC
//
//------------------------------------------------------------------------

//babar #include "BaBar/BaBar.hh"
//babar #include "DcxReco/DcxHit.hh"
//babar #include "DcxReco/DcxHel.hh"
//babar #include "AbsEnv/AbsEnv.hh"
//babar #include "DchEnv/DchEnv.hh"
//babar #include "DchGeom/DchDetector.hh"
//babar #include "DchGeom/DchLayer.hh"
//babar #include "DchData/DchDigi.hh"
//babar #include "DchCalib/DchTimeToDist.hh"
#include <iomanip>
#include <cmath>
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHit.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using std::cout;
using std::endl;
using std::ostream;

DcxHit::DcxHit(float sx, float sy, float sz, float wx, float wy, float wz,
               float c0, float cresol) 
              :_wx(wx),_wy(wy),_wz(wz),_c0(c0),_cresol(cresol)
{
//  edm::LogInfo("RoadSearch") << "try to make a cms DcxHit " << _c0 << " " << _cresol;
  _layernumber=0;
  if (_wz<0.0){_wx=-_wx;_wy=-_wy;_wz=-_wz;}
  _s = false; if (_wz<0.9975)_s=true; 
  if (_s){
    double rad=sqrt(sx*sx+sy*sy);
    if ((20.0<rad)&&(rad<30.0))_layernumber=2;
    if ((30.0<rad)&&(rad<38.0))_layernumber=4;
    if ((58.0<rad)&&(rad<66.0))_layernumber=10;
    if ((66.0<rad)&&(rad<74.0))_layernumber=12;
//    edm::LogInfo("RoadSearch") << "stereo rad layer " << rad << " " << _layernumber 
//			         << " " << _wx << " " << _wy << " " << _wz;
    _x = sx-sz*_wx/_wz; _y = sy-sz*_wy/_wz; 
  }else{
    _x = sx; _y = sy; 
    double rad=sqrt(_x*_x+_y*_y);
    if ((20.0<rad)&&(rad<30.0))_layernumber=1;
    if ((30.0<rad)&&(rad<38.0))_layernumber=3;
    if ((38.0<rad)&&(rad<46.0))_layernumber=5;
    if ((46.0<rad)&&(rad<58.0))_layernumber=7;
    if ((58.0<rad)&&(rad<66.0))_layernumber=9;
    if ((66.0<rad)&&(rad<74.0))_layernumber=11;
    if ((74.0<rad)&&(rad<83.0))_layernumber=13;
    if ((83.0<rad)&&(rad<92.0))_layernumber=15;
    if ((92.0<rad)&&(rad<100.0))_layernumber=17;
    if ((100.0<rad)&&(rad<120.0))_layernumber=19;
//    edm::LogInfo("RoadSearch") << "axial layer " << _x << " " << _y;
  }
  _wirenumber=0;
  _superlayer=1+(_layernumber-1)/4;
  _t=0.0;
  _p = 0.0;//cms layerPtr->phiOffset() + _wirenumber*layerPtr->dPhi();
  double deltaz = 0.0;//cms  is symmetric detector 
  double pw=atan2(_y,_x);
  _pw=pw; 
//  double tst=_s;  
//  _wx=-tst*sin(pw); _wy= tst*cos(pw); 
//  _wz=1.0-tst*tst; if (_wz>0.0){_wz=sqrt(_wz);}else{_wz=0.0;} 
  _x -= deltaz*_wx/_wz; _y -= deltaz*_wy/_wz;
  _sp = sin(_p); _cp = cos(_p);
  _d=d();
  _consterr = 1;
  _e = _cresol;
// note _v is a total cludge
  _v=0.0018; 
//cms  if ( (_t-_c0) > 0.0 )_v=_d/(_t-_c0);
  _xpos = _x - _d*_sp; _ypos = _y + _d*_cp;
  _xneg = _x + _d*_sp; _yneg = _y - _d*_cp;
  usedonhel=0;
}

//babar DcxHit::DcxHit(const DchDigi *pdcdatum, float c0, float cresol) 
//babar         :_dchhit(0),_pdcdatum(pdcdatum),_c0(c0),_cresol(cresol)
//babar {
//babar   process();
//babar }

//babar DcxHit::DcxHit(const DchHit  *pdchhit,  float c0,  float cresol) 
//babar         :_dchhit(pdchhit),_pdcdatum(pdchhit->digi()),
//babar          _c0(c0),_cresol(cresol)
//babar {
//babar   process();
//babar }

void 
DcxHit::process()
{
//babar   _wirenumber=_pdcdatum->wirenumber();
//babar   _layernumber=_pdcdatum->layernumber();
//babar   _t2d=gblEnv->getDch()->getDchTimeToDist(_layernumber,_wirenumber);
//babar   _superlayer=1+(_layernumber-1)/4;
//babar   _t=_pdcdatum->TdcTime();
//babar   const DchDetector* geomPtr = gblEnv->getDch()->getDchDetector(); // pointer to geometry
//babar   const DchLayer* layerPtr=geomPtr->getDchLayer(_layernumber);	   // pointer to layer
//babar   _x = layerPtr->xWire(_wirenumber);  
//babar   _y = layerPtr->yWire(_wirenumber); 
//babar   _s = layerPtr->stereo(); 
//babar   _p = layerPtr->phiOffset() + _wirenumber*layerPtr->dPhi();
//babar   double deltaz = geomPtr->zOffSet(); 
//babar   double tst=_s;  double pw=atan2(_y,_x);
//babar   _pw=pw; 
//babar   _wx=-tst*sin(pw); _wy= tst*cos(pw); 
//babar   _wz=1.0-tst*tst; if (_wz>0.0){_wz=sqrt(_wz);}else{_wz=0.0;} 
//babar   _x -= deltaz*_wx/_wz; _y -= deltaz*_wy/_wz;
//babar   _sp = sin(_p); _cp = cos(_p);
//babar   _d=d();
  _consterr = 1;
  _e = _cresol;
// note _v is a total cludge
  _v=0.0018; 
//babar   if ( (_t-_c0) > 0.0 )_v=_d/(_t-_c0);
//babar   _xpos = _x - _d*_sp; _ypos = _y + _d*_cp;
//babar   _xneg = _x + _d*_sp; _yneg = _y - _d*_cp;
  usedonhel=0;
}

//DcxHit destructor
DcxHit::~DcxHit( )
{
 ; 
}

float 
DcxHit::d(DcxHel &hel)const 
{
  hel.Doca(*this); // changes hel's internal state...
  return d(hel.Doca_Zh(),hel.Doca_Tof(),hel.T0(),
           hel.Doca_Wamb(),hel.Doca_Eang());
}//endof d

float 
DcxHit::pull(DcxHel &hel)const 
{// compute pulls for |hel|
//  float doca=hel.Doca(*this); if(hel.Mode() == 0)doca=fabs(doca);
//  return (d(hel.Doca_Zh(),hel.Doca_Tof(),hel.T0(),
//           hel.Doca_Wamb(),hel.Doca_Eang())-doca)/e(doca);
  return residual(hel)/e();
}//endof pull

float 
DcxHit::residual(DcxHel &hel)const 
{ // compute residuals for |hel|
  float doca=hel.Doca(*this);
  if(hel.Mode() == 0)doca=fabs(doca);
//  doca += v()*hel.T0();
  return d(hel.Doca_Zh(),hel.Doca_Tof(),hel.T0(),
           hel.Doca_Wamb(),hel.Doca_Eang())-doca;
}//endof residual

std::vector<float>
DcxHit::derivatives(DcxHel &hel)const 
{ // compute derivatives for |hel|
  std::vector<float> deriv=hel.derivatives(*this);
  float dtemp=d(hel.Doca_Zh(),hel.Doca_Tof(),hel.T0(),
                hel.Doca_Wamb(),hel.Doca_Eang());
  deriv[0]=dtemp-deriv[0];
//  deriv[0] -= v()*hel.T0();
  float ewire=e(dtemp);
  for(unsigned int i=0; i<deriv.size(); i++) {deriv[i]/=ewire;}
  return deriv;
}//endof derivatives

void 
DcxHit::print(ostream &o,int i)const 
{
      o << " Digi # " << i  
        << " Layer # " << Layer() 
        << " SuperLayer # " << SuperLayer() 
        << " Wire # " << WireNo() 
//        << " Drift dist (cm) " << d() 
//        << " Drift err  (cm) " << e() 
        << " Drift time (ns) " << t();
}//endof print
