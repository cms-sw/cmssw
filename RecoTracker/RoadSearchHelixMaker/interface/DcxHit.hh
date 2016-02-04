//------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DcxHit.hh,v 1.2 2006/03/22 22:47:37 stevew Exp $
//
// Description:
//	Class Header for |DcxHit|: hit that can calculate derivatives
//      and plot hits self specialized to the drift chamber
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

#ifndef _DCXHIT_
#define _DCXHIT_

//babar #include <math.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
//babar #include "DchData/DchHit.hh"
//babar #include "DchCalib/DchTimeToDist.hh"



class DcxHel;
//babar class DchDigi;
//babar class DchDisplay;
//babar class DchTimeToDist;

class DcxHit{	// A DC hit that can calculate derivatives
public:
  //constructors
//babar DcxHit(const DchDigi *pdcdatum, float c0=0,  float cresol=.0180);
//babar DcxHit(const DchHit  *pdchhit,  float c0=0,  float cresol=.0180);

DcxHit(float sx, float sy, float sz, float wx, float wy, float wz,
       float c0=0.0,  float cresol=0.0180);//cms constructor
  //destructor
virtual ~DcxHit( );
  //accessors
//babar inline const DchDigi* getDigi()const {return _pdcdatum;}
//babar inline const DchHit* getDchHit()const {return _dchhit;}
inline int WireNo()const {return _wirenumber;} // Wire#
inline int Layer()const {return _layernumber;}// layer#
inline float t()const {return _t;} // drift time
inline float x()const {return _x;} // x of wire
inline float y()const {return _y;} // y of wire
inline float xpos()const {return _xpos;}  
inline float ypos()const {return _ypos;} 
inline float xneg()const {return _xneg;} 
inline float yneg()const {return _yneg;} 
inline float wx()const {return _wx;} 
inline float wy()const {return _wy;} 
inline float wz()const {return _wz;} 
inline float pw()const {return _pw;} 
inline bool stereo()const {return _s;} // stereo angle of wire
inline float v()const {return _v;} // drift velocity
inline int type()const {return 1;} // flags |DcxHit|
inline int SuperLayer()const {return _superlayer;} //SuperLayer#
  //workers
inline void SetConstErr(int i) {_consterr=i;}
float tcor(float zh=0, float tof=0, float tzero=0)const 
      { return  _t - _c0 - tof - tzero; }
float d(float zh=0, float tof=0, float tzero=0,
        int wamb=0, float eang=0)const // |drift dist|
   { 
//     return _t2d->timeToDist(tcor(zh,tof,tzero),wamb,eang,0,zh);
     return 0.0;//cms??
   }
float d(DcxHel& hel)const; // |drift dist| (changes hel's internal state)
float e(float dd=0)const  
//babar //drift error currently in use -> //cms use _consterr for now
    { return (0!=_consterr)? _cresol: 
//babar      _t2d->resolution(dd); 
      _cresol;//cms??
    }
float pull(DcxHel& hel)const;//Chisq contribution to fit
float residual(DcxHel &hel)const;	//residual of this hit
std::vector<float> derivatives(DcxHel& hel)const; //Derivatives, etc.
void print(std::ostream &o,int i=0)const;	//print this hit
inline void SetUsedOnHel(int i) {usedonhel=i;}
inline int  GetUsedOnHel()const {return usedonhel;}
protected:
  //functions
void process();
  //data
//babar const DchDigi *_pdcdatum; //pointer to |DchDigi| defining this hit
float _t;
int _wirenumber;
int _layernumber;
int _superlayer;
float _x;
float _y;
bool _s;
float _d;
float _v;
float _e;
float _xpos,_ypos,_xneg,_yneg;
float _p,_sp,_cp;
double _pw;
double _wx,_wy,_wz;
//babar const DchHit* _dchhit; 
int _consterr; 
int usedonhel;

float _c0; // accumulated global time offset; changes in t0 go here
float _cresol;
//babar const DchTimeToDist* _t2d;

};//endof DcxHit
 
#endif
