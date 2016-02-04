//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DcxFittedHel.hh,v 1.3 2006/04/10 22:06:41 stevew Exp $
//
// Description:
//	Class Header for |DcxFittedHel| - helix fitting class
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
#ifndef _DCXFITTEDHEL_
#define _DCXFITTEDHEL_

//DcxHel & DcxFittedHel classes ...
#include <iostream>
#include <fstream>
//babar #include "DcxReco/DcxHel.hh"
//babar #include "CLHEP/Alist/AList.h"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHel.hh"
#include <vector>
 
class DcxHit;

//DcxFittedHel follows: (fitted helix class) 
class DcxFittedHel:public DcxHel{
public:
//constructors
  DcxFittedHel();

//hits+initial guess constructor
//babar  DcxFittedHel(HepAList<DcxHit> &ListOhits, DcxHel &hel, double Sfac=1.0);
  DcxFittedHel(std::vector<DcxHit*> &ListOhits, DcxHel &hel, double Sfac=1.0);

//destructor
  virtual ~DcxFittedHel( );

//accessors
  inline float Chisq()const{return chisq;}
  inline float Rcs()const{return rcs;}
  inline float Prob()const{return prob;} 
  inline float Fittime()const{return fittime;} 
  inline int Nhits()const {return nhits;}
  inline int Itofit()const {return itofit;}
  inline int Quality()const {return quality;}
  inline int Origin()const {return origin;}
  inline double Sfac()const {return sfac;}
  inline void SetQuality(const int &q) {quality=q;}
  inline void SetUsedOnHel(const int &i) {usedonhel=i;}
  inline int  GetUsedOnHel()const {return usedonhel;}
  int SuperLayer(int hitno=0)const; // return superlayer of |hitno
  int Layer(int hitno=0)const;	// return layer number of |hitno|
//babar  inline const HepAList<DcxHit> &ListOHits()const{return listohits;}
  inline const std::vector<DcxHit*> &ListOHits()const{return listohits;}

//workers
  float Residual(int i);
  float Pull(int i);
  int Fail(float Probmin=0.0)const;
  int ReFit();
  int FitPrint();
  int FitPrint(DcxHel &hel);
  void VaryRes();

//operators
  DcxFittedHel& operator=(const DcxHel&);// copy helix to fitted helix
  DcxFittedHel& operator=(const DcxFittedHel&); // copy fitted helix to fitted helix
//babar  DcxFittedHel& Grow(const DcxFittedHel&, HepAList<DcxHit> &);
  DcxFittedHel& Grow(const DcxFittedHel&, std::vector<DcxHit*> &);
  
//workers

protected:

//data
  int fail;			//fit failure codes
  float chisq;			//chisq of hit
  float rcs;			//chisq per dof
  float prob;			//chisq prob of hit
  float fittime;		//fit time in clock time (machine dep)
  int nhits;			//number of hits
  int itofit;			//number of iterations to convergence
  int quality;			// bigger quality=>great purity
  int origin;			// origin "hit", -1 if none
//babar  HepAList<DcxHit> listohits;	// list-of-hits making this |DcxFittedHel|
  std::vector<DcxHit*> listohits;	// list-of-hits making this |DcxFittedHel|
  double sfac;                  // error scale factor for fit
  int usedonhel;
  
//functions
//fitting routine
  int DoFit();
  int IterateFit();

private:

//data

//control
  void basics();
//babar  void basics(const HepAList<DcxHit> &);
  void basics(const std::vector<DcxHit*> &);

//check for included origin, if there move to end of |listohits|
//-1=>no origin; >=0 is hit number of origin
  int OriginIncluded();		// origin included

//static control parameters
  int bailout;		// bailout if chisq/ndof too big?
  float chidofbail;	// bailout cutoff
  int niter;		// max number of iterations

//static control sets
public:
  inline void SetBailOut(int i) {bailout=i;}
  inline void SetChiDofBail(float r) {chidofbail=r;}
  inline void SetNiter(int i) {niter=i;}
  
};// endof DcxFittedHel
 
#endif

