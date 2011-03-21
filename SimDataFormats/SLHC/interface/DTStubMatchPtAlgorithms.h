//
//   \class DTStubMatchPtAlgorithms 
//          
/**

    Description: abstract base class for DTStubMatch (and DTSeededTracklet).
    Various ways of obtaining  muon Pt are defined for DT triggers having
    matched stacked tracker stubs, which indeed are all available only
    in DTStubMatch (and DTSeededTracklet) objects.

**/
//   April 5, 2010             
//   I. Lazzizzera - Trento University
//
//-------------------------------------------------------------------------
#ifndef DTStubMatchPtAlgorithms_H
#define DTStubMatchPtAlgorithms_H

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>    
#include <math.h>
#include <sstream>     

#include "FWCore/ParameterSet/interface/ParameterSet.h"            
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"      
#include "DataFormats/GeometryVector/interface/GlobalVector.h"     

#include "SimDataFormats/SLHC/interface/DTStubMatchPtVariety.h"
#include "SimDataFormats/SLHC/interface/DTStubMatchPt.h"     

#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>


using namespace std;


//--------------------------------------------------------
const int StackedLayersInUseTotal = 6;

const int StackedLayersInUse[] = {0, 1, 2, 3, 8, 9};

int const our_to_tracker_lay_Id(int const l);

int const tracker_lay_Id_to_our(int const l);

static int const LENGTH = StackedLayersInUseTotal;

static int const STUBSTUBCOUPLES = 
  (StackedLayersInUseTotal*(StackedLayersInUseTotal-1))/2;
//--------------------------------------------------------



class DTStubMatchPtAlgorithms: public DTStubMatchPtVariety {

 public:
  // constructor
  DTStubMatchPtAlgorithms();
  // copy constructor
  DTStubMatchPtAlgorithms(const DTStubMatchPtAlgorithms& pts);
  // assignment operator
  DTStubMatchPtAlgorithms& operator =(const DTStubMatchPtAlgorithms& pts);
  // destructor
  virtual ~DTStubMatchPtAlgorithms() {};

  virtual void to_get_pt() = 0;         // make abstract this class!

  inline GlobalPoint  position()           const { return _position; }
  inline GlobalVector direction()          const { return _direction; }
  inline float  rho()                      const { return _rho; }
  inline float  phiCMS()                   const { return _phiCMS; }
  inline float 	thetaCMS() 		   const { return _thetaCMS; }
  inline float  eta()                      const { return _eta; }
  inline float 	bendingDT()                const { return _bendingDT; }
  inline float 	alphaDT()	           const { return _alphaDT; }
  inline double RiCoil()                   const { return _RiCoil; }
  inline double RoCoil()                   const { return _RoCoil; }
  inline double Rtilde()                   const { return _rtilde; }
  inline float 	sqrtDiscrim()	           const { return _sqrtDscrm; }
  inline float 	phiR()                     const { return _phiR; }
  inline float  PhiR()                     const { return _PhiR; }
  inline float  deltaPhiR()                const { return _deltaPhiR; }
  inline float  PhiRI()                    const { return _PhiRI; }
  inline const  float* stub_x()            const { return _stub_x; }
  inline const  float* stub_y()            const { return _stub_y; }
  inline const  float* stub_yz()           const { return _stub_z; }
  inline const  int*   stub_phi()          const { return _stub_phi; }
  inline const  int*   stub_theta()        const { return _stub_theta; }
  inline const  bool*  flagMatch()         const { return _flagMatch; }
  const GlobalVector*  stub_position()     const { return _stub_position; }
  inline float  vstub_rho()                const { return _vstub_rho; }
  inline float  vstub_phiCMS()             const { return _vstub_phiCMS; }

  float phiCMS(const GlobalVector& P) const
    { 
      float phiCMS = P.phi();
      if(phiCMS < 0.)
	phiCMS += 2. * TMath::Pi();
      if(phiCMS > 2*TMath::Pi())
	phiCMS -= 2 * TMath::Pi();
      return phiCMS;
    }

  float stub_phiCMS(size_t lay) const
    { 
      float phiCMS = static_cast<float>(_stub_phi[lay])/4096.;
      if(phiCMS < 0.)
	phiCMS += 2. * TMath::Pi();
      if(phiCMS > 2*TMath::Pi())
	phiCMS -= 2 * TMath::Pi();
      return phiCMS;
    }

  inline void set_stub_dephi(int lay, float dephi) { 	
    _stub_dephi[lay] = dephi; 
    return; 
  } 

  void set_stubstub_dephi(int L1, int L2, float dephi) { 
    // call tracker_lay_Id_to_our to be safe
    L1 = tracker_lay_Id_to_our(L1);
    L2 = tracker_lay_Id_to_our(L2);	
    if(L1 > L2) {
      int idx = (L1*(L1-1))/2 + L2;
      _stubstub_dephi[idx] = dephi;
    }
    else if(L2 > L1) {
      int idx = (L2*(L2-1))/2 + L1;
      _stubstub_dephi[idx] = -dephi;
    } 
    return; 
  } 

  float stubstub_dephi(int L1, int L2) const {
    /*
    cout << "giving stubstub_dephi: " << flush;
    */
    // call tracker_lay_Id_to_our to be safe
    L1 = tracker_lay_Id_to_our(L1);
    L2 = tracker_lay_Id_to_our(L2);
    if(L1 > L2) {
      int idx = (L1*(L1-1))/2 + L2;
      /*
      cout << "(" << L1 << ", " << L2 << ") --> idx = " << idx << ": _stubstub_dephi = " 
	   <<  _stubstub_dephi[idx] << endl;
      */
      return _stubstub_dephi[idx];
    }
    else if(L2 > L1) {
      int idx = (L2*(L2-1))/2 + L1;
      /*
      cout << "(" << L1 << ", " << L2 << ") --> idx = " << idx << ": _stubstub_dephi = " 
	   <<  _stubstub_dephi[idx] << endl;
      */
      return (- _stubstub_dephi[idx]);
    } 
    else return 0.;
  }


  virtual void setPt(const edm::ParameterSet& pSet);

  size_t matching_stubs_No() const { return _matching_stubs_No; }

  /*----------------------------------------------------------------*/
  /*   Using linear fit of dephi versus invPt                       */
  /*----------------------------------------------------------------*/

  float chi2_linearfit(int L1, int L2);     
  // chi2 of dephi vs invPt linear fit 
    
  float slope_linearfit(int L1, int L2); 
  // angular coefficient of dephi vs invPt linear fit
  
  float sigma_slope_linearfit(int L1, int L2); 
  // sigma of angular coefficient of dephi vs invPt linear fit
  
  float y_intercept_linearfit(int L1, int L2);
  // dephi @ invPt=0
  
  float sigma_y_intercept_linearfit(int L1, int L2);
  // sigma of dephi @ invPt=0
  /*----------------------------------------------------------------*/

 protected:

  // The DT trigger Id.
  // (that will effectively set in DTStubMatch class.)
  int   _wheel, _station, _sector, _bx, _code;
  GlobalPoint  _position;      
  GlobalVector _direction;
 
  // Derived DT trigger Id.
  // (that will effectively set by DTStubMatch class constructors.)
  float _rho, _phiCMS, _bendingDT;
  float _thetaCMS, _eta;

  // Matched stacked tracker stub phi and theta. 
  // (that will effectively set by DTStubMatch class constructors.)
  int _stub_phi[LENGTH];
  int _stub_theta[LENGTH];

  // Matched stacked tracker stubs
  // (that will effectively by DTStubMatch class constructors.)
  float _stub_x[LENGTH], _stub_y[LENGTH], _stub_z[LENGTH];                     
  float _stub_rho[LENGTH], _stub_phiCMS[LENGTH],  _stub_thetaCMS[LENGTH];  
  float _stub_dephi[LENGTH];       // that is fabs(_stub_phiCMS - phiCMS)
  //  triangular_matrix<float, upper> _stubstub_dephi(LENGTH, LENGTH);
  float _stubstub_dephi[STUBSTUBCOUPLES]; // phi[lay1] - phi[lay2] 
  GlobalVector _stub_position[LENGTH];                                         
  GlobalVector _stub_direction[LENGTH]; 

  // We have a match with tracker stub.
  // (It will be effectively set by DTStubMatch class constructors.)
  bool _flagMatch[LENGTH];

  size_t _matching_stubs_No;

  static const double _RiCoil, _RoCoil, _rtilde; 

  float  _alphaDT;  // tan(_alpha) is the angular coefficient of the muon trajectory 
                    // outside the inner surface of the CMS superconducting coil,  
                    // assuming that trajectory to be linear.
  // Say double delta = _phiCMS - _alphaDT; then:
  float _sqrtDscrm;    // sqrt(1. - rho*rho*sin(delta)*sin(delta)/(_rtilde*_rtilde)).
  // On the tranverse plane of CMS, we define coordinates of the intercept of the 
  // assumed straight line muon trajectory in the DT chambers volume with the 
  // cylindrical boundary surface of the CMS magnetic field.
  float _phiR;             // Polar coordinates of above; 
  float _deltaPhiR;        // _deltaPhiR = _phiR - _phiCMS; 
  // Next _sqrtDscrm set to 1. 
  float _PhiR;
  // Next using deltaPhiR_over_bending  = 1. - _rho/_rtilde;           
  float _PhiRI;
  //
  float _vstub_rho, _vstub_phiCMS;          

  /*--------------------------------------------------------------------*/
  /*   Using linear fit of dephi versus invPt                           */
  /*--------------------------------------------------------------------*/
  static const float 
    _chi2[], // chi2 of dephi vs invPt linear fit 
    _p0[],   // dephi @ invPt=0
    _e0[],   // sigma of angular coefficient of dephi vs invPt linear fit
    _p1[],   // angular coefficient of dephi vs invPt linear fit
    _e1[];   // sigma of angular coefficient of dephi vs invPt linear fit

};


#endif

