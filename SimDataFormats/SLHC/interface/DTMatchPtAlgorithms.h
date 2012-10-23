//
//   \class DTMatchPtAlgorithms 
//          
/**

    Description: abstract base class for DTMatch (and DTSeededTracklet).
    Various ways of obtaining  muon Pt are defined for DT triggers having
    matched stacked tracker stubs, which indeed are all available only
    in DTMatch (and DTSeededTracklet) objects.

**/
//   April 5, 2010             
//   I. Lazzizzera - Trento University
//
//-------------------------------------------------------------------------
#ifndef DTMatchPtAlgorithms_H
#define DTMatchPtAlgorithms_H

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

#include "SimDataFormats/SLHC/interface/DTMatchPtVariety.h"
#include "SimDataFormats/SLHC/interface/DTTrackerStub.h"
#include "SimDataFormats/SLHC/interface/DTTrackerTracklet.h"
#include "SimDataFormats/SLHC/interface/DTMatchPt.h"     

#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>


using namespace std;


//--------------------------------------------------------
int const StackedLayersInUseTotal = 6;
int const StackedLayersInUse[] = {0, 1, 2, 3, 8, 9};
int const our_to_tracker_lay_Id(int const l);
int const tracker_lay_Id_to_our(int const l);

static int const STUBSTUBCOUPLES = 
  (StackedLayersInUseTotal*(StackedLayersInUseTotal-1))/2;

size_t const TkSLinUseTotal = 3; // Tracker Super-Layers 0, 1 and 4
const int TkSLinUse[] = {0, 1, 4};
int const our_to_tracker_superlay_Id(int const l);
int const tracker_superlay_Id_to_our(int const l);
//--------------------------------------------------------



class DTMatchPtAlgorithms: public DTMatchPtVariety {
  /*
    This is abstract base class for DTMatch class. 
    It is derived from the DTMatchPtVariety, a container of muon Pt's as 
    obtained by a variety of approaches, combining stub hits to which a DT 
    muon was extrapolated.
    The class main method is setPt.
   */
 public:
  // constructor
  DTMatchPtAlgorithms();
  // copy constructor
  DTMatchPtAlgorithms(const DTMatchPtAlgorithms& pts);
  // assignment operator
  DTMatchPtAlgorithms& operator =(const DTMatchPtAlgorithms& pts);
  // destructor
  virtual ~DTMatchPtAlgorithms() {};

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

  float phiCMS(const GlobalVector& P) const;
  float stub_phiCMS(size_t lay) const;
  void set_stub_dephi(int lay, float dephi); 
  void set_stubstub_dephi(int L1, int L2, float dephi);
  float stubstub_dephi(int L1, int L2) const;

  virtual void setPt(const edm::ParameterSet& pSet);

  size_t matching_stubs_No() const { return _matching_stubs_No; }

  inline vector<TrackerTracklet*>& MatchingTkTracklets() 
    { return _MatchingTkTracklets; }

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
  /*----------------------------------------------------------------------------*/

 protected:

  // The original DT trigger Id (effectively set by DTMatch class methods).
  int   _wheel, _station, _sector, _bx, _code;
  GlobalPoint  _position;      
  GlobalVector _direction;
 
  // Derived from above DT trigger Id (effectively set by DTMatch class methods).
  float _rho, _phiCMS, _bendingDT;
  float _thetaCMS, _eta;

  /* Matching stacked tracker stubs ***********************************************
     (that will effectively set by DTMatch class constructors.)              */
  int _stub_phi[StackedLayersInUseTotal];
  int _stub_theta[StackedLayersInUseTotal];
  float 
    _stub_x[StackedLayersInUseTotal], 
    _stub_y[StackedLayersInUseTotal], 
    _stub_z[StackedLayersInUseTotal];                     
  float 
    _stub_rho[StackedLayersInUseTotal], 
    _stub_phiCMS[StackedLayersInUseTotal],  
    _stub_thetaCMS[StackedLayersInUseTotal];  
  float _stub_dephi[StackedLayersInUseTotal]; // that is fabs(_stub_phiCMS - phiCMS)
  float _stubstub_dephi[STUBSTUBCOUPLES];     // phi[lay1] - phi[lay2] 
  GlobalVector _stub_position[StackedLayersInUseTotal];                              
  GlobalVector _stub_direction[StackedLayersInUseTotal]; 
  bool _flagMatch[StackedLayersInUseTotal];  /* We have a match with tracker stub.
						(It is be effectively set by 
						DTMatch class constructors.)*/
  size_t _matching_stubs_No; /* that is No of layers with at least one valid 
				matching stubs                                  */

  /* Stubs *************************************************************************
     the closest for each layer, if any:                                          */ 
  vector<TrackerStub*> _MatchingTkStubs;

  /* Tracklets *********************************************************************
     the closest for each super-layer, if any:                                    */
  vector<TrackerTracklet*> _MatchingTkTracklets;


  // Concerning the virtual tracker layer ******************************************
  static const double _RiCoil, _RoCoil, _rtilde; 

  float  _alphaDT;  /* tan(_alphaDT) is the angular coefficient of the muon  
		       trajectory outside the inner surface of the CMS 
		       superconducting coil, assuming that trajectory to be linear.
		       Say double delta = _phiCMS - _alphaDT; then:               */
  float _sqrtDscrm; /* sqrt(1. - rho*rho*sin(delta)*sin(delta)/(_rtilde*_rtilde)).
		       On the tranverse plane of CMS, we define coordinates of 
		       the intercept of the assumed straight line muon trajectory 
		       in the DT chambers volume with the  cylindrical boundary 
		       surface of the CMS magnetic field.                      */
  float _phiR;         // Polar coordinates of above; 
  float _deltaPhiR;    // _deltaPhiR = _phiR - _phiCMS; 
  // Next is for _sqrtDscrm set to 1. 
  float _PhiR;
  // Next is for use of deltaPhiR_over_bending  = 1. - _rho/_rtilde;           
  float _PhiRI;
  //
  float _vstub_rho, _vstub_phiCMS;          

  //*******************************************************************************
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

