//-------------------------------------------------------------------
//
//   \class DTStubMatch
/**
 *   Description:  Bti triggers matched between phi and theta view
 *                 are extrapolated to stacked tracker stubs. 
*/
//   090202             
//   Sara Vanini - Padua University
//   I. Lazzizzera - Trento University
//
//-------------------------------------------------------------------
#ifndef DTStubMatch_H
#define DTStubMatch_H

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <iterator>     //Ignazio
#include <math.h>
#include <sstream>      //Ignazio

#include <TMath.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"                 //Ignazio
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"           //Ignazio
#include "DataFormats/GeometryVector/interface/GlobalVector.h"          //Ignazio

//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTTrackerStub.h" //Ignazio
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTStubMatchPt.h" //Ignazio
#include "SimDataFormats/SLHC/interface/DTTrackerStub.h"    //Ignazio
#include "SimDataFormats/SLHC/interface/DTStubMatchPt.h"    //Ignazio


using namespace std;


const int StackedLayersInUseTotal = 6;

const int StackedLayersInUse[] = {0, 1, 2, 3, 8, 9};

int const our_to_tracker_lay_Id(int const l);

int const tracker_lay_Id_to_our(int const l);

static int const LENGTH = StackedLayersInUseTotal;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTStubMatch {

 public:
  
  // trivial default constructor needed by ROOT dictionary (Ignazio)
  DTStubMatch();
  
  // constructor
  DTStubMatch(int wheel, int station, int sector,
	      int bx, int code, int phi, int phib, float theta, bool flagBxOK,
	      bool debug_dttrackmatch = false);

  // constructor
  DTStubMatch(int wheel, int station, int sector,
	      int bx, int code, int phi, int phib, float theta, 
	      GlobalPoint position, GlobalVector direction,
	      bool flagBxOK, bool debug_dttrackmatch = false) ;

  // copy constructor
  DTStubMatch(const DTStubMatch& dtsm);

  // destructor
  ~DTStubMatch(){ } 
  
  //return functions
  inline int 	wheel() 		const { return _wheel; }
  inline int	station()		const { return _station; }
  inline int 	sector() 		const { return _sector; }
  inline int 	bx() 			const { return _bx; }
  inline int	code() 			const { return _code; }
  inline int 	phi_ts()		const { return _phi_ts; }
  inline int 	phib_ts()		const { return _phib_ts; }
  inline int 	theta_ts()		const { return _theta_ts; }
  inline float 	phi_glo()               const
    { 
      float phiCMS = static_cast<float>(_phi_ts)/4096.+(_sector-1)*TMath::Pi()/6.;
      if(phiCMS < 0.)
	phiCMS += 2. * TMath::Pi();
      if(phiCMS > 2*TMath::Pi())
	phiCMS -= 2 * TMath::Pi();
      return phiCMS;
    }
  inline float 	phib_glo()              const
    { 
      return static_cast<float>(_phib_ts)/512.;                          	// 9 bits
    }
  inline float 	bendingDT()             const { return _bendingDT; }            // 9 bits

  inline GlobalPoint  position()        const { return _position; }
  inline GlobalVector direction()       const { return _direction; }
  inline float 	alphaDT()	        const { return _alphaDT; }
  inline float 	sqrtDiscrim()	        const { return _sqrtDscrm; }
  inline float  Xerre()	                const { return _Xerre; }
  inline float  Yerre()                 const { return _Yerre; }
  inline float  XerreI()	        const { return _XerreI; }
  inline float  YerreI()                const { return _YerreI; }
  inline float  xerre()	                const { return _xerre; }
  inline float  yerre()                 const { return _yerre; }
  inline float  erre()                  const { return _erre; }
  inline float  erresq()                const { return _erresq; }
  inline float  rhoDT()                 const { return _rhoDT; }
  inline float  phiCMS()                const { return _phiCMS; }
  inline float 	thetaCMS() 		const { return _thetaCMS; }
  inline float  eta()                   const { return _eta; }
  inline bool 	flagBxOK() 		const { return _flagBxOK; }
  inline int    trig_order()		const { return _trig_order; }			
  inline int    predPhi(int lay)	const { return _pred_phi[lay]; }	// 12 bits
  inline int  	predSigmaPhi(int lay)	const { return _pred_sigma_phi[lay]; }	// 12 bits
  inline int  	predTheta()		const { return _pred_theta; }		// 12 bits
  inline int  	predSigmaTheta(int lay) const { return _pred_sigma_theta[lay]; }// 12 bits
  inline float  predSigmaPhiB()		const { return _pred_sigma_phib; }	// 12 bits
  inline int  	stubPhi(int lay)	const { return _stub_phi[lay]; }
  inline int  	stubTheta(int lay)	const { return _stub_theta[lay]; }
  inline bool   isMatched(int lay)	const { return _flagMatch[lay]; }
  inline bool   flagReject()		const { return _flag_reject; }

  float const Pt(std::string const label) const; 
 
  //set rejection flag
  inline void setRejection(bool flag)
    { 
      _flag_reject = flag; 
      return; 
    }

  //set phi-eta matching order flag
  inline void setTrigOrder(int trig_order) { 
    _trig_order = trig_order; 
    return; 
  } 
 
  //set predicted tracker phi and theta in each layer
  inline void setPredStubPhi(int lay, int phi, int sigma_phi) { 	
    _pred_phi[lay] = phi; 
    _pred_sigma_phi[lay] = sigma_phi; 
    return; 
  } 

  inline void setPredStubTheta(int lay, int theta, int sigma_theta) { 	
    _pred_theta = theta; 
    _pred_sigma_theta[lay] = sigma_theta; 
    return; 
  } 

  inline void setPredSigmaPhiB(float sigma_phib) {
    _pred_sigma_phib = sigma_phib;
    return; 
  }

  //set tracker stub match function
  inline void setMatchStub(int lay, int phi, int theta) { 
    _stub_phi[lay] = phi;
    _stub_theta[lay] = theta; 
    _flagMatch[lay] = true; 
    return; 
  }

  inline void insertMatchingStubObj(TrackerStub* st) { 
    // Ignazio
    _matching_stubs.insert(st);
  }

  inline const size_t matchingStubsTotal() const { 
    // Ignazio
    return _matching_stubs.size();
  }

  inline const StubTracklet& getMatchingStubs() const { 
    // Ignazio
    return _matching_stubs;
  }

  void setMatchStub(int lay, int phi, int theta, 
		    GlobalVector position, GlobalVector direction) { 
    // Ignazio
    _stub_phi[lay]   = phi;
    _stub_theta[lay] = theta; 
    float length = 
      position.x()*position.x()+ 
      position.y()*position.y()+ 
      position.z()*position.z();
    if( length == 0. ) {
      _stub_rho[lay] = NAN;
      _stub_x[lay]   = NAN;
      _stub_y[lay]   = NAN;
      _stub_z[lay]   = NAN;
    }
    else {
    _stub_x[lay] = position.x();
    _stub_y[lay] = position.y();
    _stub_z[lay] = position.z();
    _stub_rho[lay] = 
      float(sqrt(position.x()*position.x() + position.y()*position.y()));
    }
    _stub_direction[lay] = direction;
    _flagMatch[lay] = true; 
    return; 
  }

  inline void setMatchStubPhi(int lay, int phi)	 { 	
    _stub_phi[lay] = phi; 
    _flagMatch[lay] = true; 
    return; 
  } 

  inline void setMatchStubTheta(int lay, int theta) {
    _stub_theta[lay] = theta; 
    _flagMatch[lay] = true; 
    return; 
  } 

  void extrapolateToTrackerLayer(int l);

  // SV 090505 correlate phib and error in station 1 to phib in station 2, 
  // for track rejection
  int corrPhiBend1ToCh2(int phib2);
  int corrSigmaPhiBend1ToCh2(int phib2, int sigma_phib2);

  void setPt(const edm::ParameterSet& pSet);

  // debug functions
  void print();  
  std::string writeMatchingStubs() const;
  std::string writeMatchingStubs(size_t) const;
  // end debug functions

  // debug flags
  bool _debug_dttrackmatch;


 private:
  
  void init();
  
  int   _wheel, _station, _sector, _bx, _code;
  int   _phi_ts, _phib_ts; 
  float _rhoDT, _phiCMS, _bendingDT;
  float _erre, _erresq;
  int   _theta_ts;
  float _thetaCMS, _eta;
  bool  _flagBxOK; 
  int   _trig_order;
  
  // predicted phi, theta, sigma_phi, sigma_theta, sigma_phib in tracker layers 
  // (NB wheel dependent!)
  int   _pred_phi[LENGTH];
  int   _pred_sigma_phi[LENGTH];
  int   _pred_theta;
  int   _pred_sigma_theta[LENGTH];
  float _pred_sigma_phib;
  
  // matched tracker stub phi and theta 
  int _stub_phi[LENGTH];
  int _stub_theta[LENGTH];
  // we have a match with tracker stub
  bool _flagMatch[LENGTH];


  // Ignazio begin ***
  GlobalPoint  _position;      
  GlobalVector _direction; 

  float _stub_x[LENGTH], _stub_y[LENGTH], _stub_z[LENGTH],  _stub_rho[LENGTH];  // Ignazio 
  GlobalVector _stub_direction[LENGTH];

  DTStubMatchPt Stubs_5_3_0;
  DTStubMatchPt Stubs_5_1_0;
  DTStubMatchPt Stubs_3_2_0;
  DTStubMatchPt Stubs_3_1_0;
  DTStubMatchPt Stubs_5_3_V; 
  DTStubMatchPt Stubs_5_0_V;
  DTStubMatchPt Stubs_3_0_V;
  DTStubMatchPt Mu_5_0;
  DTStubMatchPt Mu_3_0;
  DTStubMatchPt Mu_2_0; 
  DTStubMatchPt Mu_1_0; 
  DTStubMatchPt Mu_5_V;
  DTStubMatchPt Mu_3_V;
  DTStubMatchPt Mu_2_V;
  DTStubMatchPt Mu_1_V;
  DTStubMatchPt Mu_0_V;
  DTStubMatchPt IMu_5_0;
  DTStubMatchPt IMu_3_0;
  DTStubMatchPt IMu_2_0; 
  DTStubMatchPt IMu_1_0; 
  DTStubMatchPt IMu_5_V;
  DTStubMatchPt IMu_3_V;
  DTStubMatchPt IMu_2_V;
  DTStubMatchPt IMu_1_V;
  DTStubMatchPt IMu_0_V;
  DTStubMatchPt mu_5_0;
  DTStubMatchPt mu_3_0;  
  DTStubMatchPt mu_2_0;
  DTStubMatchPt mu_1_0;
  DTStubMatchPt mu_5_V;
  DTStubMatchPt mu_3_V;
  DTStubMatchPt mu_2_V;
  DTStubMatchPt mu_1_V;
  DTStubMatchPt mu_0_V;
  DTStubMatchPt only_Mu_V; 

  float  _alphaDT;  // tan(_alpha) is the angular coefficient of the muon trajectory 
                    // outside the inner surface of the CMS superconducting coil,  
                    // assuming that trajectory to be linear. 
  // Say double delta   = _phiCMS - _alphaDT; then:
  float _sqrtDscrm;    // sqrt(1. - rhoDT*rhoDT*sin(delta)*sin(delta)/(Erre*Erre)).
  // On the tranverse plane of CMS, (_X,_Y) is the intercept of the muon trajectory 
  // outside the inner surface of the CMS superconducting coil with that boundary surface.
  float _xerre, _yerre;
  float _phiR; 
  float _Xerre, _Yerre;    // _sqrtDscrm set to 1: it turns to be excellent approximation!
  float _deltaPhiR, _PhiR;// using deltaPhiR_over_bending  = 1. - _rhoDT/Erre;
  float _XerreI, _YerreI;


  float _deltaPhiR_over_bendingDT;

  float 
    _deltaPhiR_over_bendingDT_S1, 
    _deltaPhiR_over_bendingDT_S2;  
  float 
    _deltaPhiR_over_bendingDT_S1_0, 
    _deltaPhiR_over_bendingDT_S1_1,
    _deltaPhiR_over_bendingDT_S1_2;
  float
    _deltaPhiR_over_bendingDT_S2_0,
    _deltaPhiR_over_bendingDT_S2_1,
    _deltaPhiR_over_bendingDT_S2_2;

  float _deltaPhiL9_over_bendingDT;

  std::set<TrackerStub*, lt_stub> _matching_stubs; 
    
  size_t _matching_stubs_No;
  //  Ignazio end ****
  
  // rejection flags for redundancy cancellation
  bool _flag_reject;

};



typedef std::vector<DTStubMatch*> DTTracklet;


/*------------------*
 * a global method  *
 *------------------*/
bool DTStubMatchSortPredicate(const DTStubMatch* d1, const DTStubMatch* d2);

#endif
