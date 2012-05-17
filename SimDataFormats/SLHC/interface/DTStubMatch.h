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

#include "FWCore/ParameterSet/interface/ParameterSet.h"            //Ignazio
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"      //Ignazio
#include "DataFormats/GeometryVector/interface/GlobalVector.h"     //Ignazio

#include "SimDataFormats/SLHC/interface/DTTrackerStub.h"           //Ignazio
#include "SimDataFormats/SLHC/interface/DTStubMatchPtVariety.h"    //Ignazio
#include "SimDataFormats/SLHC/interface/DTStubMatchPtAlgorithms.h" //Ignazio

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

  
using namespace std;


//              ---------------------
//              -- Class Interface --
//              ---------------------


static size_t const RTSdataSize(16);


class DTStubMatch: public DTStubMatchPtAlgorithms 
{

<<<<<<< DTStubMatch.h
  // 6.5.2010 PLZ : to use Stacked Tracker PTFlag 
  // WARNING NP** typedef GlobalStub<Ref_PixelDigi_>  GlobalStubRefType;
=======
  // 6.5.2010 PLZ : to use Stacked Tracker PTFlag 
  typedef GlobalStub<Ref_PixelDigi_>  GlobalStubRefType;
>>>>>>> 1.3

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

  // assignment operator
  DTStubMatch& operator =(const DTStubMatch& dtsm);

  // destructor
  ~DTStubMatch(){ } 
  
  void to_get_pt() {}

  //return functions
  inline int 	wheel() 		const { return _wheel; }
  inline int	station()		const { return _station; }
  inline int 	sector() 		const { return _sector; }
  inline int 	bx() 			const { return _bx; }
  inline int	code() 			const { return _code; }
  inline int 	phi_ts()		const { return _phi_ts; }
  inline int 	phib_ts()		const { return _phib_ts; }
  inline int 	theta_ts()		const { return _theta_ts; }
  inline float  gunFiredSingleMuPt()    const { return _GunFiredSingleMuPt; }
  inline float 	phi_glo()               const
    { 
      float phiCMS = static_cast<float>(_phi_ts)/4096.+(_sector-1)*TMath::Pi()/6.;
      if(phiCMS <= 0.)
	phiCMS += 2. * TMath::Pi();
      if(phiCMS > 2*TMath::Pi())
	phiCMS -= 2 * TMath::Pi();
      return phiCMS;
    }

  inline float 	phib_glo()              const
    { 
      return static_cast<float>(_phib_ts)/512.;                          	// 9 bits
    }

  inline bool 	flagBxOK() 		const { return _flagBxOK; }
  inline int    trig_order()		const { return _trig_order; }           // PLZ 
  inline float  Pt_value()		const { return _Pt_value; }             // PLZ
  inline float  Pt_bin()		const { return _Pt_bin; }               // PLZ  
  inline int    predPhi(int lay)	const { return _pred_phi[lay]; }	// 12 bits
  inline int  	predSigmaPhi(int lay)	const { return _pred_sigma_phi[lay]; }	// 12 bits
  inline int  	predTheta()	        const { return _pred_theta; }   	// 12 bits
  inline int  	predSigmaTheta(int lay) const { return _pred_sigma_theta[lay]; }// 12 bits
  inline float  predSigmaPhiB()		const { return _pred_sigma_phib; }	// 12 bits
  inline int  	stubPhi(int lay)	const { return _stub_phi[lay]; }
  inline float  stubDePhi(int lay)      const { return _stub_dephi[lay]; }
  inline int  	stubTheta(int lay)	const { return _stub_theta[lay]; }
  inline bool   isMatched(int lay)	const { return _flagMatch[lay]; }
  inline bool   flagReject()		const { return _flag_reject; }
  inline bool   flagPt()		const { return _flagPt; }                // PLZ 
  inline bool   flagTheta()		const { return _flag_theta; }            // PLZ 
  inline int    deltaTheta()		const { return _delta_theta; }           // PLZ
  

  inline void setGunFiredSingleMuPt(const float Pt) {
    _GunFiredSingleMuPt = Pt;
  }

  //set theta existence flag
  inline void setTheta(float deltatheta)
    { 
      _flag_theta = false; 
      _delta_theta = static_cast<int>(deltatheta*4096./3.); 
      return; 
    }

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
		    GlobalVector position, GlobalVector direction);   // Ignazio

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

  //set Pt value from priority encoder (PLZ; Ignazio modifications)
  void choosePtValue();
  std::string writePhiStubToPredictedDistance() const;
  void assignPtBin();

  //set Pt value from priority encoder (PLZ)
  inline void setPtValue(float invPt_value) { 
    _Pt_value = 1./invPt_value; 
    _flagPt = true;
    return; 
  } 

  //assign Pt bin after priority encoder choice  (PLZ)
  inline void setPtBin(float Pt_bin) { 
    _Pt_bin = Pt_bin; 
    return; 
  }

  void extrapolateToTrackerLayer(int l);

  size_t matching_stubs_No()  const { return _matching_stubs_No; }    // Ignazio

  // SV 090505 correlate phib and error in station 1 to phib in station 2, 
  // for track rejection
  int corrPhiBend1ToCh2(int phib2);
  int corrSigmaPhiBend1ToCh2(int phib2, int sigma_phib2);

  // debug functions
  void print();  
  std::string writeMatchingStubs() const;
  std::string writeMatchingStubs(size_t) const;
  // end debug functions

  // debug flags
  bool _debug_dttrackmatch;

  void setRTSdata(size_t i, short datum) { _RTSdata[i] = datum; }
  int  RTSdata(size_t i) const { 
    if( i < RTSdataSize ) 
      return _RTSdata[i];
    else {
      cerr << "error: RTSdataSize exceeded" << endl; 
      return 9999; 
    }
  }

 private:
  
  void init();

  float _GunFiredSingleMuPt;
  
  //  int   _wheel, _station, _sector, _bx, _code;
  int   _phi_ts, _phib_ts; 
  int   _theta_ts;
  bool  _flagBxOK; 
  int   _trig_order;
  float _Pt_value;    // PLZ
  bool  _flagPt;      // PLZ
  float _Pt_bin;      // PLZ
  
  // predicted phi, theta, sigma_phi, sigma_theta, sigma_phib in tracker layers 
  // (NB wheel dependent!)
  int   _pred_phi[LENGTH];
  int   _pred_sigma_phi[LENGTH];
  int   _pred_theta;
  int   _pred_sigma_theta[LENGTH];
  float _pred_sigma_phib;
  
  // Set of matching stacked tracker stubs, used to get the Pt: for each tracker 
  // layer we take the stub which is the closest to a "predicted" position, out from 
  // those included in a suitable window around this "predicted" position.
  // Different DTStubMatch objects sharing at least three matching stubs are
  // assumed to belong to the same muon: such shared stubs are obtained by set
  // intersection of their matching stubs. 
  std::set<TrackerStub*, lt_stub> _matching_stubs;        // Ignazio

  // rejection flags for redundancy cancellation
  bool _flag_reject;
  // flag if theta missing
  bool _flag_theta;
  int _delta_theta; 

  short _RTSdata[RTSdataSize]; // data to feed RTSable neural network  (Ignazio)

};



typedef std::vector<DTStubMatch*> DTTracklet;


/*----------------------------------------------------------------------------*/
/*                          global methods                                    */
/*----------------------------------------------------------------------------*/

bool DTStubMatchSortPredicate(const DTStubMatch* d1, const DTStubMatch* d2);


ostream& operator <<(ostream &os, const DTStubMatch &obj);


#endif

