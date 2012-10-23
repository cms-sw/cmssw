//-------------------------------------------------------------------
//
//   \class DTMatch
/**
 *   Description:  Bti triggers matched between phi and theta view
 *                 are extrapolated to stacked tracker stubs. 
*/
//   090202             
//   Sara Vanini - Padua University
//   I. Lazzizzera - Trento University
//
//-------------------------------------------------------------------
#ifndef DTMatch_H
#define DTMatch_H

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
#include "SimDataFormats/SLHC/interface/DTMatchPtVariety.h"    //Ignazio
#include "SimDataFormats/SLHC/interface/DTMatchPtAlgorithms.h" //Ignazio

//#include "SimDataFormats/SLHC/interface/GlobalStub.h"
//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

  
using namespace std;


//              ---------------------
//              -- Class Interface --
//              ---------------------


static size_t const RTSdataSize(16);


class DTMatch: public DTMatchPtAlgorithms 
{
  /*
    Objects of this class do correspond to DT muons that are then extrapolated
    to the stubs on the tracker layers, including a virtual one just enclosing  
    the magnetic field volume. The main methos do aim at  getting a tracker 
    precision Pt.
    The matching stubs and tracker tracklets, indexed by tracker layer id, are 
    set as data members of the base class DTMatchPtAlgorithms. 
    Several objects of class DTMatchPt are built by methods defined in the 
    virtual base class DTMatchPtAlgorithms.
    Method encoderPt() chooses the closest stubs on layer 2 or 3 and 
    layer 0 or 1, computing the Pt accordingly, while averagePt() gives the 
    average over all of the DTMatchPt available objects.
  */

  // 6.5.2010 PLZ : to use Stacked Tracker PTFlag 
  //  typedef GlobalStub<Ref_PixelDigi_>  GlobalStubRefType;

 public:
  
  // trivial default constructor needed by ROOT dictionary (Ignazio)
  DTMatch();
  
  // constructor
  DTMatch(int wheel, int station, int sector,
	      int bx, int code, int phi, int phib, float theta, bool flagBxOK,
	      bool debug_dttrackmatch = false);
  
  // constructor
  DTMatch(int wheel, int station, int sector,
	      int bx, int code, int phi, int phib, float theta, 
	      GlobalPoint position, GlobalVector direction,
	      bool flagBxOK, bool debug_dttrackmatch = false) ;
  
  // copy constructor
  DTMatch(const DTMatch& dtsm);
  
  // assignment operator
  DTMatch& operator =(const DTMatch& dtsm);
  
  // destructor
  ~DTMatch(){ } 
  
  void to_get_pt() {}

  //return functions
  inline int 	wheel() 	      const { return _wheel; }
  inline int	station()	      const { return _station; }
  inline int 	sector() 	      const { return _sector; }
  inline int 	bx() 		      const { return _bx; }
  inline int	code() 		      const { return _code; }
  inline int 	phi_ts()	      const { return _phi_ts; }
  inline int 	phib_ts()	      const { return _phib_ts; }
  inline int 	theta_ts()	      const { return _theta_ts; }
  inline float  gunFiredSingleMuPt()  const { return _GunFiredSingleMuPt; }
  inline float 	phi_glo()             const
    { 
      float phiCMS = static_cast<float>(_phi_ts)/4096.+(_sector-1)*TMath::Pi()/6.;
      if(phiCMS <= 0.)
	phiCMS += 2. * TMath::Pi();
      if(phiCMS > 2*TMath::Pi())
	phiCMS -= 2 * TMath::Pi();
      return phiCMS;
    }
  inline float rho()                  const {return _rho;};
  inline GlobalPoint CMS_Position()   const {return _position;};

  inline float 	phib_glo()            const
    { 
      return static_cast<float>(_phib_ts)/512.;                        	      // 9 bits
    }

  inline bool 	flagBxOK() 	      const { return _flagBxOK; }
  inline int    trig_order()	      const { return _trig_order; }            // PLZ 
  inline float  Pt_encoder()	      const { return _Pt_encoder; }            // PLZ
  inline float  Pt_encoder_bin()      const { return _Pt_encoder_bin; }        // PLZ 
  inline float  Pt_average()	      const { return _Pt_average; }            // PLZ
  inline float  Pt_average_Tracklet()	      const { return _Pt_average_Tracklet; }            // PLZ
  inline float  Pt_average_bin()      const { return _Pt_average_bin; }        // PLZ 
  inline float  Pt_average_bin_Tracklet()      const { return _Pt_average_bin_Tracklet; }        // PLZ 
  inline float  Pt_majority_bin()     const { return _Pt_majority_bin; }       // PLZ 
  inline float  Pt_majority_bin_Tracklet()     const { return _Pt_majority_bin_Tracklet; }       // PLZ 
  inline float  Pt_majority_bin_Full()     const { return _Pt_majority_bin_Full; }       // PLZ 
  inline float  Pt_mixedmode_bin()     const { return _Pt_mixedmode_bin; }       // PLZ 
  inline float  Pt_mixedmode_bin_Tracklet()     const { return _Pt_mixedmode_bin_Tracklet; }       // PLZ 
  inline float  Pt_matching_track()     const { return _Pt_matching_track; }       // PLZ      // PLZ 
  inline float  Pt_matching_track_bin()     const { return _Pt_matching_track_bin; }       // PLZ 
  inline int    predPhi(int lay)      const { return _pred_phi[lay]; }	       // 12 bits
  inline int  	predSigmaPhi(int lay) const { return _pred_sigma_phi[lay]; }   // 12 bits
  inline int    predPhiVx()      const { return _pred_phi_vx; }	       // 12 bits
  inline int  	predSigmaPhiVx() const { return _pred_sigma_phi_vx; }   // 12 bits
  inline int  	predTheta()	      const { return _pred_theta; }            // 12 bits
  inline int  	predSigmaTheta(int lay) const { return _pred_sigma_theta[lay];}// 12 bits
  inline int  	predSigmaThetaVx() const { return _pred_sigma_theta_vx;}// 12 bits
  inline float  predSigmaPhiB()	      const { return _pred_sigma_phib; }       // 12 bits
  inline int  	stubPhi(int lay)      const { return _stub_phi[lay]; }
  inline float  stubDePhi(int lay)    const { return _stub_dephi[lay]; }
  inline int  	stubTheta(int lay)    const { return _stub_theta[lay]; }
  inline bool   isMatched(int lay)    const { return _flagMatch[lay]; }
  inline bool   flagReject()	      const { return _flag_reject; }
	inline bool   flagPt()	      const { return _flagPt; }                // PLZ 
	inline bool   flagPtTracklet()	      const { return _flagPtTracklet; }                // PLZ 
  inline bool   flagTheta()	      const { return _flag_theta; }            // PLZ 
  inline int    deltaTheta()	      const { return _delta_theta; }           // PLZ
  

  inline void setGunFiredSingleMuPt(const float Pt) {
    _GunFiredSingleMuPt = Pt;
  }

  //set theta existence flag
  inline void setTheta(float deltatheta) { 
    _flag_theta = false; 
    _delta_theta = static_cast<int>(deltatheta*4096./3.); 
    return; 
  }

  //set rejection flag
  inline void setRejection(bool flag) { 
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
  
  //set predicted tracker phi and theta at vertex
  inline void setPredStubPhi(int phi, int sigma_phi) { 	
    _pred_phi_vx = phi; 
    _pred_sigma_phi_vx = sigma_phi; 
    return; 
  } 

  inline void setPredStubTheta(int theta, int sigma_theta) { 	
    _pred_theta = theta; 
    _pred_sigma_theta_vx = sigma_theta; 
    return; 
  } 

  inline void setPredSigmaPhiB(float sigma_phib) {
    _pred_sigma_phib = sigma_phib;
    return; 
  }

  inline void setMatchingStub(int lay, int phi, int theta) { 
    _stub_phi[lay] = phi;
    _stub_theta[lay] = theta; 
    _flagMatch[lay] = true; 
    return; 
  }

  inline void insertMatchingStubObj(TrackerStub* st) { 
    // Ignazio
    /*
      For each DTMatch object, that is a phi-theta matched DT trigger, one has
      in principle a ("closest") matching stub per tracker layer.
      By the method setDTSeededStubTracks belonging to the DTL1SimOperations 
      class, called by getDTPrimitivesToTrackerObjectsMatches, also a method 
      belonging to the same DTL1SimOperations class, one looks for DTMatch 
      objects sharing the matching stub on at least three different tracker 
      layers: in this case such DTMatch objects are supposed to belong to the 
      same muon and the Pt of the highest rank of them is chosen.
      The present method inserts the matching stub objects into a sorted set, 
      namely std::set<TrackerStub*, lt_stub> _matching_stubs (see below), used 
      to do the set intersection operations.
    */
    _matching_stubs.insert(st);
  }

  inline const size_t matchingStubsTotal() const { 
    // Ignazio
    return _matching_stubs.size();
  }

  inline const DTMatchingStubSet& getMatchingStubs() const { 
    // Ignazio
    return _matching_stubs;
  }

  void setMatchingStub(int lay, int phi, int theta, 
		    GlobalVector position, GlobalVector direction);   // Ignazio

  inline void setMatchingStubPhi(int lay, int phi)	 { 	
    _stub_phi[lay] = phi; 
    _flagMatch[lay] = true; 
    return; 
  } 

  inline void setMatchingStubTheta(int lay, int theta) {
    _stub_theta[lay] = theta; 
    _flagMatch[lay] = true; 
    return; 
  } 

  void setMatchingTkTracklet(size_t superLayer, TrackerTracklet* aTracklet) {
    /* 
       Store the "closest" TkTracklet of those on that superlayer that are
       within a window extrapolated from a DTMatch object.
    */
    _MatchingTkTracklets[superLayer] = aTracklet;
  }

  TrackerTracklet* getMatchingTkTracklet(size_t superLayer) {
    return _MatchingTkTracklets[superLayer];
  }

  //set Pt value from priority encoder (PLZ; Ignazio modifications)
  // indeed set invPt: a lil' bit confusing denomination of methods!!!
  void encoderPt();
  void averagePt();
  void averagePtTracklet();
  std::string writePhiStubToPredictedDistance() const;
  void assign_encoderPtBin();
  void assign_averagePtBin();
  void assign_averagePtBinTracklet();
  int  assign_ptbin(float invPt,int stat);
  int  assign_L1track_ptbin(float invPt);
  void assign_majorityPtBin();
  void assign_majorityPtBinFull();
  void assign_majorityPtBinTracklet();
  void assign_mixedmodePtBin();
  void assign_mixedmodePtBinTracklet();
  
  //set Pt value from priority encoder (PLZ)
  inline void setPtEncoder(float invPt_value) { 
    _Pt_encoder = 1./invPt_value; 
    _flagPt = true;
    return; 
  } 
  //set Pt average of available measurements (PLZ)
  inline void setPtAverage(float invPt_value) {        
    _Pt_average = 1./invPt_value; 
    _flagPt = true;
    return; 
  } 
	
//set Pt average of available measurements (PLZ)
  inline void setPtAverageTracklet(float invPt_value) {        
  _Pt_average_Tracklet = 1./invPt_value; 
  _flagPtTracklet = true;
	return; 
	} 

  //assign Pt bin after priority encoder choice  (PLZ)
  inline void set_encoderPtBin(float Pt_bin) { 
    _Pt_encoder_bin = Pt_bin; 
    return; 
  }
  
  //assign Pt bin after averaging (PLZ)
  inline void set_averagePtBin(float Pt_bin) { 
    _Pt_average_bin = Pt_bin; 
    return; 
  }
	
	//assign Pt bin after averaging (PLZ)
	inline void set_averagePtBinTracklet(float Pt_bin) { 
		_Pt_average_bin_Tracklet = Pt_bin; 
		return; 
	}
  
  //assign Pt bin with majority (PLZ) - inner layers only
  inline void set_majorityPtBin(int Pt_bin) { 
    _Pt_majority_bin = Pt_bin; 
    return; 
  }
	
  //assign Pt bin with majority (PLZ) - Full longbarrel version
  inline void set_majorityPtBinFull(int Pt_bin) { 
	_Pt_majority_bin_Full = Pt_bin; 
	return; 
}	
  //assign Pt bin with majority (PLZ) - Tracklets
  inline void set_majorityPtBinTracklet(int Pt_bin) { 
	_Pt_majority_bin_Tracklet = Pt_bin; 
	return; 
}
	//assign Pt bin with majority+average (PLZ) - Inner longbarrel layers version
	inline void set_mixedmodePtBin(int Pt_bin) { 
		_Pt_mixedmode_bin = Pt_bin; 
		return; 
	}	
	//assign Pt bin with majority+average (PLZ) - Tracklets
	inline void set_mixedmodePtBinTracklet(int Pt_bin) { 
		_Pt_mixedmode_bin_Tracklet = Pt_bin; 
		return; 
	}
	
   // set Pt of best matching tracker L1 Track
   inline void setPtMatchingTrack(float pt){
         _Pt_matching_track = pt;
         _Pt_matching_track_bin = 0;
	 if (pt < 10000) _Pt_matching_track_bin = assign_L1track_ptbin(1/pt);
	 
   }
	
	

  int  DTMatch_PT(int station,int wheel, float phib);
  int  DTMatch_PTMin(int station,int wheel, float phib);
  int  DTMatch_PTMax(int station,int wheel, float phib);
  int  DTMatch_PTMin(int station, float invPT);
  int  DTMatch_PTMax(int station, float invPT);
  bool DTStubPTMatch(int DTPTMin,int DTPTMax,int TKPTMin,int TKPTMax);

  void extrapolateToTrackerLayer(int l);
  void extrapolateToVertex();

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

  float _GunFiredSingleMuPt; // to make easier analysis on gun fired muon samples
  
  //  int   _wheel, _station, _sector, _bx, _code;
  int   _phi_ts, _phib_ts; 
  int   _theta_ts;
  bool  _flagBxOK; 
  int   _trig_order;
  float _Pt_encoder;          // PLZ
	bool  _flagPt;              // PLZ
	bool  _flagPtTracklet;              // PLZ
  float _Pt_encoder_bin;      // PLZ
  float _Pt_average;          // PLZ
  float _Pt_average_Tracklet;          // PLZ
  float _Pt_average_bin;      // PLZ
  float _Pt_average_bin_Tracklet;      // PLZ
  float _Pt_majority_bin;     // PLZ
  float _Pt_majority_bin_Tracklet;     // PLZ
  float _Pt_majority_bin_Full;     // PLZ
  float _Pt_mixedmode_bin_Tracklet;     // PLZ
  float _Pt_mixedmode_bin;     // PLZ
  float _Pt_matching_track;     // PLZ
  float _Pt_matching_track_bin;     // PLZ
  
  // predicted phi, theta, sigma_phi, sigma_theta, sigma_phib in tracker layers 
  // (NB wheel dependent!)
  int   _pred_phi[StackedLayersInUseTotal];
  int   _pred_sigma_phi[StackedLayersInUseTotal];
  int   _pred_phi_vx;
  int   _pred_sigma_phi_vx;
  int   _pred_theta;
  int   _pred_sigma_theta[StackedLayersInUseTotal];
  int   _pred_sigma_theta_vx;
  float _pred_sigma_phib;
  
  // Set of matching stacked tracker stubs, used to get the Pt: for each tracker 
  // layer we take the stub which is the closest to a "predicted" position, out from 
  // those included in a suitable window around this "predicted" position.
  // Different DTMatch objects sharing at least three matching stubs are
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



//*******************************************************************************
typedef std::vector<DTMatch*> DTMatchesVector;//DTTracklet;  



/*----------------------------------------------------------------------------*/
/*                          global methods                                    */
/*----------------------------------------------------------------------------*/

bool DTMatchSortPredicate(const DTMatch* d1, const DTMatch* d2);


ostream& operator <<(ostream &os, const DTMatch &obj);


#endif

