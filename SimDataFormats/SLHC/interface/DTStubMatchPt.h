#ifndef __DT_PT_
#define __DT_PT_

#include <math.h>

#include <TMath.h>

#include <string>
#include <map>

#include "DataFormats/GeometryVector/interface/GlobalVector.h"  

#include "DataFormats/Common/interface/Ref.h"


using namespace std;


class DTStubMatch;
class DTStubMatchPtAlgorithms;


class DTStubMatchPt {
  
  friend class DTStubMatch;
  friend class DTStubMatchPtAlgorithms; 
 
 public:
  
  DTStubMatchPt() {
    _label = string();
    _Rb = NAN;
    _invRb = NAN;
    _Pt = NAN;
    _invPt = NAN;
    _alpha0 = _d = NAN;
  }
  
  DTStubMatchPt(std::string const s) {
    _label = s;
    _Rb = NAN;
    _invRb = NAN;
    _Pt = NAN;
    _invPt = NAN;
    _alpha0 = _d = NAN;
  }

  DTStubMatchPt(std::string const s, int station, 
		const edm::ParameterSet& pSet,
		const GlobalVector stub_position[],
		bool const flagMatch[]); 
  
  DTStubMatchPt(string const s, int station,
		const GlobalVector stub_position[],
		bool const flagMatch[],
		const edm::ParameterSet& pSet);
  
  DTStubMatchPt(std::string const s, int station, 
		const edm::ParameterSet& pSet,
		const float vstub_rho, const float vstub_phi,
		const GlobalVector stub_position[],
		bool const flagMatch[]); 
  
  DTStubMatchPt(std::string const s, int station,
		const edm::ParameterSet& pSet,
		float const DTmu_x, float const DTmu_y,
		float const stub_x[], float const stub_y[], 
		bool const flagMatch[]); 

  // using linear fit of stub dephi vs invPt
  DTStubMatchPt(std::string const s, 
		const float slope, const float dephi_zero,
		const int I, const int J, const float dephi, 
		const edm::ParameterSet& pSet, 
		bool const flagMatch[]); 


  // copy constructor  
  DTStubMatchPt(const DTStubMatchPt& aPt) {
    _label = aPt._label;
    _Rb = aPt._Rb;
    _invRb = aPt._invRb; 
    _Pt = aPt._Pt; 
    _invPt = aPt._invPt; 
    _alpha0 =  aPt._alpha0;
    _d =  aPt._d;
  }

  // assignment operator
  DTStubMatchPt& operator =(const DTStubMatchPt& aPt) {
  if (this == &aPt)      // Same object?
    return *this;        // Yes, so skip assignment, and just return *this.
    /*
    if(!isnan(aPt._Pt))
      cout << "DTStubMatchPt " << _label << " assignment operator called " << flush;
    */
    _label = aPt._label;
    _Rb = aPt._Rb;
    _invRb = aPt._invRb; 
    _Pt = aPt._Pt; 
    /*
    if(!isnan(_Pt))
      cout << _Pt << endl;
    */
    _invPt = aPt._invPt; 
    _alpha0 =  aPt._alpha0;
    _d =  aPt._d;
    return *this;
  }

  // destructor
  ~DTStubMatchPt() {
    _label.clear();
  }
  
  std::string const &label() const      { return _label; }
  float const Rb() const                { return _Rb; }
  float const invRb() const             { return _invRb; }
  float const Pt() const                { return _Pt; }
  float const invPt() const             { return _invPt; }
  float const alpha0() const            { return _alpha0; }
  float const d() const                 { return _d; }
 private:
  
  std::string _label;
  float _Rb, _invRb;
  float _Pt, _invPt;
  float _alpha0, _d;

  inline float phiCMS(const GlobalVector& P) const
    { 
      float phiCMS = P.phi();
      if(phiCMS < 0.)
	phiCMS += 2. * TMath::Pi();
      if(phiCMS > 2*TMath::Pi())
	phiCMS -= 2 * TMath::Pi();
      return phiCMS;
    }

  void computePt(const edm::ParameterSet& pSet, 
	     float const X[], float const Y[], float const corr = 0.0); 
  void radius_of_curvature(const edm::ParameterSet& pSet, 
			   float const x[], float const y[]);

  void computePt(const edm::ParameterSet& pSet, 
	     const GlobalVector P[], 
	     float const corr = 0.0); 
  void radius_of_curvature(const edm::ParameterSet& pSet, 
			   const GlobalVector P[]);

  void computePt(const edm::ParameterSet& pSet, 
	     const float vstub_rho, const float vstub_phi, 
	     const GlobalVector P[], 
	     float const corr = 0.0); 
  void radius_of_curvature(const edm::ParameterSet& pSet,
			   const float vstub_rho, const float vstub_phi,
			   const GlobalVector P[]);

  void computePt_etc(const edm::ParameterSet& pSet, 
		     const GlobalVector P[], 
		     float const corr = 0.0); 

};



#endif


