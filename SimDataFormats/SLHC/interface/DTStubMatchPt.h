#ifndef __DT_PT_
#define __DT_PT_

#include <math.h>

#include <string>
#include <map>

#include "DataFormats/Common/interface/Ref.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTStubMatch.h"
#include "SimDataFormats/SLHC/interface/DTStubMatch.h"


using namespace std;

class DTStubMatchPt {
  
  friend class DTStubMatch;
  
 public:
  
  DTStubMatchPt() {
    _label = string();
    _Rb = NAN;
    _invRb = NAN;
    _Pt = NAN;
    _invPt = NAN;
  }
  
  DTStubMatchPt(std::string const s) {
    _label =s;
    _Rb = NAN;
    _invRb = NAN;
    _Pt = NAN;
    _invPt = NAN;
  }
  
  DTStubMatchPt(std::string const s, int station, 
		const edm::ParameterSet& pSet,
		float const stub_x[], float const stub_y[], 
		bool const flagMatch[]); 
  
  DTStubMatchPt(std::string const s, int station,
		const edm::ParameterSet& pSet,
		float const DTmu_x, float const DTmu_y,
		float const stub_x[], float const stub_y[], 
		bool const flagMatch[]); 
  
  DTStubMatchPt(int station, 
		const edm::ParameterSet& pSet,
		float const bendingDT, float const Rb);
  
  // copy constructor 
  DTStubMatchPt(const DTStubMatchPt& aPt) {
    _label = aPt.label();
    _Rb = aPt.Rb();
    _invRb = aPt.invRb(); 
    _Pt = aPt.Pt(); 
    _invPt = aPt.invPt(); 
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

 private:
  
  std::string _label;
  float _Rb, _invRb;
  float _Pt, _invPt;

  void setPt(const edm::ParameterSet& pSet, 
	     float const X[], float const Y[], float const corr = 0.0); 
  void radius_of_curvature(const edm::ParameterSet& pSet, 
			   float const x[], float const y[]);
};



typedef std::vector<DTStubMatchPt> PtVariety;


//-------------------------------------------------------------------------------
class DTStubMatchPtVariety
{

  friend class DTStubMatch;

  typedef std::vector<DTStubMatchPt>::iterator _iterator;

 public:
  // trivial constructor
  DTStubMatchPtVariety() {
    _theVariety = std::vector<DTStubMatchPt>();
  }
  // default constructor
  DTStubMatchPtVariety(std::vector<DTStubMatchPt>& tracklets) {
    _theVariety = std::vector<DTStubMatchPt>(tracklets);
  }
 // copy constructor 
  DTStubMatchPtVariety(const DTStubMatchPtVariety& aPt) {
    _theVariety = aPt._theVariety;
  }

  void clear() { _theVariety.clear(); }
  // destructor
  ~DTStubMatchPtVariety() { clear(); } 
  
  void push_back(DTStubMatchPt aPt) { _theVariety.push_back(aPt); }

  void insert(_iterator pos, DTStubMatchPt aPt) { 
    _theVariety.insert(pos, aPt); 
  }

  void erase(_iterator pos) { _theVariety.erase(pos); }

  //return functions  
  const std::vector<DTStubMatchPt>& theVariety() const { return _theVariety; }

  DTStubMatchPtVariety& operator=(const DTStubMatchPtVariety& aPtVariety) {
    _theVariety = aPtVariety._theVariety;
    return *this;
  }

  DTStubMatchPt operator[](size_t n) {
    if( n >= _theVariety.size() ) 
      cerr << "index out of range" << endl;
    return _theVariety[n]; 
  }
  
  size_t size() const { return _theVariety.size(); }

 private:

  std::vector<DTStubMatchPt> _theVariety;

};





#endif

