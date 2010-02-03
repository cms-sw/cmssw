#ifndef _DTSeededTracklet__
#define _DTSeededTracklet__

/************************************************************

   \class DTSeededTracklet
   
   Description: stacked tracker tracklet having a core of
   at least three stubs all matched by DTStubMatch objects.

   Sept. 2009            
   
   I. Lazzizzera - Trento University

*************************************************************/
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <algorithm>
#include <vector>

//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTStubMatch.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTTrackerStub.h"
#include "SimDataFormats/SLHC/interface/DTStubMatch.h"
#include "SimDataFormats/SLHC/interface/DTTrackerStub.h"

using namespace std;


//const size_t STATIONS = 4;


class DTSeededTracklet
{
 public:

  // default trivial constructor
  DTSeededTracklet() { 
    _theDTTracklet = DTTracklet();
    _theStubTracklet = StubTracklet(); 
    _theCoreStubTracklet = StubTracklet(); 
    _init();
  }

  // constructor
  DTSeededTracklet(DTStubMatch* dtm) { 
    /**
       Start by a DTStubMatch object, then upgrade adding those 
       DTStubMatch objects sharing at least three matching stubs;
       different DTSeededTracklet objects have disjoint sets of 
       matching stubs. 
    **/
    _theDTTracklet = DTTracklet();
    _theDTTracklet.push_back(new DTStubMatch(*dtm)); 
    _theStubTracklet = StubTracklet();
    _theCoreStubTracklet = StubTracklet();
    StubTracklet::const_iterator st = dtm->getMatchingStubs().begin();
    for(size_t i=0; i<dtm->getMatchingStubs().size(); i++) {
      _theStubTracklet.insert(new TrackerStub(**st));
      _theCoreStubTracklet.insert(new TrackerStub(**st));
      ++st;
    }
    _theDTTracklet_size = _theDTTracklet.size();
    _theStubTracklet_size = _theStubTracklet.size();
    _theCoreStubTracklet_size = _theCoreStubTracklet.size();
    _init();
    ++_DTSeededTrackletsCollectionSize;
  }

  // copy constructor
  DTSeededTracklet(const DTSeededTracklet&);  

  void sort() { 
    std::sort(_theDTTracklet.begin(), _theDTTracklet.end(), DTStubMatchSortPredicate);
  }

  const DTTracklet& theDTTracklet() const {
    return _theDTTracklet;
  }
  void set_theDTTracklet_size() {
    _theDTTracklet_size = _theDTTracklet.size();
  }
  size_t const theDTTracklet_size() const {
    return _theDTTracklet.size();
  }
  const StubTracklet theStubTracklet() const {
    return _theStubTracklet;
  }
  void set_theStubTracklet_size() {
    _theStubTracklet_size = _theStubTracklet.size();
  }
  size_t const theStubTracklet_size() const {
    return _theStubTracklet.size();
  }
  const StubTracklet theCoreStubTracklet() const {
    return _theCoreStubTracklet;
  }
  void set_theCoreStubTracklet_size() {
    _theCoreStubTracklet_size = _theCoreStubTracklet.size();
  }
  size_t const theCoreStubTracklet_size() const {
    return _theCoreStubTracklet.size();
  }

  // the destructor
  ~DTSeededTracklet() {

    _theDTTracklet.clear();
    _theDTTracklet_size = 0;
    _theStubTracklet.clear();
    _theStubTracklet_size = 0;
    _theCoreStubTracklet.clear();
    _theCoreStubTracklet_size = 0;
    _DTSeededTrackletsCollectionSize = 0;
    _theAllStubsPtVariety.clear();
    _theDTmuAndStubsPtVariety.clear();
    _theDTMuAndStubsPtVariety.clear();
    _theDTIMuAndStubsPtVariety.clear();

  }

  void update(DTStubMatch* dtm, 
	      StubTracklet* tracklet, StubTracklet* core_tracklet) {
    _theDTTracklet.push_back(new DTStubMatch(*dtm));
    _theStubTracklet = StubTracklet(*tracklet); 
    _theCoreStubTracklet = StubTracklet(*core_tracklet); 
  }
 
  static size_t getDTSeededTrackletsCollectionSize() {
    return _DTSeededTrackletsCollectionSize;
  }

  static void reset_DTSeededTrackletsCollectionSize() {
    _DTSeededTrackletsCollectionSize = 0;
  }
  

  private:

  void          _init();

  /******************************************************************************
    This is the main data member: a collection of DTStubMatch objects having at 
    least three stubs in common; the component DTStubMatch objects are ordered
    according to DT trigger quality.
  *******************************************************************************/
  DTTracklet    _theDTTracklet; 
  size_t        _theDTTracklet_size;
  /******************************************************************************/

  // set union of the DTTracklet stubs   
  StubTracklet  _theStubTracklet; 
  size_t        _theStubTracklet_size;
  // set intersection of the DTTracklet stubs   
  StubTracklet  _theCoreStubTracklet; 
  size_t        _theCoreStubTracklet_size;

  // All of the following are built using the highest quality DTStubMatch object.
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

  static size_t _DTSeededTrackletsCollectionSize; 

  // The following containers suitably collect the DTStubMatchPt object above.
  PtVariety _theAllStubsPtVariety;
  PtVariety _theDTmuAndStubsPtVariety;
  PtVariety _theDTMuAndStubsPtVariety;
  PtVariety _theDTIMuAndStubsPtVariety;

  // All the following belongs to the core stub tracklet 
  float _stub_x[LENGTH], _stub_y[LENGTH], _stub_z[LENGTH]; 
  float _stub_rho[LENGTH], _stub_phi[LENGTH], _stub_theta[LENGTH];
  bool _flagMatch[LENGTH];
  GlobalVector _stub_direction[LENGTH]; 

 public:

  //  void setPt(const edm::ParameterSet& pSet);
  void setPt(const edm::ParameterSet& pSet);
  const DTStubMatchPt& getPtOBJ(std::string const label) const; 
  float const Pt(std::string const label) const; 

};
 





//--------------------------------------------------------------------------------------
class DTSeededTrackletsCollection
{

  typedef std::vector<DTSeededTracklet*>::iterator _iterator;

 public:
  // trivial constructor
  DTSeededTrackletsCollection() {
    _theCollection = std::vector<DTSeededTracklet*>();
  }
  // default constructor
  DTSeededTrackletsCollection(std::vector<DTSeededTracklet*>& tracklets) {
    _theCollection = std::vector<DTSeededTracklet*>(tracklets);
  }

  void clear() { _theCollection.clear(); }   
  // destructor
  ~DTSeededTrackletsCollection() { _theCollection.clear(); } 
  
  void push_back(DTSeededTracklet* aTracklet) {
    _theCollection.push_back(aTracklet);
  }

  void insert(_iterator pos, DTSeededTracklet* aTracklet) {
    _theCollection.insert(pos, aTracklet);
  }

  void erase(_iterator pos) { _theCollection.erase(pos); }

  //return functions  
  const std::vector<DTSeededTracklet*>& theCollection() const { return _theCollection; }

  DTSeededTracklet* operator[](size_t n) {
    if( n >= _theCollection.size() ) 
      cerr << "index out of range" << endl;
    return _theCollection[n]; 
  }
  
  size_t size() const { return _theCollection.size(); }

 private:

  std::vector<DTSeededTracklet*> _theCollection;

};


#endif


