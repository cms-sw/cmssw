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

#include "SimDataFormats/SLHC/interface/DTStubMatch.h"
#include "SimDataFormats/SLHC/interface/DTTrackerStub.h"
#include "SimDataFormats/SLHC/interface/DTStubMatchPtAlgorithms.h"


using namespace std;


//const size_t STATIONS = 4;


class DTSeededTracklet: public DTStubMatchPtVariety {
  
 public:
  
  // default trivial constructor
  DTSeededTracklet() { 
    _Pt_value = NAN;    // Ignazio
    _Pt_bin = NAN;      // Ignazio
    _theDTTracklet = DTTracklet();
    _theStubTracklet = StubTracklet(); 
    _theCoreStubTracklet = StubTracklet(); 
  }
  
  // the constructor
  DTSeededTracklet(DTStubMatch* dtm); 
  // copy constructor
  DTSeededTracklet(const DTSeededTracklet&);  
  // assignment operator
  DTSeededTracklet& operator =(const DTSeededTracklet&);
  
  void to_get_pt() {}  
  
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

  inline float  Pt_value()  const { return _Pt_value; }             
  inline float  Pt_bin()    const { return _Pt_bin; }     
  inline bool   flagPt()    const { return _flagPt; }              

  // the destructor
  ~DTSeededTracklet() {
    _theDTTracklet.clear();
    _theDTTracklet_size = 0;
    _theStubTracklet.clear();
    _theStubTracklet_size = 0;
    _theCoreStubTracklet.clear(); 
    _theCoreStubTracklet_size = 0;
    _DTSeededTrackletsCollectionSize = 0;
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

  float _Pt_value;  
  bool  _flagPt;    
  float _Pt_bin;    

  /******************************************************************************
    This is the main data member: a collection of DTStubMatch objects having at 
    least three stubs in common; the component DTStubMatch objects are ordered
    according to DT trigger quality.
  *******************************************************************************/
  DTTracklet    _theDTTracklet; 
  size_t        _theDTTracklet_size;
  
  // set union of the DTTracklet stubs   
  StubTracklet  _theStubTracklet; 
  size_t        _theStubTracklet_size;
  // set intersection of the DTTracklet stubs   
  StubTracklet  _theCoreStubTracklet; 
  size_t        _theCoreStubTracklet_size;
  /******************************************************************************/
  
  static size_t _DTSeededTrackletsCollectionSize; 
  
  
 public:
  
  void setPt(const edm::ParameterSet& pSet);
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


