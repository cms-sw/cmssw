#ifndef _DTSeededTracklet__
#define _DTSeededTracklet__

/************************************************************

   \class DTSeededTracklet
   
   Description: stacked tracker tracklet having a core of
   at least three stubs all matched by DTMatch objects.

   Sept. 2009            
   
   I. Lazzizzera - Trento University

*************************************************************/
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <algorithm>
#include <vector>

#include "SimDataFormats/SLHC/interface/DTMatch.h"
#include "SimDataFormats/SLHC/interface/DTTrackerStub.h"
#include "SimDataFormats/SLHC/interface/DTMatchPtAlgorithms.h"


using namespace std;


//const size_t STATIONS = 4;


class DTSeededStubTrack: public DTMatchPtVariety {
  
 public:
  
  // default trivial constructor
  DTSeededStubTrack() { 
    _Pt_encoder = NAN;    // Ignazio
    _Pt_encoder_bin = NAN;      // Ignazio
    _theDTMatchesVector = DTMatchesVector();
    _theDTMatchingStubSet = DTMatchingStubSet(); 
    _theCoreDTMatchingStubSet = DTMatchingStubSet(); 
  }
  
  // the constructor
  DTSeededStubTrack(DTMatch* dtm); 
  // copy constructor
  DTSeededStubTrack(const DTSeededStubTrack&);  
  // assignment operator
  DTSeededStubTrack& operator =(const DTSeededStubTrack&);
  
  void to_get_pt() {}  
  
  void sort() { 
    std::sort(_theDTMatchesVector.begin(), 
	      _theDTMatchesVector.end(), 
	      DTMatchSortPredicate);
  }
  
  const DTMatchesVector& theDTMatchesVector() const {
    return _theDTMatchesVector;
  }
  void set_theDTMatchesVector_size() {
    _theDTMatchesVector_size = _theDTMatchesVector.size();
  }
  size_t const theDTMatchesVector_size() const {
    return _theDTMatchesVector.size();
  }
  const DTMatchingStubSet theDTMatchingStubSet() const {
    return _theDTMatchingStubSet;
  }
  void set_theDTMatchingStubSet_size() {
    _theDTMatchingStubSet_size = _theDTMatchingStubSet.size();
  }
  size_t const theDTMatchingStubSet_size() const {
    return _theDTMatchingStubSet.size();
  }
  const DTMatchingStubSet theCoreDTMatchingStubSet() const {
    return _theCoreDTMatchingStubSet;
  }
  void set_theCoreDTMatchingStubSet_size() {
    _theCoreDTMatchingStubSet_size = _theCoreDTMatchingStubSet.size();
  }
  size_t const theCoreDTMatchingStubSet_size() const {
    return _theCoreDTMatchingStubSet.size();
  }

  inline float  Pt_encoder()  const { return _Pt_encoder; }             
  inline float  Pt_encoder_bin()    const { return _Pt_encoder_bin; }     
  inline bool   flagPt()    const { return _flagPt; }              

  // the destructor
  ~DTSeededStubTrack() {
    _theDTMatchesVector.clear();
    _theDTMatchesVector_size = 0;
    _theDTMatchingStubSet.clear();
    _theDTMatchingStubSet_size = 0;
    _theCoreDTMatchingStubSet.clear(); 
    _theCoreDTMatchingStubSet_size = 0;
    _DTSeededStubTracksCollectionSize = 0;
  }

  void update(DTMatch* dtm, 
	      DTMatchingStubSet* tracklet, DTMatchingStubSet* core_tracklet) {
    _theDTMatchesVector.push_back(new DTMatch(*dtm));
    _theDTMatchingStubSet = DTMatchingStubSet(*tracklet); 
    _theCoreDTMatchingStubSet = DTMatchingStubSet(*core_tracklet); 
  }
 
  static size_t getDTSeededStubTracksCollectionSize() {
    return _DTSeededStubTracksCollectionSize;
  }

  static void reset_DTSeededStubTracksCollectionSize() {
    _DTSeededStubTracksCollectionSize = 0;
  }  

  private:

  float _Pt_encoder;  
  bool  _flagPt;    
  float _Pt_encoder_bin;    

  /******************************************************************************
    This is the main data member: a collection of DTMatch objects having at 
    least three stubs in common; the component DTMatch objects are ordered
    according to DT trigger quality.
  *******************************************************************************/
  DTMatchesVector  _theDTMatchesVector; 
  size_t               _theDTMatchesVector_size;
  
  // set union of the DTMatchesVector stubs   
  DTMatchingStubSet  _theDTMatchingStubSet; 
  size_t             _theDTMatchingStubSet_size;
  // set intersection of the DTMatchesVector stubs   
  DTMatchingStubSet  _theCoreDTMatchingStubSet; 
  size_t             _theCoreDTMatchingStubSet_size;
  /******************************************************************************/
  
  static size_t _DTSeededStubTracksCollectionSize; 
  
  
 public:
  
  void setPt(const edm::ParameterSet& pSet);
};
 




//--------------------------------------------------------------------------------------
class DTSeededStubTracksCollection
{

  typedef std::vector<DTSeededStubTrack*>::iterator _iterator;

 public:
  // trivial constructor
  DTSeededStubTracksCollection() {
    _theCollection = std::vector<DTSeededStubTrack*>();
  }
  // default constructor
  DTSeededStubTracksCollection(std::vector<DTSeededStubTrack*>& tracklets) {
    _theCollection = std::vector<DTSeededStubTrack*>(tracklets);
  }

  void clear() { _theCollection.clear(); }   
  // destructor
  ~DTSeededStubTracksCollection() { _theCollection.clear(); } 
  
  void push_back(DTSeededStubTrack* aTracklet) {
    _theCollection.push_back(aTracklet);
  }

  void insert(_iterator pos, DTSeededStubTrack* aTracklet) {
    _theCollection.insert(pos, aTracklet);
  }

  void erase(_iterator pos) { _theCollection.erase(pos); }

  //return functions  
  const std::vector<DTSeededStubTrack*>& theCollection() const { return _theCollection; }

  DTSeededStubTrack* operator[](size_t n) {
    if( n >= _theCollection.size() ) 
      cerr << "index out of range" << endl;
    return _theCollection[n]; 
  }
  
  size_t size() const { return _theCollection.size(); }

 private:

  std::vector<DTSeededStubTrack*> _theCollection;

};


#endif


