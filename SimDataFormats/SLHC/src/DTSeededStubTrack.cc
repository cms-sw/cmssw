#include <algorithm>
#include <vector>

#include "SimDataFormats/SLHC/interface/DTSeededStubTrack.h"

using namespace std;


// static member init ************************************************

size_t  DTSeededStubTrack::_DTSeededStubTracksCollectionSize = 0; 

//********************************************************************


// the constructor
DTSeededStubTrack::DTSeededStubTrack(DTMatch* dtm) { 
  _Pt_encoder = NAN;    // Ignazio
  _Pt_encoder_bin = NAN;      // Ignazio
  /**
     Start by a DTMatch object, then upgrade adding those 
     DTMatch objects sharing at least three matching stubs;
     different DTSeededStubTrack objects have disjoint sets of 
     matching stubs. 
  **/
  _theDTMatchesVector = DTMatchesVector();
  _theDTMatchesVector.push_back(new DTMatch(*dtm)); 
  _theDTMatchingStubSet = DTMatchingStubSet();
  _theCoreDTMatchingStubSet = DTMatchingStubSet();
  DTMatchingStubSet::const_iterator st = dtm->getMatchingStubs().begin();
  for(size_t i=0; i<dtm->getMatchingStubs().size(); i++) {
    _theDTMatchingStubSet.insert(new TrackerStub(**st));
    _theCoreDTMatchingStubSet.insert(new TrackerStub(**st));
    ++st;
  }
  _theDTMatchesVector_size = _theDTMatchesVector.size();
  _theDTMatchingStubSet_size = _theDTMatchingStubSet.size();
  _theCoreDTMatchingStubSet_size = _theCoreDTMatchingStubSet.size();
  ++_DTSeededStubTracksCollectionSize;
}


// copy constructor
DTSeededStubTrack::DTSeededStubTrack(const DTSeededStubTrack& t): 
  DTMatchPtVariety(t) {
  _Pt_encoder = t._Pt_encoder;    // Ignazio
  _Pt_encoder_bin = t._Pt_encoder_bin;      // Ignazio
  _theDTMatchesVector = DTMatchesVector(t.theDTMatchesVector());
  _theDTMatchesVector_size = _theDTMatchesVector.size();
  _theDTMatchingStubSet = DTMatchingStubSet(t.theDTMatchingStubSet());
  _theDTMatchingStubSet_size = _theDTMatchingStubSet.size();
  _theCoreDTMatchingStubSet = DTMatchingStubSet(t.theCoreDTMatchingStubSet());
  _theCoreDTMatchingStubSet_size = _theCoreDTMatchingStubSet.size();
  _DTSeededStubTracksCollectionSize = t.getDTSeededStubTracksCollectionSize();
}


// assignment operator
DTSeededStubTrack& DTSeededStubTrack::operator =(const DTSeededStubTrack& t) {
  if (this == &t)      // Same object?
    return *this;      // Yes, so skip assignment, and just return *this.
  this->DTMatchPtVariety::operator=(t);
  _Pt_encoder = t._Pt_encoder;    // Ignazio
  _Pt_encoder_bin = t._Pt_encoder_bin;      // Ignazio
  _theDTMatchesVector = DTMatchesVector(t.theDTMatchesVector());
  _theDTMatchingStubSet = DTMatchingStubSet(t.theDTMatchingStubSet());
  _theCoreDTMatchingStubSet = DTMatchingStubSet(t.theCoreDTMatchingStubSet());
  _theDTMatchesVector_size = _theDTMatchesVector.size();
  _theDTMatchingStubSet_size = _theDTMatchingStubSet.size();
  _theCoreDTMatchingStubSet_size = _theCoreDTMatchingStubSet.size();
  _DTSeededStubTracksCollectionSize = t.getDTSeededStubTracksCollectionSize();
  return *this;
}


// Pt ***********************************************************************
void DTSeededStubTrack::setPt(const edm::ParameterSet& pSet) {
  //
  /*
  vector<string> labels = 
    pSet.getUntrackedParameter<std::vector<std::string> >("labels");
  */
  float xr = NAN, yr = NAN;
  for(size_t s=0; s<labels.size(); s++) {
    DTMatchPt* aPt = new DTMatchPt();
    if(labels[s].find(string("mu")) != string::npos) {  // using proper _sqrtDscrm
      xr = (theDTMatchesVector()[0])->Rtilde() * 
	cos((theDTMatchesVector()[0])->phiR());
      yr = (theDTMatchesVector()[0])->Rtilde() * 
	sin((theDTMatchesVector()[0])->phiR());
      aPt = new DTMatchPt(labels[s], 
			      (theDTMatchesVector()[0])->station(),
			      pSet, xr, yr,
			      //(theDTMatchesVector()[0])->xerre(), 
			      //(theDTMatchesVector()[0])->yerre(), 
			      (theDTMatchesVector()[0])->stub_x(),
			      (theDTMatchesVector()[0])->stub_y(),
			      (theDTMatchesVector()[0])->flagMatch()); 
      //  cout << " mu " << xr << " " << yr << " " 
      //       << (theDTMatchesVector()[0])->phiR() <<  endl; 
			
    }
    else if( (labels[s].find(string("Mu")) != string::npos) &&
	     (labels[s].find(string("IMu")) == string::npos) ) { // _sqrtDscrm set to 1
      xr = (theDTMatchesVector()[0])->Rtilde() * 
	cos((theDTMatchesVector()[0])->PhiR());
      yr = (theDTMatchesVector()[0])->Rtilde() * 
	sin((theDTMatchesVector()[0])->PhiR());
      aPt = new DTMatchPt(labels[s], 
			      (theDTMatchesVector()[0])->station(),
			      pSet, xr, yr,
			      //(theDTMatchesVector()[0])->Xerre(), 
			      //(theDTMatchesVector()[0])->Yerre(),
			      (theDTMatchesVector()[0])->stub_x(),
			      (theDTMatchesVector()[0])->stub_y(),
			      (theDTMatchesVector()[0])->flagMatch());       
      /*     cout << "() --> " << labels[s] 
	     << ": Pt = " << aPt->Pt() 
	     << endl ;
      */
      //     cout << " Mu " << xr << " " << yr << " " 
      //          << (theDTMatchesVector()[0])->PhiR() <<  endl; 

    }
    else if(labels[s].find(string("IMu")) !=  string::npos) { 
      // deltaPhiR_over_bending  = 1. - _rho/_erre;
      xr = (theDTMatchesVector()[0])->Rtilde() * 
	cos((theDTMatchesVector()[0])->PhiRI());
      yr = (theDTMatchesVector()[0])->Rtilde() * 
	sin((theDTMatchesVector()[0])->PhiRI());
      aPt = new DTMatchPt(labels[s], 
			      (theDTMatchesVector()[0])->station(),
			      pSet, xr, yr,
			      //(theDTMatchesVector()[0])->XerreI(), 
			      //(theDTMatchesVector()[0])->YerreI(), 
			      (theDTMatchesVector()[0])->stub_x(),
			      (theDTMatchesVector()[0])->stub_y(),
			      (theDTMatchesVector()[0])->flagMatch());
    }
    else if( (labels[s].find(string("Stubs")) != string::npos) && // all Stubs
	     (labels[s].find(string("LinStubs")) == string::npos) ) { 
      aPt = new DTMatchPt(labels[s], 
			      (theDTMatchesVector()[0])->station(),
			      pSet, 
			      (theDTMatchesVector()[0])->stub_position(), 
			      (theDTMatchesVector()[0])->flagMatch());
    }
    else if(labels[s].find(string("LinStubs")) != string::npos) {
      aPt = new DTMatchPt(labels[s], 
			      (theDTMatchesVector()[0])->station(),
			      (theDTMatchesVector()[0])->stub_position(),
 			      (theDTMatchesVector()[0])->flagMatch(),
			      pSet );
    }
    else if(labels[s].find(string("LinFit")) != string::npos)  {
      // using linear fit of stub dephi vs invPt
      const int I = tracker_lay_Id_to_our( atoi( &((labels[s])[7]) ) );
      const int J = tracker_lay_Id_to_our( atoi( &((labels[s])[9]) ) );
      /*
      cout << "(" << I << ", " << J << ") --> flagMatches = "  << "(" 
	   << ((_flagMatch[I])? "true" : "false") << ", " 
	   << ((_flagMatch[J])? "true" : "false") <<  ")" << endl;
      */
      const float dephi = fabs((theDTMatchesVector()[0])->stubstub_dephi(I, J));
      const float slope = (theDTMatchesVector()[0])->slope_linearfit(I, J);
      const float dephi_zero = 
	(theDTMatchesVector()[0])->y_intercept_linearfit(I, J);
      aPt = new DTMatchPt(labels[s], 
			      slope, dephi_zero, I, J, dephi, 
			      pSet, 
			      (theDTMatchesVector()[0])->flagMatch());
      /*
      cout << "(" << I << ", " << J << ") --> " << labels[s] 
	   << ": Pt = " << aPt->Pt() 
	   << endl << endl;
      */
    }
    assignPt(pSet, s, aPt); // to each labeled DTMatchPt object
  }
  _Pt_encoder = (theDTMatchesVector()[0])->Pt_encoder();
  _Pt_encoder_bin   = (theDTMatchesVector()[0])->Pt_encoder_bin();
  _flagPt   = (theDTMatchesVector()[0])->flagPt();
}






