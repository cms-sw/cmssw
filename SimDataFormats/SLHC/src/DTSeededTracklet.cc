#ifdef SLHC_DT_TRK_DFENABLE

#include <algorithm>
#include <vector>

#include "SimDataFormats/SLHC/interface/DTSeededTracklet.h"

using namespace std;


// static member init ************************************************

size_t  DTSeededTracklet::_DTSeededTrackletsCollectionSize = 0;

//********************************************************************


// the constructor
DTSeededTracklet::DTSeededTracklet(DTStubMatch* dtm) {
  _Pt_value = NAN;    // Ignazio
  _Pt_bin = NAN;      // Ignazio
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
  ++_DTSeededTrackletsCollectionSize;
}


// copy constructor
DTSeededTracklet::DTSeededTracklet(const DTSeededTracklet& t):
  DTStubMatchPtVariety(t) {
  _Pt_value = t._Pt_value;    // Ignazio
  _Pt_bin = t._Pt_bin;      // Ignazio
  _theDTTracklet = DTTracklet(t.theDTTracklet());
  _theDTTracklet_size = _theDTTracklet.size();
  _theStubTracklet = StubTracklet(t.theStubTracklet());
  _theStubTracklet_size = _theStubTracklet.size();
  _theCoreStubTracklet = StubTracklet(t.theCoreStubTracklet());
  _theCoreStubTracklet_size = _theCoreStubTracklet.size();
  _DTSeededTrackletsCollectionSize = t.getDTSeededTrackletsCollectionSize();
}


// assignment operator
DTSeededTracklet& DTSeededTracklet::operator =(const DTSeededTracklet& t) {
  if (this == &t)      // Same object?
    return *this;      // Yes, so skip assignment, and just return *this.
  this->DTStubMatchPtVariety::operator=(t);
  _Pt_value = t._Pt_value;    // Ignazio
  _Pt_bin = t._Pt_bin;      // Ignazio
  _theDTTracklet = DTTracklet(t.theDTTracklet());
  _theStubTracklet = StubTracklet(t.theStubTracklet());
  _theCoreStubTracklet = StubTracklet(t.theCoreStubTracklet());
  _theDTTracklet_size = _theDTTracklet.size();
  _theStubTracklet_size = _theStubTracklet.size();
  _theCoreStubTracklet_size = _theCoreStubTracklet.size();
  _DTSeededTrackletsCollectionSize = t.getDTSeededTrackletsCollectionSize();
  return *this;
}


// Pt ***********************************************************************
void DTSeededTracklet::setPt(const edm::ParameterSet& pSet) {
  //
  /*
  vector<string> labels =
    pSet.getUntrackedParameter<std::vector<std::string> >("labels");
  */
  float xr = NAN, yr = NAN;
  for(size_t s=0; s<labels.size(); s++) {
    DTStubMatchPt* aPt = new DTStubMatchPt();
    if(labels[s].find(string("mu")) != string::npos) {  // using proper _sqrtDscrm
      xr = (theDTTracklet()[0])->Rtilde() * cos((theDTTracklet()[0])->phiR());
      yr = (theDTTracklet()[0])->Rtilde() * sin((theDTTracklet()[0])->phiR());
      aPt = new DTStubMatchPt(labels[s],
			      (theDTTracklet()[0])->station(),
			      pSet, xr, yr,
			      //(theDTTracklet()[0])->xerre(),
			      //(theDTTracklet()[0])->yerre(),
			      (theDTTracklet()[0])->stub_x(),
			      (theDTTracklet()[0])->stub_y(),
			      (theDTTracklet()[0])->flagMatch());

    }
    else if( (labels[s].find(string("Mu")) != string::npos) &&
	     (labels[s].find(string("IMu")) == string::npos) ) { // _sqrtDscrm set to 1
      xr = (theDTTracklet()[0])->Rtilde() * cos((theDTTracklet()[0])->PhiR());
      yr = (theDTTracklet()[0])->Rtilde() * sin((theDTTracklet()[0])->PhiR());
      aPt = new DTStubMatchPt(labels[s],
			      (theDTTracklet()[0])->station(),
			      pSet, xr, yr,
			      //(theDTTracklet()[0])->Xerre(),
			      //(theDTTracklet()[0])->Yerre(),
			      (theDTTracklet()[0])->stub_x(),
			      (theDTTracklet()[0])->stub_y(),
			      (theDTTracklet()[0])->flagMatch());
 /*     cout << "() --> " << labels[s]
	   << ": Pt = " << aPt->Pt()
	   << endl ;*/
    }
    else if(labels[s].find(string("IMu")) !=  string::npos) {
      // deltaPhiR_over_bending  = 1. - _rho/_erre;
      xr = (theDTTracklet()[0])->Rtilde() * cos((theDTTracklet()[0])->PhiRI());
      yr = (theDTTracklet()[0])->Rtilde() * sin((theDTTracklet()[0])->PhiRI());
      aPt = new DTStubMatchPt(labels[s],
			      (theDTTracklet()[0])->station(),
			      pSet, xr, yr,
			      //(theDTTracklet()[0])->XerreI(),
			      //(theDTTracklet()[0])->YerreI(),
			      (theDTTracklet()[0])->stub_x(),
			      (theDTTracklet()[0])->stub_y(),
			      (theDTTracklet()[0])->flagMatch());
    }
    else if( (labels[s].find(string("Stubs")) != string::npos) && // all Stubs
	     (labels[s].find(string("LinStubs")) == string::npos) ) {
      aPt = new DTStubMatchPt(labels[s],
			      (theDTTracklet()[0])->station(),
			      pSet,
			      (theDTTracklet()[0])->stub_position(),
			      (theDTTracklet()[0])->flagMatch());
    }
    else if(labels[s].find(string("LinStubs")) != string::npos) {
      aPt = new DTStubMatchPt(labels[s],
			      (theDTTracklet()[0])->station(),
			      (theDTTracklet()[0])->stub_position(),
 			      (theDTTracklet()[0])->flagMatch(),
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
      const float dephi = fabs((theDTTracklet()[0])->stubstub_dephi(I, J));
      const float slope = (theDTTracklet()[0])->slope_linearfit(I, J);
      const float dephi_zero = (theDTTracklet()[0])->y_intercept_linearfit(I, J);
      aPt = new DTStubMatchPt(labels[s],
			      slope, dephi_zero, I, J, dephi,
			      pSet,
			      (theDTTracklet()[0])->flagMatch());
      /*
      cout << "(" << I << ", " << J << ") --> " << labels[s]
	   << ": Pt = " << aPt->Pt()
	   << endl << endl;
      */
    }
    assignPt(pSet, s, aPt); // to each labeled DTStubMatchPt object
  }
  _Pt_value = (theDTTracklet()[0])->Pt_value();
  _Pt_bin   = (theDTTracklet()[0])->Pt_bin();
  _flagPt   = (theDTTracklet()[0])->flagPt();
}
#endif
