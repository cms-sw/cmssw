#ifndef DTMatchPtVariety_H
#define DTMatchPtVariety_H

#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "SimDataFormats/SLHC/interface/DTMatchPt.h"  

#include <map>
#include <string.h>
   
struct ltstr
{
  bool operator()(const char* s1, const char* s2) const
  {
    return strcmp(s1, s2) < 0;
  }
};


class DTMatchPtVariety {
  /******************************************************************************
    This is the base class for DTMatchPtAlgorithms class, which is in turn
    the base for DTMatch class.
    Here the main data members are various DTMatchPt objects, which are the 
    muons Pt as computed according to different algorithms. 
    Each DTMatchPt object is characterized by a conventional id string:
    all of such id strings are stored in a vector of strings called "labels".
    A Pt is retrieved by the method "Pt(std::string const label)".
  ******************************************************************************/

 public:
  // constructor
  DTMatchPtVariety();
  // copy constructor
  DTMatchPtVariety(const DTMatchPtVariety& pts);
  // assignment operator
  DTMatchPtVariety& operator =(const DTMatchPtVariety& pts);
  // destructor
  virtual ~DTMatchPtVariety() {};

  void assignPt(const edm::ParameterSet&  pSet, const size_t s, const DTMatchPt* aPt);
  float const Pt(std::string const label) const;
  float const alpha0(std::string const label) const;
  float const d(std::string const label) const;

 protected:

  vector<string> labels;

  vector<DTMatchPt*> thePtv;

  DTMatchPt TrackletSL0;
  DTMatchPt TrackletSL1;
  DTMatchPt TrackletSL4; // our index is 2 !!!
  DTMatchPt Mu_SL4_0;
  DTMatchPt Mu_SL4_3;
  DTMatchPt Mu_SL1_0;
  DTMatchPt Mu_SL1_9;
  DTMatchPt Mu_SL0_3;
  DTMatchPt Mu_SL0_9;
  DTMatchPt Mu_SL4_V;
  DTMatchPt Mu_SL1_V;
  DTMatchPt Mu_SL0_V;
  DTMatchPt Mu_SL0_SL1;
  DTMatchPt Mu_SL0_SL4;
  DTMatchPt Mu_SL1_SL4;
  //
  DTMatchPt Stubs_9_3_0;// our index for layer 9 is 5 !!!
  DTMatchPt Stubs_9_1_0;
  DTMatchPt Stubs_3_2_0;
  DTMatchPt Stubs_3_1_0;
  DTMatchPt Stubs_9_3_V; 
  DTMatchPt Stubs_9_1_V;
  DTMatchPt Stubs_9_0_V;
  DTMatchPt Stubs_3_1_V;
  DTMatchPt Stubs_3_0_V;

  DTMatchPt Mu_9_8;
  DTMatchPt Mu_9_3;
  DTMatchPt Mu_9_2;
  DTMatchPt Mu_9_1;
  DTMatchPt Mu_9_0;
  DTMatchPt Mu_8_3; 
  DTMatchPt Mu_8_2; 
  DTMatchPt Mu_8_1; 
  DTMatchPt Mu_8_0; 
  DTMatchPt Mu_3_2;
  DTMatchPt Mu_3_1;
  DTMatchPt Mu_3_0;
  DTMatchPt Mu_2_1;
  DTMatchPt Mu_2_0;
  DTMatchPt Mu_1_0;
	
  DTMatchPt Mu_9_V;
  DTMatchPt Mu_3_V;
  DTMatchPt Mu_2_V;
  DTMatchPt Mu_1_V;
  DTMatchPt Mu_0_V;

  DTMatchPt IMu_9_0;
  DTMatchPt IMu_3_2;
  DTMatchPt IMu_3_1;
  DTMatchPt IMu_3_0;
  DTMatchPt IMu_2_1;
  DTMatchPt IMu_2_0; 
  DTMatchPt IMu_1_0; 
  DTMatchPt IMu_9_V;
  DTMatchPt IMu_3_V;
  DTMatchPt IMu_2_V;
  DTMatchPt IMu_1_V;
  DTMatchPt IMu_0_V;

  DTMatchPt mu_9_0;
  DTMatchPt mu_3_2;
  DTMatchPt mu_3_1;
  DTMatchPt mu_3_0; 
  DTMatchPt mu_2_1;
  DTMatchPt mu_2_0;
  DTMatchPt mu_1_0;
  DTMatchPt mu_9_V;
  DTMatchPt mu_3_V;
  DTMatchPt mu_2_V;
  DTMatchPt mu_1_V;
  DTMatchPt mu_0_V;

  DTMatchPt LinFitL2L0;
  DTMatchPt LinFitL2L1;
  DTMatchPt LinFitL3L0;
  DTMatchPt LinFitL3L1;
  DTMatchPt LinFitL8L0;
  DTMatchPt LinFitL8L1;
  DTMatchPt LinFitL8L2;
  DTMatchPt LinFitL8L3;
  DTMatchPt LinFitL9L0;
  DTMatchPt LinFitL9L1;
  DTMatchPt LinFitL9L2;
  DTMatchPt LinFitL9L3;

  DTMatchPt LinStubs_9_3_0;
  DTMatchPt LinStubs_9_1_0;
  DTMatchPt LinStubs_3_2_0;
  DTMatchPt LinStubs_3_1_0;

};

#endif
