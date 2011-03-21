#ifndef DTStubMatchPtVariety_H
#define DTStubMatchPtVariety_H

#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "SimDataFormats/SLHC/interface/DTStubMatchPt.h"     

class DTStubMatchPtVariety {
 public:
  // constructor
  DTStubMatchPtVariety();
  // copy constructor
  DTStubMatchPtVariety(const DTStubMatchPtVariety& pts);
  // assignment operator
  DTStubMatchPtVariety& operator =(const DTStubMatchPtVariety& pts);
  // destructor
  virtual ~DTStubMatchPtVariety() {};

  void assignPt(const edm::ParameterSet&  pSet, const size_t s, const DTStubMatchPt* aPt);
  float const Pt(std::string const label) const;
  float const alpha0(std::string const label) const;
  float const d(std::string const label) const;

 protected:

  vector<string> labels;

  DTStubMatchPt Stubs_9_3_0;
  //  DTStubMatchPt Stubs_5_3_0;
  DTStubMatchPt Stubs_9_1_0;
  //  DTStubMatchPt Stubs_5_1_0;
  DTStubMatchPt Stubs_3_2_0;
  DTStubMatchPt Stubs_3_1_0;
  DTStubMatchPt Stubs_9_3_V;
  //  DTStubMatchPt Stubs_5_3_V;
  DTStubMatchPt Stubs_9_1_V;
  //  DTStubMatchPt Stubs_5_1_V;
  DTStubMatchPt Stubs_9_0_V;
  //  DTStubMatchPt Stubs_5_0_V;
  DTStubMatchPt Stubs_3_1_V;
  DTStubMatchPt Stubs_3_0_V;
  DTStubMatchPt Mu_9_0;
  //  DTStubMatchPt Mu_5_0;
  DTStubMatchPt Mu_3_2;
  DTStubMatchPt Mu_3_1;
  DTStubMatchPt Mu_3_0;
  DTStubMatchPt Mu_2_1;
  DTStubMatchPt Mu_2_0;
  DTStubMatchPt Mu_1_0;
  DTStubMatchPt Mu_9_V;
  //  DTStubMatchPt Mu_5_V;
  DTStubMatchPt Mu_3_V;
  DTStubMatchPt Mu_2_V;
  DTStubMatchPt Mu_1_V;
  DTStubMatchPt Mu_0_V;
  DTStubMatchPt IMu_9_0;
  //  DTStubMatchPt IMu_5_0;
  DTStubMatchPt IMu_3_2;
  DTStubMatchPt IMu_3_1;
  DTStubMatchPt IMu_3_0;
  DTStubMatchPt IMu_2_1;
  DTStubMatchPt IMu_2_0; 
  DTStubMatchPt IMu_1_0; 
  DTStubMatchPt IMu_9_V;
  //  DTStubMatchPt IMu_5_V;
  DTStubMatchPt IMu_3_V;
  DTStubMatchPt IMu_2_V;
  DTStubMatchPt IMu_1_V;
  DTStubMatchPt IMu_0_V;
  DTStubMatchPt mu_9_0;
  //  DTStubMatchPt mu_5_0;
  DTStubMatchPt mu_3_2;
  DTStubMatchPt mu_3_1;
  DTStubMatchPt mu_3_0; 
  DTStubMatchPt mu_2_1;
  DTStubMatchPt mu_2_0;
  DTStubMatchPt mu_1_0;
  DTStubMatchPt mu_9_V;
  //  DTStubMatchPt mu_5_V;
  DTStubMatchPt mu_3_V;
  DTStubMatchPt mu_2_V;
  DTStubMatchPt mu_1_V;
  DTStubMatchPt mu_0_V;
  DTStubMatchPt LinFitL2L0;
  DTStubMatchPt LinFitL2L1;
  DTStubMatchPt LinFitL3L0;
  DTStubMatchPt LinFitL3L1;
  DTStubMatchPt LinFitL8L0;
  DTStubMatchPt LinFitL8L1;
  DTStubMatchPt LinFitL8L2;
  DTStubMatchPt LinFitL8L3;
  DTStubMatchPt LinFitL9L0;
  DTStubMatchPt LinFitL9L1;
  DTStubMatchPt LinFitL9L2;
  DTStubMatchPt LinFitL9L3;
  DTStubMatchPt LinStubs_9_3_0;
  DTStubMatchPt LinStubs_9_1_0;
  //  DTStubMatchPt LinStubs_5_3_0;
  //  DTStubMatchPt LinStubs_5_1_0;
  DTStubMatchPt LinStubs_3_2_0;
  DTStubMatchPt LinStubs_3_1_0;

};

#endif
