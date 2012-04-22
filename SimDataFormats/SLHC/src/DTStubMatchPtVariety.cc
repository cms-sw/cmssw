#ifdef SLHC_DT_TRK_DFENABLE
#include "SimDataFormats/SLHC/interface/DTStubMatchPtVariety.h"


using namespace std;


DTStubMatchPtVariety::DTStubMatchPtVariety() {
  Stubs_9_3_0 = DTStubMatchPt(std::string("Stubs_9_3_0"));
  labels.push_back(std::string("Stubs_9_3_0"));
  /*
  Stubs_5_3_0 = DTStubMatchPt(std::string("Stubs_5_3_0"));
  labels.push_back(std::string("Stubs_5_3_0"));
  */
  Stubs_9_1_0 = DTStubMatchPt(std::string("Stubs_9_1_0"));
  labels.push_back(std::string("Stubs_9_1_0"));
  /*
  Stubs_5_1_0 = DTStubMatchPt(std::string("Stubs_5_1_0"));
  labels.push_back(std::string("Stubs_5_1_0"));
  */
  Stubs_3_2_0 = DTStubMatchPt(std::string("Stubs_3_2_0"));
  labels.push_back(std::string("Stubs_3_2_0"));
  Stubs_3_1_0 = DTStubMatchPt(std::string("Stubs_3_1_0"));
  labels.push_back(std::string("Stubs_3_1_0"));
  Stubs_9_3_V = DTStubMatchPt(std::string("Stubs_9_3_V"));
  labels.push_back(std::string("Stubs_9_3_V"));
  /*
  Stubs_5_3_V = DTStubMatchPt(std::string("Stubs_5_3_V"));
  labels.push_back(std::string("Stubs_5_3_V"));
  */
  Stubs_9_1_V = DTStubMatchPt(std::string("Stubs_9_1_V"));
  labels.push_back(std::string("Stubs_9_1_V"));
  /*
  Stubs_5_1_V = DTStubMatchPt(std::string("Stubs_5_1_V"));
  labels.push_back(std::string("Stubs_5_1_V"));
  */
  Stubs_9_0_V = DTStubMatchPt(std::string("Stubs_9_0_V"));
  labels.push_back(std::string("Stubs_9_0_V"));
  /*
  Stubs_5_0_V = DTStubMatchPt(std::string("Stubs_5_0_V"));
  labels.push_back(std::string("Stubs_5_0_V"));
  */
  Stubs_3_1_V = DTStubMatchPt(std::string("Stubs_3_1_V"));
  labels.push_back(std::string("Stubs_3_1_V"));
  Stubs_3_0_V = DTStubMatchPt(std::string("Stubs_3_0_V"));
  labels.push_back(std::string("Stubs_3_0_V"));
  Mu_9_0   = DTStubMatchPt(std::string("Mu_9_0"));
  labels.push_back(std::string("Mu_9_0"));
  /*
  Mu_5_0   = DTStubMatchPt(std::string("Mu_5_0"));
  labels.push_back(std::string("Mu_5_0"));
  */
  Mu_3_2   = DTStubMatchPt(std::string("Mu_3_2"));
  labels.push_back(std::string("Mu_3_2"));
  Mu_3_1   = DTStubMatchPt(std::string("Mu_3_1"));
  labels.push_back(std::string("Mu_3_1"));
  Mu_3_0   = DTStubMatchPt(std::string("Mu_3_0"));
  labels.push_back(std::string("Mu_3_0"));
  Mu_2_1   = DTStubMatchPt(std::string("Mu_2_1"));
  labels.push_back(std::string("Mu_2_1"));
  Mu_2_0   = DTStubMatchPt(std::string("Mu_2_0"));
  labels.push_back(std::string("Mu_2_0"));
  Mu_1_0   = DTStubMatchPt(std::string("Mu_1_0"));
  labels.push_back(std::string("Mu_1_0"));
  Mu_9_V   = DTStubMatchPt(std::string("Mu_9_V"));
  labels.push_back(std::string("Mu_9_V"));
  /*
  Mu_5_V   = DTStubMatchPt(std::string("Mu_5_V"));
  labels.push_back(std::string("Mu_5_V"));
  */
  Mu_3_V   = DTStubMatchPt(std::string("Mu_3_V"));
  labels.push_back(std::string("Mu_3_V"));
  Mu_2_V   = DTStubMatchPt(std::string("Mu_2_V"));
  labels.push_back(std::string("Mu_2_V"));
  Mu_1_V   = DTStubMatchPt(std::string("Mu_1_V"));
  labels.push_back(std::string("Mu_1_V"));
  Mu_0_V   = DTStubMatchPt(std::string("Mu_0_V"));
  labels.push_back(std::string("Mu_0_V"));
  IMu_9_0   = DTStubMatchPt(std::string("IMu_9_0"));
  labels.push_back(std::string("IMu_9_0"));
  /*
  IMu_5_0   = DTStubMatchPt(std::string("IMu_5_0"));
  labels.push_back(std::string("IMu_5_0"));
  */
  IMu_3_2   = DTStubMatchPt(std::string("IMu_3_2"));
  labels.push_back(std::string("IMu_3_2"));
  IMu_3_1   = DTStubMatchPt(std::string("IMu_3_1"));
  labels.push_back(std::string("IMu_3_1"));
  IMu_3_0   = DTStubMatchPt(std::string("IMu_3_0"));
  labels.push_back(std::string("IMu_3_0"));
  IMu_2_1   = DTStubMatchPt(std::string("IMu_2_1"));
  labels.push_back(std::string("IMu_2_1"));
  IMu_2_0   = DTStubMatchPt(std::string("IMu_2_0"));
  labels.push_back(std::string("IMu_2_0"));
  IMu_1_0   = DTStubMatchPt(std::string("IMu_1_0"));
  labels.push_back(std::string("IMu_1_0"));
  IMu_9_V   = DTStubMatchPt(std::string("IMu_9_V"));
  labels.push_back(std::string("IMu_9_V"));
  /*
  IMu_5_V   = DTStubMatchPt(std::string("IMu_5_V"));
  labels.push_back(std::string("IMu_5_V"));
  */
  IMu_3_V   = DTStubMatchPt(std::string("IMu_3_V"));
  labels.push_back(std::string("IMu_3_V"));
  IMu_2_V   = DTStubMatchPt(std::string("IMu_2_V"));
  labels.push_back(std::string("IMu_2_V"));
  IMu_1_V   = DTStubMatchPt(std::string("IMu_1_V"));
  labels.push_back(std::string("IMu_1_V"));
  IMu_0_V   = DTStubMatchPt(std::string("IMu_0_V"));
  labels.push_back(std::string("IMu_0_V"));
  mu_9_0  = DTStubMatchPt(std::string("mu_9_0"));
  labels.push_back(std::string("mu_9_0"));
  /*
  mu_5_0  = DTStubMatchPt(std::string("mu_5_0"));
  labels.push_back(std::string("mu_5_0"));
  */
  mu_3_2  = DTStubMatchPt(std::string("mu_3_2"));
  labels.push_back(std::string("mu_3_2"));
  mu_3_1  = DTStubMatchPt(std::string("mu_3_1"));
  labels.push_back(std::string("mu_3_1"));
  mu_3_0  = DTStubMatchPt(std::string("mu_3_0"));
  labels.push_back(std::string("mu_3_0"));
  mu_2_1  = DTStubMatchPt(std::string("mu_2_1"));
  labels.push_back(std::string("mu_2_1"));
  mu_2_0  = DTStubMatchPt(std::string("mu_2_0"));
  labels.push_back(std::string("mu_2_0"));
  mu_1_0  = DTStubMatchPt(std::string("mu_1_0"));
  labels.push_back(std::string("mu_1_0"));
  mu_9_V  = DTStubMatchPt(std::string("mu_9_V"));
  labels.push_back(std::string("mu_9_V"));
  /*
  mu_5_V  = DTStubMatchPt(std::string("mu_5_V"));
  labels.push_back(std::string("mu_5_V"));
  */
  mu_3_V  = DTStubMatchPt(std::string("mu_3_V"));
  labels.push_back(std::string("mu_3_V"));
  mu_2_V  = DTStubMatchPt(std::string("mu_2_V"));
  labels.push_back(std::string("mu_2_V"));
  mu_1_V  = DTStubMatchPt(std::string("mu_1_V"));
  labels.push_back(std::string("mu_1_V"));
  mu_0_V  = DTStubMatchPt(std::string("mu_0_V"));
  labels.push_back(std::string("mu_0_V"));
  LinFitL2L0  = DTStubMatchPt(std::string("LinFitL2L0"));
  labels.push_back(std::string("LinFitL2L0"));
  LinFitL2L1  = DTStubMatchPt(std::string("LinFitL2L1"));
  labels.push_back(std::string("LinFitL2L1"));
  LinFitL3L0  = DTStubMatchPt(std::string("LinFitL3L0"));
  labels.push_back(std::string("LinFitL3L0"));
  LinFitL3L1  = DTStubMatchPt(std::string("LinFitL3L1"));
  labels.push_back(std::string("LinFitL3L1"));
  LinFitL8L0  = DTStubMatchPt(std::string("LinFitL8L0"));
  labels.push_back(std::string("LinFitL8L0"));
  LinFitL8L1  = DTStubMatchPt(std::string("LinFitL8L1"));
  labels.push_back(std::string("LinFitL8L1"));
  LinFitL8L2  = DTStubMatchPt(std::string("LinFitL8L2"));
  labels.push_back(std::string("LinFitL8L2"));
  LinFitL8L3  = DTStubMatchPt(std::string("LinFitL8L3"));
  labels.push_back(std::string("LinFitL8L3"));
  LinFitL9L0  = DTStubMatchPt(std::string("LinFitL9L0"));
  labels.push_back(std::string("LinFitL9L0"));
  LinFitL9L1  = DTStubMatchPt(std::string("LinFitL9L1"));
  labels.push_back(std::string("LinFitL9L1"));
  LinFitL9L2  = DTStubMatchPt(std::string("LinFitL9L2"));
  labels.push_back(std::string("LinFitL9L2"));
  LinFitL9L3  = DTStubMatchPt(std::string("LinFitL9L3"));
  labels.push_back(std::string("LinFitL9L3"));
  LinStubs_9_3_0  = DTStubMatchPt(std::string("LinStubs_9_3_0"));
  labels.push_back(std::string("LinStubs_9_3_0"));
  LinStubs_9_1_0  = DTStubMatchPt(std::string("LinStubs_9_1_0"));
  labels.push_back(std::string("LinStubs_9_1_0"));
  /*
  LinStubs_5_3_0  = DTStubMatchPt(std::string("LinStubs_5_3_0"));
  labels.push_back(std::string("LinStubs_5_3_0"));
  LinStubs_5_1_0  = DTStubMatchPt(std::string("LinStubs_5_1_0"));
  labels.push_back(std::string("LinStubs_5_1_0"));
  */
  LinStubs_3_2_0  = DTStubMatchPt(std::string("LinStubs_3_2_0"));
  labels.push_back(std::string("LinStubs_3_2_0"));
  LinStubs_3_1_0  = DTStubMatchPt(std::string("LinStubs_3_1_0"));
  labels.push_back(std::string("LinStubs_3_1_0"));
}




// copy constructor
DTStubMatchPtVariety::DTStubMatchPtVariety(const DTStubMatchPtVariety& pts) {
  labels = pts.labels;
  Stubs_9_3_0 = pts.Stubs_9_3_0;
  //  Stubs_5_3_0 = pts.Stubs_5_3_0;
  Stubs_9_1_0 = pts.Stubs_9_1_0;
  //  Stubs_5_1_0 = pts.Stubs_5_1_0;
  Stubs_3_2_0 = pts.Stubs_3_2_0;
  Stubs_3_1_0 = pts.Stubs_3_1_0;
  Stubs_9_3_V = pts.Stubs_9_3_V;
  //  Stubs_5_3_V = pts.Stubs_5_3_V;
  Stubs_9_1_V = pts.Stubs_9_1_V;
  //  Stubs_5_1_V = pts.Stubs_5_1_V;
  Stubs_9_0_V = pts.Stubs_9_0_V;
  //  Stubs_5_0_V = pts.Stubs_5_0_V;
  Stubs_3_1_V = pts.Stubs_3_1_V;
  Stubs_3_0_V = pts.Stubs_3_0_V;
  Mu_9_0   = pts.Mu_9_0;
  //  Mu_5_0   = pts.Mu_5_0;
  Mu_3_2   = pts.Mu_3_2;
  Mu_3_1   = pts.Mu_3_1;
  Mu_3_0   = pts.Mu_3_0;
  Mu_2_1   = pts.Mu_2_1;
  Mu_2_0   = pts.Mu_2_0;
  Mu_1_0   = pts.Mu_1_0;
  Mu_9_V   = pts.Mu_9_V;
  //  Mu_5_V   = pts.Mu_5_V;
  Mu_3_V   = pts.Mu_3_V;
  Mu_2_V   = pts.Mu_2_V;
  Mu_1_V   = pts.Mu_1_V;
  Mu_0_V   = pts.Mu_0_V;
  IMu_9_0   = pts.IMu_9_0;
  //  IMu_5_0   = pts.IMu_5_0;
  IMu_3_2   = pts.IMu_3_2;
  IMu_3_1   = pts.IMu_3_1;
  IMu_3_0   = pts.IMu_3_0;
  IMu_2_1   = pts.IMu_2_1;
  IMu_2_0   = pts.IMu_2_0;
  IMu_1_0   = pts.IMu_1_0;
  IMu_9_V   = pts.IMu_9_V;
  //  IMu_5_V   = pts.IMu_5_V;
  IMu_3_V   = pts.IMu_3_V;
  IMu_2_V   = pts.IMu_2_V;
  IMu_1_V   = pts.IMu_1_V;
  IMu_0_V   = pts.IMu_0_V;
  mu_9_0   = pts.mu_9_0;
  //  mu_5_0   = pts.mu_5_0;
  mu_3_2   = pts.mu_3_2;
  mu_3_1   = pts.mu_3_1;
  mu_3_0   = pts.mu_3_0;
  mu_2_1   = pts.mu_2_1;
  mu_2_0   = pts.mu_2_0;
  mu_1_0   = pts.mu_1_0;
  mu_9_V   = pts.mu_9_V;
  //  mu_5_V   = pts.mu_5_V;
  mu_3_V   = pts.mu_3_V;
  mu_2_V   = pts.mu_2_V;
  mu_1_V   = pts.mu_1_V;
  mu_0_V   = pts.mu_0_V;
  LinFitL2L0 = pts.LinFitL2L0;
  LinFitL2L1 = pts.LinFitL2L1;
  LinFitL3L0 = pts.LinFitL3L0;
  LinFitL3L1 = pts.LinFitL3L1;
  LinFitL8L0 = pts.LinFitL8L0;
  LinFitL8L1 = pts.LinFitL8L1;
  LinFitL8L2 = pts.LinFitL8L2;
  LinFitL8L3 = pts.LinFitL8L3;
  LinFitL9L0 = pts.LinFitL9L0;
  LinFitL9L1 = pts.LinFitL9L1;
  LinFitL9L2 = pts.LinFitL9L2;
  LinFitL9L3 = pts.LinFitL9L3;
  LinStubs_9_3_0 = pts.LinStubs_9_3_0;
  LinStubs_9_1_0 = pts.LinStubs_9_1_0;
  //  LinStubs_5_3_0 = pts.LinStubs_5_3_0;
  //  LinStubs_5_1_0 = pts.LinStubs_5_1_0;
  LinStubs_3_2_0 = pts.LinStubs_3_2_0;
  LinStubs_3_1_0 = pts.LinStubs_3_1_0;
}




// assignment constructor
  DTStubMatchPtVariety&
  DTStubMatchPtVariety::operator =(const DTStubMatchPtVariety& pts) {
  if (this == &pts)      // Same object?
    return *this;        // Yes, so skip assignment, and just return *this.
  labels = pts.labels;
  Stubs_9_3_0 = pts.Stubs_9_3_0;
  //  Stubs_5_3_0 = pts.Stubs_5_3_0;
  Stubs_9_1_0 = pts.Stubs_9_1_0;
  //  Stubs_5_1_0 = pts.Stubs_5_1_0;
  Stubs_3_2_0 = pts.Stubs_3_2_0;
  Stubs_3_1_0 = pts.Stubs_3_1_0;
  Stubs_9_3_V = pts.Stubs_9_3_V;
  //  Stubs_5_3_V = pts.Stubs_5_3_V;
  Stubs_9_1_V = pts.Stubs_9_1_V;
  //  Stubs_5_1_V = pts.Stubs_5_1_V;
  Stubs_9_0_V = pts.Stubs_9_0_V;
  //  Stubs_5_0_V = pts.Stubs_5_0_V;
  Stubs_3_1_V = pts.Stubs_3_1_V;
  Stubs_3_0_V = pts.Stubs_3_0_V;
  Mu_9_0   = pts.Mu_9_0;
  //  Mu_5_0   = pts.Mu_5_0;
  Mu_3_2   = pts.Mu_3_2;
  Mu_3_1   = pts.Mu_3_1;
  Mu_3_0   = pts.Mu_3_0;
  Mu_2_1   = pts.Mu_2_1;
  Mu_2_0   = pts.Mu_2_0;
  Mu_1_0   = pts.Mu_1_0;
  Mu_9_V   = pts.Mu_9_V;
  //  Mu_5_V   = pts.Mu_5_V;
  Mu_3_V   = pts.Mu_3_V;
  Mu_2_V   = pts.Mu_2_V;
  Mu_1_V   = pts.Mu_1_V;
  Mu_0_V   = pts.Mu_0_V;
  IMu_9_0   = pts.IMu_9_0;
  //  IMu_5_0   = pts.IMu_5_0;
  IMu_3_2   = pts.IMu_3_2;
  IMu_3_1   = pts.IMu_3_1;
  IMu_3_0   = pts.IMu_3_0;
  IMu_2_1   = pts.IMu_2_1;
  IMu_2_0   = pts.IMu_2_0;
  IMu_1_0   = pts.IMu_1_0;
  IMu_9_V   = pts.IMu_9_V;
  //  IMu_5_V   = pts.IMu_5_V;
  IMu_3_V   = pts.IMu_3_V;
  IMu_2_V   = pts.IMu_2_V;
  IMu_1_V   = pts.IMu_1_V;
  IMu_0_V   = pts.IMu_0_V;
  mu_9_0   = pts.mu_9_0;
  //  mu_5_0   = pts.mu_5_0;
  mu_3_2   = pts.mu_3_2;
  mu_3_1   = pts.mu_3_1;
  mu_3_0   = pts.mu_3_0;
  mu_2_1   = pts.mu_2_1;
  mu_2_0   = pts.mu_2_0;
  mu_1_0   = pts.mu_1_0;
  mu_9_V   = pts.mu_9_V;
  //  mu_5_V   = pts.mu_5_V;
  mu_3_V   = pts.mu_3_V;
  mu_2_V   = pts.mu_2_V;
  mu_1_V   = pts.mu_1_V;
  mu_0_V   = pts.mu_0_V;
  LinFitL2L0 = pts.LinFitL2L0;
  LinFitL2L1 = pts.LinFitL2L1;
  LinFitL3L0 = pts.LinFitL3L0;
  LinFitL3L1 = pts.LinFitL3L1;
  LinFitL8L0 = pts.LinFitL8L0;
  LinFitL8L1 = pts.LinFitL8L1;
  LinFitL8L2 = pts.LinFitL8L2;
  LinFitL8L3 = pts.LinFitL8L3;
  LinFitL9L0 = pts.LinFitL9L0;
  LinFitL9L1 = pts.LinFitL9L1;
  LinFitL9L2 = pts.LinFitL9L2;
  LinFitL9L3 = pts.LinFitL9L3;
  LinStubs_9_3_0 = pts.LinStubs_9_3_0;
  LinStubs_9_1_0 = pts.LinStubs_9_1_0;
  //  LinStubs_5_3_0 = pts.LinStubs_5_3_0;
  //  LinStubs_5_1_0 = pts.LinStubs_5_1_0;
  LinStubs_3_2_0 = pts.LinStubs_3_2_0;
  LinStubs_3_1_0 = pts.LinStubs_3_1_0;
  return *this;
  }




void DTStubMatchPtVariety::assignPt(const edm::ParameterSet&  pSet,
				    const size_t s, const DTStubMatchPt* aPt)
{
  /*
  vector<string> labels =
    pSet.getUntrackedParameter<std::vector<std::string> >("labels");
  */
  if(labels[s] == std::string("Stubs_9_3_0"))
    Stubs_9_3_0 = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("Stubs_5_3_0"))
    Stubs_5_3_0 = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("Stubs_9_1_0"))
    Stubs_9_1_0 = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("Stubs_5_1_0"))
    Stubs_5_1_0 = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("Stubs_3_2_0"))
    Stubs_3_2_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Stubs_3_1_0"))
    Stubs_3_1_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Stubs_9_3_V"))
    Stubs_9_3_V = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("Stubs_5_3_V"))
    Stubs_5_3_V = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("Stubs_9_1_V"))
    Stubs_9_1_V = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("Stubs_5_1_V"))
    Stubs_5_1_V = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("Stubs_9_0_V"))
    Stubs_9_0_V = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("Stubs_5_0_V"))
    Stubs_5_0_V = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("Stubs_3_1_V"))
    Stubs_3_1_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Stubs_3_0_V"))
    Stubs_3_0_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Mu_9_0"))
    Mu_9_0 = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("Mu_5_0"))
    Mu_5_0 = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("Mu_3_2"))
    Mu_3_2 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Mu_3_1"))
    Mu_3_1 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Mu_3_0"))
    Mu_3_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Mu_2_1"))
    Mu_2_1 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Mu_2_0"))
    Mu_2_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Mu_1_0"))
    Mu_1_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Mu_9_V"))
    Mu_9_V = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("Mu_5_V"))
    Mu_5_V = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("Mu_3_V"))
    Mu_3_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Mu_2_V"))
    Mu_2_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Mu_1_V"))
    Mu_1_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("Mu_0_V"))
    Mu_0_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("IMu_9_0"))
    IMu_9_0 = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("IMu_5_0"))
  IMu_5_0 = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("IMu_3_2"))
    IMu_3_2 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("IMu_3_1"))
    IMu_3_1 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("IMu_3_0"))
    IMu_3_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("IMu_2_1"))
    IMu_2_1 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("IMu_2_0"))
    IMu_2_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("IMu_1_0"))
    IMu_1_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("IMu_9_V"))
    IMu_9_V = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("IMu_5_V"))
    IMu_5_V = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("IMu_3_V"))
    IMu_3_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("IMu_2_V"))
    IMu_2_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("IMu_1_V"))
    IMu_1_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("IMu_0_V"))
    IMu_0_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("mu_9_0"))
    mu_9_0 = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("mu_5_0"))
    mu_5_0 = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("mu_3_2"))
    mu_3_2 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("mu_3_1"))
    mu_3_1 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("mu_3_0"))
    mu_3_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("mu_2_1"))
    mu_2_1 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("mu_2_0"))
    mu_2_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("mu_1_0"))
    mu_1_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("mu_9_V"))
    mu_9_V = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("mu_5_V"))
    mu_5_V = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("mu_3_V"))
    mu_3_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("mu_2_V"))
    mu_2_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("mu_1_V"))
    mu_1_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("mu_0_V"))
    mu_0_V = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL2L0"))
    LinFitL2L0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL2L1"))
    LinFitL2L1 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL3L0"))
    LinFitL3L0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL3L1"))
    LinFitL3L1 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL8L0"))
    LinFitL8L0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL8L1"))
    LinFitL8L1 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL8L2"))
    LinFitL8L2 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL8L3"))
    LinFitL8L3 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL9L0"))
    LinFitL9L0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL9L1"))
    LinFitL9L1 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL9L2"))
    LinFitL9L2 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinFitL9L3"))
    LinFitL9L3 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinStubs_9_3_0"))
    LinStubs_9_3_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinStubs_9_1_0"))
    LinStubs_9_1_0 = DTStubMatchPt(*aPt);
  /*
  else if(labels[s] == std::string("LinStubs_5_3_0"))
    LinStubs_5_3_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinStubs_5_1_0"))
    LinStubs_5_1_0 = DTStubMatchPt(*aPt);
  */
  else if(labels[s] == std::string("LinStubs_3_2_0"))
    LinStubs_3_2_0 = DTStubMatchPt(*aPt);
  else if(labels[s] == std::string("LinStubs_3_1_0"))
    LinStubs_3_1_0 = DTStubMatchPt(*aPt);
}





float const DTStubMatchPtVariety::Pt(std::string const label) const
{
  if(label == std::string("Stubs_9_3_0"))
    return Stubs_9_3_0.Pt();
  /*
  else if(label == std::string("Stubs_5_3_0"))
    return Stubs_5_3_0.Pt();
  */
  else if(label == std::string("Stubs_9_1_0"))
    return Stubs_9_1_0.Pt();
  /*
  else if(label == std::string("Stubs_5_1_0"))
    return Stubs_5_1_0.Pt();
  */
  else if(label == std::string("Stubs_3_2_0"))
    return Stubs_3_2_0.Pt();
  else if(label == std::string("Stubs_3_1_0"))
    return Stubs_3_1_0.Pt();
  else if(label == std::string("Stubs_9_3_V"))
    return Stubs_9_3_V.Pt();
  /*
  else if(label == std::string("Stubs_5_3_V"))
    return Stubs_5_3_V.Pt();
  */
  else if(label == std::string("Stubs_9_1_V"))
    return Stubs_9_1_V.Pt();
  /*
  else if(label == std::string("Stubs_5_1_V"))
    return Stubs_5_1_V.Pt();
  */
  else if(label == std::string("Stubs_9_0_V"))
    return Stubs_9_0_V.Pt();
  /*
  else if(label == std::string("Stubs_5_0_V"))
    return Stubs_5_0_V.Pt();
  */
  else if(label == std::string("Stubs_3_1_V"))
    return Stubs_3_1_V.Pt();
  else if(label == std::string("Stubs_3_0_V"))
    return Stubs_3_0_V.Pt();
  else if(label == std::string("Mu_9_0"))
    return Mu_9_0.Pt();
  /*
  else if(label == std::string("Mu_5_0"))
    return Mu_5_0.Pt();
  */
  else if(label == std::string("Mu_3_2"))
    return Mu_3_2.Pt();
  else if(label == std::string("Mu_3_1"))
    return Mu_3_1.Pt();
  else if(label == std::string("Mu_3_0"))
    return Mu_3_0.Pt();
  else if(label == std::string("Mu_2_1"))
    return Mu_2_1.Pt();
  else if(label == std::string("Mu_2_0"))
    return Mu_2_0.Pt();
  else if(label == std::string("Mu_1_0"))
    return Mu_1_0.Pt();
  else if(label == std::string("Mu_9_V"))
    return Mu_9_V.Pt();
  /*
  else if(label == std::string("Mu_5_V"))
    return Mu_5_V.Pt();
  */
  else if(label == std::string("Mu_3_V"))
    return Mu_3_V.Pt();
  else if(label == std::string("Mu_2_V"))
    return Mu_2_V.Pt();
  else if(label == std::string("Mu_1_V"))
    return Mu_1_V.Pt();
  else if(label == std::string("Mu_0_V"))
    return Mu_0_V.Pt();
  else if(label == std::string("IMu_9_0"))
    return IMu_9_0.Pt();
  /*
  else if(label == std::string("IMu_5_0"))
    return IMu_5_0.Pt();
  */
  else if(label == std::string("IMu_3_2"))
    return IMu_3_2.Pt();
  else if(label == std::string("IMu_3_1"))
    return IMu_3_1.Pt();
  else if(label == std::string("IMu_3_0"))
    return IMu_3_0.Pt();
  else if(label == std::string("IMu_2_1"))
    return IMu_2_1.Pt();
  else if(label == std::string("IMu_2_0"))
    return IMu_2_0.Pt();
  else if(label == std::string("IMu_1_0"))
    return IMu_1_0.Pt();
  else if(label == std::string("IMu_9_V"))
    return IMu_9_V.Pt();
  /*
  else if(label == std::string("IMu_5_V"))
    return IMu_5_V.Pt();
  */
  else if(label == std::string("IMu_3_V"))
    return IMu_3_V.Pt();
  else if(label == std::string("IMu_2_V"))
    return IMu_2_V.Pt();
  else if(label == std::string("IMu_1_V"))
    return IMu_1_V.Pt();
  else if(label == std::string("IMu_0_V"))
    return IMu_0_V.Pt();
  else if(label == std::string("mu_9_0"))
    return mu_9_0.Pt();
  /*
  else if(label == std::string("mu_5_0"))
    return mu_5_0.Pt();
  */
  else if(label == std::string("mu_3_2"))
    return mu_3_2.Pt();
  else if(label == std::string("mu_3_1"))
    return mu_3_1.Pt();
  else if(label == std::string("mu_3_0"))
    return mu_3_0.Pt();
  else if(label == std::string("mu_2_1"))
    return mu_2_1.Pt();
  else if(label == std::string("mu_2_0"))
    return mu_2_0.Pt();
  else if(label == std::string("mu_1_0"))
    return mu_1_0.Pt();
  else if(label == std::string("mu_9_V"))
    return mu_9_V.Pt();
  /*
  else if(label == std::string("mu_5_V"))
    return mu_5_V.Pt();
  */
  else if(label == std::string("mu_3_V"))
    return mu_3_V.Pt();
  else if(label == std::string("mu_2_V"))
    return mu_2_V.Pt();
  else if(label == std::string("mu_1_V"))
    return mu_1_V.Pt();
  else if(label == std::string("mu_0_V"))
    return mu_0_V.Pt();
  else if(label == std::string("LinFitL2L0"))
    return LinFitL2L0.Pt();
  else if(label == std::string("LinFitL2L1"))
    return LinFitL2L1.Pt();
  else if(label == std::string("LinFitL3L0"))
    return LinFitL3L0.Pt();
  else if(label == std::string("LinFitL3L1"))
    return LinFitL3L1.Pt();
  else if(label == std::string("LinFitL8L0"))
    return LinFitL8L0.Pt();
  else if(label == std::string("LinFitL8L1"))
    return LinFitL8L1.Pt();
  else if(label == std::string("LinFitL8L2"))
    return LinFitL8L2.Pt();
  else if(label == std::string("LinFitL8L3"))
    return LinFitL8L3.Pt();
  else if(label == std::string("LinFitL9L0"))
    return LinFitL9L0.Pt();
  else if(label == std::string("LinFitL9L1"))
    return LinFitL9L1.Pt();
  else if(label == std::string("LinFitL9L2"))
    return LinFitL9L2.Pt();
  else if(label == std::string("LinFitL9L3"))
    return LinFitL9L3.Pt();
  else if(label == std::string("LinStubs_9_3_0"))
    return LinStubs_9_3_0.Pt();
  else if(label == std::string("LinStubs_9_1_0"))
    return LinStubs_9_1_0.Pt();
  /*
  else if(label == std::string("LinStubs_5_3_0"))
    return LinStubs_5_3_0.Pt();
  else if(label == std::string("LinStubs_5_1_0"))
    return LinStubs_5_1_0.Pt();
  */
  else if(label == std::string("LinStubs_3_2_0"))
    return LinStubs_3_2_0.Pt();
  else if(label == std::string("LinStubs_3_1_0"))
    return LinStubs_3_1_0.Pt();
  return NAN;
}





float const DTStubMatchPtVariety::alpha0(std::string const label) const
{
  if(label == std::string("Stubs_9_3_0"))
    return Stubs_9_3_0.alpha0();
  /*
  else if(label == std::string("Stubs_5_3_0"))
    return Stubs_5_3_0.alpha0();
  */
  else if(label == std::string("Stubs_9_1_0"))
    return Stubs_9_1_0.alpha0();
  /*
  else if(label == std::string("Stubs_5_1_0"))
    return Stubs_5_1_0.alpha0();
  */
  else if(label == std::string("Stubs_3_2_0"))
    return Stubs_3_2_0.alpha0();
  else if(label == std::string("Stubs_3_1_0"))
    return Stubs_3_1_0.alpha0();
  else if(label == std::string("Stubs_9_3_V"))
    return Stubs_9_3_V.alpha0();
  /*
  else if(label == std::string("Stubs_5_3_V"))
    return Stubs_5_3_V.alpha0();
  */
  else if(label == std::string("Stubs_9_1_V"))
    return Stubs_9_1_V.alpha0();
  /*
  else if(label == std::string("Stubs_5_1_V"))
    return Stubs_5_1_V.alpha0();
  */
  else if(label == std::string("Stubs_9_0_V"))
    return Stubs_9_0_V.alpha0();
  /*
  else if(label == std::string("Stubs_5_0_V"))
    return Stubs_5_0_V.alpha0();
  */
  else if(label == std::string("Stubs_3_1_V"))
    return Stubs_3_1_V.alpha0();
  else if(label == std::string("Stubs_3_0_V"))
    return Stubs_3_0_V.alpha0();
  else if(label == std::string("Mu_9_0"))
    return Mu_9_0.alpha0();
  /*
  else if(label == std::string("Mu_5_0"))
    return Mu_5_0.alpha0();
  */
  else if(label == std::string("Mu_3_2"))
    return Mu_3_2.alpha0();
  else if(label == std::string("Mu_3_1"))
    return Mu_3_1.alpha0();
  else if(label == std::string("Mu_3_0"))
    return Mu_3_0.alpha0();
  else if(label == std::string("Mu_2_1"))
    return Mu_2_1.alpha0();
  else if(label == std::string("Mu_2_0"))
    return Mu_2_0.alpha0();
  else if(label == std::string("Mu_1_0"))
    return Mu_1_0.alpha0();
  else if(label == std::string("Mu_9_V"))
    return Mu_9_V.alpha0();
  /*
  else if(label == std::string("Mu_5_V"))
    return Mu_5_V.alpha0();
  */
  else if(label == std::string("Mu_3_V"))
    return Mu_3_V.alpha0();
  else if(label == std::string("Mu_2_V"))
    return Mu_2_V.alpha0();
  else if(label == std::string("Mu_1_V"))
    return Mu_1_V.alpha0();
  else if(label == std::string("Mu_0_V"))
    return Mu_0_V.alpha0();
  else if(label == std::string("IMu_9_0"))
    return IMu_9_0.alpha0();
  /*
  else if(label == std::string("IMu_5_0"))
    return IMu_5_0.alpha0();
  */
  else if(label == std::string("IMu_3_2"))
    return IMu_3_2.alpha0();
  else if(label == std::string("IMu_3_1"))
    return IMu_3_1.alpha0();
  else if(label == std::string("IMu_3_0"))
    return IMu_3_0.alpha0();
  else if(label == std::string("IMu_2_1"))
    return IMu_2_1.alpha0();
  else if(label == std::string("IMu_2_0"))
    return IMu_2_0.alpha0();
  else if(label == std::string("IMu_1_0"))
    return IMu_1_0.alpha0();
  else if(label == std::string("IMu_9_V"))
    return IMu_9_V.alpha0();
  /*
  else if(label == std::string("IMu_5_V"))
    return IMu_5_V.alpha0();
  */
  else if(label == std::string("IMu_3_V"))
    return IMu_3_V.alpha0();
  else if(label == std::string("IMu_2_V"))
    return IMu_2_V.alpha0();
  else if(label == std::string("IMu_1_V"))
    return IMu_1_V.alpha0();
  else if(label == std::string("IMu_0_V"))
    return IMu_0_V.alpha0();
  else if(label == std::string("mu_9_0"))
    return mu_9_0.alpha0();
  /*
  else if(label == std::string("mu_5_0"))
    return mu_5_0.alpha0();
  */
  else if(label == std::string("mu_3_2"))
    return mu_3_2.alpha0();
  else if(label == std::string("mu_3_1"))
    return mu_3_1.alpha0();
  else if(label == std::string("mu_3_0"))
    return mu_3_0.alpha0();
  else if(label == std::string("mu_2_1"))
    return mu_2_1.alpha0();
  else if(label == std::string("mu_2_0"))
    return mu_2_0.alpha0();
  else if(label == std::string("mu_1_0"))
    return mu_1_0.alpha0();
  else if(label == std::string("mu_9_V"))
    return mu_9_V.alpha0();
  /*
  else if(label == std::string("mu_5_V"))
    return mu_5_V.alpha0();
  */
  else if(label == std::string("mu_3_V"))
    return mu_3_V.alpha0();
  else if(label == std::string("mu_2_V"))
    return mu_2_V.alpha0();
  else if(label == std::string("mu_1_V"))
    return mu_1_V.alpha0();
  else if(label == std::string("mu_0_V"))
    return mu_0_V.alpha0();
  else if(label == std::string("LinFitL2L0"))
    return LinFitL2L0.alpha0();
  else if(label == std::string("LinFitL2L1"))
    return LinFitL2L1.alpha0();
  else if(label == std::string("LinFitL3L0"))
    return LinFitL3L0.alpha0();
  else if(label == std::string("LinFitL3L1"))
    return LinFitL3L1.alpha0();
  else if(label == std::string("LinFitL8L0"))
    return LinFitL8L0.alpha0();
  else if(label == std::string("LinFitL8L1"))
    return LinFitL8L1.alpha0();
  else if(label == std::string("LinFitL8L2"))
    return LinFitL8L2.alpha0();
  else if(label == std::string("LinFitL8L3"))
    return LinFitL8L3.alpha0();
  else if(label == std::string("LinFitL9L0"))
    return LinFitL9L0.alpha0();
  else if(label == std::string("LinFitL9L1"))
    return LinFitL9L1.alpha0();
  else if(label == std::string("LinFitL9L2"))
    return LinFitL9L2.alpha0();
  else if(label == std::string("LinFitL9L3"))
    return LinFitL9L3.alpha0();
  else if(label == std::string("LinStubs_9_3_0"))
    return LinStubs_9_3_0.alpha0();
  else if(label == std::string("LinStubs_9_1_0"))
    return LinStubs_9_1_0.alpha0();
  /*
  else if(label == std::string("LinStubs_5_3_0"))
    return LinStubs_5_3_0.alpha0();
  else if(label == std::string("LinStubs_5_1_0"))
    return LinStubs_5_1_0.alpha0();
  */
  else if(label == std::string("LinStubs_3_2_0"))
    return LinStubs_3_2_0.alpha0();
  else if(label == std::string("LinStubs_3_1_0"))
    return LinStubs_3_1_0.alpha0();
  return NAN;
}






float const DTStubMatchPtVariety::d(std::string const label) const
{
  if(label == std::string("Stubs_9_3_0"))
    return Stubs_9_3_0.d();
  /*
  else if(label == std::string("Stubs_5_3_0"))
    return Stubs_5_3_0.d();
  */
  else if(label == std::string("Stubs_9_1_0"))
    return Stubs_9_1_0.d();
  /*
  else if(label == std::string("Stubs_5_1_0"))
    return Stubs_5_1_0.d();
  */
  else if(label == std::string("Stubs_3_2_0"))
    return Stubs_3_2_0.d();
  else if(label == std::string("Stubs_3_1_0"))
    return Stubs_3_1_0.d();
  else if(label == std::string("Stubs_9_3_V"))
    return Stubs_9_3_V.d();
  /*
  else if(label == std::string("Stubs_5_3_V"))
    return Stubs_5_3_V.d();
  */
  else if(label == std::string("Stubs_9_1_V"))
    return Stubs_9_1_V.d();
  /*
  else if(label == std::string("Stubs_5_1_V"))
    return Stubs_5_1_V.d();
  */
  else if(label == std::string("Stubs_9_0_V"))
    return Stubs_9_0_V.d();
  /*
  else if(label == std::string("Stubs_5_0_V"))
    return Stubs_5_0_V.d();
  */
  else if(label == std::string("Stubs_3_1_V"))
    return Stubs_3_1_V.d();
  else if(label == std::string("Stubs_3_0_V"))
    return Stubs_3_0_V.d();
  else if(label == std::string("Mu_9_0"))
    return Mu_9_0.d();
  /*
  else if(label == std::string("Mu_5_0"))
    return Mu_5_0.d();
  */
  else if(label == std::string("Mu_3_2"))
    return Mu_3_2.d();
  else if(label == std::string("Mu_3_1"))
    return Mu_3_1.d();
  else if(label == std::string("Mu_3_0"))
    return Mu_3_0.d();
  else if(label == std::string("Mu_2_1"))
    return Mu_2_1.d();
  else if(label == std::string("Mu_2_0"))
    return Mu_2_0.d();
  else if(label == std::string("Mu_1_0"))
    return Mu_1_0.d();
  else if(label == std::string("Mu_9_V"))
    return Mu_9_V.d();
  /*
  else if(label == std::string("Mu_5_V"))
    return Mu_5_V.d();
  */
  else if(label == std::string("Mu_3_V"))
    return Mu_3_V.d();
  else if(label == std::string("Mu_2_V"))
    return Mu_2_V.d();
  else if(label == std::string("Mu_1_V"))
    return Mu_1_V.d();
  else if(label == std::string("Mu_0_V"))
    return Mu_0_V.d();
  else if(label == std::string("IMu_9_0"))
    return IMu_9_0.d();
  /*
  else if(label == std::string("IMu_5_0"))
    return IMu_5_0.d();
  */
  else if(label == std::string("IMu_3_2"))
    return IMu_3_2.d();
  else if(label == std::string("IMu_3_1"))
    return IMu_3_1.d();
  else if(label == std::string("IMu_3_0"))
    return IMu_3_0.d();
  else if(label == std::string("IMu_2_1"))
    return IMu_2_1.d();
  else if(label == std::string("IMu_2_0"))
    return IMu_2_0.d();
  else if(label == std::string("IMu_1_0"))
    return IMu_1_0.d();
  else if(label == std::string("IMu_9_V"))
    return IMu_9_V.d();
  /*
  else if(label == std::string("IMu_5_V"))
    return IMu_5_V.d();
  */
  else if(label == std::string("IMu_3_V"))
    return IMu_3_V.d();
  else if(label == std::string("IMu_2_V"))
    return IMu_2_V.d();
  else if(label == std::string("IMu_1_V"))
    return IMu_1_V.d();
  else if(label == std::string("IMu_0_V"))
    return IMu_0_V.d();
  else if(label == std::string("mu_9_0"))
    return mu_9_0.d();
  /*
  else if(label == std::string("mu_5_0"))
    return mu_5_0.d();
  */
  else if(label == std::string("mu_3_2"))
    return mu_3_2.d();
  else if(label == std::string("mu_3_1"))
    return mu_3_1.d();
  else if(label == std::string("mu_3_0"))
    return mu_3_0.d();
  else if(label == std::string("mu_2_1"))
    return mu_2_1.d();
  else if(label == std::string("mu_2_0"))
    return mu_2_0.d();
  else if(label == std::string("mu_1_0"))
    return mu_1_0.d();
  else if(label == std::string("mu_9_V"))
    return mu_9_V.d();
  /*
  else if(label == std::string("mu_5_V"))
    return mu_5_V.d();
  */
  else if(label == std::string("mu_3_V"))
    return mu_3_V.d();
  else if(label == std::string("mu_2_V"))
    return mu_2_V.d();
  else if(label == std::string("mu_1_V"))
    return mu_1_V.d();
  else if(label == std::string("mu_0_V"))
    return mu_0_V.d();
  else if(label == std::string("LinFitL2L0"))
    return LinFitL2L0.d();
  else if(label == std::string("LinFitL2L1"))
    return LinFitL2L1.d();
  else if(label == std::string("LinFitL3L0"))
    return LinFitL3L0.d();
  else if(label == std::string("LinFitL3L1"))
    return LinFitL3L1.d();
  else if(label == std::string("LinFitL8L0"))
    return LinFitL8L0.d();
  else if(label == std::string("LinFitL8L1"))
    return LinFitL8L1.d();
  else if(label == std::string("LinFitL8L2"))
    return LinFitL8L2.d();
  else if(label == std::string("LinFitL8L3"))
    return LinFitL8L3.d();
  else if(label == std::string("LinFitL9L0"))
    return LinFitL9L0.d();
  else if(label == std::string("LinFitL9L1"))
    return LinFitL9L1.d();
  else if(label == std::string("LinFitL9L2"))
    return LinFitL9L2.d();
  else if(label == std::string("LinFitL9L3"))
    return LinFitL9L3.d();
  else if(label == std::string("LinStubs_9_3_0"))
    return LinStubs_9_3_0.d();
  else if(label == std::string("LinStubs_9_1_0"))
    return LinStubs_9_1_0.d();
  /*
  else if(label == std::string("LinStubs_5_3_0"))
    return LinStubs_5_3_0.d();
  else if(label == std::string("LinStubs_5_1_0"))
    return LinStubs_5_1_0.d();
  */
  else if(label == std::string("LinStubs_3_2_0"))
    return LinStubs_3_2_0.d();
  else if(label == std::string("LinStubs_3_1_0"))
    return LinStubs_3_1_0.d();
  return NAN;
}
#endif
