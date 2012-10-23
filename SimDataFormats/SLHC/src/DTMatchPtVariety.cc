#include "SimDataFormats/SLHC/interface/DTMatchPtVariety.h"


using namespace std;


DTMatchPtVariety::DTMatchPtVariety() {
  thePtv = vector<DTMatchPt*>();
  //
  TrackletSL0 = DTMatchPt(std::string("TrackletSL0"));
  labels.push_back(std::string("TrackletSL0"));
  thePtv.push_back(&TrackletSL0);

  TrackletSL1 = DTMatchPt(std::string("TrackletSL1"));
  labels.push_back(std::string("TrackletSL1"));
  thePtv.push_back(&TrackletSL1);

  TrackletSL4 = DTMatchPt(std::string("TrackletSL4"));
  labels.push_back(std::string("TrackletSL4"));
  thePtv.push_back(&TrackletSL4);
  //
  Mu_SL4_0 = DTMatchPt(std::string("Mu_SL4_0"));
  labels.push_back(std::string("Mu_SL4_0"));
  thePtv.push_back(&Mu_SL4_0);

  Mu_SL4_3 = DTMatchPt(std::string("Mu_SL4_3"));
  labels.push_back(std::string("Mu_SL4_3"));
  thePtv.push_back(&Mu_SL4_3);

  Mu_SL1_9 = DTMatchPt(std::string("Mu_SL1_9"));
  labels.push_back(std::string("Mu_SL1_9"));
  thePtv.push_back(&Mu_SL1_9);

  Mu_SL0_3 = DTMatchPt(std::string("Mu_SL0_3"));
  labels.push_back(std::string("Mu_SL0_3"));
  thePtv.push_back(&Mu_SL0_3);

  Mu_SL0_9 = DTMatchPt(std::string("Mu_SL0_9"));
  labels.push_back(std::string("Mu_SL0_9"));
  thePtv.push_back(&Mu_SL0_9);

  Mu_SL4_V = DTMatchPt(std::string("Mu_SL4_V"));
  labels.push_back(std::string("Mu_SL4_V"));
  thePtv.push_back(&Mu_SL4_V);

  Mu_SL0_V = DTMatchPt(std::string("Mu_SL0_V"));
  labels.push_back(std::string("Mu_SL0_V"));
  thePtv.push_back(&Mu_SL0_V);
  //
  Mu_SL0_SL1 = DTMatchPt(std::string("Mu_SL0_SL1"));
  labels.push_back(std::string("Mu_SL0_SL1"));
  thePtv.push_back(&Mu_SL0_SL1);

  Mu_SL0_SL4 = DTMatchPt(std::string("Mu_SL0_SL4"));
  labels.push_back(std::string("Mu_SL0_SL4"));
  thePtv.push_back(&Mu_SL0_SL4);

  Mu_SL1_SL4 = DTMatchPt(std::string("Mu_SL1_SL4"));
  labels.push_back(std::string("Mu_SL1_SL4"));
  thePtv.push_back(&Mu_SL1_SL4);
  //
  Mu_SL0_SL1 = DTMatchPt(std::string("Mu_SL0_SL1"));
  labels.push_back(std::string("Mu_SL0_SL1"));
  thePtv.push_back(&Mu_SL0_SL1);

  Mu_SL0_SL4 = DTMatchPt(std::string("Mu_SL0_SL4"));
  labels.push_back(std::string("Mu_SL0_SL4"));
  thePtv.push_back(&Mu_SL0_SL4);

  Mu_SL1_SL4 = DTMatchPt(std::string("Mu_SL1_SL4"));
  labels.push_back(std::string("Mu_SL1_SL4"));
  thePtv.push_back(&Mu_SL1_SL4);
  //
  Stubs_9_3_0 = DTMatchPt(std::string("Stubs_9_3_0"));
  labels.push_back(std::string("Stubs_9_3_0"));
  thePtv.push_back(&Stubs_9_3_0);

  Stubs_9_1_0 = DTMatchPt(std::string("Stubs_9_1_0")); 
  labels.push_back(std::string("Stubs_9_1_0"));
  thePtv.push_back(&Stubs_9_1_0);

  Stubs_3_2_0 = DTMatchPt(std::string("Stubs_3_2_0")); 
  labels.push_back(std::string("Stubs_3_2_0"));
  thePtv.push_back(&Stubs_3_2_0);

  Stubs_3_1_0 = DTMatchPt(std::string("Stubs_3_1_0")); 
  labels.push_back(std::string("Stubs_3_1_0"));
  thePtv.push_back(&Stubs_3_1_0);

  Stubs_9_3_V = DTMatchPt(std::string("Stubs_9_3_V")); 
  labels.push_back(std::string("Stubs_9_3_V")); 
  thePtv.push_back(&Stubs_9_3_V);

  Stubs_9_1_V = DTMatchPt(std::string("Stubs_9_1_V")); 
  labels.push_back(std::string("Stubs_9_1_V")); 
  thePtv.push_back(&Stubs_9_1_V);

  Stubs_9_0_V = DTMatchPt(std::string("Stubs_9_0_V")); 
  labels.push_back(std::string("Stubs_9_0_V")); 
  thePtv.push_back(&Stubs_9_0_V);

  Stubs_3_1_V = DTMatchPt(std::string("Stubs_3_1_V")); 
  labels.push_back(std::string("Stubs_3_1_V"));
  thePtv.push_back(&Stubs_3_1_V);

  Stubs_3_0_V = DTMatchPt(std::string("Stubs_3_0_V")); 
  labels.push_back(std::string("Stubs_3_0_V"));
  thePtv.push_back(&Stubs_3_0_V);

  Mu_9_8   = DTMatchPt(std::string("Mu_9_8")); 
  labels.push_back(std::string("Mu_9_8")); 
  thePtv.push_back(&Mu_9_8);
	
  Mu_9_3   = DTMatchPt(std::string("Mu_9_3")); 
  labels.push_back(std::string("Mu_9_3"));  
  thePtv.push_back(&Mu_9_3);
	
  Mu_9_2   = DTMatchPt(std::string("Mu_9_2")); 
  labels.push_back(std::string("Mu_9_2"));  
  thePtv.push_back(&Mu_9_2);
	
  Mu_9_1   = DTMatchPt(std::string("Mu_9_1")); 
  labels.push_back(std::string("Mu_9_1"));  
  thePtv.push_back(&Mu_9_1);
	
  Mu_9_0   = DTMatchPt(std::string("Mu_9_0")); 
  labels.push_back(std::string("Mu_9_0"));
  thePtv.push_back(&Mu_9_0);
	
  Mu_8_3   = DTMatchPt(std::string("Mu_8_3"));  
  labels.push_back(std::string("Mu_8_3"));
  thePtv.push_back(&Mu_8_3);
	
  Mu_8_2   = DTMatchPt(std::string("Mu_8_2")); 
  labels.push_back(std::string("Mu_8_2")); 
  thePtv.push_back(&Mu_8_2);
	
  Mu_8_1   = DTMatchPt(std::string("Mu_8_1"));  
  labels.push_back(std::string("Mu_8_1"));
  thePtv.push_back(&Mu_8_1);
	
  Mu_8_0   = DTMatchPt(std::string("Mu_8_0")); 
  labels.push_back(std::string("Mu_8_0"));
  thePtv.push_back(&Mu_8_0);

  Mu_3_2   = DTMatchPt(std::string("Mu_3_2")); 
  labels.push_back(std::string("Mu_3_2"));
  thePtv.push_back(&Mu_3_2);

  Mu_3_1   = DTMatchPt(std::string("Mu_3_1")); 
  labels.push_back(std::string("Mu_3_1"));
  thePtv.push_back(&Mu_3_1);

  Mu_3_0   = DTMatchPt(std::string("Mu_3_0")); 
  labels.push_back(std::string("Mu_3_0"));
  thePtv.push_back(&Mu_3_0);

  Mu_2_1   = DTMatchPt(std::string("Mu_2_1")); 
  labels.push_back(std::string("Mu_2_1"));
  thePtv.push_back(&Mu_2_1);

  Mu_2_0   = DTMatchPt(std::string("Mu_2_0")); 
  labels.push_back(std::string("Mu_2_0"));
  thePtv.push_back(&Mu_2_0);

  Mu_1_0   = DTMatchPt(std::string("Mu_1_0")); 
  labels.push_back(std::string("Mu_1_0"));
  thePtv.push_back(&Mu_1_0);

  Mu_9_V   = DTMatchPt(std::string("Mu_9_V")); 
  labels.push_back(std::string("Mu_9_V"));
  thePtv.push_back(&Mu_9_V);

  Mu_3_V   = DTMatchPt(std::string("Mu_3_V")); 
  labels.push_back(std::string("Mu_3_V"));
  thePtv.push_back(&Mu_3_V);

  Mu_2_V   = DTMatchPt(std::string("Mu_2_V")); 
  labels.push_back(std::string("Mu_2_V"));
  thePtv.push_back(&Mu_2_V);

  Mu_1_V   = DTMatchPt(std::string("Mu_1_V")); 
  labels.push_back(std::string("Mu_1_V"));
  thePtv.push_back(&Mu_1_V);

  Mu_0_V   = DTMatchPt(std::string("Mu_0_V")); 
  labels.push_back(std::string("Mu_0_V"));
  thePtv.push_back(&Mu_0_V);

  IMu_9_0   = DTMatchPt(std::string("IMu_9_0")); 
  labels.push_back(std::string("IMu_9_0"));
  thePtv.push_back(&IMu_9_0);

  IMu_3_2   = DTMatchPt(std::string("IMu_3_2")); 
  labels.push_back(std::string("IMu_3_2"));
  thePtv.push_back(&IMu_3_2);

  IMu_3_1   = DTMatchPt(std::string("IMu_3_1")); 
  labels.push_back(std::string("IMu_3_1"));
  thePtv.push_back(&IMu_3_1);

  IMu_3_0   = DTMatchPt(std::string("IMu_3_0")); 
  labels.push_back(std::string("IMu_3_0"));
  thePtv.push_back(&IMu_3_0);

  IMu_2_1   = DTMatchPt(std::string("IMu_2_1")); 
  labels.push_back(std::string("IMu_2_1"));
  thePtv.push_back(&IMu_2_1);

  IMu_2_0   = DTMatchPt(std::string("IMu_2_0")); 
  labels.push_back(std::string("IMu_2_0"));
  thePtv.push_back(&IMu_2_0);

  IMu_1_0   = DTMatchPt(std::string("IMu_1_0")); 
  labels.push_back(std::string("IMu_1_0"));
  thePtv.push_back(&IMu_1_0);

  IMu_9_V   = DTMatchPt(std::string("IMu_9_V")); 
  labels.push_back(std::string("IMu_9_V"));
  thePtv.push_back(&IMu_9_V);

  IMu_3_V   = DTMatchPt(std::string("IMu_3_V")); 
  labels.push_back(std::string("IMu_3_V"));
  thePtv.push_back(&IMu_3_V);

  IMu_2_V   = DTMatchPt(std::string("IMu_2_V")); 
  labels.push_back(std::string("IMu_2_V"));
  thePtv.push_back(&IMu_2_V);

  IMu_1_V   = DTMatchPt(std::string("IMu_1_V")); 
  labels.push_back(std::string("IMu_1_V"));
  thePtv.push_back(&IMu_1_V);

  IMu_0_V   = DTMatchPt(std::string("IMu_0_V")); 
  labels.push_back(std::string("IMu_0_V"));
  thePtv.push_back(&IMu_0_V);

  mu_9_0  = DTMatchPt(std::string("mu_9_0")); 
  labels.push_back(std::string("mu_9_0"));
  thePtv.push_back(&mu_9_0);

  mu_3_2  = DTMatchPt(std::string("mu_3_2")); 
  labels.push_back(std::string("mu_3_2"));
  thePtv.push_back(&mu_3_2);

  mu_3_1  = DTMatchPt(std::string("mu_3_1")); 
  labels.push_back(std::string("mu_3_1"));
  thePtv.push_back(&mu_3_1);

  mu_3_0  = DTMatchPt(std::string("mu_3_0")); 
  labels.push_back(std::string("mu_3_0"));
  thePtv.push_back(&mu_3_0);

  mu_2_1  = DTMatchPt(std::string("mu_2_1")); 
  labels.push_back(std::string("mu_2_1"));
  thePtv.push_back(&mu_2_1);

  mu_2_0  = DTMatchPt(std::string("mu_2_0")); 
  labels.push_back(std::string("mu_2_0"));
  thePtv.push_back(&mu_2_0);

  mu_1_0  = DTMatchPt(std::string("mu_1_0")); 
  labels.push_back(std::string("mu_1_0"));
  thePtv.push_back(&mu_1_0);

  mu_9_V  = DTMatchPt(std::string("mu_9_V")); 
  labels.push_back(std::string("mu_9_V"));
  thePtv.push_back(&mu_9_V);

  mu_3_V  = DTMatchPt(std::string("mu_3_V")); 
  labels.push_back(std::string("mu_3_V"));
  thePtv.push_back(&mu_3_V);

  mu_2_V  = DTMatchPt(std::string("mu_2_V")); 
  labels.push_back(std::string("mu_2_V"));
  thePtv.push_back(&mu_2_V);

  mu_1_V  = DTMatchPt(std::string("mu_1_V")); 
  labels.push_back(std::string("mu_1_V"));
  thePtv.push_back(&mu_1_V);

  mu_0_V  = DTMatchPt(std::string("mu_0_V")); 
  labels.push_back(std::string("mu_0_V"));
  thePtv.push_back(&mu_0_V);

  LinFitL2L0  = DTMatchPt(std::string("LinFitL2L0")); 
  labels.push_back(std::string("LinFitL2L0"));
  thePtv.push_back(&LinFitL2L0);

  LinFitL2L1  = DTMatchPt(std::string("LinFitL2L1")); 
  labels.push_back(std::string("LinFitL2L1"));
  thePtv.push_back(&LinFitL2L1);

  LinFitL3L0  = DTMatchPt(std::string("LinFitL3L0")); 
  labels.push_back(std::string("LinFitL3L0"));
  thePtv.push_back(&LinFitL3L0);

  LinFitL3L1  = DTMatchPt(std::string("LinFitL3L1")); 
  labels.push_back(std::string("LinFitL3L1"));
  thePtv.push_back(&LinFitL3L1);

  LinFitL8L0  = DTMatchPt(std::string("LinFitL8L0")); 
  labels.push_back(std::string("LinFitL8L0"));
  thePtv.push_back(&LinFitL8L0);

  LinFitL8L1  = DTMatchPt(std::string("LinFitL8L1")); 
  labels.push_back(std::string("LinFitL8L1"));
  thePtv.push_back(&LinFitL8L1);

  LinFitL8L2  = DTMatchPt(std::string("LinFitL8L2")); 
  labels.push_back(std::string("LinFitL8L2"));
  thePtv.push_back(&LinFitL8L2);

  LinFitL8L3  = DTMatchPt(std::string("LinFitL8L3")); 
  labels.push_back(std::string("LinFitL8L3"));
  thePtv.push_back(&LinFitL8L3);

  LinFitL9L0  = DTMatchPt(std::string("LinFitL9L0")); 
  labels.push_back(std::string("LinFitL9L0"));
  thePtv.push_back(&LinFitL9L0);

  LinFitL9L1  = DTMatchPt(std::string("LinFitL9L1")); 
  labels.push_back(std::string("LinFitL9L1"));
  thePtv.push_back(&LinFitL9L1);
  //
  LinFitL9L2  = DTMatchPt(std::string("LinFitL9L2")); 
  labels.push_back(std::string("LinFitL9L2"));
  thePtv.push_back(&LinFitL9L2);

  LinFitL9L3  = DTMatchPt(std::string("LinFitL9L3")); 
  labels.push_back(std::string("LinFitL9L3"));
  thePtv.push_back(&LinFitL9L3);

  LinStubs_9_3_0  = DTMatchPt(std::string("LinStubs_9_3_0")); 
  labels.push_back(std::string("LinStubs_9_3_0"));
  thePtv.push_back(&LinStubs_9_3_0);

  LinStubs_9_1_0  = DTMatchPt(std::string("LinStubs_9_1_0")); 
  labels.push_back(std::string("LinStubs_9_1_0"));
  thePtv.push_back(&LinStubs_9_1_0);
  //
  LinStubs_3_2_0  = DTMatchPt(std::string("LinStubs_3_2_0")); 
  labels.push_back(std::string("LinStubs_3_2_0"));
  thePtv.push_back(&LinStubs_3_2_0);

  LinStubs_3_1_0  = DTMatchPt(std::string("LinStubs_3_1_0")); 
  labels.push_back(std::string("LinStubs_3_1_0"));
  thePtv.push_back(&LinStubs_3_1_0);
}




// copy constructor
DTMatchPtVariety::DTMatchPtVariety(const DTMatchPtVariety& pts) {
  TrackletSL0 = pts.TrackletSL0;
  TrackletSL1 = pts.TrackletSL1;
  TrackletSL4 = pts.TrackletSL4;
  //
  Mu_SL4_0 = pts.Mu_SL4_0;
  Mu_SL4_3 = pts.Mu_SL4_3;
  Mu_SL1_0 = pts.Mu_SL1_0;
  Mu_SL1_9 = pts.Mu_SL1_9;
  Mu_SL0_3 = pts.Mu_SL0_3;
  Mu_SL0_9 = pts.Mu_SL0_9;
  Mu_SL4_V = pts.Mu_SL4_V;
  Mu_SL1_V = pts.Mu_SL1_V;
  Mu_SL0_V = pts.Mu_SL0_V;
  //
  Mu_SL0_SL1 = pts.Mu_SL0_SL1;
  Mu_SL0_SL4 = pts.Mu_SL0_SL4;
  Mu_SL1_SL4 = pts.Mu_SL1_SL4;
  //
  Stubs_9_3_0 = pts.Stubs_9_3_0;
  Stubs_9_1_0 = pts.Stubs_9_1_0;
  Stubs_3_2_0 = pts.Stubs_3_2_0; 
  Stubs_3_1_0 = pts.Stubs_3_1_0; 
  Stubs_9_3_V = pts.Stubs_9_3_V;
  Stubs_9_1_V = pts.Stubs_9_1_V;
  Stubs_9_0_V = pts.Stubs_9_0_V;
  Stubs_3_1_V = pts.Stubs_3_1_V;
  Stubs_3_0_V = pts.Stubs_3_0_V;
  Mu_9_8   = pts.Mu_9_8;
  Mu_9_3   = pts.Mu_9_3;
  Mu_9_2   = pts.Mu_9_2;
  Mu_9_1   = pts.Mu_9_1;
  Mu_9_0   = pts.Mu_9_0;
  Mu_8_3   = pts.Mu_8_3;
  Mu_8_2   = pts.Mu_8_2;
  Mu_8_1   = pts.Mu_8_1;
  Mu_8_0   = pts.Mu_8_0;
  Mu_3_2   = pts.Mu_3_2;
  Mu_3_1   = pts.Mu_3_1;
  Mu_3_0   = pts.Mu_3_0;
  Mu_2_1   = pts.Mu_2_1;
  Mu_2_0   = pts.Mu_2_0;
  Mu_1_0   = pts.Mu_1_0;
  Mu_9_V   = pts.Mu_9_V;
  Mu_3_V   = pts.Mu_3_V;
  Mu_2_V   = pts.Mu_2_V;
  Mu_1_V   = pts.Mu_1_V;
  Mu_0_V   = pts.Mu_0_V;
  IMu_9_0   = pts.IMu_9_0;
  IMu_3_2   = pts.IMu_3_2;
  IMu_3_1   = pts.IMu_3_1;
  IMu_3_0   = pts.IMu_3_0;
  IMu_2_1   = pts.IMu_2_1;
  IMu_2_0   = pts.IMu_2_0;
  IMu_1_0   = pts.IMu_1_0;
  IMu_9_V   = pts.IMu_9_V;
  IMu_3_V   = pts.IMu_3_V;
  IMu_2_V   = pts.IMu_2_V;
  IMu_1_V   = pts.IMu_1_V;
  IMu_0_V   = pts.IMu_0_V;
  mu_9_0   = pts.mu_9_0;
  mu_3_2   = pts.mu_3_2;
  mu_3_1   = pts.mu_3_1;
  mu_3_0   = pts.mu_3_0;
  mu_2_1   = pts.mu_2_1;
  mu_2_0   = pts.mu_2_0;
  mu_1_0   = pts.mu_1_0;
  mu_9_V   = pts.mu_9_V;
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
  LinStubs_3_2_0 = pts.LinStubs_3_2_0;
  LinStubs_3_1_0 = pts.LinStubs_3_1_0;
  //
  labels = pts.labels;
  thePtv = pts.thePtv;
}




// assignment constructor
  DTMatchPtVariety& 
  DTMatchPtVariety::operator =(const DTMatchPtVariety& pts) {
  if (this == &pts)      // Same object?
    return *this;        // Yes, so skip assignment, and just return *this.
  //
  TrackletSL0 = pts.TrackletSL0;
  TrackletSL1 = pts.TrackletSL1;
  TrackletSL4 = pts.TrackletSL4;
  //
  Mu_SL4_0 = pts.Mu_SL4_0;
  Mu_SL4_3 = pts.Mu_SL4_3;
  Mu_SL1_0 = pts.Mu_SL1_0;
  Mu_SL1_9 = pts.Mu_SL1_9;
  Mu_SL0_3 = pts.Mu_SL0_3;
  Mu_SL0_9 = pts.Mu_SL0_9;
  Mu_SL4_V = pts.Mu_SL4_V;
  Mu_SL1_V = pts.Mu_SL1_V;
  Mu_SL0_V = pts.Mu_SL0_V;
  //
  Mu_SL0_SL1 = pts.Mu_SL0_SL1;
  Mu_SL0_SL4 = pts.Mu_SL0_SL4;
  Mu_SL1_SL4 = pts.Mu_SL1_SL4;
  //
  Stubs_9_3_0 = pts.Stubs_9_3_0;
  Stubs_9_1_0 = pts.Stubs_9_1_0;
  Stubs_3_2_0 = pts.Stubs_3_2_0; 
  Stubs_3_1_0 = pts.Stubs_3_1_0; 
  Stubs_9_3_V = pts.Stubs_9_3_V;
  Stubs_9_1_V = pts.Stubs_9_1_V;
  Stubs_9_0_V = pts.Stubs_9_0_V; 
  Stubs_3_1_V = pts.Stubs_3_1_V;
  Stubs_3_0_V = pts.Stubs_3_0_V;
  Mu_9_8   = pts.Mu_9_8;
  Mu_9_3   = pts.Mu_9_3;
  Mu_9_2   = pts.Mu_9_2;
  Mu_9_1   = pts.Mu_9_1;
  Mu_9_0   = pts.Mu_9_0;
  Mu_8_3   = pts.Mu_8_3;
  Mu_8_2   = pts.Mu_8_2;
  Mu_8_1   = pts.Mu_8_1;
  Mu_8_0   = pts.Mu_8_0;
  Mu_3_2   = pts.Mu_3_2;
  Mu_3_1   = pts.Mu_3_1;
  Mu_3_0   = pts.Mu_3_0;
  Mu_2_1   = pts.Mu_2_1;
  Mu_2_0   = pts.Mu_2_0;
  Mu_1_0   = pts.Mu_1_0;
  Mu_9_V   = pts.Mu_9_V;
  Mu_3_V   = pts.Mu_3_V;
  Mu_2_V   = pts.Mu_2_V;
  Mu_1_V   = pts.Mu_1_V;
  Mu_0_V   = pts.Mu_0_V;
  IMu_9_0   = pts.IMu_9_0;
  IMu_3_2   = pts.IMu_3_2;
  IMu_3_1   = pts.IMu_3_1;
  IMu_3_0   = pts.IMu_3_0;
  IMu_2_1   = pts.IMu_2_1;
  IMu_2_0   = pts.IMu_2_0;
  IMu_1_0   = pts.IMu_1_0;
  IMu_9_V   = pts.IMu_9_V;
  IMu_3_V   = pts.IMu_3_V;
  IMu_2_V   = pts.IMu_2_V;
  IMu_1_V   = pts.IMu_1_V;
  IMu_0_V   = pts.IMu_0_V;
  mu_9_0   = pts.mu_9_0;
  mu_3_2   = pts.mu_3_2;
  mu_3_1   = pts.mu_3_1;
  mu_3_0   = pts.mu_3_0;
  mu_2_1   = pts.mu_2_1;
  mu_2_0   = pts.mu_2_0;
  mu_1_0   = pts.mu_1_0;
  mu_9_V   = pts.mu_9_V;
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
  LinStubs_3_2_0 = pts.LinStubs_3_2_0;
  LinStubs_3_1_0 = pts.LinStubs_3_1_0;
  //
  labels = pts.labels; 
  thePtv = pts.thePtv;
  //
  return *this;
  }




void DTMatchPtVariety::assignPt(const edm::ParameterSet&  pSet, 
				    const size_t s, 
				    const DTMatchPt* aPt)
{
  *thePtv[s] = DTMatchPt(*aPt);
  return;
}





float const DTMatchPtVariety::Pt(std::string const label) const
{
  vector<string>::const_iterator itx = 
    find(labels.begin(), labels.end(), label);
  if(itx != labels.end()) {
    size_t idx = itx - labels.begin();
    return thePtv[idx]->Pt();
  }
  else
    return NAN;
} 





float const DTMatchPtVariety::alpha0(std::string const label) const
{
  vector<string>::const_iterator itx = 
    find(labels.begin(), labels.end(), label);
  if(itx != labels.end()) {
    size_t idx = itx - labels.begin();
    return thePtv[idx]->alpha0();
  }
  else
    return NAN;
} 






float const DTMatchPtVariety::d(std::string const label) const
{
  vector<string>::const_iterator itx = 
    find(labels.begin(), labels.end(), label);
  if(itx != labels.end()) {
    size_t idx = itx - labels.begin();
    return thePtv[idx]->Pt();
  }
  else
    return NAN;
} 
