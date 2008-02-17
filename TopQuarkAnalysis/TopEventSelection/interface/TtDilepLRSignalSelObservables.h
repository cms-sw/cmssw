#ifndef TtDilepLRSignalSelObservables_h
#define TtDilepLRSignalSelObservables_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <string>
#include <vector>

#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"

using namespace std;

class TtDilepLRSignalSelObservables {
  
 public:
  
  TtDilepLRSignalSelObservables();
  ~TtDilepLRSignalSelObservables();	

  typedef pair<unsigned int,bool>   IntBoolPair;    
  vector< IntBoolPair > operator()(TtDilepEvtSolution&, const edm::Event & iEvent, 
				   bool matchOnly = false);
  void jetSource(const edm::InputTag & jetSource) {jetSource_ = jetSource;}
  
 private:
  
  typedef pair<unsigned int,double> IntDblPair;
  
  double delta(double phi1, double phi2);
  void fillMinMax(double v1, double v2, int obsNbr,
		  vector< IntDblPair > & varList, bool match1, bool match2, 
		  vector< IntBoolPair > & matchList);
  
  edm::InputTag jetSource_;
  
  vector< IntDblPair > evtselectVarVal;
  vector< IntBoolPair > evtselectVarMatch;
  int count1, count2, count3, count4, count5;
};

#endif
