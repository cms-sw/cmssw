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

class TtDilepLRSignalSelObservables {
  
 public:
  
  TtDilepLRSignalSelObservables();
  ~TtDilepLRSignalSelObservables();	

  typedef std::pair<unsigned int,bool>   IntBoolPair;    
  std::vector< IntBoolPair > operator()(TtDilepEvtSolution&, const edm::Event & iEvent, 
				   bool matchOnly = false);
  void jetSource(const edm::InputTag & jetSource) {jetSource_ = jetSource;}
  
 private:
  
  typedef std::pair<unsigned int,double> IntDblPair;
  
  double delta(double phi1, double phi2);
  void fillMinMax(double v1, double v2, int obsNbr,
		  std::vector< IntDblPair > & varList, bool match1, bool match2, 
		  std::vector< IntBoolPair > & matchList);
  
  edm::InputTag jetSource_;
  
  std::vector< IntDblPair > evtselectVarVal;
  std::vector< IntBoolPair > evtselectVarMatch;
  int count1, count2, count3, count4, count5;
};

#endif
