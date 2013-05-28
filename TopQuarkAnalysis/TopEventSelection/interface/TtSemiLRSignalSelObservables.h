#ifndef TtSemiLRSignalSelObservables_h
#define TtSemiLRSignalSelObservables_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <string>
#include <vector>

#include "TLorentzVector.h"
#include "TVector.h"
#include "TVector3.h"
#include "TVectorD.h"

#include "TMatrix.h"
#include "TMatrixDSymEigen.h"
#include "TMatrixDSym.h"
#include "TMatrixTSym.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "TopQuarkAnalysis/TopTools/interface/MEzCalculator.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

const double PI=3.14159265;

class TtSemiLRSignalSelObservables{
  
 public:
  
  TtSemiLRSignalSelObservables();
  ~TtSemiLRSignalSelObservables();	
  
  void  operator()(TtSemiEvtSolution&, const std::vector<pat::Jet>&);
  
 private:
  
  std::vector<std::pair<unsigned int,double> > evtselectVarVal;
  
  // compare two jets in ET
  struct CompareET {
    bool operator()( const pat::Jet& j1, const pat::Jet& j2 ) const
    {
      return j1.et() > j2.et();
    }
  };
  
  CompareET EtComparator;
  
  // compare two jets in bdisc
  struct CompareBdisc {
    bool operator()( const pat::Jet& j1, const pat::Jet& j2 ) const
    {
      return j1.bDiscriminator("trackCountingJetTags") > j2.bDiscriminator("trackCountingJetTags");
    }
  };
  
  CompareBdisc BdiscComparator;
  
  // compare two double
  struct CompareDouble {
    bool operator()( double j1, double j2 ) const
    {
      return j1 > j2 ;
    }
  };
  
  CompareDouble dComparator;
};

#endif
