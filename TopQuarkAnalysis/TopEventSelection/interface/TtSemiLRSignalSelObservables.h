#ifndef TtSemiLRSignalSelObservables_h
#define TtSemiLRSignalSelObservables_h


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// General C++ stuff
#include <iostream>
#include <string>
#include <vector>

// ROOT classes
#include "TLorentzVector.h"
#include "TVector.h"
#include "TVector3.h"
#include "TVectorD.h"

#include "TMatrix.h"
#include "TMatrixDSymEigen.h"
#include "TMatrixDSym.h"
#include "TMatrixTSym.h"

//TQAF classes
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "TopQuarkAnalysis/TopTools/interface/MEzCalculator.h"

const double PI=3.14159265;

using namespace std;

class TtSemiLRSignalSelObservables{
  
  public:

    TtSemiLRSignalSelObservables();
    ~TtSemiLRSignalSelObservables();	

    void  operator()(TtSemiEvtSolution&, const std::vector<TopJet>&);
   
  private:

    vector<pair<unsigned int,double> > evtselectVarVal;

    // compare two jets in ET
    struct CompareET {
    	bool operator()( TopJet j1, TopJet j2 ) const
    	{
		return j1.et() > j2.et();
    	}
    };

    CompareET EtComparator;

    // compare two jets in bdisc
    struct CompareBdisc {
  	bool operator()( TopJet j1, TopJet j2 ) const
  	{
  		return j1.getBDiscriminator("trackCountingJetTags") > j2.getBDiscriminator("trackCountingJetTags");
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
