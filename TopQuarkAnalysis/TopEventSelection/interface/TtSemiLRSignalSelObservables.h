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
#include "TVectorD.h"

#include "TMatrix.h"
#include "TMatrixDSymEigen.h"
#include "TMatrixDSym.h"
#include "TMatrixTSym.h"

//own code
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"

const double PI=3.14159265;

using namespace std;

class TtSemiLRSignalSelObservables{
  
  public:

    TtSemiLRSignalSelObservables();
    ~TtSemiLRSignalSelObservables();	

    void  operator()(TtSemiEvtSolution&);
    
    
  private:

    // compare two jets in ET
    struct CompareET {
    	bool operator()( TopJet j1, TopJet j2 ) const
    	{
    		return j1.getRecJet().et() > j2.getRecJet().et();
    	}
    		};

    CompareET EtComparator;

    // compare two jets in bdisc
    struct CompareBdisc {
  	bool operator()( TopJet j1, TopJet j2 ) const
  	{
  		return j1.getBDiscriminator() > j2.getBDiscriminator();
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
  
    vector<pair<unsigned int,double> > evtselectVarVal;
    
};

#endif
