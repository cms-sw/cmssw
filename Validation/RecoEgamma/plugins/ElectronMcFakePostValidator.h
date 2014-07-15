
#ifndef Validation_RecoEgamma_ElectronMcFakePostValidator_h
#define Validation_RecoEgamma_ElectronMcFakePostValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

class ElectronMcFakePostValidator : public ElectronDqmAnalyzerBase
 {
  public:
    explicit ElectronMcFakePostValidator( const edm::ParameterSet & conf ) ;
    virtual ~ElectronMcFakePostValidator() ;
//    virtual void book() ;
    virtual void bookHistograms( DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) ;
    virtual void finalize() ; 

  private: 
    // histos
    MonitorElement *h1_ele_xOverX0VsEta ;
	
 } ;

#endif



