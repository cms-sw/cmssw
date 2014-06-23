
#ifndef Validation_RecoEgamma_ElectronMcSignalPostValidator_h
#define Validation_RecoEgamma_ElectronMcSignalPostValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

class ElectronMcSignalPostValidator : public ElectronDqmAnalyzerBase
 {
  public:
    explicit ElectronMcSignalPostValidator( const edm::ParameterSet & conf ) ;
    virtual ~ElectronMcSignalPostValidator() ;
//    virtual void book() ;
    virtual void bookHistograms( DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) ;
    virtual void finalize() ;

  private:
    // histos
    MonitorElement *h1_ele_xOverX0VsEta ;
	
 } ;

#endif



