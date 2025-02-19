
#ifndef Validation_RecoEgamma_ElectronMcFakePostValidator_h
#define Validation_RecoEgamma_ElectronMcFakePostValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

class ElectronMcFakePostValidator : public ElectronDqmAnalyzerBase
 {
  public:
    explicit ElectronMcFakePostValidator( const edm::ParameterSet & conf ) ;
    virtual ~ElectronMcFakePostValidator() ;
    virtual void book() ;
    virtual void finalize() ;
 } ;

#endif



