
#ifndef Validation_RecoEgamma_ElectronMcSignalPostValidator_h
#define Validation_RecoEgamma_ElectronMcSignalPostValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

class ElectronMcSignalPostValidator : public ElectronDqmAnalyzerBase
 {
  public:
    explicit ElectronMcSignalPostValidator( const edm::ParameterSet & conf ) ;
    virtual ~ElectronMcSignalPostValidator() ;
    virtual void book() ;
    virtual void finalize() ;
 } ;

#endif



