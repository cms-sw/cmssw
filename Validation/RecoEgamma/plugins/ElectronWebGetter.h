
#ifndef Validation_RecoEgamma_ElectronWget_h
#define Validation_RecoEgamma_ElectronWget_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

class ElectronWebGetter : public ElectronDqmAnalyzerBase
 {
  public:
    explicit ElectronWebGetter( const edm::ParameterSet & conf ) ;
    virtual ~ElectronWebGetter() ;
    virtual void finalize() ;
 } ;

#endif



