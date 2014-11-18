
#ifndef Validation_RecoEgamma_ElectronMcSignalPostValidator_h
#define Validation_RecoEgamma_ElectronMcSignalPostValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmHarvesterBase.h" 

class ElectronMcSignalPostValidator : public ElectronDqmHarvesterBase
 {
  public:
    explicit ElectronMcSignalPostValidator( const edm::ParameterSet & conf ) ;
    virtual ~ElectronMcSignalPostValidator() ;
//    virtual void book() ;
    virtual void finalize( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter ) ; 

  private:
    std::string inputFile_ ;
    std::string outputFile_ ;
    std::string inputInternalPath_ ;
    std::string outputInternalPath_ ;

    // histos limits and binning
    bool set_EfficiencyFlag ; bool set_StatOverflowFlag ;

    // histos
    MonitorElement *h1_ele_xOverX0VsEta ;
	
 } ;

#endif



