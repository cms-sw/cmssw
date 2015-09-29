
#ifndef Validation_RecoEgamma_ElectronMcMiniAODSignalPostValidator_h
#define Validation_RecoEgamma_ElectronMcMiniAODSignalPostValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmHarvesterBase.h" 

class ElectronMcMiniAODSignalPostValidator : public ElectronDqmHarvesterBase
 {
  public:
    explicit ElectronMcMiniAODSignalPostValidator( const edm::ParameterSet & conf ) ;
    virtual ~ElectronMcMiniAODSignalPostValidator() ;
    virtual void finalize( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter ) ; 

  private:
    std::string inputFile_ ;
    std::string outputFile_ ;
    std::vector<int> matchingIDs_;
    std::vector<int> matchingMotherIDs_;
    std::string inputInternalPath_ ;
    std::string outputInternalPath_ ;

    // histos limits and binning
    bool set_EfficiencyFlag ; bool set_StatOverflowFlag ;

    // histos
//    MonitorElement *h1_ele_xOverX0VsEta ;
	
 } ;

#endif



