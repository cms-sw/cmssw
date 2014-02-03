#ifndef CaloJetQualifier_h
#define CaloJetQualifier_h

#include <memory>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

class CaloJetQualifier {
  
 public:
  CaloJetQualifier(const edm::ParameterSet&);
  ~CaloJetQualifier(){};
  bool operator()(const reco::CaloJet&);

 private:

  double minEmf_, maxEmf_;
};

inline 
CaloJetQualifier::CaloJetQualifier(const edm::ParameterSet& cfg):
  minEmf_( cfg.getParameter<double>( "minEmfCaloJet" ) ),
  maxEmf_( cfg.getParameter<double>( "maxEmfCaloJet" ) )
{
}

inline bool
CaloJetQualifier::operator()(const reco::CaloJet& jet)
{
  if( !(minEmf_<=jet.emEnergyFraction() && jet.emEnergyFraction()<=maxEmf_) )
    // is the emf of the CaloJet in the specifiedc range?
    return false;   
  
  return true;
}

#endif
