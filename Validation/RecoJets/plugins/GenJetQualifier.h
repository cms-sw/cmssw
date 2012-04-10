#ifndef GenJetQualifier_h
#define GenJetQualifier_h

#include <memory>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/GenJet.h"

class GenJetQualifier {
  
 public:
  GenJetQualifier(const edm::ParameterSet&);
  ~GenJetQualifier(){};
  bool operator()(const reco::GenJet&);

 private:

  double minEmf_, maxEmf_;
};

inline 
GenJetQualifier::GenJetQualifier(const edm::ParameterSet& cfg):
  minEmf_( cfg.getParameter<double>( "minEmfGenJet" ) ),
  maxEmf_( cfg.getParameter<double>( "maxEmfGenJet" ) )
{
}

inline bool
GenJetQualifier::operator()(const reco::GenJet& jet)
{
  if( !(minEmf_<=jet.emEnergy()/jet.energy() && jet.emEnergy()/jet.energy()<=maxEmf_) )
    // is the emf of the GenJet in the specifiedc range?
    return false;
  
  return true;
}

#endif
