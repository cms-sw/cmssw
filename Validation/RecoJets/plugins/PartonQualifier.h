#ifndef PartonQualifier_h
#define PartonQualifier_h

#include <memory>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class PartonQualifier {
  
 public:
  PartonQualifier(const edm::ParameterSet&);
  ~PartonQualifier(){};
  bool operator()(const reco::GenParticle&);

 private:

  int status_;
  std::vector<int> partons_;
};

inline 
PartonQualifier::PartonQualifier(const edm::ParameterSet& cfg):
  status_ ( cfg.getParameter<int>( "status" ) ),
  partons_( cfg.getParameter<std::vector<int> >( "partons" ) )
{
}

inline bool
PartonQualifier::operator()(const reco::GenParticle& part)
{
  if( part.status()!=status_) 
    // does the particle have the correct status?
    return false;
  
  if( !(std::count(partons_.begin(), partons_.end(), fabs(part.pdgId()))>0) )
    // is the particle pdg contained in the list of partons?
    return false;
  
  return true;
}

#endif
