#include "vector"
#include "string.h"
#include <iostream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class TtDecayChannelSelector {

 public:

  /// leafs of leptonic decay channel vector decay_ 
  enum {Elec=0, Muon=1, Tau =2};
  /// typedef to simplify the decay vectors    
  typedef std::vector<int> Decay;

  /// std contructor
  TtDecayChannelSelector(const edm::ParameterSet&);
  /// default destructor
  ~TtDecayChannelSelector();
  /// operator for decay channel selection
  bool operator()(const reco::GenParticleCollection& parts, std::string inputType) const;

 private:

  /// return decay channel to select for from configuration
  unsigned int decayChannel() const;
  // return the check sum of all entries
  unsigned int checkSum(const Decay& vec) const;
  /// search for particle with pdgId in given listing (for top)
  bool search(reco::GenParticleCollection::const_iterator& part, int pdgId, std::string& inputType) const;
  /// search for particle with pdgId in given listing (for top daughters)
  bool search(reco::GenParticle::const_iterator& part, int pdgId, std::string& inputType) const;
  /// check tau decay to be leptonic, 1-prong or 3-prong
  bool tauDecay(const reco::Candidate&) const;
  /// count the number of charged particles for tau decays
  unsigned int countProngs(const reco::Candidate& part) const;

 private:

  /// invert selection
  bool  invert_;  
  /// restrict tau decays
  bool restrictTauDecays_;
  /// allow tau decays into electron
  bool allowElectron_;
  /// allow tau decays into muon
  bool allowMuon_;
  /// allow 1-prong tau decays
  bool allow1Prong_;
  /// allow 2-prong tau decays
  bool allow3Prong_;
  /// top decay branch 1
  Decay decayBranchA_;
  /// top decay branch 2
  Decay decayBranchB_;
  /// vector of allowed lepton decay channels; values
  /// may be 0,1,2 for the entries 'Elec','Muon','Tau'
  Decay allowedDecays_;
};

inline unsigned int
TtDecayChannelSelector::decayChannel() const
{
  unsigned int channel=0;
  if( std::count(decayBranchA_.begin(), decayBranchA_.end(), 1) > 0 ){ 
    ++channel; 
  }
  if( std::count(decayBranchB_.begin(), decayBranchB_.end(), 1) > 0 ){ 
    ++channel; 
  }
  return channel;
}

inline unsigned int
TtDecayChannelSelector::checkSum(const Decay& vec) const
{
  unsigned int sum=0;
  for(unsigned int d=0; d<vec.size(); ++d){
    sum+=vec[d];
  }
  return sum;
}
