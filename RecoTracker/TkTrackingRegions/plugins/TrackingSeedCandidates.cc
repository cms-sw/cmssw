#include "TrackingSeedCandidates.h"

TrackingSeedCandidates::TrackingSeedCandidates(const edm::ParameterSet& regPSet, edm::ConsumesCollector& iC) {
  // operation mode
  //
    std::string seedingModeString = regPSet.getParameter<std::string>("seedingMode");
    if      (seedingModeString == "Candidate")      m_seedingMode = SeedingMode::CANDIDATE_SEEDED;
    else if (seedingModeString == "Global")          m_seedingMode = SeedingMode::GLOBAL;
    else throw edm::Exception(edm::errors::Configuration) << "Unknown seeding mode string: "<<seedingModeString;

    m_deltaEta_Cand = regPSet.getParameter<double>("deltaEta_Cand"); 
    m_deltaPhi_Cand = regPSet.getParameter<double>("deltaPhi_Cand");
 
    // basic inputs
    if(m_seedingMode == SeedingMode::CANDIDATE_SEEDED)
      m_token_input        = iC.consumes<reco::CandidateView>(regPSet.getParameter<edm::InputTag>("input"));

}

void TrackingSeedCandidates::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<std::string>("seedingMode", "Global");
  const std::string defaultInput = ""; 
  desc.add<edm::InputTag>("input",defaultInput);
  desc.add<double>("deltaEta_Cand", -1.);
  desc.add<double>("deltaPhi_Cand", -1.);
 
}

TrackingSeedCandidates::Objects TrackingSeedCandidates::objects(const edm::Event& iEvent) const {
 
 Objects result;

 std::pair <float,float> dimensions = std::make_pair(m_deltaEta_Cand,m_deltaPhi_Cand);
 edm::Handle< reco::CandidateView > objects;
  
 if(m_seedingMode == SeedingMode::CANDIDATE_SEEDED){
      iEvent.getByToken( m_token_input, objects ); 

      result = std::make_pair(objects.product(),dimensions); 
  }
  else result = std::make_pair(nullptr,dimensions); 
  return result;
}
