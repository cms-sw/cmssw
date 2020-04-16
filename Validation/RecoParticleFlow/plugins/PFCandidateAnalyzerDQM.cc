#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <map>
#include <string>


class PFCandidateAnalyzerDQM : public DQMEDAnalyzer {
public:
  explicit PFCandidateAnalyzerDQM ( const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  //from config file
  edm::EDGetTokenT<edm::View<pat::PackedCandidate>> PFCandToken;
  std::map<std::string, MonitorElement *> me;
  
  std::map<uint32_t, std::string> pdgMap;
};



// constructor
PFCandidateAnalyzerDQM::PFCandidateAnalyzerDQM(const edm::ParameterSet& iConfig) {
  PFCandToken = consumes<edm::View<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("PFCandType"));
  
  //create map of pdgId 
  std::vector<uint32_t> pdgKeys = iConfig.getParameter<std::vector<uint32_t>>("pdgKeys");
  std::vector<std::string> pdgStrs = iConfig.getParameter<std::vector<std::string>>("pdgStrs");
  for (int i = 0, n = pdgKeys.size(); i < n; i++)
    pdgMap[pdgKeys[i]] = pdgStrs[i];
}

void PFCandidateAnalyzerDQM::bookHistograms(DQMStore::IBooker& booker, edm::Run const&, edm::EventSetup const&) {
    // all candidate
  
  booker.setCurrentFolder("ParticleFlow/PFCandidates/AllCandidates");
  
  me["CandidatePt"] = booker.book1D("CandidatePt", "CandidatePt", 1000, 0, 1000);
  me["CandidateEta"] = booker.book1D("CandidateEta", "CandidateEta", 200, -5, 5);
  me["CandidatePhi"] = booker.book1D("CandidatePhi", "CandidatePhi", 200, -M_PI, M_PI);
  me["CandidateCharge"] = booker.book1D("CandidateCharge", "CandidateCharge", 5, -2, 2);
    
  for (auto& pair: pdgMap){
    booker.setCurrentFolder("ParticleFlow/PFCandidates/" + pair.second);
    me[pair.second + "Pt"] = booker.book1D(pair.second + "Pt", pair.second + "Pt", 1000, 0, 1000);
    me[pair.second + "Eta"] = booker.book1D(pair.second + "Eta", pair.second + "Eta", 200, -5, 5);
    me[pair.second + "Phi"] = booker.book1D(pair.second + "Phi", pair.second + "Phi", 200, -M_PI, M_PI);
    me[pair.second + "Charge"] = booker.book1D(pair.second + "Charge", pair.second + "Charge", 5, -2, 2);
  }

}

void PFCandidateAnalyzerDQM::analyze (const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  
  //retrieve
  edm::Handle<edm::View<pat::PackedCandidate>> pfHandle;
  iEvent.getByToken(PFCandToken, pfHandle);
  
  if (!pfHandle.isValid()) {
    edm::LogInfo("OutputInfo") << " failed to retrieve data required by ParticleFlow Task";
    edm::LogInfo("OutputInfo") << " ParticleFlow Task cannot continue...!";
    return;
  }
  else {
    //Analyze
    // Loop Over Particle Flow Candidates
    
    for (unsigned int i = 0; i < pfHandle->size() ; i++){ 
      // Fill Histograms for Candidate Methods      
      // all candidates
      me["CandidatePt"]->Fill(pfHandle->at(i).pt());
      me["CandidateEta"]->Fill(pfHandle->at(i).eta());
      me["CandidatePhi"]->Fill(pfHandle->at(i).phi());
      me["CandidateCharge"]->Fill(pfHandle->at(i).charge());
      //me["CandidatePdgId"]->Fill(pfHandle->at(i).pdgId());
      
      // Fill Histograms for PFCandidate Specific Methods
      int pdgId = abs(pfHandle->at(i).pdgId());
      if(pdgMap.find(pdgId) != pdgMap.end()){
	me[pdgMap[pdgId] + "Pt"]->Fill(pfHandle->at(i).pt());
	me[pdgMap[pdgId] +"Eta"]->Fill(pfHandle->at(i).eta());
	me[pdgMap[pdgId] +"Phi"]->Fill(pfHandle->at(i).phi());
	me[pdgMap[pdgId] +"Charge"]->Fill(pfHandle->at(i).charge());
      }


    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateAnalyzerDQM);
