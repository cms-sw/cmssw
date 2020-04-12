
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

//#include "DataFormats/Common/interface/RefToBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
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
  //typedef dqm::legacy::DQMStore DQMStore;
  //typedef dqm::legacy::MonitorElement MonitorElement;

  explicit PFCandidateAnalyzerDQM ( const edm::ParameterSet&);

  void analyze(const edm::Event&, const edm::EventSetup&) override;

protected:
  //Book histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  //from config file
  edm::InputTag PFCandType;
  edm::EDGetTokenT<edm::View<pat::PackedCandidate>> PFCandToken;
  //edm::EDGetTokenT<pat::PackedCandidate> PFCandToken;
  //  DQMStore *dbe_;
  std::map<std::string, MonitorElement *> me;
};



// constructor
PFCandidateAnalyzerDQM::PFCandidateAnalyzerDQM(const edm::ParameterSet& iConfig) {
  PFCandType = iConfig.getParameter<edm::InputTag>("PFCandType");
  PFCandToken = consumes<edm::View<pat::PackedCandidate>>(PFCandType);
  //PFCandToken = consumes<pat::PackedCandidate>(PFCandType);  
}

void PFCandidateAnalyzerDQM::bookHistograms(DQMStore::IBooker& booker, edm::Run const&, edm::EventSetup const&) {
    booker.setCurrentFolder("RecoParticleFlow/PFCandidates");   //find out how to do PFCandType.label()

    me["CandidatePt"] = booker.book1D("CandidatePt", "CandidatePt", 1000, 0, 1000);
    me["CandidateEta"] = booker.book1D("CandidateEta", "CandidateEta", 200, -5, 5);
    me["CandidatePhi"] = booker.book1D("CandidatePhi", "CandidatePhi", 200, -M_PI, M_PI);
    me["CandidateCharge"] = booker.book1D("CandidateCharge", "CandidateCharge", 5, -2, 2);
    me["PFCandidateType"] = booker.book1D("PFCandidateType", "PFCandidateType", 10, 0, 10);
}

void PFCandidateAnalyzerDQM::analyze (const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  //const pat::PackedCandidateCollection *pflow_candidates;

  //retrieve
  edm::Handle<pat::PackedCandidate> pflow_handle;
  // edm::Handle<pat::PackedCandidateCollection> pflow_handle;
  iEvent.getByToken(PFCandToken, pflow_handle);
  
  if (!pflow_handle.isValid()) {
    edm::LogInfo("OutputInfo") << " failed to retrieve data required by ParticleFlow Task";
    edm::LogInfo("OutputInfo") << " ParticleFlow Task cannot continue...!";
    return;
  }
  else if (pflow_handle.isValid()) {
    std::cout << "it's valid?????" << std::endl; 
    //auto pflow_candidates = *pflow_handle;
    
    if(pflow_handle){
      std::cout << "if pflow_handle" << std::endl;
    }
    
    //Analyze

    // Loop Over Particle Flow Candidates
    /*
    pat::PackedCandidate::const_iterator pf;
    for (pf = pflow_candidates.begin(); pf != pflow_candidates.end(); pf++) {
      const pat::PackedCandidate *particle = &(*pf);

      // Fill Histograms for Candidate Methods
      me["CandidatePt"]->Fill(particle->pt());
      me["CandidateEta"]->Fill(particle->eta());
      me["CandidatePhi"]->Fill(particle->phi());
      me["CandidateCharge"]->Fill(particle->charge());
      me["CandidatePdgId"]->Fill(particle->pdgId());
      
      // Fill Histograms for PFCandidate Specific Methods
      //me["PFCandidateType"]->Fill(particle->particleId());
    }
    */
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateAnalyzerDQM);
