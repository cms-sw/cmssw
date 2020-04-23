// Producer for particle flow candidates. Plots Eta, Phi, Charge, Pt (log freq, bin)
// for different types of particles described in python/defaults_cfi.py
// note: for pt, log freq is done in this producer, but log freq is done by running
// compare.py
// author: Chosila Sutantawibul, April 23, 2020

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

#include "TH1F.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <map>
#include <string>
#include <cstring>

class PFCandidateAnalyzerDQM : public DQMEDAnalyzer {
public:
  explicit PFCandidateAnalyzerDQM(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  //from config file
  edm::InputTag PFCandTag;
  edm::EDGetTokenT<edm::View<pat::PackedCandidate>> PFCandToken;
  std::vector<double> etabins;
  std::map<std::string, MonitorElement*> me;

  std::map<uint32_t, std::string> pdgMap;
};

// constructor
PFCandidateAnalyzerDQM::PFCandidateAnalyzerDQM(const edm::ParameterSet& iConfig) {
  PFCandTag = iConfig.getParameter<edm::InputTag>("PFCandType");
  PFCandToken = consumes<edm::View<pat::PackedCandidate>>(PFCandTag);
  etabins = iConfig.getParameter<std::vector<double>>("etabins");

  //create map of pdgId
  std::vector<uint32_t> pdgKeys = iConfig.getParameter<std::vector<uint32_t>>("pdgKeys");
  std::vector<std::string> pdgStrs = iConfig.getParameter<std::vector<std::string>>("pdgStrs");
  for (int i = 0, n = pdgKeys.size(); i < n; i++)
    pdgMap[pdgKeys[i]] = pdgStrs[i];
}

void PFCandidateAnalyzerDQM::bookHistograms(DQMStore::IBooker& booker, edm::Run const&, edm::EventSetup const&) {
  // all candidate
  booker.setCurrentFolder("ParticleFlow/PackedCandidates/AllCandidate");

  // for eta binning
  int n = etabins.size() - 1;
  float etabinArray[etabins.size()];
  std::copy(etabins.begin(), etabins.end(), etabinArray);

  //eta has variable bin sizes, use 4th def of TH1F constructor
  TH1F* etaHist = new TH1F("AllCandidateEta", "AllCandidateEta", n, etabinArray);
  me["AllCandidateEta"] = booker.book1D("AllCandidateEta", etaHist);

  me["AllCandidatePt(log GeV)"] = booker.book1D("AllCandidatePt(log GeV)", "AllCandidatePt(log GeV)", 140, -2, 4);
  me["AllCandidatePt(log y)"] = booker.book1D("AllCandidatePt(log y)", "AllCandidatePt(log y)", 500, 0, 1000);
  me["AllCandidatePhi"] = booker.book1D("AllCandidatePhi", "AllCandidatePhi", 72, -M_PI, M_PI);
  me["AllCandidateCharge"] = booker.book1D("AllCandidateCharge", "AllCandidateCharge", 3, -1.5, 1.5);

  for (auto& pair : pdgMap) {
    booker.setCurrentFolder("ParticleFlow/PackedCandidates/" + pair.second);

    //TH1F only takes char*, so have to do conversions for histogram name
    const char* etaHistName = (pair.second + "Eta").c_str();
    TH1F* etaHist = new TH1F(etaHistName, etaHistName, n, etabinArray);
    me[pair.second + "Eta"] = booker.book1D(pair.second + "Eta", etaHist);

    me[pair.second + "Pt(log GeV)"] =
        booker.book1D(pair.second + "Pt(log GeV)", pair.second + "Pt(log GeV)", 140, -2, 4);
    me[pair.second + "Pt(log y)"] = booker.book1D(pair.second + "Pt(log y)", pair.second + "Pt(log y)", 500, 0, 1000);
    me[pair.second + "Phi"] = booker.book1D(pair.second + "Phi", pair.second + "Phi", 72, -M_PI, M_PI);
    me[pair.second + "Charge"] = booker.book1D(pair.second + "Charge", pair.second + "Charge", 3, -1.5, 1.5);
  }
}

void PFCandidateAnalyzerDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //retrieve
  edm::Handle<edm::View<pat::PackedCandidate>> pfHandle;
  iEvent.getByToken(PFCandToken, pfHandle);

  if (!pfHandle.isValid()) {
    edm::LogInfo("OutputInfo") << " failed to retrieve data required by ParticleFlow Task";
    edm::LogInfo("OutputInfo") << " ParticleFlow Task cannot continue...!";
    return;
  } else {
    //Analyze
    // Loop Over Particle Flow Candidates

    for (unsigned int i = 0; i < pfHandle->size(); i++) {
      // Fill Histograms for Candidate Methods
      // all candidates
      me["AllCandidatePt(log GeV)"]->Fill(log10(pfHandle->at(i).pt()));
      me["AllCandidatePt(log y)"]->Fill(pfHandle->at(i).pt());
      me["AllCandidateEta"]->Fill(pfHandle->at(i).eta());
      me["AllCandidatePhi"]->Fill(pfHandle->at(i).phi());
      me["AllCandidateCharge"]->Fill(pfHandle->at(i).charge());
      int pdgId = abs(pfHandle->at(i).pdgId());
      if (pdgMap.find(pdgId) != pdgMap.end()) {
        me[pdgMap[pdgId] + "Pt(log GeV)"]->Fill(log10(pfHandle->at(i).pt()));
        me[pdgMap[pdgId] + "Pt(log y)"]->Fill(pfHandle->at(i).pt());
        me[pdgMap[pdgId] + "Eta"]->Fill(pfHandle->at(i).eta());
        me[pdgMap[pdgId] + "Phi"]->Fill(pfHandle->at(i).phi());
        me[pdgMap[pdgId] + "Charge"]->Fill(pfHandle->at(i).charge());
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateAnalyzerDQM);
