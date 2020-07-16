// Producer for particle flow candidates. Plots Eta, Phi, Charge, Pt (log freq, bin)
// for different types of particles described in python/defaults_cfi.py
// It actually uses packedCandidates so that we need only MINIAOD contents to run this DQMAnalyzer.
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

  me["AllCandidateLog10Pt"] = booker.book1D("AllCandidateLog10Pt", "AllCandidateLog10Pt", 120, -2, 4);

  //for phi binnings
  double nPhiBins = 73;
  double phiBinWidth = M_PI / (nPhiBins - 1) * 2.;
  me["AllCandidatePhi"] = booker.book1D(
      "AllCandidatePhi", "AllCandidatePhi", nPhiBins, -M_PI - 0.25 * phiBinWidth, +M_PI + 0.75 * phiBinWidth);

  me["AllCandidateCharge"] = booker.book1D("AllCandidateCharge", "AllCandidateCharge", 3, -1.5, 1.5);
  me["AllCandidatePtLow"] = booker.book1D("AllCandidatePtLow", "AllCandidatePtLow", 100, 0., 5.);
  me["AllCandidatePtMid"] = booker.book1D("AllCandidatePtMid", "AllCandidatePtMid", 100, 0., 200.);
  me["AllCandidatePtHigh"] = booker.book1D("AllCandidatePtHigh", "AllCandidatePtHigh", 100, 0., 1000.);

  std::string etaHistName;
  for (auto& pair : pdgMap) {
    booker.setCurrentFolder("ParticleFlow/PackedCandidates/" + pair.second);

    //TH1F only takes char*, so have to do conversions for histogram name
    etaHistName = pair.second + "Eta";
    TH1F* etaHist = new TH1F(etaHistName.c_str(), etaHistName.c_str(), n, etabinArray);
    me[pair.second + "Eta"] = booker.book1D(pair.second + "Eta", etaHist);

    me[pair.second + "Log10Pt"] = booker.book1D(pair.second + "Log10Pt", pair.second + "Log10Pt", 120, -2, 4);
    me[pair.second + "Phi"] = booker.book1D(
        pair.second + "Phi", pair.second + "Phi", nPhiBins, -M_PI - 0.25 * phiBinWidth, +M_PI + 0.75 * phiBinWidth);
    me[pair.second + "Charge"] = booker.book1D(pair.second + "Charge", pair.second + "Charge", 3, -1.5, 1.5);
    me[pair.second + "PtLow"] = booker.book1D(pair.second + "PtLow", pair.second + "PtLow", 100, 0., 5.);
    me[pair.second + "PtMid"] = booker.book1D(pair.second + "PtMid", pair.second + "PtMid", 100, 0., 200.);
    me[pair.second + "PtHigh"] = booker.book1D(pair.second + "PtHigh", pair.second + "PtHigh", 100, 0., 1000.);
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
      me["AllCandidateLog10Pt"]->Fill(log10(pfHandle->at(i).pt()));
      me["AllCandidateEta"]->Fill(pfHandle->at(i).eta());
      me["AllCandidatePhi"]->Fill(pfHandle->at(i).phi());
      me["AllCandidateCharge"]->Fill(pfHandle->at(i).charge());
      me["AllCandidatePtLow"]->Fill(pfHandle->at(i).pt());
      me["AllCandidatePtMid"]->Fill(pfHandle->at(i).pt());
      me["AllCandidatePtHigh"]->Fill(pfHandle->at(i).pt());

      int pdgId = abs(pfHandle->at(i).pdgId());
      if (pdgMap.find(pdgId) != pdgMap.end()) {
        me[pdgMap[pdgId] + "Log10Pt"]->Fill(log10(pfHandle->at(i).pt()));
        me[pdgMap[pdgId] + "Eta"]->Fill(pfHandle->at(i).eta());
        me[pdgMap[pdgId] + "Phi"]->Fill(pfHandle->at(i).phi());
        me[pdgMap[pdgId] + "Charge"]->Fill(pfHandle->at(i).charge());
        me[pdgMap[pdgId] + "PtLow"]->Fill(pfHandle->at(i).pt());
        me[pdgMap[pdgId] + "PtMid"]->Fill(pfHandle->at(i).pt());
        me[pdgMap[pdgId] + "PtHigh"]->Fill(pfHandle->at(i).pt());
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateAnalyzerDQM);
