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
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

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

class PFCandidateAnalyzerHLTDQM : public DQMEDAnalyzer {
public:
  explicit PFCandidateAnalyzerHLTDQM(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  //from config file
  edm::InputTag PFCandTag;
  edm::EDGetTokenT<reco::PFCandidateCollection> PFCandToken;
  std::vector<double> etabins;
  std::map<std::string, MonitorElement*> me;

  std::map<uint32_t, std::string> pdgMap;
};

// constructor
PFCandidateAnalyzerHLTDQM::PFCandidateAnalyzerHLTDQM(const edm::ParameterSet& iConfig) {
  PFCandTag = iConfig.getParameter<edm::InputTag>("PFCandType");
  PFCandToken = consumes<reco::PFCandidateCollection>(PFCandTag);
  etabins = iConfig.getParameter<std::vector<double>>("etabins");

  //create map of pdgId
  std::vector<uint32_t> pdgKeys = iConfig.getParameter<std::vector<uint32_t>>("pdgKeys");
  std::vector<std::string> pdgStrs = iConfig.getParameter<std::vector<std::string>>("pdgStrs");
  for (int i = 0, n = pdgKeys.size(); i < n; i++)
    pdgMap[pdgKeys[i]] = pdgStrs[i];
}

void PFCandidateAnalyzerHLTDQM::bookHistograms(DQMStore::IBooker& booker, edm::Run const&, edm::EventSetup const&) {
  // all candidate
  booker.setCurrentFolder("ParticleFlow/PFCandidate/AllCandidate");

  // for eta binning
  int n = etabins.size() - 1;
  float etabinArray[etabins.size()];
  std::copy(etabins.begin(), etabins.end(), etabinArray);

  //eta has variable bin sizes, use 4th def of TH1F constructor
  TH1F* etaHist = new TH1F("AllCandidateEta", "AllCandidateEta", n, etabinArray);
//  me["AllCandidateEta"] = booker.book1D("AllCandidateEta", etaHist);
  me["AllCandidateEta"] = booker.book1D("AllCandidateEta", "AllCandidateEta", 20, -3.0, 3.0);
  me["AllCandidateHFEta"] = booker.book1D("AllCandidateHFEta", "AllCandidateHFEta", 20, -5.0, 5.0);
  me["AllCandidateLinear10Pt"] = booker.book1D("AllCandidateLinear10Pt", "AllCandidateLinear10Pt", 120, 0.05, 50);
  me["AllCandidateHFLinear10Pt"] = booker.book1D("AllCandidateHFLinear10Pt", "AllCandidateHFLinear10Pt", 120, 0.05, 50);

  me["AllCandidateLog10Pt"] = booker.book1D("AllCandidateLog10Pt", "AllCandidateLog10Pt", 120, -2, 4);
  me["AllCandidateHFLog10Pt"] = booker.book1D("AllCandidateHFLog10Pt", "AllCandidateHFLog10Pt", 120, -2, 4);

  //for phi binnings
  double nPhiBins = 73;
  double phiBinWidth = M_PI / (nPhiBins - 1) * 2.;
  me["AllCandidatePhi"] = booker.book1D(
      "AllCandidatePhi", "AllCandidatePhi", nPhiBins, -M_PI - 0.25 * phiBinWidth, +M_PI + 0.75 * phiBinWidth);
  me["AllCandidateHFPhi"] = booker.book1D(
      "AllCandidateHFPhi", "AllCandidateHFPhi", nPhiBins, -M_PI - 0.25 * phiBinWidth, +M_PI + 0.75 * phiBinWidth);

  me["AllCandidateCharge"] = booker.book1D("AllCandidateCharge", "AllCandidateCharge", 3, -1.5, 1.5);
  me["AllCandidatePtLow"] = booker.book1D("AllCandidatePtLow", "AllCandidatePtLow", 100, 0., 5.);
  me["AllCandidatePtMid"] = booker.book1D("AllCandidatePtMid", "AllCandidatePtMid", 100, 5., 100.);
  me["AllCandidatePtHigh"] = booker.book1D("AllCandidatePtHigh", "AllCandidatePtHigh", 100, 100., 1000.);
  me["AllCandidateECALEnergyLow"] = booker.book1D("AllCandidateECALEnergyLow", "AllCandidateECALEnergy", 100, 0., 5.);
  me["AllCandidateECALEnergyMid"] = booker.book1D("AllCandidateECALEnergyMid", "AllCandidateECALEnergy", 100, 5., 100.);
  me["AllCandidateECALEnergyHigh"] = booker.book1D("AllCandidateECALEnergyHigh", "AllCandidateECALEnergy", 100, 100., 1000.);
  me["AllCandidateHCALEnergyLow"] = booker.book1D("AllCandidateHCALEnergyLow", "AllCandidateHCALEnergy", 100, 0., 5.);
  me["AllCandidateHCALEnergyMid"] = booker.book1D("AllCandidateHCALEnergyMid", "AllCandidateHCALEnergy", 100, 5., 100.);
  me["AllCandidateHCALEnergyHigh"] = booker.book1D("AllCandidateHCALEnergyHigh", "AllCandidateHCALEnergy", 100, 100., 1000.);
  me["AllCandidateHCALEnergy"] = booker.book1D("AllCandidateHCALEnergy", "AllCandidateHCALEnergy", 100, 0., 1000.);
  me["AllCandidateECALEnergyCorrLow"] = booker.book1D("AllCandidateECALEnergyCorrLow", "AllCandidateECALEnergyCorr", 100, 0., 5.);
  me["AllCandidateECALEnergyCorrMid"] = booker.book1D("AllCandidateECALEnergyCorrMid", "AllCandidateECALEnergyCorr", 100, 5., 100.);
  me["AllCandidateECALEnergyCorrHigh"] = booker.book1D("AllCandidateECALEnergyCorrHigh", "AllCandidateECALEnergyCorr", 100, 100., 1000.);
  me["AllCandidateHCALEnergyCorrLow"] = booker.book1D("AllCandidateHCALEnergyCorrLow", "AllCandidateHCALEnergyCorr", 100, 0., 5.);
  me["AllCandidateHCALEnergyCorrMid"] = booker.book1D("AllCandidateHCALEnergyCorrMid", "AllCandidateHCALEnergyCorr", 100, 5., 100.);
  me["AllCandidateHCALEnergyCorrHigh"] = booker.book1D("AllCandidateHCALEnergyCorrHigh", "AllCandidateHCALEnergyCorr", 100, 100., 1000.);
  me["AllCandidateHCALEnergyCorr"] = booker.book1D("AllCandidateHCALEnergyCorr", "AllCandidateHCALEnergyCorr", 100, 0., 1000.);
  me["AllCandidateHOEnergyCorr"] = booker.book1D("AllCandidateHOEnergyCorr", "AllCandidateHOEnergyCorr", 50, 0., 0.1);
  me["AllCandidateHOEnergy"] = booker.book1D("AllCandidateHOEnergy", "AllCandidateHOEnergy", 50, 0., 0.1);
  me["AllCandidatePSEnergy"] = booker.book1D("AllCandidatePSEnergy", "AllCandidatePSEnergy", 50, 0., 0.1);

  me["AllCandidateHFCharge"] = booker.book1D("AllCandidateHFCharge", "AllCandidateHFCharge", 3, -1.5, 1.5);
  me["AllCandidateHFPtLow"] = booker.book1D("AllCandidateHFPtLow", "AllCandidateHFPtLow", 100, 0., 5.);
  me["AllCandidateHFPtMid"] = booker.book1D("AllCandidateHFPtMid", "AllCandidateHFPtMid", 100, 5., 100.);
  me["AllCandidateHFPtHigh"] = booker.book1D("AllCandidateHFPtHigh", "AllCandidateHFPtHigh", 100, 100., 1000.);
  me["AllCandidateHFECALEnergyLow"] = booker.book1D("AllCandidateHFECALEnergyLow", "AllCandidateHFECALEnergy", 100, 0., 5.);
  me["AllCandidateHFECALEnergyMid"] = booker.book1D("AllCandidateHFECALEnergyMid", "AllCandidateHFECALEnergy", 100, 5., 100.);
  me["AllCandidateHFECALEnergyHigh"] = booker.book1D("AllCandidateHFECALEnergyHigh", "AllCandidateHFECALEnergy", 100, 100., 1000.);
  me["AllCandidateHFHCALEnergyLow"] = booker.book1D("AllCandidateHFHCALEnergyLow", "AllCandidateHFHCALEnergy", 100, 0., 5.);
  me["AllCandidateHFHCALEnergyMid"] = booker.book1D("AllCandidateHFHCALEnergyMid", "AllCandidateHFHCALEnergy", 100, 5., 100.);
  me["AllCandidateHFHCALEnergyHigh"] = booker.book1D("AllCandidateHFHCALEnergyHigh", "AllCandidateHFHCALEnergy", 100, 100., 1000.);
  me["AllCandidateHFHCALEnergy"] = booker.book1D("AllCandidateHFHCALEnergy", "AllCandidateHFHCALEnergy", 100, 0., 1000.);
  me["AllCandidateHFHOEnergy"] = booker.book1D("AllCandidateHFHOEnergy", "AllCandidateHFHOEnergy", 50, 0., 0.1);
  me["AllCandidateHFECALEnergyCorrLow"] = booker.book1D("AllCandidateHFECALEnergyCorrLow", "AllCandidateHFECALEnergyCorr", 100, 0., 5.);
  me["AllCandidateHFECALEnergyCorrMid"] = booker.book1D("AllCandidateHFECALEnergyCorrMid", "AllCandidateHFECALEnergyCorr", 100, 5., 100.);
  me["AllCandidateHFECALEnergyCorrHigh"] = booker.book1D("AllCandidateHFECALEnergyCorrHigh", "AllCandidateHFECALEnergyCorr", 100, 100., 1000.);
  me["AllCandidateHFHCALEnergyCorrLow"] = booker.book1D("AllCandidateHFHCALEnergyCorrLow", "AllCandidateHFHCALEnergyCorr", 100, 0., 5.);
  me["AllCandidateHFHCALEnergyCorrMid"] = booker.book1D("AllCandidateHFHCALEnergyCorrMid", "AllCandidateHFHCALEnergyCorr", 100, 5., 100.);
  me["AllCandidateHFHCALEnergyCorrHigh"] = booker.book1D("AllCandidateHFHCALEnergyCorrHigh", "AllCandidateHFHCALEnergyCorr", 100, 100., 1000.);
  me["AllCandidateHFHCALEnergyCorr"] = booker.book1D("AllCandidateHFHCALEnergyCorr", "AllCandidateHFHCALEnergyCorr", 100, 0., 1000.);
  me["AllCandidateHFHOEnergyCorr"] = booker.book1D("AllCandidateHFHOEnergyCorr", "AllCandidateHFHOEnergyCorr", 50, 0., 0.1);
  me["AllCandidateHFPSEnergy"] = booker.book1D("AllCandidateHFPSEnergy", "AllCandidateHFPSEnergy", 50, 0., 0.1);
  booker.setCurrentFolder("ParticleFlow/PFCandidate/Undefined");

  me["UndefinedPhi"] = booker.book1D(
      "UndefinedPhi", "UndefinedPhi", nPhiBins, -M_PI - 0.25 * phiBinWidth, +M_PI + 0.75 * phiBinWidth);

  me["UndefinedCharge"] = booker.book1D("UndefinedCharge", "UndefinedCharge", 3, -1.5, 1.5);
  me["UndefinedPtLow"] = booker.book1D("UndefinedPtLow", "UndefinedPtLow", 100, 0., 5.);
  me["UndefinedPtMid"] = booker.book1D("UndefinedPtMid", "UndefinedPtMid", 100, 4., 100.);
  me["UndefinedPtHigh"] = booker.book1D("UndefinedPtHigh", "UndefinedPtHigh", 100, 100., 1000.);
  me["UndefinedECALEnergyLow"] = booker.book1D("UndefinedECALEnergyLow", "UndefinedECALEnergy", 100, 0., 5.);
  me["UndefinedECALEnergyMid"] = booker.book1D("UndefinedECALEnergyMid", "UndefinedECALEnergy", 100, 5., 100.);
  me["UndefinedECALEnergyHigh"] = booker.book1D("UndefinedECALEnergyHigh", "UndefinedECALEnergy", 100, 100., 1000.);
  me["UndefinedHCALEnergyLow"] = booker.book1D("UndefinedHCALEnergyLow", "UndefinedHCALEnergy", 100, 0., 5.);
  me["UndefinedHCALEnergyMid"] = booker.book1D("UndefinedHCALEnergyMid", "UndefinedHCALEnergy", 100, 5., 100.);
  me["UndefinedHCALEnergyHigh"] = booker.book1D("UndefinedHCALEnergyHigh", "UndefinedHCALEnergy", 100, 100., 1000.);
  me["UndefinedHCALEnergy"] = booker.book1D("UndefinedHCALEnergy", "UndefinedHCALEnergy", 100, 0., 1000.);
  me["UndefinedHOEnergy"] = booker.book1D("UndefinedHOEnergy", "UndefinedHOEnergy", 50, 0., 0.1);
  me["UndefinedECALEnergyCorrLow"] = booker.book1D("UndefinedECALEnergyCorrLow", "UndefinedECALEnergyCorr", 100, 0., 5.);
  me["UndefinedECALEnergyCorrMid"] = booker.book1D("UndefinedECALEnergyCorrMid", "UndefinedECALEnergyCorr", 100, 5., 100.);
  me["UndefinedECALEnergyCorrHigh"] = booker.book1D("UndefinedECALEnergyCorrHigh", "UndefinedECALEnergyCorr", 100, 100., 1000.);
  me["UndefinedHCALEnergyCorrLow"] = booker.book1D("UndefinedHCALEnergyCorrLow", "UndefinedHCALEnergyCorr", 100, 0., 5.);
  me["UndefinedHCALEnergyCorrMid"] = booker.book1D("UndefinedHCALEnergyCorrMid", "UndefinedHCALEnergyCorr", 100, 5., 100.);
  me["UndefinedHCALEnergyCorrHigh"] = booker.book1D("UndefinedHCALEnergyCorrHigh", "UndefinedHCALEnergyCorr", 100, 100., 1000.);
  me["UndefinedHCALEnergyCorr"] = booker.book1D("UndefinedHCALEnergyCorr", "UndefinedHCALEnergyCorr", 100, 0., 1000.);
  me["UndefinedHOEnergyCorr"] = booker.book1D("UndefinedHOEnergyCorr", "UndefinedHOEnergyCorr", 50, 0., 0.1);
  me["UndefinedPSEnergy"] = booker.book1D("UndefinedPSEnergy", "UndefinedPSEnergy", 50, 0., 0.1);

  std::string etaHistName;
  for (auto& pair : pdgMap) {
    booker.setCurrentFolder("ParticleFlow/PFCandidate/" + pair.second);
    if(pair.second == "muon"){
      me["Muon20GeVEta"] = booker.book1D("Muon20GeVEta", "Muon20GeVEta",  20, -3.0, 3.0);
      me["Muon20GeVLinear10Pt"] = booker.book1D("Muon20GeVLinear10Pt", "Muon20GeVLinear10Pt", 120, 0.05, 50);
      me["Muon20GeVLog10Pt"] = booker.book1D("Muon20GeVLog10Pt", "Muon20GeVLog10Pt", 120, -2, 4);
      me["Muon20GeVPhi"] = booker.book1D(
          "Muon20GeVPhi", "Muon20GeVPhi", 30, -M_PI, +M_PI);
      me["Muon20GeVCharge"] = booker.book1D("Muon20GeVCharge", "Muon20GeVCharge", 3, -1.5, 1.5);
      me["Muon20GeVPtLow"] = booker.book1D("Muon20GeVPtLow", "Muon20GeVPtLow", 100, 0., 5.);
      me["Muon20GeVPtMid"] = booker.book1D("Muon20GeVPtMid", "Muon20GeVPtMid", 100, 20., 200.);
      me["Muon20GeVPtHigh"] = booker.book1D("Muon20GeVPtHigh", "Muon20GeVPtHigh", 100, 20., 1000.);
      me["Muon20GeVECALEnergyLow"] = booker.book1D("Muon20GeVECALEnergyLow", "Muon20GeVECALEnergyLow", 100, 0., 5.);
      me["Muon20GeVECALEnergyMid"] = booker.book1D("Muon20GeVECALEnergyMid", "Muon20GeVECALEnergyMid", 50, 5., 100.);
      me["Muon20GeVECALEnergyHigh"] = booker.book1D("Muon20GeVECALEnergyHigh", "Muon20GeVECALEnergyHigh", 30, 100., 1000.);
      me["Muon20GeVHCALEnergyLow"] = booker.book1D("Muon20GeVHCALEnergyLow", "Muon20GeVHCALEnergyLow", 100, 0., 5.);
      me["Muon20GeVHCALEnergyMid"] = booker.book1D("Muon20GeVHCALEnergyMid", "Muon20GeVHCALEnergyMid", 50, 5., 100.);
      me["Muon20GeVHCALEnergyHigh"] = booker.book1D("Muon20GeVHCALEnergyHigh", "Muon20GeVHCALEnergyHigh", 30, 100., 1000.);
      me["Muon20GeVHOEnergy"] = booker.book1D("Muon20GeVHOEnergy", "Muon20GeVHOEnergy", 50, 0., 0.1);
      me["Muon20GeVECALEnergyCorrLow"] = booker.book1D("Muon20GeVECALEnergyCorrLow", "Muon20GeVECALEnergyCorrLow", 100, 0., 5.);
      me["Muon20GeVECALEnergyCorrMid"] = booker.book1D("Muon20GeVECALEnergyCorrMid", "Muon20GeVECALEnergyCorrMid", 50, 5., 100.);
      me["Muon20GeVECALEnergyCorrHigh"] = booker.book1D("Muon20GeVECALEnergyCorrHigh", "Muon20GeVECALEnergyCorrHigh", 30, 100., 1000.);
      me["Muon20GeVHCALEnergyCorrLow"] = booker.book1D("Muon20GeVHCALEnergyCorrLow", "Muon20GeVHCALEnergyCorrLow", 100, 0., 5.);
      me["Muon20GeVHCALEnergyCorrMid"] = booker.book1D("Muon20GeVHCALEnergyCorrMid", "Muon20GeVHCALEnergyCorrMid", 50, 5., 100.);
      me["Muon20GeVHCALEnergyCorrHigh"] = booker.book1D("Muon20GeVHCALEnergyCorrHigh", "Muon20GeVHCALEnergyCorrHigh", 30, 100., 1000.);
      me["Muon20GeVHOEnergyCorr"] = booker.book1D("Muon20GeVHOEnergyCorr", "Muon20GeVHOEnergyCorr", 50, 0., 0.1);
      me["Muon20GeVPSEnergy"] = booker.book1D("Muon20GeVPSEnergy", "Muon20GeVPSEnergy", 50, 0., 0.1);
      me["muonEta"] = booker.book1D("muonEta", "muonEta",  20, -3.0, 3.0);
      me["muonLinear10Pt"] = booker.book1D("muonLinear10Pt", "muonLinear10Pt", 60, 0.05, 50);
      me["muonLog10Pt"] = booker.book1D("muonLog10Pt", "muonLog10Pt", 80, -2, 4);
      me["muonPhi"] = booker.book1D(
          "muonPhi", "muonPhi", 30, -M_PI, +M_PI);
      me["muonCharge"] = booker.book1D("muonCharge", "muonCharge", 3, -1.5, 1.5);
      me["muonPtLow"] = booker.book1D("muonPtLow", "muonPtLow", 20, 0., 5.);
      me["muonPtMid"] = booker.book1D("muonPtMid", "muonPtMid", 80, 20., 200.);
      me["muonPtHigh"] = booker.book1D("muonPtHigh", "muonPtHigh", 100, 20., 1000.);
      me["muonECALEnergyLow"] = booker.book1D("muonECALEnergyLow", "muonECALEnergyLow", 100, 0., 5.);
      me["muonECALEnergyMid"] = booker.book1D("muonECALEnergyMid", "muonECALEnergyMid", 50, 5., 100.);
      me["muonECALEnergyHigh"] = booker.book1D("muonECALEnergyHigh", "muonECALEnergyHigh", 30, 100., 1000.);
      me["muonHCALEnergyLow"] = booker.book1D("muonHCALEnergyLow", "muonHCALEnergyLow", 50, 0., 5.);
      me["muonHCALEnergyMid"] = booker.book1D("muonHCALEnergyMid", "muonHCALEnergyMid", 50, 5., 100.);
      me["muonHCALEnergyHigh"] = booker.book1D("muonHCALEnergyHigh", "muonHCALEnergyHigh", 30, 100., 1000.);
      me["muonHOEnergy"] = booker.book1D("muonHOEnergy", "muonHOEnergy", 50, 0., 0.1);
      me["muonECALEnergyCorrLow"] = booker.book1D("muonECALEnergyCorrLow", "muonECALEnergyCorrLow", 100, 0., 5.);
      me["muonECALEnergyCorrMid"] = booker.book1D("muonECALEnergyCorrMid", "muonECALEnergyCorrMid", 50, 5., 100.);
      me["muonECALEnergyCorrHigh"] = booker.book1D("muonECALEnergyCorrHigh", "muonECALEnergyCorrHigh", 30, 100., 1000.);
      me["muonHCALEnergyCorrLow"] = booker.book1D("muonHCALEnergyCorrLow", "muonHCALEnergyCorrLow", 50, 0., 5.);
      me["muonHCALEnergyCorrMid"] = booker.book1D("muonHCALEnergyCorrMid", "muonHCALEnergyCorrMid", 50, 5., 100.);
      me["muonHCALEnergyCorrHigh"] = booker.book1D("muonHCALEnergyCorrHigh", "muonHCALEnergyCorrHigh", 30, 100., 1000.);
      me["muonHOEnergyCorr"] = booker.book1D("muonHOEnergyCorr", "muonHOEnergyCorr", 50, 0., 0.1);
      me["muonPSEnergy"] = booker.book1D("muonPSEnergy", "muonPSEnergy", 50, 0., 0.1);

    }
    else{
      //TH1F only takes char*, so have to do conversions for histogram name
      etaHistName = pair.second + "Eta";
      TH1F* etaHist = new TH1F(etaHistName.c_str(), etaHistName.c_str(), n, etabinArray);
//      me[pair.second + "Eta"] = booker.book1D(pair.second + "Eta", etaHist);
      if(pair.second == "chargedHadron")
        me[pair.second + "Eta"] = booker.book1D(pair.second + "Eta", pair.second + "Eta",  50, -3.0, 3.0);
      me[pair.second + "Eta"] = booker.book1D(pair.second + "Eta", pair.second + "Eta",  20, -3.0, 3.0);
      me[pair.second + "Linear10Pt"] = booker.book1D(pair.second + "Linear10Pt", pair.second + "Linear10Pt", 120, 0.05, 50);
      me[pair.second + "Log10Pt"] = booker.book1D(pair.second + "Log10Pt", pair.second + "Log10Pt", 120, -2, 4);
      if(pair.second == "photon"){
      me[pair.second + "Phi"] = booker.book1D(
          pair.second + "Phi", pair.second + "Phi", 50, -M_PI, +M_PI);
      }
      else{
      me[pair.second + "Phi"] = booker.book1D(
          pair.second + "Phi", pair.second + "Phi", nPhiBins, -M_PI - 0.25 * phiBinWidth, +M_PI + 0.75 * phiBinWidth);
      }
      me[pair.second + "Charge"] = booker.book1D(pair.second + "Charge", pair.second + "Charge", 3, -1.5, 1.5);
      me[pair.second + "PtLow"] = booker.book1D(pair.second + "PtLow", pair.second + "PtLow", 100, 0., 5.);
      me[pair.second + "PtMid"] = booker.book1D(pair.second + "PtMid", pair.second + "PtMid", 100, 5., 100.);
      me[pair.second + "PtHigh"] = booker.book1D(pair.second + "PtHigh", pair.second + "PtHigh", 100, 100., 1000.);
      me[pair.second + "ECALEnergyLow"] = booker.book1D(pair.second + "ECALEnergyLow", pair.second + "ECALEnergyLow", 100, 0., 5.);
      me[pair.second + "ECALEnergyMid"] = booker.book1D(pair.second + "ECALEnergyMid", pair.second + "ECALEnergyMid", 50, 5., 100.);
      me[pair.second + "ECALEnergyHigh"] = booker.book1D(pair.second + "ECALEnergyHigh", pair.second + "ECALEnergyHigh", 30, 100., 1000.);
      me[pair.second + "HCALEnergyLow"] = booker.book1D(pair.second + "HCALEnergyLow", pair.second + "HCALEnergyLow", 100, 0., 5.);
      me[pair.second + "HCALEnergyMid"] = booker.book1D(pair.second + "HCALEnergyMid", pair.second + "HCALEnergyMid", 50, 5., 100.);
      me[pair.second + "HCALEnergyHigh"] = booker.book1D(pair.second + "HCALEnergyHigh", pair.second + "HCALEnergyHigh", 30, 100., 1000.);
      me[pair.second + "HOEnergy"] = booker.book1D(pair.second + "HOEnergy", pair.second + "HOEnergy", 50, 0., 0.1);
      me[pair.second + "ECALEnergyCorrLow"] = booker.book1D(pair.second + "ECALEnergyCorrLow", pair.second + "ECALEnergyCorrLow", 100, 0., 5.);
      me[pair.second + "ECALEnergyCorrMid"] = booker.book1D(pair.second + "ECALEnergyCorrMid", pair.second + "ECALEnergyCorrMid", 50, 5., 100.);
      me[pair.second + "ECALEnergyCorrHigh"] = booker.book1D(pair.second + "ECALEnergyCorrHigh", pair.second + "ECALEnergyCorrHigh", 30, 100., 1000.);
      me[pair.second + "HCALEnergyCorrLow"] = booker.book1D(pair.second + "HCALEnergyCorrLow", pair.second + "HCALEnergyCorrLow", 100, 0., 5.);
      me[pair.second + "HCALEnergyCorrMid"] = booker.book1D(pair.second + "HCALEnergyCorrMid", pair.second + "HCALEnergyCorrMid", 50, 5., 100.);
      me[pair.second + "HCALEnergyCorrHigh"] = booker.book1D(pair.second + "HCALEnergyCorrHigh", pair.second + "HCALEnergyCorrHigh", 30, 100., 1000.);
      me[pair.second + "HOEnergyCorr"] = booker.book1D(pair.second + "HOEnergyCorr", pair.second + "HOEnergyCorr", 50, 0., 0.1);
      me[pair.second + "PSEnergy"] = booker.book1D(pair.second + "PSEnergy", pair.second + "PSEnergy", 50, 0., 0.1);
    }
  }
}

void PFCandidateAnalyzerHLTDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //retrieve
  edm::Handle<reco::PFCandidateCollection> pfHandle;
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
      if(abs(pfHandle->at(i).pdgId()) != 1 and abs(pfHandle->at(i).pdgId()) != 2){
        me["AllCandidateLinear10Pt"]->Fill(pfHandle->at(i).pt());
        me["AllCandidateLog10Pt"]->Fill(log10(pfHandle->at(i).pt()));
        me["AllCandidateEta"]->Fill(pfHandle->at(i).eta());
        me["AllCandidatePhi"]->Fill(pfHandle->at(i).phi());
        me["AllCandidateCharge"]->Fill(pfHandle->at(i).charge());
        me["AllCandidatePtLow"]->Fill(pfHandle->at(i).pt());
        me["AllCandidatePtMid"]->Fill(pfHandle->at(i).pt());
        me["AllCandidatePtHigh"]->Fill(pfHandle->at(i).pt());
        if(pfHandle->at(i).rawEcalEnergy() > 0.){
          me["AllCandidateECALEnergyLow"]->Fill(pfHandle->at(i).rawEcalEnergy());
          me["AllCandidateECALEnergyMid"]->Fill(pfHandle->at(i).rawEcalEnergy());
          me["AllCandidateECALEnergyHigh"]->Fill(pfHandle->at(i).rawEcalEnergy());
        }
        if(pfHandle->at(i).rawHcalEnergy() > 0.){
          me["AllCandidateHCALEnergyLow"]->Fill(pfHandle->at(i).rawHcalEnergy());
          me["AllCandidateHCALEnergyMid"]->Fill(pfHandle->at(i).rawHcalEnergy());
          me["AllCandidateHCALEnergyHigh"]->Fill(pfHandle->at(i).rawHcalEnergy());
        }
        me["AllCandidateHOEnergy"]->Fill(pfHandle->at(i).rawHoEnergy()); 
        if(pfHandle->at(i).ecalEnergy() > 0.){
          me["AllCandidateECALEnergyCorrLow"]->Fill(pfHandle->at(i).ecalEnergy());
          me["AllCandidateECALEnergyCorrMid"]->Fill(pfHandle->at(i).ecalEnergy());
          me["AllCandidateECALEnergyCorrHigh"]->Fill(pfHandle->at(i).ecalEnergy());
        }
        if(pfHandle->at(i).hcalEnergy() > 0.){
          me["AllCandidateHCALEnergyCorrLow"]->Fill(pfHandle->at(i).hcalEnergy());
          me["AllCandidateHCALEnergyCorrMid"]->Fill(pfHandle->at(i).hcalEnergy());
          me["AllCandidateHCALEnergyCorrHigh"]->Fill(pfHandle->at(i).hcalEnergy());
        }
        me["AllCandidateHOEnergyCorr"]->Fill(pfHandle->at(i).hoEnergy()); 
        me["AllCandidatePSEnergy"]->Fill(pfHandle->at(i).pS1Energy() + pfHandle->at(i).pS2Energy());
      }
      else{
        me["AllCandidateHFLinear10Pt"]->Fill(pfHandle->at(i).pt());
        me["AllCandidateHFLog10Pt"]->Fill(log10(pfHandle->at(i).pt()));
        me["AllCandidateHFEta"]->Fill(pfHandle->at(i).eta());
        me["AllCandidateHFPhi"]->Fill(pfHandle->at(i).phi());
        me["AllCandidateHFCharge"]->Fill(pfHandle->at(i).charge());
        me["AllCandidateHFPtLow"]->Fill(pfHandle->at(i).pt());
        me["AllCandidateHFPtMid"]->Fill(pfHandle->at(i).pt());
        me["AllCandidateHFPtHigh"]->Fill(pfHandle->at(i).pt());
        if(pfHandle->at(i).rawEcalEnergy() > 0.){
          me["AllCandidateHFECALEnergyLow"]->Fill(pfHandle->at(i).rawEcalEnergy());
          me["AllCandidateHFECALEnergyMid"]->Fill(pfHandle->at(i).rawEcalEnergy());
          me["AllCandidateHFECALEnergyHigh"]->Fill(pfHandle->at(i).rawEcalEnergy());
        }
        if(pfHandle->at(i).rawHcalEnergy() > 0.){
          me["AllCandidateHFHCALEnergyLow"]->Fill(pfHandle->at(i).rawHcalEnergy());
          me["AllCandidateHFHCALEnergyMid"]->Fill(pfHandle->at(i).rawHcalEnergy());
          me["AllCandidateHFHCALEnergyHigh"]->Fill(pfHandle->at(i).rawHcalEnergy());
        }
        me["AllCandidateHFHOEnergy"]->Fill(pfHandle->at(i).rawHoEnergy()); 
        if(pfHandle->at(i).ecalEnergy() > 0.){
          me["AllCandidateHFECALEnergyCorrLow"]->Fill(pfHandle->at(i).ecalEnergy());
          me["AllCandidateHFECALEnergyCorrMid"]->Fill(pfHandle->at(i).ecalEnergy());
          me["AllCandidateHFECALEnergyCorrHigh"]->Fill(pfHandle->at(i).ecalEnergy());
        }
        if(pfHandle->at(i).hcalEnergy() > 0.){
          me["AllCandidateHFHCALEnergyCorrLow"]->Fill(pfHandle->at(i).hcalEnergy());
          me["AllCandidateHFHCALEnergyCorrMid"]->Fill(pfHandle->at(i).hcalEnergy());
          me["AllCandidateHFHCALEnergyCorrHigh"]->Fill(pfHandle->at(i).hcalEnergy());
        }
        me["AllCandidateHFHOEnergyCorr"]->Fill(pfHandle->at(i).hoEnergy()); 
        me["AllCandidateHFPSEnergy"]->Fill(pfHandle->at(i).pS1Energy() + pfHandle->at(i).pS2Energy());

      }

      int pdgId = abs(pfHandle->at(i).pdgId());
      if (pdgMap.find(pdgId) != pdgMap.end()) {
        me[pdgMap[pdgId] + "Linear10Pt"]->Fill(pfHandle->at(i).pt());
        me[pdgMap[pdgId] + "Log10Pt"]->Fill(log10(pfHandle->at(i).pt()));
        me[pdgMap[pdgId] + "Eta"]->Fill(pfHandle->at(i).eta());
        me[pdgMap[pdgId] + "Phi"]->Fill(pfHandle->at(i).phi());
        me[pdgMap[pdgId] + "Charge"]->Fill(pfHandle->at(i).charge());
        me[pdgMap[pdgId] + "PtLow"]->Fill(pfHandle->at(i).pt());
        me[pdgMap[pdgId] + "PtMid"]->Fill(pfHandle->at(i).pt());
        me[pdgMap[pdgId] + "PtHigh"]->Fill(pfHandle->at(i).pt());
        if(pfHandle->at(i).rawEcalEnergy() > 0.){
          me[pdgMap[pdgId] + "ECALEnergyLow"]->Fill(pfHandle->at(i).rawEcalEnergy());
          me[pdgMap[pdgId] + "ECALEnergyMid"]->Fill(pfHandle->at(i).rawEcalEnergy());
          me[pdgMap[pdgId] + "ECALEnergyHigh"]->Fill(pfHandle->at(i).rawEcalEnergy());
        }
        if(pfHandle->at(i).rawHcalEnergy() > 0.){
          me[pdgMap[pdgId] + "HCALEnergyLow"]->Fill(pfHandle->at(i).rawHcalEnergy());
          me[pdgMap[pdgId] + "HCALEnergyMid"]->Fill(pfHandle->at(i).rawHcalEnergy());
          me[pdgMap[pdgId] + "HCALEnergyHigh"]->Fill(pfHandle->at(i).rawHcalEnergy());
        }
        me[pdgMap[pdgId] + "HOEnergy"]->Fill(pfHandle->at(i).rawHoEnergy()); 
        if(pfHandle->at(i).ecalEnergy() > 0.){
          me[pdgMap[pdgId] + "ECALEnergyCorrLow"]->Fill(pfHandle->at(i).ecalEnergy());
          me[pdgMap[pdgId] + "ECALEnergyCorrMid"]->Fill(pfHandle->at(i).ecalEnergy());
          me[pdgMap[pdgId] + "ECALEnergyCorrHigh"]->Fill(pfHandle->at(i).ecalEnergy());
        }
        if(pfHandle->at(i).hcalEnergy() > 0.){
          me[pdgMap[pdgId] + "HCALEnergyCorrLow"]->Fill(pfHandle->at(i).hcalEnergy());
          me[pdgMap[pdgId] + "HCALEnergyCorrMid"]->Fill(pfHandle->at(i).hcalEnergy());
          me[pdgMap[pdgId] + "HCALEnergyCorrHigh"]->Fill(pfHandle->at(i).hcalEnergy());
        }
        me[pdgMap[pdgId] + "HOEnergyCorr"]->Fill(pfHandle->at(i).hoEnergy()); 
        me[pdgMap[pdgId] + "PSEnergy"]->Fill(pfHandle->at(i).pS1Energy() + pfHandle->at(i).pS2Energy());
        if(pdgId == 13 and pfHandle->at(i).pt() >= 20.f){
          me["Muon20GeVLinear10Pt"]->Fill(pfHandle->at(i).pt());
          me["Muon20GeVLog10Pt"]->Fill(log10(pfHandle->at(i).pt()));
          me["Muon20GeVEta"]->Fill(pfHandle->at(i).eta());
          me["Muon20GeVPhi"]->Fill(pfHandle->at(i).phi());
          me["Muon20GeVCharge"]->Fill(pfHandle->at(i).charge());
          me["Muon20GeVPtLow"]->Fill(pfHandle->at(i).pt());
          me["Muon20GeVPtMid"]->Fill(pfHandle->at(i).pt());
          me["Muon20GeVPtHigh"]->Fill(pfHandle->at(i).pt());
          me["Muon20GeVECALEnergyLow"]->Fill(pfHandle->at(i).rawEcalEnergy());
          me["Muon20GeVECALEnergyMid"]->Fill(pfHandle->at(i).rawEcalEnergy());
          me["Muon20GeVECALEnergyHigh"]->Fill(pfHandle->at(i).rawEcalEnergy());
          me["Muon20GeVHCALEnergyLow"]->Fill(pfHandle->at(i).rawHcalEnergy());
          me["Muon20GeVHCALEnergyMid"]->Fill(pfHandle->at(i).rawHcalEnergy());
          me["Muon20GeVHCALEnergyHigh"]->Fill(pfHandle->at(i).rawHcalEnergy());
          me["Muon20GeVHOEnergy"]->Fill(pfHandle->at(i).rawHoEnergy()); 
          me["Muon20GeVECALEnergyCorrLow"]->Fill(pfHandle->at(i).ecalEnergy());
          me["Muon20GeVECALEnergyCorrMid"]->Fill(pfHandle->at(i).ecalEnergy());
          me["Muon20GeVECALEnergyCorrHigh"]->Fill(pfHandle->at(i).ecalEnergy());
          me["Muon20GeVHCALEnergyCorrLow"]->Fill(pfHandle->at(i).hcalEnergy());
          me["Muon20GeVHCALEnergyCorrMid"]->Fill(pfHandle->at(i).hcalEnergy());
          me["Muon20GeVHCALEnergyCorrHigh"]->Fill(pfHandle->at(i).hcalEnergy());
          me["Muon20GeVHOEnergyCorr"]->Fill(pfHandle->at(i).hoEnergy()); 
          me["Muon20GeVPSEnergy"]->Fill(pfHandle->at(i).pS1Energy() + pfHandle->at(i).pS2Energy());
        }
      }
      else{
        me["UndefinedLog10Pt"]->Fill(log10(pfHandle->at(i).pt()));
        me["UndefinedEta"]->Fill(pfHandle->at(i).eta());
        me["UndefinedPhi"]->Fill(pfHandle->at(i).phi());
        me["UndefinedCharge"]->Fill(pfHandle->at(i).charge());
        me["UndefinedPtLow"]->Fill(pfHandle->at(i).pt());
        me["UndefinedPtMid"]->Fill(pfHandle->at(i).pt());
        me["UndefinedPtHigh"]->Fill(pfHandle->at(i).pt());
        me["UndefinedECALEnergyLow"]->Fill(pfHandle->at(i).rawEcalEnergy());
        me["UndefinedECALEnergyMid"]->Fill(pfHandle->at(i).rawEcalEnergy());
        me["UndefinedECALEnergyHigh"]->Fill(pfHandle->at(i).rawEcalEnergy());
        me["UndefinedHCALEnergyLow"]->Fill(pfHandle->at(i).rawHcalEnergy());
        me["UndefinedHCALEnergyMid"]->Fill(pfHandle->at(i).rawHcalEnergy());
        me["UndefinedHCALEnergyHigh"]->Fill(pfHandle->at(i).rawHcalEnergy());
        me["UndefinedHOEnergy"]->Fill(pfHandle->at(i).rawHoEnergy()); 
        me["UndefinedECALEnergyCorrLow"]->Fill(pfHandle->at(i).ecalEnergy());
        me["UndefinedECALEnergyCorrMid"]->Fill(pfHandle->at(i).ecalEnergy());
        me["UndefinedECALEnergyCorrHigh"]->Fill(pfHandle->at(i).ecalEnergy());
        me["UndefinedHCALEnergyCorrLow"]->Fill(pfHandle->at(i).hcalEnergy());
        me["UndefinedHCALEnergyCorrMid"]->Fill(pfHandle->at(i).hcalEnergy());
        me["UndefinedHCALEnergyCorrHigh"]->Fill(pfHandle->at(i).hcalEnergy());
        me["UndefinedHOEnergyCorr"]->Fill(pfHandle->at(i).hoEnergy()); 
        me["UndefinedPSEnergy"]->Fill(pfHandle->at(i).pS1Energy() + pfHandle->at(i).pS2Energy());
      }
    }
  }
}
#
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateAnalyzerHLTDQM);
