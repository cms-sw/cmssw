// -*- C++ -*-
//
// Package:    TauTagValidation
// Class:      TauTagValidation
//
/**\class TauTagValidation TauTagValidation.cc

 Description: <one line class summary>

 Class used to do the Validation of the TauTag

 Implementation:
 <Notes on implementation>
 */
//
// Original Author:  Ricardo Vasquez Sierra
//         Created:  October 8, 2008
//
//Modified by: Atanu Modak to include extra plots
// user include files

#include "Validation/RecoTau/interface/TauTagValidation.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include <DataFormats/VertexReco/interface/Vertex.h>
#include <DataFormats/VertexReco/interface/VertexFwd.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "RecoTauTag/RecoTau/interface/PFTauDecayModeTools.h"

using namespace edm;
using namespace std;
using namespace reco;

TauTagValidation::TauTagValidation(const edm::ParameterSet& iConfig)
    : moduleLabel_(iConfig.getParameter<std::string>("@module_label")),
      // What do we want to use as source Leptons or Jets (the only difference is the matching criteria)
      dataType_(iConfig.getParameter<string>("DataType")),
      // We need different matching criteria if we talk about leptons or jets
      matchDeltaR_Leptons_(iConfig.getParameter<double>("MatchDeltaR_Leptons")),
      matchDeltaR_Jets_(iConfig.getParameter<double>("MatchDeltaR_Jets")),
      TauPtCut_(iConfig.getParameter<double>("TauPtCut")),
      //flexible cut interface to filter reco and gen collection. use an empty string to select all.
      recoCuts_(iConfig.getParameter<std::string>("recoCuts")),
      genCuts_(iConfig.getParameter<std::string>("genCuts")),
      // The output histograms can be stored or not
      saveoutputhistograms_(iConfig.getParameter<bool>("SaveOutputHistograms")),
      // Here it can be pretty much anything either a lepton or a jet
      refCollectionInputTag_(iConfig.getParameter<InputTag>("RefCollection")),
      // The extension name has information about the Reference collection used
      extensionName_(iConfig.getParameter<string>("ExtensionName")),
      // Here is the reconstructed product of interest.
      TauProducerInputTag_(iConfig.getParameter<InputTag>("TauProducer")),
      // Get the discriminators and their cuts
      discriminators_(iConfig.getParameter<std::vector<edm::ParameterSet> >("discriminators")) {
  turnOnTrigger_ = iConfig.exists("turnOnTrigger") && iConfig.getParameter<bool>("turnOnTrigger");
  genericTriggerEventFlag_ =
      (iConfig.exists("GenericTriggerSelection") && turnOnTrigger_)
          ? new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("GenericTriggerSelection"),
                                        consumesCollector(),
                                        *this,
                                        l1t::UseEventSetupIn::Run)
          : nullptr;
  if (genericTriggerEventFlag_ != nullptr)
    LogDebug(moduleLabel_) << "--> GenericTriggerSelection parameters found in " << moduleLabel_ << "."
                           << std::endl;  //move to LogDebug
  else
    LogDebug(moduleLabel_) << "--> GenericTriggerSelection not found in " << moduleLabel_ << "."
                           << std::endl;  //move to LogDebug to keep track of modules that fail and pass

  //InputTag to strings
  refCollection_ = refCollectionInputTag_.label();
  TauProducer_ = TauProducerInputTag_.label();

  histoSettings_ = (iConfig.exists("histoSettings")) ? iConfig.getParameter<edm::ParameterSet>("histoSettings")
                                                     : edm::ParameterSet();
  edm::InputTag PrimaryVertexCollection_ = (iConfig.exists("PrimaryVertexCollection"))
                                               ? iConfig.getParameter<InputTag>("PrimaryVertexCollection")
                                               : edm::InputTag("offlinePrimaryVertices");  //TO-DO

  refCollectionInputTagToken_ = consumes<edm::View<reco::Candidate> >(iConfig.getParameter<InputTag>("RefCollection"));
  primaryVertexCollectionToken_ = consumes<VertexCollection>(PrimaryVertexCollection_);  //TO-DO
  tauProducerInputTagToken_ = consumes<reco::PFTauCollection>(iConfig.getParameter<InputTag>("TauProducer"));
  for (const auto& it : discriminators_) {
    currentDiscriminatorToken_.push_back(
        consumes<reco::PFTauDiscriminator>(edm::InputTag(it.getParameter<string>("discriminator"))));
  }

  tversion = edm::getReleaseVersion();

  if (!saveoutputhistograms_) {
    LogInfo("OutputInfo") << " TauVisible histograms will NOT be saved";
  } else {
    outPutFile_ = TauProducer_;
    outPutFile_.append("_");
    tversion.erase(0, 1);
    tversion.erase(tversion.size() - 1, 1);
    outPutFile_.append(tversion);
    outPutFile_.append("_" + refCollection_);
    if (!extensionName_.empty()) {
      outPutFile_.append("_" + extensionName_);
    }
    outPutFile_.append(".root");

    LogInfo("OutputInfo") << " TauVisiblehistograms will be saved to file:" << outPutFile_;
  }

  //---- book-keeping information ---
  numEvents_ = 0;

  // Check if we want to "chain" the discriminator requirements (i.e. all
  // prveious discriminators must pass)
  chainCuts_ = iConfig.exists("chainCuts") ? iConfig.getParameter<bool>("chainCuts") : true;
}

TauTagValidation::~TauTagValidation() {
  if (genericTriggerEventFlag_)
    delete genericTriggerEventFlag_;
}

void TauTagValidation::bookHistograms(DQMStore::IBooker& ibooker,
                                      edm::Run const& iRun,
                                      edm::EventSetup const& /* iSetup */) {
  MonitorElement *ptTemp, *etaTemp, *phiTemp, *pileupTemp, *tmpME, *summaryTemp;

  ibooker.setCurrentFolder("RecoTauV/" + TauProducer_ + extensionName_ + "_Summary");
  auto n_disc = !discriminators_.empty() ? discriminators_.size() : 21;
  hinfo summaryHinfo = (histoSettings_.exists("summary"))
                           ? hinfo(histoSettings_.getParameter<edm::ParameterSet>("summary"))
                           : hinfo(n_disc, -0.5, n_disc - 0.5);
  summaryTemp =
      ibooker.book1D("summaryPlotNum", "summaryPlotNum", summaryHinfo.nbins, summaryHinfo.min, summaryHinfo.max);
  summaryMap.insert(std::make_pair(refCollection_ + "Num", summaryTemp));
  summaryTemp =
      ibooker.book1D("summaryPlotDen", "summaryPlotDen", summaryHinfo.nbins, summaryHinfo.min, summaryHinfo.max);
  summaryMap.insert(std::make_pair(refCollection_ + "Den", summaryTemp));
  summaryTemp = ibooker.book1D("summaryPlot", "summaryPlot", summaryHinfo.nbins, summaryHinfo.min, summaryHinfo.max);
  summaryMap.insert(std::make_pair(refCollection_, summaryTemp));

  ibooker.setCurrentFolder("RecoTauV/" + TauProducer_ + extensionName_ + "_ReferenceCollection");

  //Histograms settings
  hinfo ptHinfo = (histoSettings_.exists("pt")) ? hinfo(histoSettings_.getParameter<edm::ParameterSet>("pt"))
                                                : hinfo(500, 0., 1000.);
  hinfo etaHinfo = (histoSettings_.exists("eta")) ? hinfo(histoSettings_.getParameter<edm::ParameterSet>("eta"))
                                                  : hinfo(60, -3.0, 3.0);
  hinfo phiHinfo = (histoSettings_.exists("phi")) ? hinfo(histoSettings_.getParameter<edm::ParameterSet>("phi"))
                                                  : hinfo(40, -200., 200.);
  hinfo pileupHinfo = (histoSettings_.exists("pileup"))
                          ? hinfo(histoSettings_.getParameter<edm::ParameterSet>("pileup"))
                          : hinfo(100, 0., 100.);
  //hinfo dRHinfo = (histoSettings_.exists("deltaR")) ? hinfo(histoSettings_.getParameter<edm::ParameterSet>("deltaR")) : hinfo(10, 0., 0.5);

  // What kind of Taus do we originally have!

  ptTemp =
      ibooker.book1D("nRef_Taus_vs_ptTauVisible", "nRef_Taus_vs_ptTauVisible", ptHinfo.nbins, ptHinfo.min, ptHinfo.max);
  etaTemp = ibooker.book1D(
      "nRef_Taus_vs_etaTauVisible", "nRef_Taus_vs_etaTauVisible", etaHinfo.nbins, etaHinfo.min, etaHinfo.max);
  phiTemp = ibooker.book1D(
      "nRef_Taus_vs_phiTauVisible", "nRef_Taus_vs_phiTauVisible", phiHinfo.nbins, phiHinfo.min, phiHinfo.max);
  pileupTemp = ibooker.book1D("nRef_Taus_vs_pileupTauVisible",
                              "nRef_Taus_vs_pileupTauVisible",
                              pileupHinfo.nbins,
                              pileupHinfo.min,
                              pileupHinfo.max);

  ptTauVisibleMap.insert(std::make_pair(refCollection_, ptTemp));
  etaTauVisibleMap.insert(std::make_pair(refCollection_, etaTemp));
  phiTauVisibleMap.insert(std::make_pair(refCollection_, phiTemp));
  pileupTauVisibleMap.insert(std::make_pair(refCollection_, pileupTemp));

  // Number of Tau Candidates matched to MC Taus

  ibooker.setCurrentFolder("RecoTauV/" + TauProducer_ + extensionName_ + "_Matched");

  ptTemp = ibooker.book1D(TauProducer_ + "Matched_vs_ptTauVisible",
                          TauProducer_ + "Matched_vs_ptTauVisible",
                          ptHinfo.nbins,
                          ptHinfo.min,
                          ptHinfo.max);
  etaTemp = ibooker.book1D(TauProducer_ + "Matched_vs_etaTauVisible",
                           TauProducer_ + "Matched_vs_etaTauVisible",
                           etaHinfo.nbins,
                           etaHinfo.min,
                           etaHinfo.max);
  phiTemp = ibooker.book1D(TauProducer_ + "Matched_vs_phiTauVisible",
                           TauProducer_ + "Matched_vs_phiTauVisible",
                           phiHinfo.nbins,
                           phiHinfo.min,
                           phiHinfo.max);
  pileupTemp = ibooker.book1D(TauProducer_ + "Matched_vs_pileupTauVisible",
                              TauProducer_ + "Matched_vs_pileupTauVisible",
                              pileupHinfo.nbins,
                              pileupHinfo.min,
                              pileupHinfo.max);

  ptTauVisibleMap.insert(std::make_pair(TauProducer_ + "Matched", ptTemp));
  etaTauVisibleMap.insert(std::make_pair(TauProducer_ + "Matched", etaTemp));
  phiTauVisibleMap.insert(std::make_pair(TauProducer_ + "Matched", phiTemp));
  pileupTauVisibleMap.insert(std::make_pair(TauProducer_ + "Matched", pileupTemp));

  int j = 0;
  for (const auto& it : discriminators_) {
    string DiscriminatorLabel = it.getParameter<string>("discriminator");
    std::string histogramName;
    stripDiscriminatorLabel(DiscriminatorLabel, histogramName);

    //Summary plots
    string DiscriminatorLabelReduced = it.getParameter<string>("discriminator");
    DiscriminatorLabelReduced.erase(0, 24);
    summaryMap.find(refCollection_ + "Den")->second->setBinLabel(j + 1, DiscriminatorLabelReduced);
    summaryMap.find(refCollection_ + "Num")->second->setBinLabel(j + 1, DiscriminatorLabelReduced);
    summaryMap.find(refCollection_)->second->setBinLabel(j + 1, DiscriminatorLabelReduced);

    ibooker.setCurrentFolder("RecoTauV/" + TauProducer_ + extensionName_ + "_" + DiscriminatorLabel);

    ptTemp = ibooker.book1D(DiscriminatorLabel + "_vs_ptTauVisible",
                            histogramName + "_vs_ptTauVisible",
                            ptHinfo.nbins,
                            ptHinfo.min,
                            ptHinfo.max);
    etaTemp = ibooker.book1D(DiscriminatorLabel + "_vs_etaTauVisible",
                             histogramName + "_vs_etaTauVisible",
                             etaHinfo.nbins,
                             etaHinfo.min,
                             etaHinfo.max);
    phiTemp = ibooker.book1D(DiscriminatorLabel + "_vs_phiTauVisible",
                             histogramName + "_vs_phiTauVisible",
                             phiHinfo.nbins,
                             phiHinfo.min,
                             phiHinfo.max);
    pileupTemp = ibooker.book1D(DiscriminatorLabel + "_vs_pileupTauVisible",
                                histogramName + "_vs_pileupTauVisible",
                                pileupHinfo.nbins,
                                pileupHinfo.min,
                                pileupHinfo.max);

    ptTauVisibleMap.insert(std::make_pair(DiscriminatorLabel, ptTemp));
    etaTauVisibleMap.insert(std::make_pair(DiscriminatorLabel, etaTemp));
    phiTauVisibleMap.insert(std::make_pair(DiscriminatorLabel, phiTemp));
    pileupTauVisibleMap.insert(std::make_pair(DiscriminatorLabel, pileupTemp));

    tmpME = ibooker.book1D(
        DiscriminatorLabel + "_TauCandMass", histogramName + "_TauCandMass" + ";Cand Mass" + ";Frequency", 30, 0., 2.0);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + "_TauCandMass", tmpME));

    // momentum resolution for several decay modes

    std::string plotType = "_pTRatio_";  //use underscores (this allows to parse plot type in later stages)
    std::string xaxisLabel = ";p_{T}^{reco}/p_{T}^{gen}";
    std::string yaxislabel = ";Frequency";
    std::string plotName = plotType + "allHadronic";
    int bins = 40;
    tmpME =
        ibooker.book1D(DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 2.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "oneProng0Pi0";
    tmpME =
        ibooker.book1D(DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 2.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "oneProng1Pi0";
    tmpME =
        ibooker.book1D(DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 2.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "oneProng2Pi0";
    tmpME =
        ibooker.book1D(DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 2.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));

    plotName = plotType + "twoProng0Pi0";
    tmpME =
        ibooker.book1D(DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 2.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "twoProng1Pi0";
    tmpME =
        ibooker.book1D(DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 2.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "twoProng2Pi0";
    tmpME =
        ibooker.book1D(DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 2.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));

    plotName = plotType + "threeProng0Pi0";
    tmpME =
        ibooker.book1D(DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 2.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "threeProng1Pi0";
    tmpME =
        ibooker.book1D(DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 2.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));

    // Tau Multiplicity for several decay modes

    plotType = "_nTaus_";  //use underscores (this allows to parse plot type in later stages)
    xaxisLabel = ";Tau Multiplicity";
    yaxislabel = ";Frequency";
    plotName = plotType + "allHadronic";
    bins = 50;
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "oneProng0Pi0";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "oneProng1Pi0";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "oneProng2Pi0";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "twoProng0Pi0";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "twoProng1Pi0";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "twoProng2Pi0";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "threeProng0Pi0";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "threeProng1Pi0";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));

    //size and sumPt within tau isolation

    plotType = "_Size_";
    xaxisLabel = ";size";
    yaxislabel = ";Frequency";
    bins = 20;
    plotName = plotType + "signalCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, -0.5, bins - 0.5);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "signalChargedHadrCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, -0.5, bins - 0.5);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "signalNeutrHadrCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, -0.5, bins - 0.5);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));

    plotName = plotType + "isolationCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, -0.5, bins - 0.5);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "isolationChargedHadrCands";
    bins = 10;
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, -0.5, bins - 0.5);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "isolationNeutrHadrCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, -0.5, bins - 0.5);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "isolationGammaCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, -0.5, bins - 0.5);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));

    plotType = "_SumPt_";
    xaxisLabel = ";p_{T}^{sum}/ GeV";
    yaxislabel = ";Frequency";
    bins = 20;
    plotName = plotType + "signalCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "signalChargedHadrCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "signalNeutrHadrCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "isolationCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 50.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "isolationChargedHadrCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 10.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "isolationNeutrHadrCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 30.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));
    plotName = plotType + "isolationGammaCands";
    tmpME = ibooker.book1D(
        DiscriminatorLabel + plotName, histogramName + plotName + xaxisLabel + yaxislabel, bins, 0., 20.);
    plotMap_.insert(std::make_pair(DiscriminatorLabel + plotName, tmpME));

    //deprecated!

    if (DiscriminatorLabel.find("LeadingTrackPtCut") != string::npos) {
      if (TauProducer_.find("PFTau") != string::npos) {
        nPFJet_LeadingChargedHadron_ChargedHadronsSignal_ = ibooker.book1D(
            DiscriminatorLabel + "_ChargedHadronsSignal", DiscriminatorLabel + "_ChargedHadronsSignal", 21, -0.5, 20.5);
        nPFJet_LeadingChargedHadron_ChargedHadronsIsolAnnulus_ =
            ibooker.book1D(DiscriminatorLabel + "_ChargedHadronsIsolAnnulus",
                           DiscriminatorLabel + "_ChargedHadronsIsolAnnulus",
                           21,
                           -0.5,
                           20.5);
        nPFJet_LeadingChargedHadron_GammasSignal_ =
            ibooker.book1D(DiscriminatorLabel + "_GammasSignal", DiscriminatorLabel + "_GammasSignal", 21, -0.5, 20.5);
        nPFJet_LeadingChargedHadron_GammasIsolAnnulus_ = ibooker.book1D(
            DiscriminatorLabel + "_GammasIsolAnnulus", DiscriminatorLabel + "_GammasIsolAnnulus", 21, -0.5, 20.5);
        nPFJet_LeadingChargedHadron_NeutralHadronsSignal_ = ibooker.book1D(
            DiscriminatorLabel + "_NeutralHadronsSignal", DiscriminatorLabel + "_NeutralHadronsSignal", 21, -0.5, 20.5);
        nPFJet_LeadingChargedHadron_NeutralHadronsIsolAnnulus_ =
            ibooker.book1D(DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",
                           DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",
                           21,
                           -0.5,
                           20.5);
      }
    }

    if (DiscriminatorLabel.find("ByIsolationLater") != string::npos) {
      if (TauProducer_.find("PFTau") != string::npos) {
        nIsolated_NoChargedHadrons_ChargedHadronsSignal_ = ibooker.book1D(
            DiscriminatorLabel + "_ChargedHadronsSignal", DiscriminatorLabel + "_ChargedHadronsSignal", 21, -0.5, 20.5);
        nIsolated_NoChargedHadrons_GammasSignal_ =
            ibooker.book1D(DiscriminatorLabel + "_GammasSignal", DiscriminatorLabel + "_GammasSignal", 21, -0.5, 20.5);
        nIsolated_NoChargedHadrons_GammasIsolAnnulus_ = ibooker.book1D(
            DiscriminatorLabel + "_GammasIsolAnnulus", DiscriminatorLabel + "_GammasIsolAnnulus", 21, -0.5, 20.5);
        nIsolated_NoChargedHadrons_NeutralHadronsSignal_ = ibooker.book1D(
            DiscriminatorLabel + "_NeutralHadronsSignal", DiscriminatorLabel + "_NeutralHadronsSignal", 21, -0.5, 20.5);
        nIsolated_NoChargedHadrons_NeutralHadronsIsolAnnulus_ =
            ibooker.book1D(DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",
                           DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",
                           21,
                           -0.5,
                           20.5);
      }
    }

    if (DiscriminatorLabel.find("ByIsolation") != string::npos) {
      if (TauProducer_.find("PFTau") != string::npos) {
        nIsolated_NoChargedNoGammas_ChargedHadronsSignal_ = ibooker.book1D(
            DiscriminatorLabel + "_ChargedHadronsSignal", DiscriminatorLabel + "_ChargedHadronsSignal", 21, -0.5, 20.5);
        nIsolated_NoChargedNoGammas_GammasSignal_ =
            ibooker.book1D(DiscriminatorLabel + "_GammasSignal", DiscriminatorLabel + "_GammasSignal", 21, -0.5, 20.5);
        nIsolated_NoChargedNoGammas_NeutralHadronsSignal_ = ibooker.book1D(
            DiscriminatorLabel + "_NeutralHadronsSignal", DiscriminatorLabel + "_NeutralHadronsSignal", 21, -0.5, 20.5);
        nIsolated_NoChargedNoGammas_NeutralHadronsIsolAnnulus_ =
            ibooker.book1D(DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",
                           DiscriminatorLabel + "_NeutralHadronsIsolAnnulus",
                           21,
                           -0.5,
                           20.5);
      }
    }
    j++;
  }
}

void TauTagValidation::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (genericTriggerEventFlag_) {
    if (genericTriggerEventFlag_->on()) {
      genericTriggerEventFlag_->initRun(iRun, iSetup);
    }
  }
}

void TauTagValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (genericTriggerEventFlag_) {
    if (!genericTriggerEventFlag_->on())
      std::cout << "TauTagValidation::analyze: No working genericTriggerEventFlag. Did you specify a valid globaltag?"
                << std::endl;  //move to LogDebug?
  }

  numEvents_++;
  double matching_criteria = -1.0;

  //Initialize the Tau Multiplicity Counter
  for (const auto& it : discriminators_) {
    string DiscriminatorLabel = it.getParameter<string>("discriminator");
    tauDecayCountMap_["allHadronic" + DiscriminatorLabel] = 0;
    tauDecayCountMap_["oneProng0Pi0" + DiscriminatorLabel] = 0;
    tauDecayCountMap_["oneProng1Pi0" + DiscriminatorLabel] = 0;
    tauDecayCountMap_["oneProng2Pi0" + DiscriminatorLabel] = 0;
    tauDecayCountMap_["twoProng0Pi0" + DiscriminatorLabel] = 0;
    tauDecayCountMap_["twoProng1Pi0" + DiscriminatorLabel] = 0;
    tauDecayCountMap_["twoProng2Pi0" + DiscriminatorLabel] = 0;
    tauDecayCountMap_["threeProng0Pi0" + DiscriminatorLabel] = 0;
    tauDecayCountMap_["threeProng1Pi0" + DiscriminatorLabel] = 0;
  }

  typedef edm::View<reco::Candidate> genCandidateCollection;

  // ----------------------- Reference product -----------------------------------------------------------------------

  Handle<genCandidateCollection> ReferenceCollection;
  bool isGen = iEvent.getByToken(refCollectionInputTagToken_, ReferenceCollection);

  Handle<VertexCollection> pvHandle;
  iEvent.getByToken(primaryVertexCollectionToken_, pvHandle);  //TO-DO

  if (!isGen) {
    std::cerr << " Reference collection: " << refCollection_ << " not found while running TauTagValidation.cc "
              << std::endl;
    return;
  }

  if (dataType_ == "Leptons") {
    matching_criteria = matchDeltaR_Leptons_;
  } else {
    matching_criteria = matchDeltaR_Jets_;
  }

  // ------------------------------ PFTauCollection Matched and other discriminators ---------------------------------------------------------

  if (TauProducer_.find("PFTau") != string::npos || TauProducer_.find("hpsTancTaus") != string::npos) {
    Handle<PFTauCollection> thePFTauHandle;
    iEvent.getByToken(tauProducerInputTagToken_, thePFTauHandle);

    const PFTauCollection* pfTauProduct;
    pfTauProduct = thePFTauHandle.product();

    PFTauCollection::size_type thePFTauClosest;

    std::map<std::string, MonitorElement*>::const_iterator element = plotMap_.end();

    //Run the Reference Collection
    for (genCandidateCollection::const_iterator RefJet = ReferenceCollection->begin();
         RefJet != ReferenceCollection->end();
         RefJet++) {
      ptTauVisibleMap.find(refCollection_)->second->Fill(RefJet->pt());
      etaTauVisibleMap.find(refCollection_)->second->Fill(RefJet->eta());
      phiTauVisibleMap.find(refCollection_)->second->Fill(RefJet->phi() * 180.0 / TMath::Pi());
      pileupTauVisibleMap.find(refCollection_)->second->Fill(pvHandle->size());

      const reco::Candidate* gen_particle = &(*RefJet);

      double delta = TMath::Pi();

      thePFTauClosest = pfTauProduct->size();

      for (PFTauCollection::size_type iPFTau = 0; iPFTau < pfTauProduct->size(); iPFTau++) {
        if (algo_->deltaR(gen_particle, &pfTauProduct->at(iPFTau)) < delta) {
          delta = algo_->deltaR(gen_particle, &pfTauProduct->at(iPFTau));
          thePFTauClosest = iPFTau;
        }
      }

      // Skip if there is no reconstructed Tau matching the Reference
      if (thePFTauClosest == pfTauProduct->size())
        continue;

      double deltaR = algo_->deltaR(gen_particle, &pfTauProduct->at(thePFTauClosest));

      // Skip if the delta R difference is larger than the required criteria
      if (deltaR > matching_criteria && matching_criteria != -1.0)
        continue;

      ptTauVisibleMap.find(TauProducer_ + "Matched")->second->Fill(RefJet->pt());
      etaTauVisibleMap.find(TauProducer_ + "Matched")->second->Fill(RefJet->eta());
      phiTauVisibleMap.find(TauProducer_ + "Matched")->second->Fill(RefJet->phi() * 180.0 / TMath::Pi());
      pileupTauVisibleMap.find(TauProducer_ + "Matched")->second->Fill(pvHandle->size());

      PFTauRef thePFTau(thePFTauHandle, thePFTauClosest);

      Handle<PFTauDiscriminator> currentDiscriminator;

      //filter the candidates
      if (thePFTau->pt() < TauPtCut_)
        continue;  //almost deprecated, since recoCuts_ provides more flexibility
                   //reco
      StringCutObjectSelector<PFTauRef> selectReco(recoCuts_);
      bool pass = selectReco(thePFTau);
      if (!pass)
        continue;
      //gen
      StringCutObjectSelector<reco::Candidate> selectGen(genCuts_);
      pass = selectGen(*gen_particle);
      if (!pass)
        continue;

      int j = 0;
      for (const auto& it : discriminators_) {
        string currentDiscriminatorLabel = it.getParameter<string>("discriminator");
        iEvent.getByToken(currentDiscriminatorToken_[j], currentDiscriminator);
        summaryMap.find(refCollection_ + "Den")->second->Fill(j);

        if ((*currentDiscriminator)[thePFTau] >= it.getParameter<double>("selectionCut")) {
          ptTauVisibleMap.find(currentDiscriminatorLabel)->second->Fill(RefJet->pt());
          etaTauVisibleMap.find(currentDiscriminatorLabel)->second->Fill(RefJet->eta());
          phiTauVisibleMap.find(currentDiscriminatorLabel)->second->Fill(RefJet->phi() * 180.0 / TMath::Pi());
          pileupTauVisibleMap.find(currentDiscriminatorLabel)->second->Fill(pvHandle->size());
          summaryMap.find(refCollection_ + "Num")->second->Fill(j);

          //fill the momentum resolution plots
          double tauPtRes = thePFTau->pt() / gen_particle->pt();  //WARNING: use only the visible parts!
          plotMap_.find(currentDiscriminatorLabel + "_pTRatio_allHadronic")->second->Fill(tauPtRes);

          //Fill Tau Cand Mass
          TLorentzVector TAU;
          TAU.SetPtEtaPhiE(thePFTau->pt(), thePFTau->eta(), thePFTau->phi(), thePFTau->energy());
          plotMap_.find(currentDiscriminatorLabel + "_TauCandMass")->second->Fill(TAU.M());

          //Tau Counter, allHadronic mode
          tauDecayCountMap_.find("allHadronic" + currentDiscriminatorLabel)->second++;

          /*
          //is there a better way than casting the candidate?
          const reco::GenJet *tauGenJet = dynamic_cast<const reco::GenJet*>(gen_particle);
          if(tauGenJet){
            std::string genTauDecayMode =  JetMCTagUtils::genTauDecayMode(*tauGenJet); // gen_particle is the tauGenJet matched to the reconstructed tau
            element = plotMap_.find( currentDiscriminatorLabel + "_pTRatio_" + genTauDecayMode );
            if( element != plotMap_.end() ) element->second->Fill(tauPtRes);
            tauDecayCountMap_.find( genTauDecayMode + currentDiscriminatorLabel)->second++;
          }else{
            LogInfo("TauTagValidation") << " Failed to cast the MC candidate.";
	  }*/

          if (thePFTau->decayMode() == reco::PFTau::kOneProng0PiZero) {
            tauDecayCountMap_.find("oneProng0Pi0" + currentDiscriminatorLabel)->second++;
            plotMap_.find(currentDiscriminatorLabel + "_pTRatio_" + "oneProng0Pi0")->second->Fill(tauPtRes);
          } else if (thePFTau->decayMode() == reco::PFTau::kOneProng1PiZero) {
            tauDecayCountMap_.find("oneProng1Pi0" + currentDiscriminatorLabel)->second++;
            plotMap_.find(currentDiscriminatorLabel + "_pTRatio_" + "oneProng1Pi0")->second->Fill(tauPtRes);
          } else if (thePFTau->decayMode() == reco::PFTau::kOneProng2PiZero) {
            tauDecayCountMap_.find("oneProng2Pi0" + currentDiscriminatorLabel)->second++;
            plotMap_.find(currentDiscriminatorLabel + "_pTRatio_" + "oneProng2Pi0")->second->Fill(tauPtRes);
          } else if (thePFTau->decayMode() == reco::PFTau::kTwoProng0PiZero) {
            tauDecayCountMap_.find("twoProng0Pi0" + currentDiscriminatorLabel)->second++;
            plotMap_.find(currentDiscriminatorLabel + "_pTRatio_" + "twoProng0Pi0")->second->Fill(tauPtRes);
          } else if (thePFTau->decayMode() == reco::PFTau::kTwoProng1PiZero) {
            tauDecayCountMap_.find("twoProng1Pi0" + currentDiscriminatorLabel)->second++;
            plotMap_.find(currentDiscriminatorLabel + "_pTRatio_" + "twoProng1Pi0")->second->Fill(tauPtRes);
          } else if (thePFTau->decayMode() == reco::PFTau::kTwoProng2PiZero) {
            tauDecayCountMap_.find("twoProng2Pi0" + currentDiscriminatorLabel)->second++;
            plotMap_.find(currentDiscriminatorLabel + "_pTRatio_" + "twoProng2Pi0")->second->Fill(tauPtRes);
          } else if (thePFTau->decayMode() == reco::PFTau::kThreeProng0PiZero) {
            tauDecayCountMap_.find("threeProng0Pi0" + currentDiscriminatorLabel)->second++;
            plotMap_.find(currentDiscriminatorLabel + "_pTRatio_" + "threeProng0Pi0")->second->Fill(tauPtRes);
          } else if (thePFTau->decayMode() == reco::PFTau::kThreeProng1PiZero) {
            tauDecayCountMap_.find("threeProng1Pi0" + currentDiscriminatorLabel)->second++;
            plotMap_.find(currentDiscriminatorLabel + "_pTRatio_" + "threeProng1Pi0")->second->Fill(tauPtRes);
          }
          //fill: size and sumPt within tau isolation
          std::string plotType = "_Size_";
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "signalCands");
          if (element != plotMap_.end())
            element->second->Fill(thePFTau->signalCands().size());
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "signalChargedHadrCands");
          if (element != plotMap_.end())
            element->second->Fill(thePFTau->signalChargedHadrCands().size());
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "signalNeutrHadrCands");
          if (element != plotMap_.end())
            element->second->Fill(thePFTau->signalNeutrHadrCands().size());
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "isolationCands");
          if (element != plotMap_.end())
            element->second->Fill(thePFTau->isolationCands().size());
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "isolationChargedHadrCands");
          if (element != plotMap_.end())
            element->second->Fill(thePFTau->isolationChargedHadrCands().size());
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "isolationNeutrHadrCands");
          if (element != plotMap_.end())
            element->second->Fill(thePFTau->isolationNeutrHadrCands().size());
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "isolationGammaCands");
          if (element != plotMap_.end())
            element->second->Fill(thePFTau->isolationGammaCands().size());

          plotType = "_SumPt_";
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "signalCands");
          if (element != plotMap_.end())
            element->second->Fill(getSumPt(thePFTau->signalCands()));
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "signalChargedHadrCands");
          if (element != plotMap_.end())
            element->second->Fill(getSumPt(thePFTau->signalChargedHadrCands()));
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "signalNeutrHadrCands");
          if (element != plotMap_.end())
            element->second->Fill(getSumPt(thePFTau->signalNeutrHadrCands()));
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "isolationCands");
          if (element != plotMap_.end())
            element->second->Fill(getSumPt(thePFTau->isolationCands()));
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "isolationChargedHadrCands");
          if (element != plotMap_.end())
            element->second->Fill(getSumPt(thePFTau->isolationChargedHadrCands()));
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "isolationNeutrHadrCands");
          if (element != plotMap_.end())
            element->second->Fill(getSumPt(thePFTau->isolationNeutrHadrCands()));
          element = plotMap_.find(currentDiscriminatorLabel + plotType + "isolationGammaCands");
          if (element != plotMap_.end())
            element->second->Fill(getSumPt(thePFTau->isolationGammaCands()));

          //deprecated

          if (TauProducer_.find("PFTau") != string::npos) {
            if (currentDiscriminatorLabel.find("LeadingTrackPtCut") != string::npos) {
              nPFJet_LeadingChargedHadron_ChargedHadronsSignal_->Fill((*thePFTau).signalChargedHadrCands().size());
              nPFJet_LeadingChargedHadron_ChargedHadronsIsolAnnulus_->Fill(
                  (*thePFTau).isolationChargedHadrCands().size());
              nPFJet_LeadingChargedHadron_GammasSignal_->Fill((*thePFTau).signalGammaCands().size());
              nPFJet_LeadingChargedHadron_GammasIsolAnnulus_->Fill((*thePFTau).isolationGammaCands().size());
              nPFJet_LeadingChargedHadron_NeutralHadronsSignal_->Fill((*thePFTau).signalNeutrHadrCands().size());
              nPFJet_LeadingChargedHadron_NeutralHadronsIsolAnnulus_->Fill(
                  (*thePFTau).isolationNeutrHadrCands().size());
            } else if (currentDiscriminatorLabel.find("ByIsolation") != string::npos) {
              nIsolated_NoChargedNoGammas_ChargedHadronsSignal_->Fill((*thePFTau).signalChargedHadrCands().size());
              nIsolated_NoChargedNoGammas_GammasSignal_->Fill((*thePFTau).signalGammaCands().size());
              nIsolated_NoChargedNoGammas_NeutralHadronsSignal_->Fill((*thePFTau).signalNeutrHadrCands().size());
              nIsolated_NoChargedNoGammas_NeutralHadronsIsolAnnulus_->Fill(
                  (*thePFTau).isolationNeutrHadrCands().size());
            }
          }
        } else {
          if (chainCuts_)
            break;
        }
        j++;
      }
    }  //End of Reference Collection Loop

    //Fill the Tau Multiplicity Histograms
    for (const auto& it : discriminators_) {
      string currentDiscriminatorLabel = it.getParameter<string>("discriminator");
      plotMap_.find(currentDiscriminatorLabel + "_nTaus_allHadronic")
          ->second->Fill(tauDecayCountMap_.find("allHadronic" + currentDiscriminatorLabel)->second);
      plotMap_.find(currentDiscriminatorLabel + "_nTaus_oneProng0Pi0")
          ->second->Fill(tauDecayCountMap_.find("oneProng0Pi0" + currentDiscriminatorLabel)->second);
      plotMap_.find(currentDiscriminatorLabel + "_nTaus_oneProng1Pi0")
          ->second->Fill(tauDecayCountMap_.find("oneProng1Pi0" + currentDiscriminatorLabel)->second);
      plotMap_.find(currentDiscriminatorLabel + "_nTaus_oneProng2Pi0")
          ->second->Fill(tauDecayCountMap_.find("oneProng2Pi0" + currentDiscriminatorLabel)->second);
      plotMap_.find(currentDiscriminatorLabel + "_nTaus_twoProng0Pi0")
          ->second->Fill(tauDecayCountMap_.find("twoProng0Pi0" + currentDiscriminatorLabel)->second);
      plotMap_.find(currentDiscriminatorLabel + "_nTaus_twoProng1Pi0")
          ->second->Fill(tauDecayCountMap_.find("twoProng1Pi0" + currentDiscriminatorLabel)->second);
      plotMap_.find(currentDiscriminatorLabel + "_nTaus_twoProng2Pi0")
          ->second->Fill(tauDecayCountMap_.find("twoProng2Pi0" + currentDiscriminatorLabel)->second);
      plotMap_.find(currentDiscriminatorLabel + "_nTaus_threeProng0Pi0")
          ->second->Fill(tauDecayCountMap_.find("threeProng0Pi0" + currentDiscriminatorLabel)->second);
      plotMap_.find(currentDiscriminatorLabel + "_nTaus_threeProng1Pi0")
          ->second->Fill(tauDecayCountMap_.find("threeProng1Pi0" + currentDiscriminatorLabel)->second);
    }

  }  //End of PFTau Collection If Loop
}

double TauTagValidation::getSumPt(const std::vector<edm::Ptr<reco::Candidate> >& candidates) {
  double sumPt = 0.;
  for (std::vector<edm::Ptr<reco::Candidate> >::const_iterator candidate = candidates.begin();
       candidate != candidates.end();
       ++candidate) {
    sumPt += (*candidate)->pt();
  }
  return sumPt;
}

bool TauTagValidation::stripDiscriminatorLabel(const std::string& discriminatorLabel, std::string& newLabel) {
  std::string separatorString = "DiscriminationBy";
  std::string::size_type separator = discriminatorLabel.find(separatorString);
  if (separator == std::string::npos) {
    separatorString = "Discrimination";  //DiscriminationAgainst, keep the 'against' here
    separator = discriminatorLabel.find(separatorString);
    if (separator == std::string::npos) {
      return false;
    }
  }
  std::string prefix = discriminatorLabel.substr(0, separator);
  std::string postfix = discriminatorLabel.substr(separator + separatorString.size());
  newLabel = prefix + postfix;
  return true;
}
