#include <iostream>
//

#include "Validation/RecoEgamma/plugins/ConversionPostprocessing.h"

//

/** \class ConversionPostprocessing
 **
 **
 **  $Id: ConversionPostprocessing
 **  author:
 **   Nancy Marinelli, U. of Notre Dame, US
 **
 **
 ***/

using namespace std;

ConversionPostprocessing::ConversionPostprocessing(const edm::ParameterSet& pset) {
  dbe_ = nullptr;
  dbe_ = edm::Service<DQMStore>().operator->();
  parameters_ = pset;

  standAlone_ = pset.getParameter<bool>("standAlone");
  batch_ = pset.getParameter<bool>("batch");
  outputFileName_ = pset.getParameter<string>("OutputFileName");
  inputFileName_ = pset.getParameter<std::string>("InputFileName");

  etMin = parameters_.getParameter<double>("etMin");
  etMax = parameters_.getParameter<double>("etMax");
  etBin = parameters_.getParameter<int>("etBin");

  etaMin = parameters_.getParameter<double>("etaMin");
  etaMax = parameters_.getParameter<double>("etaMax");
  etaBin = parameters_.getParameter<int>("etaBin");
  etaBin2 = parameters_.getParameter<int>("etaBin2");

  phiMin = parameters_.getParameter<double>("phiMin");
  phiMax = parameters_.getParameter<double>("phiMax");
  phiBin = parameters_.getParameter<int>("phiBin");

  rMin = parameters_.getParameter<double>("rMin");
  rMax = parameters_.getParameter<double>("rMax");
  rBin = parameters_.getParameter<int>("rBin");

  zMin = parameters_.getParameter<double>("zMin");
  zMax = parameters_.getParameter<double>("zMax");
  zBin = parameters_.getParameter<int>("zBin");
}

ConversionPostprocessing::~ConversionPostprocessing() {}

void ConversionPostprocessing::beginJob() {}

void ConversionPostprocessing::analyze(const edm::Event& e, const edm::EventSetup& esup) {}

void ConversionPostprocessing::endJob() {
  if (standAlone_)
    runPostprocessing();
}

void ConversionPostprocessing::endRun(const edm::Run& run, const edm::EventSetup& setup) {
  if (!standAlone_)
    runPostprocessing();
}

void ConversionPostprocessing::runPostprocessing() {
  std::string simInfoPathName = "EgammaV/ConversionValidator/SimulationInfo/";
  std::string convPathName = "EgammaV/ConversionValidator/ConversionInfo/";
  std::string effPathName = "EgammaV/ConversionValidator/EfficienciesAndFakeRate/";

  if (batch_)
    dbe_->open(inputFileName_);

  dbe_->setCurrentFolder(effPathName);
  // Conversion reconstruction efficiency
  std::string histname = "convEffVsEtaTwoTracks";
  convEffEtaTwoTracks_ = dbe_->book1D(histname, histname, etaBin2, etaMin, etaMax);

  histname = "convEffVsPhiTwoTracks";
  convEffPhiTwoTracks_ = dbe_->book1D(histname, histname, phiBin, phiMin, phiMax);

  histname = "convEffVsRTwoTracks";
  convEffRTwoTracks_ = dbe_->book1D(histname, histname, rBin, rMin, rMax);

  histname = "convEffVsZTwoTracks";
  convEffZTwoTracks_ = dbe_->book1D(histname, histname, zBin, zMin, zMax);

  histname = "convEffVsEtTwoTracks";
  convEffEtTwoTracks_ = dbe_->book1D(histname, histname, etBin, etMin, etMax);
  //
  histname = "convEffVsEtaTwoTracksAndVtxProbGT0";
  convEffEtaTwoTracksAndVtxProbGT0_ = dbe_->book1D(histname, histname, etaBin2, etaMin, etaMax);
  histname = "convEffVsEtaTwoTracksAndVtxProbGT0005";
  convEffEtaTwoTracksAndVtxProbGT0005_ = dbe_->book1D(histname, histname, etaBin2, etaMin, etaMax);
  histname = "convEffVsRTwoTracksAndVtxProbGT0";
  convEffRTwoTracksAndVtxProbGT0_ = dbe_->book1D(histname, histname, rBin, rMin, rMax);
  histname = "convEffVsRTwoTracksAndVtxProbGT0005";
  convEffRTwoTracksAndVtxProbGT0005_ = dbe_->book1D(histname, histname, rBin, rMin, rMax);
  //
  // Fake rate
  histname = "convFakeRateVsEtaTwoTracks";
  convFakeRateEtaTwoTracks_ = dbe_->book1D(histname, histname, etaBin2, etaMin, etaMax);
  histname = "convFakeRateVsPhiTwoTracks";
  convFakeRatePhiTwoTracks_ = dbe_->book1D(histname, histname, phiBin, phiMin, phiMax);
  histname = "convFakeRateVsRTwoTracks";
  convFakeRateRTwoTracks_ = dbe_->book1D(histname, histname, rBin, rMin, rMax);
  histname = "convFakeRateVsZTwoTracks";
  convFakeRateZTwoTracks_ = dbe_->book1D(histname, histname, zBin, zMin, zMax);
  histname = "convFakeRateVsEtTwoTracks";
  convFakeRateEtTwoTracks_ = dbe_->book1D(histname, histname, etBin, etMin, etMax);

  // efficiencies
  dividePlots(dbe_->get(effPathName + "convEffVsEtaTwoTracks"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksEta"),
              dbe_->get(simInfoPathName + "h_VisSimConvEta"),
              "effic");
  dividePlots(dbe_->get(effPathName + "convEffVsPhiTwoTracks"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksPhi"),
              dbe_->get(simInfoPathName + "h_VisSimConvPhi"),
              "effic");
  dividePlots(dbe_->get(effPathName + "convEffVsRTwoTracks"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksR"),
              dbe_->get(simInfoPathName + "h_VisSimConvR"),
              "effic");
  dividePlots(dbe_->get(effPathName + "convEffVsZTwoTracks"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksZ"),
              dbe_->get(simInfoPathName + "h_VisSimConvZ"),
              "effic");
  dividePlots(dbe_->get(effPathName + "convEffVsEtTwoTracks"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksEt"),
              dbe_->get(simInfoPathName + "h_VisSimConvEt"),
              "effic");
  dividePlots(dbe_->get(effPathName + "convEffVsEtaTwoTracksAndVtxProbGT0"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksEtaAndVtxPGT0"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksEta"),
              "effic");
  dividePlots(dbe_->get(effPathName + "convEffVsEtaTwoTracksAndVtxProbGT0005"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksEtaAndVtxPGT0005"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksEta"),
              "effic");
  dividePlots(dbe_->get(effPathName + "convEffVsRTwoTracksAndVtxProbGT0"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksRAndVtxPGT0"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksR"),
              "effic");
  dividePlots(dbe_->get(effPathName + "convEffVsRTwoTracksAndVtxProbGT0005"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksRAndVtxPGT0005"),
              dbe_->get(simInfoPathName + "h_SimConvTwoMTracksR"),
              "effic");

  // fake rate
  dividePlots(dbe_->get(effPathName + "convFakeRateVsEtaTwoTracks"),
              dbe_->get(convPathName + "convEtaAss2"),
              dbe_->get(convPathName + "convEta2"),
              "fakerate");
  dividePlots(dbe_->get(effPathName + "convFakeRateVsPhiTwoTracks"),
              dbe_->get(convPathName + "convPhiAss"),
              dbe_->get(convPathName + "convPhi"),
              "fakerate");
  dividePlots(dbe_->get(effPathName + "convFakeRateVsRTwoTracks"),
              dbe_->get(convPathName + "convRAss"),
              dbe_->get(convPathName + "convR"),
              "fakerate");
  dividePlots(dbe_->get(effPathName + "convFakeRateVsZTwoTracks"),
              dbe_->get(convPathName + "convZAss"),
              dbe_->get(convPathName + "convZ"),
              "fakerate");
  dividePlots(dbe_->get(effPathName + "convFakeRateVsEtTwoTracks"),
              dbe_->get(convPathName + "convPtAss"),
              dbe_->get(convPathName + "convPt"),
              "fakerate");

  if (standAlone_)
    dbe_->save(outputFileName_);
  else if (batch_)
    dbe_->save(inputFileName_);
}

void ConversionPostprocessing::dividePlots(MonitorElement* dividend,
                                           MonitorElement* numerator,
                                           MonitorElement* denominator,
                                           std::string type) {
  double value, err;

  //quick fix to avoid seg. faults due to null pointers.
  if (dividend == nullptr || numerator == nullptr || denominator == nullptr)
    return;

  for (int j = 1; j <= numerator->getNbinsX(); j++) {
    if (denominator->getBinContent(j) != 0) {
      if (type == "effic")
        value = ((double)numerator->getBinContent(j)) / ((double)denominator->getBinContent(j));
      else if (type == "fakerate")
        value = 1 - ((double)numerator->getBinContent(j)) / ((double)denominator->getBinContent(j));
      else
        return;
      err = sqrt(value * (1 - value) / ((double)denominator->getBinContent(j)));
      dividend->setBinContent(j, value);
      if (err != 0)
        dividend->setBinError(j, err);
    } else {
      dividend->setBinContent(j, 0);
      dividend->setBinError(j, 0);
    }
  }
}

void ConversionPostprocessing::dividePlots(MonitorElement* dividend, MonitorElement* numerator, double denominator) {
  double value, err;

  //quick fix to avoid seg. faults due to null pointers.
  if (dividend == nullptr || numerator == nullptr)
    return;

  for (int j = 1; j <= numerator->getNbinsX(); j++) {
    if (denominator != 0) {
      value = ((double)numerator->getBinContent(j)) / denominator;
      err = sqrt(value * (1 - value) / denominator);
      dividend->setBinContent(j, value);
      dividend->setBinError(j, err);
    } else {
      dividend->setBinContent(j, 0);
    }
  }
}
