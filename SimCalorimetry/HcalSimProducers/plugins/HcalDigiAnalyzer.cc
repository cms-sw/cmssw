/** Studies Hcal digis

  \Author Rick Wilkinson, Caltech
*/

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloValidationStatistics.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCHitFilter.h"

#include <iostream>
#include <string>

class HcalDigiStatistics {
public:
  HcalDigiStatistics(std::string name,
                     int maxBin,
                     float amplitudeThreshold,
                     float expectedPedestal,
                     float binPrevToBinMax,
                     float binNextToBinMax,
                     CaloHitAnalyzer &amplitudeAnalyzer)
      : maxBin_(maxBin),
        amplitudeThreshold_(amplitudeThreshold),
        pedestal_(name + " pedestal", expectedPedestal, 0.),
        binPrevToBinMax_(name + " binPrevToBinMax", binPrevToBinMax, 0.),
        binNextToBinMax_(name + " binNextToBinMax", binNextToBinMax, 0.),
        amplitudeAnalyzer_(amplitudeAnalyzer) {}

  template <class Digi>
  void analyze(const Digi &digi);

private:
  int maxBin_;
  float amplitudeThreshold_;
  CaloValidationStatistics pedestal_;
  CaloValidationStatistics binPrevToBinMax_;
  CaloValidationStatistics binNextToBinMax_;
  CaloHitAnalyzer &amplitudeAnalyzer_;
};

template <class Digi>
void HcalDigiStatistics::analyze(const Digi &digi) {
  pedestal_.addEntry(digi[0].adc());
  pedestal_.addEntry(digi[1].adc());

  double pedestal_fC = 0.5 * (digi[0].nominal_fC() + digi[1].nominal_fC());

  double maxAmplitude = digi[maxBin_].nominal_fC() - pedestal_fC;

  if (maxAmplitude > amplitudeThreshold_) {
    double binPrevToBinMax = (digi[maxBin_ - 1].nominal_fC() - pedestal_fC) / maxAmplitude;
    binPrevToBinMax_.addEntry(binPrevToBinMax);

    double binNextToBinMax = (digi[maxBin_ + 1].nominal_fC() - pedestal_fC) / maxAmplitude;
    binNextToBinMax_.addEntry(binNextToBinMax);

    double amplitude = digi[maxBin_].nominal_fC() + digi[maxBin_ + 1].nominal_fC() - 2 * pedestal_fC;

    amplitudeAnalyzer_.analyze(digi.id().rawId(), amplitude);
  }
}

class HcalDigiAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalDigiAnalyzer(edm::ParameterSet const &conf);
  void analyze(edm::Event const &e, edm::EventSetup const &c) override;

private:
  std::string hitReadoutName_;
  HcalSimParameterMap simParameterMap_;
  HBHEHitFilter hbheFilter_;
  HOHitFilter hoFilter_;
  HFHitFilter hfFilter_;
  ZDCHitFilter zdcFilter_;
  CaloHitAnalyzer hbheHitAnalyzer_;
  CaloHitAnalyzer hoHitAnalyzer_;
  CaloHitAnalyzer hfHitAnalyzer_;
  CaloHitAnalyzer zdcHitAnalyzer_;
  HcalDigiStatistics hbheDigiStatistics_;
  HcalDigiStatistics hoDigiStatistics_;
  HcalDigiStatistics hfDigiStatistics_;
  HcalDigiStatistics zdcDigiStatistics_;

  edm::InputTag hbheDigiCollectionTag_;
  edm::InputTag hoDigiCollectionTag_;
  edm::InputTag hfDigiCollectionTag_;
};

HcalDigiAnalyzer::HcalDigiAnalyzer(edm::ParameterSet const &conf)
    : hitReadoutName_("HcalHits"),
      simParameterMap_(),
      hbheFilter_(),
      hoFilter_(),
      hfFilter_(),
      hbheHitAnalyzer_("HBHEDigi", 1., &simParameterMap_, &hbheFilter_),
      hoHitAnalyzer_("HODigi", 1., &simParameterMap_, &hoFilter_),
      hfHitAnalyzer_("HFDigi", 1., &simParameterMap_, &hfFilter_),
      zdcHitAnalyzer_("ZDCDigi", 1., &simParameterMap_, &zdcFilter_),
      hbheDigiStatistics_("HBHEDigi", 4, 10., 6., 0.1, 0.5, hbheHitAnalyzer_),
      hoDigiStatistics_("HODigi", 4, 10., 6., 0.1, 0.5, hoHitAnalyzer_),
      hfDigiStatistics_("HFDigi", 3, 10., 6., 0.1, 0.5, hfHitAnalyzer_),
      zdcDigiStatistics_("ZDCDigi", 3, 10., 6., 0.1, 0.5, zdcHitAnalyzer_),
      hbheDigiCollectionTag_(conf.getParameter<edm::InputTag>("hbheDigiCollectionTag")),
      hoDigiCollectionTag_(conf.getParameter<edm::InputTag>("hoDigiCollectionTag")),
      hfDigiCollectionTag_(conf.getParameter<edm::InputTag>("hfDigiCollectionTag")) {}

namespace HcalDigiAnalyzerImpl {
  template <class Collection>
  void analyze(edm::Event const &e, HcalDigiStatistics &statistics, edm::InputTag &tag) {
    edm::Handle<Collection> digis;
    e.getByLabel(tag, digis);
    for (unsigned i = 0; i < digis->size(); ++i) {
      std::cout << (*digis)[i] << std::endl;
      statistics.analyze((*digis)[i]);
    }
  }
}  // namespace HcalDigiAnalyzerImpl

void HcalDigiAnalyzer::analyze(edm::Event const &e, edm::EventSetup const &c) {
  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PCaloHit>> cf, zdccf;
  e.getByLabel("mix", "HcalHits", cf);
  // e.getByLabel("mix", "ZDCHits", zdccf);

  // test access to SimHits for HcalHits and ZDC hits
  std::unique_ptr<MixCollection<PCaloHit>> hits(new MixCollection<PCaloHit>(cf.product()));
  // std::unique_ptr<MixCollection<PCaloHit> > zdcHits(new
  // MixCollection<PCaloHit>(zdccf.product()));
  hbheHitAnalyzer_.fillHits(*hits);
  hoHitAnalyzer_.fillHits(*hits);
  hfHitAnalyzer_.fillHits(*hits);
  // zdcHitAnalyzer_.fillHits(*zdcHits);
  HcalDigiAnalyzerImpl::analyze<HBHEDigiCollection>(e, hbheDigiStatistics_, hbheDigiCollectionTag_);
  HcalDigiAnalyzerImpl::analyze<HODigiCollection>(e, hoDigiStatistics_, hoDigiCollectionTag_);
  HcalDigiAnalyzerImpl::analyze<HFDigiCollection>(e, hfDigiStatistics_, hfDigiCollectionTag_);
  // HcalDigiAnalyzerImpl::analyze<ZDCDigiCollection>(e, zdcDigiStatistics_);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalDigiAnalyzer);
