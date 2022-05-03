/**  Castor digis
 Author: Panos Katsas
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
#include "SimCalorimetry/CastorSim/interface/CastorHitFilter.h"
#include "SimCalorimetry/CastorSim/interface/CastorSimParameterMap.h"

#include <iostream>
#include <string>

class CastorDigiStatistics {
public:
  CastorDigiStatistics(std::string name,
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
void CastorDigiStatistics::analyze(const Digi &digi) {
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

class CastorDigiAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CastorDigiAnalyzer(edm::ParameterSet const &conf);
  void analyze(edm::Event const &e, edm::EventSetup const &c) override;

private:
  std::string hitReadoutName_;
  CastorSimParameterMap simParameterMap_;
  CastorHitFilter castorFilter_;
  CaloHitAnalyzer castorHitAnalyzer_;
  CastorDigiStatistics castorDigiStatistics_;
  const edm::EDGetTokenT<CastorDigiCollection> castordigiToken_;
  const edm::EDGetTokenT<CrossingFrame<PCaloHit>> castorcfToken_;
};

CastorDigiAnalyzer::CastorDigiAnalyzer(edm::ParameterSet const &conf)
    : hitReadoutName_("CastorHits"),
      simParameterMap_(),
      castorHitAnalyzer_("CASTORDigi", 1., &simParameterMap_, &castorFilter_),
      castorDigiStatistics_("CASTORDigi", 3, 10., 6., 0.1, 0.5, castorHitAnalyzer_),
      castordigiToken_(consumes<CastorDigiCollection>(conf.getParameter<edm::InputTag>("castorDigiCollectionTag"))),
      castorcfToken_(consumes<CrossingFrame<PCaloHit>>(edm::InputTag("mix", "g4SimHitsCastorFI"))) {}

namespace CastorDigiAnalyzerImpl {
  template <class Collection>
  void analyze(edm::Event const &e, CastorDigiStatistics &statistics, const edm::EDGetTokenT<Collection> &token) {
    const edm::Handle<Collection> &digis = e.getHandle(token);
    if (!digis.isValid()) {
      edm::LogError("CastorDigiAnalyzer") << "Could not find Castor Digi Container ";
    } else {
      for (unsigned i = 0; i < digis->size(); ++i) {
        statistics.analyze((*digis)[i]);
      }
    }
  }
}  // namespace CastorDigiAnalyzerImpl

void CastorDigiAnalyzer::analyze(edm::Event const &e, edm::EventSetup const &c) {
  //  edm::Handle<edm::PCaloHitContainer> hits;
  const edm::Handle<CrossingFrame<PCaloHit>> &castorcf = e.getHandle(castorcfToken_);

  // access to SimHits
  std::unique_ptr<MixCollection<PCaloHit>> hits(new MixCollection<PCaloHit>(castorcf.product()));
  //  if (hits.isValid()) {
  castorHitAnalyzer_.fillHits(*hits);
  CastorDigiAnalyzerImpl::analyze<CastorDigiCollection>(e, castorDigiStatistics_, castordigiToken_);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CastorDigiAnalyzer);
