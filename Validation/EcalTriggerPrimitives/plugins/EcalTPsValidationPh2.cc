#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalDigi/interface/EcalEBPhase2TriggerPrimitiveDigi.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <format>
#include <string>

class EcalTPsValidationPh2 : public DQMEDAnalyzer {
public:
  EcalTPsValidationPh2(const edm::ParameterSet& ps);
  ~EcalTPsValidationPh2() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) override;

protected:
  void analyze(edm::Event const& event, edm::EventSetup const& eventSetup) override;

private:
  // match DataFormats/EcalDigi/src/EcalEBPhase2TriggerPrimitiveDigi.cc
  static constexpr unsigned int kMaxSamples_ = 20;

  const edm::EDGetTokenT<EcalEBPhase2TrigPrimDigiCollection> tpDigiCollectionToken_;

  MonitorElement* meTPDigisMultiplicity_;
  MonitorElement* meTPDigiOccupancy_;
  MonitorElement* meTPDigiSize_;
  MonitorElement* meTPDigiEt_;
  MonitorElement* meTPDigiSpike_;
  MonitorElement* meTPDigiTime_;
  MonitorElement* meTPDigiDebugFlag_;
  MonitorElement* meTPDigiSOI_;

  MonitorElement* meTPSampleEt_[kMaxSamples_];
  MonitorElement* meTPSampleSpike_[kMaxSamples_];
  MonitorElement* meTPSampleTime_[kMaxSamples_];
};

EcalTPsValidationPh2::EcalTPsValidationPh2(const edm::ParameterSet& ps)
    : tpDigiCollectionToken_(
          consumes<EcalEBPhase2TrigPrimDigiCollection>(ps.getParameter<edm::InputTag>("tpDigiCollection"))),
      meTPDigisMultiplicity_(nullptr),
      meTPDigiOccupancy_(nullptr),
      meTPDigiSize_(nullptr),
      meTPDigiEt_(nullptr),
      meTPDigiSpike_(nullptr),
      meTPDigiTime_(nullptr),
      meTPDigiDebugFlag_(nullptr),
      meTPDigiSOI_(nullptr) {
  for (unsigned int i = 0; i < kMaxSamples_; ++i) {
    meTPSampleEt_[i] = nullptr;
    meTPSampleSpike_[i] = nullptr;
    meTPSampleTime_[i] = nullptr;
  }
}

void EcalTPsValidationPh2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tpDigiCollection", edm::InputTag("simEcalEBTriggerPrimitivePhase2Digis"));
  descriptions.add("ecalTPsValidationPh2", desc);
}

void EcalTPsValidationPh2::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder("EcalDigisV/EcalTriggerPrimitivesTask");

  std::string histo("EcalTPDigiTask TP digis multiplicity");
  meTPDigisMultiplicity_ = ibooker.book1D(histo, histo, 613, 0, 61300);

  histo = "EcalTPDigiTask TP digi occupancy";
  meTPDigiOccupancy_ = ibooker.book2D(histo, histo, 360, 0., 360., 170, -85., 85.);

  histo = "EcalTPDigiTask TP digi size";
  meTPDigiSize_ = ibooker.book1D(histo, histo, kMaxSamples_, 0, kMaxSamples_);

  histo = "EcalTPDigiTask TP digi encoded ET";
  meTPDigiEt_ = ibooker.book1D(histo, histo, 1025, -1, 1024);

  histo = "EcalTPDigiTask TP digi spike flag";
  meTPDigiSpike_ = ibooker.book1D(histo, histo, 2, 0, 2);

  histo = "EcalTPDigiTask TP digi time";
  meTPDigiTime_ = ibooker.book1D(histo, histo, 33, -1, 32);

  histo = "EcalTPDigiTask TP digi debug flag";
  meTPDigiDebugFlag_ = ibooker.book1D(histo, histo, 2, 0, 2);

  histo = "EcalTPDigiTask TP digi sample of interest";
  meTPDigiSOI_ = ibooker.book1D(histo, histo, kMaxSamples_ + 1, -1, kMaxSamples_);

  for (unsigned int i = 0; i < kMaxSamples_; ++i) {
    histo = std::format("EcalTPDigiTask TP sample {:02d} encoded ET", i);
    meTPSampleEt_[i] = ibooker.book1D(histo, histo, 1024, 0, 1024);

    histo = std::format("EcalTPDigiTask TP sample {:02d} spike flag", i);
    meTPSampleSpike_[i] = ibooker.book1D(histo, histo, 2, 0, 2);

    histo = std::format("EcalTPDigiTask TP sample {:02d} time", i);
    meTPSampleTime_[i] = ibooker.book1D(histo, histo, 32, 0, 32);
  }
}

void EcalTPsValidationPh2::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Handle<EcalEBPhase2TrigPrimDigiCollection> ecalTPDigis;
  const EcalEBPhase2TrigPrimDigiCollection* tpDigis = nullptr;

  event.getByToken(tpDigiCollectionToken_, ecalTPDigis);
  if (ecalTPDigis.isValid()) {
    tpDigis = ecalTPDigis.product();

    meTPDigisMultiplicity_->Fill(tpDigis->size());

    // loop over TP digis
    for (unsigned int tpDigiIdx = 0; tpDigiIdx < tpDigis->size(); ++tpDigiIdx) {
      auto const& tpDigi = (*tpDigis)[tpDigiIdx];

      const EBDetId ebid(tpDigi.id());
      meTPDigiOccupancy_->Fill(ebid.iphi(), ebid.ieta());

      meTPDigiSize_->Fill(tpDigi.size());

      meTPDigiEt_->Fill(tpDigi.encodedEt());
      meTPDigiSpike_->Fill(tpDigi.l1aSpike());
      meTPDigiTime_->Fill(tpDigi.time());
      meTPDigiDebugFlag_->Fill(tpDigi.isDebug());
      meTPDigiSOI_->Fill(tpDigi.sampleOfInterest());

      // loop over TP digi samples
      for (int sample = 0; sample < tpDigi.size(); ++sample) {
        auto const tpSample = tpDigi[sample];
        meTPSampleEt_[sample]->Fill(tpSample.encodedEt());
        meTPSampleSpike_[sample]->Fill(tpSample.l1aSpike());
        meTPSampleTime_[sample]->Fill(tpSample.time());
      }
    }
  }
}

DEFINE_FWK_MODULE(EcalTPsValidationPh2);
