#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalCATIAGainRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalCATIAGainRatios.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include <array>
#include <cstdlib>
#include <format>
#include <string>
#include <vector>
#include <map>

class EcalDigisValidationPh2 : public DQMEDAnalyzer {
  typedef std::map<uint32_t, float> MapType;

public:
  EcalDigisValidationPh2(const edm::ParameterSet& ps);
  ~EcalDigisValidationPh2() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) override;

protected:
  void analyze(edm::Event const& event, edm::EventSetup const& eventSetup) override;

private:
  static auto constexpr kMaxSamples_ = EcalDataFrame_Ph2::MAXSAMPLES;

  const edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;
  const edm::EDGetTokenT<EBDigiCollectionPh2> digiCollectionToken_;
  const edm::EDGetTokenT<CrossingFrame<PCaloHit> > crossingFramePCaloHitToken_;

  const edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> adcToGeVToken_;
  const edm::ESGetToken<EcalCATIAGainRatios, EcalCATIAGainRatiosRcd> gainRatiosToken_;

  MonitorElement* meGunEnergy_;
  MonitorElement* meGunEta_;
  MonitorElement* meGunPhi_;

  MonitorElement* meDigiSimRatio_;
  MonitorElement* meDigiSimRatiogt10ADC_;
  MonitorElement* meDigiSimRatiogt100ADC_;

  MonitorElement* meDigiOccupancy_;
  MonitorElement* meDigiMultiplicity_;
  MonitorElement* meDigiADCGlobal_;
  MonitorElement* meDigiADCAnalog_[kMaxSamples_];
  MonitorElement* meDigiADCg10_[kMaxSamples_];
  MonitorElement* meDigiADCg1_[kMaxSamples_];
  MonitorElement* meDigiGain_[kMaxSamples_];

  MonitorElement* mePedestal_;
  MonitorElement* meMaximumgt10ADC_;
  MonitorElement* meMaximumgt100ADC_;
  MonitorElement* menADCafterSwitch_;

  MonitorElement* meSm1OverSmax_;
  MonitorElement* meSm2OverSmax_;
  MonitorElement* meSp1OverSmax_;
  MonitorElement* meSp2OverSmax_;
  MonitorElement* meSp1OverSmaxvsSm1OverSmax_;
  MonitorElement* meSp2OverSmaxvsSm2OverSmax_;
};

EcalDigisValidationPh2::EcalDigisValidationPh2(const edm::ParameterSet& ps)
    : hepMCToken_(consumes<edm::HepMCProduct>(edm::InputTag(ps.getParameter<std::string>("moduleLabelMC")))),
      digiCollectionToken_(consumes<EBDigiCollectionPh2>(ps.getParameter<edm::InputTag>("digiCollection"))),
      crossingFramePCaloHitToken_(consumes<CrossingFrame<PCaloHit> >(
          edm::InputTag("mix", ps.getParameter<std::string>("moduleLabelG4") + std::string("EcalHitsEB")))),
      adcToGeVToken_(esConsumes()),
      gainRatiosToken_(esConsumes()),
      meGunEnergy_(nullptr),
      meGunEta_(nullptr),
      meGunPhi_(nullptr),
      meDigiSimRatio_(nullptr),
      meDigiSimRatiogt10ADC_(nullptr),
      meDigiSimRatiogt100ADC_(nullptr),
      meDigiOccupancy_(nullptr),
      meDigiMultiplicity_(nullptr),
      meDigiADCGlobal_(nullptr),
      mePedestal_(nullptr),
      meMaximumgt10ADC_(nullptr),
      meMaximumgt100ADC_(nullptr),
      menADCafterSwitch_(nullptr),
      meSm1OverSmax_(nullptr),
      meSm2OverSmax_(nullptr),
      meSp1OverSmax_(nullptr),
      meSp2OverSmax_(nullptr),
      meSp1OverSmaxvsSm1OverSmax_(nullptr),
      meSp2OverSmaxvsSm2OverSmax_(nullptr) {
  for (int i = 0; i < kMaxSamples_; ++i) {
    meDigiADCAnalog_[i] = nullptr;
    meDigiADCg10_[i] = nullptr;
    meDigiADCg1_[i] = nullptr;
    meDigiGain_[i] = nullptr;
  }
}

void EcalDigisValidationPh2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiCollection", edm::InputTag("simEcalUnsuppressedDigis"));
  desc.add<std::string>("moduleLabelMC", std::string("generatorSmeared"));
  desc.add<std::string>("moduleLabelG4", std::string("g4SimHits"));
  descriptions.add("ecalDigisValidationPh2", desc);
}

void EcalDigisValidationPh2::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder("EcalDigisV/EcalDigiTask");

  std::string histo("EcalDigiTask Gun Momentum");
  meGunEnergy_ = ibooker.book1D(histo, histo, 100, 0., 1000.);

  histo = "EcalDigiTask Gun Eta";
  meGunEta_ = ibooker.book1D(histo, histo, 700, -3.5, 3.5);

  histo = "EcalDigiTask Gun Phi";
  meGunPhi_ = ibooker.book1D(histo, histo, 360, 0., 360.);

  histo = "EcalDigiTask maximum Digi over Sim ratio";
  meDigiSimRatio_ = ibooker.book1D(histo, histo, 100, 0., 2.);

  histo = "EcalDigiTask maximum Digi over Sim ratio gt 10 ADC";
  meDigiSimRatiogt10ADC_ = ibooker.book1D(histo, histo, 100, 0., 2.);

  histo = "EcalDigiTask maximum Digi over Sim ratio gt 100 ADC";
  meDigiSimRatiogt100ADC_ = ibooker.book1D(histo, histo, 100, 0., 2.);

  histo = "EcalDigiTask occupancy";
  meDigiOccupancy_ = ibooker.book2D(histo, histo, 360, 0., 360., 170, -85., 85.);

  histo = "EcalDigiTask digis multiplicity";
  meDigiMultiplicity_ = ibooker.book1D(histo, histo, 612, 0., 61200);

  histo = "EcalDigiTask global pulse shape";
  meDigiADCGlobal_ = ibooker.bookProfile(histo, histo, kMaxSamples_, 0, kMaxSamples_, 10000, 0., 1000.);

  for (int i = 0; i < kMaxSamples_; ++i) {
    histo = std::format("EcalDigiTask analog pulse {:02d}", i + 1);
    meDigiADCAnalog_[i] = ibooker.book1DD(histo, histo, 4000, 0., 400.);

    histo = std::format("EcalDigiTask ADC pulse {:02d} Gain 10", i + 1);
    meDigiADCg10_[i] = ibooker.book1DD(histo, histo, 4096, -0.5, 4095.5);

    histo = std::format("EcalDigiTask ADC pulse {:02d} Gain 1", i + 1);
    meDigiADCg1_[i] = ibooker.book1D(histo, histo, 4096, -0.5, 4095.5);

    histo = std::format("EcalDigiTask gain pulse {:02d}", i + 1);
    meDigiGain_[i] = ibooker.book1DD(histo, histo, 2, 0, 2);
  }

  histo = "EcalDigiTask pedestal for pre-sample";
  mePedestal_ = ibooker.book1DD(histo, histo, 4096, -0.5, 4095.5);

  histo = "EcalDigiTask maximum position gt 10 ADC";
  meMaximumgt10ADC_ = ibooker.book1D(histo, histo, kMaxSamples_, 0., static_cast<double>(kMaxSamples_));

  histo = "EcalDigiTask maximum position gt 100 ADC";
  meMaximumgt100ADC_ = ibooker.book1D(histo, histo, kMaxSamples_, 0., static_cast<double>(kMaxSamples_));

  histo = "EcalDigiTask ADC counts after gain switch";
  menADCafterSwitch_ = ibooker.book1DD(histo, histo, kMaxSamples_, 0., static_cast<double>(kMaxSamples_));

  histo = "EcalDigiTask ADC sample ratio A_maxminus1 over A_max gt 100 ADC";
  meSm1OverSmax_ = ibooker.book1D(histo, histo + ";A_{max-1}/A_{max}", 110, 0., 1.1);

  histo = "EcalDigiTask ADC sample ratio A_maxminus2 over A_max gt 100 ADC";
  meSm2OverSmax_ = ibooker.book1D(histo, histo + ";A_{max-2}/A_{max}", 110, 0., 1.1);

  histo = "EcalDigiTask ADC sample ratio A_maxplus1 over A_max gt 100 ADC";
  meSp1OverSmax_ = ibooker.book1D(histo, histo + ";A_{max+1}/A_{max}", 110, 0., 1.1);

  histo = "EcalDigiTask ADC sample ratio A_maxplus2 over A_max gt 100 ADC";
  meSp2OverSmax_ = ibooker.book1D(histo, histo + ";A_{max+2}/A_{max}", 110, 0., 1.1);

  histo = "EcalDigiTask ADC sample ratio comparison R_plus1 vs R_minus1 gt 100 ADC";
  meSp1OverSmaxvsSm1OverSmax_ =
      ibooker.book2D(histo, histo + ";A_{max-1}/A_{max};A_{max+1}/A_{max}", 100, 0., 1., 100, 0., 1.);

  histo = "EcalDigiTask ADC sample ratio comparison R_plus2 vs R_minus2 gt 100 ADC";
  meSp2OverSmaxvsSm2OverSmax_ =
      ibooker.book2D(histo, histo + ";A_{max-2}/A_{max};A_{max+2}/A_{max}", 100, 0., 1., 100, 0., 1.);
}

void EcalDigisValidationPh2::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Handle<edm::HepMCProduct> mcEvt;
  edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;
  edm::Handle<EBDigiCollectionPh2> ecalDigis;

  bool skipMC = false;
  event.getByToken(hepMCToken_, mcEvt);
  if (!mcEvt.isValid()) {
    skipMC = true;
  }

  const EBDigiCollectionPh2* digis = nullptr;

  bool doDigis = true;
  event.getByToken(digiCollectionToken_, ecalDigis);
  if (ecalDigis.isValid()) {
    digis = ecalDigis.product();
    if (digis->empty())
      doDigis = false;
  } else {
    doDigis = false;
  }

  if (!skipMC) {
    for (HepMC::GenEvent::particle_const_iterator p = mcEvt->GetEvent()->particles_begin();
         p != mcEvt->GetEvent()->particles_end();
         ++p) {
      auto const theGunEnergy = (*p)->momentum().e();
      auto const htheta = (*p)->momentum().theta();
      auto const heta = -log(tan(htheta * 0.5));
      auto hphi = (*p)->momentum().phi();
      hphi = (hphi >= 0) ? hphi : hphi + 2 * M_PI;
      hphi = hphi / M_PI * 180.;
      LogDebug("EventInfo") << "Particle gun type form MC = " << std::abs((*p)->pdg_id()) << "\n"
                            << "Energy = " << (*p)->momentum().e() << " Eta = " << heta << " Phi = " << hphi;

      meGunEnergy_->Fill(theGunEnergy);
      meGunEta_->Fill(heta);
      meGunPhi_->Fill(hphi);
    }
  }

  if (doDigis) {
    event.getByToken(crossingFramePCaloHitToken_, crossingFrame);
    const MixCollection<PCaloHit> barrelHits(crossingFrame.product());

    MapType ebSimMap;
    // loop over simHits
    for (auto const& iHit : barrelHits) {
      auto const ebid = EBDetId(iHit.id());

      LogDebug("HitInfo") << " CaloHit " << iHit.getName() << "\n"
                          << " DetID = " << iHit.id() << " EBDetId = " << ebid.ieta() << " " << ebid.iphi() << "\n"
                          << " Time = " << iHit.time() << " Event id. = " << iHit.eventId().rawId() << "\n"
                          << " Track Id = " << iHit.geantTrackId() << "\n"
                          << " Energy = " << iHit.energy();

      auto const crystid = ebid.rawId();
      ebSimMap[crystid] += iHit.energy();
    }

    // get conditions
    auto const adcToGeV = eventSetup.getData(adcToGeVToken_).getEBValue();
    auto const& gainRatios = eventSetup.getData(gainRatiosToken_);
    // The nominal CATIA gains are 10 and 1 when the gain bit is 0 and 1, respectively
    std::array<EcalCATIAGainRatio, 2> gainConv = {1., 10.};

    meDigiMultiplicity_->Fill(digis->size());

    // loop over Digis
    for (unsigned int digi = 0; digi < digis->size(); ++digi) {
      auto const& ebdf = (*digis)[digi];
      auto const nrSamples = ebdf.size();

      const EBDetId ebid(ebdf.id());

      meDigiOccupancy_->Fill(ebid.iphi(), ebid.ieta());

      gainConv[1] = gainRatios[ebid];

      double emax = 0.;
      int pmax = 0;
      double pedestalPreSample = 0.;
      double pedestalPreSampleAnalog = 0.;
      int countsAfterGainSwitch = 0;
      unsigned int higherGainSample = 0;
      int prevGainId = 0;

      std::vector<double> ebAnalogSignal(nrSamples, 0.);

      for (unsigned int sample = 0; sample < nrSamples; ++sample) {
        const EcalLiteDTUSample mySample = ebdf[sample];
        ebAnalogSignal[sample] = mySample.adc() * gainConv[mySample.gainId()] * adcToGeV;
        if (emax < ebAnalogSignal[sample]) {
          emax = ebAnalogSignal[sample];
          pmax = sample;
        }
        if (sample < 3) {
          pedestalPreSample += mySample.adc();
          pedestalPreSampleAnalog += mySample.adc() * gainConv[mySample.gainId()] * adcToGeV;
        }

        if (sample > 0 && mySample.gainId() > prevGainId) {
          higherGainSample = sample;
          countsAfterGainSwitch = 1;
        }
        if (mySample.gainId() > 0 && sample != higherGainSample) {
          ++countsAfterGainSwitch;
        }

        LogDebug("DigiInfo") << " sample " << sample << " ADC counts = " << mySample.adc()
                             << " Gain Id = " << mySample.gainId() << " Analog eq = " << ebAnalogSignal[sample];

        meDigiADCAnalog_[sample]->Fill(ebAnalogSignal[sample]);

        if (mySample.gainId() == 0) {
          meDigiADCg10_[sample]->Fill(mySample.adc());
        } else if (mySample.gainId() == 1) {
          meDigiADCg1_[sample]->Fill(mySample.adc());
        }
        meDigiGain_[sample]->Fill(mySample.gainId());

        prevGainId = mySample.gainId();
      }

      pedestalPreSample /= 3.;
      pedestalPreSampleAnalog /= 3.;
      auto const pmaxGainId = static_cast<EcalLiteDTUSample>(ebdf[pmax]).gainId();
      auto const Erec = emax - pedestalPreSampleAnalog * gainConv[pmaxGainId];

      for (unsigned int sample = 0; sample < nrSamples; ++sample) {
        if (Erec > 100. * adcToGeV)
          meDigiADCGlobal_->Fill(sample, ebAnalogSignal[sample]);
      }

      auto const simHit = ebSimMap.find(ebid.rawId());
      if (simHit != ebSimMap.end() && simHit->second != 0.) {
        LogDebug("DigiInfo") << " Digi / Hit = " << Erec << " / " << simHit->second << " gainConv "
                             << gainConv[pmaxGainId];
        meDigiSimRatio_->Fill(Erec / simHit->second);
        if (Erec > 10. * adcToGeV) {
          meDigiSimRatiogt10ADC_->Fill(Erec / simHit->second);
          if (Erec > 100. * adcToGeV)
            meDigiSimRatiogt100ADC_->Fill(Erec / simHit->second);
        }
      }

      mePedestal_->Fill(pedestalPreSample);
      if (Erec > 10. * adcToGeV) {
        meMaximumgt10ADC_->Fill(pmax);
        if (Erec > 100. * adcToGeV)
          meMaximumgt100ADC_->Fill(pmax);
      }
      menADCafterSwitch_->Fill(countsAfterGainSwitch);

      // Sample ratios
      if (Erec > 100. * adcToGeV) {
        double erecp1 = 0.;
        double erecp2 = 0.;
        if (pmax < static_cast<int>(nrSamples) - 1) {
          erecp1 = ebAnalogSignal[pmax + 1] -
                   pedestalPreSampleAnalog * gainConv[static_cast<EcalLiteDTUSample>(ebdf[pmax + 1]).gainId()];
          meSp1OverSmax_->Fill(erecp1 / Erec);
          if (pmax < static_cast<int>(nrSamples) - 2) {
            erecp2 = ebAnalogSignal[pmax + 2] -
                     pedestalPreSampleAnalog * gainConv[static_cast<EcalLiteDTUSample>(ebdf[pmax + 2]).gainId()];
            meSp2OverSmax_->Fill(erecp2 / Erec);
          }
        }
        if (pmax > 1) {
          auto const erecm1 =
              ebAnalogSignal[pmax - 1] -
              pedestalPreSampleAnalog * gainConv[static_cast<EcalLiteDTUSample>(ebdf[pmax - 1]).gainId()];
          meSm1OverSmax_->Fill(erecm1 / Erec);
          if (pmax < static_cast<int>(nrSamples) - 1) {
            meSp1OverSmaxvsSm1OverSmax_->Fill(erecm1 / Erec, erecp1 / Erec);
          }
          if (pmax > 2) {
            auto const erecm2 =
                ebAnalogSignal[pmax - 2] -
                pedestalPreSampleAnalog * gainConv[static_cast<EcalLiteDTUSample>(ebdf[pmax - 2]).gainId()];
            meSm2OverSmax_->Fill(erecm2 / Erec);
            if (pmax < static_cast<int>(nrSamples) - 2) {
              meSp2OverSmaxvsSm2OverSmax_->Fill(erecm2 / Erec, erecp2 / Erec);
            }
          }
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(EcalDigisValidationPh2);
