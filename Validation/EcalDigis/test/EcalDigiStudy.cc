
/*
 * \file EcalDigiStudy.h
 *
 * \author Sunanda Banerjee
 *
*/

#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#include "TH2D.h"
#include "TH1D.h"

//#define EDM_ML_DEBUG

class EcalDigiStudy : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {

public:
  EcalDigiStudy(const edm::ParameterSet& ps);
  ~EcalDigiStudy() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void endJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void checkCalibrations(edm::EventSetup const& c);

private:
  bool verbose_;

  std::string outputFile_;

  edm::EDGetTokenT<EBDigiCollection> EBdigiCollection_;
  edm::EDGetTokenT<EEDigiCollection> EEdigiCollection_;
  edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> pAgc;
  std::map<int, double, std::less<int> > gainConv_;

  double barrelADCtoGeV_;
  double endcapADCtoGeV_;

  TH2D* meEBDigiOccupancy_;
  TH1D* meEBDigiMultiplicity_;
  TH1D* meEBDigiADCAnalog_[10];
  TH1D* meEBDigiADCgS_[10];
  TH1D* meEBDigiADCg1_[10];
  TH1D* meEBDigiADCg6_[10];
  TH1D* meEBDigiADCg12_[10];
  TH1D* meEBDigiGain_[10];
  TH1D* meEBPedestal_;
  TH1D* meEBMaximumgt100ADC_;
  TH1D* meEBMaximumgt10ADC_;
  TH1D* meEBnADCafterSwitch_;

  TH2D* meEEDigiOccupancyzp_;
  TH2D* meEEDigiOccupancyzm_;
  TH1D* meEEDigiMultiplicityzp_;
  TH1D* meEEDigiMultiplicityzm_;
  TH1D* meEEDigiADCAnalog_[10];
  TH1D* meEEDigiADCgS_[10];
  TH1D* meEEDigiADCg1_[10];
  TH1D* meEEDigiADCg6_[10];
  TH1D* meEEDigiADCg12_[10];
  TH1D* meEEDigiGain_[10];
  TH1D* meEEPedestal_;
  TH1D* meEEMaximumgt100ADC_;
  TH1D* meEEMaximumgt20ADC_;
  TH1D* meEEnADCafterSwitch_;
};

EcalDigiStudy::EcalDigiStudy(const edm::ParameterSet& ps)
    : EBdigiCollection_(consumes<EBDigiCollection>(ps.getParameter<edm::InputTag>("EBdigiCollection"))),
      EEdigiCollection_(consumes<EEDigiCollection>(ps.getParameter<edm::InputTag>("EEdigiCollection"))),
      pAgc(esConsumes<edm::Transition::BeginRun>()) {
  usesResource(TFileService::kSharedResource);
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // get hold of back-end interface

  gainConv_[1] = 1.;
  gainConv_[2] = 2.;
  gainConv_[3] = 12.;
  gainConv_[0] = 12.;  // saturated channels
  barrelADCtoGeV_ = 0.035;
  endcapADCtoGeV_ = 0.06;

}

void EcalDigiStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("EBdigiCollection", edm::InputTag("simEcalDigis","ebDigis"));
  desc.add<edm::InputTag>("EEdigiCollection", edm::InputTag("simEcalDigis","eeDigis"));
  desc.addUntracked<bool>("verbose", "false");
  descriptions.add("ecalDigiStudy", desc);
}

void EcalDigiStudy::beginRun(const edm::Run&, const edm::EventSetup& es) {

  checkCalibrations(es);
  
  edm::Service<TFileService> fs;
  Char_t histo[200];

  // ECAL Barrel
  sprintf(histo, "EcalDigiTask Barrel occupancy");
  meEBDigiOccupancy_ = fs->make<TH2D>(histo, histo, 360, 0., 360., 170, -85., 85.);
  sprintf(histo, "EcalDigiTask Barrel digis multiplicity");
  meEBDigiMultiplicity_ = fs->make<TH1D>(histo, histo, 612, 0., 61200);
  sprintf(histo, "EcalDigiTask Barrel global pulse shape");
  for (int i = 0; i < 10; i++) {
    sprintf(histo, "EcalDigiTask Barrel analog pulse %02d", i + 1);
    meEBDigiADCAnalog_[i] = fs->make<TH1D>(histo, histo, 4000, 0., 400.);
    sprintf(histo, "EcalDigiTask Barrel ADC pulse %02d Gain 0 - Saturated", i + 1);
    meEBDigiADCgS_[i] = fs->make<TH1D>(histo, histo, 4096, -0.5, 4095.5);
    sprintf(histo, "EcalDigiTask Barrel ADC pulse %02d Gain 1", i + 1);
    meEBDigiADCg1_[i] = fs->make<TH1D>(histo, histo, 4096, -0.5, 4095.5);
    sprintf(histo, "EcalDigiTask Barrel ADC pulse %02d Gain 6", i + 1);
    meEBDigiADCg6_[i] = fs->make<TH1D>(histo, histo, 4096, -0.5, 4095.5);
    sprintf(histo, "EcalDigiTask Barrel ADC pulse %02d Gain 12", i + 1);
    meEBDigiADCg12_[i] = fs->make<TH1D>(histo, histo, 4096, -0.5, 4095.5);
    sprintf(histo, "EcalDigiTask Barrel gain pulse %02d", i + 1);
    meEBDigiGain_[i] = fs->make<TH1D>(histo, histo, 4, 0, 4);
  }
  sprintf(histo, "EcalDigiTask Barrel pedestal for pre-sample");
  meEBPedestal_ = fs->make<TH1D>(histo, histo, 4096, -0.5, 4095.5);
  sprintf(histo, "EcalDigiTask Barrel maximum position gt 100 ADC");
  meEBMaximumgt100ADC_ = fs->make<TH1D>(histo, histo, 10, 0., 10.);
  sprintf(histo, "EcalDigiTask Barrel maximum position gt 10 ADC");
  meEBMaximumgt10ADC_ = fs->make<TH1D>(histo, histo, 10, 0., 10.);
  sprintf(histo, "EcalDigiTask Barrel ADC counts after gain switch");
  meEBnADCafterSwitch_ = fs->make<TH1D>(histo, histo, 10, 0., 10.);

  //ECAL Endcap
  sprintf(histo, "EcalDigiTask Endcap occupancy z+");
  meEEDigiOccupancyzp_ = fs->make<TH2D>(histo, histo, 100, 0., 100., 100, 0., 100.);
  sprintf(histo, "EcalDigiTask Endcap occupancy z-");
  meEEDigiOccupancyzm_ = fs->make<TH2D>(histo, histo, 100, 0., 100., 100, 0., 100.);
  sprintf(histo, "EcalDigiTask Endcap multiplicity z+");
  meEEDigiMultiplicityzp_ = fs->make<TH1D>(histo, histo, 100, 0., 7324.);
  sprintf(histo, "EcalDigiTask Endcap multiplicity z-");
  meEEDigiMultiplicityzm_ = fs->make<TH1D>(histo, histo, 100, 0., 7324.);
  sprintf(histo, "EcalDigiTask Endcap global pulse shape");
  for (int i = 0; i < 10; i++) {
    sprintf(histo, "EcalDigiTask Endcap analog pulse %02d", i + 1);
    meEEDigiADCAnalog_[i] = fs->make<TH1D>(histo, histo, 4000, 0., 400.);
    sprintf(histo, "EcalDigiTask Endcap ADC pulse %02d Gain 0 - Saturated", i + 1);
    meEEDigiADCgS_[i] = fs->make<TH1D>(histo, histo, 4096, -0.5, 4095.5);
    sprintf(histo, "EcalDigiTask Endcap ADC pulse %02d Gain 1", i + 1);
    meEEDigiADCg1_[i] = fs->make<TH1D>(histo, histo, 4096, -0.5, 4095.5);
    sprintf(histo, "EcalDigiTask Endcap ADC pulse %02d Gain 6", i + 1);
    meEEDigiADCg6_[i] = fs->make<TH1D>(histo, histo, 4096, -0.5, 4095.5);
    sprintf(histo, "EcalDigiTask Endcap ADC pulse %02d Gain 12", i + 1);
    meEEDigiADCg12_[i] = fs->make<TH1D>(histo, histo, 4096, -0.5, 4095.5);
    sprintf(histo, "EcalDigiTask Endcap gain pulse %02d", i + 1);
    meEEDigiGain_[i] = fs->make<TH1D>(histo, histo, 4, 0, 4);
  }
  sprintf(histo, "EcalDigiTask Endcap pedestal for pre-sample");
  meEEPedestal_ = fs->make<TH1D>(histo, histo, 4096, -0.5, 4095.5);
  sprintf(histo, "EcalDigiTask Endcap maximum position gt 100 ADC");
  meEEMaximumgt100ADC_ = fs->make<TH1D>(histo, histo, 10, 0., 10.);
  sprintf(histo, "EcalDigiTask Endcap maximum position gt 20 ADC");
  meEEMaximumgt20ADC_ = fs->make<TH1D>(histo, histo, 10, 0., 10.);
  sprintf(histo, "EcalDigiTask Endcap ADC counts after gain switch");
  meEEnADCafterSwitch_ = fs->make<TH1D>(histo, histo, 10, 0., 10.);
}

void EcalDigiStudy::analyze(edm::Event const& e, edm::EventSetup const& c) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalDigiStudy") << " Run = " << e.id().run() << " Event = " << e.id().event();
#endif

  edm::Handle<EBDigiCollection> EcalDigiEB;
  e.getByToken(EBdigiCollection_, EcalDigiEB);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalDigiStudy") << " EB collection " << EcalDigiEB.isValid();
#endif
  
  if (EcalDigiEB.isValid()) {
    // BARREL: Loop over Digis

    const EBDigiCollection* barrelDigi = EcalDigiEB.product();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalDigiStudy") << " EB Digi size " << EcalDigiEB->size();
#endif
  
    std::vector<double> ebAnalogSignal, ebADCCounts, ebADCGains;
    ebAnalogSignal.reserve(EBDataFrame::MAXSAMPLES);
    ebADCCounts.reserve(EBDataFrame::MAXSAMPLES);
    ebADCGains.reserve(EBDataFrame::MAXSAMPLES);

    int nDigis(0);
    for (unsigned int digis = 0; digis < EcalDigiEB->size(); ++digis) {
      EBDataFrame ebdf = (*barrelDigi)[digis];
      int nrSamples = ebdf.size();
      EBDetId ebid = ebdf.id();

      nDigis++;
      meEBDigiOccupancy_->Fill(ebid.iphi(), ebid.ieta());
      double Emax(0.);
      int Pmax(0);
      double pedestalPreSample(0.), pedestalPreSampleAnalog(0.);
      int countsAfterGainSwitch = -1;
      double higherGain = 1.;
      int higherGainSample = 0;

      for (int sample = 0; sample < nrSamples; ++sample) {
        ebAnalogSignal[sample] = 0.;
        ebADCCounts[sample] = 0.;
        ebADCGains[sample] = 0.;
      }

      for (int sample = 0; sample < nrSamples; ++sample) {
        EcalMGPASample thisSample = ebdf[sample];

        ebADCCounts[sample] = (thisSample.adc());
        ebADCGains[sample] = (thisSample.gainId());
        ebAnalogSignal[sample] = (ebADCCounts[sample] * gainConv_[(int)ebADCGains[sample]] * barrelADCtoGeV_);

        if (Emax < ebAnalogSignal[sample]) {
          Emax = ebAnalogSignal[sample];
          Pmax = sample;
        }

        if (sample < 3) {
          pedestalPreSample += ebADCCounts[sample];
          pedestalPreSampleAnalog += ebADCCounts[sample] * gainConv_[(int)ebADCGains[sample]] * barrelADCtoGeV_;
        }

        if (sample > 0 && (((ebADCGains[sample] > ebADCGains[sample - 1]) && (ebADCGains[sample - 1] != 0)) ||
                           (countsAfterGainSwitch < 0 && ebADCGains[sample] == 0))) {
          higherGain = ebADCGains[sample];
          higherGainSample = sample;
          countsAfterGainSwitch = 1;
        }

        if ((higherGain > 1 && (higherGainSample != sample) && (ebADCGains[sample] == higherGain)) ||
            (higherGain == 3 && (higherGainSample != sample) && (ebADCGains[sample] == 0)) ||
            (higherGain == 0 && (higherGainSample != sample) &&
	     ((ebADCGains[sample] == 3) || (ebADCGains[sample] == 0)))) {
          countsAfterGainSwitch++;
        }
      }

      pedestalPreSample /= 3.;
      pedestalPreSampleAnalog /= 3.;

#ifdef EDM_ML_DEBUG
      if (verbose_) {
	edm::LogVerbatim("EcalDigiStudy") << "Barrel Digi for EBDetId = " << ebid.rawId() << " eta,phi " << ebid.ieta() << " " << ebid.iphi();
	for (int i = 0; i < 10; i++) 
	  edm::LogVerbatim("EcalDigiSTudy") << "sample " << i << " ADC = " << ebADCCounts[i] << " gain = " << ebADCGains[i] << " Analog = " << ebAnalogSignal[i];
	edm::LogVerbatim("EcalDigiStudy") << "Maximum energy = " << Emax << " in sample " << Pmax << " Pedestal from pre-sample = " << pedestalPreSampleAnalog;
	if (countsAfterGainSwitch > 0)
	  edm::LogVerbatim("EcalDigiStudy") << "Counts after switch " << countsAfterGainSwitch;
      }
#endif
      if (countsAfterGainSwitch > 0 && countsAfterGainSwitch < 5) {
        edm::LogWarning("EcalDigiStudy") << "Wrong number of counts after gain switch before next switch! " << countsAfterGainSwitch;
        for (int i = 0; i < 10; i++) 
          edm::LogWarning("EcalDigiStudy") << "sample " << i << " ADC = " << ebADCCounts[i] << " gain = " << ebADCGains[i] << " Analog = " << ebAnalogSignal[i];
      }

      for (int i = 0; i < 10; i++) {
        meEBDigiADCAnalog_[i]->Fill(ebAnalogSignal[i]);

        if (ebADCGains[i] == 0) {
          meEBDigiADCgS_[i]->Fill(ebADCCounts[i]);
        } else if (ebADCGains[i] == 3) {
          meEBDigiADCg1_[i]->Fill(ebADCCounts[i]);
        } else if (ebADCGains[i] == 2) {
          meEBDigiADCg6_[i]->Fill(ebADCCounts[i]);
        } else if (ebADCGains[i] == 1) {
          meEBDigiADCg12_[i]->Fill(ebADCCounts[i]);
        }
        meEBDigiGain_[i]->Fill(ebADCGains[i]);
      }

      meEBPedestal_->Fill(pedestalPreSample);
      if ((Emax - pedestalPreSampleAnalog * gainConv_[(int)ebADCGains[Pmax]]) > 10. * barrelADCtoGeV_)
        meEBMaximumgt10ADC_->Fill(Pmax);
      if ((Emax - pedestalPreSampleAnalog * gainConv_[(int)ebADCGains[Pmax]]) > 100. * barrelADCtoGeV_)
        meEBMaximumgt100ADC_->Fill(Pmax);
      meEBnADCafterSwitch_->Fill(countsAfterGainSwitch);
    }
    meEBDigiMultiplicity_->Fill(nDigis);
}
 
  edm:: Handle<EEDigiCollection> EcalDigiEE;
  e.getByToken(EEdigiCollection_, EcalDigiEE);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalDigiStudy") << " EE collection " << EcalDigiEE.isValid();
#endif

  if (EcalDigiEE.isValid()) {

    // ENDCAP: Loop over Digis
    const EEDigiCollection* endcapDigi = EcalDigiEE.product();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalDigiStudy") << " EE Digi size " << EcalDigiEE->size();
#endif

    std::vector<double> eeAnalogSignal, eeADCCounts, eeADCGains;
    eeAnalogSignal.reserve(EEDataFrame::MAXSAMPLES);
    eeADCCounts.reserve(EEDataFrame::MAXSAMPLES);
    eeADCGains.reserve(EEDataFrame::MAXSAMPLES);

    int nDigiszp(0), nDigiszm(0);

    for (unsigned int digis = 0; digis < EcalDigiEE->size(); ++digis) {
      EEDataFrame eedf = (*endcapDigi)[digis];
      int nrSamples = eedf.size();

      EEDetId eeid = eedf.id();

      if (eeid.zside() > 0) {
        meEEDigiOccupancyzp_->Fill(eeid.ix(), eeid.iy());
        nDigiszp++;
      } else if (eeid.zside() < 0) {
        meEEDigiOccupancyzm_->Fill(eeid.ix(), eeid.iy());
        nDigiszm++;
      }

      double Emax(0.);
      int Pmax(0);
      double pedestalPreSample(0.), pedestalPreSampleAnalog(0.);
      int countsAfterGainSwitch(-1);
      double higherGain(1.);
      int higherGainSample(0);

      for (int sample = 0; sample < nrSamples; ++sample) {
        eeAnalogSignal[sample] = 0.;
        eeADCCounts[sample] = 0.;
        eeADCGains[sample] = 0.;
      }

      for (int sample = 0; sample < nrSamples; ++sample) {
        EcalMGPASample mySample = eedf[sample];

        eeADCCounts[sample] = (mySample.adc());
        eeADCGains[sample] = (mySample.gainId());
        eeAnalogSignal[sample] = (eeADCCounts[sample] * gainConv_[(int)eeADCGains[sample]] * endcapADCtoGeV_);

        if (Emax < eeAnalogSignal[sample]) {
          Emax = eeAnalogSignal[sample];
          Pmax = sample;
        }

        if (sample < 3) {
          pedestalPreSample += eeADCCounts[sample];
          pedestalPreSampleAnalog += eeADCCounts[sample] * gainConv_[(int)eeADCGains[sample]] * endcapADCtoGeV_;
        }

        if (sample > 0 && (((eeADCGains[sample] > eeADCGains[sample - 1]) && (eeADCGains[sample - 1] != 0)) ||
                           (countsAfterGainSwitch < 0 && eeADCGains[sample] == 0))) {
          higherGain = eeADCGains[sample];
          higherGainSample = sample;
          countsAfterGainSwitch = 1;
        }

        if ((higherGain > 1 && (higherGainSample != sample) && (eeADCGains[sample] == higherGain)) ||
            (higherGain == 3 && (higherGainSample != sample) && (eeADCGains[sample] == 0)) ||
            (higherGain == 0 && (higherGainSample != sample) && ((eeADCGains[sample] == 0) || (eeADCGains[sample] == 3))))
          countsAfterGainSwitch++;
      }
      pedestalPreSample /= 3.;
      pedestalPreSampleAnalog /= 3.;

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EcalDigiStudy") << "Endcap Digi for EEDetId = " << eeid.rawId() << " x,y " << eeid.ix() << " " << eeid.iy();
      for (int i = 0; i < 10; i++)
        edm::LogVerbatim("EcalDigiStudy") << "sample " << i << " ADC = " << eeADCCounts[i] << " gain = " << eeADCGains[i]
                             << " Analog = " << eeAnalogSignal[i];
      edm::LogVerbatim("EcalDigiStudy") << "Maximum energy = " << Emax << " in sample " << Pmax
                                        << " Pedestal from pre-sample = " << pedestalPreSampleAnalog;
      if (countsAfterGainSwitch > 0)
        edm::LogVerbatim("EcalDigiStudy") << "Counts after switch " << countsAfterGainSwitch;
#endif
      if (countsAfterGainSwitch > 0 && countsAfterGainSwitch < 5) {
        edm::LogWarning("EcalDigiStudy") << "Wrong number of counts after gain switch before next switch! "
                                         << countsAfterGainSwitch;
        for (int i = 0; i < 10; i++)
          edm::LogWarning("EcalDigiStudy") << "sample " << i << " ADC = " << eeADCCounts[i] << " gain = " << eeADCGains[i]
                                           << " Analog = " << eeAnalogSignal[i];
      }

      for (int i = 0; i < 10; i++) {
        meEEDigiADCAnalog_[i]->Fill(eeAnalogSignal[i]);
        if (eeADCGains[i] == 0) {
          meEEDigiADCgS_[i]->Fill(eeADCCounts[i]);
        } else if (eeADCGains[i] == 3) {
          meEEDigiADCg1_[i]->Fill(eeADCCounts[i]);
        } else if (eeADCGains[i] == 2) {
          meEEDigiADCg6_[i]->Fill(eeADCCounts[i]);
        } else if (eeADCGains[i] == 1) {
          meEEDigiADCg12_[i]->Fill(eeADCCounts[i]);
        }
        meEEDigiGain_[i]->Fill(eeADCGains[i]);
      }

      meEEPedestal_->Fill(pedestalPreSample);
      if ((Emax - pedestalPreSampleAnalog * gainConv_[(int)eeADCGains[Pmax]]) > 20. * endcapADCtoGeV_)
        meEEMaximumgt20ADC_->Fill(Pmax);
      if ((Emax - pedestalPreSampleAnalog * gainConv_[(int)eeADCGains[Pmax]]) > 100. * endcapADCtoGeV_)
        meEEMaximumgt100ADC_->Fill(Pmax);
      meEEnADCafterSwitch_->Fill(countsAfterGainSwitch);
    }

    meEEDigiMultiplicityzp_->Fill(nDigiszp);
    meEEDigiMultiplicityzm_->Fill(nDigiszm);
  }
}

void EcalDigiStudy::checkCalibrations(edm::EventSetup const& eventSetup) {
  // ADC -> GeV Scale

  const EcalADCToGeVConstant* agc = &eventSetup.getData(pAgc);
  EcalMGPAGainRatio* defaultRatios = new EcalMGPAGainRatio();

  gainConv_[1] = 1.;
  gainConv_[2] = defaultRatios->gain12Over6();
  gainConv_[3] = gainConv_[2] * (defaultRatios->gain6Over1());
  gainConv_[0] = gainConv_[2] * (defaultRatios->gain6Over1());

  edm::LogVerbatim("EcalDigiStudy") << " Gains conversions: "
                                    << "\n"
                                    << " g0 = " << gainConv_[0] << "\n"
                                    << " g1 = " << gainConv_[1] << "\n"
                                    << " g2 = " << gainConv_[2] << "\n"
                                    << " g3 = " << gainConv_[3];

  delete defaultRatios;
 
  edm::LogVerbatim("EcalDigiStudy") << " Barrel GeV/ADC = " << agc->getEBValue();
  edm::LogVerbatim("EcalDigiStudy") << " Endcap GeV/ADC = " << agc->getEEValue();
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalDigiStudy);
