/*
 * \file EcalDigisValidation.cc
 *
 * \author F. Cossutti
 *
*/

#include "EcalDigisValidation.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"

EcalDigisValidation::EcalDigisValidation(const edm::ParameterSet& ps)
    : HepMCToken_(consumes<edm::HepMCProduct>(edm::InputTag(ps.getParameter<std::string>("moduleLabelMC")))),
      g4TkInfoToken_(consumes<edm::SimTrackContainer>(edm::InputTag(ps.getParameter<std::string>("moduleLabelG4")))),
      g4VtxInfoToken_(consumes<edm::SimVertexContainer>(edm::InputTag(ps.getParameter<std::string>("moduleLabelG4")))),
      EBdigiCollectionToken_(consumes<EBDigiCollection>(ps.getParameter<edm::InputTag>("EBdigiCollection"))),
      EEdigiCollectionToken_(consumes<EEDigiCollection>(ps.getParameter<edm::InputTag>("EEdigiCollection"))),
      ESdigiCollectionToken_(consumes<ESDigiCollection>(ps.getParameter<edm::InputTag>("ESdigiCollection"))),
      pAgc(esConsumes<edm::Transition::BeginRun>()),
      crossingFramePCaloHitEBToken_(consumes<CrossingFrame<PCaloHit> >(edm::InputTag(
          std::string("mix"), ps.getParameter<std::string>("moduleLabelG4") + std::string("EcalHitsEB")))),
      crossingFramePCaloHitEEToken_(consumes<CrossingFrame<PCaloHit> >(edm::InputTag(
          std::string("mix"), ps.getParameter<std::string>("moduleLabelG4") + std::string("EcalHitsEE")))),
      crossingFramePCaloHitESToken_(consumes<CrossingFrame<PCaloHit> >(edm::InputTag(
          std::string("mix"), ps.getParameter<std::string>("moduleLabelG4") + std::string("EcalHitsES")))) {
  // DQM ROOT output
  outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");

  if (!outputFile_.empty()) {
    edm::LogInfo("OutputInfo") << " Ecal Digi Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Ecal Digi Task histograms will NOT be saved";
  }

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  gainConv_[1] = 1.;
  gainConv_[2] = 2.;
  gainConv_[3] = 12.;
  gainConv_[0] = 12.;  // saturated channels
  barrelADCtoGeV_ = 0.035;
  endcapADCtoGeV_ = 0.06;

  meGunEnergy_ = nullptr;
  meGunEta_ = nullptr;
  meGunPhi_ = nullptr;

  meEBDigiSimRatio_ = nullptr;
  meEEDigiSimRatio_ = nullptr;

  meEBDigiSimRatiogt10ADC_ = nullptr;
  meEEDigiSimRatiogt20ADC_ = nullptr;

  meEBDigiSimRatiogt100ADC_ = nullptr;
  meEEDigiSimRatiogt100ADC_ = nullptr;
}

EcalDigisValidation::~EcalDigisValidation() {}

void EcalDigisValidation::dqmBeginRun(edm::Run const&, edm::EventSetup const& c) { checkCalibrations(c); }

void EcalDigisValidation::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  Char_t histo[200];

  ibooker.setCurrentFolder("EcalDigisV/EcalDigiTask");

  sprintf(histo, "EcalDigiTask Gun Momentum");
  meGunEnergy_ = ibooker.book1D(histo, histo, 100, 0., 1000.);

  sprintf(histo, "EcalDigiTask Gun Eta");
  meGunEta_ = ibooker.book1D(histo, histo, 700, -3.5, 3.5);

  sprintf(histo, "EcalDigiTask Gun Phi");
  meGunPhi_ = ibooker.book1D(histo, histo, 360, 0., 360.);

  sprintf(histo, "EcalDigiTask Barrel maximum Digi over Sim ratio");
  meEBDigiSimRatio_ = ibooker.book1D(histo, histo, 100, 0., 2.);

  sprintf(histo, "EcalDigiTask Endcap maximum Digi over Sim ratio");
  meEEDigiSimRatio_ = ibooker.book1D(histo, histo, 100, 0., 2.);

  sprintf(histo, "EcalDigiTask Barrel maximum Digi over Sim ratio gt 10 ADC");
  meEBDigiSimRatiogt10ADC_ = ibooker.book1D(histo, histo, 100, 0., 2.);

  sprintf(histo, "EcalDigiTask Endcap maximum Digi over Sim ratio gt 20 ADC");
  meEEDigiSimRatiogt20ADC_ = ibooker.book1D(histo, histo, 100, 0., 2.);

  sprintf(histo, "EcalDigiTask Barrel maximum Digi over Sim ratio gt 100 ADC");
  meEBDigiSimRatiogt100ADC_ = ibooker.book1D(histo, histo, 100, 0., 2.);

  sprintf(histo, "EcalDigiTask Endcap maximum Digi over Sim ratio gt 100 ADC");
  meEEDigiSimRatiogt100ADC_ = ibooker.book1D(histo, histo, 100, 0., 2.);
}

void EcalDigisValidation::analyze(edm::Event const& e, edm::EventSetup const& c) {
  edm::LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  std::vector<SimTrack> theSimTracks;
  std::vector<SimVertex> theSimVertexes;

  edm::Handle<edm::HepMCProduct> MCEvt;
  edm::Handle<edm::SimTrackContainer> SimTk;
  edm::Handle<edm::SimVertexContainer> SimVtx;
  edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;
  edm::Handle<EBDigiCollection> EcalDigiEB;
  edm::Handle<EEDigiCollection> EcalDigiEE;
  edm::Handle<ESDigiCollection> EcalDigiES;

  bool skipMC = false;
  e.getByToken(HepMCToken_, MCEvt);
  if (!MCEvt.isValid()) {
    skipMC = true;
  }
  e.getByToken(g4TkInfoToken_, SimTk);
  e.getByToken(g4VtxInfoToken_, SimVtx);

  const EBDigiCollection* EBdigis = nullptr;
  const EEDigiCollection* EEdigis = nullptr;
  const ESDigiCollection* ESdigis = nullptr;

  bool isBarrel = true;
  e.getByToken(EBdigiCollectionToken_, EcalDigiEB);
  if (EcalDigiEB.isValid()) {
    EBdigis = EcalDigiEB.product();
    LogDebug("DigiInfo") << "total # EBdigis: " << EBdigis->size();
    if (EBdigis->empty())
      isBarrel = false;
  } else {
    isBarrel = false;
  }

  bool isEndcap = true;
  e.getByToken(EEdigiCollectionToken_, EcalDigiEE);
  if (EcalDigiEE.isValid()) {
    EEdigis = EcalDigiEE.product();
    LogDebug("DigiInfo") << "total # EEdigis: " << EEdigis->size();
    if (EEdigis->empty())
      isEndcap = false;
  } else {
    isEndcap = false;
  }

  bool isPreshower = true;
  e.getByToken(ESdigiCollectionToken_, EcalDigiES);
  if (EcalDigiES.isValid()) {
    ESdigis = EcalDigiES.product();
    LogDebug("DigiInfo") << "total # ESdigis: " << ESdigis->size();
    if (ESdigis->empty())
      isPreshower = false;
  } else {
    isPreshower = false;
  }

  theSimTracks.insert(theSimTracks.end(), SimTk->begin(), SimTk->end());
  theSimVertexes.insert(theSimVertexes.end(), SimVtx->begin(), SimVtx->end());

  if (!skipMC) {
    double theGunEnergy = 0.;
    for (HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin();
         p != MCEvt->GetEvent()->particles_end();
         ++p) {
      theGunEnergy = (*p)->momentum().e();
      double htheta = (*p)->momentum().theta();
      double heta = -log(tan(htheta * 0.5));
      double hphi = (*p)->momentum().phi();
      hphi = (hphi >= 0) ? hphi : hphi + 2 * M_PI;
      hphi = hphi / M_PI * 180.;
      LogDebug("EventInfo") << "Particle gun type form MC = " << abs((*p)->pdg_id()) << "\n"
                            << "Energy = " << (*p)->momentum().e() << " Eta = " << heta << " Phi = " << hphi;

      if (meGunEnergy_)
        meGunEnergy_->Fill(theGunEnergy);
      if (meGunEta_)
        meGunEta_->Fill(heta);
      if (meGunPhi_)
        meGunPhi_->Fill(hphi);
    }
  }

  int nvtx = 0;
  for (std::vector<SimVertex>::iterator isimvtx = theSimVertexes.begin(); isimvtx != theSimVertexes.end(); ++isimvtx) {
    LogDebug("EventInfo") << " Vertex index = " << nvtx << " event Id = " << isimvtx->eventId().rawId() << "\n"
                          << " vertex dump: " << *isimvtx;
    ++nvtx;
  }

  int ntrk = 0;
  for (std::vector<SimTrack>::iterator isimtrk = theSimTracks.begin(); isimtrk != theSimTracks.end(); ++isimtrk) {
    LogDebug("EventInfo") << " Track index = " << ntrk << " track Id = " << isimtrk->trackId()
                          << " event Id = " << isimtrk->eventId().rawId() << "\n"
                          << " track dump: " << *isimtrk;
    ++ntrk;
  }

  // BARREL

  // loop over simHits

  if (isBarrel) {
    e.getByToken(crossingFramePCaloHitEBToken_, crossingFrame);
    const MixCollection<PCaloHit> barrelHits(crossingFrame.product());

    MapType ebSimMap;
    for (auto const& iHit : barrelHits) {
      EBDetId ebid = EBDetId(iHit.id());

      LogDebug("HitInfo") << " CaloHit " << iHit.getName() << "\n"
                          << " DetID = " << iHit.id() << " EBDetId = " << ebid.ieta() << " " << ebid.iphi() << "\n"
                          << " Time = " << iHit.time() << " Event id. = " << iHit.eventId().rawId() << "\n"
                          << " Track Id = " << iHit.geantTrackId() << "\n"
                          << " Energy = " << iHit.energy();

      uint32_t crystid = ebid.rawId();
      ebSimMap[crystid] += iHit.energy();
    }

    // loop over Digis

    const EBDigiCollection* barrelDigi = EcalDigiEB.product();

    std::vector<double> ebAnalogSignal;
    std::vector<double> ebADCCounts;
    std::vector<double> ebADCGains;
    ebAnalogSignal.reserve(EBDataFrame::MAXSAMPLES);
    ebADCCounts.reserve(EBDataFrame::MAXSAMPLES);
    ebADCGains.reserve(EBDataFrame::MAXSAMPLES);

    for (unsigned int digis = 0; digis < EcalDigiEB->size(); ++digis) {
      EBDataFrame ebdf = (*barrelDigi)[digis];
      int nrSamples = ebdf.size();

      EBDetId ebid = ebdf.id();

      double Emax = 0.;
      int Pmax = 0;
      double pedestalPreSampleAnalog = 0.;

      for (int sample = 0; sample < nrSamples; ++sample) {
        ebAnalogSignal[sample] = 0.;
        ebADCCounts[sample] = 0.;
        ebADCGains[sample] = -1.;
      }

      for (int sample = 0; sample < nrSamples; ++sample) {
        EcalMGPASample mySample = ebdf[sample];

        ebADCCounts[sample] = (mySample.adc());
        ebADCGains[sample] = (mySample.gainId());
        ebAnalogSignal[sample] = (ebADCCounts[sample] * gainConv_[(int)ebADCGains[sample]] * barrelADCtoGeV_);
        if (Emax < ebAnalogSignal[sample]) {
          Emax = ebAnalogSignal[sample];
          Pmax = sample;
        }
        if (sample < 3) {
          pedestalPreSampleAnalog += ebADCCounts[sample] * gainConv_[(int)ebADCGains[sample]] * barrelADCtoGeV_;
        }
        LogDebug("DigiInfo") << "EB sample " << sample << " ADC counts = " << ebADCCounts[sample]
                             << " Gain Id = " << ebADCGains[sample] << " Analog eq = " << ebAnalogSignal[sample];
      }

      pedestalPreSampleAnalog /= 3.;
      double Erec = Emax - pedestalPreSampleAnalog * gainConv_[(int)ebADCGains[Pmax]];

      if (ebSimMap[ebid.rawId()] != 0.) {
        LogDebug("DigiInfo") << " Digi / Hit = " << Erec << " / " << ebSimMap[ebid.rawId()] << " gainConv "
                             << gainConv_[(int)ebADCGains[Pmax]];
        if (meEBDigiSimRatio_)
          meEBDigiSimRatio_->Fill(Erec / ebSimMap[ebid.rawId()]);
        if (Erec > 10. * barrelADCtoGeV_ && meEBDigiSimRatiogt10ADC_)
          meEBDigiSimRatiogt10ADC_->Fill(Erec / ebSimMap[ebid.rawId()]);
        if (Erec > 100. * barrelADCtoGeV_ && meEBDigiSimRatiogt100ADC_)
          meEBDigiSimRatiogt100ADC_->Fill(Erec / ebSimMap[ebid.rawId()]);
      }
    }
  }

  // ENDCAP

  // loop over simHits

  if (isEndcap) {
    e.getByToken(crossingFramePCaloHitEEToken_, crossingFrame);
    const MixCollection<PCaloHit> endcapHits(crossingFrame.product());

    MapType eeSimMap;
    for (auto const& iHit : endcapHits) {
      EEDetId eeid = EEDetId(iHit.id());

      LogDebug("HitInfo") << " CaloHit " << iHit.getName() << "\n"
                          << " DetID = " << iHit.id() << " EEDetId side = " << eeid.zside() << " = " << eeid.ix() << " "
                          << eeid.iy() << "\n"
                          << " Time = " << iHit.time() << " Event id. = " << iHit.eventId().rawId() << "\n"
                          << " Track Id = " << iHit.geantTrackId() << "\n"
                          << " Energy = " << iHit.energy();

      uint32_t crystid = eeid.rawId();
      eeSimMap[crystid] += iHit.energy();
    }

    // loop over Digis

    const EEDigiCollection* endcapDigi = EcalDigiEE.product();

    std::vector<double> eeAnalogSignal;
    std::vector<double> eeADCCounts;
    std::vector<double> eeADCGains;
    eeAnalogSignal.reserve(EEDataFrame::MAXSAMPLES);
    eeADCCounts.reserve(EEDataFrame::MAXSAMPLES);
    eeADCGains.reserve(EEDataFrame::MAXSAMPLES);

    for (unsigned int digis = 0; digis < EcalDigiEE->size(); ++digis) {
      EEDataFrame eedf = (*endcapDigi)[digis];
      int nrSamples = eedf.size();

      EEDetId eeid = eedf.id();

      double Emax = 0.;
      int Pmax = 0;
      double pedestalPreSampleAnalog = 0.;

      for (int sample = 0; sample < nrSamples; ++sample) {
        eeAnalogSignal[sample] = 0.;
        eeADCCounts[sample] = 0.;
        eeADCGains[sample] = -1.;
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
          pedestalPreSampleAnalog += eeADCCounts[sample] * gainConv_[(int)eeADCGains[sample]] * endcapADCtoGeV_;
        }
        LogDebug("DigiInfo") << "EE sample " << sample << " ADC counts = " << eeADCCounts[sample]
                             << " Gain Id = " << eeADCGains[sample] << " Analog eq = " << eeAnalogSignal[sample];
      }
      pedestalPreSampleAnalog /= 3.;
      double Erec = Emax - pedestalPreSampleAnalog * gainConv_[(int)eeADCGains[Pmax]];

      if (eeSimMap[eeid.rawId()] != 0.) {
        LogDebug("DigiInfo") << " Digi / Hit = " << Erec << " / " << eeSimMap[eeid.rawId()] << " gainConv "
                             << gainConv_[(int)eeADCGains[Pmax]];
        if (meEEDigiSimRatio_)
          meEEDigiSimRatio_->Fill(Erec / eeSimMap[eeid.rawId()]);
        if (Erec > 20. * endcapADCtoGeV_ && meEEDigiSimRatiogt20ADC_)
          meEEDigiSimRatiogt20ADC_->Fill(Erec / eeSimMap[eeid.rawId()]);
        if (Erec > 100. * endcapADCtoGeV_ && meEEDigiSimRatiogt100ADC_)
          meEEDigiSimRatiogt100ADC_->Fill(Erec / eeSimMap[eeid.rawId()]);
      }
    }
  }

  if (isPreshower) {
    e.getByToken(crossingFramePCaloHitESToken_, crossingFrame);
    const MixCollection<PCaloHit> preshowerHits(crossingFrame.product());
    for (auto const& iHit : preshowerHits) {
      ESDetId esid = ESDetId(iHit.id());

      LogDebug("HitInfo") << " CaloHit " << iHit.getName() << "\n"
                          << " DetID = " << iHit.id() << "ESDetId: z side " << esid.zside() << "  plane "
                          << esid.plane() << esid.six() << ',' << esid.siy() << ':' << esid.strip() << "\n"
                          << " Time = " << iHit.time() << " Event id. = " << iHit.eventId().rawId() << "\n"
                          << " Track Id = " << iHit.geantTrackId() << "\n"
                          << " Energy = " << iHit.energy();
    }
  }
}

void EcalDigisValidation::checkCalibrations(edm::EventSetup const& eventSetup) {
  // ADC -> GeV Scale
  const EcalADCToGeVConstant* agc = &eventSetup.getData(pAgc);

  EcalMGPAGainRatio* defaultRatios = new EcalMGPAGainRatio();

  gainConv_[1] = 1.;
  gainConv_[2] = defaultRatios->gain12Over6();
  gainConv_[3] = gainConv_[2] * (defaultRatios->gain6Over1());
  gainConv_[0] = gainConv_[2] * (defaultRatios->gain6Over1());  // saturated channels

  LogDebug("EcalDigi") << " Gains conversions: "
                       << "\n"
                       << " g1 = " << gainConv_[1] << "\n"
                       << " g2 = " << gainConv_[2] << "\n"
                       << " g3 = " << gainConv_[3];
  LogDebug("EcalDigi") << " Gains conversions: "
                       << "\n"
                       << " saturation = " << gainConv_[0];

  delete defaultRatios;

  const double barrelADCtoGeV_ = agc->getEBValue();
  LogDebug("EcalDigi") << " Barrel GeV/ADC = " << barrelADCtoGeV_;
  const double endcapADCtoGeV_ = agc->getEEValue();
  LogDebug("EcalDigi") << " Endcap GeV/ADC = " << endcapADCtoGeV_;
}
