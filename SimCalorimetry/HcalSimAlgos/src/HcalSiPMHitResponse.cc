#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"

#include "CLHEP/Random/RandPoissonQ.h"

#include <cmath>
#include <list>

HcalSiPMHitResponse::HcalSiPMHitResponse(const CaloVSimParameterMap* parameterMap,
                                         const CaloShapes* shapes,
                                         bool PreMix1,
                                         bool HighFidelity)
    : CaloHitResponse(parameterMap, shapes),
      theSiPM(),
      PreMixDigis(PreMix1),
      HighFidelityPreMix(HighFidelity),
      nbins((PreMixDigis and HighFidelityPreMix) ? 1 : BUNCHSPACE * HcalPulseShapes::invDeltaTSiPM_),
      dt(HcalPulseShapes::deltaTSiPM_),
      invdt(HcalPulseShapes::invDeltaTSiPM_) {
  //fill shape map
  shapeMap.emplace(HcalShapes::ZECOTEK, HcalShapes::ZECOTEK);
  shapeMap.emplace(HcalShapes::HAMAMATSU, HcalShapes::HAMAMATSU);
  shapeMap.emplace(HcalShapes::HE2017, HcalShapes::HE2017);
  shapeMap.emplace(HcalShapes::HE2018, HcalShapes::HE2018);
}

HcalSiPMHitResponse::~HcalSiPMHitResponse() {}

void HcalSiPMHitResponse::initializeHits() { precisionTimedPhotons.clear(); }

int HcalSiPMHitResponse::getReadoutFrameSize(const DetId& id) const {
  const CaloSimParameters& parameters = theParameterMap->simParameters(id);
  int readoutFrameSize = parameters.readoutFrameSize();
  if (PreMixDigis and HighFidelityPreMix) {
    //preserve fidelity of time info
    readoutFrameSize *= BUNCHSPACE * HcalPulseShapes::invDeltaTSiPM_;
  }
  return readoutFrameSize;
}

void HcalSiPMHitResponse::finalizeHits(CLHEP::HepRandomEngine* engine) {
  //do not add PE noise for initial premix
  if (!PreMixDigis)
    addPEnoise(engine);

  photonTimeMap::iterator channelPhotons;
  for (channelPhotons = precisionTimedPhotons.begin(); channelPhotons != precisionTimedPhotons.end();
       ++channelPhotons) {
    CaloSamples signal(makeSiPMSignal(channelPhotons->first, channelPhotons->second, engine));
    bool keep(keepBlank());
    if (!keep) {
      const unsigned int size(signal.size());
      if (0 != size) {
        for (unsigned int i(0); i != size; ++i) {
          keep = keep || signal[i] > 1.e-7;
        }
      }
    }

    LogDebug("HcalSiPMHitResponse") << HcalDetId(signal.id()) << ' ' << signal;

    //if we don't want to keep precise info at the end
    if (!HighFidelityPreMix) {
      signal.setPreciseSize(0);
    }

    if (keep)
      CaloHitResponse::add(signal);
  }
}

//used for premixing - premixed CaloSamples have fine time binning
void HcalSiPMHitResponse::add(const CaloSamples& signal) {
  if (!HighFidelityPreMix) {
    CaloHitResponse::add(signal);
    return;
  }
  DetId id(signal.id());
  int photonTimeHistSize = nbins * getReadoutFrameSize(id);
  assert(photonTimeHistSize == signal.size());
  if (precisionTimedPhotons.find(id) == precisionTimedPhotons.end()) {
    precisionTimedPhotons.insert(std::pair<DetId, photonTimeHist>(id, photonTimeHist(photonTimeHistSize, 0)));
  }
  for (int i = 0; i < signal.size(); ++i) {
    unsigned int photons(signal[i] + 0.5);
    precisionTimedPhotons[id][i] += photons;
  }
}

void HcalSiPMHitResponse::add(const PCaloHit& hit, CLHEP::HepRandomEngine* engine) {
  if (!edm::isNotFinite(hit.time()) && ((theHitFilter == nullptr) || (theHitFilter->accepts(hit)))) {
    HcalDetId id(hit.id());
    const HcalSimParameters& pars = dynamic_cast<const HcalSimParameters&>(theParameterMap->simParameters(id));
    //divide out mean of crosstalk distribution 1/(1-lambda) = multiply by (1-lambda)
    double signal(analogSignalAmplitude(id, hit.energy(), pars, engine) * (1 - pars.sipmCrossTalk(id)));
    unsigned int photons(signal + 0.5);
    double tof(timeOfFlight(id));
    double time(hit.time());
    if (ignoreTime)
      time = tof;

    if (photons > 0)
      if (precisionTimedPhotons.find(id) == precisionTimedPhotons.end()) {
        precisionTimedPhotons.insert(
            std::pair<DetId, photonTimeHist>(id, photonTimeHist(nbins * getReadoutFrameSize(id), 0)));
      }

    LogDebug("HcalSiPMHitResponse") << id;
    LogDebug("HcalSiPMHitResponse") << " fCtoGeV: " << pars.fCtoGeV(id)
                                    << " samplingFactor: " << pars.samplingFactor(id)
                                    << " photoelectronsToAnalog: " << pars.photoelectronsToAnalog(id)
                                    << " simHitToPhotoelectrons: " << pars.simHitToPhotoelectrons(id);
    LogDebug("HcalSiPMHitResponse") << " energy: " << hit.energy() << " photons: " << photons << " time: " << time;
    LogDebug("HcalSiPMHitResponse") << " timePhase: " << pars.timePhase() << " tof: " << tof
                                    << " binOfMaximum: " << pars.binOfMaximum() << " phaseShift: " << thePhaseShift_;
    double tzero(0.0 + pars.timePhase() - (time - tof) - BUNCHSPACE * (pars.binOfMaximum() - thePhaseShift_));
    LogDebug("HcalSiPMHitResponse") << " tzero: " << tzero;
    double tzero_bin(-tzero * invdt);
    LogDebug("HcalSiPMHitResponse") << " corrected tzero: " << tzero_bin << '\n';
    double t_pe(0.);
    int t_bin(0);
    unsigned signalShape = pars.signalShape(id);
    for (unsigned int pe(0); pe < photons; ++pe) {
      t_pe = HcalPulseShapes::generatePhotonTime(engine, signalShape);
      t_bin = int(t_pe * invdt + tzero_bin + 0.5);
      LogDebug("HcalSiPMHitResponse") << "t_pe: " << t_pe << " t_pe + tzero: " << (t_pe + tzero_bin * dt)
                                      << " t_bin: " << t_bin << '\n';
      if ((t_bin >= 0) && (static_cast<unsigned int>(t_bin) < precisionTimedPhotons[id].size()))
        precisionTimedPhotons[id][t_bin] += 1;
    }
  }
}

void HcalSiPMHitResponse::addPEnoise(CLHEP::HepRandomEngine* engine) {
  // Add SiPM dark current noise to all cells
  for (std::vector<DetId>::const_iterator idItr = theDetIds->begin(); idItr != theDetIds->end(); ++idItr) {
    HcalDetId id(*idItr);
    const HcalSimParameters& pars = static_cast<const HcalSimParameters&>(theParameterMap->simParameters(id));

    // uA * ns / (fC/pe) = pe!
    double dc_pe_avg = pars.sipmDarkCurrentuA(id) * dt / pars.photoelectronsToAnalog(id);

    if (dc_pe_avg <= 0.)
      continue;

    int nPreciseBins = nbins * getReadoutFrameSize(id);

    unsigned int sumnoisePE(0);
    for (int tprecise(0); tprecise < nPreciseBins; ++tprecise) {
      int noisepe = CLHEP::RandPoissonQ::shoot(engine, dc_pe_avg);  // add dark current noise

      if (noisepe > 0) {
        if (precisionTimedPhotons.find(id) == precisionTimedPhotons.end()) {
          photonTimeHist photons(nPreciseBins, 0);
          photons[tprecise] = noisepe;
          precisionTimedPhotons.insert(std::pair<DetId, photonTimeHist>(id, photons));
        } else {
          precisionTimedPhotons[id][tprecise] += noisepe;
        }

        sumnoisePE += noisepe;
      }

    }  // precise time loop

    LogDebug("HcalSiPMHitResponse") << id;
    LogDebug("HcalSiPMHitResponse") << " total noise (PEs): " << sumnoisePE;

  }  // detId loop
}  // HcalSiPMHitResponse::addPEnoise()

CaloSamples HcalSiPMHitResponse::makeBlankSignal(const DetId& detId) const {
  const CaloSimParameters& parameters = theParameterMap->simParameters(detId);
  int readoutFrameSize = getReadoutFrameSize(detId);
  int preciseSize(readoutFrameSize * nbins);
  CaloSamples result(detId, readoutFrameSize, preciseSize);
  result.setPresamples(parameters.binOfMaximum() - 1);
  result.setPrecise(result.presamples() * nbins, dt);
  return result;
}

CaloSamples HcalSiPMHitResponse::makeSiPMSignal(DetId const& id,
                                                photonTimeHist const& photonTimeBins,
                                                CLHEP::HepRandomEngine* engine) {
  const HcalSimParameters& pars = static_cast<const HcalSimParameters&>(theParameterMap->simParameters(id));
  theSiPM.setNCells(pars.pixels(id));
  theSiPM.setTau(pars.sipmTau());
  theSiPM.setCrossTalk(pars.sipmCrossTalk(id));
  theSiPM.setSaturationPars(pars.sipmNonlinearity(id));

  //use to make signal
  CaloSamples signal(makeBlankSignal(id));
  int sampleBin(0), preciseBin(0);
  signal.resetPrecise();
  unsigned int pe(0);
  double hitPixels(0.), elapsedTime(0.);

  auto const& sipmPulseShape(shapeMap[pars.signalShape(id)]);

  LogDebug("HcalSiPMHitResponse") << "makeSiPMSignal for " << HcalDetId(id);

  const int nptb = photonTimeBins.size();
  double sum[nptb];
  for (auto i = 0; i < nptb; ++i)
    sum[i] = 0;
  for (int tbin(0); tbin < nptb; ++tbin) {
    pe = photonTimeBins[tbin];
    if (pe <= 0)
      continue;
    preciseBin = tbin;
    sampleBin = preciseBin / nbins;
    //skip saturation/recovery and pulse smearing for premix stage 1
    if (PreMixDigis and HighFidelityPreMix) {
      signal[sampleBin] += pe;
      signal.preciseAtMod(preciseBin) += pe;
      elapsedTime += dt;
      continue;
    }

    hitPixels = theSiPM.hitCells(engine, pe, 0., elapsedTime);
    LogDebug("HcalSiPMHitResponse") << " elapsedTime: " << elapsedTime << " sampleBin: " << sampleBin
                                    << " preciseBin: " << preciseBin << " pe: " << pe << " hitPixels: " << hitPixels;
    if (!pars.doSiPMSmearing()) {
      signal[sampleBin] += hitPixels;
      signal.preciseAtMod(preciseBin) += 0.6 * hitPixels;
      if (preciseBin > 0)
        signal.preciseAtMod(preciseBin - 1) += 0.2 * hitPixels;
      if (preciseBin < signal.preciseSize() - 1)
        signal.preciseAtMod(preciseBin + 1) += 0.2 * hitPixels;
    } else {
      // add "my" smearing to future bins...
      // this loop can vectorize....
      for (auto i = tbin; i < nptb; ++i) {
        auto itdiff = i - tbin;
        if (itdiff == sipmPulseShape.nBins())
          break;
        auto shape = sipmPulseShape[itdiff];
        auto pulseBit = shape * hitPixels;
        sum[i] += pulseBit;
        if (shape < 1.e-7 && itdiff > int(HcalPulseShapes::invDeltaTSiPM_))
          break;
      }
    }
    elapsedTime += dt;
  }
  if (pars.doSiPMSmearing())
    for (auto i = 0; i < nptb; ++i) {
      auto iSampleBin = i / nbins;
      signal[iSampleBin] += sum[i];
      signal.preciseAtMod(i) += sum[i];
    }

#ifdef EDM_ML_DEBUG
  LogDebug("HcalSiPMHitResponse") << nbins << ' ' << nptb << ' ' << HcalDetId(id);
  for (auto i = 0; i < nptb; ++i) {
    auto iSampleBin = (nbins > 1) ? i / nbins : i;
    LogDebug("HcalSiPMHitResponse") << i << ' ' << iSampleBin << ' ' << signal[iSampleBin] << ' '
                                    << signal.preciseAtMod(i);
  }
#endif

  return signal;
}

void HcalSiPMHitResponse::setDetIds(const std::vector<DetId>& detIds) { theDetIds = &detIds; }
