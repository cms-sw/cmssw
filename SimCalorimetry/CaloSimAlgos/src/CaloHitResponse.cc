#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapes.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>

CaloHitResponse::CaloHitResponse(const CaloVSimParameterMap *parametersMap, const CaloVShape *shape)
    : theAnalogSignalMap(),
      theParameterMap(parametersMap),
      theShapes(nullptr),
      theShape(shape),
      theHitCorrection(nullptr),
      thePECorrection(nullptr),
      theHitFilter(nullptr),
      theGeometry(nullptr),
      theMinBunch(-10),
      theMaxBunch(10),
      thePhaseShift_(1.),
      storePrecise(false),
      ignoreTime(false) {}

CaloHitResponse::CaloHitResponse(const CaloVSimParameterMap *parametersMap, const CaloShapes *shapes)
    : theAnalogSignalMap(),
      theParameterMap(parametersMap),
      theShapes(shapes),
      theShape(nullptr),
      theHitCorrection(nullptr),
      thePECorrection(nullptr),
      theHitFilter(nullptr),
      theGeometry(nullptr),
      theMinBunch(-10),
      theMaxBunch(10),
      thePhaseShift_(1.),
      storePrecise(false),
      ignoreTime(false) {}

CaloHitResponse::~CaloHitResponse() {}

void CaloHitResponse::setBunchRange(int minBunch, int maxBunch) {
  theMinBunch = minBunch;
  theMaxBunch = maxBunch;
}

void CaloHitResponse::run(const MixCollection<PCaloHit> &hits, CLHEP::HepRandomEngine *engine) {
  for (MixCollection<PCaloHit>::MixItr hitItr = hits.begin(); hitItr != hits.end(); ++hitItr) {
    if (withinBunchRange(hitItr.bunch())) {
      add(*hitItr, engine);
    }  // loop over hits
  }
}

void CaloHitResponse::add(const PCaloHit &hit, CLHEP::HepRandomEngine *engine) {
  // check the hit time makes sense
  if (edm::isNotFinite(hit.time())) {
    return;
  }

  // maybe it's not from this subdetector
  if (theHitFilter == nullptr || theHitFilter->accepts(hit)) {
    LogDebug("CaloHitResponse") << hit;
    CaloSamples signal(makeAnalogSignal(hit, engine));
    bool keep(keepBlank());  // here we  check for blank signal if not keeping them
    if (!keep) {
      const unsigned int size(signal.size());
      if (0 != size) {
        for (unsigned int i(0); i != size; ++i) {
          keep = keep || signal[i] > 1.e-7;
        }
      }
    }

    if (keep)
      add(signal);
  }
}

void CaloHitResponse::add(const CaloSamples &signal) {
  DetId id(signal.id());
  CaloSamples *oldSignal = findSignal(id);
  if (oldSignal == nullptr) {
    theAnalogSignalMap[id] = signal;

  } else {
    (*oldSignal) += signal;
  }
}

CaloSamples CaloHitResponse::makeAnalogSignal(const PCaloHit &hit, CLHEP::HepRandomEngine *engine) const {
  DetId detId(hit.id());
  const CaloSimParameters &parameters = theParameterMap->simParameters(detId);
  double signal = analogSignalAmplitude(detId, hit.energy(), parameters, engine);

  double time = hit.time();
  double tof = timeOfFlight(detId);
  if (ignoreTime)
    time = tof;
  if (theHitCorrection != nullptr) {
    time += theHitCorrection->delay(hit, engine);
  }
  double jitter = time - tof;

  const CaloVShape *shape = theShape;
  if (!shape) {
    shape = theShapes->shape(detId, storePrecise);
  }
  // assume bins count from zero, go for center of bin
  const double tzero = (shape->timeToRise() + parameters.timePhase() - jitter -
                        BUNCHSPACE * (parameters.binOfMaximum() - thePhaseShift_));
  double binTime = tzero;

  CaloSamples result(makeBlankSignal(detId));

  if (storePrecise) {
    result.resetPrecise();
    int sampleBin(0);
    // use 1ns binning for precise sample
    for (int bin = 0; bin < result.size() * BUNCHSPACE; bin++) {
      sampleBin = bin / BUNCHSPACE;
      double pulseBit = (*shape)(binTime)*signal;
      result[sampleBin] += pulseBit;
      result.preciseAtMod(bin) += pulseBit;
      binTime += 1.0;
    }
  } else {
    for (int bin = 0; bin < result.size(); bin++) {
      result[bin] += (*shape)(binTime)*signal;
      binTime += BUNCHSPACE;
    }
  }
  return result;
}

double CaloHitResponse::analogSignalAmplitude(const DetId &detId,
                                              float energy,
                                              const CaloSimParameters &parameters,
                                              CLHEP::HepRandomEngine *engine) const {
  // OK, the "energy" in the hit could be a real energy, deposited energy,
  // or pe count.  This factor converts to photoelectrons
  // GMA Smeared in photon production it self
  double npe = energy * parameters.simHitToPhotoelectrons(detId);
  // do we need to doPoisson statistics for the photoelectrons?
  if (parameters.doPhotostatistics()) {
    npe = CLHEP::RandPoissonQ::shoot(engine, npe);
  }
  if (thePECorrection)
    npe = thePECorrection->correctPE(detId, npe, engine);
  return npe;
}

CaloSamples *CaloHitResponse::findSignal(const DetId &detId) {
  CaloSamples *result = nullptr;
  AnalogSignalMap::iterator signalItr = theAnalogSignalMap.find(detId);
  if (signalItr == theAnalogSignalMap.end()) {
    result = nullptr;
  } else {
    result = &(signalItr->second);
  }
  return result;
}

CaloSamples CaloHitResponse::makeBlankSignal(const DetId &detId) const {
  const CaloSimParameters &parameters = theParameterMap->simParameters(detId);
  int preciseSize(storePrecise ? parameters.readoutFrameSize() * BUNCHSPACE : 0);
  CaloSamples result(detId, parameters.readoutFrameSize(), preciseSize);
  result.setPresamples(parameters.binOfMaximum() - 1);
  if (storePrecise)
    result.setPrecise(result.presamples() * BUNCHSPACE, 1.0);
  return result;
}

double CaloHitResponse::timeOfFlight(const DetId &detId) const {
  // not going to assume there's one of these per subdetector.
  // Take the whole CaloGeometry and find the right subdet
  double result = 0.;
  if (theGeometry == nullptr) {
    edm::LogWarning("CaloHitResponse") << "No Calo Geometry set, so no time of flight correction";
  } else {
    auto cellGeometry = theGeometry->getSubdetectorGeometry(detId)->getGeometry(detId);
    if (cellGeometry == nullptr) {
      edm::LogWarning("CaloHitResponse") << "No Calo cell found for ID" << detId.rawId()
                                         << " so no time-of-flight subtraction will be done";
    } else {
      double distance = cellGeometry->getPosition().mag();
      result = distance * cm / c_light;  // Units of c_light: mm/ns
    }
  }
  return result;
}
