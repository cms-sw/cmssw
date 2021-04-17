#include "SimCalorimetry/EcalSimAlgos/interface/EcalHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVPECorrection.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

EcalHitResponse::EcalHitResponse(const CaloVSimParameterMap* parameterMap, const CaloVShape* shape)
    : m_parameterMap(parameterMap),
      m_shape(shape),
      m_hitCorrection(nullptr),
      m_PECorrection(nullptr),
      m_hitFilter(nullptr),
      m_geometry(nullptr),
      m_lasercals(nullptr),
      m_minBunch(-32),
      m_maxBunch(10),
      m_phaseShift(1),
      m_iTime(0),
      m_useLCcorrection(false) {}

EcalHitResponse::~EcalHitResponse() {}

const CaloSimParameters* EcalHitResponse::params(const DetId& detId) const {
  assert(nullptr != m_parameterMap);
  return &m_parameterMap->simParameters(detId);
}

const CaloVShape* EcalHitResponse::shape() const {
  assert(nullptr != m_shape);
  return m_shape;
}

const CaloSubdetectorGeometry* EcalHitResponse::geometry() const {
  assert(nullptr != m_geometry);
  return m_geometry;
}

void EcalHitResponse::setBunchRange(int minBunch, int maxBunch) {
  m_minBunch = minBunch;
  m_maxBunch = maxBunch;
}

void EcalHitResponse::setGeometry(const CaloSubdetectorGeometry* geometry) { m_geometry = geometry; }

void EcalHitResponse::setPhaseShift(double phaseShift) { m_phaseShift = phaseShift; }

double EcalHitResponse::phaseShift() const { return m_phaseShift; }

void EcalHitResponse::setHitFilter(const CaloVHitFilter* filter) { m_hitFilter = filter; }

void EcalHitResponse::setHitCorrection(const CaloVHitCorrection* hitCorrection) { m_hitCorrection = hitCorrection; }

void EcalHitResponse::setPECorrection(const CaloVPECorrection* peCorrection) { m_PECorrection = peCorrection; }

void EcalHitResponse::setEventTime(const edm::TimeValue_t& iTime) {
  m_iTime = iTime;
  //clear the laser cache for each event time
  CalibCache().swap(m_laserCalibCache);
}

void EcalHitResponse::setLaserConstants(const EcalLaserDbService* laser, bool& useLCcorrection) {
  m_lasercals = laser;
  m_useLCcorrection = useLCcorrection;
}

void EcalHitResponse::blankOutUsedSamples()  // blank out previously used elements
{
  const unsigned int size(m_index.size());

  for (unsigned int i(0); i != size; ++i) {
    vSamAll(m_index[i])->setZero();
  }
  m_index.erase(m_index.begin(),  // done and make ready to start over
                m_index.end());
}

void EcalHitResponse::add(const PCaloHit& hit, CLHEP::HepRandomEngine* engine) {
  if (!edm::isNotFinite(hit.time()) && (nullptr == m_hitFilter || m_hitFilter->accepts(hit))) {
    putAnalogSignal(hit, engine);
  }
}

void EcalHitResponse::add(const CaloSamples& hit) {
  const DetId detId(hit.id());

  EcalSamples& result(*findSignal(detId));

  const int rsize(result.size());

  if (rsize != hit.size()) {
    throw cms::Exception("EcalDigitization") << "CaloSamples and EcalSamples have different sizes. Type Mismatach";
  }

  for (int bin(0); bin != rsize; ++bin) {
    result[bin] += hit[bin];
  }
}

bool EcalHitResponse::withinBunchRange(int bunchCrossing) const {
  return (m_minBunch <= bunchCrossing && m_maxBunch >= bunchCrossing);
}

void EcalHitResponse::initializeHits() { blankOutUsedSamples(); }

void EcalHitResponse::finalizeHits() {}

void EcalHitResponse::run(MixCollection<PCaloHit>& hits, CLHEP::HepRandomEngine* engine) {
  blankOutUsedSamples();

  for (MixCollection<PCaloHit>::MixItr hitItr(hits.begin()); hitItr != hits.end(); ++hitItr) {
    const PCaloHit& hit(*hitItr);
    const int bunch(hitItr.bunch());
    if (withinBunchRange(bunch) && !edm::isNotFinite(hit.time()) &&
        (nullptr == m_hitFilter || m_hitFilter->accepts(hit)))
      putAnalogSignal(hit, engine);
  }
}

void EcalHitResponse::putAnalogSignal(const PCaloHit& hit, CLHEP::HepRandomEngine* engine) {
  const DetId detId(hit.id());

  const CaloSimParameters* parameters(params(detId));

  const double signal(analogSignalAmplitude(detId, hit.energy(), engine));

  double time = hit.time();

  if (m_hitCorrection) {
    time += m_hitCorrection->delay(hit, engine);
  }

  const double jitter(time - timeOfFlight(detId));

  const double tzero = (shape()->timeToRise() + parameters->timePhase() - jitter -
                        kSamplePeriod * (parameters->binOfMaximum() - m_phaseShift));
  double binTime(tzero);

  EcalSamples& result(*findSignal(detId));

  const unsigned int rsize(result.size());

  for (unsigned int bin(0); bin != rsize; ++bin) {
    result[bin] += (*shape())(binTime)*signal;
    binTime += kSamplePeriod;
  }
}

double EcalHitResponse::findLaserConstant(const DetId& detId) const {
  const edm::Timestamp& evtTimeStamp = edm::Timestamp(m_iTime);
  return (m_lasercals->getLaserCorrection(detId, evtTimeStamp));
}

EcalHitResponse::EcalSamples* EcalHitResponse::findSignal(const DetId& detId) {
  const unsigned int di(CaloGenericDetId(detId).denseIndex());
  EcalSamples* result(vSamAll(di));
  if (result->zero())
    m_index.push_back(di);
  return result;
}

double EcalHitResponse::analogSignalAmplitude(const DetId& detId, double energy, CLHEP::HepRandomEngine* engine) {
  const CaloSimParameters& parameters(*params(detId));

  // OK, the "energy" in the hit could be a real energy, deposited energy,
  // or pe count.  This factor converts to photoelectrons

  double lasercalib = 1.;
  if (m_useLCcorrection == true && detId.subdetId() != 3) {
    auto cache = m_laserCalibCache.find(detId);
    if (cache != m_laserCalibCache.end()) {
      lasercalib = cache->second;
    } else {
      lasercalib = 1.0 / findLaserConstant(detId);
      m_laserCalibCache.emplace(detId, lasercalib);
    }
  }

  double npe(energy * lasercalib * parameters.simHitToPhotoelectrons(detId));

  // do we need to doPoisson statistics for the photoelectrons?
  if (parameters.doPhotostatistics()) {
    npe = CLHEP::RandPoissonQ::shoot(engine, npe);
  }
  if (nullptr != m_PECorrection)
    npe = m_PECorrection->correctPE(detId, npe, engine);

  return npe;
}

double EcalHitResponse::timeOfFlight(const DetId& detId) const {
  auto cellGeometry(geometry()->getGeometry(detId));
  assert(nullptr != cellGeometry);
  return cellGeometry->getPosition().mag() * cm / c_light;  // Units of c_light: mm/ns
}

void EcalHitResponse::add(const EcalSamples* pSam) {
  EcalSamples& sam(*findSignal(pSam->id()));
  sam += (*pSam);
}

int EcalHitResponse::minBunch() const { return m_minBunch; }

int EcalHitResponse::maxBunch() const { return m_maxBunch; }

EcalHitResponse::VecInd& EcalHitResponse::index() { return m_index; }

const EcalHitResponse::VecInd& EcalHitResponse::index() const { return m_index; }

const CaloVHitFilter* EcalHitResponse::hitFilter() const { return m_hitFilter; }

const EcalHitResponse::EcalSamples* EcalHitResponse::findDetId(const DetId& detId) const {
  const unsigned int di(CaloGenericDetId(detId).denseIndex());
  return vSamAll(di);
}
