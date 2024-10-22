#ifndef EcalSimAlgos_EcalHitResponse_h
#define EcalSimAlgos_EcalHitResponse_h

#include "CalibFormats/CaloObjects/interface/CaloTSamplesBase.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ComponentShapeCollection.h"

#include <unordered_map>
#include <vector>

typedef unsigned long long TimeValue_t;

class CaloVShape;
class CaloVSimParameterMap;
class CaloVHitCorrection;
class CaloVHitFilter;
class CaloSimParameters;
class CaloSubdetectorGeometry;
class CaloVPECorrection;
namespace CLHEP {
  class HepRandomEngine;
}

class EcalHitResponse {
public:
  typedef CaloTSamplesBase<float> EcalSamples;

  typedef std::vector<unsigned int> VecInd;

  typedef std::unordered_map<uint32_t, double> CalibCache;

  EcalHitResponse(const CaloVSimParameterMap* parameterMap, const CaloVShape* shape);

  virtual ~EcalHitResponse();

  const float kSamplePeriod = ecalPh1::Samp_Period;

  void setBunchRange(int minBunch, int maxBunch);

  void setGeometry(const CaloSubdetectorGeometry* geometry);

  void setPhaseShift(double phaseShift);

  void setHitFilter(const CaloVHitFilter* filter);

  void setHitCorrection(const CaloVHitCorrection* hitCorrection);

  void setPECorrection(const CaloVPECorrection* peCorrection);

  void setEventTime(const edm::TimeValue_t& iTime);

  void setLaserConstants(const EcalLaserDbService* laser, bool& useLCcorrection);

  void add(const EcalSamples* pSam);

  virtual void add(const PCaloHit& hit, CLHEP::HepRandomEngine*);

  virtual void add(const CaloSamples& hit);

  virtual void initializeHits();

  virtual void finalizeHits();

  virtual void run(MixCollection<PCaloHit>& hits, CLHEP::HepRandomEngine*);

  virtual unsigned int samplesSize() const = 0;

  virtual EcalSamples* operator[](unsigned int i) = 0;

  virtual const EcalSamples* operator[](unsigned int i) const = 0;

  const EcalSamples* findDetId(const DetId& detId) const;

  bool withinBunchRange(int bunchCrossing) const;

protected:
  virtual unsigned int samplesSizeAll() const = 0;

  virtual EcalSamples* vSam(unsigned int i) = 0;

  virtual EcalSamples* vSamAll(unsigned int i) = 0;

  virtual const EcalSamples* vSamAll(unsigned int i) const = 0;

  virtual void putAnalogSignal(const PCaloHit& inputHit, CLHEP::HepRandomEngine*);

  double findLaserConstant(const DetId& detId) const;

  EcalSamples* findSignal(const DetId& detId);

  double analogSignalAmplitude(const DetId& id, double energy, CLHEP::HepRandomEngine*);

  double timeOfFlight(const DetId& detId) const;

  double phaseShift() const;

  void blankOutUsedSamples();

  const CaloSimParameters* params(const DetId& detId) const;

  const CaloVShape* shape() const;

  const CaloSubdetectorGeometry* geometry() const;

  int minBunch() const;

  int maxBunch() const;

  VecInd& index();

  const VecInd& index() const;

  const CaloVHitFilter* hitFilter() const;

  const CaloVSimParameterMap* m_parameterMap;
  const CaloVShape* m_shape;
  const CaloVHitCorrection* m_hitCorrection;
  const CaloVPECorrection* m_PECorrection;
  const CaloVHitFilter* m_hitFilter;
  const CaloSubdetectorGeometry* m_geometry;
  const EcalLaserDbService* m_lasercals;

private:
  int m_minBunch;
  int m_maxBunch;
  double m_phaseShift;

  edm::TimeValue_t m_iTime;
  bool m_useLCcorrection;
  CalibCache m_laserCalibCache;

  VecInd m_index;
};

#endif
