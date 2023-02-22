#ifndef EcalSimAlgos_EcalTimeMapDigitizer_h
#define EcalSimAlgos_EcalTimeMapDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface analogToDigital(const CaloSamples &, Digi &);
*/

#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "CalibFormats/CaloObjects/interface/CaloTSamplesBase.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ComponentShapeCollection.h"

class CaloSubdetectorGeometry;

class EcalTimeMapDigitizer {
public:
  struct time_average {
    static const unsigned short time_average_capacity = 10;  // this corresponds to the number of BX
    static const unsigned short waveform_capacity = EcalTimeDigi::WAVEFORMSAMPLES;  // this will give a waveform with
    static constexpr double waveform_granularity = 1.;                              // a granularity of 1ns

    const DetId id;
    float average_time[time_average_capacity];
    unsigned int nhits[time_average_capacity];
    float tot_energy[time_average_capacity];
    float waveform[waveform_capacity];

    time_average(const DetId& myId) : id(myId) {
      for (unsigned int i(0); i < time_average_capacity; ++i) {
        average_time[i] = 0;
        nhits[i] = 0;
        tot_energy[i] = 0;
      }
      for (unsigned int i(0); i < waveform_capacity; ++i) {
        waveform[i] = 0;
      }
    };

    void calculateAverage() {
      for (unsigned int i(0); i < time_average_capacity; ++i) {
        if (nhits[i] > 0)
          average_time[i] = average_time[i] / tot_energy[i];
        else
          average_time[i] = 0;
      }
    };

    void setZero() {
      for (unsigned int i(0); i < time_average_capacity; ++i) {
        average_time[i] = 0;
        nhits[i] = 0;
        tot_energy[i] = 0;
      }
      for (unsigned int i(0); i < waveform_capacity; ++i) {
        waveform[i] = 0;
      }
    };

    bool zero() {
      for (unsigned int i(0); i < time_average_capacity; ++i) {
        if (nhits[i] > 0)
          return false;
      }
      return true;
    };
  };

  typedef time_average TimeSamples;

  typedef EcalTimeDigi Digi;

  typedef std::vector<unsigned int> VecInd;

  explicit EcalTimeMapDigitizer(EcalSubdetector myDet, ComponentShapeCollection* componentShapes);

  virtual ~EcalTimeMapDigitizer();

  void add(const std::vector<PCaloHit>& hits, int bunchCrossing);

  void setGeometry(const CaloSubdetectorGeometry* geometry);

  void setEventSetup(const edm::EventSetup& eventSetup);

  void initializeMap();

  void run(EcalTimeDigiCollection& output);

  void finalizeHits();

  inline void setTimeLayerId(const int& layerId) { m_timeLayerId = layerId; };

  int getTimeLayerId() { return m_timeLayerId; };

  EcalSubdetector subdetector() { return m_subDet; };

  int minBunch() const;

  int maxBunch() const;

  void blankOutUsedSamples();  // blank out previously used elements

  /*   const CaloVHitFilter* hitFilter() const ; */

private:
  TimeSamples* findSignal(const DetId& detId);

  VecInd& index();

  const VecInd& index() const;

  unsigned int samplesSize() const;

  unsigned int samplesSizeAll() const;

  const TimeSamples* operator[](unsigned int i) const;

  TimeSamples* operator[](unsigned int i);

  TimeSamples* vSam(unsigned int i);

  TimeSamples* vSamAll(unsigned int i);

  const TimeSamples* vSamAll(unsigned int i) const;

  EcalSubdetector m_subDet;
  //time difference between bunches

  static const int BUNCHSPACE = 25;

  static const float MIN_ENERGY_THRESHOLD;  //50 KeV threshold to consider a valid hit in the timing detector

  static const int m_minBunch = -4;
  static const int m_maxBunch = 5;

  int m_timeLayerId;

  ComponentShapeCollection* m_ComponentShapes;

  const CaloSubdetectorGeometry* m_geometry;

  double timeOfFlight(const DetId& detId, int layer) const;

  const ComponentShapeCollection* shapes() const;

  std::vector<TimeSamples> m_vSam;

  VecInd m_index;
};

#endif
