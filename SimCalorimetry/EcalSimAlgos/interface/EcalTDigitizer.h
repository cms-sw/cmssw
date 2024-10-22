#ifndef EcalSimAlgos_EcalTDigitizer_h
#define EcalSimAlgos_EcalTDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface analogToDigital(CLHEP::HepRandomEngine*, const CaloSamples &, Digi &);
*/

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CalibFormats/CaloObjects/interface/CaloTSamplesBase.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalBaseSignalGenerator.h"

class EcalHitResponse;
class EcalBaseSignalGenerator;

namespace CLHEP {
  class HepRandomEngine;
}

template <class Traits>
class EcalTDigitizer {
public:
  typedef typename Traits::ElectronicsSim ElectronicsSim;
  typedef typename Traits::Digi Digi;
  typedef typename Traits::DigiCollection DigiCollection;
  typedef typename Traits::EcalSamples EcalSamples;

  EcalTDigitizer(EcalHitResponse* hitResponse, ElectronicsSim* electronicsSim, bool addNoise);

  virtual ~EcalTDigitizer();

  void add(const std::vector<PCaloHit>& hits, int bunchCrossing, CLHEP::HepRandomEngine*);

  virtual void initializeHits();

  virtual void run(DigiCollection& output, CLHEP::HepRandomEngine*);

  virtual void run(MixCollection<PCaloHit>& input, DigiCollection& output) { assert(0); }

  void setNoiseSignalGenerator(EcalBaseSignalGenerator* noiseSignalGenerator);

  void addNoiseSignals();

protected:
  bool addNoise() const;

  const EcalHitResponse* hitResponse() const;

  const ElectronicsSim* elecSim() const;

private:
  EcalHitResponse* m_hitResponse;
  ElectronicsSim* m_electronicsSim;
  bool m_addNoise;

  EcalBaseSignalGenerator* theNoiseSignalGenerator;
};

#endif
