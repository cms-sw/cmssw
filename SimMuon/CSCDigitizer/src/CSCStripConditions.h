#ifndef CSCDigitizer_CSCStripConditions_h
#define CSCDigitizer_CSCStripConditions_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"

namespace CLHEP {
  class HepRandomEngine;
}

class CSCStripConditions {
public:
  typedef math::ErrorD<8>::type CSCCorrelatedNoiseMatrix;
  typedef CorrelatedNoisifier<CSCCorrelatedNoiseMatrix> CSCCorrelatedNoisifier;
  CSCStripConditions();

  virtual ~CSCStripConditions();

  /// superimposes noise, in fC, on the signal
  void noisify(const CSCDetId &detId, CSCAnalogSignal &signal, CLHEP::HepRandomEngine *);

  virtual void initializeEvent(const edm::EventSetup &es) {}

  /// channels count from 1
  /// gain is the ratio that takes us from fC to ADC.  Nominally around 2
  virtual float gain(const CSCDetId &detId, int channel) const = 0;
  virtual float gainSigma(const CSCDetId &detId, int channel) const = 0;
  virtual float smearedGain(const CSCDetId &detId, int channel, CLHEP::HepRandomEngine *) const;

  /// in ADC counts
  virtual float pedestal(const CSCDetId &detId, int channel) const = 0;
  virtual float pedestalSigma(const CSCDetId &detId, int channel) const = 0;

  /// calculated from pedestalSigma & gain
  float analogNoise(const CSCDetId &detId, int channel) const;

  virtual void crosstalk(const CSCDetId &detId,
                         int channel,
                         double stripLength,
                         bool leftRight,
                         float &capacitive,
                         float &resistive) const = 0;

  /// is supplied layer/chamber flagged as bad? (default impl. is no)
  virtual bool isInBadChamber(const CSCDetId &id) const { return false; }

protected:
  virtual void fetchNoisifier(const CSCDetId &detId, int istrip) = 0;

  CSCCorrelatedNoisifier *theNoisifier;
};

#endif
