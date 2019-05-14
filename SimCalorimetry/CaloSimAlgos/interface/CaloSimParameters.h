#ifndef CaloSimAlgos_CaloSimParameters_h
#define CaloSimAlgos_CaloSimParameters_h

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iosfwd>
/**

   \class CaloSimParameters

   \brief Main class for Parameters in different subdetectors.

*/
class CaloSimParameters {
public:
  // note: sampling factor not used
  CaloSimParameters(double simHitToPhotoelectrons,
                    double photoelectronsToAnalog,
                    double samplingFactor,
                    double timePhase,
                    int readoutFrameSize,
                    int binOfMaximum,
                    bool doPhotostatistics,
                    bool syncPhase = true);

  CaloSimParameters(const edm::ParameterSet &p, bool skipPe2Fc = false);

  virtual ~CaloSimParameters(){};

  /// the factor which goes from whatever units the SimHit amplitudes
  /// are in (could be deposited GeV, real GeV, or photoelectrons)
  /// and converts to photoelectrons
  /// probably should make everything virtual, but this is enough for HCAL
  double simHitToPhotoelectrons() const { return simHitToPhotoelectrons_; }
  virtual double simHitToPhotoelectrons(const DetId &) const { return simHitToPhotoelectrons_; }

  /// the factor which goes from photoelectrons to whatever gets read by ADCs
  double photoelectronsToAnalog() const { return photoelectronsToAnalog_; }
  virtual double photoelectronsToAnalog(const DetId &detId) const { return photoelectronsToAnalog_; }

  /// the adjustment you need to apply to get the signal where you want it
  double timePhase() const { return timePhase_; }

  /// for now, the LinearFrames and trhe digis will be one-to-one.
  int readoutFrameSize() const { return readoutFrameSize_; }

  int binOfMaximum() const { return binOfMaximum_; }

  /// some datamixing apps need this to be set dynamically
  void setReadoutFrameSize(int frameSize) { readoutFrameSize_ = frameSize; }
  void setBinOfMaximum(int binOfMax) { binOfMaximum_ = binOfMax; }

  /// whether or not to apply Poisson statistics to photoelectrons
  bool doPhotostatistics() const { return doPhotostatistics_; }

  /// choice of the ADC time alignment (synchronous for LHC, asynchronous for
  /// test beams)
  bool syncPhase() const { return syncPhase_; }

private:
  double simHitToPhotoelectrons_;
  double photoelectronsToAnalog_;
  double timePhase_;
  int readoutFrameSize_;
  int binOfMaximum_;
  bool doPhotostatistics_;
  bool syncPhase_;
};

std::ostream &operator<<(std::ostream &os, const CaloSimParameters &p);

#endif
