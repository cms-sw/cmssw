#ifndef HcalSimAlgos_HcalSimParameters_h
#define HcalSimAlgos_HcalSimParameters_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

typedef std::vector<std::pair<double,double> > HcalTimeSmearSettings;

class HcalSimParameters : public CaloSimParameters
{
public:
  HcalSimParameters(double simHitToPhotoelectrons,
                    const std::vector<double> & photoelectronsToAnalog,
                    double samplingFactor, double timePhase,
                    int readoutFrameSize, int binOfMaximum,
                    bool doPhotostatistics, bool syncPhase,
                    int firstRing, const std::vector<double> & samplingFactors,
                    double sipmTau
                    );
  HcalSimParameters(const edm::ParameterSet & p);

  virtual ~HcalSimParameters() {}

  void setDbService(const HcalDbService * service);

  virtual double simHitToPhotoelectrons(const DetId & detId) const;
  virtual double photoelectronsToAnalog(const DetId & detId) const;

  double fCtoGeV(const DetId & detId) const;

  /// the ratio of actual incident energy to deposited energy
  /// in the SimHit
  virtual double samplingFactor(const DetId & detId) const;

  bool doTimeSmear() const { return doTimeSmear_; }

  double timeSmearRMS(double ampl) const;

  int pixels(const DetId & detId) const;
  bool doSiPMSmearing() const { return theSiPMSmearing; }

  double sipmTau() const { return theSiPMTau; }
  double sipmDarkCurrentuA(const DetId & detId) const;
  double sipmCrossTalk(const DetId & detId) const;
  std::vector<float> sipmNonlinearity(const DetId & detId) const;
  unsigned int signalShape(const DetId & detId) const;

  friend class HcalSimParameterMap;

private:
  void defaultTimeSmearing();
  const HcalDbService * theDbService;
  const HcalSiPMCharacteristics* theSiPMcharacteristics;
  int theFirstRing;
  std::vector<double> theSamplingFactors;
  std::vector<double> thePE2fCByRing;
  bool theSiPMSmearing;
  bool doTimeSmear_;
  HcalTimeSmearSettings theSmearSettings;
  double theSiPMTau;
};

#endif
  
