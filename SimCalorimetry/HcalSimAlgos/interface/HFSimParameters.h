#ifndef HcalSimAlgos_HFSimParameters_h
#define HcalSimAlgos_HFSimParameters_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

class HFSimParameters : public CaloSimParameters {
public:
  HFSimParameters(double simHitToPhotoelectrons,
                  double photoelectronsToAnalog,
                  double samplingFactor,
                  double timePhase,
                  bool syncPhase);
  HFSimParameters(const edm::ParameterSet& p);

  ~HFSimParameters() override {}

  void setDbService(const HcalDbService* service) { theDbService = service; }

  double photoelectronsToAnalog(const DetId& detId) const override;

  double fCtoGeV(const DetId& detId) const;

  double samplingFactor() const;

private:
  const HcalDbService* theDbService;
  double theSamplingFactor;
};

#endif
