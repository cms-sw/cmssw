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
  double threshold_currentTDC() const { return threshold_currentTDC_; }

private:
  const HcalDbService* theDbService;
  double theSamplingFactor;
  double threshold_currentTDC_;
};

#endif
