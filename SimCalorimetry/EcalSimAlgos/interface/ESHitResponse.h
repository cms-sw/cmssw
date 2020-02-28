#ifndef EcalSimAlgos_ESHitResponse_h
#define EcalSimAlgos_ESHitResponse_h

#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalHitResponse.h"

class ESHitResponse : public EcalHitResponse {
public:
  typedef CaloTSamples<float, 3> ESSamples;

  ESHitResponse(const CaloVSimParameterMap* parameterMap, const CaloVShape* shape);

  ~ESHitResponse() override;

  virtual bool keepBlank() const { return false; }

  unsigned int samplesSize() const override;

  EcalSamples* operator[](unsigned int i) override;

  const EcalSamples* operator[](unsigned int i) const override;

protected:
  unsigned int samplesSizeAll() const override;

  EcalSamples* vSamAll(unsigned int i) override;

  const EcalSamples* vSamAll(unsigned int i) const override;

  EcalSamples* vSam(unsigned int i) override;

private:
  std::vector<ESSamples> m_vSam;
};
#endif
