#ifndef SimCalorimetry_HcalSimProducers_HcalHitRelabeller_h
#define SimCalorimetry_HcalSimProducers_HcalHitRelabeller_h 1

#include <vector>
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalHitRelabeller {
public:
  HcalHitRelabeller(const edm::ParameterSet& ps);
  void process(std::vector<PCaloHit> & hcalHits);
  void setGeometry(const CaloGeometry *&, const HcalDDDRecConstants *&);
  DetId relabel(const uint32_t testId) const;

private:

  const CaloGeometry* theGeometry;
  const HcalDDDRecConstants* theRecNumber;
};
#endif
