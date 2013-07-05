#ifndef SimCalorimetry_HcalSimProducers_HcalHitRelabeller_h
#define SimCalorimetry_HcalSimProducers_HcalHitRelabeller_h 1

#include <vector>
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalHitRelabeller {
public:
  HcalHitRelabeller(const edm::ParameterSet& ps);
  void process(std::vector<PCaloHit> & hcalHits);
  void setGeometry(const CaloGeometry *& theGeometry);

private:
  DetId relabel(const uint32_t testId) const;

  const CaloGeometry* theGeometry;

  std::vector<std::vector<int> > m_segmentation;
  bool                           m_CorrectPhi;
};
#endif
