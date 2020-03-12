#ifndef EcalSimAlgos_ESDigitizer_h
#define EcalSimAlgos_ESDigitizer_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"

namespace CLHEP {
  class RandGeneral;
  class HepRandomEngine;
}  // namespace CLHEP

#include <vector>

class ESDigitizer : public EcalTDigitizer<ESDigitizerTraits> {
public:
  typedef ESDigitizerTraits::ElectronicsSim ElectronicsSim;

  ESDigitizer(EcalHitResponse* hitResponse, ElectronicsSim* electronicsSim, bool addNoise);

  ~ESDigitizer() override;

  void run(ESDigiCollection& output, CLHEP::HepRandomEngine*) override;

  void setDetIds(const std::vector<DetId>& detIds);

  void setGain(const int gain);

private:
  void createNoisyList(std::vector<DetId>& abThreshCh, CLHEP::HepRandomEngine*);

  const std::vector<DetId>* m_detIds;
  CLHEP::RandGeneral* m_ranGeneral;
  int m_ESGain;
  double m_histoBin;
  double m_histoInf;
  double m_histoWid;
  double m_meanNoisy;

  class Triplet {
  public:
    Triplet() : first(0), second(0), third(0) {}
    Triplet(uint32_t a0, uint32_t a1, uint32_t a2) : first(a0), second(a1), third(a2) {}
    ~Triplet(){};
    uint32_t first;
    uint32_t second;
    uint32_t third;
  };

  std::vector<Triplet> m_trip;
};

#endif
