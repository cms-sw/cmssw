#ifndef SimDataFormatsValidationFormatsMaterialAccountingCalo_h
#define SimDataFormatsValidationFormatsMaterialAccountingCalo_h

#include <string>
#include <vector>

// struct to keep material accounting information along a track
class MaterialAccountingCalo {
public:
  MaterialAccountingCalo(void) { clear(); }

  void clear(void) {
    m_eta = m_phi = 0.;
    m_stepLen.clear();
    m_radLen.clear();
    m_intLen.clear();
    m_layers.clear();
  }

  double m_eta, m_phi;
  std::vector<double> m_stepLen, m_radLen, m_intLen;
  std::vector<int> m_layers;
};

typedef std::vector<MaterialAccountingCalo> MaterialAccountingCaloCollection;

#endif  // MaterialAccountingCalo_h
