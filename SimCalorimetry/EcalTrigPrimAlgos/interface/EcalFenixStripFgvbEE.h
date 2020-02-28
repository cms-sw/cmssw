#ifndef ECAL_FENIXSTRIP_FGVB_EE_H
#define ECAL_FENIXSTRIP_FGVB_EE_H
#include <CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h>

#include <vector>

class EEDataFrame;
// class EcalTPGFineGrainStripEE;

/**
    \class EcalFenixStripFgvbEE
    \brief calculation of Fgvb for the endcap in Fenix Strip
    *  calculates fgvb for the endcap in Fenix Strip
    *
    *
    *  input: 5X18 bits
    *  output: 1 bit
    *
    *
    */
class EcalFenixStripFgvbEE {
private:
  int threshold_fg_;
  int lut_fg_;
  int identif_;
  bool flagBadStripMiss_;

public:
  EcalFenixStripFgvbEE();
  virtual ~EcalFenixStripFgvbEE();
  void setParameters(int identif, uint32_t id, const EcalTPGFineGrainStripEE *);
  void process(std::vector<std::vector<int>> &lin_out, std::vector<int> &output);

  void setbadStripMissing(bool flag) { flagBadStripMiss_ = flag; }
  bool getMissedStripFlag() const { return flagBadStripMiss_; }
};

#endif
