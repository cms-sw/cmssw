#ifndef HcalSimAlgos_HcalTDCParameters_h
#define HcalSimAlgos_HcalTDCParameters_h

class HcalTDCParameters {

public:
  HcalTDCParameters() : nbits_(6), nbins_(50) {}

  int nbits() const {return nbits_;}
  int nbins() const {return nbins_;}
  float deltaT() const {return 25./nbins();}

  int alreadyTransitionCode() const { return 62; }
  int noTransitionCode() const { return 63; }
  int unlockedCode() const { return 61; }

private:
  int nbits_;
  int nbins_;
};

#endif

