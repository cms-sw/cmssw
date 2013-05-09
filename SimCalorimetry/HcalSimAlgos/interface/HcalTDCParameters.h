#ifndef HcalSimAlgos_HcalTDCParameters_h
#define HcalSimAlgos_HcalTDCParameters_h

class HcalTDCParameters {

public:
  HcalTDCParameters() : nbits_(6) {}

  int nbits() const {return nbits_;}
  int nbins() const {return 1 << nbits_;}
  float deltaT() const {return 25./nbins();}

private:
  int nbits_;
};

#endif

