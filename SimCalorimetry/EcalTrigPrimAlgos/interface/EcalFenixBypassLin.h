#ifndef ECAL_FENIX_BYPASS_LIN_H
#define ECAL_FENIX_BYPASS_LIN_H

#include <vector>

/**
    \class EcalFenixBypassLin
    \brief Linearisation for Tcp
    *  input: 16 bits
    *  output: 12 bits +1 going to fgvb (???)
    *
    *      ----> c un output de 13 bits, le 13 eme bit est le resultat du fvgb
   du FenixStrip
    */

class EcalFenixBypassLin {
public:
  EcalFenixBypassLin();
  virtual ~EcalFenixBypassLin();

  void process(std::vector<int> &, std::vector<int> &);
};

#endif
