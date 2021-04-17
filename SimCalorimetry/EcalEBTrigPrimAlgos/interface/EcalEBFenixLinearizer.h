#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBFenixLinearizer_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBFenixLinearizer_h

#include <DataFormats/EcalDigi/interface/EcalMGPASample.h>
#include <CondFormats/EcalObjects/interface/EcalTPGPedestals.h>
#include <CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h>
#include <CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h>

#include <vector>

/** 
   \class EcalEBFenixLinearizer
   \brief Linearisation for Fenix strip
   *  input: 16 bits  corresponding to input EBDataFrame
   *  output: 18 bits 
   *  
   */

class EcalEBFenixLinearizer {
private:
  bool famos_;
  int uncorrectedSample_;
  int gainID_;
  int base_;
  int mult_;
  int shift_;
  int strip_;
  bool init_;

  const EcalTPGLinearizationConstant *linConsts_;
  const EcalTPGPedestal *peds_;
  const EcalTPGCrystalStatusCode *badXStatus_;

  std::vector<const EcalTPGCrystalStatusCode *> vectorbadXStatus_;

  int setInput(const EcalMGPASample &RawSam);
  int process();

public:
  EcalEBFenixLinearizer(bool famos);
  virtual ~EcalEBFenixLinearizer();

  template <class T>
  void process(const T &, std::vector<int> &);
  void setParameters(uint32_t raw,
                     const EcalTPGPedestals *ecaltpPed,
                     const EcalTPGLinearizationConst *ecaltpLin,
                     const EcalTPGCrystalStatus *ecaltpBadX);
};

template <class T>
void EcalEBFenixLinearizer::process(const T &df, std::vector<int> &output_percry) {
  //We know a tower numbering is:
  // S1 S2 S3 S4 S5
  //
  // 4  5  14 15 24
  // 3  6  13 16 23
  // 2  7  12 17 22
  // 1  8  11 18 21
  // 0  9  10 19 20

  for (int i = 0; i < df.size(); i++) {
    setInput(df[i]);
    output_percry[i] = process();
  }

  return;
}

#endif
