#ifndef SimDataFormats_GeneratorProducts_GenEventInfoProduct3_h
#define SimDataFormats_GeneratorProducts_GenEventInfoProduct3_h

#include <vector>
#include <memory>

#include "SimDataFormats/GeneratorProducts/interface/PdfInfo.h"

namespace HepMC3 {
  class GenEvent;
}  // namespace HepMC3

/** \class GenEventInfoProduct3
 *
 */

class GenEventInfoProduct3 {
public:
  GenEventInfoProduct3();
  GenEventInfoProduct3(const HepMC3::GenEvent *evt);
  GenEventInfoProduct3(const GenEventInfoProduct3 &other);
  GenEventInfoProduct3(GenEventInfoProduct3 &&other);
  virtual ~GenEventInfoProduct3();

  GenEventInfoProduct3 &operator=(const GenEventInfoProduct3 &other);
  GenEventInfoProduct3 &operator=(GenEventInfoProduct3 &&other);

  typedef gen::PdfInfo PDF;

  // getters

  std::vector<double> &weights() { return weights_; }
  const std::vector<double> &weights() const { return weights_; }

  double weight() const { return weights_.empty() ? 1.0 : weights_[0]; }

  double weightProduct() const;

  unsigned int signalProcessID() const { return signalProcessID_; }

  double qScale() const { return qScale_; }
  double alphaQCD() const { return alphaQCD_; }
  double alphaQED() const { return alphaQED_; }

  const PDF *pdf() const { return pdf_.get(); }
  bool hasPDF() const { return pdf() != nullptr; }

  const std::vector<double> &binningValues() const { return binningValues_; }
  bool hasBinningValues() const { return !binningValues_.empty(); }

  const std::vector<float> &DJRValues() const { return DJRValues_; }
  bool hasDJRValues() const { return !DJRValues_.empty(); }

  int nMEPartons() const { return nMEPartons_; }

  int nMEPartonsFiltered() const { return nMEPartonsFiltered_; }

  // setters

  void setWeights(const std::vector<double> &weights) { weights_ = weights; }

  void setSignalProcessID(unsigned int procID) { signalProcessID_ = procID; }

  void setScales(double q = -1., double qcd = -1., double qed = -1.) { qScale_ = q, alphaQCD_ = qcd, alphaQED_ = qed; }

  void setPDF(const PDF *pdf) { pdf_.reset(pdf ? new PDF(*pdf) : nullptr); }

  void setBinningValues(const std::vector<double> &values) { binningValues_ = values; }

  void setDJR(const std::vector<float> &values) { DJRValues_ = values; }

  void setNMEPartons(int n) { nMEPartons_ = n; }

  void setNMEPartonsFiltered(int n) { nMEPartonsFiltered_ = n; }

private:
  // HepMC3::GenEvent provides a list of weights
  std::vector<double> weights_;

  // generator-dependent process ID
  unsigned int signalProcessID_;

  // information about scales
  double qScale_;
  double alphaQCD_, alphaQED_;

  // optional PDF info
  std::unique_ptr<PDF> pdf_;

  // If event was produced in bis, this contains
  // the values that were used to define which
  // bin the event belongs in
  // This replaces the genEventScale, which only
  // corresponds to Pythia pthat.  The RunInfo
  // will contain the information what physical
  // quantity these values actually belong to
  std::vector<double> binningValues_;
  std::vector<float> DJRValues_;
  int nMEPartons_;
  int nMEPartonsFiltered_;
};

#endif  // SimDataFormats_GeneratorProducts_GenEventInfoProduct3_h
