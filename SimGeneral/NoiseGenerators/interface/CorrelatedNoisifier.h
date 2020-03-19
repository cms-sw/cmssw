#ifndef SimAlgos_CorrelatedNoisifier_h
#define SimAlgos_CorrelatedNoisifier_h

/**
   \class CorrelatedNoisifier

   \brief adds noise to the given frame.

Takes input correlation matrix C and from it creates
an upper-triangular matrix H such that H*Htranspose = C.

Algorithm taken from
http://cg.scs.carleton.ca/~luc/chapter_eleven.pdf, p 564-567
Uses a Cholesky decomposition

Any input array f is "noisified" with f += H*r
where r is an array of random numbers

The above matrix multiplication is expedited in the
trivial cases of a purely diagonal or identity correlation matrix.
*/

#include "DataFormats/Math/interface/Error.h"
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

template <class M>
class CorrelatedNoisifier {
public:
  typedef std::vector<double> VecDou;

  CorrelatedNoisifier(const M &symCorMat);  // correlation matrix

  CorrelatedNoisifier(int *dummy, const M &cholDecMat);  // decomposition matrix

  virtual ~CorrelatedNoisifier();

  void resetCorrelationMatrix(const M &symCorMat);

  void resetCholDecompMatrix(const M &cholMat);

  template <class T>
  void noisify(T &frame,  // applies random noise to frame
               CLHEP::HepRandomEngine *,
               const VecDou *rangau = nullptr) const;  // use these

  const M &cholMat() const;  // return decomposition

  const VecDou &vecgau() const;

private:
  static const double k_precision;  // precision to which 0 & 1 are compared

  void init(const M &symCorMat);

  void initChol();

  bool computeDecomposition(const M &symCorMat);

  bool checkDecomposition(const M &symCorMat, M &HHtDiff) const;

  void checkOffDiagonal(const M &symCorMat);

  mutable VecDou m_vecgau;

  bool m_isDiagonal;
  bool m_isIdentity;

  M m_H;
};

#endif
