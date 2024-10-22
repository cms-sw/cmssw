#include "CLHEP/Random/JamesRandom.h"
#include "CommonTools/Statistics/interface/AutocorrelationAnalyzer.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.icc"

namespace CLHEP {
  class HepRandomEngine;
}

typedef math::ErrorD<10>::type MyMat;

template class CorrelatedNoisifier<MyMat>;

template void CorrelatedNoisifier<MyMat>::noisify(std::vector<double> &,
                                                  CLHEP::HepRandomEngine *,
                                                  const std::vector<double> *) const;

int main() {
  math::ErrorD<10>::type input;
  for (int k = 0; k < 10; k++) {
    for (int kk = k; kk < 10; kk++) {
      input(k, kk) = kk == k       ? 1
                     : kk == k + 1 ? 0.67
                     : kk == k + 2 ? 0.53
                     : kk == k + 3 ? 0.44
                     : kk == k + 4 ? 0.39
                     : kk == k + 5 ? 0.36
                     : kk == k + 6 ? 0.38
                     : kk == k + 7 ? 0.35
                     : kk == k + 8 ? 0.36
                     : kk == k + 9 ? 0.32
                                   : 0.;
    }
  }
  CLHEP::HepJamesRandom engine;

  typedef math::ErrorD<10>::type MatType;
  typedef CorrelatedNoisifier<MatType> Noisifier;

  MatType chol;

  for (unsigned int itry(0); itry != 2; ++itry) {
    //      Noisifier noisifier ( input, &engine ) ;
    Noisifier noisifier(0 == itry ? Noisifier(input) : Noisifier(nullptr, chol));

    if (0 == itry)
      chol = noisifier.cholMat();

    AutocorrelationAnalyzer analyzer(10);

    const unsigned int nTotal = 1000000;
    for (unsigned int i = 0; i < nTotal; ++i) {
      std::vector<double> samples(10);
      noisifier.noisify(samples, &engine);
      analyzer.analyze(samples);
    }

    double big(-999);
    math::ErrorD<10>::type ratdif;
    for (int k = 0; k < 10; k++) {
      for (int kk = k; kk < 10; kk++) {
        const double diff(input(k, kk) - analyzer.correlation(k, kk));
        ratdif(k, kk) = diff / input(k, kk);
        if (fabs(ratdif(k, kk)) > big)
          big = fabs(ratdif(k, kk));
      }
    }

    std::cout << "In " << nTotal << " trials, the biggest fractional deviation\n"
              << "of observed correlations from input correlations =" << big << std::endl;

    std::cout << std::endl
              << "Initial correlations:"
              << "\n"
              << input << std::endl;
    ;

    std::cout << analyzer << std::endl;

    std::cout << ratdif << std::endl;
    std::cout << std::endl << "\nSQUARE of initial matrix:\n" << input * input << std::endl;
  }
}
