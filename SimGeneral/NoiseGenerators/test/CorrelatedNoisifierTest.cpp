#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "CommonTools/Statistics/interface/AutocorrelationAnalyzer.h"
#include "CLHEP/Random/JamesRandom.h"



class BoringSignal
{
public:
  BoringSignal(int size)
    : theVector(size, 0.)
    {
    }

  double & operator[](int i) { return theVector[i]; }
  const double & operator[](int i) const  { return theVector[i]; }
  int size() const {return theVector.size();}
private:
  std::vector<double> theVector;
};


int main()
{
  math::ErrorD<10>::type input;
  for (int k = 0; k < 10; k++) {
    for (int kk = k; kk < 10; kk++) {
      input(k,kk) =
        kk == k ? 1
        : kk == k+1 ? 0.67
        : kk == k+2 ? 0.53
        : kk == k+3 ? 0.44
        : kk == k+4 ? 0.39
        : kk == k+5 ? 0.36
        : kk == k+6 ? 0.38
        : kk == k+7 ? 0.35
        : kk == k+8 ? 0.36
        : kk == k+9 ? 0.32
        : 0.;
    }
  }

  std::cout << std::endl << "Initial correlations:" << "\n" << input << std::endl;;
  CLHEP::HepJamesRandom  engine;
  CorrelatedNoisifier<math::ErrorD<10>::type> noisifier(input, engine);
  AutocorrelationAnalyzer analyzer(10);

  int nTotal = 10000;
  for (int i=0; i<nTotal; i++) {
    BoringSignal samples(10);
    noisifier.noisify(samples);
    analyzer.analyze(samples);
  }

  std::cout << analyzer << std::endl;
  ROOT::Math::SMatrixIdentity a;
  math::ErrorD<10>::type input2(a);
  for (int k = 0; k < 10; k++) {
    for (int kk = k; kk < 10; kk++) {
      for (int ix = 0; ix < 10; ix++) {
        input2(k,kk)  += input(k,ix)*input(ix,kk);
      }
    }
  }
  std::cout << std::endl << "SQUARE of initial matrix:" << std::endl << input2;
}


