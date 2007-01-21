#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimGeneral/NoiseGenerators/interface/NoiseStatistics.h"
#include "CommonTools/Statistics/interface/AutocorrelationAnalyzer.h"

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
 HepSymMatrix input (10);
  for (int k = 0; k < 10; k++) {
    for (int kk = k; kk < 10; kk++) {
      input[k][kk] =
        kk == k ? 5
        : kk == k+1 ? -0.2
        : kk == k+2 ? -0.1
        : 0.;
    }
  }
  std::cout << std::endl << "Initial correlations:" << std::endl << input;

  CorrelatedNoisifier noisifier(input);

  AutocorrelationAnalyzer analyzer(10);

  int nTotal = 4000000;
  for (int i=0; i<nTotal; i++) {
    BoringSignal samples(10);
    noisifier.noisify(samples);
    analyzer.analyze(samples);
  }

  std::cout << analyzer << std::endl;
  HepSymMatrix input2 (10, 0);
  for (int k = 0; k < 10; k++) {
    for (int kk = k; kk < 10; kk++) {
      for (int ix = 0; ix < 10; ix++) {
        input2 [k][kk] += input[k][ix]*input[ix][kk];
      }
    }
  }
  std::cout << std::endl << "SQUARE of initial matrix:" << std::endl << input2;
}


