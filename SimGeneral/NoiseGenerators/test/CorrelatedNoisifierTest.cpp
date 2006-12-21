#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimGeneral/NoiseGenerators/interface/NoiseStatistics.h"

class BoringSignal
{
public:
  BoringSignal(int size)
    : theVector(size, 0.)
    {
    }

  double & operator[](int i) { return theVector[i]; }
  int size() const {return theVector.size();}
private:
  std::vector<double> theVector;
};


int main() {
  //typedef accumulator_set<double, stats<tag::mean, tag::variance>, int > Stats
  //Stats stats0, stats1;

  HepSymMatrix matrix(2);
  matrix[0][0] = 16;
  matrix[0][1] = 4;
  matrix[1][1] = 9;

  CorrelatedNoisifier noisifier(matrix);

  NoiseStatistics stats0("Bin 0", 0., 4.);
  NoiseStatistics stats1("Bin 1", 0., 3.);

  for(int i = 0; i < 100000; ++i)
  {
    BoringSignal signal(2);
    noisifier.noisify(signal);
    stats0.addEntry(signal[0]);
    stats1.addEntry(signal[1]);
  }


}


