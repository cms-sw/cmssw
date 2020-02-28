#ifndef GaussNoiseProducerFP420_h
#define GaussNoiseProducerFP420_h _ 1

#include <map>

class GaussNoiseProducerFP420 {
public:
  GaussNoiseProducerFP420() {}
  ~GaussNoiseProducerFP420() {}

  void generate(int NumberOfchannels, float threshold, float noiseRMS, std::map<int, float, std::less<int>> &theMap);
};

#endif
