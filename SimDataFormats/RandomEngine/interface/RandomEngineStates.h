// -*- C++ -*-
#ifndef RandomEngine_RandomEngineStates_h
#define RandomEngine_RandomEngineStates_h

/** \class edm::RandomEngineStates

Description: Holds the states of multiple random number
engines and associated seeds and module labels.

Usage:  This should only be used by the Random Number Generator
Service.

\author W. David Dagenhart, created December 3, 2010
*/ 

#include <vector>
#include <string>

class RandomEngineState;

namespace edm {

  class RandomEngineStates {
  public:

    RandomEngineStates();
    ~RandomEngineStates();

    void getRandomEngineStates(std::vector<RandomEngineState> & states) const;
    void setRandomEngineStates(std::vector<RandomEngineState> const& states);

    bool isProductEqual(RandomEngineStates const& randomEngineStates) const;

  private:

    std::vector<std::string> moduleLabels_;

    std::vector<unsigned> seedLengths_;
    std::vector<unsigned> seedVectors_;

    std::vector<unsigned> stateLengths_;
    std::vector<unsigned> stateVectors_;
  };
}
#endif
