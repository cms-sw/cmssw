// -*- C++ -*-
#include "SimDataFormats/RandomEngine/interface/RandomEngineStates.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>

namespace edm {

  RandomEngineStates::RandomEngineStates() {
  }

  RandomEngineStates::~RandomEngineStates() {
  }

  void
  RandomEngineStates::getRandomEngineStates(std::vector<RandomEngineState> & states) const {

    // First check for data corruption so that the following code
    // does not encounter out of range errors.
    bool corrupt = false;

    if (moduleLabels_.size() != seedLengths_.size()) corrupt = true;
    if (moduleLabels_.size() != stateLengths_.size()) corrupt = true;

    unsigned int sum = 0U;
    for (std::vector<unsigned>::const_iterator i = seedLengths_.begin(),
	                                    iEnd = seedLengths_.end();
         i != iEnd; ++i) {
      sum += *i;
    }
    if (sum != seedVectors_.size()) corrupt = true;

    sum = 0U;
    for (std::vector<unsigned>::const_iterator i = stateLengths_.begin(),
	                                    iEnd = stateLengths_.end();
         i != iEnd; ++i) {
      sum += *i;
    }
    if (sum != stateVectors_.size()) corrupt = true;

    if (corrupt) {
      throw edm::Exception(errors::EventCorruption)
        << "RandomEngineStates data is corrupted.\n";
    }

    // Done with error checks.  Now do the work.

    std::vector<unsigned>::const_iterator seedLength = seedLengths_.begin();
    std::vector<unsigned>::const_iterator seedBegin = seedVectors_.begin();
    std::vector<unsigned>::const_iterator seedEnd = seedVectors_.begin();

    std::vector<unsigned>::const_iterator stateLength = stateLengths_.begin();
    std::vector<unsigned>::const_iterator stateBegin = stateVectors_.begin();
    std::vector<unsigned>::const_iterator stateEnd = stateVectors_.begin();

    for (std::vector<std::string>::const_iterator label = moduleLabels_.begin(),
	                                       labelEnd = moduleLabels_.end();
         label != labelEnd;
         ++label, ++seedLength, ++stateLength) {

      seedBegin = seedEnd;
      seedEnd += *seedLength;

      stateBegin = stateEnd;
      stateEnd += *stateLength;

      RandomEngineState randomEngineState;
      randomEngineState.setLabel(*label);
      std::vector<RandomEngineState>::iterator state = 
        std::lower_bound(states.begin(), states.end(), randomEngineState);

      if (state != states.end() && *label == state->getLabel()) {
        if (*seedLength != state->getSeed().size() ||
            *stateLength != state->getState().size()) {
          throw edm::Exception(edm::errors::Configuration)
            << "When attempting to replay processing with the RandomNumberGeneratorService,\n"
            << "the engine type for each module must be the same in the replay configuration\n"
            << "and the original configuration.  If this is not the problem, then the data\n"
            << "is somehow corrupted or there is a bug because the vector in the data containing\n"
            << "the seeds or engine state is the incorrect size for the type of random engine.\n";
        }

        state->clearSeedVector();
        state->reserveSeedVector(*seedLength);
        for (std::vector<unsigned int>::const_iterator i = seedBegin;
             i != seedEnd; ++i) {
          state->push_back_seedVector(*i);
        }

        state->clearStateVector();
        state->reserveStateVector(*stateLength);
        for (std::vector<unsigned int>::const_iterator i = stateBegin;
             i != stateEnd; ++i) {
          state->push_back_stateVector(*i);
        }
      }
    }
  }


  void
  RandomEngineStates::setRandomEngineStates(std::vector<RandomEngineState> const& states) {

    moduleLabels_.resize(states.size());
    seedLengths_.resize(states.size());
    seedVectors_.clear();
    stateLengths_.resize(states.size());
    stateVectors_.clear();

    std::vector<std::string>::iterator label = moduleLabels_.begin();
    std::vector<unsigned>::iterator seedLength = seedLengths_.begin();
    std::vector<unsigned>::iterator stateLength = stateLengths_.begin();


    for (std::vector<RandomEngineState>::const_iterator state = states.begin(),
	                                                 iEnd = states.end();
         state != iEnd; ++state, ++label, ++seedLength, ++stateLength) {

      *label = state->getLabel();

      std::vector<uint32_t> const& seedVector = state->getSeed();
      *seedLength = seedVector.size();

      for (std::vector<uint32_t>::const_iterator j = seedVector.begin(),
                                              jEnd = seedVector.end();
           j != jEnd; ++j) {
        seedVectors_.push_back(*j);
      }

      std::vector<uint32_t> const& stateVector = state->getState();
      *stateLength = stateVector.size();

      for (std::vector<uint32_t>::const_iterator j = stateVector.begin(),
                                              jEnd = stateVector.end();
           j != jEnd; ++j) {
        stateVectors_.push_back(*j);
      }
    }
  }

  bool
  RandomEngineStates::isProductEqual(RandomEngineStates const& randomEngineStates) const {
    if (moduleLabels_ == randomEngineStates.moduleLabels_ &&
        seedLengths_ == randomEngineStates.seedLengths_ &&
        seedVectors_ == randomEngineStates.seedVectors_ &&
        stateLengths_ == randomEngineStates.stateLengths_ &&
        stateVectors_ == randomEngineStates.stateVectors_) {
      return true;
    }
    return false;
  }
}
