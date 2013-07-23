#ifndef RandomEngine_RandomEngineState_h
#define RandomEngine_RandomEngineState_h
// -*- C++ -*-
//
// Package:     RandomEngine
// Class  :     RandomEngineState
// 
/**\class RandomEngineState RandomEngineState.h SimDataFormats/RandomEngine/interface/RandomEngineState.h

 Description: Holds the state of a CLHEP random number engine
and the label of the module it is associated with.

 Usage:  This should only be used by the Random Number Generator
service.

*/
//
// Original Author:  W. David Dagenhart, Fermilab
//         Created:  Tue Oct  3 09:56:36 CDT 2006
// $Id: RandomEngineState.h,v 1.1 2006/10/17 20:45:52 wdd Exp $
//

#include <vector>
#include <string>
#include <boost/cstdint.hpp>


class RandomEngineState {

  public:

  RandomEngineState();

  ~RandomEngineState();

  const std::string& getLabel() const { return label_; }
  const std::vector<uint32_t>& getState() const { return state_; }
  const std::vector<uint32_t>& getSeed() const { return seed_; }

  void setLabel(const std::string& value) { label_ = value; }
  void setState(const std::vector<uint32_t>& value) { state_ = value; }
  void setSeed(const std::vector<uint32_t>& value) { seed_ = value; }

  void clearSeedVector() { seed_.clear(); }
  void reserveSeedVector(std::vector<uint32_t>::size_type n) { seed_.reserve(n); }
  void push_back_seedVector(uint32_t v) { seed_.push_back(v); }

  void clearStateVector() { state_.clear(); }
  void reserveStateVector(std::vector<uint32_t>::size_type n) { state_.reserve(n); }
  void push_back_stateVector(uint32_t v) { state_.push_back(v); }

  bool operator<(RandomEngineState const& rhs) { return label_ < rhs.label_; }

  private:

  std::string label_;
  std::vector<uint32_t> state_;
  std::vector<uint32_t> seed_;
};

#endif
