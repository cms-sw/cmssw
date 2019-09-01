
#include <iostream>

#include "SimTracker/TrackHistory/interface/VertexCategories.h"

const char *const VertexCategories::Names[] = {"Fake",
                                               "SignalEvent",
                                               "BWeakDecay",
                                               "CWeakDecay",
                                               "TauDecay",
                                               "KsDecay",
                                               "LambdaDecay",
                                               "JpsiDecay",
                                               "XiDecay",
                                               "OmegaDecay",
                                               "SigmaPlusDecay",
                                               "SigmaMinusDecay",
                                               "LongLivedDecay",
                                               "KnownProcess",
                                               "UndefinedProcess",
                                               "UnknownProcess",
                                               "PrimaryProcess",
                                               "HadronicProcess",
                                               "DecayProcess",
                                               "ComptonProcess",
                                               "AnnihilationProcess",
                                               "EIoniProcess",
                                               "HIoniProcess",
                                               "MuIoniProcess",
                                               "PhotonProcess",
                                               "MuPairProdProcess",
                                               "ConversionsProcess",
                                               "EBremProcess",
                                               "SynchrotronRadiationProcess",
                                               "MuBremProcess",
                                               "MuNuclProcess",
                                               "PrimaryVertex",
                                               "SecondaryVertex",
                                               "TertiaryVertex",
                                               "Unknown"};

void VertexCategories::unknownVertex() {
  // Check for all flags down
  for (std::size_t index = 0; index < flags_.size() - 1; ++index)
    if (flags_[index])
      return;
  // If all of them are down then it is a unkown track.
  flags_[Unknown] = true;
}

std::ostream &operator<<(std::ostream &os, VertexCategories const &classifier) {
  bool init = true;

  const VertexCategories::Flags &flags = classifier.flags();

  // Print out the classification for the track
  for (std::size_t index = 0; index < flags.size(); ++index) {
    if (flags[index]) {
      if (init) {
        os << VertexCategories::Names[index];
        init = false;
      } else
        os << "::" << VertexCategories::Names[index];
    }
  }
  os << std::endl;

  return os;
}
