#include "Utilities/General/interface/GeneralVerbosity.h"
#include "Utilities/General/interface/DecoratedSB.h"
#include <iostream>

template <>
envSwitch & GeneralVerbosity::me() {
  static envSwitch l("GenUtilVerbose");
  return l;
}

template <>
envSwitch & GeneralVerbosity::silence() {
  static envSwitch l("GenUtilSilent");
  return l;
}

namespace {
  static DefDecoratedSB lsb(std::cout.rdbuf(),"","");

}

std::ostream GeneralUtilities::cout(&lsb);
