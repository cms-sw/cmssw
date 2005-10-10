#include "Utilities/General/interface/GenUtilVerbosity.h"
#include<iostream>

template <>
envSwitch & GenUtilVerbosity::me() {
  static envSwitch l("GenUtilVerbose");
  return l;
}

template <>
envSwitch & GenUtilVerbosity::silence() {
  static envSwitch l("GenUtilSilent");
  return l;
}

#include "Utilities/General/interface/DecoratedSB.h"
namespace {
  static DefDecoratedSB lsb(std::cout.rdbuf(),"","");

}
std::ostream GenUtil::cout(&lsb);
