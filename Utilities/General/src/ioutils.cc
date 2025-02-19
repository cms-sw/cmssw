#include "Utilities/General/interface/ioutils.h"
#include "Utilities/General/interface/MutexUtils.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

bool  ato<bool>::operator () (const std::string &c) const {
  bool ret;
  std::string loc = c;
  std::transform(loc.begin(),loc.end(),loc.begin(),::tolower);
  std::stringstream in(loc.c_str());
  in >> ret;;
  if (in.fail()) {
    in.clear();
    in >> std::setiosflags(std::ios::boolalpha) >> ret;
  }
  return ret;
}

#ifndef CMS_CHAR_STREAM
#define CSTR .c_str()
#else
#define CSTR
#endif

#ifdef CMS_CHAR_STREAM //linux
toa::~toa() {
  oss().rdbuf()->freeze(0);
}
#else
toa::~toa() {
}
#endif

std::string & toa::localS() {
  static boost::thread_specific_ptr<std::string> local_s;
  if (!local_s.get()) local_s.reset(new std::string);
  return *local_s;
}

const char * toa::operator () (const int&c) const {
  oss() << c;
#ifdef CMS_CHAR_STREAM
  oss() << ends;
#endif
  localS() = oss().str();
  return localS().c_str();
}

const char * toa::operator () (bool c) const {
  oss() << std::setiosflags(std::ios::boolalpha) << c;
#ifdef CMS_CHAR_STREAM
  oss() << ends;
#endif
  localS() = oss().str();
  return localS().c_str();
}


const char * toa::operator () (const double&c) const {
  oss() << c;
#ifdef CMS_CHAR_STREAM
  oss() << ends;
#endif
  localS() = oss().str();
  return localS().c_str();
}

const char * toa::operator () (const char c[], int n) const {
  localS().assign(c,n);
  return localS().c_str();
}
