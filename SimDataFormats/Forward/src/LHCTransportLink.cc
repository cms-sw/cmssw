#include "SimDataFormats/Forward/interface/LHCTransportLink.h"
#include <iomanip>

void LHCTransportLink::fill(int& afterHector, int& beforeHector) {
  afterHector_ = afterHector;
  beforeHector_ = beforeHector;
}

int LHCTransportLink::beforeHector() const { return beforeHector_; }

int LHCTransportLink::afterHector() const { return afterHector_; }

void LHCTransportLink::clear() {
  afterHector_ = 0;
  beforeHector_ = 0;
}

std::ostream& operator<<(std::ostream& o, const LHCTransportLink& t) {
  o << "before Hector " << t.beforeHector() << " after Hector " << t.afterHector();
  return o;
}
