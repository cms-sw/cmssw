#ifndef SimG4Core_LumiMonitorFilter_H
#define SimG4Core_LumiMonitorFilter_H

#include "HepMC/GenParticle.h"

class LumiMonitorFilter {
public:
  LumiMonitorFilter();
  virtual ~LumiMonitorFilter();

  void Describe() const;
  bool isGoodForLumiMonitor(const HepMC::GenParticle *) const;

private:
};

#endif
