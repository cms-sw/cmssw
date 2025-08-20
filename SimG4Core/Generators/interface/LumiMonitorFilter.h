#ifndef SimG4Core_LumiMonitorFilter_H
#define SimG4Core_LumiMonitorFilter_H

#include "HepMC/GenParticle.h"
#include "HepMC3/GenParticle.h"

class LumiMonitorFilter {
public:
  LumiMonitorFilter();
  virtual ~LumiMonitorFilter();

  void Describe() const;
  bool isGoodForLumiMonitor(const HepMC::GenParticle *) const;
  bool isGoodForLumiMonitor(const HepMC3::GenParticle *) const;

private:
};

#endif
