#include "SimG4Core/Generators/interface/LumiMonitorFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
// using std::cout;
// using std::endl;

LumiMonitorFilter::LumiMonitorFilter() {}

LumiMonitorFilter::~LumiMonitorFilter() {}

void LumiMonitorFilter::Describe() const { edm::LogInfo("LumiMonitorFilter") << " is active "; }

bool LumiMonitorFilter::isGoodForLumiMonitor(const HepMC::GenParticle *) const { return true; }

bool LumiMonitorFilter::isGoodForLumiMonitor(const HepMC3::GenParticle *) const { return true; }
