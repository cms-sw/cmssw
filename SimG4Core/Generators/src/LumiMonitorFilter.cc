#include "SimG4Core/Generators/interface/LumiMonitorFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HepMC/GenParticle.h"

#include "G4Event.hh"

using namespace edm;
//using std::cout;
//using std::endl;

LumiMonitorFilter::LumiMonitorFilter() 
{} 

LumiMonitorFilter::~LumiMonitorFilter() 
{}

void LumiMonitorFilter::Describe() 
{
  edm::LogInfo("LumiMonitorFilter") 
    << " is active ";
}

bool LumiMonitorFilter::isGoodForLumiMonitor(const GenParticle*) const
{
  return true;
}
