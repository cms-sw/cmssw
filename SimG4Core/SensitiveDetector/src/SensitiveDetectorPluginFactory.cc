#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

SensitiveDetectorPluginFactory SensitiveDetectorPluginFactory::s_instance;

SensitiveDetectorPluginFactory::SensitiveDetectorPluginFactory () : 
  Base ("CMS Simulation SensitiveDetectorFactory") {}
SensitiveDetectorPluginFactory*SensitiveDetectorPluginFactory::get (){
  return &s_instance; 
}
