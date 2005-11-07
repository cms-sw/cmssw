#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

SensitiveDetectorPluginFactory SensitiveDetectorPluginFactory::s_instance;

SensitiveDetectorPluginFactory::SensitiveDetectorPluginFactory () : 
  seal::PluginFactory<SensitiveDetector *(std::string, const DDCompactView & cpv, edm::ParameterSet const & p)> ("SensitiveDetectorFactory") {}
SensitiveDetectorPluginFactory*SensitiveDetectorPluginFactory::get (){
  return &s_instance; 
}
