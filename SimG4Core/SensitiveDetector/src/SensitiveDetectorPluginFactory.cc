#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

SensitiveDetectorPluginFactory SensitiveDetectorPluginFactory::s_instance;

SensitiveDetectorPluginFactory::SensitiveDetectorPluginFactory () : 
  seal::PluginFactory<SensitiveDetector *(std::string)> ("SensitiveDetectorFactory")
{}
SensitiveDetectorPluginFactory*SensitiveDetectorPluginFactory::get (){
  return &s_instance; 
}
