#ifndef SimG4Core_SensitiveDetector_SensitiveDetectorPluginFactory_H
#define SimG4Core_SensitiveDetector_SensitiveDetectorPluginFactory_H

# include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
# include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMaker.h"
# include "PluginManager/PluginFactory.h"

#include <string>


class SensitiveDetectorPluginFactory : public seal::PluginFactory<SensitiveDetectorMakerBase *()>{
 public:
  typedef seal::PluginFactory<SensitiveDetectorMakerBase*()> Base;
    static  SensitiveDetectorPluginFactory* get (void);
    SensitiveDetectorPluginFactory();
 private:
    static SensitiveDetectorPluginFactory s_instance;
};

#define DEFINE_SENSITIVEDETECTOR(type) \
  DEFINE_SEAL_PLUGIN(SensitiveDetectorPluginFactory, SensitiveDetectorMaker<type>, #type)
#endif
