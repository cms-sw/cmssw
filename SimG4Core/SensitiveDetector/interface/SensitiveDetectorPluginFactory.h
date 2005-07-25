#ifndef SimG4Core_SensitiveDetector_SensitiveDetectorPluginFactory_H
#define SimG4Core_SensitiveDetector_SensitiveDetectorPluginFactory_H

# include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
# include "PluginManager/PluginFactory.h"

#include <string>


class SensitiveDetectorPluginFactory : public seal::PluginFactory<SensitiveDetector *(std::string)>{
 public:
    static  SensitiveDetectorPluginFactory* get (void);
    SensitiveDetectorPluginFactory();
 private:
    static SensitiveDetectorPluginFactory s_instance;
};


#endif
