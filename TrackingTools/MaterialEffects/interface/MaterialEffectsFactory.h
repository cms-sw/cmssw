#ifndef MaterialEffectsFactory_H
#define MaterialEffectsFactory_H

#include <string>

class MaterialEffectsUpdator;
/**
 * Pseudo-factory for MaterialEffectsUpdator (components are
 * created by the factory itself).
 */
class MaterialEffectsFactory {

public:
  MaterialEffectsFactory();
  ~MaterialEffectsFactory() {}

  MaterialEffectsUpdator* constructComponent();
  MaterialEffectsUpdator* constructComponent(const float);

private:
  std::string theUpdatorName;
};

#endif

