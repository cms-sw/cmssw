#ifndef SimG4Core_Application_RunManagerMTWorker_H
#define SimG4Core_Application_RunManagerMTWorker_H

#include "SimG4Core/Generators/interface/Generator.h"

#include <memory>

namespace edm {
  class ParameterSet;
}
class Generator;

class RunManagerMTWorker {
public:
  RunManagerMTWorker(const edm::ParameterSet& iConfig);
  ~RunManagerMTWorker();

private:
  Generator m_generator;
};

#endif
