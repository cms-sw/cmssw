#include "SimG4Core/Application/interface/RunManagerMTWorker.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

RunManagerMTWorker::RunManagerMTWorker(const edm::ParameterSet& iConfig):
  m_generator(iConfig.getParameter<edm::ParameterSet>("Generator"))
{}

RunManagerMTWorker::~RunManagerMTWorker() {}
