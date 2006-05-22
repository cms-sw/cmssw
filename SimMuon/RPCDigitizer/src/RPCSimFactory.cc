#include "SimMuon/RPCDigitizer/src/RPCSimFactory.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSimple.h"

RPCSimFactory::RPCSimFactory() :
  seal::PluginFactory<RPCSim*(const edm::ParameterSet&)>("RPCSimFactory")
{
}

RPCSimFactory::~RPCSimFactory()
{
}

RPCSimFactory* 
RPCSimFactory::get()
{
  return &factory;
}

RPCSimFactory RPCSimFactory::factory;
