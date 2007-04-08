#ifndef RPCDigitizer_RPCSimFactory_h
#define RPCDigitizer_RPCSimFactory_h

/** \class RPCSimFactory
 * Factory of seal plugins for RPCDigitizer
 * \author M. Maggi -- INFN Bari
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm{
  class ParameterSet;
}


class RPCSim;
class RPCSimFactory : public seal::PluginFactory<RPCSim*(const edm::ParameterSet&)>{
 public:
  RPCSimFactory();
  virtual ~RPCSimFactory();
  static RPCSimFactory* get();
 private:
  static RPCSimFactory factory;

};

#endif
