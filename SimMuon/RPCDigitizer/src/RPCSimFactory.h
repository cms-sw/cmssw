#ifndef RPCDigitizer_RPCSimFactory_h
#define RPCDigitizer_RPCSimFactory_h

class RPCSim;
class RPCSimFactory{
 public:
  RPCSimFactory(){}
  virtual ~RPCSimFactory(){}
  RPCSim* rpcSim();

};

#endif
