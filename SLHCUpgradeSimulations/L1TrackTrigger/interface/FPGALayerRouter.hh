//This class implementes the layer router
#ifndef FPGALAYERROUTER_H
#define FPGALAYERROUTER_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGALayerRouter:public FPGAProcessBase{

public:

  FPGALayerRouter(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    FPGAStubLayer* tmp=dynamic_cast<FPGAStubLayer*>(memory);
    assert(tmp!=0);
    if (output=="stuboutL1") L1_=tmp;
    if (output=="stuboutL2") L2_=tmp;
    if (output=="stuboutL3") L3_=tmp;
    if (output=="stuboutL4") L4_=tmp;
    if (output=="stuboutL5") L5_=tmp;
    if (output=="stuboutL6") L6_=tmp;
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    FPGAInputLink* tmp=dynamic_cast<FPGAInputLink*>(memory);
    assert(tmp!=0);
    inputlink_=tmp;
  }

  void execute(){
    for(unsigned int i=0;i<inputlink_->nStubs();i++){
      //cout << "i nStubs : "<<i<<" "<<inputlink_->nStubs()<<endl;
      std::pair<FPGAStub*,L1TStub*> stub=inputlink_->getStub(i);
      int layer=stub.first->layer().value()+1;
      //cout << "layer = "<<layer<<endl;
      assert(layer>=1);
      assert(layer<=6);
      if (layer==1) L1_->addStub(stub);
      if (layer==2) L2_->addStub(stub);
      if (layer==3) L3_->addStub(stub);
      if (layer==4) L4_->addStub(stub);
      if (layer==5) L5_->addStub(stub);
      if (layer==6) L6_->addStub(stub);
    }
  }

private:
  
  FPGAInputLink* inputlink_;

  FPGAStubLayer* L1_;
  FPGAStubLayer* L2_;
  FPGAStubLayer* L3_;
  FPGAStubLayer* L4_;
  FPGAStubLayer* L5_;
  FPGAStubLayer* L6_;
  

};

#endif

