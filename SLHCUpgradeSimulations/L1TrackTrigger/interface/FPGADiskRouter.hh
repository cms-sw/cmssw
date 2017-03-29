//This class implementes the layer router
#ifndef FPGADISKROUTER_H
#define FPGADISKROUTER_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGADiskRouter:public FPGAProcessBase{

public:

  FPGADiskRouter(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    FPGAStubDisk* tmp=dynamic_cast<FPGAStubDisk*>(memory);
    assert(tmp!=0);
    if (output=="stuboutD1") D1_=tmp;
    if (output=="stuboutD2") D2_=tmp;
    if (output=="stuboutD3") D3_=tmp;
    if (output=="stuboutD4") D4_=tmp;
    if (output=="stuboutD5") D5_=tmp;
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
      int disk=stub.first->disk().value();
      //cout << "disk = "<<disk<<endl;
      assert(fabs(disk)>=1);
      assert(fabs(disk)<=5);
      if (fabs(disk)==1) D1_->addStub(stub);
      if (fabs(disk)==2) D2_->addStub(stub);
      if (fabs(disk)==3) D3_->addStub(stub);
      if (fabs(disk)==4) D4_->addStub(stub);
      if (fabs(disk)==5) D5_->addStub(stub);
    }
  }

private:
  
  FPGAInputLink* inputlink_;

  FPGAStubDisk* D1_;
  FPGAStubDisk* D2_;
  FPGAStubDisk* D3_;
  FPGAStubDisk* D4_;
  FPGAStubDisk* D5_;
  

};

#endif

