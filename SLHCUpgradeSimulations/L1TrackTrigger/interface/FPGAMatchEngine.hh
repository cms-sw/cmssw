//This class implementes the tracklet engine
#ifndef FPGAMATCHENGINE_H
#define FPGAMATCHENGINE_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAMatchEngine:public FPGAProcessBase{

public:

  FPGAMatchEngine(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="matchout") {
      FPGACandidateMatch* tmp=dynamic_cast<FPGACandidateMatch*>(memory);
      assert(tmp!=0);
      candmatches_=tmp;
      return;
    }
    assert(0);

  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="vmstubin") {
      FPGAVMStubs* tmp=dynamic_cast<FPGAVMStubs*>(memory);
      assert(tmp!=0);
      vmstubs_=tmp;
      return;
    }
    if (input=="vmprojin") {
      FPGAVMProjections* tmp=dynamic_cast<FPGAVMProjections*>(memory);
      assert(tmp!=0);
      vmprojs_=tmp;
      return;
    }
    cout << "Could not find input : "<<input<<endl;
    assert(0);
  }

  void execute() {

    unsigned int count=0;

    for(unsigned int i=0;i<vmstubs_->nStubs();i++){
      std::pair<FPGAStub*,L1TStub*> stub=vmstubs_->getStub(i);
      for(unsigned int j=0;j<vmprojs_->nTracklets();j++){
	FPGATracklet* proj=vmprojs_->getFPGATracklet(j);
	//cout << "Adding match in "<<getName()<<endl;
	candmatches_->addMatch(proj,stub);
	count++;
	if (count>=NMAXME) break;
      }
      if (count>=NMAXME) break;
    }

    if (writeME) {
      static ofstream out("matchengine.txt");
      out << getName()<<" "<<count<<endl;
    }

  }


private:

  FPGAVMStubs* vmstubs_;
  FPGAVMProjections* vmprojs_;

  FPGACandidateMatch* candmatches_;

 
};

#endif
