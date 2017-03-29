//Base class for processing modules
#ifndef FPGAPROCESSBASE_H
#define FPGAPROCESSBASE_H

using namespace std;

class FPGAProcessBase{

public:

  FPGAProcessBase(string name, unsigned int iSector){
    name_=name;
    iSector_=iSector;
  }

  virtual void addOutput(FPGAMemoryBase* memory,string output)=0;

  virtual void addInput(FPGAMemoryBase* memory,string input)=0;

  string getName() const {return name_;}


protected:

  string name_;
  unsigned int iSector_;


};

#endif
