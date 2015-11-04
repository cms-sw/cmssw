//Base class for processing modules
#ifndef FPGAMEMORYBASE_H
#define FPGAMEMORYBASE_H

using namespace std;

class FPGAMemoryBase{

public:

  FPGAMemoryBase(string name, unsigned int iSector){
    name_=name;
    iSector_=iSector;
    bx_=0;
    event_=0;
  }

  virtual ~FPGAMemoryBase(){}

  string getName() const {return name_;}

  virtual void clean()=0;

protected:

  string name_;
  unsigned int iSector_;

  ofstream out_;
  int bx_;
  int event_;


};

#endif
