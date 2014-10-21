#ifndef G4ProcessTypeEnumerator_H
#define G4ProcessTypeEnumerator_H

#include "G4VProcess.hh"

class G4ProcessTypeEnumerator {

public:

  G4ProcessTypeEnumerator();
  ~G4ProcessTypeEnumerator();

  inline unsigned int processId(const G4VProcess* p)
    {
      unsigned int id = 0;
      if(p) { id = p->GetProcessSubType(); }
      return id;
    }
  inline int processIdLong(const G4VProcess* p) 
    {
      int id = 0;
      if(p) { id = p->GetProcessSubType(); }
      return id;
    }

  std::string processG4Name(int);

  int processId(const std::string& name);

};
#endif 

