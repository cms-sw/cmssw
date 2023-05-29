#ifndef G4ProcessTypeEnumerator_H
#define G4ProcessTypeEnumerator_H

#include "G4VProcess.hh"

class G4ProcessTypeEnumerator {
public:
  G4ProcessTypeEnumerator();
  ~G4ProcessTypeEnumerator() = default;

  inline unsigned int processId(const G4VProcess* p) const { return (p) ? p->GetProcessSubType() : 0; }
  inline int processIdLong(const G4VProcess* p) const { return (p) ? p->GetProcessSubType() : 0; }

  std::string processG4Name(int) const;

  int processId(const std::string& name) const;
};
#endif
