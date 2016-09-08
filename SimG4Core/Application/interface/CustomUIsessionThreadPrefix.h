#ifndef SimG4Core_CustomUIsessionThreadPrefix_H
#define SimG4Core_CustomUIsessionThreadPrefix_H 

#include "SimG4Core/Application/interface/CustomUIsession.h"


/**
 * This class is intended for debugging of multithreaded simulation
 * when the amount of output is small to moderate. The output of
 * Geant4 is forwarded to MessageLogger as in CustomUIsession, but a
 * thread-specific prefix is added before each line of output. This
 * makes it easier to grab the output of a specific thread.
 */
class CustomUIsessionThreadPrefix : public CustomUIsession
{

 public:

  explicit CustomUIsessionThreadPrefix(const std::string& threadPrefix, int threadId);
  virtual ~CustomUIsessionThreadPrefix();

  G4int ReceiveG4cout(const G4String& coutString) override;
  G4int ReceiveG4cerr(const G4String& cerrString) override;

private:
  const std::string m_threadPrefix;
};

#endif
