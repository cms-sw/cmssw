#ifndef SimG4Core_CustomUIsessionToFile_H
#define SimG4Core_CustomUIsessionToFile_H 

#include "SimG4Core/Application/interface/CustomUIsession.h"

#include <fstream>

/**
 * This class is intended for debugging of multithreaded simulation
 * when the amount of output is moderate to large. Output of Geant4 in
 * each thread is forwarded to a thread-specific file. Compared to
 * using MessageLogger, this way the synchronization is avoided.
 * Thread safety is ensured by
 * - each thread gets its own file name
 * - each thread gets its own std::ofstream object (as
 *   RunManagerMTWorker creates a CustomUIsessionToFile object
 *   separately for each thread).
 */
class CustomUIsessionToFile : public CustomUIsession
{

 public:

  CustomUIsessionToFile(const std::string& filePrefix, int threadId);
  ~CustomUIsessionToFile();

  G4int ReceiveG4cout(const G4String& coutString) override;
  G4int ReceiveG4cerr(const G4String& cerrString) override;

private:
  std::ofstream m_output;
};

#endif
