#include "SimG4Core/Application/interface/CustomUIsessionToFile.h"

CustomUIsessionToFile::CustomUIsessionToFile(const std::string& filePrefix, int threadId):
  CustomUIsession(),
  m_output(filePrefix+"_"+std::to_string(threadId)+".txt")
{}

CustomUIsessionToFile::~CustomUIsessionToFile() {}

G4int CustomUIsessionToFile::ReceiveG4cout(const G4String& coutString)
{
  m_output << coutString;
  return 0;
}

G4int CustomUIsessionToFile::ReceiveG4cerr(const G4String& cerrString)
{
  m_output << cerrString;
  return 0;
}
