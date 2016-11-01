#include "SimG4Core/Application/interface/CustomUIsessionThreadPrefix.h"

CustomUIsessionThreadPrefix::CustomUIsessionThreadPrefix(const std::string& threadPrefix, int threadId):
  CustomUIsession(),
  m_threadPrefix(threadPrefix+std::to_string(threadId)+">> ")
{}

CustomUIsessionThreadPrefix::~CustomUIsessionThreadPrefix() {}

namespace {
  std::string addThreadPrefix(const std::string& threadPrefix, const std::string str) {
    // Add thread prefix to each line beginning
    std::string ret;
    std::string::size_type beg = 0;
    std::string::size_type end = str.find('\n');
    while(end != std::string::npos) {
      ret += threadPrefix + str.substr(beg, end-beg) + "\n";
      beg = end+1;
      end = str.find('\n', beg);
    }
    ret += threadPrefix + str.substr(beg, end);
    return ret;
  }
}

G4int CustomUIsessionThreadPrefix::ReceiveG4cout(const G4String& coutString)
{
  // edm::LogInfo("G4cout") << addThreadPrefix(m_threadPrefix, trim(coutString));
  edm::LogVerbatim("G4cout") << addThreadPrefix(m_threadPrefix, trim(coutString));
  return 0;
}

G4int CustomUIsessionThreadPrefix::ReceiveG4cerr(const G4String& cerrString)
{
  edm::LogWarning("G4cerr") << addThreadPrefix(m_threadPrefix, trim(cerrString));
  return 0;
}
