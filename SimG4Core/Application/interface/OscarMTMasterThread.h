#ifndef SimG4Core_OscarMTMasterThread_H
#define SimG4Core_OscarMTMasterThread_H

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace edm {
  class EventSetup;
}

class RunManagerMTInit;
class RunManagerMT;

class OscarMTMasterThread {
public:
  OscarMTMasterThread(std::shared_ptr<RunManagerMTInit> runManager, const edm::EventSetup& iSetup);
  ~OscarMTMasterThread();

  const RunManagerMTInit& runManagerInit() const { return *m_runManagerInit; }
  const RunManagerMT& runManagerMaster() const { return *m_runManagerMaster; }
  const RunManagerMT *runManagerMasterPtr() const { return m_runManagerMaster.get(); }
  
private:
  std::shared_ptr<RunManagerMTInit> m_runManagerInit;
  std::shared_ptr<RunManagerMT> m_runManagerMaster;
  std::thread m_masterThread;
  std::mutex m_startMutex;
  std::mutex m_stopMutex;
  std::condition_variable m_startCv;
  std::condition_variable m_stopCv;
  bool m_startCanProceed;
  bool m_stopCanProceed;
};


#endif
