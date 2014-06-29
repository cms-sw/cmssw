#ifndef SimG4Core_OscarMTMasterThread_H
#define SimG4Core_OscarMTMasterThread_H

#include "SimG4Core/Application/interface/RunManagerMTInit.h"
class RunManagerMTInit;

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace edm {
  class EventSetup;
}

class RunManagerMT;

class OscarMTMasterThread {
public:
  OscarMTMasterThread(const RunManagerMTInit *runManagerInit, const edm::EventSetup& iSetup);
  ~OscarMTMasterThread();

  void stopThread() const;

  const RunManagerMTInit& runManagerInit() const { return *m_runManagerInit; }
  const RunManagerMT& runManagerMaster() const { return *m_runManagerMaster; }
  const RunManagerMT *runManagerMasterPtr() const { return m_runManagerMaster.get(); }
  
private:
  enum class ThreadState {
    NotExist=0, BeginRun=1, EndRun=2, Destruct=3
  };

  const RunManagerMTInit *m_runManagerInit;
  mutable std::shared_ptr<RunManagerMT> m_runManagerMaster;
  mutable std::thread m_masterThread;
  mutable RunManagerMTInit::ESProducts m_esProducts;

  mutable std::mutex m_protectMutex;
  mutable std::mutex m_threadMutex;
  //mutable std::mutex m_notifyMasterMutex;
  //mutable std::mutex m_notifyMainMutex;
  mutable std::condition_variable m_notifyMasterCv;
  mutable std::condition_variable m_notifyMainCv;

  mutable ThreadState m_masterThreadState;

  mutable bool m_masterCanProceed;
  mutable bool m_mainCanProceed;
  mutable bool m_stopped;
};


#endif
