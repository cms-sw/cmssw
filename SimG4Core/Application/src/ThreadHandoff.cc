// -*- C++ -*-
//
// Package:     SimG4Core/Application
// Class  :     ThreadHandoff
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 16 Aug 2021 14:37:29 GMT
//

// system include files

// user include files
#include "SimG4Core/Application/interface/ThreadHandoff.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <array>

//
// constants, enums and typedefs
//

namespace {
  std::string errorMessage(int erno) {
    std::array<char, 1024> buffer;
    strerror_r(erno, &buffer[0], buffer.size());
    buffer.back() = '\0';
    return std::string(&buffer[0]);
  }
}  // namespace
//
// static data member definitions
//

//
// constructors and destructor
//
using namespace omt;

ThreadHandoff::ThreadHandoff(int stackSize) {
  pthread_attr_t attr;
  int erno;
  if (0 != (erno = pthread_attr_init(&attr))) {
    throw cms::Exception("ThreadInitFailed")
        << "Failed to initialize thread attributes (" << erno << ") " << errorMessage(erno);
  }

  if (0 != (erno = pthread_attr_setstacksize(&attr, stackSize))) {
    throw cms::Exception("ThreadStackSizeFailed")
        << "Failed to set stack size " << stackSize << " " << errorMessage(erno);
  }
  std::unique_lock<std::mutex> lk(m_mutex);

  erno = pthread_create(&m_thread, &attr, threadLoop, this);
  if (0 != erno) {
    throw cms::Exception("ThreadCreateFailed") << " failed to create a pthread (" << erno << ") " << errorMessage(erno);
  }
  m_loopReady = false;
  m_threadHandoff.wait(lk, [this]() { return m_loopReady; });
}

// ThreadHandoff::ThreadHandoff(const ThreadHandoff& rhs)
// {
//    // do actual copying here;
// }

ThreadHandoff::~ThreadHandoff() {
  if (not m_stopThread) {
    stopThread();
  }
  void* ret;
  pthread_join(m_thread, &ret);
}

//
// assignment operators
//
// const ThreadHandoff& ThreadHandoff::operator=(const ThreadHandoff& rhs)
// {
//   //An exception safe implementation is
//   ThreadHandoff temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
void* ThreadHandoff::threadLoop(void* iArgs) {
  auto theThis = reinterpret_cast<ThreadHandoff*>(iArgs);

  //need to hold lock until wait to avoid both threads
  // being stuck in wait
  std::unique_lock<std::mutex> lck(theThis->m_mutex);
  theThis->m_loopReady = true;
  theThis->m_threadHandoff.notify_one();

  do {
    theThis->m_toRun = nullptr;
    theThis->m_threadHandoff.wait(lck, [theThis]() { return nullptr != theThis->m_toRun; });
    theThis->m_toRun->execute();
    theThis->m_loopReady = true;
    theThis->m_threadHandoff.notify_one();
  } while (not theThis->m_stopThread);
  theThis->m_loopReady = true;
  return nullptr;
}
