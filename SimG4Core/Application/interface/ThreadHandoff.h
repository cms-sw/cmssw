#ifndef SimG4Core_Application_ThreadHandoff_h
#define SimG4Core_Application_ThreadHandoff_h
// -*- C++ -*-
//
// Package:     SimG4Core/Application
// Class  :     ThreadHandoff
//
/**\class ThreadHandoff ThreadHandoff.h "SimG4Core/Application/interface/ThreadHandoff.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 16 Aug 2021 13:51:53 GMT
//

// system include files
#include <condition_variable>
#include <cstring>  //strerror_r
#include <exception>
#include <mutex>
#include <pthread.h>

// user include files

// forward declarations

namespace omt {
  class ThreadHandoff {
  public:
    explicit ThreadHandoff(int stackSize);
    ~ThreadHandoff();

    ThreadHandoff(const ThreadHandoff&) = delete;                   // stop default
    const ThreadHandoff& operator=(const ThreadHandoff&) = delete;  // stop default

    template <typename F>
    void runAndWait(F&& iF) {
      Functor<F> f{std::move(iF)};

      std::unique_lock<std::mutex> lck(m_mutex);
      m_loopReady = false;
      m_toRun = &f;

      m_threadHandoff.notify_one();

      m_threadHandoff.wait(lck, [this]() { return m_loopReady; });
      auto e = f.exception();
      if (e) {
        std::rethrow_exception(e);
      }
    }

    void stopThread() {
      runAndWait([this]() { m_stopThread = true; });
    }

  private:
    class FunctorBase {
    public:
      virtual ~FunctorBase() {}
      virtual void execute() = 0;
    };
    template <typename F>
    class Functor : public FunctorBase {
    public:
      explicit Functor(F&& iF) : m_f(std::move(iF)) {}
      void execute() final {
        try {
          m_f();
        } catch (...) {
          m_except = std::current_exception();
        }
      }
      std::exception_ptr exception() { return m_except; }

    private:
      F m_f;
      std::exception_ptr m_except;
    };

    static void* threadLoop(void* iArgs);

    // ---------- member data --------------------------------
    pthread_t m_thread;
    std::mutex m_mutex;
    std::condition_variable m_threadHandoff;

    FunctorBase* m_toRun{nullptr};
    bool m_loopReady{false};
    bool m_stopThread{false};
  };
}  // namespace omt
#endif
