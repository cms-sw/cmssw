#ifndef SimG4Core_BeginOfJob_H
#define SimG4Core_BeginOfJob_H

namespace edm {
  class EventSetup;
}

class BeginOfJob {
public:
  BeginOfJob(const edm::EventSetup* tJob) : anJob(tJob) {}
  const edm::EventSetup* operator()() const { return anJob; }

private:
  const edm::EventSetup* anJob;
};

#endif
