#ifndef SimG4Core_OscarMTProducer_H
#define SimG4Core_OscarMTProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "SimG4Core/Application/interface/RunManagerMTInit.h"
#include "SimG4Core/Application/interface/OscarMTMasterThread.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <memory>

class SimProducer;
class RunManagerMTWorker;

class OscarMTGlobalCache {
public:
  OscarMTGlobalCache(const edm::ParameterSet& iConfig);
  ~OscarMTGlobalCache();

  OscarMTMasterThread& masterThread() { return m_masterThread; }
  const OscarMTMasterThread& masterThread() const { return m_masterThread; }

private:
  RunManagerMTInit m_runManagerInit;
  OscarMTMasterThread m_masterThread;
};

class OscarMTProducer : public edm::stream::EDProducer<
  edm::GlobalCache<OscarMTGlobalCache>,
  edm::RunCache<int> // for some reason void doesn't compile
>
{
public:
  typedef std::vector<boost::shared_ptr<SimProducer> > Producers;

  explicit OscarMTProducer(edm::ParameterSet const & p, const OscarMTGlobalCache *);
  virtual ~OscarMTProducer();

  static std::unique_ptr<OscarMTGlobalCache> initializeGlobalCache(const edm::ParameterSet& iConfig);
  static std::shared_ptr<int> globalBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const OscarMTGlobalCache *globalCache);
  static void globalEndRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const RunContext *iContext);
  static void globalEndJob(OscarMTGlobalCache *globalCache);


  virtual void endRun(const edm::Run & r,const edm::EventSetup& c) override;
  virtual void produce(edm::Event & e, const edm::EventSetup& c) override;

private:
  Producers     m_producers;
  std::unique_ptr<RunManagerMTWorker> m_runManagerWorker;
  //edm::EDGetTokenT<edm::HepMCProduct> m_HepMC;
};

#endif
