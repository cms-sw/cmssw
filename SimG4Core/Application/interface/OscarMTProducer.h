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

class OscarMTProducer : public edm::stream::EDProducer<
  edm::GlobalCache<RunManagerMTInit>,
  edm::RunCache<OscarMTMasterThread>
>
{
public:
  typedef std::vector<boost::shared_ptr<SimProducer> > Producers;

  explicit OscarMTProducer(edm::ParameterSet const & p, const RunManagerMTInit *);
  virtual ~OscarMTProducer();

  static std::unique_ptr<RunManagerMTInit> initializeGlobalCache(const edm::ParameterSet& iConfig);
  static std::shared_ptr<OscarMTMasterThread> globalBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const RunManagerMTInit *runManagerInit);
  static void globalEndRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const RunContext *iContext);
  static void globalEndJob(RunManagerMTInit *runManagerInit);


  virtual void endRun(const edm::Run & r,const edm::EventSetup& c) override;
  virtual void produce(edm::Event & e, const edm::EventSetup& c) override;

private:
  Producers     m_producers;
  std::unique_ptr<RunManagerMTWorker> m_runManagerWorker;
  //edm::EDGetTokenT<edm::HepMCProduct> m_HepMC;
};

#endif
