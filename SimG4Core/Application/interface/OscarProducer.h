#ifndef SimG4Core_OscarProducer_H
#define SimG4Core_OscarProducer_H

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "SimG4Core/Application/interface/RunManager.h"
#include "SimG4Core/Application/interface/CustomUIsession.h"

#include <memory>

class OscarProducer : public edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns>
{
public:
  typedef std::vector<std::shared_ptr<SimProducer> > Producers;

  explicit OscarProducer(edm::ParameterSet const & p);
  virtual ~OscarProducer();
  virtual void beginRun(const edm::Run & r,const edm::EventSetup& c) override;
  virtual void endRun(const edm::Run & r,const edm::EventSetup& c) override;
  virtual void produce(edm::Event & e, const edm::EventSetup& c) override;

private:
  std::unique_ptr<RunManager> m_runManager;
  Producers     m_producers;
  std::unique_ptr<CustomUIsession> m_UIsession;
};

#endif
