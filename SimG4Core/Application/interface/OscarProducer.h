#ifndef SimG4Core_OscarProducer_H
#define SimG4Core_OscarProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
// #include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "SimG4Core/Application/interface/RunManager.h"

#include "SimG4Core/Application/interface/CustomUIsession.h"

namespace CLHEP {
    class HepRandomEngine;
}

class OscarProducer : public edm::EDProducer
{
public:
    typedef std::vector<boost::shared_ptr<SimProducer> > Producers;

    explicit OscarProducer(edm::ParameterSet const & p);
    virtual ~OscarProducer();
    virtual void beginRun(const edm::Run & r,const edm::EventSetup& c) override;
    virtual void beginJob();
    virtual void endJob();
    virtual void produce(edm::Event & e, const edm::EventSetup& c) override;
protected:
    RunManager*   m_runManager;
    Producers     m_producers;
    CustomUIsession* m_UIsession;

private:
    CLHEP::HepRandomEngine*  m_engine;
};

#endif
