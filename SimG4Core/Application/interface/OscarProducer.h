#ifndef SimG4Core_OscarProducer_H
#define SimG4Core_OscarProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimG4Core/Application/interface/RunManager.h"

namespace CLHEP {
    class HepRandomEngine;
}

class OscarProducer : public edm::EDProducer
{
public:
    typedef std::vector<boost::shared_ptr<SimProducer> > Producers;

    explicit OscarProducer(edm::ParameterSet const & p);
    virtual ~OscarProducer();
    virtual void beginJob(const edm::EventSetup & c);
    virtual void endJob();
    virtual void produce(edm::Event & e, const edm::EventSetup & c);
protected:
    RunManager*   m_runManager;
    Producers     m_producers;

private:
    CLHEP::HepRandomEngine*  m_engine;
};

#endif
