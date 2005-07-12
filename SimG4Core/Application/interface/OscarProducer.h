#ifndef SimG4Core_OscarProducer_H
#define SimG4Core_OscarProducer_H

#include "FWCore/CoreFramework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/Handle.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"
#include "FWCore/CoreFramework/interface/EventSetup.h"

#include "SimG4Core/Application/interface/RunManager.h"
 
class OscarProducer : public edm::EDProducer
{
public:
    explicit OscarProducer(edm::ParameterSet const & p);
    virtual ~OscarProducer();
    virtual void produce(edm::Event & e, const edm::EventSetup & c);
protected:
    RunManager * m_runManager;
};

#endif
