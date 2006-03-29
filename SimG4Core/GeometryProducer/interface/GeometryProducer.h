#ifndef SimG4Core_GeometryProducer_H
#define SimG4Core_GeometryProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/EDProduct.h"
 
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include <memory>
#include "boost/shared_ptr.hpp"

namespace sim { class FieldBuilder; }
class SimWatcher;
class SimProducer;
class DDDWorld;
class G4RunManagerKernel;

class GeometryProducer : public edm::EDProducer
{
public:
    typedef std::vector<boost::shared_ptr<SimProducer> > Producers;
    explicit GeometryProducer(edm::ParameterSet const & p);
    virtual ~GeometryProducer();
    virtual void beginJob(const edm::EventSetup & c);
    virtual void endJob();
    virtual void produce(edm::Event & e, const edm::EventSetup & c);
    std::vector<boost::shared_ptr<SimProducer> > producers() const
    { return m_producers; }
private:
    G4RunManagerKernel * m_kernel;
    bool m_pUseMagneticField;
    edm::ParameterSet m_pField;
    edm::ParameterSet m_p;    
    SimActivityRegistry m_registry;
    std::vector<boost::shared_ptr<SimWatcher> > m_watchers;
    std::vector<boost::shared_ptr<SimProducer> > m_producers;    
    std::auto_ptr<sim::FieldBuilder> m_fieldBuilder;
};

#endif
