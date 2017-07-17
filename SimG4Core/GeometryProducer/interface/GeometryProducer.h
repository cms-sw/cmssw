#ifndef SimG4Core_GeometryProducer_H
#define SimG4Core_GeometryProducer_H

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"

#include <memory>

namespace sim { class FieldBuilder; }

class SimWatcher;
class SimProducer;
class DDDWorld;
class G4RunManagerKernel;
class SimTrackManager;

class GeometryProducer : public edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns>
{
public:
    typedef std::vector<std::shared_ptr<SimProducer> > Producers;

    explicit GeometryProducer(edm::ParameterSet const & p);
    virtual ~GeometryProducer() override;
    virtual void beginRun(const edm::Run & r,const edm::EventSetup& c) override;
    virtual void endRun(const edm::Run & r,const edm::EventSetup& c) override;
    virtual void produce(edm::Event & e, const edm::EventSetup & c) override;
    void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
   
    std::vector<std::shared_ptr<SimProducer> > producers() const
    { return m_producers; }
    std::vector<SensitiveTkDetector*>& sensTkDetectors() { return m_sensTkDets; }
    std::vector<SensitiveCaloDetector*>& sensCaloDetectors() { return m_sensCaloDets; }

private:

    void updateMagneticField( edm::EventSetup const& es );

    G4RunManagerKernel * m_kernel;
    bool m_pUseMagneticField;
    edm::ParameterSet m_pField;
    bool m_pUseSensitiveDetectors;
    SimActivityRegistry m_registry;
    std::vector<std::shared_ptr<SimWatcher> > m_watchers;
    std::vector<std::shared_ptr<SimProducer> > m_producers;
    std::auto_ptr<sim::FieldBuilder> m_fieldBuilder;
    std::auto_ptr<SimTrackManager> m_trackManager;
    AttachSD * m_attach;
    std::vector<SensitiveTkDetector*> m_sensTkDets;
    std::vector<SensitiveCaloDetector*> m_sensCaloDets;
    edm::ParameterSet m_p;
    bool m_firstRun;
};

#endif
