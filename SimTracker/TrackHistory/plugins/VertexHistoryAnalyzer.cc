/*
 *  VertexHistoryAnalyzer.C
 *
 *  Created by Victor Eduardo Bazterra on 5/31/07.
 *
 */

// system include files
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

// user include files
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimTracker/TrackHistory/interface/VertexClassifier.h"

//
// class decleration
//

class VertexHistoryAnalyzer : public edm::EDAnalyzer
{
public:

    explicit VertexHistoryAnalyzer(const edm::ParameterSet&);

private:

  virtual void beginRun(const edm::Run&,const edm::EventSetup&);
    virtual void beginJob() ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    // Member data

    VertexClassifier classifier_;

    edm::InputTag vertexProducer_;

    edm::ESHandle<ParticleDataTable> pdt_;

    std::string particleString(int) const;

    std::string vertexString(
        const TrackingParticleRefVector&,
        const TrackingParticleRefVector&
    ) const;

    std::string vertexString(
        HepMC::GenVertex::particles_in_const_iterator,
        HepMC::GenVertex::particles_in_const_iterator,
        HepMC::GenVertex::particles_out_const_iterator,
        HepMC::GenVertex::particles_out_const_iterator
    ) const;
};


VertexHistoryAnalyzer::VertexHistoryAnalyzer(const edm::ParameterSet& config) : classifier_(config)
{
    vertexProducer_ = config.getUntrackedParameter<edm::InputTag> ("vertexProducer");
}


void VertexHistoryAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
    // Set the classifier for a new event
    classifier_.newEvent(event, setup);

    // Vertex collection
    edm::Handle<edm::View<reco::Vertex> > vertexCollection;
    event.getByLabel(vertexProducer_, vertexCollection);

    // Get a constant reference to the track history associated to the classifier
    VertexHistory const & tracer = classifier_.history();

    // Loop over the track collection.
    for (std::size_t index = 0; index < vertexCollection->size(); index++)
    {
        std::cout << std::endl << "History for vertex #" << index << " : " << std::endl;

        // Classify the track and detect for fakes
        if ( !classifier_.evaluate( reco::VertexBaseRef(vertexCollection, index) ).is(VertexClassifier::Fake) )
        {
            // Get the list of TrackingParticles associated to
            VertexHistory::SimParticleTrail simParticles(tracer.simParticleTrail());

            // Loop over all simParticles
            for (std::size_t hindex=0; hindex<simParticles.size(); hindex++)
            {
                std::cout << "  simParticles [" << hindex << "] : "
                          << particleString(simParticles[hindex]->pdgId())
                          << std::endl;
            }

            // Get the list of TrackingVertexes associated to
            VertexHistory::SimVertexTrail simVertexes(tracer.simVertexTrail());

            // Loop over all simVertexes
            if ( !simVertexes.empty() )
            {
                for (std::size_t hindex=0; hindex<simVertexes.size(); hindex++)
                {
                    std::cout << "  simVertex    [" << hindex << "] : "
                              << vertexString(
                                  simVertexes[hindex]->sourceTracks(),
                                  simVertexes[hindex]->daughterTracks()
                              )
                              << std::endl;
                }
            }
            else
                std::cout << "  simVertex no found" << std::endl;

            // Get the list of GenParticles associated to
            VertexHistory::GenParticleTrail genParticles(tracer.genParticleTrail());

            // Loop over all genParticles
            for (std::size_t hindex=0; hindex<genParticles.size(); hindex++)
            {
                std::cout << "  genParticles [" << hindex << "] : "
                          << particleString(genParticles[hindex]->pdg_id())
                          << std::endl;
            }

            // Get the list of TrackingVertexes associated to
            VertexHistory::GenVertexTrail genVertexes(tracer.genVertexTrail());

            // Loop over all simVertexes
            if ( !genVertexes.empty() )
            {
                for (std::size_t hindex=0; hindex<genVertexes.size(); hindex++)
                {
                    std::cout << "  genVertex    [" << hindex << "] : "
                              << vertexString(
                                  genVertexes[hindex]->particles_in_const_begin(),
                                  genVertexes[hindex]->particles_in_const_end(),
                                  genVertexes[hindex]->particles_out_const_begin(),
                                  genVertexes[hindex]->particles_out_const_end()
                              )
                              << std::endl;
                }
            }
            else
                std::cout << "  genVertex no found" << std::endl;
        }
        else
            std::cout << "  fake vertex" << std::endl;

        std::cout << "  vertex categories : " << classifier_;
        std::cout << std::endl;
    }
}


void
VertexHistoryAnalyzer::beginRun(const edm::Run& run, const edm::EventSetup& setup)
{
    // Get the particles table.
    setup.getData( pdt_ );
}

void
VertexHistoryAnalyzer::beginJob()
{

}


std::string VertexHistoryAnalyzer::particleString(int pdgId) const
{
    ParticleData const * pid;

    std::ostringstream vDescription;

    HepPDT::ParticleID particleType(pdgId);

    if (particleType.isValid())
    {
        pid = pdt_->particle(particleType);
        if (pid)
            vDescription << pid->name();
        else
            vDescription << pdgId;
    }
    else
        vDescription << pdgId;

    return vDescription.str();
}


std::string VertexHistoryAnalyzer::vertexString(
    const TrackingParticleRefVector& in,
    const TrackingParticleRefVector& out
) const
{
    ParticleData const * pid;

    std::ostringstream vDescription;

    for (std::size_t j = 0; j < in.size(); j++)
    {
        if (!j) vDescription << "(";

        HepPDT::ParticleID particleType(in[j]->pdgId());

        if (particleType.isValid())
        {
            pid = pdt_->particle(particleType);
            if (pid)
                vDescription << pid->name();
            else
                vDescription << in[j]->pdgId();
        }
        else
            vDescription << in[j]->pdgId();

        if (j == in.size() - 1) vDescription << ")";
        else vDescription << ",";
    }

    vDescription << "->";

    for (std::size_t j = 0; j < out.size(); j++)
    {
        if (!j) vDescription << "(";

        HepPDT::ParticleID particleType(out[j]->pdgId());

        if (particleType.isValid())
        {
            pid = pdt_->particle(particleType);
            if (pid)
                vDescription << pid->name();
            else
                vDescription << out[j]->pdgId();
        }
        else
            vDescription << out[j]->pdgId();

        if (j == out.size() - 1) vDescription << ")";
        else vDescription << ",";
    }

    return vDescription.str();
}


std::string VertexHistoryAnalyzer::vertexString(
    HepMC::GenVertex::particles_in_const_iterator in_begin,
    HepMC::GenVertex::particles_in_const_iterator in_end,
    HepMC::GenVertex::particles_out_const_iterator out_begin,
    HepMC::GenVertex::particles_out_const_iterator out_end
) const
{
    ParticleData const * pid;

    std::ostringstream vDescription;

    std::size_t j = 0;

    HepMC::GenVertex::particles_in_const_iterator in, itmp;

    for (in = in_begin; in != in_end; in++, j++)
    {
        if (!j) vDescription << "(";

        HepPDT::ParticleID particleType((*in)->pdg_id());

        if (particleType.isValid())
        {
            pid = pdt_->particle(particleType);
            if (pid)
                vDescription << pid->name();
            else
                vDescription << (*in)->pdg_id();
        }
        else
            vDescription << (*in)->pdg_id();

        itmp = in;

        if (++itmp == in_end) vDescription << ")";
        else vDescription << ",";
    }

    vDescription << "->";
    j = 0;

    HepMC::GenVertex::particles_out_const_iterator out, otmp;

    for (out = out_begin; out != out_end; out++, j++)
    {
        if (!j) vDescription << "(";

        HepPDT::ParticleID particleType((*out)->pdg_id());

        if (particleType.isValid())
        {
            pid = pdt_->particle(particleType);
            if (pid)
                vDescription << pid->name();
            else
                vDescription << (*out)->pdg_id();
        }
        else
            vDescription << (*out)->pdg_id();

        otmp = out;

        if (++otmp == out_end) vDescription << ")";
        else vDescription << ",";
    }

    return vDescription.str();
}


DEFINE_FWK_MODULE(VertexHistoryAnalyzer);
