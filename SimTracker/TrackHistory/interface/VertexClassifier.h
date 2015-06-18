
#ifndef VertexClassifier_h
#define VertexClassifier_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "SimTracker/TrackHistory/interface/CMSProcessTypes.h"
#include "SimTracker/TrackHistory/interface/VertexCategories.h"
#include "SimTracker/TrackHistory/interface/VertexHistory.h"

//! Get track history and classify it in function of their .
class VertexClassifier : public VertexCategories
{

public:

    //! Type to the associate category
    typedef VertexCategories Categories;

    //! Constructor by ParameterSet
    VertexClassifier(edm::ParameterSet const & pset,
                     edm::ConsumesCollector&&);

    virtual ~VertexClassifier() {}

    //! Pre-process event information (for accessing reconstraction information)
    virtual void newEvent(edm::Event const &, edm::EventSetup const &);

    //! Classify the RecoVertex in categories.
    VertexClassifier const & evaluate (reco::VertexBaseRef const &);

    //! Classify the TrackingVertex in categories.
    VertexClassifier const & evaluate (TrackingVertexRef const &);

    //! Classify the RecoVertex in categories.
    VertexClassifier const & evaluate (reco::VertexRef const & vertex)
    {
        return evaluate( reco::VertexBaseRef(vertex) );
    }

    //! Returns a reference to the vertex history used in the classification.
    VertexHistory const & history() const
    {
        return tracer_;
    }

private:

    VertexHistory tracer_;

    const G4toCMSLegacyProcTypeMap g4toCMSProcMap_;

    const edm::InputTag hepMCLabel_;

    double longLivedDecayLength_;
    double vertexClusteringDistance_;

    edm::Handle<edm::HepMCProduct> mcInformation_;

    edm::ESHandle<ParticleDataTable> particleDataTable_;

    //! Get reconstruction information
    void reconstructionInformation(reco::TrackBaseRef const &);

    //! Get all the information related to the simulation details
    void simulationInformation();

    //! Get all the information related to decay process
    void processesAtGenerator();

    //! Get information about conversion and other interactions
    void processesAtSimulation();

    //! Get geometrical information about the vertices
    void vertexInformation();

    //! Auxiliary class holding simulated primary vertices
    struct GeneratedPrimaryVertex
    {
        GeneratedPrimaryVertex(double x1,double y1,double z1): x(x1), y(y1), z(z1), ptsq(0), nGenTrk(0) {}

        bool operator< ( GeneratedPrimaryVertex const & reference) const
        {
            return ptsq < reference.ptsq;
        }

        double x, y, z;
        double ptsq;
        int nGenTrk;

        HepMC::FourVector ptot;

        std::vector<int> finalstateParticles;
        std::vector<int> simTrackIndex;
        std::vector<int> genVertex;
    };

    std::vector<GeneratedPrimaryVertex> genpvs_;

    // Auxiliary function to get the generated primary vertex
    bool isFinalstateParticle(const HepMC::GenParticle *);
    bool isCharged(const HepMC::GenParticle *);
    void genPrimaryVertices();

};

#endif
