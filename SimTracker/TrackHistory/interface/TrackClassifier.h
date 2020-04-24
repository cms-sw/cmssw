
#ifndef TrackClassifier_h
#define TrackClassifier_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "SimTracker/TrackHistory/interface/CMSProcessTypes.h"
#include "SimTracker/TrackHistory/interface/TrackCategories.h"
#include "SimTracker/TrackHistory/interface/TrackHistory.h"
#include "SimTracker/TrackHistory/interface/TrackQuality.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"


class TrackTopology;

//! Get track history and classify it in function of their .
class TrackClassifier : public TrackCategories
{

public:

    //! Type to the associate category
    typedef TrackCategories Categories;

    //! Constructor by ParameterSet
    TrackClassifier(edm::ParameterSet const &,
                    edm::ConsumesCollector&& );

    //! Pre-process event information (for accessing reconstraction information)
    void newEvent(edm::Event const &, edm::EventSetup const &);

    //! Classify the RecoTrack in categories.
    TrackClassifier const & evaluate (reco::TrackBaseRef const &);

    //! Classify the TrackingParticle in categories.
    TrackClassifier const & evaluate (TrackingParticleRef const &);

    //! Classify the RecoTrack in categories.
    TrackClassifier const & evaluate (reco::TrackRef const & track)
    {
      return evaluate( reco::TrackBaseRef(track));
    }

    //! Returns a reference to the track history used in the classification.
    TrackHistory const & history() const
    {
        return tracer_;
    }

    //! Returns a reference to the track quality used in the classification.
    TrackQuality const & quality() const
    {
        return quality_;
    }

private:

    const edm::InputTag hepMCLabel_;
    const edm::InputTag beamSpotLabel_;

    double badPull_;
    double longLivedDecayLength_;
    double vertexClusteringSqDistance_;
    unsigned int numberOfInnerLayers_;
    unsigned int minTrackerSimHits_;

    TrackHistory tracer_;

    TrackQuality quality_;

    const G4toCMSLegacyProcTypeMap g4toCMSProcMap_;

    edm::ESHandle<MagneticField> magneticField_;

    edm::Handle<edm::HepMCProduct> mcInformation_;

    edm::ESHandle<ParticleDataTable> particleDataTable_;

    edm::ESHandle<TransientTrackBuilder> transientTrackBuilder_;

    edm::Handle<reco::BeamSpot> beamSpot_;

    const TrackerTopology *tTopo_;

    //! Classify all the tracks by their association and reconstruction information
    void reconstructionInformation(reco::TrackBaseRef const &);

    //! Get all the information related to the simulation details
    void simulationInformation();

    //! Classify all the tracks by their reconstruction quality
    void qualityInformation(reco::TrackBaseRef const &);

    //! Get hadron flavor of the initial hadron
    void hadronFlavor();

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
