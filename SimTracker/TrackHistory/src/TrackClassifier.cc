
#include <math.h>
#include <cstdlib>
#include <iostream>

#include "HepPDT/ParticleID.hh"

#include "SimTracker/TrackHistory/interface/TrackClassifier.h"

#define update(a, b) do { (a) = (a) | (b); } while(0)

TrackClassifier::TrackClassifier(edm::ParameterSet const & pset) : TrackCategories(),
        hepMCLabel_( pset.getUntrackedParameter<edm::InputTag>("hepMC") ),
        beamSpotLabel_( pset.getUntrackedParameter<edm::InputTag>("beamSpot") ),
        tracer_(pset),
        quality_(pset)
{
    // Set the history depth after hadronization
    tracer_.depth(-2);

    // Set the maximum d0pull for the bad category
    badD0Pull_ = pset.getUntrackedParameter<double>("badD0Pull");

    // Set the minimum decay length for detecting long decays
    longLivedDecayLength_ = pset.getUntrackedParameter<double>("longLivedDecayLength");

    // Set the distance for clustering vertices
    float vertexClusteringDistance = pset.getUntrackedParameter<double>("vertexClusteringDistance");
    vertexClusteringSqDistance_ = vertexClusteringDistance * vertexClusteringDistance;

    // Set the number of innermost layers to check for bad hits
    numberOfInnerLayers_ = pset.getUntrackedParameter<unsigned int>("numberOfInnerLayers");
}


void TrackClassifier::newEvent ( edm::Event const & event, edm::EventSetup const & setup )
{
    // Get the new event information for the tracer
    tracer_.newEvent(event, setup);

    // Get the new event information for the track quality analyser
    quality_.newEvent(event, setup);

    // Get hepmc of the event
    event.getByLabel(hepMCLabel_, mcInformation_);

    // Magnetic field
    setup.get<IdealMagneticFieldRecord>().get(magneticField_);

    // Get the partivle data table
    setup.getData(particleDataTable_);

    // get the beam spot
    edm::Handle<reco::BeamSpot> beamSpot;
    event.getByLabel(beamSpotLabel_, beamSpot);
    beamSpot_ = reco::TrackBase::Point(
                    beamSpot->x0(), beamSpot->y0(), beamSpot->z0()
                );

    // Transient track builder
    setup.get<TransientTrackRecord>().get("TransientTrackBuilder", transientTrackBuilder_);

    // Create the list of primary vertices associated to the event
    genPrimaryVertices();
}


TrackClassifier const & TrackClassifier::evaluate (reco::TrackBaseRef const & track)
{
    // Initializing the category vector
    reset();

    // Associate and evaluate the track history (check for fakes)
    if ( tracer_.evaluate(track) )
    {
        // Classify all the tracks by their association and reconstruction information
        reconstructionInformation(track);

        // Get all the information related to the simulation details
        simulationInformation();

        // Analyse the track reconstruction quality
        qualityInformation(track);

        // Get hadron flavor of the initial hadron
        hadronFlavor();

        // Get all the information related to decay process
        processesAtGenerator();

        // Get information about conversion and other interactions
        processesAtSimulation();

        // Get geometrical information about the vertices
        vertexInformation();

        // Check for unkown classification
        unknownTrack();
    }
    else
        flags_[Fake] = true;

    return *this;
}


TrackClassifier const & TrackClassifier::evaluate (TrackingParticleRef const & track)
{
    // Initializing the category vector
    reset();

    // Trace the history for the given TP
    tracer_.evaluate(track);

    // Collect the associated reco track
    const reco::TrackBaseRef & recotrack = tracer_.recoTrack();

    // If there is a reco truck then evaluate the simulated history
    if ( recotrack.isNonnull() )
    {
        flags_[Reconstructed] = true;
        // Classify all the tracks by their association and reconstruction information
        reconstructionInformation(recotrack);
        // Analyse the track reconstruction quality
        qualityInformation(recotrack);
    }
    else
        flags_[Reconstructed] = false;

    // Get all the information related to the simulation details
    simulationInformation();

    // Get hadron flavor of the initial hadron
    hadronFlavor();

    // Get all the information related to decay process
    processesAtGenerator();

    // Get information about conversion and other interactions
    processesAtSimulation();

    // Get geometrical information about the vertices
    vertexInformation();

    // Check for unkown classification
    unknownTrack();

    return *this;
}


void TrackClassifier::reconstructionInformation(reco::TrackBaseRef const & track)
{
    TrackingParticleRef tpr = tracer_.simParticle();

    // Compute tracking particle parameters at point of closest approach to the beamline
    const SimTrack * assocTrack = &(*tpr->g4Track_begin());

    FreeTrajectoryState ftsAtProduction(
        GlobalPoint(
            tpr->vertex().x(),
            tpr->vertex().y(),
            tpr->vertex().z()
        ),
        GlobalVector(
            assocTrack->momentum().x(),
            assocTrack->momentum().y(),
            assocTrack->momentum().z()
        ),
        TrackCharge(track->charge()),
        magneticField_.product()
    );

    TSCPBuilderNoMaterial tscpBuilder;

    GlobalPoint theBS(beamSpot_.x(), beamSpot_.y(), beamSpot_.z());

    TrajectoryStateClosestToPoint tsAtClosestApproach = tscpBuilder(
                ftsAtProduction, theBS );


    GlobalVector v = tsAtClosestApproach.theState().position() - theBS;
    GlobalVector p = tsAtClosestApproach.theState().momentum();

    // Simulated d0
    double d0Sim = - (-v.x()*sin(p.phi()) + v.y()*cos(p.phi()));

    // Calculate the d0 pull
    double d0Pull = std::abs(-track->dxy(beamSpot_) - d0Sim) / track->d0Error();

    // Return true if d0Pull > badD0Pull sigmas
    flags_[Bad] = (d0Pull > badD0Pull_);
}


void TrackClassifier::simulationInformation()
{
    // Get the event id for the initial TP.
    EncodedEventId eventId = tracer_.simParticle()->eventId();
    // Check for signal events
    flags_[SignalEvent] = !eventId.bunchCrossing() && !eventId.event();
}


void TrackClassifier::qualityInformation(reco::TrackBaseRef const & track)
{
    // run the hit-by-hit reconstruction quality analysis
    quality_.evaluate(tracer_.simParticleTrail(), track);

    unsigned int maxLayers = std::min(numberOfInnerLayers_,
                                      quality_.numberOfLayers());

    // check the innermost layers for bad hits
    for (unsigned int i = 0; i < maxLayers; i++)
    {
        const TrackQuality::Layer &layer = quality_.layer(i);

        // check all hits in that layer
        for (unsigned int j = 0; j < layer.hits.size(); j++)
        {
            const TrackQuality::Layer::Hit &hit = layer.hits[j];

            // In those cases the bad hit was used by track reconstruction
            if (hit.state == TrackQuality::Layer::Noise ||
                    hit.state == TrackQuality::Layer::Misassoc)
                flags_[BadInnerHits] = true;
            else if (hit.state == TrackQuality::Layer::Shared)
                flags_[SharedInnerHits] = true;
        }
    }
}


void TrackClassifier::hadronFlavor()
{
    // Get the initial hadron
    const HepMC::GenParticle * particle = tracer_.genParticle();

    // Check for the initial hadron
    if (particle)
    {
        HepPDT::ParticleID pid(particle->pdg_id());
        flags_[Bottom] = pid.hasBottom();
        flags_[Charm] =  pid.hasCharm();
        flags_[Light] = !pid.hasCharm() && !pid.hasBottom();
    }
}


void TrackClassifier::processesAtGenerator()
{
    // Get the generated vetices from track history
    TrackHistory::GenVertexTrail const & genVertexTrail = tracer_.genVertexTrail();

    // Loop over the generated vertices
    for (TrackHistory::GenVertexTrail::const_iterator ivertex = genVertexTrail.begin(); ivertex != genVertexTrail.end(); ++ivertex)
    {
        // Get the pointer to the vertex by removing the const-ness (no const methos in HepMC::GenVertex)
        HepMC::GenVertex * vertex = const_cast<HepMC::GenVertex *>(*ivertex);

        // Loop over the sources looking for specific decays
        for (
            HepMC::GenVertex::particle_iterator iparent = vertex->particles_begin(HepMC::parents);
            iparent != vertex->particles_end(HepMC::parents);
            ++iparent
        )
        {
            // Collect the pdgid of the parent
            int pdgid = std::abs((*iparent)->pdg_id());
            // Get particle type
            HepPDT::ParticleID particleID(pdgid);

            // Check if the particle type is valid one
            if (particleID.isValid())
            {
                // Get particle data
                ParticleData const * particleData = particleDataTable_->particle(particleID);
                // Check if the particle exist in the table
                if (particleData)
                {
                    bool longlived = false;
                    // Check if their life time is bigger than longLivedDecayLength_
                    if ( particleData->lifetime() > longLivedDecayLength_ )
                    {
                        // Check for B, C weak decays and long lived decays
                        update(flags_[BWeakDecay], particleID.hasBottom());
                        update(flags_[CWeakDecay], particleID.hasCharm());
                        longlived = true;
                    }
                    // Check Tau, Ks and Lambda decay
                    update(flags_[TauDecay], pdgid == 15);
                    update(flags_[KsDecay], pdgid == 310);
                    update(flags_[LambdaDecay], pdgid == 3122);
                    update(flags_[Jpsi], pdgid == 443);
                    update(
                        flags_[LongLivedDecay],
                        !flags_[BWeakDecay] &&
                        !flags_[CWeakDecay] &&
                        !flags_[TauDecay] &&
                        !flags_[KsDecay] &&
                        !flags_[LambdaDecay] &&
                        !flags_[Jpsi] &&
                        longlived
                    );
                }
            }
        }
    }
}


void TrackClassifier::processesAtSimulation()
{
    TrackHistory::SimParticleTrail const & simParticleTrail = tracer_.simParticleTrail();

    // Loop over the simulated particles
    for (
        TrackHistory::SimParticleTrail::const_iterator iparticle = simParticleTrail.begin();
        iparticle != simParticleTrail.end();
        ++iparticle
    )
    {
        if ( (*iparticle)->matchedHit() )
        {
            // pdgid of the real source parent vertex
            int pdgid = 0;

            // Get a reference to the TP's parent vertex
            TrackingVertexRef const & parentVertex = (*iparticle)->parentVertex();

            // Look for the original source track
            if ( parentVertex.isNonnull() )
            {
                // select the original source in case of combined vertices
                bool flag = false;
                TrackingVertex::tp_iterator itd, its;

                for (its = parentVertex->sourceTracks_begin(); its != parentVertex->sourceTracks_end(); ++its)
                {
                    for (itd = parentVertex->daughterTracks_begin(); itd != parentVertex->daughterTracks_end(); ++itd)
                        if (itd != its)
                        {
                            flag = true;
                            break;
                        }
                    if (flag)
                        break;
                }
                // Collect the pdgid of the original source track
                if ( its != parentVertex->sourceTracks_end() )
                    pdgid = std::abs((*its)->pdgId());
                else
                    pdgid = 0;
            }

            // Collect the G4 process of the first psimhit (it should be the same for all of them)
            unsigned short process = (*iparticle)->pSimHit_begin()->processType();

            // Look for conversion process
            flags_[Conversion] = (process == G4::Conversions);

            // Special treatment for decays
            if (process == G4::Decay)
            {
                // Get particle type
                HepPDT::ParticleID particleID(pdgid);
                // Check if the particle type is valid one
                if (particleID.isValid())
                {
                    // Get particle data
                    ParticleData const * particleData = particleDataTable_->particle(particleID);
                    // Check if the particle exist in the table
                    if (particleData)
                    {
                        bool longlived = false;
                        // Check if their life time is bigger than 1e-14
                        if ( particleDataTable_->particle(particleID)->lifetime() > longLivedDecayLength_ )
                        {
                            // Check for B, C weak decays and long lived decays
                            update(flags_[BWeakDecay], particleID.hasBottom());
                            update(flags_[CWeakDecay], particleID.hasCharm());
                            longlived = true;
                        }
                        // Check Tau, Ks and Lambda decay
                        update(flags_[TauDecay], pdgid == 15);
                        update(flags_[KsDecay], pdgid == 310);
                        update(flags_[LambdaDecay], pdgid == 3122);
                        update(flags_[Jpsi], pdgid == 443);
                        update(
                            flags_[LongLivedDecay],
                            !flags_[BWeakDecay] &&
                            !flags_[CWeakDecay] &&
                            !flags_[TauDecay] &&
                            !flags_[KsDecay] &&
                            !flags_[LambdaDecay] &&
                            !flags_[Jpsi] &&
                            longlived
                        );
                    }
                }
                update(
                    flags_[Interaction],
                    !flags_[BWeakDecay] &&
                    !flags_[CWeakDecay] &&
                    !flags_[LongLivedDecay] &&
                    !flags_[TauDecay] &&
                    !flags_[KsDecay] &&
                    !flags_[LambdaDecay] &&
                    !flags_[Jpsi]
                );
            }
            else
            {
                update(
                    flags_[Interaction],
                    process != G4::Undefined &&
                    process != G4::Unknown &&
                    process != G4::Primary &&
                    process != G4::Hadronic &&
                    process != G4::Conversions
                );
            }
        }
    }
}


void TrackClassifier::vertexInformation()
{
    // Get the main primary vertex from the list
    GeneratedPrimaryVertex const & genpv = genpvs_.back();

    // Get the generated history of the tracks
    TrackHistory::GenParticleTrail & genParticleTrail = const_cast<TrackHistory::GenParticleTrail &> (tracer_.genParticleTrail());

    // Vertex counter
    int counter = 0;

    // Unit transformation from mm to cm
    double const mm = 0.1;

    double oldX = genpv.x;
    double oldY = genpv.y;
    double oldZ = genpv.z;

    // Loop over the generated particles
    for (
        TrackHistory::GenParticleTrail::reverse_iterator iparticle = genParticleTrail.rbegin();
        iparticle != genParticleTrail.rend();
        ++iparticle
    )
    {
        // Look for those with production vertex
        HepMC::GenVertex * parent = (*iparticle)->production_vertex();
        if (parent)
        {
            HepMC::ThreeVector p = parent->point3d();

            double distance2   = pow(p.x() * mm - genpv.x, 2) + pow(p.y() * mm - genpv.y, 2) + pow(p.z() * mm - genpv.z, 2);
            double difference2 = pow(p.x() * mm - oldX, 2)    + pow(p.y() * mm - oldY, 2)    + pow(p.z() * mm - oldZ, 2);

            // std::cout << "Distance2 : " << distance2 << " (" << p.x() * mm << "," << p.y() * mm << "," << p.z() * mm << ")" << std::endl;
            // std::cout << "Difference2 : " << difference2 << std::endl;

            if ( difference2 > vertexClusteringSqDistance_ )
            {
                if ( distance2 > vertexClusteringSqDistance_ ) counter++;
                oldX = p.x() * mm;
                oldY = p.y() * mm;
                oldZ = p.z() * mm;
            }
        }
    }

    TrackHistory::SimParticleTrail & simParticleTrail = const_cast<TrackHistory::SimParticleTrail &> (tracer_.simParticleTrail());

    // Loop over the generated particles
    for (
        TrackHistory::SimParticleTrail::reverse_iterator iparticle = simParticleTrail.rbegin();
        iparticle != simParticleTrail.rend();
        ++iparticle
    )
    {
        // Look for those with production vertex
        TrackingParticle::Point p = (*iparticle)->vertex();

        double distance2   = pow(p.x() - genpv.x, 2) + pow(p.y() - genpv.y, 2) + pow(p.z() - genpv.z, 2);
        double difference2 = pow(p.x() - oldX, 2)    + pow(p.y() - oldY, 2)    + pow(p.z() - oldZ, 2);

        // std::cout << "Distance2 : " << distance2 << " (" << p.x() << "," << p.y() << "," << p.z() << ")" << std::endl;
        // std::cout << "Difference2 : " << difference2 << std::endl;

        if ( difference2 > vertexClusteringSqDistance_ )
        {
            if ( distance2 > vertexClusteringSqDistance_ ) counter++;
            oldX = p.x();
            oldY = p.y();
            oldZ = p.z();
        }
    }

    if ( !counter )
        flags_[PrimaryVertex] = true;
    else if ( counter == 1 )
        flags_[SecondaryVertex] = true;
    else
        flags_[TertiaryVertex] = true;
}


bool TrackClassifier::isFinalstateParticle(const HepMC::GenParticle * p)
{
    return !p->end_vertex() && p->status() == 1;
}


bool TrackClassifier::isCharged(const HepMC::GenParticle * p)
{
    const ParticleData * part = particleDataTable_->particle( p->pdg_id() );
    if (part)
        return part->charge()!=0;
    else
    {
        // the new/improved particle table doesn't know anti-particles
        return  particleDataTable_->particle( -p->pdg_id() ) != 0;
    }
}


void TrackClassifier::genPrimaryVertices()
{
    genpvs_.clear();

    const HepMC::GenEvent * event = mcInformation_->GetEvent();

    if (event)
    {
        int idx = 0;

        // Loop over the different GenVertex
        for ( HepMC::GenEvent::vertex_const_iterator ivertex = event->vertices_begin(); ivertex != event->vertices_end(); ++ivertex )
        {
            bool hasParentVertex = false;

            // Loop over the parents looking to see if they are coming from a production vertex
            for (
                HepMC::GenVertex::particle_iterator iparent = (*ivertex)->particles_begin(HepMC::parents);
                iparent != (*ivertex)->particles_end(HepMC::parents);
                ++iparent
            )
                if ( (*iparent)->production_vertex() )
                {
                    hasParentVertex = true;
                    break;
                }

            // Reject those vertices with parent vertices
            if (hasParentVertex) continue;

            // Get the position of the vertex
            HepMC::FourVector pos = (*ivertex)->position();

            double const mm = 0.1;

            GeneratedPrimaryVertex pv(pos.x()*mm, pos.y()*mm, pos.z()*mm);

            std::vector<GeneratedPrimaryVertex>::iterator ientry = genpvs_.begin();

            // Search for a VERY close vertex in the list
            for (; ientry != genpvs_.end(); ++ientry)
            {
                double distance2 = pow(pv.x - ientry->x, 2) + pow(pv.y - ientry->y, 2) + pow(pv.z - ientry->z, 2);
                if ( distance2 < vertexClusteringSqDistance_ )
                    break;
            }

            // Check if there is not a VERY close vertex and added to the list
            if (ientry == genpvs_.end())
                ientry = genpvs_.insert(ientry,pv);

            // Add the vertex barcodes to the new or existent vertices
            ientry->genVertex.push_back((*ivertex)->barcode());

            // Collect final state descendants
            for (
                HepMC::GenVertex::particle_iterator idecendants  = (*ivertex)->particles_begin(HepMC::descendants);
                idecendants != (*ivertex)->particles_end(HepMC::descendants);
                ++idecendants
            )
            {
                if (isFinalstateParticle(*idecendants))
                    if ( find(ientry->finalstateParticles.begin(), ientry->finalstateParticles.end(), (*idecendants)->barcode()) == ientry->finalstateParticles.end() )
                    {
                        ientry->finalstateParticles.push_back((*idecendants)->barcode());
                        HepMC::FourVector m = (*idecendants)->momentum();

                        ientry->ptot.setPx(ientry->ptot.px() + m.px());
                        ientry->ptot.setPy(ientry->ptot.py() + m.py());
                        ientry->ptot.setPz(ientry->ptot.pz() + m.pz());
                        ientry->ptot.setE(ientry->ptot.e() + m.e());
                        ientry->ptsq += m.perp() * m.perp();

                        if ( m.perp() > 0.8 && std::abs(m.pseudoRapidity()) < 2.5 && isCharged(*idecendants) ) ientry->nGenTrk++;
                    }
            }
            idx++;
        }
    }

    std::sort(genpvs_.begin(), genpvs_.end());
}
