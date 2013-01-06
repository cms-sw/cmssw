
#include <math.h>
#include <cstdlib>
#include <iostream>

#include "HepPDT/ParticleID.hh"

#include "SimTracker/TrackHistory/interface/TrackClassifier.h"

#define update(a, b) do { (a) = (a) | (b); } while(0)

TrackClassifier::TrackClassifier(edm::ParameterSet const & config) : TrackCategories(),
        hepMCLabel_( config.getUntrackedParameter<edm::InputTag>("hepMC") ),
        beamSpotLabel_( config.getUntrackedParameter<edm::InputTag>("beamSpot") ),
        tracer_(config),
        quality_(config)
{
    // Set the history depth after hadronization
    tracer_.depth(-2);

    // Set the maximum d0pull for the bad category
    badPull_ = config.getUntrackedParameter<double>("badPull");

    // Set the minimum decay length for detecting long decays
    longLivedDecayLength_ = config.getUntrackedParameter<double>("longLivedDecayLength");

    // Set the distance for clustering vertices
    float vertexClusteringDistance = config.getUntrackedParameter<double>("vertexClusteringDistance");
    vertexClusteringSqDistance_ = vertexClusteringDistance * vertexClusteringDistance;

    // Set the number of innermost layers to check for bad hits
    numberOfInnerLayers_ = config.getUntrackedParameter<unsigned int>("numberOfInnerLayers");

    // Set the minimum number of simhits in the tracker
    minTrackerSimHits_ = config.getUntrackedParameter<unsigned int>("minTrackerSimHits");
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
    event.getByLabel(beamSpotLabel_, beamSpot_);

    // Transient track builder
    setup.get<TransientTrackRecord>().get("TransientTrackBuilder", transientTrackBuilder_);

    // Create the list of primary vertices associated to the event
    genPrimaryVertices();

    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHand;
    setup.get<IdealGeometryRecord>().get(tTopoHand);
    tTopo_=tTopoHand.product();
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

    try
    {
        TSCPBuilderNoMaterial tscpBuilder;
        TrajectoryStateClosestToPoint tsAtClosestApproach = tscpBuilder(
                    ftsAtProduction,
                    GlobalPoint(beamSpot_->x0(), beamSpot_->y0(), beamSpot_->z0())
                );

        GlobalVector v = tsAtClosestApproach.theState().position()
                         - GlobalPoint(beamSpot_->x0(), beamSpot_->y0(), beamSpot_->z0());
        GlobalVector p = tsAtClosestApproach.theState().momentum();

        // Simulated dxy
        double dxySim = -v.x()*sin(p.phi()) + v.y()*cos(p.phi());

        // Simulated dz
        double dzSim = v.z() - (v.x()*p.x() + v.y()*p.y())*p.z()/p.perp2();

        // Calculate the dxy pull
        double dxyPull = std::abs(
                             track->dxy( reco::TrackBase::Point(beamSpot_->x0(), beamSpot_->y0(), beamSpot_->z0()) ) - dxySim
                         ) / track->dxyError();

        // Calculate the dx pull
        double dzPull = std::abs(
                            track->dz( reco::TrackBase::Point(beamSpot_->x0(), beamSpot_->y0(), beamSpot_->z0()) ) - dzSim
                        ) / track->dzError();

        // Return true if d0Pull > badD0Pull sigmas
        flags_[Bad] = (dxyPull > badPull_ || dzPull > badPull_);

    }
    catch (cms::Exception exception)
    {
        flags_[Bad] = true;
    }
}


void TrackClassifier::simulationInformation()
{
    // Get the event id for the initial TP.
    EncodedEventId eventId = tracer_.simParticle()->eventId();
    // Check for signal events
    flags_[SignalEvent] = !eventId.bunchCrossing() && !eventId.event();
    // Check for muons
    flags_[Muon] = (abs(tracer_.simParticle()->pdgId()) == 13);
    // Check for the number of psimhit in tracker
    flags_[TrackerSimHits] = tracer_.simParticle()->matchedHit() >= (int)minTrackerSimHits_;
}


void TrackClassifier::qualityInformation(reco::TrackBaseRef const & track)
{
    // run the hit-by-hit reconstruction quality analysis
  quality_.evaluate(tracer_.simParticleTrail(), track, tTopo_);

    unsigned int maxLayers = std::min(numberOfInnerLayers_, quality_.numberOfLayers());

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
    // pdgid of the "in" particle to the production vertex
    int pdgid = 0;

    // Get the generated particles from track history
    TrackHistory::GenParticleTrail const & genParticleTrail = tracer_.genParticleTrail();

    // Loop over the generated particles
    for (TrackHistory::GenParticleTrail::const_iterator iparticle = genParticleTrail.begin(); iparticle != genParticleTrail.end(); ++iparticle)
    {
        // Get the source vertex for the particle
        HepMC::GenVertex * productionVertex = (*iparticle)->production_vertex();

        // Get the pointer to the vertex by removing the const-ness (no const methos in HepMC::GenVertex)
        // HepMC::GenVertex * vertex = const_cast<HepMC::GenVertex *>(*ivertex);

        // Check for a non-null pointer to the production vertex
        if (productionVertex)
        {
            // Only case track history will navegate (one in or source particle per vertex)
            if ( productionVertex->particles_in_size() == 1 )
            {
                // Look at the pdgid of the first "in" particle to the vertex
                pdgid = std::abs((*productionVertex->particles_in_const_begin())->pdg_id());
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
                        // Check if their life time is bigger than longLivedDecayLength_
                        if ( particleData->lifetime() > longLivedDecayLength_ )
                            update(flags_[LongLivedDecay], true);
                        // Check for B and C weak decays
                        update(flags_[BWeakDecay], particleID.hasBottom());
                        update(flags_[CWeakDecay], particleID.hasCharm());
                        // Check for B and C pure leptonic decay
                        int daughterId = abs((*iparticle)->pdg_id());
                        update(flags_[FromBWeakDecayMuon], particleID.hasBottom() && daughterId == 13);
                        update(flags_[FromCWeakDecayMuon], particleID.hasCharm() && daughterId == 13);
                    }
                    // Check Tau, Ks and Lambda decay
                    update(flags_[ChargePionDecay], pdgid == 211);
                    update(flags_[ChargeKaonDecay], pdgid == 321);
                    update(flags_[TauDecay], pdgid == 15);
                    update(flags_[KsDecay], pdgid == 310);
                    update(flags_[LambdaDecay], pdgid == 3122);
                    update(flags_[JpsiDecay], pdgid == 443);
                    update(flags_[XiDecay], pdgid == 3312);
                    update(flags_[SigmaPlusDecay], pdgid == 3222);
                    update(flags_[SigmaMinusDecay], pdgid == 3112);
                }
            }
        }
    }
    // Decays in flight
    update(flags_[FromChargePionMuon], flags_[Muon] && flags_[ChargePionDecay]);
    update(flags_[FromChargeKaonMuon], flags_[Muon] && flags_[ChargeKaonDecay]);
    update(flags_[DecayOnFlightMuon], (flags_[FromChargePionMuon] || flags_[FromChargeKaonMuon]));
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

        // Check for the number of psimhit if different from zero
        if ((*iparticle)->trackPSimHit().empty()) continue;

        // Collect the G4 process of the first psimhit (it should be the same for all of them)
        unsigned short process = (*iparticle)->pSimHit_begin()->processType();

        // Flagging all the different processes

        update(
            flags_[KnownProcess],
            process != G4::Undefined &&
            process != G4::Unknown &&
            process != G4::Primary
        );

        update(flags_[UndefinedProcess], process == G4::Undefined);
        update(flags_[UnknownProcess], process == G4::Unknown);
        update(flags_[PrimaryProcess], process == G4::Primary);
        update(flags_[HadronicProcess], process == G4::Hadronic);
        update(flags_[DecayProcess], process == G4::Decay);
        update(flags_[ComptonProcess], process == G4::Compton);
        update(flags_[AnnihilationProcess], process == G4::Annihilation);
        update(flags_[EIoniProcess], process == G4::EIoni);
        update(flags_[HIoniProcess], process == G4::HIoni);
        update(flags_[MuIoniProcess], process == G4::MuIoni);
        update(flags_[PhotonProcess], process == G4::Photon);
        update(flags_[MuPairProdProcess], process == G4::MuPairProd);
        update(flags_[ConversionsProcess], process == G4::Conversions);
        update(flags_[EBremProcess], process == G4::EBrem);
        update(flags_[SynchrotronRadiationProcess], process == G4::SynchrotronRadiation);
        update(flags_[MuBremProcess], process == G4::MuBrem);
        update(flags_[MuNuclProcess], process == G4::MuNucl);

        // Get particle type
        HepPDT::ParticleID particleID(pdgid);

        // Check if the particle type is valid one
        if (particleID.isValid())
        {
            // Get particle data
            ParticleData const * particleData = particleDataTable_->particle(particleID);
            // Special treatment for decays
            if (process == G4::Decay)
            {
                // Check if the particle exist in the table
                if (particleData)
                {
                    // Check if their life time is bigger than 1e-14
                    if ( particleDataTable_->particle(particleID)->lifetime() > longLivedDecayLength_ )
                        update(flags_[LongLivedDecay], true);

                    // Check for B and C weak decays
                    update(flags_[BWeakDecay], particleID.hasBottom());
                    update(flags_[CWeakDecay], particleID.hasCharm());

                    // Check for B or C pure leptonic decays
                    int daughtId = abs((*iparticle)->pdgId());
                    update(flags_[FromBWeakDecayMuon], particleID.hasBottom() && daughtId == 13);
                    update(flags_[FromCWeakDecayMuon], particleID.hasCharm() && daughtId == 13);
                }
                // Check decays
                update(flags_[ChargePionDecay], pdgid == 211);
                update(flags_[ChargeKaonDecay], pdgid == 321);
                update(flags_[TauDecay], pdgid == 15);
                update(flags_[KsDecay], pdgid == 310);
                update(flags_[LambdaDecay], pdgid == 3122);
                update(flags_[JpsiDecay], pdgid == 443);
                update(flags_[XiDecay], pdgid == 3312);
                update(flags_[OmegaDecay], pdgid == 3334);
                update(flags_[SigmaPlusDecay], pdgid == 3222);
                update(flags_[SigmaMinusDecay], pdgid == 3112);
            }
        }
    }
    // Decays in flight
    update(flags_[FromChargePionMuon], flags_[Muon] && flags_[ChargePionDecay]);
    update(flags_[FromChargeKaonMuon], flags_[Muon] && flags_[ChargeKaonDecay]);
    update(flags_[DecayOnFlightMuon], flags_[FromChargePionMuon] || flags_[FromChargeKaonMuon]);
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
