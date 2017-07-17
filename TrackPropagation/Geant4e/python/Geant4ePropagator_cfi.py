import FWCore.ParameterSet.Config as cms

# Default Geant4e propagator setup, uses Muons as particle hypotesis
# ParticleName can be any particle described in the Geant4 documentation
# http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch05s03.html
# The chargen ( e.g. mu+ or mu- ) will be added by the propagator, depending on the fitted track's charge
Geant4ePropagator = cms.ESProducer("GeantPropagatorESProducer",
                                   ComponentName = cms.string("Geant4ePropagator"),
                                   PropagationDirection=cms.string("alongMomentum"),
                                   ParticleName=cms.string("mu")
                                   )
