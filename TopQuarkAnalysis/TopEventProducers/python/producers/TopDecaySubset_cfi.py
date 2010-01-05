import FWCore.ParameterSet.Config as cms

#
# module to make a persistent copy of the genParticles from the top decay,
# using either status 2 equivalent particles (default) or status 3 particles
#
decaySubset = cms.EDProducer("TopDecaySubset",
  ## input particle collection of type edm::View<reco::GenParticle>
  src = cms.InputTag("genParticles"),
  ## define fill mode. The following modes are available:
  ## 'kStable' : status 2 equivalents (after parton shower) are
  ##             calculated and saved (as status 2 particles)
  ## 'kME'     : status 3 particles (from matrix element, before
  ##             parton shower) are saved (as status 3 particles)
  fillMode = cms.string("kStable"),
  ## choose whether to save additionally radiated gluons in the
  ## decay chain or not
  addRadiation = cms.bool(True)
)

