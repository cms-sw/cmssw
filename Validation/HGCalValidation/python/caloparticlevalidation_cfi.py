import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.caloparticlevalidationDefault_cfi import caloparticlevalidationDefault as _caloparticlevalidationDefault
caloparticlevalidation = _caloparticlevalidationDefault.clone()

# TODO: The following would be needed to use the signal+pileup
# CaloParticles for premixing. However, the code uses SimVertices, and
# - we don't propagate pileup SimVertices (actually we don't do that even in classical mixing?)
# - the code will either produce garbage or throw an exception
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(caloparticlevalidation,
    caloParticles = "mixData:MergedCaloTruth"
)
