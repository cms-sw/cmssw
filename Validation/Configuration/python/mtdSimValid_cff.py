import FWCore.ParameterSet.Config as cms

# --- Cluster associations maps producers
from SimFastTiming.MtdAssociatorProducers.mtdRecoClusterToSimLayerClusterAssociatorByHits_cfi import mtdRecoClusterToSimLayerClusterAssociatorByHits
from SimFastTiming.MtdAssociatorProducers.mtdRecoClusterToSimLayerClusterAssociation_cfi import mtdRecoClusterToSimLayerClusterAssociation
from SimFastTiming.MtdAssociatorProducers.mtdSimLayerClusterToTPAssociatorByTrackId_cfi import mtdSimLayerClusterToTPAssociatorByTrackId
from SimFastTiming.MtdAssociatorProducers.mtdSimLayerClusterToTPAssociation_cfi import mtdSimLayerClusterToTPAssociation
mtdAssociationProducers = cms.Sequence( mtdRecoClusterToSimLayerClusterAssociatorByHits +
                                        mtdRecoClusterToSimLayerClusterAssociation +
                                        mtdSimLayerClusterToTPAssociatorByTrackId +
                                        mtdSimLayerClusterToTPAssociation
                                       )

# MTD validation sequences
from Validation.MtdValidation.btlSimHitsValid_cfi import btlSimHitsValid
from Validation.MtdValidation.btlDigiHitsValid_cfi import btlDigiHitsValid
from Validation.MtdValidation.btlLocalRecoValid_cfi import btlLocalRecoValid
from Validation.MtdValidation.etlLocalRecoValid_cfi import etlLocalRecoValid
from Validation.MtdValidation.etlSimHitsValid_cfi import etlSimHitsValid
from Validation.MtdValidation.etlDigiHitsValid_cfi import etlDigiHitsValid
from Validation.MtdValidation.mtdTracksValid_cfi import mtdTracksValid
from Validation.MtdValidation.vertices4DValid_cff import vertices4DValid

mtdSimValid  = cms.Sequence(btlSimHitsValid  + etlSimHitsValid )
mtdDigiValid = cms.Sequence(btlDigiHitsValid + etlDigiHitsValid)
mtdRecoValid = cms.Sequence(mtdAssociationProducers + btlLocalRecoValid  + etlLocalRecoValid + mtdTracksValid + vertices4DValid)

