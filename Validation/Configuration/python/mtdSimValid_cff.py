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
from Validation.MtdValidation.btlDigiSoAHitsValid_cfi import btlDigiSoAHitsValid
from Validation.MtdValidation.btlLocalRecoValid_cfi import btlLocalRecoValid
from Validation.MtdValidation.btlLocalRecoSoAValid_cfi import btlLocalRecoSoAValid
from Validation.MtdValidation.etlSimHitsValid_cfi import etlSimHitsValid
from Validation.MtdValidation.etlDigiHitsValid_cfi import etlDigiHitsValid
from Validation.MtdValidation.etlDigiSoAHitsValid_cfi import etlDigiSoAHitsValid
from Validation.MtdValidation.etlLocalRecoValid_cfi import etlLocalRecoValid
from Validation.MtdValidation.etlLocalRecoSoAValid_cfi import etlLocalRecoSoAValid
from Validation.MtdValidation.mtdTracksValid_cfi import mtdTracksValid
from Validation.MtdValidation.vertices4DValid_cff import vertices4DValid

mtdSimValid  = cms.Sequence(btlSimHitsValid  + etlSimHitsValid )
mtdDigiValid = cms.Sequence(btlDigiHitsValid + btlDigiSoAHitsValid + etlDigiHitsValid + etlDigiSoAHitsValid)
mtdRecoValid = cms.Sequence(mtdAssociationProducers + btlLocalRecoValid + btlLocalRecoSoAValid + etlLocalRecoValid + etlLocalRecoSoAValid + mtdTracksValid + vertices4DValid)
