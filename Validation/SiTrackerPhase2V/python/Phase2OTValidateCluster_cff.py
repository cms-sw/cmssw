import FWCore.ParameterSet.Config as cms
from Validation.SiTrackerPhase2V.Phase2OTValidateCluster_cfi import Phase2OTValidateCluster
clusterValidOT = Phase2OTValidateCluster.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(clusterValidOT,
    OuterTrackerDigiSimLinkSource = "mixData:Phase2OTDigiSimLink",
)
