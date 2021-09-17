import FWCore.ParameterSet.Config as cms
from Validation.SiTrackerPhase2V.Phase2ITValidateCluster_cfi import Phase2ITValidateCluster
clusterValidIT = Phase2ITValidateCluster.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(clusterValidIT,
    InnerTrackerDigiSimLinkSource = "mixData:PixelDigiSimLink",
)
