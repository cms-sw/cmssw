import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

gemSimHarvesting = DQMEDHarvester("MuonGEMHitsHarvestor")
MuonGEMHitsPostProcessors = cms.Sequence( gemSimHarvesting ) 
# foo bar baz
# 9J1oZi1pTvFxD
# UJVFA769sADni
