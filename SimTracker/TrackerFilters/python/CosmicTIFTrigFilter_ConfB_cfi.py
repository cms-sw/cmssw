import FWCore.ParameterSet.Config as cms

fil = cms.EDFilter("CosmicTIFTrigFilter",
    filter = cms.bool(True),
    #trigger configurations: A==1, B==2, C==3
    trig_conf = cms.int32(2)
)


# foo bar baz
# evkRk57DoBLUd
# kcJHP2XtRhaW6
