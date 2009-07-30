from Configuration.StandardSequences.Geometry_cff import *
from Configuration.StandardSequences.MagneticField_cff import *


from JetMETCorrections.Configuration.JetPlusTrackCorrections_cff import *
from JetMETCorrections.Configuration.ZSPJetCorrections152_cff import *
recoJPTJets = cms.Sequence(
        ZSPJetCorrections+JetPlusTrackCorrections
        )
