import FWCore.ParameterSet.Config as cms

OuterTrackerStub = cms.EDAnalyzer('OuterTrackerStub',
    
    TopFolderName = cms.string('Phase2OuterTrackerV'),
    
# Stub Stacks
    TH1TTStub_Stack = cms.PSet(
        Nbinsx = cms.int32(6),
        xmax = cms.double(6.5),                      
        xmin = cms.double(0.5)
        ),

# Stub Eta
    TH1TTStub_Eta = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3),                      
        xmin = cms.double(-3)
        ),          

)
