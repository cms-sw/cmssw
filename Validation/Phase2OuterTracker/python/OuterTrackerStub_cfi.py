import FWCore.ParameterSet.Config as cms

OuterTrackerStub = cms.EDAnalyzer('OuterTrackerStub',
    
    TopFolderName = cms.string('Phase2OuterTrackerV'),
    TTStubs       = cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
    TTStubMCTruth = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),    
    verbosePlots   = cms.untracked.bool(False),


# Stub Layers
    TH1TTStub_Layer = cms.PSet(
        Nbinsx = cms.int32(6),
        xmax = cms.double(6.5),                      
        xmin = cms.double(0.5)
        ),
    
# Stub Disks
    TH1TTStub_Disk= cms.PSet(
        Nbinsx = cms.int32(5),
        xmax = cms.double(5.5),                      
        xmin = cms.double(0.5)
        ),
    
# Stub Rings
    TH1TTStub_Ring = cms.PSet(
        Nbinsx = cms.int32(16),
        xmax = cms.double(16.5),                      
        xmin = cms.double(0.5)
        ),

# Stub Eta
    TH1TTStub_Eta = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3),                      
        xmin = cms.double(-3)
        ),          

)
