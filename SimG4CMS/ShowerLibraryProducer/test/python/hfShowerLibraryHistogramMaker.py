import FWCore.ParameterSet.Config as cms

process = cms.Process("ANA")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

process.source = cms.Source("EmptySource",
  numberEventsInRun = cms.untracked.uint32(1),
  firstRun = cms.untracked.uint32(1)
)


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('hfShowerLibHistograms.root')
)

process.ana = cms.EDAnalyzer("AnalyzeTuples",
     HFShowerLibrary = cms.PSet(
         FileName        = cms.FileInPath('myOutputFile.root'),
#FileName        = cms.FileInPath('SimG4CMS/ShowerLibraryProducer/test/python/hfshowerlibrary_lhep_140_edm.root'),
         BackProbability = cms.double(0.2),
         TreeEMID        = cms.string('emParticles'),
         TreeHadID       = cms.string('hadParticles'),
         Verbosity       = cms.untracked.bool(False),
         BranchPost      = cms.untracked.string('_HFSHOWERLIBRARY.obj'),
         #BranchEvt       = cms.untracked.string('HFShowerLibraryEventInfos_hfshowerlib_HFShowerLibraryEventInfo'),
        # BranchPre       = cms.untracked.string('ints_hfshowerlib_')
          BranchEvt       = cms.untracked.string('HFShowerLibraryEventInfos_photon_HFShowerLibraryEventInfo'),
          BranchPre       = cms.untracked.string('HFShowerPhotons_photon_')
     )
)

process.p  = cms.Path(process.ana)
