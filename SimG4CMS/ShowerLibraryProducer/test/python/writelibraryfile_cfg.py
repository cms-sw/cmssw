import FWCore.ParameterSet.Config as cms

process = cms.Process("HFSHOWERLIBRARY")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))


process.photon = cms.EDProducer('HcalForwardLibWriter',
    HcalForwardLibWriterParameters = cms.PSet(
	FileName = cms.FileInPath('SimG4CMS/ShowerLibraryProducer/data/fileList.txt')
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
)

process.p = cms.Path(process.photon)
process.e = cms.EndPath(process.out)
