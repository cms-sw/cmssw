import FWCore.ParameterSet.Config as cms

process = cms.Process("HFSHOWERLIBRARY")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.TFileService = cms.Service("TFileService",fileName = cms.string('HFShowerLibrary.root') )

process.photon = cms.EDAnalyzer('HcalForwardLibWriter',
    HcalForwardLibWriterParameters = cms.PSet(
	FileName = cms.FileInPath('SimG4CMS/ShowerLibraryProducer/data/fileList.txt'),
	Nbins = cms.int32(16),
	Nshowers = cms.int32(5000)
    )
)

process.p = cms.Path(process.photon)
