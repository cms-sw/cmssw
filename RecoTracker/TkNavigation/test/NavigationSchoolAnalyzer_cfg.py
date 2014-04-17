import FWCore.ParameterSet.Config as cms

process = cms.Process("NavigationSchoolAnalyze")

#process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DPixel10DReco_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'STARTUP_V4::All'
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoTracker.TkNavigation.NavigationSchoolESProducer_cff")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.navigationSchoolAnalyzer = cms.EDAnalyzer("NavigationSchoolAnalyzer",
    #navigationSchoolName = cms.string('BeamHaloNavigationSchool')
    navigationSchoolName = cms.string('SimpleNavigationSchool')
)

process.p = cms.Path(process.navigationSchoolAnalyzer)

process.trackerNumberingSLHCGeometry.layerNumberPXB = cms.uint32(20)
process.trackerTopologyConstants.pxb_layerStartBit = cms.uint32(20)
process.trackerTopologyConstants.pxb_ladderStartBit = cms.uint32(12)
process.trackerTopologyConstants.pxb_moduleStartBit = cms.uint32(2)
process.trackerTopologyConstants.pxb_layerMask = cms.uint32(15)
process.trackerTopologyConstants.pxb_ladderMask = cms.uint32(255)
process.trackerTopologyConstants.pxb_moduleMask = cms.uint32(1023)
process.trackerTopologyConstants.pxf_diskStartBit = cms.uint32(18)
process.trackerTopologyConstants.pxf_bladeStartBit = cms.uint32(12)
process.trackerTopologyConstants.pxf_panelStartBit = cms.uint32(10)
process.trackerTopologyConstants.pxf_moduleMask = cms.uint32(255)

#in case you want to distinguish between inner and outer tracker
#perl -p -i -e "s/PixelBarrel_5/OuterBarrel_1/g" detailedInfo.log
#perl -p -i -e "s/PixelBarrel_6/OuterBarrel_2/g" detailedInfo.log
#perl -p -i -e "s/PixelBarrel_7/OuterBarrel_3/g" detailedInfo.log
#perl -p -i -e "s/PixelBarrel_8/OuterBarrel_4/g" detailedInfo.log
#perl -p -i -e "s/PixelBarrel_9/OuterBarrel_5/g" detailedInfo.log
#perl -p -i -e "s/PixelBarrel_10/OuterBarrel_6/g" detailedInfo.log
#perl -p -i -e "s/PixelEndcapMinus_11/OuterEndcapMinus_1/g" detailedInfo.log
#perl -p -i -e "s/PixelEndcapMinus_12/OuterEndcapMinus_2/g" detailedInfo.log
#perl -p -i -e "s/PixelEndcapMinus_13/OuterEndcapMinus_3/g" detailedInfo.log
#perl -p -i -e "s/PixelEndcapMinus_14/OuterEndcapMinus_4/g" detailedInfo.log
#perl -p -i -e "s/PixelEndcapMinus_15/OuterEndcapMinus_5/g" detailedInfo.log
#perl -p -i -e "s/PixelEndcapPlus_11/OuterEndcapPlus_1/g" detailedInfo.log
#perl -p -i -e "s/PixelEndcapPlus_12/OuterEndcapPlus_2/g" detailedInfo.log
#perl -p -i -e "s/PixelEndcapPlus_13/OuterEndcapPlus_3/g" detailedInfo.log
#perl -p -i -e "s/PixelEndcapPlus_14/OuterEndcapPlus_4/g" detailedInfo.log
#perl -p -i -e "s/PixelEndcapPlus_15/OuterEndcapPlus_5/g" detailedInfo.log
