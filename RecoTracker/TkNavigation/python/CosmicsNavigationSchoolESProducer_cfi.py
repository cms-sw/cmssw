#do not comment that line
import FWCore.ParameterSet.Config as cms

#cosmicsNavigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
#    ComponentName = cms.string('CosmicNavigationSchool')
#)

from RecoTracker.TkNavigation.skippingLayerCosmicNavigationSchoolESProducer_cfi import skippingLayerCosmicNavigationSchoolESProducer as _skippingLayerCosmicNavigationSchoolESProducer
cosmicsNavigationSchoolESProducer = _skippingLayerCosmicNavigationSchoolESProducer.clone(
    ComponentName = 'CosmicNavigationSchool',
    noPXB = False,
    noPXF = False,
    noTIB = False,
    noTID = False,
    noTOB = False,
    noTEC = False,
    selfSearch = True,
    allSelf = True
)


