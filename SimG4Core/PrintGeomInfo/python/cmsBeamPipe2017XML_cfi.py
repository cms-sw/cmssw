import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/cmsextent.xml',
        'Geometry/CMSCommonData/data/cms/2017/v1/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/beampipe/2017/v1/beampipe.xml',
        'Geometry/CMSCommonData/data/cmsBeam.xml',
#        'Geometry/DTGeometryBuilder/data/dtSpecsFilter.xml',
#        'Geometry/CSCGeometryBuilder/data/cscSpecsFilter.xml',
#        'Geometry/CSCGeometryBuilder/data/cscSpecs.xml',
#        'Geometry/RPCGeometryBuilder/data/RPCSpecs.xml',
#        'Geometry/GEMGeometryBuilder/data/GEMSpecsFilter17.xml',
#        'Geometry/GEMGeometryBuilder/data/v4/GEMSpecs.xml',
        'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


