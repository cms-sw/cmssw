import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
         'Geometry/CMSCommonData/data/rotations.xml',
         'Geometry/CMSCommonData/data/extend/cmsextent.xml',
         'Geometry/CMSCommonData/data/cms.xml',
         'Geometry/CMSCommonData/data/cmsMother.xml',
         'Geometry/ForwardCommonData/data/forward.xml',
         'Geometry/ForwardCommonData/data/totemMaterials.xml',
         'Geometry/ForwardCommonData/data/totemRotations.xml',
         'Geometry/ForwardCommonData/data/totemt1.xml',
         'Geometry/ForwardCommonData/data/totemt2.xml',
         'Geometry/ForwardCommonData/data/ionpump.xml',
         'Geometry/ForwardSimData/data/totemsensT1.xml',
         'Geometry/ForwardSimData/data/totemsensT2.xml',
         'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('cms:OCMS')
)
