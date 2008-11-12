import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
 	geomXMLFiles = (cms.vstring('Geometry/CMSCommonData/data/materials.xml',
		'Geometry/ForwardCommonData/data/zdcmaterials.xml',
		'Geometry/ForwardCommonData/data/lumimaterials.xml',
		'Geometry/ForwardCommonData/data/zdcrotations.xml',
		'Geometry/ForwardCommonData/data/lumirotations.xml',
		'Geometry/ForwardCommonData/data/zdc.xml',
		'Geometry/ForwardCommonData/data/zdclumi.xml',
		'Geometry/ForwardCommonData/data/zdcworld.xml',
		'SimG4CMS/Forward/test/data/CaloUtil.xml',
		'Geometry/ForwardSimData/data/zdcsens.xml',
		'Geometry/ForwardSimData/data/zdcProdCuts.xml')
	),
	rootNodeName = cms.string('zdcworld:ZDCWorld')
)
