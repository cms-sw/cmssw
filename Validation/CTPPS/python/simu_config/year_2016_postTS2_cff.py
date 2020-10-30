import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2016_cff import *

profile_2016_postTS2=profile.clone()

#LHCInfo
profile_2016_postTS2.ctppsLHCInfo.xangleBetaStarHistogramObject=cms.string("2016_postTS2/h2_betaStar_vs_xangle")

#Optics
profile_2016_postTS2.ctppsOpticalFunctions.opticalFunctions = cms.VPSet(
    		cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2016_postTS2/version2/140urad.root") )
  	)



profile_2016_postTS2.ctppsOpticalFunctions.scoringPlanes = cms.VPSet(
	      # z in cm
	      cms.PSet( rpId = cms.uint32(0x76100000), dirName = cms.string("XRPH_C6L5_B2"), z = cms.double(-20382.6) ),  # RP 002, strip
	      cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003, strip
	      cms.PSet( rpId = cms.uint32(0x77100000), dirName = cms.string("XRPH_C6R5_B1"), z = cms.double(+20382.6) ),  # RP 102, strip
	      cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103, strip
  	)

#geometry
profile_2016_postTS2.xmlIdealGeometry.geomXMLFiles = totemGeomXMLFiles + ctppsDiamondGeomXMLFiles + ctppsUFSDGeomXMLFiles + ctppsPixelGeomXMLFiles
profile_2016_postTS2.xmlIdealGeometry.rootNodeName = cms.string('cms:CMSE')

#alignment
profile_2016_postTS2.ctppsRPAlignmentCorrectionsDataXML.RealFiles=cms.vstring("Validation/CTPPS/alignment/2016_postTS2.xml")
profile_2016_postTS2.ctppsRPAlignmentCorrectionsDataXML.MisalignedFiles=cms.vstring("Validation/CTPPS/alignment/2016_postTS2.xml")


#direct simu data
profile_2016_postTS2.ctppsDirectSimuData.useEmpiricalApertures=cms.bool(True)
profile_2016_postTS2.ctppsDirectSimuData.empiricalAperture45=cms.string("6.10374E-05+(([xi]<0.113491)*0.00795942+([xi]>=0.113491)*0.01935)*([xi]-0.113491)")
profile_2016_postTS2.ctppsDirectSimuData.empiricalAperture56=cms.string("([xi]-0.110)/130.0")
profile_2016_postTS2.ctppsDirectSimuData.timeResolutionDiamonds45=cms.string("0.200")
profile_2016_postTS2.ctppsDirectSimuData.timeResolutionDiamonds56=cms.string("0.200")


profile_2016_postTS2.xmlIdealGeometry.geomXMLFiles.append("Geometry/VeryForwardData/data/2016_ctpps_15sigma_margin0/RP_Dist_Beam_Cent.xml")

