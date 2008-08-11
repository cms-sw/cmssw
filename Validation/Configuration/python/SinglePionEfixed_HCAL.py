import FWCore.ParameterSet.Config as cms
def customise(process):

# HCAL geometry
    
    process.XMLIdealGeometryESSource.geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMother.xml', 
        'Geometry/CMSCommonData/data/caloBase.xml', 
        'Geometry/CMSCommonData/data/cmsCalo.xml', 
        'Geometry/CMSCommonData/data/muonBase.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/HcalCommonData/data/hcalalgo.xml', 
        'Geometry/HcalCommonData/data/hcalbarrelalgo.xml', 
        'Geometry/HcalCommonData/data/hcalendcapalgo.xml', 
        'Geometry/HcalCommonData/data/hcalouteralgo.xml', 
        'Geometry/HcalCommonData/data/hcalforwardalgo.xml', 
        'Geometry/HcalCommonData/data/hcalforwardfibre.xml', 
        'Geometry/HcalCommonData/data/hcalforwardmaterial.xml', 
        'Geometry/HcalCommonData/data/hcalsens.xml', 
        'Geometry/HcalSimData/data/CaloUtil.xml', 
        'Geometry/HcalSimData/data/HcalProdCuts.xml', 
        'Geometry/CMSCommonData/data/FieldParameters.xml')

    process.CaloGeometryBuilder.SelectedCalos = cms.vstring ('HCAL','TOWER')

# extend the particle gun acceptance

    process.source.AddAntiParticle = cms.untracked.bool(False)

# no magnetic field

    process.g4SimHits.UseMagneticField = cms.bool(False)
    process.UniformMagneticFieldESProducer = cms.ESProducer("UniformMagneticFieldESProducer",
                                                            ZFieldInTesla = cms.double(0.0)
                                                                )

    process.prefer("UniformMagneticFieldESProducer") 

# modify the content

    process.output.outputCommands.append("keep *_simHcalUnsuppressedDigis_*_*")

# user schedule: use only calorimeters digitization and local reconstruction

    del process.schedule[:]

    process.schedule.append(process.generation_step)
    process.schedule.append(process.simulation_step)

    process.ecalWeightUncalibRecHit.EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis")
    process.ecalWeightUncalibRecHit.EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis")
    process.ecalPreshowerRecHit.ESdigiCollection = cms.InputTag("simEcalPreshowerDigis")

    process.hbhereco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")
    process.horeco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")
    process.hfreco.digiLabel = cms.InputTag("simHcalUnsuppressedDigis")

    process.local_digireco = cms.Path(process.mix * process.calDigi * process.calolocalreco * process.caloTowersRec)

    process.schedule.append(process.local_digireco)

    process.load("Validation/Configuration/hcalSimValid_cff")
    process.local_validation = cms.Path(process.hcalSimValid*process.MEtoEDMConverter)
    process.schedule.append(process.local_validation)

    process.schedule.append(process.out_step)
 
# drop the plain root file outputs of all analyzers
# Note: all the validation "analyzers" are EDFilters!
    for filter in (getattr(process,f) for f in process.filters_()):
        if hasattr(filter,"outputFile"):
            filter.outputFile=""
        #Catch the problem with valid_HB.root that uses OutputFile instead of outputFile
        if hasattr(filter,"OutputFile"):
            filter.OutputFile=""
       
    return(process)
