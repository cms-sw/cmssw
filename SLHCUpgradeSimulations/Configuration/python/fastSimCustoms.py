import FWCore.ParameterSet.Config as cms

def customise_fastSimPostLS1(process):

    if hasattr(process,'famosSimHits'):
       process=customise_fastSimProducer(process)
       
    return process


def customise_fastSimProducer(process): 

    # enable 2015 HF shower library
    process.famosSimHits.Calorimetry.HFShowerLibrary.FileName = cms.FileInPath('SimG4CMS/Calo/data/HFShowerLibrary_npmt_noatt_eta4_16en_v4.root')
    
    return process


