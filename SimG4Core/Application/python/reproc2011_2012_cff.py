import FWCore.ParameterSet.Config as cms

def customiseG4(process):

  if hasattr(process,'g4SimHits'):
    process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/QGSP_FTFP_BERT_EML_New')
    # use HF shower library instead of GFlash parameterization
    process.g4SimHits.HCalSD.UseShowerLibrary = cms.bool(True)
    process.g4SimHits.HCalSD.UseParametrize = cms.bool(False)
    process.g4SimHits.HCalSD.UsePMTHits = cms.bool(False)
    process.g4SimHits.HCalSD.UseFibreBundleHits = cms.bool(False)
    process.g4SimHits.HFShowerLibrary.FileName  = 'SimG4CMS/Calo/data/HFShowerLibrary_oldpmt_eta4_16en.root'
    process.g4SimHits.HFShowerLibrary.BranchPost= ''
    process.g4SimHits.HFShowerLibrary.BranchPre = ''
    process.g4SimHits.HFShowerLibrary.BranchEvt = ''

    return(process)
