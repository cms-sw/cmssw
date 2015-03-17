def customise_HFSLrun1(process):
    #enable Run 1 HF shower library
    process.g4SimHits.HCalSD.UseShowerLibrary   = True
    process.g4SimHits.HCalSD.UseParametrize     = False
    process.g4SimHits.HCalSD.UsePMTHits         = False
    process.g4SimHits.HCalSD.UseFibreBundleHits = False
    process.g4SimHits.HFShowerLibrary.FileName  = 'SimG4CMS/Calo/data/HFShowerLibrary_oldpmt_noatt_eta4_16en_v3.root'  
    process.g4SimHits.HFShowerLibrary.BranchPost= ''
    process.g4SimHits.HFShowerLibrary.BranchPre = ''
    process.g4SimHits.HFShowerLibrary.BranchEvt = ''
    return process
