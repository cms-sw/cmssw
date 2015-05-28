import FWCore.ParameterSet.Config as cms
import SLHCUpgradeSimulations.Configuration.customise_PFlow as customise_PFlow

#GEN-SIM so far...
def customise(process):
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    n=0
    if hasattr(process,'reconstruction') or hasattr(process,'dqmoffline_step'):
        if hasattr(process,'mix'):
            if hasattr(process.mix,'input'):
                n=process.mix.input.nbPileupEvents.averageNumber.value()
        else:
            print 'phase1TkCustoms requires a --pileup option to cmsDriver to run the reconstruction/dqm'
            print 'Please provide one!'
            sys.exit(1)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process,float(n))
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process,n)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process,float(n))
    process=customise_condOverRides(process)
    
    return process

def customise_Digi(process):
    process.mix.digitizers.pixel.MissCalibrate = False
    process.mix.digitizers.pixel.LorentzAngle_DB = False
    process.mix.digitizers.pixel.killModules = True
    

    dead=cms.VPSet()
    dead.extend([ 
        cms.PSet(Dead_detID = cms.int32(307245144), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307245140), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307245192), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307245188), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307249240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307249236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307253272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307253268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307253552), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307253548), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307253560), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307253556), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307257796), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307257800), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307273756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307273760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307274196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307274200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307277852), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307277856), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307278168), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307278164), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307278228), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307278232), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307278232), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307278228), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307282236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307282240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307282400), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307282396), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307286424), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307286420), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307290144), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307290140), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307298728), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307298724), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308310312), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308310308), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308322724), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308322728), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308347064), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308347060), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308355268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308355272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308359416), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308359412), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308379680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308379676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309334348), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309334352), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309346328), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309346324), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309354712), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309354708), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309363076), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309363080), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309371196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309371200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309391716), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309391720), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309391744), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309391740), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309395688), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309395684), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309412032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309412028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309420264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309420260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309424424), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309424420), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309436808), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309436804), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309444784), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309444780), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309465404), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309465408), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382620), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382624), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382628), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382632), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382636), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382640), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382644), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382648), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382652), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382656), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382660), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382664), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382668), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382672), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382684), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310382688), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310394944), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310394940), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310394980), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310394984), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310403088), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310403084), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310407208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310407204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310444156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310444160), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310460476), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310460480), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310472840), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310472836), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310485164), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310485168), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310501400), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310501396), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310505512), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310505508), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509572), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509576), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509580), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509584), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509588), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509592), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509604), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509608), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509612), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509616), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509620), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509624), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509628), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509632), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509636), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509640), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509644), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509648), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509652), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509656), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509660), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310509664), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513668), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513672), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513684), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513688), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513692), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513696), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513700), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513704), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513708), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513712), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513716), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513720), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513724), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513728), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513732), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513736), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513740), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513744), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310513760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517764), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517768), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517772), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517776), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517780), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517784), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517788), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517792), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517796), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517800), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517804), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517808), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517812), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517816), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517820), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517824), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517828), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517832), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517836), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517840), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517844), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517848), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517852), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310517856), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521860), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521864), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521868), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521872), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521876), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521880), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521884), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521888), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521892), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521896), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521900), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521904), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521908), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521912), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521916), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521920), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521924), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521928), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521932), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521936), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521940), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521944), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521948), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521952), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525956), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525960), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525964), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525968), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525972), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525976), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525980), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525984), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525988), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525992), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310525996), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526000), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526004), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526008), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526012), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526016), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526020), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526024), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526036), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526040), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526044), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310526048), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530052), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530056), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530060), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530064), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530068), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530072), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530076), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530080), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530084), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530088), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530092), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530096), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530100), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530104), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530108), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530112), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530116), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530120), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530124), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530128), Module = cms.string('whole')),
    	 ]);
    dead.extend([ 
        cms.PSet(Dead_detID = cms.int32(310530132), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530136), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530140), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310530144), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534148), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534152), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534160), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534164), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534168), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534172), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534176), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534180), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534184), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534188), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534192), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534212), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534216), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534220), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534224), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534228), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534232), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310534240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538244), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538248), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538252), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538256), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538276), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538280), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538284), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538288), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538292), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538296), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538300), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538304), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538308), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538312), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538316), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538320), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538324), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538328), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538332), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310538336), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542340), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542344), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542348), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542352), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542356), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542360), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542364), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542368), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542372), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542376), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542380), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542384), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542388), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542392), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542396), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542400), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542404), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542408), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542412), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542416), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542420), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542424), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542428), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310542432), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546436), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546440), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546444), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546448), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546460), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546464), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546468), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546472), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546476), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546480), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546484), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546488), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546492), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546496), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546508), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546512), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546516), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546520), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546524), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310546528), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550540), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550544), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550548), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550552), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550556), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550560), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550564), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550568), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550572), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550576), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550580), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550584), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550588), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550592), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550604), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550608), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550612), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550616), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550620), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310550624), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554628), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554632), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554636), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554640), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554644), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554648), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554652), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554656), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554660), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554664), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554668), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554672), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554684), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554688), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554692), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554696), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554700), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554704), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554708), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554712), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554716), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310554720), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558724), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558728), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558732), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558736), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558740), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558744), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558764), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558768), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558772), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558776), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558780), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558784), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558788), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558792), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558796), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558800), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558804), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558808), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558812), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310558816), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562820), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562824), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562828), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562832), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562836), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562840), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562844), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562848), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562852), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562856), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562860), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562864), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562868), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562872), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562876), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562880), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562884), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562888), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562892), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562896), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562900), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562904), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562908), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562912), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566916), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566920), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566924), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566928), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566932), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566936), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566940), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566944), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566948), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566952), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566956), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566960), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566964), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566968), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566972), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566976), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566980), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566984), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566988), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566992), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310566996), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310567000), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310567004), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310567008), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571012), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571016), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571020), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571024), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571036), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571040), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571044), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571048), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571052), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571056), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571060), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571064), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571068), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571072), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571076), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571080), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571084), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571088), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571092), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571096), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571100), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310571104), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575108), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575112), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575116), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575120), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575124), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575128), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575132), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575136), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575140), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575144), Module = cms.string('whole')),
    	 ]);
    dead.extend([ 
        cms.PSet(Dead_detID = cms.int32(310575148), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575152), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575160), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575164), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575168), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575172), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575176), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575180), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575184), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575188), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575192), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310575200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579212), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579216), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579220), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579224), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579228), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579232), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579244), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579248), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579252), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579256), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579276), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579280), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579284), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579288), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579292), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310579296), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583300), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583304), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583308), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583312), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583316), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583320), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583324), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583328), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583332), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583336), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583340), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583344), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583348), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583352), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583356), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583360), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583364), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583368), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583372), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583376), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583380), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583384), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583388), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310583392), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587396), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587400), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587404), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587408), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587412), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587416), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587420), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587424), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587428), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587432), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587436), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587440), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587444), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587448), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587460), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587464), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587468), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587472), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587476), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587480), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587484), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310587488), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591492), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591496), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591508), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591512), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591516), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591520), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591524), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591528), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591540), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591544), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591548), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591552), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591556), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591560), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591564), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591568), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591572), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591576), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591580), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310591584), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595588), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595592), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595604), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595608), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595612), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595616), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595620), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595624), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595628), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595632), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595636), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595640), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595644), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595648), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595652), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595656), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595660), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595664), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595668), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595672), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310595680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599684), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599688), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599692), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599696), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599700), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599704), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599708), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599712), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599716), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599720), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599724), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599728), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599732), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599736), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599740), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599744), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599764), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599768), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599772), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310599776), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603780), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603784), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603788), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603792), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603796), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603800), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603804), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603808), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603812), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603816), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603820), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603824), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603828), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603832), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603836), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603840), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603844), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603848), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603852), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603856), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603860), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603864), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603868), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310603872), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607876), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607880), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607884), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607888), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607892), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607896), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607900), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607904), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607908), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607912), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607916), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607920), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607924), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607928), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607932), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607936), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607940), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607944), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607948), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607952), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607956), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607960), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607964), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310607968), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310611972), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310611976), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310611980), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310611984), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310611988), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310611992), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310611996), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612000), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612004), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612008), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612012), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612016), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612020), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612024), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612036), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612040), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612044), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612048), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612052), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612056), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612060), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310612064), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616068), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616072), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616076), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616080), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616084), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616088), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616092), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616096), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616100), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616104), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616108), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616112), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616116), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616120), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616124), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616128), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616132), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616136), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616140), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616144), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616148), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616152), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310616160), Module = cms.string('whole')),
    	 ]);
    dead.extend([ 
        cms.PSet(Dead_detID = cms.int32(310620164), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620168), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620172), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620176), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620180), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620184), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620188), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620192), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620212), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620216), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620220), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620224), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620228), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620232), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620244), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620248), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620252), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310620256), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624276), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624280), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624284), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624288), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624292), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624296), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624300), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624304), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624308), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624312), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624316), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624320), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624324), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624328), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624332), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624336), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624340), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624344), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624348), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310624352), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628356), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628360), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628364), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628368), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628372), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628376), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628380), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628384), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628388), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628392), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628396), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628400), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628404), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628408), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628412), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628416), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628420), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628424), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628428), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628432), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628436), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628440), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628444), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310628448), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632460), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632464), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632468), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632472), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632476), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632480), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632484), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632488), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632492), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632496), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632508), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632512), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632516), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632520), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632524), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632528), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632540), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310632544), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311443580), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311443584), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311459912), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311459908), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311484608), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311484604), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311500988), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311500992), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311509120), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311509116), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311509148), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311509152), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311537768), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311537764), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311550092), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311550096), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619652), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619656), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623692), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623696), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636128), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636124), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644336), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644332), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656468), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656472), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656476), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656480), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656512), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656508), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660648), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660644), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676932), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676936), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681040), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681036), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311697580), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311697584), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311705644), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311705648), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311726204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311726208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311726244), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311726248), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311738540), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311738544), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355210380), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355210384), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355214412), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355214416), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355226680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355226676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355230908), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355230912), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355247280), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355247276), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355255564), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355255568), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355255612), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355255616), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355263572), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355263576), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355263996), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355264000), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355267880), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355267876), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355268052), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355268056), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355497024), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355497020), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355497108), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355497112), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355497156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355497160), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355501092), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355501096), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355509500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355509504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355521748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355521752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355521760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355521756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355746876), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355746880), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355750988), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355750992), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355751044), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355751048), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355755028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355755032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355996700), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355996704), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356004868), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356004872), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356013132), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356013136), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356017204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356017208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356021532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356021536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356025360), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356025356), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356029680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356029676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356033588), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356033592), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356033716), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356033720), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356038056), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356038052), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356271164), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356271168), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356271204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356271208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356275320), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356275316), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356279368), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356279364), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356283400), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356283396), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356283608), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356283604), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356296012), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356296016), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356300008), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356300004), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356300020), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356300024), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356303928), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356303924), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356308064), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356308060), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356308184), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356308180), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356308420), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356308424), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356316748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356316752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346838216), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346838212), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346846452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346846456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346858596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346858600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346862704), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346862700), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346871204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346871208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346871264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346871260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346875328), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346875324), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346879132), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346879136), Module = cms.string('whole')),
    	 ]);
    dead.extend([ 
        cms.PSet(Dead_detID = cms.int32(347108580), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347108584), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347137040), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347137036), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347137576), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347137572), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347370600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347370596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347391260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347391264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347391320), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347391316), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347403532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347403536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347403568), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347403564), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347616336), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347616332), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347620500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347620504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347624516), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347624520), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347624668), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347624672), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347641056), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347641052), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347649188), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347649192), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347649344), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347649340), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347653436), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347653440), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347657624), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347657620), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347665456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347665452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347874452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347874456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347907100), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347907104), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347923972), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347923976), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347927608), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347927604), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347927820), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347927824), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347928184), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347928180), Module = cms.string('whole')),
                ]);
    process.mix.digitizers.pixel.DeadModules = cms.VPSet(dead)



 




    process.mix.digitizers.pixel.useDB = False
    process.mix.digitizers.pixel.DeadModules_DB = False
    process.mix.digitizers.pixel.NumPixelBarrel = cms.int32(10)
    process.mix.digitizers.pixel.NumPixelEndcap = cms.int32(14)
    process.mix.digitizers.pixel.ThresholdInElectrons_FPix = cms.double(2000.0)
    process.mix.digitizers.pixel.ThresholdInElectrons_BPix = cms.double(2000.0)
    process.mix.digitizers.pixel.ThresholdInElectrons_BPix_L1 = cms.double(2000.0)
    process.mix.digitizers.pixel.thePixelColEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelColEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.AddPixelInefficiencyFromPython = cms.bool(False)
    process.mix.digitizers.strip.ROUList = cms.vstring("g4SimHitsTrackerHitsPixelBarrelLowTof",
                         'g4SimHitsTrackerHitsPixelEndcapLowTof')
    # Check if mergedtruth is in the sequence first, could be taken out depending on cmsDriver options
    if hasattr(process.mix.digitizers,"mergedtruth") :    
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"))
    
    return process


def customise_DigiToRaw(process):
    process.digi2raw_step.remove(process.siPixelRawData)
    process.digi2raw_step.remove(process.rpcpacker)
    return process

def customise_RawToDigi(process):
    process.raw2digi_step.remove(process.siPixelDigis)
    return process

def customise_Reco(process,pileup):



    #use with latest pixel geometry
    process.ClusterShapeHitFilterESProducer.PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1Tk.par')
    # Need this line to stop error about missing siPixelDigis.
    process.MeasurementTracker.inactivePixelDetectorLabels = cms.VInputTag()

    # new layer list (3/4 pixel seeding) in InitialStep and pixelTracks
    process.pixellayertriplets.layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
						       'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
						       'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
						       'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
						       'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
						       'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
						       'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
						       'FPix2_pos+FPix3_pos+FPix4_pos', 'FPix2_neg+FPix3_neg+FPix4_neg',
						       'FPix3_pos+FPix4_pos+FPix5_pos', 'FPix3_neg+FPix4_neg+FPix5_neg',
						       'FPix4_pos+FPix5_pos+FPix6_pos', 'FPix4_neg+FPix5_neg+FPix6_neg',
						       'FPix5_pos+FPix6_pos+FPix7_pos', 'FPix5_neg+FPix6_neg+FPix7_neg',
						       'FPix6_pos+FPix7_pos+FPix8_pos', 'FPix6_neg+FPix7_neg+FPix8_neg',
						       'FPix6_pos+FPix7_pos+FPix9_pos', 'FPix6_neg+FPix7_neg+FPix9_neg')

    # New tracking.  This is really ugly because it redefines globalreco and reconstruction.
    # It can be removed if change one line in Configuration/StandardSequences/python/Reconstruction_cff.py
    # from RecoTracker_cff.py to RecoTrackerPhase1PU140_cff.py

    # remove all the tracking first
    itIndex=process.globalreco.index(process.trackingGlobalReco)
    grIndex=process.reconstruction.index(process.globalreco)

    process.reconstruction.remove(process.globalreco)
    process.globalreco.remove(process.iterTracking)
    process.globalreco.remove(process.electronSeedsSeq)
    process.reconstruction_fromRECO.remove(process.trackingGlobalReco)
    del process.iterTracking
    del process.ckftracks
    del process.ckftracks_woBH
    del process.ckftracks_wodEdX
    del process.ckftracks_plus_pixelless
    del process.trackingGlobalReco
    del process.electronSeedsSeq
    del process.InitialStep
    del process.LowPtTripletStep
    del process.PixelPairStep
    del process.DetachedTripletStep
    del process.MixedTripletStep
    del process.PixelLessStep
    del process.TobTecStep
    del process.earlyGeneralTracks
    del process.ConvStep
    del process.earlyMuons
    del process.muonSeededStepCore
    del process.muonSeededStepExtra 
    del process.muonSeededStep
    del process.muonSeededStepDebug
    
    # add the correct tracking back in
    process.load("RecoTracker.Configuration.RecoTrackerPhase2BEPixel10D_cff")

    process.globalreco.insert(itIndex,process.trackingGlobalReco)
    process.reconstruction.insert(grIndex,process.globalreco)
    #Note process.reconstruction_fromRECO is broken
    
    # End of new tracking configuration which can be removed if new Reconstruction is used.


    process.reconstruction.remove(process.castorreco)
    process.reconstruction.remove(process.CastorTowerReco)
    process.reconstruction.remove(process.ak7BasicJets)
    process.reconstruction.remove(process.ak7CastorJetID)

    #the quadruplet merger configuration     
    process.load("RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff")
    process.pixelseedmergerlayers.BPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.BPix.HitProducer = cms.string("siPixelRecHits" )
    process.pixelseedmergerlayers.FPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.FPix.HitProducer = cms.string("siPixelRecHits" )    
    process.pixelseedmergerlayers.layerList = cms.vstring('BPix1+BPix2+BPix3+BPix4',
						       'BPix1+BPix2+BPix3+FPix1_pos','BPix1+BPix2+BPix3+FPix1_neg',
						       'BPix1+BPix2+FPix1_pos+FPix2_pos', 'BPix1+BPix2+FPix1_neg+FPix2_neg',
						       'BPix1+FPix1_pos+FPix2_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix2_neg+FPix3_neg',
						       'FPix1_pos+FPix2_pos+FPix3_pos+FPix4_pos', 'FPix1_neg+FPix2_neg+FPix3_neg+FPix4_neg',
						       'FPix2_pos+FPix3_pos+FPix4_pos+FPix5_pos', 'FPix2_neg+FPix3_neg+FPix4_neg+FPix5_neg',
						       'FPix3_pos+FPix4_pos+FPix5_pos+FPix6_pos', 'FPix3_neg+FPix4_neg+FPix5_neg+FPix6_pos',
						       'FPix4_pos+FPix5_pos+FPix6_pos+FPix7_pos', 'FPix4_neg+FPix5_neg+FPix6_neg+FPix7_neg',
						       'FPix5_pos+FPix6_pos+FPix7_pos+FPix8_pos', 'FPix5_neg+FPix6_neg+FPix7_neg+FPix8_neg',
						       'FPix5_pos+FPix6_pos+FPix7_pos+FPix9_pos', 'FPix5_neg+FPix6_neg+FPix7_neg+FPix9_neg',
						       'FPix6_pos+FPix7_pos+FPix8_pos+FPix9_pos', 'FPix6_neg+FPix7_neg+FPix8_neg+FPix9_neg')
    
    
    # Need these until pixel templates are used
    process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
    # PixelCPEGeneric #
    process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
    process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
    process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
    process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
    process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
    process.PixelCPEGenericESProducer.DoCosmics = False
    # CPE for other steps
    process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
    # Turn of template use in tracking (iterative steps handled inside their configs)
    process.mergedDuplicateTracks.TTRHBuilder = 'WithTrackAngle'
    process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
    process.muonSeededSeedsInOut.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.muonSeededTracksInOut.TTRHBuilder=cms.string('WithTrackAngle')
    process.muons1stStep.TrackerKinkFinderParameters.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.regionalCosmicTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.cosmicsVetoTracksRaw.TTRHBuilder=cms.string('WithTrackAngle')
    # End of pixel template needed section
    
    process.regionalCosmicTrackerSeeds.OrderedHitsFactoryPSet.LayerPSet.layerList  = cms.vstring('BPix9+BPix8')  # Optimize later
    process.regionalCosmicTrackerSeeds.OrderedHitsFactoryPSet.LayerPSet.BPix = cms.PSet(
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.006),
        useErrorsFromParam = cms.bool(True),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        skipClusters = cms.InputTag("pixelPairStepClusters"),
        hitErrorRPhi = cms.double(0.0027)
    )
    # Make pixelTracks use quadruplets
    process.pixelTracks.SeedMergerPSet = cms.PSet(
        layerListName = cms.string('PixelSeedMergerQuadruplets'),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
        )
    process.pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
    process.pixelTracks.FilterPSet.chi2 = cms.double(50.0)
    process.pixelTracks.FilterPSet.tipMax = cms.double(0.05)
    process.pixelTracks.RegionFactoryPSet.RegionPSet.originRadius =  cms.double(0.02)

    # Particle flow needs to know that the eta range has increased, for
    # when linking tracks to HF clusters
    process=customise_PFlow.customise_extendedTrackerBarrel( process )

    return process

def customise_condOverRides(process):
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_BarrelEndcap5DPixel10DLHCCCooling_cff')
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
    return process


def l1EventContent(process):
    #extend the event content

    alist=['RAWSIM','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):

            getattr(process,b).outputCommands.append('keep *_TTClustersFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTStubsFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTTracksFromPixelDigis_*_*')

            getattr(process,b).outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')

            getattr(process,b).outputCommands.append('drop PixelDigiSimLinkedmDetSetVector_mix_*_*')
            getattr(process,b).outputCommands.append('drop PixelDigiedmDetSetVector_mix_*_*')

            getattr(process,b).outputCommands.append('keep *_simSiPixelDigis_*_*')

    return process

def customise_DQM(process,pileup):
    # We cut down the number of iterative tracking steps
#    process.dqmoffline_step.remove(process.TrackMonStep3)
#    process.dqmoffline_step.remove(process.TrackMonStep4)
#    process.dqmoffline_step.remove(process.TrackMonStep5)
#    process.dqmoffline_step.remove(process.TrackMonStep6)
    			    #The following two steps were removed
                            #process.PixelLessStep*
                            #process.TobTecStep*
#    process.dqmoffline_step.remove(process.muonAnalyzer)
    process.dqmoffline_step.remove(process.jetMETAnalyzer)
#    process.dqmoffline_step.remove(process.TrackMonStep9)
#    process.dqmoffline_step.remove(process.TrackMonStep10)
#    process.dqmoffline_step.remove(process.PixelTrackingRecHitsValid)

    #put isUpgrade flag==true
    process.SiPixelRawDataErrorSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelDigiSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelClusterSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelRecHitSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelTrackResidualSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelHitEfficiencySource.isUpgrade = cms.untracked.bool(True)
    
    from DQM.TrackingMonitor.customizeTrackingMonitorSeedNumber import customise_trackMon_IterativeTracking_PHASE1PU140
    process=customise_trackMon_IterativeTracking_PHASE1PU140(process)
    process.dqmoffline_step.remove(process.Phase1Pu70TrackMonStep2)
    process.dqmoffline_step.remove(process.Phase1Pu70TrackMonStep4)
    if hasattr(process,"globalrechitsanalyze") : # Validation takes this out if pileup is more than 30
       process.globalrechitsanalyze.ROUList = cms.vstring(
          'g4SimHitsTrackerHitsPixelBarrelLowTof', 
          'g4SimHitsTrackerHitsPixelBarrelHighTof', 
          'g4SimHitsTrackerHitsPixelEndcapLowTof', 
          'g4SimHitsTrackerHitsPixelEndcapHighTof')
    return process

def customise_Validation(process,pileup):
    process.validation_step.remove(process.PixelTrackingRecHitsValid)
    process.validation_step.remove(process.stripRecHitsValid)
    process.validation_step.remove(process.trackerHitsValid)
    process.validation_step.remove(process.StripTrackingRecHitsValid)
    # We don't run the HLT
    process.validation_step.remove(process.HLTSusyExoVal)
    process.validation_step.remove(process.hltHiggsValidator)
    process.validation_step.remove(process.relvalMuonBits)
    if pileup>30:
        process.trackValidator.label=cms.VInputTag(cms.InputTag("cutsRecoTracksHp"))
        process.tracksValidationSelectors = cms.Sequence(process.cutsRecoTracksHp)
        process.globalValidation.remove(process.recoMuonValidation)
        process.validation.remove(process.recoMuonValidation)
        process.validation_preprod.remove(process.recoMuonValidation)
        process.validation_step.remove(process.recoMuonValidation)
        process.validation.remove(process.globalrechitsanalyze)
        process.validation_prod.remove(process.globalrechitsanalyze)
        process.validation_step.remove(process.globalrechitsanalyze)
        process.validation.remove(process.stripRecHitsValid)
        process.validation_step.remove(process.stripRecHitsValid)
        process.validation_step.remove(process.StripTrackingRecHitsValid)
        process.globalValidation.remove(process.vertexValidation)
        process.validation.remove(process.vertexValidation)
        process.validation_step.remove(process.vertexValidation)
        process.mix.input.nbPileupEvents.averageNumber = cms.double(0.0)
        process.mix.minBunch = cms.int32(0)
        process.mix.maxBunch = cms.int32(0)

    if hasattr(process,'simHitTPAssocProducer'):    
        process.simHitTPAssocProducer.simHitSrc=cms.VInputTag(cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
                                                              cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"))

    return process

def customise_harvesting(process):
    process.dqmHarvesting.remove(process.jetMETDQMOfflineClient)
    process.dqmHarvesting.remove(process.dataCertificationJetMET)
    process.dqmHarvesting.remove(process.sipixelEDAClient)
    process.dqmHarvesting.remove(process.sipixelCertification)
    return (process)

