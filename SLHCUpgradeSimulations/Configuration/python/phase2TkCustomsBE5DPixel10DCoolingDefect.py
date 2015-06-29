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
        cms.PSet(Dead_detID = cms.int32(307237188), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307237192), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307245204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307245208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307245268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307245272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307253284), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307253288), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307253592), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307253588), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307261820), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307261824), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307273808), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307273804), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307278272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307278268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307286156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(307286160), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308302160), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308302156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308302188), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308302192), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308302196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308302200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308314288), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308314284), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308314344), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308314340), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308330904), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308330900), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308343108), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308343112), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308355344), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308355340), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308375872), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308375868), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308379748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308379752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308379980), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(308379984), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309346572), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309346576), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309346736), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309346732), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309350528), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309350524), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309354672), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309354668), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309358616), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309358612), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309375196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309375200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309379108), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309379112), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309383452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309383456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309383476), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309383480), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309391532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309391536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309416196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309416200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309416248), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309416244), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309420452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309420456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309444712), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309444708), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309448752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309448748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309448908), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(309448912), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310386720), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310386716), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310407292), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310407296), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310415544), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310415540), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310435872), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310435868), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310435896), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310435892), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310440040), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310440036), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310501524), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310501528), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310505500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310505504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521968), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521964), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310522000), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310521996), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562936), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(310562932), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431212), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431216), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431220), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431224), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431228), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431232), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431244), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431248), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431252), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431256), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311431264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311439444), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311439448), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311455784), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311455780), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311492728), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311492724), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311496816), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311496812), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311541808), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311541804), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311541880), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311541876), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558148), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558152), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558160), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558164), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558168), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558172), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558176), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558180), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558184), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558188), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558192), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558212), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558216), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558220), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558224), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558228), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558232), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311558240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562244), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562248), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562252), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562256), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562276), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562280), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562284), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562288), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562292), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562296), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562300), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562304), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562308), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562312), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562316), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562320), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562324), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562328), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562332), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562336), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562404), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311562408), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566340), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566344), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566348), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566352), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566356), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566360), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566364), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566368), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566372), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566376), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566380), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566384), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566388), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566392), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566396), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566400), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566404), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566408), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566412), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566416), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566420), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566424), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566428), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311566432), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570436), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570440), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570444), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570448), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570460), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570464), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570468), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570472), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570476), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570480), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570484), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570488), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570492), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570496), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570508), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570512), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570516), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570520), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570524), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311570528), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574540), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574544), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574548), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574552), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574556), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574560), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574564), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574568), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574572), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574576), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574580), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574584), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574588), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574592), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574604), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574608), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574612), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574616), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574620), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311574624), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578628), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578632), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578636), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578640), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578644), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578648), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578652), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578656), Module = cms.string('whole')),
    	 ]);
    dead.extend([ 
        cms.PSet(Dead_detID = cms.int32(311578660), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578664), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578668), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578672), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578684), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578688), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578692), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578696), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578700), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578704), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578708), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578712), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578716), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311578720), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582724), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582728), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582732), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582736), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582740), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582744), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582764), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582768), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582772), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582776), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582780), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582784), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582788), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582792), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582796), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582800), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582804), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582808), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582812), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311582816), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586820), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586824), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586828), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586832), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586836), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586840), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586844), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586848), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586852), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586856), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586860), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586864), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586868), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586872), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586876), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586880), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586884), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586888), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586892), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586896), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586900), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586904), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586908), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311586912), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590916), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590920), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590924), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590928), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590932), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590936), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590940), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590944), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590948), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590952), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590956), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590960), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590964), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590968), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590972), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590976), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590980), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590984), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590988), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590992), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311590996), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311591000), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311591004), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311591008), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595012), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595016), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595020), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595024), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595036), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595040), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595044), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595048), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595052), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595056), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595060), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595064), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595068), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595072), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595076), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595080), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595084), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595088), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595092), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595096), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595100), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311595104), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599108), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599112), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599116), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599120), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599124), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599128), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599132), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599136), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599140), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599144), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599148), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599152), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599160), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599164), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599168), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599172), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599176), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599180), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599184), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599188), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599192), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311599200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603212), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603216), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603220), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603224), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603228), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603232), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603244), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603248), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603252), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603256), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603276), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603280), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603284), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603288), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603292), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603296), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603380), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311603384), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607300), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607304), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607308), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607312), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607316), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607320), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607324), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607328), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607332), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607336), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607340), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607344), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607348), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607352), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607356), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607360), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607364), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607368), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607372), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607376), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607380), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607384), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607388), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311607392), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611396), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611400), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611404), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611408), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611412), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611416), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611420), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611424), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611428), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611432), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611436), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611440), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611444), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611448), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611460), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611464), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611468), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611472), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611476), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611480), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611484), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311611488), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615492), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615496), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615508), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615512), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615516), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615520), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615524), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615528), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615540), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615544), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615548), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615552), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615556), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615560), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615564), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615568), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615572), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615576), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615580), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311615584), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619588), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619592), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619604), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619608), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619612), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619616), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619620), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619624), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619628), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619632), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619636), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619640), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619644), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619648), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619652), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619656), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619660), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619664), Module = cms.string('whole')),
    	 ]);
    dead.extend([ 
        cms.PSet(Dead_detID = cms.int32(311619668), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619672), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311619680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623684), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623688), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623692), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623696), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623700), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623704), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623708), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623712), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623716), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623720), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623724), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623728), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623732), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623736), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623740), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623744), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623764), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623768), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623772), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311623776), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627780), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627784), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627788), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627792), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627796), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627800), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627804), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627808), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627812), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627816), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627820), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627824), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627828), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627832), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627836), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627840), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627844), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627848), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627852), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627856), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627860), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627864), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627868), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627872), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627968), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311627964), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631876), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631880), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631884), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631888), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631892), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631896), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631900), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631904), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631908), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631912), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631916), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631920), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631924), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631928), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631932), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631936), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631940), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631944), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631948), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631952), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631956), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631960), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631964), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311631968), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311635972), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311635976), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311635980), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311635984), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311635988), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311635992), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311635996), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636000), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636004), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636008), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636012), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636016), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636020), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636024), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636036), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636040), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636044), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636048), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636052), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636056), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636060), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311636064), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640068), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640072), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640076), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640080), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640084), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640088), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640092), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640096), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640100), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640104), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640108), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640112), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640116), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640120), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640124), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640128), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640132), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640136), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640140), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640144), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640148), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640152), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311640160), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644164), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644168), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644172), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644176), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644180), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644184), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644188), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644192), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644196), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644200), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644212), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644216), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644220), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644224), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644228), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644232), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644244), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644248), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644252), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311644256), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648276), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648280), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648284), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648288), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648292), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648296), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648300), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648304), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648308), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648312), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648316), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648320), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648324), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648328), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648332), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648336), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648340), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648344), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648348), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311648352), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652356), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652360), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652364), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652368), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652372), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652376), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652380), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652384), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652388), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652392), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652396), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652400), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652404), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652408), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652412), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652416), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652420), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652424), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652428), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652432), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652436), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652440), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652444), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311652448), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656452), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656460), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656464), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656468), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656472), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656476), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656480), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656484), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656488), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656492), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656496), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656508), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656512), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656516), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656520), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656524), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656528), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656540), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656544), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311656600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660548), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660552), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660556), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660560), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660564), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660568), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660572), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660576), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660580), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660584), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660588), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660592), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660604), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660608), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660612), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660616), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660620), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660624), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660628), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660632), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660636), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311660640), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664644), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664648), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664652), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664656), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664660), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664664), Module = cms.string('whole')),
    	 ]);
    dead.extend([ 
        cms.PSet(Dead_detID = cms.int32(311664668), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664672), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664684), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664688), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664692), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664696), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664700), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664704), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664708), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664712), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664716), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664720), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664724), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664728), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664732), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311664736), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668740), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668744), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668764), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668768), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668772), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668776), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668780), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668784), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668788), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668792), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668796), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668800), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668804), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668808), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668812), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668816), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668820), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668824), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668828), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311668832), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672836), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672840), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672844), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672848), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672852), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672856), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672860), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672864), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672868), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672872), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672876), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672880), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672884), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672888), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672892), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672896), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672900), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672904), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672908), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672912), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672916), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672920), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672924), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311672928), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676932), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676936), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676940), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676944), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676948), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676952), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676956), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676960), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676964), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676968), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676972), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676976), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676980), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676984), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676988), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676992), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311676996), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311677000), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311677004), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311677008), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311677012), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311677016), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311677020), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311677024), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681036), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681040), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681044), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681048), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681052), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681056), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681060), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681064), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681068), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681072), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681076), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681080), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681084), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681088), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681092), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681096), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681100), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681104), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681108), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681112), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681116), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(311681120), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312496152), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312496148), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312524980), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312524984), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312537116), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312537120), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312553492), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312553496), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312557640), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312557636), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312582324), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312582328), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312586332), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312586336), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312615032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312615028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312623292), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312623296), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312639528), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312639524), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312643588), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312643592), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312643644), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312643648), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312651924), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312651928), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312696856), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312696852), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312701024), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312701020), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312709152), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312709148), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312741976), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312741972), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312742012), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312742016), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312750124), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312750128), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312774748), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312774752), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312778760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(312778756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355214420), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355214424), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355226632), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355226628), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355239068), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355239072), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355239176), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355239172), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355243300), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355243304), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355251544), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355251540), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355255804), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355255808), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355259892), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355259896), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355263916), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355263920), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355264064), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355264060), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355268056), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355268052), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355480740), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355480744), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355497072), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355497068), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355517472), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355517468), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355522068), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355522072), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355530156), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355530160), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355763236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355763240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355763500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355763504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355775760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355775756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355780060), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(355780064), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356009088), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356009084), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356021476), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356021480), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356037828), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356037832), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356050220), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356050224), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356054536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356054532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356283600), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356283596), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356287684), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356287688), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356295756), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356295760), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356308284), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356308288), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356312204), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356312208), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356312352), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356312348), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356316404), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(356316408), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346838272), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346838268), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346854436), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346854440), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346858676), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346858680), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346858868), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346858872), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346879024), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(346879020), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347092128), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347092124), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347096136), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347096132), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347104532), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347104536), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347112544), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347112540), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347120824), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347120820), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347120924), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347120928), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347124860), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347124864), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347129096), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347129092), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347137236), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347137240), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347137400), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347137396), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347362484), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347362488), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347366456), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347366452), Module = cms.string('whole')),
    	 ]);
    dead.extend([ 
        cms.PSet(Dead_detID = cms.int32(347378924), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347378928), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347387288), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347387284), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347399504), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347399500), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347399652), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347399656), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347403468), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347403472), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347403784), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347403780), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347649032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347649028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347657260), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347657264), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347665552), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347665548), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347895028), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347895032), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347907428), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347907432), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347927768), Module = cms.string('whole')),
        cms.PSet(Dead_detID = cms.int32(347927764), Module = cms.string('whole')),
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
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_BarrelEndcap5DPixel10DLHCC_cff')
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

