
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import GenParticle

from PhysicsTools.Heppy.physicsutils.genutils import *
import PhysicsTools.HeppyCore.framework.config as cfg
        
class TTHtoTauTauGeneratorAnalyzer( Analyzer ):
    """Do generator-level analysis of ttH, H -> tautau decay:

       Creates in the event:
         event.genTTHtoTauTauDecayMode = 0 for 2b_2taul_2wl
                                         1 for 2b_2taulh_2wl
                                         2 for 2b_2tauh_2wl
                                         3 for 2b_2taul_2wlj
                                         4 for 2b_2taulh_2wlj
                                         5 for 2b_2tauh_2wlj
                                         6 for 2b_2taul_2wj
                                         7 for 2b_2taulh_2wj
                                         8 for 2b_2tauh_2wj
                                        -1 other
    """
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(TTHtoTauTauGeneratorAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

    #---------------------------------------------
    # DECLARATION OF HANDLES OF GEN LEVEL OBJECTS 
    #---------------------------------------------

    def declareHandles(self):
        super(TTHtoTauTauGeneratorAnalyzer, self).declareHandles()

        #mc information
        self.mchandles['genParticles'] = AutoHandle( 'prunedGenParticles',
                                                     'std::vector<reco::GenParticle>' )

    def beginLoop(self,setup):
        super(TTHtoTauTauGeneratorAnalyzer,self).beginLoop(setup)

    def findFirstDaughterGivenPdgId(self, givenMother, givenMotherPdgId, givenPdgIds):
        """Get the first gen level daughter particle with given pdgId"""

        mothers = []
        if givenMother:
            mothers.append(givenMother)
        while len(mothers) > 0:
            daughters = []
            for mother in mothers:
                if givenMotherPdgId and mother.pdgId() != givenMotherPdgId:
                    continue
                for idxDaughter in range(mother.numberOfDaughters()):
                    daughter = mother.daughter(idxDaughter)
                    daughters.append(daughter)
            for daughter in daughters:
                if daughter.pdgId() in givenPdgIds:
                    return daughter
            mothers = daughters

        # no daughter particle with given pdgId found
        return None

    def fillGenTTHtoTauTauDecayMode(self, event, genParticles):
        """Determine gen level ttH, H -> tautau decay mode"""

        t = None
        tbar = None
        H = None

        for genParticle in genParticles:
            pdgId = genParticle.pdgId()
            if pdgId == +6 and not t:
                t = genParticle
            if pdgId == -6 and not tbar:
                tbar = genParticle
            if pdgId in [ 25, 35, 36 ] and not H:
                H = genParticle

        event.genH_decayMode = -1
        if H:
            for pdgIds in [ [ +5, -5 ], [ +24, -24 ], [ 21, 21 ], [ +15, -15 ], [ 23, 23 ], [ +4, -4 ], [ 22, 22 ], [ 23, 22 ] ]:
                if self.findFirstDaughterGivenPdgId(H, H.pdgId(), [ pdgIds[0] ]) and self.findFirstDaughterGivenPdgId(H, H.pdgId(), [ pdgIds[1] ]):
                    event.genH_decayMode = abs(pdgIds[0]*pdgIds[1])
                    break

        numElectrons_or_Muons_fromH = 0
        numHadTaus_fromH = 0
        if H:
            if self.findFirstDaughterGivenPdgId(H, None, [ -11, -13 ]):
                numElectrons_or_Muons_fromH = numElectrons_or_Muons_fromH + 1
            elif self.findFirstDaughterGivenPdgId(H, H.pdgId(), [ -15 ]):
                numHadTaus_fromH = numHadTaus_fromH + 1
            if self.findFirstDaughterGivenPdgId(H, None, [ +11, +13 ]):
                numElectrons_or_Muons_fromH = numElectrons_or_Muons_fromH + 1
            elif self.findFirstDaughterGivenPdgId(H, H.pdgId(), [ +15 ]):
                numHadTaus_fromH = numHadTaus_fromH + 1

        numBs = 0
        if t and tbar:
            if self.findFirstDaughterGivenPdgId(t, t.pdgId(), [ +5 ]):
                numBs = numBs + 1
            if self.findFirstDaughterGivenPdgId(tbar, tbar.pdgId(), [ -5 ]):
                numBs = numBs + 1

        Wplus = None
        Wminus = None
        if t and tbar:
            Wplus = self.findFirstDaughterGivenPdgId(t, t.pdgId(), [ +24 ])
            Wminus = self.findFirstDaughterGivenPdgId(tbar, tbar.pdgId(), [ -24 ])
        
        numElectrons_or_Muons_fromW = 0
        numHadTaus_fromW = 0
        if Wplus and Wminus:
            if self.findFirstDaughterGivenPdgId(Wplus, None, [ -11, -13 ]):
                numElectrons_or_Muons_fromW = numElectrons_or_Muons_fromW + 1
            elif self.findFirstDaughterGivenPdgId(Wplus, Wplus.pdgId(), [ -15 ]):
                numHadTaus_fromW = numHadTaus_fromW + 1
            if self.findFirstDaughterGivenPdgId(Wminus, None, [ +11, +13 ]):
                numElectrons_or_Muons_fromW = numElectrons_or_Muons_fromW + 1
            elif self.findFirstDaughterGivenPdgId(Wminus, Wminus.pdgId(), [ +15 ]):
                numHadTaus_fromW = numHadTaus_fromW + 1

        event.genTTHtoTauTauDecayMode = -1
        if numElectrons_or_Muons_fromH == 2 and (numElectrons_or_Muons_fromW + numHadTaus_fromW) == 2:
            event.genTTHtoTauTauDecayMode = 0
        elif numElectrons_or_Muons_fromH == 1 and numHadTaus_fromH == 1 and (numElectrons_or_Muons_fromW + numHadTaus_fromW) == 2:
            event.genTTHtoTauTauDecayMode = 1
        elif numHadTaus_fromH == 2 and numElectrons_or_Muons_fromW == 2:
            event.genTTHtoTauTauDecayMode = 2
        elif numElectrons_or_Muons_fromH == 2 and (numElectrons_or_Muons_fromW + numHadTaus_fromW) == 1:
            event.genTTHtoTauTauDecayMode = 3
        elif numElectrons_or_Muons_fromH == 1 and numHadTaus_fromH == 1 and (numElectrons_or_Muons_fromW + numHadTaus_fromW) == 1:
            event.genTTHtoTauTauDecayMode = 4
        elif numHadTaus_fromH == 2 and (numElectrons_or_Muons_fromW + numHadTaus_fromW) == 1:
            event.genTTHtoTauTauDecayMode = 5
        elif numElectrons_or_Muons_fromH == 2 and (numElectrons_or_Muons_fromW + numHadTaus_fromW) == 0:
            event.genTTHtoTauTauDecayMode = 6
        elif numElectrons_or_Muons_fromH == 1 and numHadTaus_fromH == 1 and numElectrons_or_Muons_fromW == 0:
            event.genTTHtoTauTauDecayMode = 7
        elif numHadTaus_fromH == 2 and numElectrons_or_Muons_fromW == 0:
            event.genTTHtoTauTauDecayMode = 8

    def makeMCInfo(self, event):
        genParticles = list(self.mchandles['genParticles'].product() )
        self.fillGenTTHtoTauTauDecayMode(event, genParticles)

    def process(self, event):
        self.readCollections(event.input)

        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True

        # do MC level analysis
        self.makeMCInfo(event)

        return True

setattr(TTHtoTauTauGeneratorAnalyzer, "defaultConfig", cfg.Analyzer(
    class_object = TTHtoTauTauGeneratorAnalyzer,
    verbose = False,
))
