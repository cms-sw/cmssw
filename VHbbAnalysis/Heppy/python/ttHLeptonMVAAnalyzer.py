from math import *

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi
from VHbbAnalysis.Heppy.leptonMVA import LeptonMVA
from VHbbAnalysis.Heppy.signedSip import *
import os
        
class ttHLeptonMVAAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHLeptonMVAAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.leptonMVAKindTTH = getattr(self.cfg_ana, "leptonMVAKindTTH", "forMoriond16")
        self.leptonMVAPathTTH = getattr(self.cfg_ana, "leptonMVAPathTTH", "VHbbAnalysis/Heppy/data/leptonMVA/tth/%s_BDTG.weights.xml")
        if self.leptonMVAPathTTH[0] != "/": self.leptonMVAPathTTH = "%s/src/%s" % ( os.environ['CMSSW_BASE'], self.leptonMVAPathTTH)
        self.leptonMVATTH = LeptonMVA(self.leptonMVAKindTTH, self.leptonMVAPathTTH, self.cfg_comp.isMC)

    def declareHandles(self):
        super(ttHLeptonMVAAnalyzer, self).declareHandles()

    def beginLoop(self, setup):
        super(ttHLeptonMVAAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')

    def process(self, event):
        self.readCollections( event.input )
        self.counters.counter('events').inc('all events')

        for lep in event.selectedLeptons:
            lep.mvaValueTTH = self.leptonMVATTH(lep, event.allJets)
        for lep in event.inclusiveLeptons:
            if lep not in event.selectedLeptons:
                lep.mvaValueTTH = self.leptonMVATTH(lep, event.allJets)

        return True
