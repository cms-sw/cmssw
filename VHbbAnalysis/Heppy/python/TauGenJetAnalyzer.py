
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import GenParticle

from PhysicsTools.Heppy.physicsutils.genutils import *
import PhysicsTools.HeppyCore.framework.config as cfg

from VHbbAnalysis.Heppy.genTauDecayMode import genTauDecayMode
        
class TauGenJetAnalyzer( Analyzer ):
    """Determine generator level tau decay mode."""
    
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(TauGenJetAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

    #---------------------------------------------
    # DECLARATION OF HANDLES OF GEN LEVEL OBJECTS 
    #---------------------------------------------

    def declareHandles(self):
        super(TauGenJetAnalyzer, self).declareHandles()

        #mc information
        self.mchandles['tauGenJetsSelectorAllHadrons'] = AutoHandle( 'tauGenJetsSelectorAllHadrons',
                                                                     'vector<reco::GenJet>' )

    def beginLoop(self,setup):
        super(TauGenJetAnalyzer,self).beginLoop(setup)
            
    def makeMCInfo(self, event):
        event.tauGenJets = []
        tauGenJets = list(self.mchandles['tauGenJetsSelectorAllHadrons'].product() )
        for tauGenJet in tauGenJets:
            tauGenJet.decayMode = genTauDecayMode(tauGenJet)[0]
            event.tauGenJets.append(tauGenJet)

    def process(self, event):
        self.readCollections(event.input)

        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True

        # do MC level analysis
        self.makeMCInfo(event)

        return True

setattr(TauGenJetAnalyzer, "defaultConfig", cfg.Analyzer(
    class_object = TauGenJetAnalyzer,
    verbose = False,
))
