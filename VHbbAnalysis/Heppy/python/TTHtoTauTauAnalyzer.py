from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Tau import Tau

from PhysicsTools.HeppyCore.utils.deltar import deltaR, matchObjectCollection3

import PhysicsTools.HeppyCore.framework.config as cfg

class TTHtoTauTauAnalyzer( Analyzer ):
   '''Analyze ttH, H -> tautau events'''

   def declareHandles(self):
      super(TTHtoTauTauAnalyzer, self).declareHandles()

      ##self.handles['taus'] = AutoHandle( ('slimmedTaus',''), 'std::vector<pat::Tau>' )

      #mc information      
      self.mchandles['genParticles'] = AutoHandle( 'prunedGenParticles',
                                                   'std::vector<reco::GenParticle>' )

   def addTau_genMatchType(self, event, tau):      
      '''Determine if given tau matched gen level hadronic tau decay or is due to a misidentified jet, electron or muon
             tau.genMatchType = 0 for matched to gen level hadronic tau decay
                              = 1 for matched to gen level jet
                              = 2 for matched to gen level electron
                              = 3 for matched to gen level muon
      '''

      genParticles = list(self.mchandles['genParticles'].product() )

      genMatchType = 1 # assume hadronic tau to be due to misidentified jet per default
      if tau.genJet():
         genMatchType = 0
      if genMatchType == 1:
         match = matchObjectCollection3([ tau ], genParticles, deltaRMax = 0.4, filter = lambda x,y : True if (y.pt() > 0.5*x.pt() and abs(y.pdgId()) == 11) else False)
         if match[tau]:
            genMatchType = 2
      if genMatchType == 1:
         match = matchObjectCollection3([ tau ], genParticles, deltaRMax = 0.4, filter = lambda x,y : True if (y.pt() > 0.5*x.pt() and abs(y.pdgId()) == 13) else False)
         if match[tau]:
            genMatchType = 3

      return genMatchType

   def process(self, event):
      #print "<TTHtoTauTauAnalyzer::process>:"
      
      self.readCollections( event.input )

      ##taus = list( self.handles['taus'].product() )
      taus = event.inclusiveTaus
      taus_modified = []
      for idxTau in range(len(taus)):
         tau = Tau(taus[idxTau])
         #print "processing tau #%i: Pt = %1.2f, eta = %1.2f, phi = %1.2f" % (idxTau, tau.pt(), tau.eta(), tau.phi())
         # if not MC, nothing to do
         if self.cfg_comp.isMC:
            tau.genMatchType = self.addTau_genMatchType(event, tau)
         else:
            tau.genMatchType = -1
         #print " genMatchType = %i" % tau.genMatchType
         taus_modified.append(tau)

      event.inclusiveTaus = taus_modified

      return True
