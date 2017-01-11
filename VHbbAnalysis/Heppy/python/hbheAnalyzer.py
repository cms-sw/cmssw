import operator 
import itertools
import copy

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
import PhysicsTools.HeppyCore.framework.config as cfg

# New filter parameters taken from slide 11 of 
# https://indico.cern.ch/event/433302/contribution/0/attachments/1126451/1608346/2015-07-15_slides_v3.pdf
#
# HBHE filter requires:
#
# either  maxHPDHits >= 17
# or      maxHPDNoOtherHits >= 10
# or      maxZeros >= 10 (to be removed here)
# or      (HasBadRBXTS4TS5 and not goodJetFoundInLowBVRegion)


class hbheAnalyzer( Analyzer ):

    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(hbheAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.IgnoreTS4TS5ifJetInLowBVRegion = cfg_ana.IgnoreTS4TS5ifJetInLowBVRegion

    def declareHandles(self):
        super(hbheAnalyzer, self).declareHandles()
        self.handles['hcalnoise'] = AutoHandle( 'hcalnoise', 'HcalNoiseSummary' )

    def beginLoop(self, setup):
        super(hbheAnalyzer,self).beginLoop( setup )

    def process(self, event):
        self.readCollections( event.input )

        event.hbheGoodJetFoundInLowBVRegion = False

        event.hbheMaxZeros          = self.handles['hcalnoise'].product().maxZeros()
        event.hbheMaxHPDHits        = self.handles['hcalnoise'].product().maxHPDHits()
        event.hbheMaxHPDNoOtherHits = self.handles['hcalnoise'].product().maxHPDNoOtherHits()
        event.hbheHasBadRBXTS4TS5   = self.handles['hcalnoise'].product().HasBadRBXTS4TS5()
        event.hbheHasBadRBXRechitR45Loose   = self.handles['hcalnoise'].product().HasBadRBXRechitR45Loose()
        if self.IgnoreTS4TS5ifJetInLowBVRegion: event.hbheGoodJetFoundInLowBVRegion = self.handles['hcalnoise'].product().goodJetFoundInLowBVRegion()
        event.hbhenumIsolatedNoiseChannels  = self.handles['hcalnoise'].product().numIsolatedNoiseChannels()
        event.hbheisolatedNoiseSumE         = self.handles['hcalnoise'].product().isolatedNoiseSumE()
        event.hbheisolatedNoiseSumEt        = self.handles['hcalnoise'].product().isolatedNoiseSumEt()

        event.hbheFilterNew25ns = 1
        event.hbheFilterNew50ns = 1

        if event.hbheMaxHPDHits >= 17: 
            event.hbheFilterNew25ns = 0
            event.hbheFilterNew50ns = 0
        if event.hbheMaxHPDNoOtherHits >= 10 or (event.hbheHasBadRBXTS4TS5 and not event.hbheGoodJetFoundInLowBVRegion): 
            event.hbheFilterNew50ns = 0
        if event.hbheMaxHPDNoOtherHits >= 10 or (event.hbheHasBadRBXRechitR45Loose and not event.hbheGoodJetFoundInLowBVRegion): 
            event.hbheFilterNew25ns = 0

        event.hbheFilterIso = 1
        if event.hbhenumIsolatedNoiseChannels >= 10: event.hbheFilterIso = 0
        if event.hbheisolatedNoiseSumE        >= 50: event.hbheFilterIso = 0
        if event.hbheisolatedNoiseSumEt       >= 25: event.hbheFilterIso = 0 


#        event.hbheFilterNew = event.hbheFilterNew50ns # to be updated later with automatic choice based on PileupSummaryInfo or run number
        event.hbheFilterNew = event.hbheFilterNew25ns # to be updated later with automatic choice based on PileupSummaryInfo or run number


        return True

setattr(hbheAnalyzer,"defaultConfig", cfg.Analyzer(
        class_object = hbheAnalyzer,
        IgnoreTS4TS5ifJetInLowBVRegion = False,
        )
)
