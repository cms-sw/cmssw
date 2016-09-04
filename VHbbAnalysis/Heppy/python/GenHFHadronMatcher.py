import itertools

import ROOT

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObject import PhysicsObject

class Hadron(object):
    """
    Represents a hadron in the GenHFHadronMatcher.cc context.
    Specified by an index in the hadron vector and an index to a jet.
    """
    def __init__(self, idx, jetIdx, flavour, fromTopWeak):
        self.index = idx
        self.jetIndex = jetIdx
        self.flavour = flavour
        self.fromTopWeak = fromTopWeak
        #self.lepViaTau = lepViaTau
        self.kind = "B"

    def __str__(self):
        s = self.kind
        s += " bidx={0}".format(self.index)
        s += " jidx={0}".format(self.jetIndex)
        s += " fl={0}".format(self.flavour)
        s += " fromTopWeak={0}".format(self.fromTopWeak)
        #s += " lepViaTau={0}".format(self.lepViaTau)
        return s

class CHadron(Hadron):
    def __init__(self, idx, jetIdx, flavour, fromTopWeak, parentBHadronIdx):
        super(CHadron, self).__init__(idx, jetIdx, flavour, fromTopWeak)
        self.kind = "C"
        self.bHadronIdx = parentBHadronIdx

    def __str__(self):
        s = super(CHadron, self).__str__()
        s += " bHadronIdx={0}".format(self.bHadronIdx)
        return s

class GenHFHadronMatcher( Analyzer ):

    def declareHandles(self):
        super(GenHFHadronMatcher, self).declareHandles()
        #self.genJetsSrc = "slimmedGenJets"
        self.keys_b = [
            "genBHadIndex",
            "genBHadJetIndex",
            "genBHadFlavour",
            "genBHadFromTopWeakDecay",
            #"genBHadLeptonViaTau"
        ]
        self.keys_c = [
            "genCHadIndex",
            "genCHadJetIndex",
            "genCHadFlavour",
            "genCHadFromTopWeakDecay",
            #"genCHadLeptonViaTau",
            "genCHadBHadronId",
        ]
        for x in self.keys_b:
            self.handles[x] = AutoHandle(
                ("matchGenBHadron",x,"EX"), "std::vector<int>"
            )
        for x in self.keys_c:
            self.handles[x] = AutoHandle(
                ("matchGenCHadron",x,"EX"), "std::vector<int>"
            )
        self.handles['genBHadJetIndex'] = AutoHandle( ("matchGenBHadron","genBHadJetIndex","EX"), "std::vector<int>")
        
        self.handles['ttbarCategory'] = AutoHandle( ("categorizeGenTtbar","genTtbarId","EX"), "int")

        self.genJetMinPt = 20
        self.genJetMaxEta = 2.4

    def process(self, event):

        # if not MC, nothing to do
        if not self.cfg_comp.isMC:
            return True

        self.readCollections( event.input )

        genJets = event.genJets
        genJetsIndexed = {gj.index: gj for gj in genJets}

        arrs_b = {k: self.handles[k].product() for k in self.keys_b}
        arrs_c = {k: self.handles[k].product() for k in self.keys_c}
        event.ttbarCategory = self.handles["ttbarCategory"].product()[0]
        nbhad = arrs_b["genBHadJetIndex"].size()
        nchad = arrs_c["genCHadJetIndex"].size()

        bhads = [Hadron(*[arrs_b[k][i] for k in self.keys_b]) for i in range(nbhad)]
        chads = [CHadron(*[arrs_c[k][i] for k in self.keys_c]) for i in range(nchad)]

        #pre-create the counters on gen-jets
        for gj in genJets:
            for fl in ["b", "c"]:
                d = {}
                for sel in ["nosel", "pteta"]:
                    d[sel] = {}
                    d[sel]["fromTop"] = 0
                    d[sel]["beforeTop"] = 0
                    d[sel]["afterTop"] = 0
                setattr(gj, fl+"HadMatch", d)

        #get b-hadrons not coming from W, save the ones that are before the top
        for had in bhads:

            #hadron had no associated jet
            if had.jetIndex < 0:
                continue

            #rare b from W
            if abs(had.flavour) == 24:
                continue
            isTop = abs(had.flavour) == 6

            had.genJet = genJetsIndexed[had.jetIndex]

            if isTop:
                had.genJet.bHadMatch["nosel"]["fromTop"] += 1

            if not self.jetCut(had.genJet):
                continue

            if isTop:
                had.genJet.bHadMatch["pteta"]["fromTop"] += 1
            else:
                if had.fromTopWeak:
                    had.genJet.bHadMatch["pteta"]["afterTop"] += 1
                else:
                    had.genJet.bHadMatch["pteta"]["beforeTop"] += 1

        #get c-hadrons not coming from W
        for had in chads:
            if had.bHadronIdx >= 0:
                continue
            if abs(had.flavour) == 24:
                continue
            if had.jetIndex < 0:
                continue
            had.genJet = genJetsIndexed[had.jetIndex]

            if not self.jetCut(had.genJet):
                continue
            had.genJet.cHadMatch["pteta"]["beforeTop"] += 1

        #save gen jets that contain an interesting b/c hadron
        bBeforeTop = []
        bAfterTop = []
        cBeforeTop = []
        for gj in genJets:
            if not self.jetCut(gj):
                continue

            if (gj.bHadMatch["pteta"]["beforeTop"] >= 1 and
                gj.bHadMatch["pteta"]["fromTop"] == 0):
                bBeforeTop += [gj]
            if (gj.bHadMatch["pteta"]["afterTop"] >= 1 and
                gj.bHadMatch["pteta"]["fromTop"] == 0):
                bAfterTop += [gj]
            if (gj.cHadMatch["pteta"]["beforeTop"] >= 1 and
                gj.cHadMatch["pteta"]["fromTop"] == 0):
                cBeforeTop += [gj]


        #get the number of b/c hadrons per jets
        nh = [gj.bHadMatch["pteta"]["beforeTop"] for gj in bBeforeTop]
        nhc = [gj.cHadMatch["pteta"]["beforeTop"] for gj in cBeforeTop]

        cls = -1
        if len(bBeforeTop) >= 2:
            if max(nh) == 1:
                cls = 53 #two b-jets with one b hadron each
            elif min(nh)==1 and max(nh)>1:
                cls = 54 #two b-jets with one jet having 2 b hadrons
            elif min(nh)>1:
                cls = 55 #two b-jets with both having 2 b hadrons
        elif len(bBeforeTop) == 1:
            if max(nh) == 1:
                cls = 51 #one b jet with one b hadron
            else:
                cls = 52 #one b jet with two b hadrons
        elif len(bBeforeTop) == 0:
            if len(cBeforeTop) == 1:
                if max(nhc) == 1:
                    cls = 41 #1 c jet, 1 c hadron
                else:
                    cls = 42 #1 c jet, 2 c hadrons
            elif len(cBeforeTop) > 1:
                if max(nhc) == 1:
                    cls = 43 #2 c jets, 1 c hadron
                elif min(nhc)==1 and max(nhc)>1:
                    cls = 44 #2 c jets, one with one and the other with two c hadrons
                elif min(nhc)>1:
                    cls = 45 #2 c jets, 2 c hadrons
            else:
                cls = 0

        for gj in genJets:
            gj.numBHadronsBeforeTop = gj.bHadMatch["pteta"]["beforeTop"]
            gj.numCHadronsBeforeTop = gj.cHadMatch["pteta"]["beforeTop"]
            gj.numBHadronsAfterTop = gj.bHadMatch["pteta"]["afterTop"]
            gj.numCHadronsAfterTop = gj.cHadMatch["pteta"]["afterTop"]
            gj.numBHadronsFromTop = gj.bHadMatch["pteta"]["fromTop"]
            gj.numCHadronsFromTop = gj.cHadMatch["pteta"]["fromTop"]

        #event.genJetsHadronMatcher = sorted(genJets, key=lambda j: j.pt(), reverse=True)
        #classification as done by this code
        event.ttbarCls = cls

    def jetCut(self, jet):
        return jet.pt() > self.genJetMinPt and abs(jet.eta()) < self.genJetMaxEta
