import os
import json

# A class to apply SF's tabulated in json files
class LeptonSF:
    def __init__(self, lep_json, lep_name, lep_binning, extrapolateFromClosestBin=True) :
        if not os.path.isfile(lep_json):
            self.valid = False
            if lep_json!="":
                print "[LeptonSF]: Warning: ", lep_json, " is not a valid file. Return."
            else:
                print "[LeptonSF]: No file has been specified. Return."
        else:
            self.init(lep_json, lep_name, lep_binning, extrapolateFromClosestBin)

    def init(self, lep_json, lep_name, lep_binning, extrapolateFromClosestBin) :
        f = open(lep_json, 'r')             
        print '[LeptonSF]: Initialize with the following parameters:'
        print '\tfile:',lep_json
        print '\titem:', lep_name
        print '\tbinning:', lep_binning
        results = json.load(f)
        if lep_name not in results.keys():
            self.valid = False
            print "[LeptonSF]: Warning: ", lep_name , " is not a valid item. Return."
            return False
        self.res = results[lep_name]
        self.lep_name = lep_name
        self.lep_binning = lep_binning

        self.valid = True
        
        # use the closest bin with twice the uncertainty
        self.extrapolateFromClosestBin = extrapolateFromClosestBin

        # number of characters for correct string parsing
        self.stripForEta = 5
        if "abseta" in self.lep_binning:
            self.stripForEta = 8

        # map between eta-pt bin and scale factors
        self.eta_edge = {}

        # for 2D reweighting, allow each eta bin to have its own pt binning
        if lep_binning.find("pt")>-1 and lep_binning.find("eta")>-1:
            for etaKey, values in sorted(self.res[self.lep_binning].iteritems()) :
                self.eta_edge[etaKey] = {}
                self.eta_edge[etaKey]['low'] = float(((etaKey[self.stripForEta:]).rstrip(']').split(',')[0]))
                self.eta_edge[etaKey]['high'] = float(((etaKey[self.stripForEta:]).rstrip(']').split(',')[1]))
                self.eta_edge[etaKey]['pt_edge'] = {}
                for ptKey, result in sorted(values.iteritems()) :
                    self.eta_edge[etaKey]['pt_edge'][ptKey] = {}
                    self.eta_edge[etaKey]['pt_edge'][ptKey]['low'] = float(((ptKey[4:]).rstrip(']').split(',')[0]))
                    self.eta_edge[etaKey]['pt_edge'][ptKey]['high'] = float(((ptKey[4:]).rstrip(']').split(',')[1]))
        else:
            for ptKey, result in sorted(self.res[self.lep_binning].iteritems()) :
                self.eta_edge[ptKey] = {}
                self.eta_edge[ptKey]['low'] = float(((ptKey[4:]).rstrip(']').split(',')[0]))
                self.eta_edge[ptKey]['high'] = float(((ptKey[4:]).rstrip(']').split(',')[1]))        

        f.close()

    # method to get 1D factors
    def get_1D(self, pt):

        if not self.valid or self.lep_binning not in self.res.keys():
            return [1.0, 0.0]        

        # if no bin is found, search for closest one, and double the uncertainty
        closestPtBin = ""
        closestPt = 9999.
        ptFound = False

        for ptKey, result in sorted(self.res[self.lep_binning].iteritems()) :

            ptL = self.eta_edge[ptKey]['low']
            ptH = self.eta_edge[ptKey]['high']

            if abs(ptL-pt)<closestPt or abs(ptH-pt)<closestPt and not ptFound:
                closestPt = min(abs(ptL-pt), abs(ptH-pt))
                closestPtBin = ptKey
            if (pt>ptL and pt<ptH):
                closestPtBin = ptKey
                ptFound = True
            if ptFound:
                return [result["value"], result["error"]]

        if self.extrapolateFromClosestBin and not (closestPtBin==""):
            return [self.res[self.lep_binning][closestPtBin]["value"],2*self.res[self.lep_binning][closestPtBin]["error"]]
        else:
            return [1.0, 0.0]
                    
    # method to get 2D factors
    def get_2D(self, pt, eta):

        if not self.valid or self.lep_binning not in self.res.keys():
            return [1.0, 0.0]        

        if "abseta" in self.lep_binning:
            eta = abs(eta)

        # if no bin is found, search for closest one, and double the uncertainty
        closestEtaBin = ""
        closestPtBin = ""
        closestEta = 9999.
        closestPt = 9999.
        etaFound = False

        for etaKey, values in sorted(self.res[self.lep_binning].iteritems()) :

            etaL = self.eta_edge[etaKey]['low']
            etaH = self.eta_edge[etaKey]['high']

            ptFound = False

            if abs(etaL-eta)<closestEta or abs(etaH-eta)<closestEta and not etaFound:
                closestEta = min(abs(etaL-eta), abs(etaH-eta))
                closestEtaBin = etaKey
            if (eta>etaL and eta<etaH):
                closestEtaBin = etaKey
                etaFound = True                

            for ptKey, result in sorted(values.iteritems()) :

                ptL = self.eta_edge[etaKey]['pt_edge'][ptKey]['low']
                ptH = self.eta_edge[etaKey]['pt_edge'][ptKey]['high']

                if abs(ptL-pt)<closestPt or abs(ptH-pt)<closestPt and not ptFound:
                    closestPt = min(abs(ptL-pt), abs(ptH-pt))
                    closestPtBin = ptKey

                if (pt>ptL and pt<ptH):
                    closestPtBin = ptKey
                    ptFound = True

                if etaFound and ptFound:
                    return [result["value"], result["error"]]

        if self.extrapolateFromClosestBin and not (closestPtBin=="" or closestEtaBin==""):
            return [self.res[self.lep_binning][closestEtaBin][closestPtBin]["value"], 
                    2*self.res[self.lep_binning][closestEtaBin][closestPtBin]["error"]] 
        else:
            return [1.0, 0.0]


##################################################################################################
# EXAMPLE 
#

if __name__ == "__main__":

    jsonpath = os.environ['CMSSW_BASE']+"/src/VHbbAnalysis/Heppy/data/leptonSF/"
    jsons = {    
        #'muEff_HLT_RunC' : [ jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15_eff.json' , 'runC_IsoMu20_OR_IsoTkMu20_PtEtaBins', 'abseta_pt_MC' ],
        #'muEff_HLT_RunD4p2' : [ jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15_eff.json' , 'runD_IsoMu20_OR_IsoTkMu20_HLTv4p2_PtEtaBins', 'abseta_pt_MC' ],
        #'muEff_HLT_RunD4p3' : [ jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15_eff.json' , 'runD_IsoMu20_OR_IsoTkMu20_HLTv4p3_PtEtaBins', 'abseta_pt_MC' ],
        #'muSF_HLT_RunC' : [ jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15.json' , 'runC_IsoMu20_OR_IsoTkMu20_PtEtaBins', 'abseta_pt_ratio' ],
        #'eleEff_HLT_RunC' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
        #'eleEff_HLT_RunD4p2' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
        #'eleEff_HLT_RunD4p3' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
        #'eleSF_HLT_RunC' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
        #'eleSF_HLT_RunD4p2' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
        'eleSF_HLT_RunD4p3' : [jsonpath+'ScaleFactor_HLT_Ele23_WPLoose_Gsf_v.json','ScaleFactor_HLT_Ele23_WPLoose_Gsf_v', 'eta_pt_ratio'],
        #'eleSF_IdCutLoose' : [jsonpath+'CutBasedID_LooseWP.json', 'CutBasedID_LooseWP', 'abseta_pt_ratio'],
        #'eleSF_IdCutTight' : [jsonpath+'CutBasedID_TightWP.json', 'CutBasedID_TightWP', 'abseta_pt_ratio'],
        #'eleSF_IdMVALoose' : [jsonpath+'ScaleFactor_egammaEff_WP80.json', 'ScaleFactor_egammaEff_WP80', 'eta_pt_ratio'],
        #'eleSF_IdMVATight' : [jsonpath+'ScaleFactor_egammaEff_WP90.json', 'ScaleFactor_egammaEff_WP90', 'eta_pt_ratio'],
        #'eleSF_trk_eta' : [jsonpath+'EleGSFTrk_80X.json','ScaleFactor_GsfTracking_80X','eta_ratio'],
        #'muSF_HLT_RunD4p2' : [ jsonpath+'SingleMuonTrigger_Z_RunBCD_prompt80X_7p65.json' , 'IsoMu22_OR_IsoTkMu22_PtEtaBins_Run273158_to_274093', 'abseta_pt_DATA' ],
        #'muSF_HLT_RunD4p3' : [ jsonpath+'SingleMuonTrigger_Z_RunBCD_prompt80X_7p65.json' , 'IsoMu22_OR_IsoTkMu22_PtEtaBins_Run274094_to_276097', 'abseta_pt_DATA' ],
        #'muSF_IsoLoose' : [ jsonpath+'MuonIso_Z_RunBCD_prompt80X_7p65.json' , 'MC_NUM_LooseRelIso_DEN_TightID_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
        #'muSF_IsoTight' : [ jsonpath+'MuonIso_Z_RunBCD_prompt80X_7p65.json' , 'MC_NUM_TightRelIso_DEN_TightID_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
        #'muSF_IdCutLoose' : [ jsonpath+'MuonID_Z_RunBCD_prompt80X_7p65.json' , 'MC_NUM_LooseID_DEN_genTracks_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
        #'muSF_IdCutTight' : [ jsonpath+'MuonID_Z_RunBCD_prompt80X_7p65.json' , 'MC_NUM_TightIDandIPCut_DEN_genTracks_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
        #'muSF_trk_eta' : [ jsonpath+'MuonTrkHIP_80X_Jul28.json' , 'ScaleFactor_MuonTrkHIP_80X_eta', 'eta_ratio' ],
        }

    for j, name in jsons.iteritems():
        lepCorr = LeptonSF(name[0] , name[1], name[2])
        if name[2].find('pt')>-1 and name[2].find('eta')>-1 :
            weight = lepCorr.get_2D( 50 , 0.9)
        else:
            weight = lepCorr.get_1D(-0.7)
        val = weight[0]
        err = weight[1]
        print 'SF: ',  val, ' +/- ', err
    
    
    #jsons = {
    #    'SingleMuonTrigger_Z_RunCD_Reco74X_Dec1.json' : ['runD_IsoMu20_OR_IsoTkMu20_HLTv4p3_PtEtaBins', 'abseta_pt_ratio'],
    #    'MuonIso_Z_RunCD_Reco74X_Dec1.json' : ['NUM_LooseRelIso_DEN_LooseID_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'], 
    #    'MuonID_Z_RunCD_Reco74X_Dec1.json' : ['NUM_LooseID_DEN_genTracks_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'] ,
    #    'CutBasedID_LooseWP.json' : ['CutBasedID_LooseWP', 'eta_pt_ratio'],
    #    'CutBasedID_TightWP.json' : ['CutBasedID_TightWP', 'eta_pt_ratio'],
    #    'SingleMuonTrigger_Z_RunCD_Reco74X_Dec1_MC.json' : ['runD_IsoMu20_OR_IsoTkMu20_HLTv4p3_PtEtaBins', 'abseta_pt_MC'],
    #    }
    #
    #for j, name in jsons.iteritems():
    #    lepCorr = LeptonSF(j , name[0], name[1])
    #    weight = lepCorr.get_2D( 35. , 1.0)
    #    val = weight[0]
    #    err = weight[1]
    #    print j, name[0], ': ',  val, ' +/- ', err
    
