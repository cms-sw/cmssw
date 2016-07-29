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
        self.extrapolateFromClosestBin = extrapolateFromClosestBin
        f.close()

    def get_2D(self, pt, eta):
        if not self.valid:
            return [1.0, 0.0]        

        stripForEta = 5
        if self.lep_binning not in self.res.keys():
            return [1.0, 0.0]

        if "abseta" in self.lep_binning:
            eta = abs(eta)
            stripForEta = 8

        # if no bin is found, search for closest one, and double the uncertainty
        closestEtaBin = ""
        closestPtBin = ""
        closestEta = 9999.
        closestPt = 9999.

        etaFound = False
        for etaKey, values in sorted(self.res[self.lep_binning].iteritems()) :
            etaL = float(((etaKey[stripForEta:]).rstrip(']').split(',')[0]))
            etaH = float(((etaKey[stripForEta:]).rstrip(']').split(',')[1]))

            ptFound = False

            if abs(etaL-eta)<closestEta or abs(etaH-eta)<closestEta and not etaFound:
                closestEta = min(abs(etaL-eta), abs(etaH-eta))
                closestEtaBin = etaKey

            if (eta>etaL and eta<etaH):
                closestEtaBin = etaKey
                #print 'etaL is', etaL
                #print 'etaH is', etaH
                etaFound = True                

            #print etaL, etaH
            for ptKey, result in sorted(values.iteritems()) :
                ptL = float(((ptKey[4:]).rstrip(']').split(',')[0]))
                ptH = float(((ptKey[4:]).rstrip(']').split(',')[1]))                

                if abs(ptL-pt)<closestPt or abs(ptH-pt)<closestPt and not ptFound:
                    closestPt = min(abs(ptL-pt), abs(ptH-pt))
                    closestPtBin = ptKey

                if (pt>ptL and pt<ptH):
                    closestPtBin = ptKey
                    #print 'ptL is', ptL
                    #print 'ptH is', ptH
                    #print 'results value is', result['value']
                    ptFound = True

                #print ptL, ptH
                #print "|eta| bin: %s  pT bin: %s\tdata/MC SF: %f +/- %f" % (etaKey, ptKey, result["value"], result["error"])
                if etaFound and ptFound:
                    #print 'both are true'
                    return [result["value"], result["error"]]

        if self.extrapolateFromClosestBin and not (closestPtBin=="" or closestEtaBin==""):
            #print 'closest bin for (%s,%s) is %s,%s' % (pt, eta , closestEtaBin, closestPtBin)
            #print '\t return ', [self.res[self.lep_binning][closestEtaBin][closestPtBin]["value"], self.res[self.lep_binning][closestEtaBin][closestPtBin]["error"]]
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
        #jsonpath+'SingleMuonTrigger_Z_RunBCD_prompt80X_7p65.json' :['IsoMu22_OR_IsoTkMu22_PtEtaBins_Run273158_to_274093', 'abseta_pt_DATA' ],
        #jsonpath+'MuonIso_Z_RunBCD_prompt80X_7p65.json' : ['MC_NUM_LooseRelIso_DEN_TightID_PAR_pt_spliteta_bin1', 'abseta_pt_ratio'],
        #jsonpath+'SingleMuonTrigger_Z_RunBCD_prompt80X_7p65.json' :['IsoMu22_OR_IsoTkMu22_PtEtaBins_Run274094_to_276097', 'abseta_pt_DATA' ],
        #jsonpath+'MuonTrkHIP_80X_Jul28.json' :[ 'ratio_eta', 'ratio_eta' ],
        #jsonpath+'MuonTrkHIP_80X_Jul28.json' :['ratio_vtx', 'ratio_vtx' ],
        #jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15.json' : ['runC_IsoMu20_OR_IsoTkMu20_PtEtaBins', 'abseta_pt_ratio' ]
        #jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15.json' : ['runD_IsoMu20_OR_IsoTkMu20_HLTv4p2_PtEtaBins', 'abseta_pt_ratio' ],
        #jsonpath+'SingleMuonTrigger_Z_RunCD_Reco76X_Feb15.json' : ['runD_IsoMu20_OR_IsoTkMu20_HLTv4p3_PtEtaBins', 'abseta_pt_ratio' ]
        }

    for j, name in jsons.iteritems():
        lepCorr = LeptonSF(j , name[0], name[1])
        weight = lepCorr.get_2D( 65 , -1.5)
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
    
