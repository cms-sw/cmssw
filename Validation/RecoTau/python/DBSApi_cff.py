def GetDbsInfo(toFind, requirements):
    "Interface with the DBS API to get the whatever you want of a requirements. ALWAYS RETURN A LIST OF STRINGS"
    from xml.dom.minidom import parseString
    from DBSAPI.dbsApi import DbsApi
    args = {}
    args['url']='http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet'
    args['version']='DBS_2_0_9'
    args['mode']='POST'
    api = DbsApi(args)
    data = api.executeQuery("find %s where %s" % (toFind, requirements))
    domresults = parseString(data)
    dbs = domresults.getElementsByTagName('dbs')
    result = dbs[0].getElementsByTagName('results')
    rows=result[0].getElementsByTagName('row')
    retList = []
    for row in rows:
        resultXML = row.getElementsByTagName(toFind)[0]
        node=(resultXML.childNodes)[0] #childNodes should be a one element array
        retList.append(str(node.nodeValue))
    return retList

#Matching names to real datasetNames
datasetDict={
    #GEN-SIM-RECO tier
    'ZTT' : { 'primds' : 'RelValZTT', 'tier' : 'GEN-SIM-RECO',},
    'QCD' : { 'primds' : 'RelValQCD_FlatPt_15_3000', 'tier' : 'GEN-SIM-RECO',},
    'ZMM' : { 'primds' : 'RelValZMM', 'tier' : 'GEN-SIM-RECO',},
    'ZEE' : { 'primds' : 'RelValZEE', 'tier' : 'GEN-SIM-RECO',},
    #Data
    'RealData'          : { 'primds' : 'Jet'           , 'tier' : 'RECO', 'dataset' : '*RelVal*'},
    'RealMuonsData'     : { 'primds' : 'SingleMu'      , 'tier' : 'RECO', 'dataset' : '*RelVal*'},
    'RealTausData'      : { 'primds' : 'TauPlusX'      , 'tier' : 'RECO', 'dataset' : '*RelVal*'},
    'RealElectronsData' : { 'primds' : 'SingleElectron', 'tier' : 'RECO', 'dataset' : '*RelVal*'},
    #FastSim
    'ZTTFastSim' : { 'primds' : 'RelValZTT', 'tier' : 'GEN-SIM-DIGI-RECO','dataset' : '*FastSim*',},
    #'FastSimQCD' : { 'primds' : 'RelValQCD_FlatPt_15_3000', 'tier' : 'GEN-SIM-DIGI-RECO','dataset' : '*FastSim*',}, NOT YET IN RELVAL CONTENT
    #'FastSimZMM' : { 'primds' : 'RelValZMM', 'tier' : 'GEN-SIM-DIGI-RECO','dataset' : '*FastSim*',},
    'ZEEFastSim' : { 'primds' : 'RelValZEE', 'tier' : 'GEN-SIM-DIGI-RECO','dataset' : '*FastSim*',},
    }

def FillSource(eventType,source):
    import os
    requirements = ''
    for item in datasetDict[eventType].items():
        requirements += item[0]+' = '+item[1]+' and '
    requirements += 'release = %s' % os.environ['CMSSW_VERSION']
    foundDs = GetDbsInfo('dataset', requirements)
    selDs = ''
    if len(foundDs) > 1:
        print "Multiple datasets found for %s! Which one you would like to use?" % eventType
        for ds in foundDs:
            print "%s  :  %s" % (foundDs.index(ds),ds)
        cnum = int(raw_input("\nselect Dataset: "))
        selDs = foundDs[cnum]
    elif len(foundDs) == 0:
        print "Sorry! No Dataset found, exiting..."
        return None
    else:
        selDs = foundDs[0]
    requirements = 'dataset = %s' % selDs
    files = GetDbsInfo('file', requirements)
    for entry in files:
        source.fileNames.append(entry)

def serialize(root):
        xmlstr = ''
        for key in root.keys():
            if isinstance(root[key], dict):
                xmlstr = '%s<%s>%s</%s>' % (xmlstr, key, serialize(root[key]), key)
            elif isinstance(root[key], list):
                xmlstr = '%s<%s>' % (xmlstr, key)
                for item in root[key]:
                    xmlstr = '%s%s' % (xmlstr, serialize(item))
                xmlstr = '%s</%s>' % (xmlstr, key)
            else:
                value = root[key]
                xmlstr = '%s<%s>%s</%s>' % (xmlstr, key, value, key)
        return xmlstr

def DictToXML(root):
    from xml.dom.minidom import parseString    
    outdom = parseString(serialize(root)) #closure test to check incopatibilities, and better printing
    return outdom.toprettyxml()

def loadXML(xml,eventType,source):
    from xml.dom.minidom import parse
    wrappedCont = parse(xml)
    content = wrappedCont.getElementsByTagName('dataFiles')[0]
    byType  = content.getElementsByTagName(eventType)
    if len(byType) == 0:
        return None
    fnames  = byType[0].getElementsByTagName('file')
    for fname in fnames:
        node = (fname.childNodes)[0] #childNodes should be a one element array
        source.fileNames.append(str(node.nodeValue).replace('\n','').replace('\t',''))
