#! /usr/bin/env python

import os
import shutil
import sys
import fileinput
import string
import userparams

##############################################################
# Input parameters
#

print "Reading input parameters" 

#
#specify macros used to plot here
#
macro='macro/TrackValHistoPublisher.C'
macroSeed='macro/SeedValHistoPublisher.C'
macroReco='macro/RecoValHistoPublisher.C'
macroIsol='macro/IsoValHistoPublisher.C'
macroMuonReco='macro/RecoMuonValHistoPublisher.C'


##############################################################
# Helper functions
#
def GetGuiRepository(param):
    splitrelease=param['Release'].split('_')
    genericrelease='_'.join(splitrelease[0:3])+'_x/'
    dqmguirepository=param['DqmGuiBaseRepo']+genericrelease
    return dqmguirepository

def GetLabel(params):
    label = params['Condition']
    if (params['Condition'] == 'STARTUP'):
        label = 'START'
    elif (params['Condition']=='MC'):
        label ='MC_52_V1'
    label+=params['Label']
    if (params['FastSim']):
        label += '_FastSim'

    label += '-' + params['Version']
   
    if (params['PileUp']!='no'):
   	label = 'PU' + params['PileUp']+ '_' + label

    return label


def GetTag(params):
    tag = params['Condition']
    if (params['PileUp']=='no'):
        tag += '_noPU'
    else:
        tag += '_PU'+params['PileUp']
    if (params['FastSim']):
        tag +='_FSIM'
    return tag



def GetFastSimSuffix(params):
    if (params['FastSim']):
        return 'FS'
    else:
        return ''
  



def replace(map, filein, fileout):
    replace_items = map.items()
    while True:
        line = filein.readline()
        if not line: break
        for old, new in replace_items:
            line = string.replace(line, old, new)
        fileout.write(line)
    fileout.close()
    filein.close()

def myrootsubmit(cfgfile):
    command='root -b -q -l '+ cfgfile + '.C' + '>  macro.' + cfgfile +'.log'
    print('>> Executing \'' + command + '\'')
    os.system(command)
    print('   ... executed \'' + command + '\'');

def downloadfile(url):
    print('   + Downloading "' + url + '"...')
    stream = os.popen('/usr/bin/curl -k -O -L --capath $X509_CERT_DIR --key $X509_USER_PROXY --cert $X509_USER_PROXY -w "%{http_code}" '+ url)
    output=stream.readlines()
    if output[0]=='200':
        print('   + OK!')
	return True
    else:
        print('   + ERROR! ' + str(output[0]))
	print "Skipping " + url
	print "Please check the name of the file in the repository: "+GetGuiRepository(userparams.NewParams)
        sys.exit('Exiting...');
	# return False

def GetSamplePath(params, sample):
    return params['Release']+'/'+GetTag(params)+'/'+sample

def GetLocalSampleName(params, sample):
    return GetSamplePath(params, sample) + '/val.' + sample + '.root'


def createSampleDirectory(params, sample):
    path = GetSamplePath(params, sample)
    print('>> Creating directory ' + path + '...')
    if(os.path.exists(path)==False):
        os.makedirs(path)
    else:
        print('NOTE: Directory \'' + path + '\' already exists')

    return path


    
def getSampleFiles(params, sample):
    path = GetSamplePath(params,sample)
    print('>> Creating directory ' + path + '...')
    if(os.path.exists(path)==False):
        os.makedirs(path)
    else:
        print('NOTE: Directory \'' + path + '\' already exists')

    print('\n')

    # Check if pdfs have already been produced
    checkFile = path + '/general_tpToTkmuAssociation' + GetFastSimSuffix(params)+'.pdf'
    print('>> Checking if ' + checkFile + ' already exists...')
    if (os.path.isfile(checkFile)==True):
        print('   + Files of type ' + checkFile + ' exist already.')
        print('     Delete them first, if you really want to overwrite them')
        return
      
    print('   + The file does not exist so we continue')
    localsample=GetLocalSampleName(params, sample)
    sampleOnWeb=userparams.WebRepository+'/'+ localsample

    if (os.path.isfile(localsample)==True):
        print '   + ' + params['Type'] + 'sample file found at: ' +localsample + '. Using that one'
    elif (userparams.NewParams['GetFilesFrom']=='GUI'):
        theGuiSample = sample
        
        if (params['Condition'].find('PRE_LS1')!=-1 or params['Condition'].find('POSTLS1')!=-1):
            if (userparams.NewParams['FastSim']| userparams.RefParams['FastSim']):
                if ((sample=="RelValTTbar")|(sample=="RelValJpsiMM")|(sample=="RelValZMM")):
                    #theGuiPostFixLS1 = "_UPGpostls1_14"                                                                                                                               
                    theGuiPostFixLS1 = "_13"
                    theGuiSample = theGuiSample+theGuiPostFixLS1
                print "Sample is "+theGuiSample
            else:
                # Temporary fix due to the different names used for JPsiMM and TTbar samples in the DQM GUI
                if(sample=="RelValZpMM_2250_13TeV_Tauola"):
                    print "Subfix not added"     
                    theGuiPostFixLS1 = ""
                elif ((sample=="RelValTTbar")|(sample=="RelValJpsiMM")|(sample=="RelValZMM")):
                    #theGuiPostFixLS1 = "_UPGpostls1_14"
                    theGuiPostFixLS1 = "_13"
                else:
                    theGuiPostFixLS1 = "_UP15"
                    
            theGuiSample = theGuiSample+theGuiPostFixLS1

        guiFileName='DQM_V0001_R000000001__'+theGuiSample+'__'+params['Release']+'-'+GetLabel(params)+'__'+params['Format']+'.root'
        guiFullURL=GetGuiRepository(params)+guiFileName
        print ">> Downloading file from the GUI: " + guiFullURL
        #os.system('wget --ca-directory $X509_CERT_DIR/ --certificate=$X509_USER_PROXY --private-key=$X509_USER_PROXY '+GetGuiRepository(params)+guiFileName)
        #os.system('/usr/bin/curl -O -L --capath $X509_CERT_DIR --key $X509_USER_PROXY --cert $X509_USER_PROXY '+ guiFullURL)
        if (downloadfile(guiFullURL)==True):
		print('   + Moving ' + guiFileName + ' to ' + localsample)
        	shutil.move(guiFileName,localsample)

    elif (params['GetFilesFrom']=='CASTOR'):
        print '   + Getting new file from castor'
        params['Condition']=params['Condition']+GetFastSimSuffix(params)
        os.system('rfcp '+params['CastorRepository']+'/'+params['Release']+'_'+params['Condition']+'_'+sample+'_val.'+sample+'.root '+localsample)
    elif ((params['GetFilesFrom']=='WEB') & (os.path.isfile(sampleOnWeb))) :
        print "NOTE: New file found at: "+newSample+' -> Copy that one'
        os.system('cp '+sampleOnWeb+' '+path)
    else:
        print '*** WARNING: no signal file was found'


def getReplaceMap(newparams, refparams, sample, datatype, cfgkey, cfgfile):
    newLocalSample=GetLocalSampleName(newparams, sample)
    refLocalSample=GetLocalSampleName(refparams, sample)
    replace_map = { 'DATATYPE': datatype,
                    'NEW_FILE':newLocalSample,
                    'REF_FILE':refLocalSample,
                    'REF_LABEL':sample,
                    'NEW_LABEL': sample,
                    'REF_RELEASE':refparams['Release'],
                    'NEW_RELEASE':newparams['Release'],
                    'REFSELECTION':GetTag(refparams),
                    'NEWSELECTION':GetTag(newparams),
                    cfgkey: cfgfile
                    }
    return replace_map


##############################################################
# Main program
#

# Initial checks
# + Copying missing parameters from New to Ref
print('>> Checking input parameters...')
for i in userparams.NewParams:
    if i in userparams.RefParams:
        print('   + ' + i + ' set to:')
        print('     -' + str(userparams.NewParams[i]) + ' (new)')
        print('     -' + str(userparams.RefParams[i]) + ' (ref)')
    else:
        userparams.RefParams[i]=userparams.NewParams[i]
        print('   + ' + i + ' set to ' + str(userparams.NewParams[i]) + ' (both)')

# + Things needed by GUI
if ((userparams.NewParams['GetFilesFrom']=='GUI')|(userparams.RefParams['GetFilesFrom']=='GUI')):
    if os.getenv('X509_USER_PROXY','') == '':
        print "ERROR: It seems you did not configure your environment to be able"
        print "       to download files from the GUI. Your should follow these steps:"
        print " > source /afs/cern.ch/project/gd/LCG-share/sl5/etc/profile.d/grid_env.csh"
        print " > voms-proxy-init"
        quit()

if (userparams.NewParams['FastSim']|userparams.RefParams['FastSim']):
    userparams.ValidateDQM=False


# Choose samples based on input parameters
#print('>> Selecting samples...')
#if (userparams.NewParams['Condition']=='MC'):
#    userparams.samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValSingleMuPt1000','RelValTTbar']
#    if (userparams.NewParams['FastSim']|userparams.RefParams['FastSim']):
#        userparams.samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValTTbar']
#elif (userparams.NewParams['Condition']=='STARTUP'):
#    userparams.samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValSingleMuPt1000','RelValTTbar','RelValZMM','RelValJpsiMM']
#    if (userparams.NewParams['FastSim']|userparams.RefParams['FastSim']):
#        userparams.samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValTTbar']
#if ((userparams.NewParams['Condition'].find("PRE_LS1")!=-1)|(userparams.RefParams['Condition'].find("POSTS1")!=-1)):
#    #userparams.samples= ['RelValSingleMuPt10','RelValSingleMuPt100','RelValSingleMuPt1000','RelValTTbar','RelValZMM','RelValJpsiMM','RelValZpMM_2250_13TeV_Tauola']
#    userparams.samples= ['RelValZpMM_2250_13TeV_Tauola']
#if ((userparams.NewParams['PileUp'] != 'no')|(userparams.RefParams['PileUp']!='no')):
#    userparams.samples= ['RelValTTbar']
#    if (userparams.NewParams['FastSim']|userparams.RefParams['FastSim']):
#        userparams.samples= ['RelValTTbar']

# Itereate over userparams.samples
print('>> Samples selected:')
for sample in userparams.samples:
    print('   + ' + sample)
    
print('>> Processing userparams.samples:')
for sample in userparams.samples :
    print('###################################################')
    print('# ' + sample)
    print('###################################################')
    print('\n')



    # Create the directories to store the userparams.samples and pdfs
    newpath=createSampleDirectory(userparams.NewParams, sample)
    refpath=createSampleDirectory(userparams.RefParams, sample)
    print('\n')

    getSampleFiles(userparams.NewParams, sample)
    getSampleFiles(userparams.RefParams, sample)

        
    print('   + Producing macro files...')
    cfgFileName=sample+'_'+userparams.NewParams['Release']+'_'+userparams.RefParams['Release']
    hltcfgFileName='HLT'+sample+'_'+userparams.NewParams['Release']+'_'+userparams.RefParams['Release']
    seedcfgFileName='DQMSEED'+sample+'_'+userparams.NewParams['Release']+'_'+userparams.RefParams['Release']
    recocfgFileName='DQMRECO'+sample+'_'+userparams.NewParams['Release']+'_'+userparams.RefParams['Release']
    recomuoncfgFileName='RECO'+sample+'_'+userparams.NewParams['Release']+'_'+userparams.RefParams['Release']
    isolcfgFileName='ISOL'+sample+'_'+userparams.NewParams['Release']+'_'+userparams.RefParams['Release']
    

    if os.path.isfile(GetLocalSampleName(userparams.RefParams,sample)):
        replace_map = getReplaceMap(userparams.NewParams, userparams.RefParams, sample, 'RECO', 'TrackValHistoPublisher', cfgFileName)
        if (userparams.ValidateHLT):
            replace_map_HLT = getReplaceMap(userparams.NewParams, userparams.RefParams, sample, 'HLT', 'TrackValHistoPublisher', hltcfgFileName)
        if (userparams.ValidateDQM):
            replace_map_DIST = getReplaceMap(userparams.NewParams, userparams.RefParams, sample, 'RECO', 'RecoValHistoPublisher', recocfgFileName)
            replace_map_SEED = getReplaceMap(userparams.NewParams, userparams.RefParams, sample, 'RECO', 'SeedValHistoPublisher', seedcfgFileName)
        if (userparams.ValidateISO):
            replace_map_ISOL = getReplaceMap(userparams.NewParams, userparams.RefParams, sample, 'RECO', 'IsoValHistoPublisher', isolcfgFileName)
        if (userparams.ValidateRECO):
            replace_map_RECO = getReplaceMap(userparams.NewParams, userparams.RefParams, sample, 'RECO', 'RecoMuonValHistoPublisher', recomuoncfgFileName)
            replace_map_RECO['IS_FSIM']=''
    else:
        print "No reference file found at: ", refpath
        replace_map = getReplaceMap(userparams.NewParams, userparams.NewParams, sample, 'RECO', 'TrackValHistoPublisher', cfgFileName)
        if (userparams.ValidateHLT):
            replace_map_HLT = getReplaceMap(userparams.NewParams, userparams.NewParams, sample, 'HLT', 'TrackValHistoPublisher', hltcfgFileName)
        if (userparams.ValidateDQM):
            replace_map_DIST = getReplaceMap(userparams.NewParams, userparams.NewParams, sample, 'RECO', 'RecoValHistoPublisher', recocfgFileName)
            replace_map_SEED = getReplaceMap(userparams.NewParams, userparams.NewParams, sample, 'RECO', 'SeedValHistoPublisher', seedcfgFileName)
        if (userparams.ValidateISO):
            replace_map_ISOL = getReplaceMap(userparams.NewParams, userparams.NewParams, sample, 'RECO', 'IsoValHistoPublisher', isolcfgFileName)
        if (userparams.ValidateRECO):
            replace_map_RECO = getReplaceMap(userparams.NewParams, userparams.NewParams, sample, 'RECO', 'RecoMuonValHistoPublisher', recomuoncfgFileName)
            replace_map_RECO['IS_FSIM']=''

    templatemacroFile = open(macro, 'r')
    macroFile = open(cfgFileName+'.C' , 'w' )
    replace(replace_map, templatemacroFile, macroFile)

    if (userparams.ValidateHLT):
        templatemacroFile = open(macro, 'r')
        hltmacroFile = open(hltcfgFileName+'.C' , 'w' )
        replace(replace_map_HLT, templatemacroFile, hltmacroFile)

    if (userparams.ValidateDQM):
        templatemacroFile = open(macroReco, 'r')
        recomacroFile = open(recocfgFileName+'.C' , 'w' )
        replace(replace_map_DIST, templatemacroFile, recomacroFile)
        templatemacroFile = open(macroSeed, 'r')
        seedmacroFile = open(seedcfgFileName+'.C' , 'w' )
        replace(replace_map_SEED, templatemacroFile, seedmacroFile)

    if (userparams.ValidateISO):
        templatemacroFile = open(macroIsol, 'r')
        isolmacroFile = open(isolcfgFileName+'.C' , 'w' )
        replace(replace_map_ISOL, templatemacroFile, isolmacroFile)

    if (userparams.ValidateRECO):
        templatemacroFile = open(macroMuonReco, 'r')
        recomuonmacroFile = open(recomuoncfgFileName+'.C' , 'w' )
        replace(replace_map_RECO, templatemacroFile, recomuonmacroFile)

    if(userparams.Submit):
        myrootsubmit(cfgFileName)
        if (userparams.ValidateHLT):
            myrootsubmit(hltcfgFileName)
        if (userparams.ValidateDQM):
            myrootsubmit(recocfgFileName)
            myrootsubmit(seedcfgFileName)
        if (userparams.ValidateISO):
            myrootsubmit(isolcfgFileName)
            if (userparams.NewParams['FastSim']&userparams.RefParams['FastSim']):
                shutil.move(newpath+'/MuonIsolationV_inc.pdf',newpath+'/MuonIsolationV_inc_FS.pdf')
        if (userparams.ValidateRECO):
            myrootsubmit(recomuoncfgFileName)
            if (userparams.NewParams['FastSim']&userparams.RefParams['FastSim']):
                if (os.path.isfile(newpath+'/RecoMuonV.pdf') == True):
                    os.rename(newpath+'/RecoMuonV.pdf',newpath+'/RecoMuonV_FS.pdf')
                else:
                    print('ERROR: Could not find "' + newpath + '/RecoMuonV.pdf')

    if(userparams.Publish):
        newpath = GetSamplePath(userparams.NewParams,sample)
        newlocalsample = GetLocalSampleName(userparams.NewParams, sample)
        newdir=userparams.WebRepository + '/' + newpath
        print('>> Publishing to ' + newdir + '...')
        if(os.path.exists(newdir)==False):
            os.system('ssh '+userparams.User+'@lxplus.cern.ch mkdir -p ' + newdir)
            # os.makedirs(newdir)
        # os.system('rm '+ newlocalsample)  
        # os.system('scp -r '+newpath+'/* ' + newdir)
        os.system('scp -r '+newpath+'/* '+userparams.User+'@lxplus.cern.ch:' + newdir)    
        print('Newpath is' + newlocalsample + ' and ' + newpath)
