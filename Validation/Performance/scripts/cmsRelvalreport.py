#! /usr/bin/env python

r'''
cmsRelvalreport.py: a script to run performance tests and produce reports in a automated way.
'''

import glob,re #Used for IgProf Analyse work with IgProf Mem profile dumps
# Configuration parameters:#############################################################

# Perfreport 3 and 2 coordinates:
PR3_BASE='/afs/cern.ch/user/d/dpiparo/w0/perfreport3installation/'
PR3=PR3_BASE+'/bin/perfreport'# executable
PERFREPORT3_PATH=PR3_BASE+'/share/perfreport' #path to xmls
#PR3_PRODUCER_PLUGIN=PR3_BASE+'/lib/libcmssw_by_producer.so' #plugin for fpage
PR3_PRODUCER_PLUGIN='/afs/cern.ch/user/d/dpiparo/w0/pr3/perfreport/plugins/cmssw_by_producer/libcmssw_by_producer.so'

#PR2_BASE='/afs/cern.ch/user/d/dpiparo/w0/perfreport2.1installation/'
PR2_BASE='/afs/cern.ch/user/g/gbenelli/public/PerfReport2/2.0.1/'
PR2='%s' %(PR2_BASE+'/bin/perfreport')# executable
PERFREPORT2_PATH=PR2_BASE+'/share/perfreport' #path to xmls

import os
cmssw_base=os.environ["CMSSW_BASE"]
cmssw_release_base=os.environ["CMSSW_RELEASE_BASE"]
pyrelvallocal=cmssw_base+"/src/Configuration/PyReleaseValidation"
valperf=cmssw_base+"/src/Validation/Performance"
#Set the path depending on the presence of a locally checked out version of PyReleaseValidation
if os.path.exists(pyrelvallocal):
    RELEASE='CMSSW_BASE'
    print "Using LOCAL version of Configuration/PyReleaseValidation instead of the RELEASE version"
elif not os.path.exists(pyrelvallocal):
    RELEASE='CMSSW_RELEASE_BASE'
    
#Eliminating the full paths to all the scripts used they are assumed to be in the release by default, being in the scripts/ directory of their package    
# Valgrind Memcheck Parser coordinates:
#VMPARSER='%s/src/Utilities/ReleaseScripts/scripts/valgrindMemcheckParser.pl' %os.environ['CMSSW_RELEASE_BASE']#This has to always point to 'CMSSW_RELEASE_BASE'
VMPARSER='valgrindMemcheckParser.pl'

#Not a script... need to keep the full path to the release
# Valgrind Memcheck Parser output style file coordinates:
VMPARSERSTYLE='%s/src/Utilities/ReleaseScripts/data/valgrindMemcheckParser.css' %os.environ['CMSSW_RELEASE_BASE']#This has to always point to 'CMSSW_RELEASE_BASE'

# IgProf_Analysis coordinates:
#IGPROFANALYS='%s/src/Validation/Performance/scripts/cmsIgProf_Analysis.py'%os.environ[RELEASE]
IGPROFANALYS='cmsIgProf_Analysis.py'

# Timereport parser
#TIMEREPORTPARSER='%s/src/Validation/Performance/scripts/cmsTimeReport.pl'%os.environ[RELEASE]
TIMEREPORTPARSER='cmsTimeReport.pl'

# Simple memory parser
#SIMPLEMEMPARSER='%s/src/Validation/Performance/scripts/cmsSimplememchecker_parser.py' %os.environ[RELEASE]
SIMPLEMEMPARSER='cmsSimplememchecker_parser.py'

# Timing Parser
#TIMINGPARSER='%s/src/Validation/Performance/scripts/cmsTiming_parser.py' %os.environ[RELEASE]
TIMINGPARSER='cmsTiming_parser.py'

# makeSkimDriver
MAKESKIMDRIVERDIR='%s/src/Configuration/EventContent/test' %os.environ[RELEASE]
MAKESKIMDRIVER='%s/makeSkimDriver.py'%MAKESKIMDRIVERDIR

########################################################################################


# Library to include to run valgrind fce
VFCE_LIB='/afs/cern.ch/user/m/moserro/public/vgfcelib' 
PERL5_LIB='/afs/cern.ch/user/d/dpiparo/w0/PERLlibs/5.8.0'



# the profilers that use the stout of the app..
STDOUTPROFILERS=['Memcheck_Valgrind',
                 'Timereport_Parser',
                 'Timing_Parser',
                 'SimpleMem_Parser']
# Profilers list
PROFILERS=['ValgrindFCE',
           'IgProf_perf',
           'IgProf_mem',
           'Edm_Size']+STDOUTPROFILERS
                            

# name of the executable to benchmark. It can be different from cmsRun in future           
EXECUTABLE='cmsRun'

# Command execution and debug switches
EXEC=True
DEBUG=True

#Handy dictionaries to handle the mapping between IgProf Profiles and counters:
IgProfCounters={'IgProfPerf':['PERF_TICKS'],
                'IgProfMem':['MEM_TOTAL','MEM_LIVE','MEM_MAX']
                }
IgProfProfiles={'PERF_TICKS':'IgProfPerf',
               'MEM_TOTAL':'IgProfMem',
               'MEM_LIVE':'IgProfMem',
               'MEM_MAX':'IgProfMem'
               }
import time   
import optparse 
import sys


#######################################################################
def red(string):
    return '%s%s%s' %('\033[1;31m',string,'\033[1;0m')    
def green(string):
    return '%s%s%s' %('\033[1;32m',string,'\033[1;0m') 
def yellow(string):
    return '%s%s%s' %('\033[1;33m',string,'\033[1;0m')     
#######################################################################

def clean_name(name):
    '''
    Trivially removes an underscore if present as last char of a string
    '''
    i=-1
    is_dirty=True
    while(is_dirty):
        if name[i]=='_':
            name=name[:-1]
        else:
            return name
        i-=1

#######################################################################

def execute(command):
        '''
        It executes command if the EXEC switch is True. 
        Catches exitcodes different from 0.
        '''
        logger('%s %s ' %(green('[execute]'),command))
        if EXEC:
            exit_code=os.system(command)
            if exit_code!=0:
                logger(red('*** Seems like "%s" encountered problems.' %command))
            return exit_code
        else:
            return 0
            
#######################################################################            
            
def logger(message,level=0):
    '''
    level=0 output, level 1 debug.
    '''                  
    message='%s %s' %(yellow('[RelValreport]'),message)
    
    sys.stdout.flush()
    
    if level==0:
        print message
    if level==1 and DEBUG:
        print message    
    
    sys.stdout.flush()

#######################################################################

class Candles_file:
    '''
    Class to read the trivial ASCII file containing the candles
    '''
    def __init__(self, filename):
        
        self.commands_profilers_meta_list=[]    
    
        candlesfile=open(filename,'r')
        
        if filename[-3:]=='xml':
            command=''
            profiler=''
            meta=''
            db_meta=''
            reuse=False    
            
            from xml.dom import minidom
            
            # parse the config
            xmldoc = minidom.parse(filename)
            
            # get the candles
            candles_list = xmldoc.getElementsByTagName('candle')
            
            # a list of dictionaries to store the info
            candles_dict_list=[]
            
            for candle in candles_list:
                info_dict={}
                for child in candle.childNodes:# iteration over candle node children
                    if not child.__dict__.has_key('nodeName'):# if just a text node skip!
                        #print 'CONTINUE!'
                        continue
                    # We pick the info from the node
                    tag_name=child.tagName
                    #print 'Manipulating a %s ...'%tag_name
                    data=child.firstChild.data
                    #print 'Found the data: %s !' %data
                    # and we put it in the dictionary
                    info_dict[tag_name]=data
                # to store it in a list
                candles_dict_list.append(info_dict)
            
            # and now process what was parsed
                        
            for candle_dict in candles_dict_list:
                # compulsory params!!
                command=candle_dict['command']
                profiler=candle_dict['profiler']
                meta=candle_dict['meta']
                # other params
                try:
                    db_meta=candle_dict['db_meta']
                except:
                    db_meta=None
                try:
                    reuse=candle_dict['reuse']
                except:
                    reuse=False    
                            
                self.commands_profilers_meta_list.append([command,profiler,meta,reuse,db_meta])
        
        # The file is a plain ASCII
        else:
            for candle in candlesfile.readlines():
                # Some parsing of the file
                if candle[0]!='#' and candle.strip(' \n\t')!='': # if not a comment or an empty line
                    if candle[-1]=='\n': #remove trail \n if it's there
                        candle=candle[:-1] 
                    splitted_candle=candle.split('@@@') #separate
                    
                    # compulsory fields
                    command=splitted_candle[0]
                    profiler=splitted_candle[1].strip(' \t')
                    meta=splitted_candle[2].strip(' \t')        
                    info=[command,profiler,meta]
                    
                    # FIXME: AN .ini or xml config??
                    # do we have something more?
                    len_splitted_candle=len(splitted_candle)
                    reuse=False
                    if len_splitted_candle>3:
                        # is it a reuse statement?
                        if 'reuse' in splitted_candle[3]:
                            reuse=True
                        info.append(reuse)
                    else:
                        info.append(reuse)               
                    
                    # we have one more field or we hadn't a reuse in the last one    
                    if len_splitted_candle>4 or (len_splitted_candle>3 and not reuse):
                        cmssw_scram_version_string=splitted_candle[-1].strip(' \t')
                        info.append(cmssw_scram_version_string)
                    else:
                        info.append(None)
    
                        
                    self.commands_profilers_meta_list.append(info)
                    
    #----------------------------------------------------------------------
        
    def get_commands_profilers_meta_list(self):
        return self.commands_profilers_meta_list
            
#######################################################################

class Profile:
    '''
    Class that represents the procedure of performance report creation
    '''
    def __init__(self,command,profiler,profile_name):
        self.command=command
        self.profile_name=profile_name
        self.profiler=profiler
    
    #------------------------------------------------------------------
    # edit here if more profilers added
    def make_profile(self):
        '''
        Launch the right function according to the profiler name.
        '''  
        if self.profiler=='ValgrindFCE':
            return self._profile_valgrindfce()
        elif self.profiler.find('IgProf')!=-1:
            return self._profile_igprof()    
        elif self.profiler.find('Edm_Size')!=-1:
            return self._profile_edmsize()
        elif self.profiler=='Memcheck_Valgrind':
            return self._profile_Memcheck_Valgrind()
        elif self.profiler=='Timereport_Parser':
            return self._profile_Timereport_Parser()
        elif self.profiler=='Timing_Parser':
            return self._profile_Timing_Parser()        
        elif self.profiler=='SimpleMem_Parser':
            return self._profile_SimpleMem_Parser()
        elif self.profiler=='':
            return self._profile_None()
        elif self.profiler=='None': #adding this for the case of candle ASCII file non-profiling commands
            return self._profile_None()
        else:
            raise('No %s profiler found!' %self.profiler)
    #------------------------------------------------------------------
    def _profile_valgrindfce(self):
        '''
        Valgrind profile launcher.
        '''
        # ValgrindFCE needs a special library to run
        os.environ["VALGRIND_LIB"]=VFCE_LIB
        
        profiler_line=''
        valgrind_options= 'time valgrind '+\
                          '--tool=callgrind '+\
                          '--fce=%s ' %(self.profile_name)
        
        # If we are using cmsDriver we should use the prefix switch        
        if EXECUTABLE=='cmsRun' and self.command.find('cmsDriver.py')!=-1:
            profiler_line='%s --prefix "%s"' %(self.command,valgrind_options)
                            
        else:                          
            profiler_line='%s %s' %(valgrind_options,self.command)
                        #'--trace-children=yes '+\
        
        return execute(profiler_line)
    
    #------------------------------------------------------------------
    def _profile_igprof(self): 
        '''
        IgProf profiler launcher.
        '''
        profiler_line=''
        
        igprof_options='igprof -d -t %s ' \
                    %EXECUTABLE # IgProf profile not general only for CMSRUN!
        
        # To handle Igprof memory and performance profiler in one function
        if self.profiler=='IgProf_perf':
            igprof_options+='-pp '
        elif self.profiler=='IgProf_mem':
            igprof_options+='-mp '
        else:
            raise ('Unknown IgProf flavour: %s !'%self.profiler)
        igprof_options+='-z -o %s' %(self.profile_name)
        
        # If we are using cmsDriver we should use the prefix switch 
        if EXECUTABLE=='cmsRun' and self.command.find('cmsDriver.py')!=-1:
            profiler_line='%s --prefix "%s"' %(self.command,igprof_options) 
        else:
            profiler_line='%s %s' %(igprof_options, self.command)  
            
        return execute(profiler_line)
    
    #------------------------------------------------------------------
    
    def _profile_edmsize(self):
        '''
        Launch edm size profiler
        '''
        # In this case we replace the name to be clear
        input_rootfile=self.command
        
        # Skim the content if requested!
        if '.' in self.profiler:
            
            clean_profiler_name,options=self.profiler.split('.')
            content,nevts=options.split(',')
            outfilename='%s_%s.root'%(os.path.basename(self.command)[:-6],content)
            oldpypath=os.environ['PYTHONPATH']
            os.environ['PYTHONPATH']+=':%s' %MAKESKIMDRIVERDIR
            execute('%s -i %s -o %s --outputcommands %s -n %s' %(MAKESKIMDRIVER,
                                                                 self.command,
                                                                 outfilename,
                                                                 content,
                                                                 nevts)) 
            os.environ['PYTHONPATH']=oldpypath
            #execute('rm %s' %outfilename)
            self.command=outfilename
            self.profiler=clean_profiler_name
                                                                                                        
        
        profiler_line='edmEventSize -o %s -d %s'\
                            %(self.profile_name,self.command)
        
        return execute(profiler_line)
   
   #------------------------------------------------------------------
   
    def _profile_Memcheck_Valgrind(self):
        '''
        Valgrind Memcheck profile launcher
        '''
        profiler_line=''
        #Adding cms suppression of useless messages (cmsvgsupp)
        #Removing leak-checking (done with igprof)
        #'--leak-check=no '+\ (no is the default)
        #'--show-reachable=yes '+\
        #'--track-fds=yes '
        #Adding xml logging
        xmlFileName = self.profile_name.replace(",","-")[:-4] + ".xml"
        valgrind_options='time valgrind --track-origins=yes '+\
                               '--tool=memcheck `cmsvgsupp` '+\
                               '--num-callers=20 '+\
                               '--xml=yes '+\
                               '--xml-file=%s '%xmlFileName
        
        # If we are using cmsDriver we should use the prefix switch        
        if EXECUTABLE=='cmsRun' and self.command.find('cmsDriver.py')!=-1:
            #Replacing 2>&1 |tee with >& in the shell command to preserve the return code significance:
            # using tee return would be 0 even if the command failed before the pipe:
            profiler_line='%s --prefix "%s" >& %s' %(self.command,valgrind_options,self.profile_name)
                            
        else:                          
            profiler_line='%s %s >& %s' %(valgrind_options,self.command,self.profile_name)
                        #'--trace-children=yes '+\
        exec_status = execute(profiler_line)

        # compact the xml by removing the Leak_* errors
        if not exec_status:
            newFileName = xmlFileName.replace('valgrind.xml', 'vlgd.xml')
            compactCmd = 'xsltproc --output %s %s/test/filterOutValgrindLeakErrors.xsl %s' %(newFileName, valperf, xmlFileName)
            execute(compactCmd)
        
        return exec_status
    #-------------------------------------------------------------------
    
    def _profile_Timereport_Parser(self):
        return self._save_output()
        
    #-------------------------------------------------------------------
    
    def _profile_SimpleMem_Parser(self):
        return self._save_output()

    #-------------------------------------------------------------------
    
    def _profile_Timing_Parser(self):
        return self._save_output()     
    
    #-------------------------------------------------------------------
        
    def _save_output(self):
        '''
        Save the output of cmsRun on a file!
        '''               
#         # a first maquillage about the profilename:
#         if self.profile_name[-4:]!='.log':
#             self.profile_name+='.log'
        #Replacing 2>&1 |tee with >& in the shell command to preserve the return code significance:
        # using tee return would be 0 even if the command failed before the pipe:
        profiler_line='%s  >& %s' %(self.command,self.profile_name)
        return execute(profiler_line)    

    #-------------------------------------------------------------------                    
    
    def _profile_None(self):
        '''
        Just Run the command!
        '''
        return execute(self.command)
    
    #-------------------------------------------------------------------
    
    def make_report(self,
                    fill_db=False,
                    db_name=None,
                    tmp_dir=None,
                    outdir=None,
                    IgProf_option=None,
                    metastring=None):
        '''
        Make a performance report with CMSSW scripts for CMSSW internal profiling (Timing/SimpleMemoryCheck) and Memcheck, PR2 for edmEventSize and Callgrind (NOTE PR2 is not supported anymore and is not currently in the CMSSW external, running froma privat AFS!), igprof-analyse for all IgProf profiling.
        '''

        if outdir==None or outdir==self.profile_name:
            outdir=self.profile_name+'_outdir'
            
        #Create the directory where the report will be stored:
        if not os.path.exists(outdir) and not fill_db and not IgProf_option:
            #Added an IgProf condition to avoid the creation of a directory that we will not need anymore, since we will put all info in the filenames
            execute('mkdir %s' %outdir)
        
        if fill_db:
            db_option='-a'
            if not os.path.exists(db_name):
                db_option='-A'
        
        # temp in the local dir for PR
        tmp_switch=''    
        if tmp_dir!='':
            execute('mkdir %s' %tmp_dir)
            tmp_switch=' -t %s' %tmp_dir

        #Handle the various profilers:
        
        ##################################################################### 
        
        # Profiler is ValgrindFCE:
        if self.profiler=='ValgrindFCE':
            perfreport_command=''
            # Switch for filling the db
            if not fill_db:
                os.environ["PERFREPORT_PATH"]='%s/' %PERFREPORT2_PATH
                perfreport_command='%s %s -ff -i %s -o %s' %(PR2,
                                                             tmp_switch,
                                                             self.profile_name,
                                                             outdir)
            else:
                os.environ["PERFREPORT_PATH"]='%s/' %PERFREPORT3_PATH
                perfreport_command='%s %s -n5000 -u%s  -ff  -m \'scram_cmssw_version_string,%s\' -i %s %s -o %s' \
                                                    %(PR3,
                                                      tmp_switch,
                                                      PR3_PRODUCER_PLUGIN,
                                                      metastring,
                                                      self.profile_name,
                                                      db_option,
                                                      db_name)
            return execute(perfreport_command)
        
        #####################################################################            
            
        # Profiler is IgProf:
        if self.profiler.find('IgProf')!=-1:
            #First the case of IgProf PERF and MEM reporting:
            if not 'ANALYSE' in IgProf_option:
                #Switch to the use of igprof-analyse instead of PerfReport!
                #Will use the ANALYSE case for regressions between early event dumps and late event dumps of the profiles
                #Following Andreas suggestion, add the number of events for the EndOfJob report
                NumberOfEvents=self.command.split()[3] #FIXME: this is quite hardcoded... but should be stable...
                sqlite_outputfile=self.profile_name.split(".")[0].replace(IgProfProfiles[IgProf_option[0]],IgProf_option[0])+'___'+NumberOfEvents+'_EndOfJob.sql3'
                logger("Executing the report of the IgProf end of job profile")
                exit=execute('igprof-analyse --sqlite -d -v -g -r %s %s | sqlite3 %s'%(IgProf_option[0],self.profile_name,sqlite_outputfile)) 
                return exit
            #Then the "ANALYSE" case that we want to use to add to the same directories (Perf, MemTotal, MemLive)
            #also some other analyses and in particular:
            #1-the first 7 lines of the ASCII analysis of the IgProf profile dumps (total of the counters)
            #2-the dump at different event numbers,
            #3-the diff between the first and last dump,
            #4-the counters grouped by library using regexp at the last dump:
            else: #We use IgProf Analysis
                #Set the IgProfCounter from the ANALYSE.MEM_TOT style IgProf_option
                #print IgProf_option
                IgProfCounter=IgProf_option[1]
                #Add here the handling of the new IgProf.N.gz files so that they will get preserved and not overwritten:
                logger("Looking for IgProf intermediate event profile dumps")
                #Check if there are IgProf.N.gz dump files:
                IgProfDumps=glob.glob("IgProf.*.gz")
                #in case there are none check if they have already been mv'd to another name to avoid overwriting
                #(MEM_LIVE usually re-uses MEM_TOTAL, so the IgProf.N.gz files will already have a MemTotal name...)
                if not IgProfDumps:
                    localFiles=os.listdir('.')
                    IgProfDumpProfilesPrevious=re.compile(r"\w+.\d+.gz")
                    IgProfDumps=filter(lambda x: IgProfDumpProfilesPrevious.search(x),localFiles)
                #Now if there are dumps execute the following analyses:
                if IgProfDumps:
                    IgProfDumps.sort()
                    logger("Found the following IgProf intermediate event profile dumps:")
                    logger(IgProfDumps)
                    FirstDumpEvent=9999999
                    LastDumpEvent=0
                    exit=0
                    for dump in IgProfDumps:
                        if "___" in dump:
                            DumpEvent=dump.split(".")[0].split("___")[-1]
                        else:
                            DumpEvent=dump.split(".")[1]
                        #New naming convention using ___ as separator
                        DumpedProfileName=self.profile_name[:-3]+"___"+DumpEvent+".gz"
                        if dump.startswith("IgProf"):
                            execute('mv %s %s'%(dump,DumpedProfileName))
                        #Keep a tab of the first and last dump event for the diff analysis
                        if int(DumpEvent) < FirstDumpEvent:
                            FirstDumpEvent = int(DumpEvent)
                        if int(DumpEvent) > LastDumpEvent:
                            LastDumpEvent = int(DumpEvent)
                        #Eliminating the ASCII analysis to get the totals, Giulio will provide this information in igprof-navigator with a special query
                        #First type of analysis: dump first 7 lines of ASCII analysis:
                        #logger("Executing the igprof-analyse analysis to dump the ASCII 7 lines output with the totals for the IgProf counter")
                        #exit=execute('%s -o%s -i%s -t%s' %(IGPROFANALYS,outdir,DumpedProfileName,"ASCII"))
                        #Second type of analysis: dump the report in sqlite format to be ready to be browsed with igprof-navigator
                        logger("Executing the igprof-analyse analysis saving into igprof-navigator browseable SQLite3 format")
                        #exit=exit+execute('%s -o%s -i%s -t%s' %(IGPROFANALYS,outdir,DumpedProfileName,"SQLite3"))
                        #Execute all types of analyses available with the current profile (using the dictionary IgProfProfile):
                        #To avoid this we should use a further input in SimulationCandles.txt IgProfMem.ANALYSE.MEM_TOTAL maybe the cleanest solution.
                        #for IgProfile in IgProfCounters.keys():
                        #    if DumpedProfileName.find(IgProfile)>0:
                        #        for counter in IgProfCounters[IgProfile]:
                        #Check that the file does not exist:
                        #if not os.path.exists(DumpedProfileName.split(".")[0].replace(IgProfProfiles[counter],counter)+".sql3"):
                        exit=exit+execute('%s -c%s -i%s -t%s' %(IGPROFANALYS,IgProfCounter,DumpedProfileName,"SQLite3"))
                                    #else:
                                    #    print "File %s already exists will not process profile"%DumpedProfileName.split(".")[0].replace(IgProfProfiles[counter],counter)+".sql3"
                    #FIXME:
                    #Issue with multiple profiles in the same dir: assuming Perf and Mem will always be in separate dirs
                    #Potential ssue with multiple steps?
                    #Adapting to new igprof naming scheme:
                    FirstDumpProfile=self.profile_name[:-3]+"___"+str(FirstDumpEvent)+".gz"
                    LastDumpProfile=self.profile_name[:-3]+"___"+str(LastDumpEvent)+".gz"
                    #Third type of analysis: execute the diff analysis:
                    #Check there are at least 2 IgProf intermediate event dump profiles to do a regression!
                    if len(IgProfDumps)>1:
                        logger("Executing the igprof-analyse regression between the first IgProf profile dump and the last one")
                        #for IgProfile in IgProfCounters.keys():
                        #    if DumpedProfileName.find(IgProfile)>0:
                        #        IgProfCounter=IgProfCounters[IgProfile]
                        exit=exit+execute('%s -c%s -i%s -r%s' %(IGPROFANALYS,IgProfCounter,LastDumpProfile,FirstDumpProfile))
                    else:
                        logger("CANNOT execute any regressions: not enough IgProf intermediate event profile dumps!")
                    #Fourth type of analysis: execute the grouped by library igprof-analyse:
                    logger("Executing the igprof-analyse analysis merging the results by library via regexp and saving the result in igprof-navigator browseable SQLite3 format")
                    #for IgProfile in IgProfCounters.keys():
                    #        if DumpedProfileName.find(IgProfile)>0:
                    #            IgProfCounter=IgProfCounters[IgProfile]
                    exit=exit+execute('%s -c%s -i%s --library' %(IGPROFANALYS,IgProfCounter,LastDumpProfile))
                #If they are not there at all (no dumps)
                else:
                    logger("No IgProf intermediate event profile dumps found!")
                    exit=0
                
                return exit
                

        #####################################################################                     
            
        # Profiler is EdmSize:        
        if 'Edm_Size' in self.profiler:
            perfreport_command=''
            if not fill_db:
                os.environ["PERFREPORT_PATH"]='%s/' \
                                            %PERFREPORT2_PATH
                perfreport_command='%s %s -fe -i %s -o %s' \
                                            %(PR2,
                                              tmp_switch,
                                              self.profile_name,
                                              outdir)
            else:
                os.environ["PERFREPORT_PATH"]='%s/' \
                                            %PERFREPORT3_PATH
                perfreport_command='%s %s -n5000 -u%s -fe -i %s -a -o %s' \
                                            %(PR3,
                                              tmp_switch,
                                              PR3_PRODUCER_PLUGIN,
                                              self.profile_name,
                                              db_name)             

            return execute(perfreport_command)    

        #FIXME: probably need to move this somewhere else now that we use return statements
        if tmp_dir!='':
            execute('rm -r %s' %tmp_dir)
            
        #####################################################################    
                            
        # Profiler is Valgrind Memcheck
        if self.profiler=='Memcheck_Valgrind':
            # Three pages will be produced:
            os.environ['PERL5LIB']=PERL5_LIB
            report_coordinates=(VMPARSER,self.profile_name,outdir)
            # Copy the Valgrind Memcheck parser style file in the outdir
            copyStyleFile='cp -pR %s %s'%(VMPARSERSTYLE,outdir)
            execute(copyStyleFile)
            report_commands=('%s --preset +prod,-prod1 %s > %s/edproduce.html'\
                                %report_coordinates,
                             '%s --preset +prod1 %s > %s/esproduce.html'\
                                %report_coordinates,
                             '%s -t beginJob %s > %s/beginjob.html'\
                                %report_coordinates)
            exit=0
            for command in report_commands:
                exit= exit + execute(command)      
            return exit
        #####################################################################                
                
        # Profiler is TimeReport parser
        
        if self.profiler=='Timereport_Parser':
            return execute('%s %s %s' %(TIMEREPORTPARSER,self.profile_name,outdir))

        #####################################################################
        
        # Profiler is Timing Parser            
        
        if self.profiler=='Timing_Parser':
            return execute('%s -i %s -o %s' %(TIMINGPARSER,self.profile_name,outdir))
        
                    
        #####################################################################
        
        # Profiler is Simple memory parser
        
        if self.profiler=='SimpleMem_Parser':
            return execute('%s -i %s -o %s' %(SIMPLEMEMPARSER,self.profile_name,outdir))
            
        #####################################################################
        
        # no profiler
            
        if self.profiler=='' or self.profiler=='None': #Need to catch the None case, since otherwise we get no return code (crash for pre-requisite step running).
            return 0         #Used to be pass, but we need a return 0 to handle exit code properly!         
                                                                
#############################################################################################

def principal(options):
    '''
    Here the objects of the Profile class are istantiated.
    '''
    #Add a global exit code variable, that is the sum of all exit codes, to return it at the end:
    exitCodeSum=0
    # Build a list of commands for programs to benchmark.
    # It will be only one if -c option is selected
    commands_profilers_meta_list=[]
    
    # We have only one
    if options.infile=='':
        logger('Single command found...')
        commands_profilers_meta_list.append([options.command,'','',False,''])
    
    # We have more: we parse the list of candles    
    else:
        logger('List of commands found. Processing %s ...' %options.infile)
        
        # an object that represents the candles file:
        candles_file = Candles_file(options.infile)
        
        commands_profilers_meta_list=candles_file.get_commands_profilers_meta_list()
       
        
    logger('Iterating through commands of executables to profile ...')
    
    # Cycle through the commands
    len_commands_profilers_meta_list=len(commands_profilers_meta_list)    
    
    commands_counter=1
    precedent_profile_name=''
    precedent_reuseprofile=False
    for command,profiler_opt,meta,reuseprofile,db_metastring in commands_profilers_meta_list:
                  
        exit_code=0
        
        logger('Processing command %d/%d' \
                    %(commands_counter,len_commands_profilers_meta_list))
        logger('Process started on %s' %time.asctime())
        
        # for multiple directories and outputs let's put the meta
        # just before the output profile and the outputdir
        profile_name=''
        profiler=''
        reportdir=options.output
        IgProf_counter=options.IgProf_counter
           
        
        if options.infile!='': # we have a list of commands

            reportdir='%s_%s' %(meta,options.output) #Usually options.output is not used
            reportdir=clean_name(reportdir)          #Remove _
         
            profile_name=clean_name('%s_%s'%(meta,options.profile_name)) #Also options.profile_name is usually not used... should clean up...
                    
            # profiler is igprof: we need to disentangle the profiler and the counter
            
            if profiler_opt.find('.')!=-1 and profiler_opt.find('IgProf')!=-1:
                profiler_opt_split=profiler_opt.split('.')
                profiler=profiler_opt_split[0]
                IgProf_counter=profiler_opt_split[1:] #This way can handle IgProfMem.ANALYSE.MEM_TOT etc.
                if profile_name[-3:]!='.gz':
                    profile_name+='.gz'
            
            # Profiler is Timereport_Parser
            elif profiler_opt in STDOUTPROFILERS:
                # a first maquillage about the profilename:
                if profile_name[:-4]!='.log':
                    profile_name+='.log'
                profiler=profiler_opt
                        
            # profiler is not igprof
            else:
                profiler=profiler_opt
            
            if precedent_reuseprofile:
                profile_name=precedent_profile_name
            if reuseprofile:
                precedent_profile_name=profile_name 

                                
                        
        else: # we have a single command: easy job!
            profile_name=options.profile_name
            reportdir=options.output
            profiler=options.profiler

        
        
        # istantiate a Profile object    
        if precedent_profile_name!='':
            if os.path.exists(precedent_profile_name):
                logger('Reusing precedent profile: %s ...' %precedent_profile_name)
                if profile_name!=precedent_profile_name:
                    logger('Copying the old profile to the new name %s ...' %profile_name)
                    execute('cp %s %s' %(precedent_profile_name, profile_name))
                
        performance_profile=Profile(command,
                                    profiler,
                                    profile_name)   

        # make profile if needed
        if options.profile:                                         
            if reuseprofile:
                logger('Saving profile name to reuse it ...')
                precedent_profile_name=profile_name
            else:
                precedent_profile_name=''                                
            
            if not precedent_reuseprofile:
                logger('Creating profile for command %d using %s ...' \
                                                %(commands_counter,profiler))     
                exit_code=performance_profile.make_profile()
                print exit_code
                logger('The exit code was %s'%exit_code)
                exitCodeSum=exitCodeSum+exit_code #Add all exit codes into the global exitCodeSum in order to return it on cmsRelvareport.py exit.
                logger('The exit code sum is %s'%exitCodeSum)
            
        
        # make report if needed   
        if options.report:
            if exit_code!=0:
                logger('Halting report creation procedure: unexpected exit code %s from %s ...' \
                                            %(exit_code,profiler))
            else:   
                logger('Creating report for command %d using %s ...' \
                                                %(commands_counter,profiler))     
                                               
                # Write into the db instead of producing html if this is the case:
                if options.db:
                    exit_code=performance_profile.make_report(fill_db=True,
                                                    db_name=options.output,
                                                    metastring=db_metastring,
                                                    tmp_dir=options.pr_temp,
                                                    IgProf_option=IgProf_counter)
                    exitCodeSum=exitCodeSum+exit_code #this is to also check that the reporting works... a little more ambitious testing... could do without for release integration
                else:
                    exit_code=performance_profile.make_report(outdir=reportdir,
                                                    tmp_dir=options.pr_temp,
                                                    IgProf_option=IgProf_counter)
                    exitCodeSum=exitCodeSum+exit_code #this is to also check that the reporting works... a little more ambitious testing... could do without for release integration
                    
        commands_counter+=1                                                
        precedent_reuseprofile=reuseprofile
        if not precedent_reuseprofile:
            precedent_profile_name=''
        
        logger('Process ended on %s\n' %time.asctime())
    
    logger('Procedure finished on %s' %time.asctime())
    logger("Exit code sum is %s"%exitCodeSum)
    return exitCodeSum

###################################################################################################    
        
if __name__=="__main__":

    usage='\n'+\
          '----------------------------------------------------------------------------\n'+\
          ' RelValreport: a tool for automation of benchmarking and report generation. \n'+\
          '----------------------------------------------------------------------------\n\n'+\
          'relvalreport.py <options>\n'+\
          'relvalreport.py -i candles_150.txt -R -P -n 150.out -o 150_report\n'+\
          ' - Executes the candles contained in the file candles_150.txt, create\n'+\
          '   profiles, specified by -n, and reports, specified by -o.\n\n'+\
          'Candles file grammar:\n'+\
          'A candle is specified by the syntax:\n'+\
          'executable_name @@@ profiler_name @@@ meta\n'+\
          ' - executable_name: the name of the executable to benchmark.\n'+\
          ' - profiler_name: the name of the profiler to use. The available are: %s.\n' %str(PROFILERS)+\
          '   In case you want to use IgProf_mem or IgProf_perf, the counter (MEM_TOTAL,PERF_TICKS...)\n'+\
          '   must be added with a ".": IgProf_mem.MEM_TOTAL.\n'+\
          ' - meta: metastring that is used to change the name of the names specified in the command line\n'+\
          '   in case of batch execution.'+\
          'An example of candle file:\n\n'+\
          '  ># My list of candles:\n'+\
          '  >\n'+\
          '  >cmsDriver.py MU- -sSIM  -e10_20 @@@ IgProf_perf.PERF_TICKS @@@ QCD_sim_IgProfperf\n'+\
          '  >cmsDriver.py MU- -sRECO -e10_20 @@@ ValgrindFCE            @@@ QCD_reco_Valgrind\n'+\
          '  >cmsRun mycfg.cfg                @@@ IgProf_mem.MEM_TOTAL   @@@ Mycfg\n'
             
             

    parser = optparse.OptionParser(usage)

    parser.add_option('-p', '--profiler',
                      help='Profilers are: %s' %str(PROFILERS) ,
                      default='',
                      dest='profiler')
                                            
    parser.add_option('-c', '--command',
                      help='Command to profile. If specified the infile is ignored.' ,
                      default='',
                      dest='command') 
                      
    parser.add_option('-t',
                      help='The temp directory to store the PR service files. Default is PR_TEMP Ignored if PR is not used.',
                      default='',
                      dest='pr_temp')
    
    #Flags                      
                      
    parser.add_option('--db',
                      help='EXPERIMENTAL: Write results on the db.',
                      action='store_true',
                      default=False,
                      dest='db')               

    parser.add_option('-R','--Report',
                      help='Create a static html report. If db switch is on this is ignored.',
                      action='store_true',
                      default=False,
                      dest='report')                      
    
    parser.add_option('-P','--Profile',
                      help='Create a profile for the selected profiler.',
                      action='store_true',
                      default=False,
                      dest='profile')                        
    
    # Output location for profile and report                      
    
    parser.add_option('-n', '--profile_name',
                      help='Profile name' ,
                      default='',
                      dest='profile_name') 
                                                                                                                            
    parser.add_option('-o', '--output',
                      help='Outdir for the html report or db filename.' ,
                      default='',
                      dest='output')
                      
    #Batch functionality                       
                                               
    parser.add_option('-i', '--infile',
                      help='Name of the ASCII file containing the commands to profile.' ,
                      default='',
                      dest='infile')    
    
    # ig prof options

    parser.add_option('-y', 
                      help='Specify the IgProf counter or the CMSSW. '+\
                           'If a profiler different from '+\
                           'IgProf is selected this is ignored.' ,
                      default=None,
                      dest='IgProf_counter')                        
                      
    parser.add_option('--executable',
                      help='Specify executable to monitor if different from cmsRun. '+\
                           'Only valid for IgProf.',
                      default='',
                      dest='executable')                               
              
    # Debug options
    parser.add_option('--noexec',
                      help='Do not exec commands, just display them!',
                      action='store_true',
                      default=False,
                      dest='noexec')   
                                        
    (options,args) = parser.parse_args()
    
    # FAULT CONTROLS
    if options.infile=='' and options.command=='' and not (options.report and not options.profile):
        raise('Specify at least one command to profile!')
    if options.profile_name=='' and options.infile=='':
        raise('Specify a profile name!')
    if not options.db and options.output=='' and options.infile=='':
        raise('Specify a db name or an output dir for the static report!')
    
    if not options.profile:
        if not os.path.exists(options.profile_name) and options.infile=='':
            raise('Profile %s does not exist!' %options.profile_name)
        logger("WARNING: No profile will be generated. An existing one will be processed!")
    
    if options.command!='' and options.infile!='':
        raise('-c and -i options cannot coexist!')    
    
    if options.profiler=='Memcheck_Valgrind' and not os.path.exists(VMPARSER):
        raise('Couldn\'t find Valgrind Memcheck Parser Script! Please install it from Utilities/ReleaseScripts.')
            
    if options.executable!='':
        globals()['EXECUTABLE']=options.executable
    
    if options.noexec:
        globals()['EXEC']=False
        
    logger('Procedure started on %s' %time.asctime())                               
    
    if options.infile == '':
        logger('Script options:')
        for key,val in options.__dict__.items():
            if val!='':
                logger ('\t\t|- %s = %s' %(key, str(val)))
                logger ('\t\t|')
    exit=principal(options)
    logger("Exit code received from principal is: %s"%exit)
    #Mind you! exit codes in Linux are all 0 if they are even! We can easily make the code 1
    if exit: #This is different than 0 only if there have been at least one non-zero exit(return) code in the cmsRelvalreport.py
        exit=1
    sys.exit(exit)
                
