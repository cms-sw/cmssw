from __future__ import print_function
from __future__ import absolute_import
################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################

from builtins import range
from copy import deepcopy
from os import chdir,getcwd,makedirs
from os.path import abspath,exists,join, basename
from re import sub,search
from re import compile as recompile
from sys import exit,stderr,version_info
from threading import Thread,activeCount
from time import sleep
if version_info[0]==2:
  from urllib2  import Request,build_opener,urlopen
else:
  from urllib.request  import Request,build_opener,urlopen

import sys
argv=sys.argv
import ROOT
sys.argv=argv

ROOT.gROOT.SetBatch(True)

from .authentication import X509CertOpen
from .dirstructure import Comparison,Directory,tcanvas_print_processes
from .utils import Chi2,KS,BinToBin,Statistical_Tests,literal2root

#-------------------------------------------------------------------------------  

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class DQM_DB_Communication(Error):
    """Exception occurs in case of problems of communication with the server.
    """
    def __init__(self,msg):
        self.msg = msg

class InvalidNumberOfArguments(Error):

    def __init__(self,msg):
        self.msg = msg

#-----------------------------------------------------------------------------    

class DQMcommunicator(object):

    """Communicate with the DQM Document server"""

    #-----------------------------------------------------------------------------

    base_dir='/data/json/archive/'

    def __init__(self,
                 server,
                 is_private=False,
                 ident="DQMToJson/1.0 python/%d.%d.%d" % version_info[:3]):
        self.ident = ident
        self.server = server
        self.is_private = is_private
        self.DQMpwd=DQMcommunicator.base_dir
        self.prevDQMpwd=self.DQMpwd
        self.opener=None
        if not self.is_private:
            self.opener=build_opener(X509CertOpen())
    #-----------------------------------------------------------------------------

    def open_url(self,url):
        url=url.replace(' ','%20')
        datareq = Request(url)
        datareq.add_header('User-agent', self.ident)    
        url_obj=0
        if not self.is_private:
            url_obj=self.opener.open(datareq)   
            #url_obj=build_opener(X509CertOpen()).open(datareq) 
        else:
            url_obj=urlopen(datareq)

        return url_obj

    #-----------------------------------------------------------------------------

    def get_data(self, full_url):
        #print "getting data from %s" %full_url
        data = self.open_url(full_url).read()

        data = sub("-inf", '0', data)
        data = sub("\s+inf", '0', data)
        data = sub("\s+nan", '0', data)
        data = sub('""(CMSSW.*?)""', '"\\1"', data)

        return data

    #-----------------------------------------------------------------------------

    def ls_url(self, url):
        url=url.replace(" ","%20")
        url=self.server+url
        #print "listing "+url
        form_folder={}
        raw_folder=None
        try:
            raw_folder=eval(self.get_data(url))
        except:
            print("Retrying..")
            for ntrials in range(5):
                try:
                    if ntrials!=0:
                        sleep(2)
                    #raw_folder=loads(self.get_data(url))
                    raw_folder=eval(self.get_data(url))
                    break
                except:
                    print("Could not fetch %s. Retrying" %url)

        #raw_folder=loads(self.get_data(url))
        for content_dict in raw_folder["contents"]:      
            if "subdir" in content_dict:
                form_folder[content_dict["subdir"]]={"type":'dir'}
            elif "obj" in content_dict:
                properties=content_dict["properties"]
                obj_name=content_dict["obj"]
                obj_type=properties["type"]
                obj_kind=properties["kind"]
                obj_as_string=''
                if "rootobj" in content_dict:
                    obj_as_string=content_dict["rootobj"]
                form_folder[obj_name]={'type':obj_type,'obj_as_string':obj_as_string,"kind":obj_kind}
        #for k,v in form_folder.items():
            #print "* %s --> %s" %(k,v["type"])

        return form_folder        

    #-----------------------------------------------------------------------------

    def ls(self, url='', fetch_root=False):
        if len(url)==0:
            url=join(self.DQMpwd,url)

        form_folder={}   

        if fetch_root:
            url='%s?rootcontent=1'%url
        form_folder=self.ls_url(url)

        return form_folder

    #-----------------------------------------------------------------------------

    def cd(self, *args):
        len_args=len(args)
        full_url=""
        if len_args!=1 and len_args!=3:
            raise InvalidNumberOfArguments
        if len_args==3:
            dataset, run, folder = args    
            full_url='%s/data/json/archive/%s/%s/%s' % (self.server, dataset, run, folder)
        if len_args==1:
            folder=args[0]
            if folder==self.DQMpwd:
                full_url=self.DQMpwd
            elif folder=="..":
                full_url=self.DQMpwd[:self.DQMpwd.rfind("/")]
            elif folder=="-":
                full_url=self.oldDQMpwd
            elif folder=="":
                full_url=DQMcommunicator.base_dir
            else:
                full_url=self.DQMpwd+"/"+folder

        full_url=full_url.replace(' ','%20')
        #print "cd: "+full_url

        self.oldDQMpwd=self.DQMpwd
        self.DQMpwd=full_url   
        #print "In %s" %self.DQMpwd

    #-----------------------------------------------------------------------------

    def get_samples(self, samples_string="*"):
        """
        A sample contains, among the other things, a data type, a dataset name 
        and a run.
        """
        full_url='%s/data/json/samples?match=%s' % (self.server, samples_string)
        samples_dict=eval(self.get_data(full_url))
        return samples_dict["samples"]

    #-----------------------------------------------------------------------------

    def get_datasets_list(self, dataset_string=""):
        samples_list=self.get_samples(dataset_string)    
        datasets_list=[]
        for sample in samples_list:
            temp_datasets_list =  map(lambda item:item["dataset"] ,sample['items'])
            for temp_dataset in temp_datasets_list:
                if not temp_dataset in datasets_list:
                    datasets_list.append(temp_dataset)
        return datasets_list

    #-----------------------------------------------------------------------------

    def get_RelVal_CMSSW_versions(self,query):
        """Get the available cmssw versions for the relvals.
        """
        relvals_list=self.get_datasets_list(query)
        # The samples are of the form /RelValTHISISMYFAVOURITECHANNEL/CMSSW_VERSION/GEN-SIM-WHATEVER-RECO
        cmssw_versions_with_duplicates=map (lambda x: x.split("/")[2],relvals_list)
        return list(set(cmssw_versions_with_duplicates))

    #-----------------------------------------------------------------------------    

    def get_runs_list(self, dataset_string):
        slash="/"
        while(dataset_string.endswith(slash) or dataset_string.beginswith(slash)):
            dataset_string=dataset_string.strip("/")
        samples_list=self.get_samples(dataset_string)
        runlist=[]
        # Get all the runs in all the items which are in every sample
        map( lambda sample: map (lambda item: runlist.append(item['run']), sample['items']), samples_list)
        return runlist

    #-----------------------------------------------------------------------------  

    def get_dataset_runs(self,dataset_string):
        dataset_runs={}
        for dataset in self.get_datasets_list(dataset_string):
            dataset_runs[dataset]=self.get_runs_list(dataset)
        return dataset_runs

    #-----------------------------------------------------------------------------  

    def get_common_runs(self,dataset_string1,dataset_string2):
        set1=set(self.get_runs_list(dataset_string1))
        set2=set(self.get_runs_list(dataset_string2))
        set1.intersection_update(set2)
        return list (set2)

    #-----------------------------------------------------------------------------  

    def get_root_objects_list(self, url=""):
        if len(url)==0:
            url=self.DQMpwd
        else:
            url="/"+url    
        url = url.replace(" ","%20")
        objects=[]
        for name,description in self.ls(url,True).items():     
            if "dir" not in description["type"]  and "ROOT" in description["kind"]:
                objects.append(literal2root(description["obj_as_string"],description["type"]))
        return objects

    #-----------------------------------------------------------------------------  

    def get_root_objects(self, url=""):
        if len(url)==0:
            url=self.DQMpwd
        else:
            url=self.server+"/"+url    
        url = url.replace(" ","%20")
        objects={}
        for name,description in self.ls(url,True).items():     
            if "dir" not in description["type"] and "ROOT" in description["kind"]:
                objects[name]=literal2root(description["obj_as_string"],description["type"])
        return objects

     #-------------------------------------------------------------------------------

    def get_root_objects_list_recursive(self, url=""):
        null_url = (len(url)==0)    
        if len(url)==0:
            url=self.DQMpwd
        else:
            url="/"+url    
        url = url.replace(" ","%20")      
        if not null_url: 
            self.cd(url)
        objects=[]
        for name,description in self.ls("",True).items():     
            if "dir" in description["type"]:
                objects+=self.get_root_objects_list_recursive(name)
                self.cd("..")
            elif  "ROOT" in description["kind"]:
                objects.append(literal2root(description["obj_as_string"],description["type"]))
        if not null_url: 
            self.cd("..")
        return objects

     #-------------------------------------------------------------------------------

    def get_root_objects_names_list_recursive(self, url="",present_url=""):
        null_url = (len(url)==0)
        if (not null_url):
            if len(present_url)==0:
                present_url=url
            else:
                present_url+="_%s"%url
        if len(url)==0:
            url=self.DQMpwd
        else:
            url="/"+url    
        url = url.replace(" ","%20")
        if not null_url:
            self.cd(url)
        objects_names=[]
        for name,description in self.ls("",False).items():     
            if "dir" in description["type"]:        
                objects_names+=self.get_root_objects_names_list_recursive(name,present_url)
                self.cd("..")
            elif  "ROOT" in description["kind"]:
                objects_names.append("%s_%s"%(present_url,name))
        if not null_url: 
            self.cd("..")
        return objects_names

     #-------------------------------------------------------------------------------

    def get_root_objects_recursive(self, url="",present_url=""):
        null_url = (len(url)==0)
        if (not null_url):
            if len(present_url)==0:
                present_url=url
            else:
                present_url+="_%s"%url
        if len(url)==0:
            url=self.DQMpwd
        else:
            url="/"+url    
        url = url.replace(" ","%20")
        #if not null_url:
        self.cd(url)
        objects={}
        for name,description in self.ls("",True).items():     
            if "dir" in description["type"]:
                objects.update(self.get_root_objects_recursive(name,present_url))
                self.cd("..")
            elif  "ROOT" in description["kind"]:
                objects["%s_%s"%(present_url,name)]=literal2root(description["obj_as_string"],description["type"])
        #if not null_url:
        self.cd("..")
        return objects

#-------------------------------------------------------------------------------

class DirID(object):
    """Structure used to identify a directory in the walked tree,
    It carries the name and depth information.
    """
    def __init__(self,name,depth,mother=""):
        self.name=name
        self.compname=recompile(name)
        self.mother=mother
        self.depth=depth
    def __eq__(self,dirid):
        depth2=dirid.depth
        compname2=dirid.compname
        name2=dirid.name
        is_equal = False
        #if self.name in name2 or name2 in self.name:
        if search(self.compname,name2)!=None or search(compname2,self.name)!=None:
            is_equal = self.depth*depth2 <0 or self.depth==depth2
        if len(self.mother)*len(dirid.mother)>0:
            is_equal = is_equal and self.mother==dirid.mother
        return is_equal

    def __repr__(self):
        return "Directory %s at level %s" %(self.name,self.depth)

#-------------------------------------------------------------------------------
class DirFetcher(Thread):
    """ Fetch the content of the single "directory" in the dqm.
    """
    def __init__ (self,comm,directory):
        Thread.__init__(self)
        self.comm = comm
        self.directory = directory
        self.contents=None    
    def run(self):
        self.contents = self.comm.ls(self.directory,True)

#-------------------------------------------------------------------------------

class DirWalkerDB(Thread):
    """An interface to the DQM document db. It is threaded to compensate the 
    latency introduced by the finite response time of the server.
    """
    def __init__ (self,comm1,comm2,base1,base2,directory,depth=0,do_pngs=True,stat_test="KS",test_threshold=.5,black_list=[]):
        Thread.__init__(self)
        self.comm1 = deepcopy(comm1)
        self.comm2 = deepcopy(comm2)
        self.base1,self.base2 = base1,base2
        self.directory = directory
        self.depth=depth
        self.do_pngs=do_pngs
        self.test_threshold=test_threshold
        self.stat_test=stat_test
        self.black_list=black_list
        # name of the thread
        self.name+="_%s" %directory.name

    def run(self):

        this_dir=DirID(self.directory.name,self.depth)
        if this_dir in self.black_list: 
            print("Skipping %s since blacklisted!" %this_dir)
            return 0 

        self.depth+=1

        the_test=Statistical_Tests[self.stat_test](self.test_threshold)
        #print "Test %s with threshold %s" %(self.stat_test,self.test_threshold)

        directory1=self.base1+"/"+self.directory.mother_dir+"/"+self.directory.name
        directory2=self.base2+"/"+self.directory.mother_dir+"/"+self.directory.name

        fetchers =(DirFetcher(self.comm1,directory1),DirFetcher(self.comm2,directory2))
        for fetcher in fetchers:
            fetcher.start()
        for fetcher in fetchers:  
            fetcher.join()

        contents1 = fetchers[0].contents
        contents2 = fetchers[1].contents
        set1= set(contents1.keys())
        set2= set(contents2.keys())  

        walkers=[]
        self_directory_directories=self.directory.subdirs
        self_directory_comparisons=self.directory.comparisons
        contents_names=list(set1.intersection(set2))

        for name in contents_names:
            content = contents1[name]
            if "dir" in content["type"]:
                #if this_dir not in DirWalker.white_list:continue              
                subdir=Directory(name,join(self.directory.mother_dir,self.directory.name))        
                dirwalker=DirWalkerDB(self.comm1,self.comm2,self.base1,self.base2,subdir,self.depth,
                                      self.do_pngs,self.stat_test,self.test_threshold,self.black_list)
                dirwalker.start()
                walkers.append(dirwalker)
                n_threads=activeCount()
                if n_threads>5:
                    #print >> stderr, "Threads that are running: %s. Joining them." %(n_threads)    
                    dirwalker.join()
            elif content["kind"]=="ROOT":
#	print directory1,name
                comparison=Comparison(name,
                                      join(self.directory.mother_dir,self.directory.name),
                                      literal2root(content["obj_as_string"],content["type"]),
                                      literal2root(contents2[name]["obj_as_string"],content["type"]),
                                      deepcopy(the_test),
                                      do_pngs=self.do_pngs)
                self_directory_comparisons.append(comparison)


        for walker in walkers:
            walker.join()
            walker_directory=walker.directory
            if not walker_directory.is_empty():
                self_directory_directories.append(walker_directory)

#-------------------------------------------------------------------------------

class DQMRootFile(object):
    """ Class acting as interface between the user and the harvested DQMRootFile.  
    It skips the directories created by the DQM infrastructure so to provide an
    interface as similar as possible to a real direcory structure and to the 
    directory structure provided by the db interface.
    """
    def __init__(self,rootfilename):
        dqmdatadir="DQMData"
        self.rootfile=ROOT.TFile(rootfilename)
        self.rootfilepwd=self.rootfile.GetDirectory(dqmdatadir)
        self.rootfileprevpwd=self.rootfile.GetDirectory(dqmdatadir)
        if self.rootfilepwd == None:
            print("Directory %s does not exist: skipping. Is this a custom rootfile?" %dqmdatadir)
            self.rootfilepwd=self.rootfile
            self.rootfileprevpwd=self.rootfile

    def __is_null(self,directory,name):
        is_null = not directory
        if is_null:
            print("Directory %s does not exist!" %name, file=stderr)
        return is_null

    def ls(self,directory_name=""):
        contents={}
        directory=None
        if len(directory_name)==0:
            directory=self.rootfilepwd      

        directory=self.rootfilepwd.GetDirectory(directory_name)    
        if self.__is_null(directory,directory_name):
            return contents

        for key in directory.GetListOfKeys():
            contents[key.GetName()]=key.GetClassName()
        return contents

    def cd(self,directory_name):
        """Change the current TDirectoryFile. The familiar "-" and ".." directories 
        can be accessed as well.
        """
        if directory_name=="-":
            tmp=self.rootfilepwd
            self.rootfilepwd=self.rootfileprevpwd
            self.rootfileprevpwd=tmp
        if directory_name=="..":
            #print "Setting prevpwd"
            self.rootfileprevpwd=self.rootfilepwd
            #print "The mom"
            mom=self.rootfilepwd.GetMotherDir()
            #print "In directory +%s+" %self.rootfilepwd
            #print "Deleting the TFileDir"
            if "Run " not in self.rootfilepwd.GetName():
                self.rootfilepwd.Delete()
            #print "Setting pwd to mom"
            self.rootfilepwd=mom
        else:
            new_directory=self.rootfilepwd.GetDirectory(directory_name)
            if not self.__is_null(new_directory,directory_name):
                self.rootfileprevpwd=self.rootfilepwd
                self.rootfilepwd=new_directory

    def getObj(self,objname):
        """Get a TObject from the rootfile.
        """
        obj=self.rootfilepwd.Get(objname)
        if not self.__is_null(obj,objname):
            return obj

#-------------------------------------------------------------------------------

class DirWalkerFile(object):
    def __init__(self, name, topdirname,rootfilename1, rootfilename2, run=-1, black_list=[], stat_test="KS", test_threshold=.5,draw_success=True,do_pngs=False, black_list_histos=[]):
        self.name=name
        self.dqmrootfile1=DQMRootFile(abspath(rootfilename1))
        self.dqmrootfile2=DQMRootFile(abspath(rootfilename2))
        self.run=run
        self.stat_test=Statistical_Tests[stat_test](test_threshold)
        self.workdir=getcwd()
        self.black_list=black_list
        self.directory=Directory(topdirname)
        #print "DIRWALKERFILE %s %s" %(draw_success,do_pngs)
        self.directory.draw_success=draw_success
        self.directory.do_pngs=do_pngs
        self.black_list_histos = black_list_histos
        self.different_histograms = {}
        self.filename1 = basename(rootfilename2)
        self.filename2 = basename(rootfilename1)

    def __del__(self):
        chdir(self.workdir)

    def cd(self,directory_name, on_disk=False, regexp=False,):
        if regexp == True:
            if len(directory_name)!=0:
                if on_disk:
                    if not exists(directory_name):
                        makedirs(directory_name)
                        chdir(directory_name)  
                tmp = self.dqmrootfile2.ls().keys()
                for elem in tmp:
                    if "Run" in elem:
                        next_dir = elem
                self.dqmrootfile2.cd(next_dir)
                tmp = self.dqmrootfile1.ls().keys()
                for elem in tmp:
                    if "Run" in elem:
                        next_dir = elem
                self.dqmrootfile1.cd(next_dir)
        else:
            if len(directory_name)!=0:
                if on_disk:
                    if not exists(directory_name):
                        makedirs(directory_name)
                        chdir(directory_name)
                self.dqmrootfile2.cd(directory_name)
                self.dqmrootfile1.cd(directory_name)

    def ls(self,directory_name=""):
        """Return common objects to the 2 files.
        """
        contents1=self.dqmrootfile1.ls(directory_name)
        contents2=self.dqmrootfile2.ls(directory_name)
        #print "cont1: %s"%(contents1)
        #print "cont2: %s"%(contents2)
        contents={}
        self.different_histograms['file1']= {}
        self.different_histograms['file2']= {}
        keys = [key for key in contents2.keys() if key in contents1] #set of all possible contents from both files
        #print " ## keys: %s" %(keys)
        for key in keys:  #iterate on all unique keys
            if contents1[key]!=contents2[key]:
                diff_file1 = set(contents1.keys()) - set(contents2.keys()) #set of contents that file1 is missing
                diff_file2 = set(contents2.keys()) - set(contents1.keys()) #--'-- that file2 is missing
                for key1 in diff_file1:
                    obj_type = contents1[key1]
                    if obj_type == "TDirectoryFile":
                        self.different_histograms['file1'][key1] = contents1[key1] #if direcory
                        #print "\n Missing inside a dir: ", self.ls(key1)
                        #contents[key] = contents1[key1]
                    if obj_type[:2]!="TH" and obj_type[:3]!="TPr" : #if histogram
                        continue
                    self.different_histograms['file1'][key1] = contents1[key1]
                for key1 in diff_file2:
                    obj_type = contents2[key1]
                    if obj_type == "TDirectoryFile":
                        self.different_histograms['file2'][key1] = contents2[key1] #if direcory
                        #print "\n Missing inside a dir: ", self.ls(key1)
                        #contents[key] = contents2[key1]
                    if obj_type[:2]!="TH" and obj_type[:3]!="TPr" : #if histogram
                        continue
                    self.different_histograms['file2'][key1] = contents2[key1]
            contents[key]=contents1[key]
        return contents

    def getObjs(self,name):
        h1=self.dqmrootfile1.getObj(name)
        h2=self.dqmrootfile2.getObj(name)
        return h1,h2

    def __fill_single_dir(self,dir_name,directory,mother_name="",depth=0):
        #print "MOTHER NAME  = +%s+" %mother_name
     #print "About to study %s (in dir %s)" %(dir_name,getcwd())

        # see if in black_list
        this_dir=DirID(dir_name,depth)
        #print "  ## this_dir: %s"%(this_dir)
        if this_dir in self.black_list: 
            #print "Directory %s skipped because black-listed" %dir_name
            return 0        

        depth+=1

        self.cd(dir_name)
        #if dir_name == 'HLTJETMET':
        #    print self.ls()

        #print "Test %s with thre %s" %(self.stat_test.name, self.stat_test.threshold)

        contents=self.ls()
        if depth==1:
            n_top_contents=len(contents)

        #print contents
        cont_counter=1
        comparisons=[]
        for name,obj_type in contents.items():
            if obj_type=="TDirectoryFile":        
                #We have a dir, launch recursion!
                #Some feedback on the progress
                if depth==1:
                    print("Studying directory %s, %s/%s" %(name,cont_counter,n_top_contents))
                    cont_counter+=1          

                #print "Studying directory",name
                # ok recursion on!
                subdir=Directory(name)
                subdir.draw_success=directory.draw_success
                subdir.do_pngs=directory.do_pngs
                self.__fill_single_dir(name,subdir,join(mother_name,dir_name),depth)
                if not subdir.is_empty():
                    if depth==1:
                        print(" ->Appending %s..." %name, end=' ')
                    directory.subdirs.append(subdir)
                    if depth==1:
                        print("Appended.")
            else:
                # We have probably an histo. Let's make the plot and the png.        
                if obj_type[:2]!="TH" and obj_type[:3]!="TPr" :
                    continue
                h1,h2=self.getObjs(name)
                #print "COMPARISON : +%s+%s+" %(mother_name,dir_name)
                path = join(mother_name,dir_name,name)
                if path in self.black_list_histos:
                    print("  Skipping %s" %(path))
                    directory.comparisons.append(Comparison(name,
                                        join(mother_name,dir_name),
                                        h1,h2,
                                        deepcopy(self.stat_test),
                                        draw_success=directory.draw_success,
                                        do_pngs=directory.do_pngs, skip=True))
                else:
                    directory.comparisons.append(Comparison(name,
                                          join(mother_name,dir_name),
                                          h1,h2,
                                          deepcopy(self.stat_test),
                                          draw_success=directory.draw_success,
                                          do_pngs=directory.do_pngs, skip=False))
                    directory.filename1 = self.filename1
                    directory.filename2 = self.filename2
                    directory.different_histograms['file1'] = self.different_histograms['file1']
                    directory.different_histograms['file2'] = self.different_histograms['file2']

        self.cd("..")

    def walk(self):
        # Build the top dir in the rootfile first
        rundir=""
        if self.run<0:
            # change dir in the first one...
            #print  self.ls().keys()
            first_run_dir = ""
            try:
                first_run_dir = list(filter(lambda k: "Run " in k, self.ls().keys()))[0]
            except:
                print("\nRundir not there: Is this a generic rootfile?\n")
            rundir=first_run_dir
            try:
                self.run= int(rundir.split(" ")[1])
            except:
                print("Setting run number to 0")
                self.run= 0
        else:
            rundir="Run %s"%self.run

        try:
            self.cd(rundir, False, True) #True -> for checking the Rundir in case of different runs
        except:
            print("\nRundir not there: Is this a generic rootfile?\n")

        # Let's rock!
        self.__fill_single_dir(self.directory.name,self.directory)
        print("Finished")
        n_left_threads=len(tcanvas_print_processes)
        if n_left_threads>0:
            print("Waiting for %s threads to finish..." %n_left_threads)
            for p in tcanvas_print_processes:
                p.join()  

#-------------------------------------------------------------------------------

class DirWalkerFile_thread_wrapper(Thread):
    def __init__(self, walker):
        Thread.__init__(self)
        self.walker=walker
    def run(self):
        self.walker.walk()

#-------------------------------------------------------------------------------

def string2blacklist(black_list_str):
    black_list=[]
    # replace the + with " ":
    black_list_str=black_list_str.replace("__"," ")
    if len(black_list_str)>0:
        for ele in black_list_str.split(","):
            dirname,level=ele.split("@")
            level=int(level)
            dirid=None
            if "/" not in dirname:
                dirid=DirID(dirname,level)
            else:
                mother,daughter=dirname.split("/")
                dirid=DirID(daughter,level,mother)
            if not dirid in black_list:
                black_list.append(dirid)

    return black_list

#-------------------------------------------------------------------------------

