#!/usr/bin/env python
# set of tools to create and submit validation webpages 
# author: Colin

import shutil, sys, os, re, glob, string 

from optparse import OptionParser


class webpage:

    # sets variables related to the creation of the local webpage. 
    def __init__(self):
        self.parser_ = OptionParser()
        self.parser_.add_option("-c", "--comments", dest="comments",
                               help="name of another release",
                               default=None)
        self.parser_.add_option("-f", "--force", dest="force",
                                action="store_true",
                                help="overwrites the local benchmark page.",
                                default=False)
        self.rootFile_ = 'benchmark.root'
        self.dirPlots_ = './'
        self.templates_ = '../Tools/templates'
        self.date_ =  os.popen( 'date' ).read()
        self.benchmarkName_ = os.path.basename( os.getcwd() ) 

    def parseArgs(self):
        (self.options_, self.args_) = self.parser_.parse_args()

    #create output dir, if it does not exist
    def setOutputDir(self, outputDir ):

        self.outputDir_ = outputDir
        if os.path.isdir( outputDir ):
            print outputDir, "already exists"
            if self.options_.force == False:
                print 'sorry... run the script with the -h option for more information' 
                sys.exit(3)
            else:
                print 'overwriting local output directory...'
        else:
            os.makedirs( outputDir )

    # read the caption file and produce the html
    # code for the Plots section
    def readCaptions(self, captions):
        imgTemplate = '<IMG src="%s" width="500" align="left" border="0"><br clear="ALL">'
        images = ''
        captionsContents = open( captions )
        for line in captionsContents:
            try:
                (picfile, caption) = self.readCaption( line )
                img = imgTemplate % os.path.basename(picfile)
                images = "%s<h3>%s:</h3>\n%s\n" % (images, caption, img)
                # what to do if the file's not there? 
                # : print a warning
                shutil.copy(picfile, self.outputDir_) 
            except Exception:
                print 'File %s does not exist. Did you generate the comparison plots?' % picfile
                print 'Aborting the script.\n'
                print 'Solution 1: run without the -m "" option, to run the compare.C macro'
                print 'Solution 2: run with the -m "myMacro.C" option, to run another macro'
                sys.exit(1)
                raise
        return images

    # decode a caption line, and return
    # the caption, and the filename
    # COULD HANDLE SEVERAL FILES
    def readCaption( self, line ):
        
        if( re.compile('^\s*$').match(line) ):
            raise Exception
        
        p = re.compile('^\s*(\S+)\s*\"(.*)\"');
        m = p.match(line)
        if m:
            pic = m.group(1)
            caption = m.group(2)
            return (pic, caption)
        else:
            print 'bad caption format: "%s"' % line
            raise Exception

class website:
    def __init__(self):
        self.website_ = '/afs/cern.ch/cms/Physics/particleflow/Validation/cms-project-pfvalidation/Releases'
        self.url_ = 'http://cern.ch/pfvalidation/Releases'
    
    def __str__(self):
        return self.website_


    def writeAccess(self):
        if( os.access(self.website_, os.W_OK)==False ):
            print 'cannot write to the website. Please ask Colin to give you access.'
            sys.exit(1)

    def listBenchmarks(self, pattern, afs=False, url=False):
        for bench in glob.glob(self.website_ + '/' + pattern):
            # strip off the root path of the website
            p = re.compile('^%s/(\S+)$' % self.website_);
            m = p.match( bench )
            if m:
                (release, benchName, extension) = decodePath( m.group(1) )
                if release == None:
                    # this is a comparison
                    continue
                print
                bench = benchmark(m.group(1))
                print bcolors.OKGREEN + m.group(1) + bcolors.ENDC
                if afs or url:                        
                    if afs: print '  ',bench.benchmarkOnWebSite( self )
                    if url: print '  ',bench.benchmarkUrl( self )
        
    def listComparisons(self, benchmark):

        comparisons = []
        find = 'find %s -type d' % self.website_
        for dir in os.popen( find ):
            dir = dir.rstrip()
            #print dir 
            comp = '%s/\S+/\S+/%s' % (self.website_,
                                      benchmark.fullName() )
            #print "** ", comp
            p = re.compile(comp)
            m = p.match(dir)
            if m:
                comparisons.append(dir)
                #print ' ',dir
        return comparisons

class benchmark:
 
    # arg can be either the full name of a benchmark, or 
    # an extension, in which case, the release and benchmark name are guessed 
    # from the environment variables. 
    def __init__(self, arg=None):

        release = None
        benchName = None
        extension = None
        self.indexHtml_ = 'index.html'
        
        if arg != None:
            (release, benchName, extension) = decodePath( arg )

        if release == None:
            # we get there if:
            # - arg == None
            # - the decoding of arg as a full benchmark name has failed. 
            self.release_ = os.environ['CMSSW_VERSION']
        
            # benchmark directory, as the current working directory
            self.benchmark_ = os.path.basename( os.getcwd() )

            # underscore are not allowed in extension names 
            if arg!=None and arg.count('_'):
                print 'sorry, as said many times, underscores are not allowed in the extension ;P'
                sys.exit(5)
            
            extension = arg
        else:
            self.release_ = release
            self.benchmark_ = benchName

        self.benchmarkWithExt_ = self.benchmark_
        if( extension != None ):
            self.benchmarkWithExt_ = '%s_%s' % (self.benchmark_, extension)

        
    def __str__(self):
        return self.release_ + '/' + self.benchmarkWithExt_

    def fullName(self):
        return self.release_ + '/' + self.benchmarkWithExt_
  
    def releaseOnWebSite( self, website ):
        return '%s/%s'  % ( website, self.release_ )

    def benchmarkOnWebSite( self, website ):
        return  '%s/%s' % ( website, self.fullName() )
#        return '%s/%s'  % ( self.releaseOnWebSite(website), 
#                            self.benchmarkWithExt_ )

    def rootFileOnWebSite( self, website ):
        return '%s/%s'  % ( self.benchmarkOnWebSite(website), 
                            'benchmark.root' )
    
    def releaseUrl( self, website ):
        return '%s/%s'  % ( website.url_, self.release_ )

    
    def benchmarkUrl( self, website ):
        return '%s/%s'  % ( self.releaseUrl( website ), 
                            self.benchmarkWithExt_ )
    
    def makeRelease( self, website):
        rel = self.releaseOnWebSite(website)
        if( os.path.isdir( rel )==False):
            print 'creating release %s' % self.release_
            print rel
            os.mkdir( rel )

    def exists( self, website): 
        if( os.path.isdir( self.benchmarkOnWebSite(website) )):
            print 'benchmark %s exists for release %s' % (self.benchmarkWithExt_, self.release_)
            return True
        else:
            print 'benchmark %s does not exist for release %s' % (self.benchmarkWithExt_, self.release_)
            return False

    def addLinkToComparison( self, website, comparison ):
        url = comparison.comparisonUrl( website )
        index = self.benchmarkOnWebSite(website) + '/' + self.indexHtml_
        indexTmp = self.benchmarkOnWebSite(website) + '/index.tmp.html' 
        indexFile = open( index)
        indexFileTmp = open( indexTmp, 'w')
        for line in indexFile:
            p = re.compile('<h2>Comparisons:</h2>')
            m = p.match(line)
            indexFileTmp.write(line)
            if m:
                link = '<A href="%s">%s</A><BR>\n' % (url, url)
                indexFileTmp.write(link)
        shutil.move( indexTmp, index)

class comparison:

    def __init__(self, benchmark, comparisonPath):
        self.benchmark_ = benchmark
        self.path_ = comparisonPath

    def comparisonOnWebSite(self, website):
        return  '%s/%s' % ( self.benchmark_.benchmarkOnWebSite(website),
                            self.path_ )

    def comparisonUrl(self, website):
        return '%s/%s'  % ( self.benchmark_.benchmarkUrl(website),
                            self.path_ )

    def submit(self, website, force=False):
        print 'Submitting comparison:'
        print '  from: ',self.path_
        print '  to  : ',self.comparisonOnWebSite(website)

        if( os.path.isdir(self.comparisonOnWebSite(website) ) ):
            print 'comparison already exists'
            if force:
                print 'overwriting comparison on the website...'
            else:
                print 'submission cancelled. run with -h for a solution.'
                return False
        else:
            print 'comparison directory does not yet exist. creating it.'
            mkdir = 'mkdir -p ' + self.comparisonOnWebSite(website)
            print mkdir    
            if os.system( mkdir ):
                print 'problem creating the output directory on the website. Aborting.'
                return False
        cp = 'cp %s %s' % (self.path_ + '/*',
                           self.comparisonOnWebSite(website))
        if os.system(cp):
            print 'problem copying the files to the website aborting'
            return False

        print 'access your comparison here:'
        print '  ', self.comparisonUrl(website)



# pathname in the form: CMSSW_3_1_0_pre7/TauBenchmarkGeneric_Extension
def decodePath( path ):
    p = re.compile('^(\S+)/([^\s_]+)_(\S+)');
    m = p.match(path)
    if m:
        release = m.group(1)
        benchmarkname = m.group(2)
        extension = m.group(3)
        return (release, benchmarkname, extension )
    else:
        return (None, None, None)

#test that a given file is a file with the correct extenstion, e.g. .root
def testFileType( file, ext ):

     if file == "None":
          return
     
     if os.path.isfile( file ) == False:
          print '%s is not a file' % file
          sys.exit(2)
     
     (fileroot, fileext) = os.path.splitext( file )
     if fileext != ext:
          print '%s does not end with %s' % (file, ext) 
          sys.exit(3)


# copy a file to a destination directory, and return the basename
# that is the filename without the path. that name is used
# to set a relative link in the html code
def processFile( file, outputDir ):
 
     if file == "None":
          return 'infoNotFound.html'
     else:
          if os.path.isfile(file):
               shutil.copy(file, outputDir)
               return os.path.basename(file)
          else:
               return file

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''
     
