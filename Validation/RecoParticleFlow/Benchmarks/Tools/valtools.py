#!/usr/bin/env python
# to remove a benchmark webpage from the validation website
# author: Colin

import shutil, sys, os

from optparse import OptionParser

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
    

class benchmark:

    def __init__(self, extension = None, release=None):
        
        if release==None:
            self.release_ = os.environ['CMSSW_VERSION']
        else:
            self.release_ = release

        # benchmark directory, as the current working directory
        self.benchmark_ = os.path.basename( os.getcwd() )
        self.benchmarkWithExt_ = self.benchmark_
        if( extension != None ):
            self.benchmarkWithExt_ = '%s_%s' % (self.benchmark_, extension)

    def __str__(self):
        return self.benchmark_
   
    def releaseOnWebSite( self, website ):
        return '%s/%s'  % ( website, self.release_ )

    def benchmarkOnWebSite( self, website ):
        return '%s/%s'  % ( self.releaseOnWebSite(website), 
                            self.benchmarkWithExt_ )

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
            print 'benchmark %s already exists for release %s' % (self.benchmarkWithExt_, self.release_)
            return True
        else:
            return False


