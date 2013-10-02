---
title: CMS Offline Software
layout: default
related:
 - { name: Project page, link: 'https://github.com/cms-sw/cmssw' }
 - { name: Feedback, link: 'https://github.com/cms-sw/cmssw/issues/new' }
---

## Tutorial: using your UserCode repository in CMSSW

This tutorial will guide you through the use of your usercode repository,
created via the [usercode FAQ](usercode-faq.html) into CMSSW.

### Disclaimer

The instructions contained in this tutorial are illustrative only. They of
course do not produce any sensible analysis result.

### Before you start.

Please make sure you registered to GitHub and that you have provided them
a ssh public key to access your private repository. For more information see
the [FAQ](faq.html).

We will also assume that you are in the src area of some CMSSW release, i.e. you have 
set up a scram area with:

    scram project CMSSW_6_2_0
    cd CMSSW_6_2_0/src
    cmsenv

### The analysis use case.

There are two ways you can use your usercode repository together in CMSSW:

* by checking it out as a separate entity.
* by merging a branch of your usercode repository as a branch of CMSSW, which
  works on top some defined release.

the former approach is the easiest and is best for simple cases, where you do
not overlap with CMSSW and where you do not have any particular need of
commiting back to the official CMSSW.

the second approach is slighly more complex to start with, however it's the one
which fits best the git workflow. Moreover you'll probably find it the easiest
to work with, after the initial setup required. This approach is best suited
for more complex cases, where your usercode area is really an extension of
CMSSW and when you would like at some point to propagate your changes back to
the official CMSSW.

### The easy one: checking out in parallel.

This is simply done by doing:

    git cms-addpkg FWCore/Version
    git clone git@github.com:<your-user-name>/usercode src/UserCode/<your-name>

notice the initial `git cms-addpkg` is only needeed to create an initial area.
It will be soon replaced by a `git cms-init`.

Done this you can add additional CMSSW packages by simply doing:

    git cms-addpkg SubSystem/Package

You also can modify your usercode area by going into it and using the standard
git commands.

As we said, this approach is the easiest to do, however it has the drawback
that the two repositories are really separate so all the git commands (e.g. git
diff) need to be issued separately.

### Working on top of CMSSW

A more git-like approach is to consider your usercode area as an addition on
top of a given CMSSW release i.e. a branch spawing from some CMSSW release. 

If you followed the instructions in [the UserCode FAQ](usercode-faq.html) to
create the repository, the first step will be to merge a branch of your
usercode repository in a CMSSW branch. While this is admittedly a non trivial
operation it's still straighforward for most common cases and most important
has to be done only once per branch, tag. Done that, you can use the `git
cms-merge-topic` command to bring in your changes.

The first step, once again is to checkout some CMSSW package to setup CMS
environment (again this will be replaced by `git cms-init`):

   git cms-addpkg FWCore/Version

done this you'll have to add your usercode repository as one of the available
ones. This is done via the standard [git remote
add](https://www.kernel.org/pub/software/scm/git/docs/git-remote.html) command:

    git remote add -f usercode https://github.com/<your-github-username>/usercode

you can then proceed at merging a given branch (say `master`) into the CMSSW area:

    git merge -s ours --no-commit usercode/master
    git read-tree --prefix=UserCode/<my-name> -u usercode/master 

notice that the argument of `--prefix` is where your repository will be merged
and that an empty prefix (i.e. `--prefix= ` ) is a valid one which can be used
in the case your usercode area already contains a standard
`<sub-system>/<package>` layout.

You also can repeat these steps multiple times if you have more than one
repository you would like to merge.  For example you might want to use those
present in https://github.com/cms-analysis.

Once you are happy with the merge (use `git log --graph --oneline` or `git
status` to see what has been done by the various merges) you can commit
your work via:

    git commit -m 'Merge usercode in CMSSW'

and push the branch to your private CMSSW repository:

    git push my-cmssw HEAD:some-branch-name

You are done. From this moment on you can merge your changes in a workarea by simply doing:

    git cms-merge-topic <your-username>:some-branch-name

For more information and the details on how all this works you can also look
[here](http://git-scm.com/book/ch6-7.html),
[here](https://help.github.com/articles/working-with-subtree-merge) and
[here](https://www.kernel.org/pub/software/scm/git/docs/howto/using-merge-subtree.html).
