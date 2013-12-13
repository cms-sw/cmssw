---
title: CMS Offline Software
layout: default
related:
 - { name: Project page, link: 'https://github.com/cms-sw/cmssw' }
 - { name: Feedback, link: 'https://github.com/cms-sw/cmssw/issues/new' }
---
# Rosetta Stone

### Converting CVS-speak to git-speak

| CMS CVS                              | CMS GIT                                      |
| -                                    | -                                            |
| tagset                               | topic branch                                 |
| publish a tagset                     | create a pull request                        |
| `addpkg`                             | `git cms-addpkg`                             |
| `checkdeps`                          | `git cms-checkdeps`                          |
| `cmstc tagset <tagset-id> -a`        | `git cms-merge-topic <topic-id>`             |
| `cvs log`                            | `git log`                                    |
| `cd $CMSSW_BASE ; cvs diff -r HEAD`  | `git diff --staged`                          |
| `cvs diff `                          | `git diff --staged .`                        |
| `cvs diff -r some tag`               | `git diff -r <some-tag>`                     |
| `cvs diff -r TAG1 -r TAG2 <path>`    | `git diff TAG1..TAG2 -- <path>`              |


### Useful aliases to ease transition:

You can set up the given aliases by doing:

    git config --global alias.cms-<alias-name> '<alias code>'

then you can reuse them via:

    git cms-<alias-name>

it's recommended you prefix your aliases with "cms-" or "your-name-" so that
people coming from the real world actually know about the fact it's not an
official git command.

| CVS command                           | git alias                                                                                     |
|-                                      | -                                                                                             |
| `cvs rdiff -r TAG1 -r TAG2 <package>` | `git config --global alias.cms-rdiff '!git-rdiff $@' ; git cms-rdiff TAG1..TAG2 -- <package>` |



### Converting "Real World" git to "CMS" git

| Rest of the world                    | CMS                                          |
| -                                    | -                                            |
| `upstream`                           | `official-cmssw`                             |
| `origin`                             | `my-cmssw`                                   |
