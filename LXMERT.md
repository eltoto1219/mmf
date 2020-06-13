# TODO list for LXMERT

I think it would be a good idea to follow a gitstyle workflow:
https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow
except since its just us three, no need for us to approve pull requests for eachother
we can just add our own feature in seperate branches and then merge into master

However, I am impartial, so if you think its too much work, we should still be fine

6/12/2020 - Antonio
1. added lxmert.py file to mmf/mmf/models, checkout my branch for development
2. created this todo file to keep  track of tasks
3. all code that ive added should be compliant with mmf's code of conduct
4. read code of conduct to make sure we have everything need to edit
5. the lxmert.py file contains all the classes from  src/lxrt/modeling.py in Hao's repo
6. Originally, I made a copy of the vilbert.py class, I've saved all the config options \n
so that we know exactly what we should be providing to our LXMERT code \n
(all in docstrings at the bottom of the file
7. so now I think the next steps will be to make the various .yaml files  for \n
LXMERT making we incorporate everything from the VILBERT config
8.  after that we can tidy up and and add the attn viz and make sure all our inputs \n
are correct for LXMERT in lxmert.py
9. once that is done, we can implement .yaml files for pretraining/finetuning \n
execution
10. after that we can use their testing suite to test our code

Questions I have:

Hao, I have labeled all of the classes/errors that cant be used in MMF in lxmert.py
including your logging class and load\_tf\_bert\_weights class, I am thinking we can get rid
of some of these but just wanted to check with you

Also,  ive annotated classes in the lxmert.py file that you have directly copied over from hugging face or have \n
made modifications too, I am wondering how many of these classes we can just simply import
instead of having to copy them from scratch
