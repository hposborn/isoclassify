import pdb
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import random
from .priors import *

def plotinit():
    fig_post= plt.figure('posteriors',figsize=(8,12))
    fig_post.clf()
    fig_hrd = plt.figure('hrd',figsize=(8,12))
    fig_hrd.clf()

    #plt.subplots_adjust(left=0.08, bottom=0.04, right=0.96, top=0.96, wspace=0.27, \
    #                    hspace=0.6)

    #fig1.set_tight_layout(True)
    #plt.draw()
    #plt.show()

    fig_post.subplots_adjust(left=0.08, bottom=0.04, right=0.96, top=0.96, wspace=0.27, \
                            hspace=0.6)
    fig_hrd.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.97, wspace=0.3, \
                            hspace=0.5)
    return fig_post,fig_hrd

def plotclear():
    #raw_input(':')
    plt.clf()
    plt.figure('posteriors')
    plt.clf()

#def plotposterior(x,y,res,err1,err2,avs,model,model_red,names,j,medav,stdav,grcol,ricol,grcole,ricole,Mg,Mge,ix,iy):
def plotposterior(x,y,res,err1,err2,names,j,ix,iy,fig_post=None):
    if fig_post is None:
        fig = plt.figure(1,figsize=(8,12))
    else:
        fig=fig_post
    ax1 = fig.add_subplot(len(names),2,ix)
    ax1.plot(x,np.cumsum(y))
    ax1.plot([res,res],[0,1],'r')
    ax1.plot([res+err1,res+err1],[0,1],'--r')
    ax1.plot([res-err2,res-err2],[0,1],'--r')
    ax1.set_ylim([0,1])
    ax1.set_title(names[j])
    if fnmatch.fnmatch(names[j],'*rho*'):
        ax1.set_xscale('log')
    if fnmatch.fnmatch(names[j],'*lum*'):
        ax1.set_xscale('log')


    ax2 = fig.add_subplot(len(names),2,iy)
    ax2.plot(x,y)
    ax2.plot([res,res],[0,1],'r')
    ax2.plot([res+err1,res+err1],[0,1],'--r')
    ax2.plot([res-err2,res-err2],[0,1],'--r')
    if np.isnan(y).sum()!=len(y):
        ax2.set_ylim([0,np.nanmax(y)+np.nanmax(y)*0.1])
    ax2.set_title(names[j])
    if fnmatch.fnmatch(names[j],'*rho*'):
        ax2.set_xscale('log')
    if fnmatch.fnmatch(names[j],'*lum*'):
        ax2.set_xscale('log')

    if fnmatch.fnmatch(names[j],'*feh*'):
        xt=np.arange(-2.,1.,0.01)
        yt=fehprior(xt)
        ax2.plot(xt,yt*np.nanmax(y)/np.nanmax(yt),'--g')

    '''
    if fnmatch.fnmatch(names[j],'*avs*'):
        xt=np.arange(np.nanmin(avs),np.nanmax(avs),0.001)
        yt=gaussian(xt,1.,medav,stdav,0.)
        #yt=avprior(xt,data,i,dust,dist)
        plt.plot(xt,yt*np.nanmax(y)/np.nanmax(yt),'--g')
    '''
    #fig.set_tight_layout(True)
    return fig

def plothrd(model,input,mabs,mabse,ix,iy,fig_hrd=None):
    if fig_hrd is None:
        fig=plt.figure('hrd')
    else:
        fig=fig_hrd

    fig.subplots_adjust(left=0.08, bottom=0.05, right=0.96, top=0.96, wspace=0.31, \
                        hspace=0.26)
    ax1=fig.add_subplot(2,3,1)
    frac=0.01

    ran=np.array(random.sample(range(len(model['teff'])),\
    int(len(model['teff'])*frac)))

    ### Sloan color-color
    d=np.where(model['logg'][ran] > 3.5)[0]
    ax1.plot(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model['rmag'][ran[d]]-model['imag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)
    g=np.where(model['logg'][ran] < 3.5)[0]
    ax1.plot(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model['rmag'][ran[g]]-model['imag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)

    if ((input.gmag > -99) & (input.rmag > -99)):
        ax1.errorbar([input.gmag-input.rmag], [input.rmag-input.imag], \
                 xerr=np.sqrt(input.gmage**2+input.rmage**2), \
                 yerr=np.sqrt(input.rmage**2+input.image**2),color='green',elinewidth=5)

    ax1.set_xlabel('g-r')
    ax1.set_ylabel('r-i')
    ax1.set_xlim([-0.5,2.5])
    ax1.set_ylim([-0.5,2])


    ### Sloan color-color
    ax2=fig.add_subplot(2,3,2)
    d=np.where(model['logg'][ran] > 3.5)[0]
    ax2.plot(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model['imag'][ran[d]]-model['zmag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)
    g=np.where(model['logg'][ran] < 3.5)[0]
    ax2.plot(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model['imag'][ran[g]]-model['zmag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)

    if ((input.imag > -99) & (input.zmag > -99)):
        ax2.errorbar([input.gmag-input.rmag], [input.imag-input.zmag], \
                 xerr=np.sqrt(input.gmage**2+input.rmage**2), \
                 yerr=np.sqrt(input.image**2+input.zmage**2),color='green',elinewidth=5)

    ax2.set_xlabel('g-r')
    ax2.set_ylabel('i-z')
    ax2.set_xlim([-0.5,2.5])
    ax2.set_ylim([-0.5,2])


### Sloan color-color
    ax3=fig.add_subplot(2,3,3)
    d=np.where(model['logg'][ran] > 3.5)[0]
    ax3.plot(model['hmag'][ran[d]]-model['kmag'][ran[d]],\
             model['jmag'][ran[d]]-model['hmag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)
    g=np.where(model['logg'][ran] < 3.5)[0]
    ax3.plot(model['hmag'][ran[g]]-model['kmag'][ran[g]],\
             model['jmag'][ran[g]]-model['hmag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)

    if ((input.jmag > -99) & (input.hmag > -99)):
        ax3.errorbar([input.hmag-input.kmag], [input.jmag-input.hmag], \
                 xerr=np.sqrt(input.hmage**2+input.kmage**2), \
                 yerr=np.sqrt(input.jmage**2+input.hmage**2),color='green',elinewidth=5)

    ax3.set_xlabel('H-K')
    ax3.set_ylabel('J-H')
    ax3.set_xlim([-0.1,0.5])
    ax3.set_ylim([-0.3,1.3])


    ### 2MASS color-color
    ax4=fig.add_subplot(2,3,4)
    ax4.plot(model['btmag'][ran[d]]-model['vtmag'][ran[d]],\
             model['jmag'][ran[d]]-model['hmag'][ran[d]],\
             '.',color='blue',markersize=1,zorder=-32)
    ax4.set_xlim([-0.1,2.5])
    ax4.set_ylim([-0.2,1.2])
    ax4.plot(model['btmag'][ran[g]]-model['vtmag'][ran[g]],\
             model['jmag'][ran[g]]-model['hmag'][ran[g]],\
             '.',color='red',markersize=1,zorder=-32)

    if ((input.vtmag > -99) & (input.btmag > -99)):
        ax4.errorbar([input.btmag-input.vtmag], [input.jmag-input.hmag], \
                 xerr=np.sqrt(input.btmage**2+input.vtmage**2), \
                 yerr=np.sqrt(input.jmage**2+input.hmage**2),color='green',elinewidth=5)

    ax4.set_xlabel('Bt-Vt')
    ax4.set_ylabel('J-H')

    # CMD
    ax5=fig.add_subplot(2,3,5)
    mag1='bmag'
    mag2='vmag'
    absmag='vmag'
    col=None

    if (input.vmag > 0):
        mag1='bmag'
        mag2='vmag'
        absmag='vmag'
        col=input.bmag-input.vmag
        cole=np.sqrt(input.bmage**2+input.vmage**2)

    if (input.vtmag > 0):
        mag1='btmag'
        mag2='vtmag'
        absmag='vtmag'
        col=input.btmag-input.vtmag
        cole=np.sqrt(input.btmage**2+input.vtmage**2)

    if (input.gmag > 0):
        mag1='gmag'
        mag2='rmag'
        absmag='gmag'
        col=input.gmag-input.rmag
        cole=np.sqrt(input.gmage**2+input.rmage**2)

    if (input.jmag > 0):
        mag1='jmag'
        mag2='kmag'
        absmag='jmag'
        col=input.jmag-input.kmag
        cole=np.sqrt(input.jmage**2+input.kmage**2)

    ax5.plot(model[mag1][ran[d]]-model[mag2][ran[d]],\
             model[absmag][ran[d]],'.',color='blue',markersize=1,zorder=-32)

    ax5.plot(model[mag1][ran[g]]-model[mag2][ran[g]], \
             model[absmag][ran[g]],'.',color='red',markersize=1,zorder=-32)

    if (input.plx > 0.) and col is not None:
        ax5.errorbar([col], [mabs], xerr=cole, yerr=mabse,color='green',elinewidth=5)

    ax5.set_xlim([-0.5,2])
    ax5.set_ylim([np.nanmax(model[absmag]),np.nanmin(model[absmag])])
    ax5.set_xlabel(mag1+'-'+mag2)
    ax5.set_ylabel(absmag)

    # HRD
    ax6=fig.add_subplot(2,3,6)

    if (input.numax == 0):
        ax6.plot(model['teff'][ran[d]],model['logg'][ran[d]],\
                 '.',color='blue',markersize=1,zorder=-32)
        ax6.set_xlim([10000,2000])
        ax6.set_ylim([6,0])
        ax6.set_xlabel('TEFF')
        ax6.set_ylabel('logg')
        ax6.plot(model['teff'][ran[g]],model['logg'][ran[g]],\
                 '.',color='red',markersize=1,zorder=-32)

        ax6.errorbar([input.teff], [input.logg], xerr=input.teffe, yerr=input.logge, \
                 color='green',elinewidth=5)

    else:
        mod_numax=3090*(10**model['logg']/27420.)*(model['teff']/5777.)**(-0.5)
        ax6.semilogy(model['teff'][ran[d]],mod_numax[ran[d]],\
                 '.',color='blue',markersize=1,zorder=-32)
        ax6.set_xlim([10000,2000])
        ax6.set_xlabel('TEFF')
        ax6.set_ylabel('nu max')
        ax6.set_ylim([100000,0.1])
        ax6.plot(model['teff'][ran[g]],mod_numax[ran[g]],\
                 '.',color='red',markersize=1,zorder=-32)

        ax6.errorbar([input.teff], [input.numax], xerr=input.teffe, yerr=input.numaxe, \
                 color='green',elinewidth=5)
    return fig




























def plothrdold(model,grcol,ricol,grcole,ricole,Mg,Mge,ix,iy):

    fig=plt.figure('hrd')
    ax1=fig.add_subplot(3,1,1)
    frac=0.01

    ran=np.array(random.sample(range(len(model['teff'])),\
    int(len(model['teff'])*frac)))

    d=np.where(model['logg'][ran] > 3.5)[0]
    ax1.plot(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model['rmag'][ran[d]]-model['imag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)

    g=np.where(model['logg'][ran] < 3.5)[0]
    ax1.plot(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model['rmag'][ran[g]]-model['imag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)
    ax1.errorbar([grcol], [ricol], xerr=grcole, yerr=ricole,color='green',elinewidth=5)
    ax1.set_xlabel('g-r')
    ax1.set_ylabel('r-i')
    ax1.set_xlim([-0.5,2.5])
    ax1.set_ylim([-0.5,2])


    ax2=fig.add_subplot(3,1,2)
    ax2.plot(model['hmag'][ran[d]]-model['kmag'][ran[d]],\
             model['jmag'][ran[d]]-model['hmag'][ran[d]],\
             '.',color='blue',markersize=1,zorder=-32)
    ax2.set_xlim([-0.1,0.4])
    ax2.set_ylim([-0.2,1.2])

    ax2.plot(model['hmag'][ran[g]]-model['kmag'][ran[g]],\
             model['jmag'][ran[g]]-model['hmag'][ran[g]],\
             '.',color='red',markersize=1,zorder=-32)

    ax2.set_xlabel('H-K')
    ax2.set_ylabel('J-H')

    #ran=np.array(random.sample(range(len(model_red['teff'])),\
    #int(len(model_red['teff'])*frac)))

    '''
    plt.plot(model_red['gmag'][ran]-model_red['rmag'][ran],\
    model_red['rmag'][ran]-model_red['imag'][ran]\
    ,'.',color='red',markersize=3,zorder=-32)
    '''
    '''
    um=np.where(model_red['avs'][ran] == np.nanmax(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['rmag'][ran[um]]-model_red['imag'][ran[um]]\
    ,'.',color='blue',markersize=3,zorder=-32)

    um=np.where(model_red['avs'][ran] == np.nanmin(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['rmag'][ran[um]]-model_red['imag'][ran[um]]\
    ,'.',color='yellow',markersize=3,zorder=-32)
   '''

    ax3=fig.add_subplot(3,1,3)
    #plt.errorbar([grcol], [Mg], xerr=grcole, yerr=Mge,color='green',elinewidth=15)

    #ran=np.array(random.sample(range(len(model['teff'])),\
    #int(len(model['teff'])*frac)))

    ax3.plot(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model['gmag'][ran[d]],\
             '.',color='blue',markersize=1,zorder=-32)

    ax3.plot(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model['gmag'][ran[g]],\
             '.',color='red',markersize=1,zorder=-32)

    '''
    ran=np.array(random.sample(range(len(model_red['teff'])),\
    int(len(model_red['teff'])*frac)))

    plt.plot(model_red['gmag'][ran]-model_red['rmag'][ran],\
    model_red['gmag'][ran]\
    ,'.',color='red',markersize=3,zorder=-32)
    '''
    '''
    um=np.where(model_red['avs'][ran] == np.nanmax(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['gmag'][ran[um]]\
    ,'.',color='blue',markersize=3,zorder=-32)

    um=np.where(model_red['avs'][ran] == np.nanmin(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['gmag'][ran[um]]\
    ,'.',color='yellow',markersize=3,zorder=-32)
    '''

    ax3.errorbar([grcol], [Mg], xerr=grcole, yerr=Mge,color='green',elinewidth=5)

    ax3.set_xlim([-0.5,2])
    ax3.set_ylim([15,-5])
    ax3.set_xlabel('g-r')
    ax3.set_ylabel('Mg')
    return fig













def plothrd2(x,y,res,err1,err2,avs,model,model_red,names,j,medav,stdav,grcol,ricol,grcole,ricole,plx,plxe,ix,iy,model_plx):

    plt.figure('hrd')
    plt.subplot(3,1,1)
    frac=0.01

    ran=np.array(random.sample(range(len(model['teff'])),\
    int(len(model['teff'])*frac)))

    d=np.where(model['logg'][ran] > 3.5)[0]
    plt.plot(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model['rmag'][ran[d]]-model['imag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)

    g=np.where(model['logg'][ran] < 3.5)[0]
    plt.plot(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model['rmag'][ran[g]]-model['imag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)
    plt.errorbar([grcol], [ricol], xerr=grcole, yerr=ricole,color='green',elinewidth=5)
    plt.xlabel('g-r')
    plt.ylabel('r-i')
    plt.xlim([-0.5,2.5])
    plt.ylim([-0.5,2])


    plt.subplot(3,1,2)
    plt.plot(model['hmag'][ran[d]]-model['kmag'][ran[d]],\
             model['jmag'][ran[d]]-model['hmag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)
    plt.xlim([-0.1,0.4])
    plt.ylim([-0.2,1.2])

    plt.plot(model['hmag'][ran[g]]-model['kmag'][ran[g]],\
             model['jmag'][ran[g]]-model['hmag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)

    plt.xlabel('H-K')
    plt.ylabel('J-H')

    #ran=np.array(random.sample(range(len(model_red['teff'])),\
    #int(len(model_red['teff'])*frac)))

    '''
    plt.plot(model_red['gmag'][ran]-model_red['rmag'][ran],\
    model_red['rmag'][ran]-model_red['imag'][ran]\
    ,'.',color='red',markersize=3,zorder=-32)
    '''
    '''
    um=np.where(model_red['avs'][ran] == np.nanmax(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['rmag'][ran[um]]-model_red['imag'][ran[um]]\
    ,'.',color='blue',markersize=3,zorder=-32)

    um=np.where(model_red['avs'][ran] == np.nanmin(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['rmag'][ran[um]]-model_red['imag'][ran[um]]\
    ,'.',color='yellow',markersize=3,zorder=-32)
   '''

    plt.subplot(3,1,3)
    #plt.errorbar([grcol], [Mg], xerr=grcole, yerr=Mge,color='green',elinewidth=15)

    #ran=np.array(random.sample(range(len(model['teff'])),\
    #int(len(model['teff'])*frac)))

    plt.semilogy(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model_plx[ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)

    plt.semilogy(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model_plx[ran[g]],\
    '.',color='red',markersize=1,zorder=-32)

    '''
    ran=np.array(random.sample(range(len(model_red['teff'])),\
    int(len(model_red['teff'])*frac)))

    plt.plot(model_red['gmag'][ran]-model_red['rmag'][ran],\
    model_red['gmag'][ran]\
    ,'.',color='red',markersize=3,zorder=-32)
    '''
    '''
    um=np.where(model_red['avs'][ran] == np.nanmax(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['gmag'][ran[um]]\
    ,'.',color='blue',markersize=3,zorder=-32)

    um=np.where(model_red['avs'][ran] == np.nanmin(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['gmag'][ran[um]]\
    ,'.',color='yellow',markersize=3,zorder=-32)
    '''

    plt.errorbar([grcol], [plx], xerr=grcole, yerr=plxe,color='green',elinewidth=5)

    plt.xlim([-0.5,2])
    plt.ylim([np.nanmax(model_plx),np.nanmin(model_plx)])
    plt.xlabel('g-r')
    plt.ylabel('plx')
