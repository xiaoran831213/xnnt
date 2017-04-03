# fine-tuner for neural networks
import numpy as np
import os
import sys
from os import path as pt
from tnr.cmb import Comb as Tnr
from hlp import cv_msk
from xutl import spz, lpz
from sae import SAE
sys.path.extend(['..'] if '..' not in sys.path else [])
from pdb import set_trace


def ftn_sae(w, x, u=None, gdy=False, ae0=None, ae1=None, **kwd):
    """ layer-wise unsupervised pre-training for
    stacked autoencoder.
    w: the stacked autoencoders
    x: the inputs.

    nft: maximum number of epochs to go through for fine-tuning.
    gdy: number of greedy layer-wise pre-training.
    ae0: which autoencoder in the stack to start the tuning?
    ae1: which autoencoder in the stack to end the tuning?

    By default, the the entire SAE of all layers are tuned, that is,
    start = 0, depth = len(w.sa)

    kwd: additional key words to pass on to the trainer.
    """
    # number of epoch to go through
    nft = kwd.get('nft', 20)
    lrt = kwd.pop('lrt', 0.0001)
    hte = kwd.pop('hte', 0.005)

    # select sub-stack
    # w = w.sub(ae1, ae0) if ae0 or ae1 else w
    # x = w.sub(ae0, 0).ec(x).eval() if ae0 else x
    # u = w.sub(ae0, 0).ec(u).eval() if ae0 and u else u

    # a layer-wise greedy training
    if gdy:
        for i in range(len(w.sa) - 1):
            sw = w.sub(i + 1, 0)
            print('stack:', sw)
            ftn = Tnr(sw, x, u=u, lrt=lrt, hte=hte, **kwd)
            ftn.tune(gdy)
            
    # whole stack training
    ftn = Tnr(w, x, u=u, lrt=lrt, hte=hte, **kwd)
    ftn.tune(nft)

    kwd.update(ftn=ftn, nft=nft)
    kwd = dict((k, v) for k, v in kwd.iteritems() if v is not None)

    return kwd


def main(fnm='../../raw/W09/1004', **kwd):
    """ the fine-tune procedure for Stacked Autoencoder(SAE).

    -- fnm: pathname to the input, supposingly the saved progress after the
    pre-training. If {fnm} points to a directory, a file is randomly chosen
    from it.

    ** ae1: depth of the sub SA.
    """
    new_lrt = kwd.pop('lrt', None)  # new learning rate
    new_hte = kwd.pop('hte', None)  # new halting error

    # randomly pick data file if {fnm} is a directory and no record
    # exists in the saved progress:
    if pt.isdir(fnm):
        fnm = pt.join(fnm, np.random.choice(os.listdir(fnm)))
    kwd.update(fnm=fnm)

    # load data from {fnm}, but parameters in {kwd} takes precedence.
    kwd.update((k, v) for k, v in lpz(fnm).iteritems() if k not in kwd.keys())

    # check saved progress and overwrite options:
    sav = kwd.get('sav', '.')
    if pt.isdir(sav):
        sav = pt.join(sav, pt.basename(fnm).split('.')[0])
    if pt.exists(sav + '.pgz'):
        print(sav, ": exists,")
        ovr = kwd.pop('ovr', 0)  # overwrite?

        if ovr is 0 or ovr > 2:  # do not overwrite the progress
            print(" skipped.")
            return kwd
    else:
        ovr = 2

    # resume progress, use network stored in {sav}.
    if ovr is 1:
        kwd.pop('cvw', None)    # use saved networks for CV
        kwd.pop('cvl', None)    # use saved CV LRT
        kwd.pop('cvh', None)    # use saved CV halting state
        kwd.pop('cve', None)    # use saved CV halting error
        kwd.pop('lrt', None)    # use saved learning rate for training
        kwd.pop('nwk', None)    # use saved network for training

        # remaining options in {kwd} take precedence over {sav}.
        sdt = lpz(sav)
        sdt.update(kwd)
        kwd = sdt
        print("continue training.")
    else:  # restart the training
        kwd.pop('lrt', None)    # do not use archived NT LRT
        kwd.pop('cvl', None)    # do not use archived CV LRT
        kwd.pop('cve', None)    # do not use archived CV errors
        kwd.pop('cvh', None)    # do not use archived CV halting state
        print("restart training.")

    # <-- __x, w, npt, ptn, ... do it.
    gmx = kwd['gmx']
    nsb = gmx.shape[0]                     # sample size
    xmx = gmx.reshape(nsb, -1).astype('f')  # training data
    ngv = xmx.shape[-1]                     # feature size
    mdp = kwd.pop('wdp', 16)                # maximum network depth
    # learing rates
    lrt = new_lrt if new_lrt else kwd.pop('lrt', 1e-4)
    dim = [ngv//2**_ for _ in range(mdp) if 2**_ <= ngv]

    # cross-validation networks
    cvk = kwd.get('cvk', 2)                    # K
    cvm = kwd.get('cvm', cv_msk(xmx, cvk))     # mask
    cvh = kwd.pop('cvh', [None] * cvk)         # halting
    cvl = kwd.pop('cvl', [lrt] * cvk)          # learning rates
    cvw = kwd.pop('cvw', [None] * cvk)         # slots for CV networks
    cve = kwd.pop('cve', np.ndarray((cvk, 2)))  # error

    # tune the network: (1) CV
    for i, m in enumerate(cvm):
        msg = 'CV: {:02d}/{:02d}'.format(i + 1, cvk)
        if cvh[i]:
            msg = msg + ' halted.'
            print(msg)
            continue

        print(msg)
        if cvw[i] is None:
            cvw[i] = SAE.from_dim(dim, s='relu', **kwd)
            cvw[i][-1].s = 'sigmoid'
            cvw[i].ec[-1].s = 'sigmoid'
            
            # suggest no layer-wise treatment (relu)
            gdy = kwd.get('gdy', False)
        else:
            # suggest no layer-wise treatment
            gdy = kwd.get('gdy', False)
        kwd = ftn_sae(cvw[i], xmx[-m], xmx[+m], gdy=gdy, lrt=cvl[i], **kwd)

        # collect the output
        ftn = kwd.pop('ftn')
        cvl[i] = ftn.lrt.get_value()  # CV learning rate
        cve[i, 0] = ftn.terr()        # CV training error
        cve[i, 1] = ftn.verr()        # CV validation error
        cvh[i] = ftn.hlt              # CV halting?
    # update
    kwd.update(cvk=cvk, cvm=cvm, cvh=cvh, cvl=cvl, cve=cve, cvw=cvw)

    # (2) normal training
    # force continue of training till new halting error?
    if new_hte:
        [kwd.pop(_, None) for _ in ['hte', 'hof', 'eot', 'eov']]
        hte = new_hte
    else:
        # mean CV training error as halting error
        hte = kwd.pop('hte', cve[:, 0].mean())
        
    # NT only happens when all CV is halted.
    if all(cvh):
        # create normal network if necessary
        nwk = kwd.pop('nwk', None)
        if nwk is None:
            nwk = SAE.from_dim(dim, s='relu', **kwd)
            nwk[-1].s = 'sigmoid'
            nwk.ec[-1].s = 'sigmoid'

            # suggest no layer-wise treatment (relu)
            gdy = kwd.get('gdy', False)
        else:
            # suggest no layer-wise treatment
            gdy = kwd.get('gdy', False)

        print('NT: HTE = {}'.format(hte))
        kwd = ftn_sae(nwk, xmx, xmx, gdy=gdy, lrt=lrt, hte=hte, **kwd)
        ftn = kwd.pop('ftn')
        lrt = ftn.lrt.get_value()  # learning rate
        hof = nwk.ec(xmx).eval()   # high order features
        eot = ftn.terr()           # error of training
        eov = ftn.verr()           # error of validation

        # update
        kwd.update(nwk=nwk, lrt=lrt, hte=hte, hof=hof, eot=eot, eov=eov)

        # when NT halt, report.
        if ftn.hlt:
            print('NT: halted.')
    else:
        print('NT: Not Ready.')  # not ready for NT

    # save
    if sav:
        print("write to: ", sav)
        spz(sav, kwd)

    kwd = dict((k, v) for k, v in kwd.iteritems() if v is not None)
    return kwd


def ept(fnm, out=None):
    """ Export training result in text format.
    -- fnm: filename of training progress.
    -- out: where to save the export.
    """
    pwd = os.getcwd()
    fnm = pt.abspath(fnm)

    import tempfile
    tpd = tempfile.mkdtemp()
    if out is None:
        out = pwd
    if pt.isdir(out):
        out = pt.join(out, pt.basename(fnm).split('.')[0])
    if not out.endswith('.tgz') or out.endswith('.tar.gz'):
        out = out + '.tgz'
    out = pt.abspath(out)

    # read the training progress
    dat = lpz(fnm)
    [dat.pop(_) for _ in ['nwk', 'cvm', 'cvw', 'nft', 'ovr']]

    dat['fnm'] = fnm
    dat['out'] = out

    # genomic matrix
    gmx = dat.pop('gmx').astype('i1')
    np.savetxt(pt.join(tpd, 'gx0.txt'), gmx[:, 0, :], '%d')
    np.savetxt(pt.join(tpd, 'gx1.txt'), gmx[:, 1, :], '%d')

    # genomic map
    np.savetxt(pt.join(tpd, 'gmp.txt'), dat.pop('gmp'), '%d\t%d\t%s')

    # subjects
    np.savetxt(pt.join(tpd, 'sbj.txt'), dat.pop('sbj'), '%s')

    # untyped subjects (indices)
    # np.savetxt(pt.join(tpd, 'usb.txt'), dat.pop('usb'), '%d')

    # untyped variants (indices)
    # np.savetxt(pt.join(tpd, 'ugv.txt'), dat.pop('ugv'), '%d')

    # CV-errors
    cve = dat.pop('cve')
    np.savetxt(pt.join(tpd, 'cve.txt'), cve, '%.8f')

    # high-order features
    hof = dat.pop('hof')
    np.savetxt(pt.join(tpd, 'hof.txt'), hof, '%.8f')

    # meta information
    inf = open(pt.join(tpd, 'inf.txt'), 'w')
    for k, v in dat.iteritems():
        inf.write('{}={}\n'.format(k, v))
    inf.close()  # done

    # pack the output, delete invididual files
    import tarfile
    import shutil

    # packing
    os.chdir(tpd)  # goto the packing dir
    try:
        tar = tarfile.open(out, 'w:gz')
        [tar.add(_) for _ in os.listdir('.')]
        shutil.rmtree(tpd, True)
        tar.close()
    except Exception as e:
        print(e)
    os.chdir(pwd)  # back to the working dir
