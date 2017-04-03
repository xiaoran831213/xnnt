# fine-tuner for neural networks
import numpy as np
import os
import sys
from os import path as pt
from tnr.cmb import Comb as Tnr
from xutl import spz, lpz
from sae import SAE
sys.path.extend(['..'] if '..' not in sys.path else [])
from pdb import set_trace


def ftn_sae(w, x, u=None, gdy=False, lrt=.001, hte=.001, nft=50, **kwd):
    """ layer-wise unsupervised pre-training for
    stacked autoencoder.
    w: the stacked autoencoders
    x: the inputs.

    nft: maximum number of epochs to go through.
    hte: halting error
    lrt: learning rate

    By default, the the entire SAE of all layers are tuned, that is,
    start = 0, depth = len(w.sa)

    kwd: additional key words to pass on to the trainer.
    """
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


def main(fnm='../../raw/W09/1001', **kwd):
    """ the fine-tune procedure for Stacked Autoencoder(SAE).

    -- fnm: pathname to the input, supposingly the saved progress after the
    pre-training. If {fnm} points to a directory, a file is randomly chosen
    from it.

    ** ae1: depth of the sub SA.
    """
    new_lrt = kwd.pop('lrt', 1e-2)  # new learning rate

    # randomly pick data file if {fnm} is a directory and no record
    # exists in the saved progress:
    if pt.isdir(fnm):
        fnm = pt.join(fnm, np.random.choice(os.listdir(fnm)))
    kwd.update(fnm=fnm)

    # load data from {fnm}, but parameters in {kwd} takes precedence.
    kwd.update((k, v) for k, v in lpz(fnm).iteritems() if k not in kwd.keys())

    # check saved progress and overwrite options:
    sav = kwd.get('sav', pt.basename(fnm).split('.')[0])
    sav = pt.abspath(sav)
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
        kwd.pop('lrt', None)  # use saved learning rate for training
        kwd.pop('nwk', None)  # use saved network for training

        # remaining options in {kwd} take precedence over {sav}.
        sdt = lpz(sav)
        sdt.update(kwd)
        kwd = sdt
        print("continue training.")
    else:  # restart the training
        kwd['lrt'] = new_lrt    # do not use archived NT LRT
        print("restart training.")

    # <-- __x, w, npt, ptn, ... do it.
    gmx = kwd['gmx']
    nsb = gmx.shape[0]                     # sample size
    xmx = gmx.reshape(nsb, -1).astype('f')  # training data
    ngv = xmx.shape[-1]                     # feature size
    mdp = kwd.pop('wdp', 16)                # maximum network depth
    lrt = kwd.pop('lrt', 1e-2)              # learing rates
    dim = [ngv//2**_ for _ in range(mdp) if 2**_ <= ngv]

    # normal training
    # create normal network if necessary
    nwk = kwd.pop('nwk', None)
    if nwk is None:
        nwk = SAE.from_dim(dim, s='relu', **kwd)
        nwk[-1].s = None

        hte = kwd.pop('hte', 0.0005)
        print('NT: HTE = {}'.format(hte))
        kwd = ftn_sae(nwk, xmx, xmx, gdy=0, lrt=lrt, hte=hte, **kwd)
        ftn = kwd.pop('ftn')
        lrt = ftn.lrt.get_value()  # learning rate
        hof = nwk.ec(xmx).eval()   # high order features
        eot = ftn.terr()           # error of training
        eov = ftn.verr()           # error of validation

        # update
        kwd.update(nwk=nwk, lrt=lrt, hte=hte, hof=hof, eot=eot, eov=eov)

    # report halting
    if ftn.hlt:
        print('NT: halted.')

    # save
    if sav:
        print("write to: ", sav)
        spz(sav, kwd)

    kwd = dict((k, v) for k, v in kwd.iteritems() if v is not None)
    return kwd


def ept(fnm='../../sim/W09/SS1_DP5/0004.pgz', out=None):
    """ Export training result in text format.
    -- fnm: filename of training progress.
    -- out: where to export the progress.
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
