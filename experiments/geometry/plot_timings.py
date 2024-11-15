import numpy as np
import sys
sys.path.append('../../')

from matplotlib import pyplot as plt

def collect_timings():
    data = np.loadtxt("results/geometry/corner/mapping_timings.dat")
    data = data[data[:,0].argsort()]
    nv = data[:,0]

    nv = np.unique(data[:,0])

    nc = np.zeros((nv.size,3))
    af = nc.copy()
    ab = nc.copy()
    tf = nc.copy()
    tb = nc.copy()
    of = nc.copy()
    sf = nc.copy()
    qf = nc.copy()

    time_list = [nc,af,ab,tf,tb,of,sf,qf]
    for ii in np.arange(len(nv)):
        selector = data[:,0] == nv[ii]
        for jj in [1,2,3,4,5,6,7,8]:
            curr = data[selector,jj]
            curr = curr[curr > -1]
            if curr.size != 0:
                time_list[jj-1][ii][0] = np.mean(curr) # otherwise keep 0
                time_list[jj-1][ii][1] = np.quantile(curr,0.25)
                time_list[jj-1][ii][2] = np.quantile(curr,0.75)

    nc, af, ab, tf, tb, of, sf, qf = time_list

    fig, (ax1) = plt.subplots(1, 1, figsize=(18.0, 6.0))
    ax1.set_aspect('equal')
    ax1.plot(nv, af[:,0], c="k", label='dQP Forward')
    ax1.plot(nv, ab[:,0], c="k",linestyle='--', label='dQP Backward')
    ax1.plot(nv[of[:,0]  > 0], of[of[:,0]  > 0,0], c="r", label='Optnet Forward')
    ax1.plot(nv[sf[:,0] > 0], sf[sf[:,0]  > 0,0], c="g", label='SCQPTH Forward')
    ax1.plot(nv[qf[:,0]  > 0], qf[qf[:,0]  > 0,0], c="purple", label='QPLayer Forward')

    ax1.scatter(nv, af[:,0], c="k", label='dQP Forward')
    ax1.scatter(nv, ab[:,0], c="k", label='dQP Backward')
    ax1.scatter(nv[of[:,0] > 0], of[of[:,0] > 0,0], c="r", label='Optnet Forward')
    ax1.scatter(nv[sf[:,0] > 0], sf[sf[:,0] > 0,0], c="g", label='SCQPTH Forward')
    ax1.scatter(nv[qf[:,0] > 0], qf[qf[:,0] > 0,0], c="purple", label='QPLayer Forward')

    ax1.fill_between(nv, af[:,1],  af[:,2], alpha=0.2, color="k") # needs to be along an axis
    ax1.fill_between(nv, ab[:,1],  ab[:,2], alpha=0.2, color="k")
    ax1.fill_between(nv[of[:,0] > 0], of[of[:,0] > 0,1],  of[of[:,0] > 0,2], alpha=0.2, color="r")
    ax1.fill_between(nv[sf[:,0] > 0], sf[sf[:,0] > 0,1],  sf[sf[:,0] > 0,2], alpha=0.2, color="g")
    ax1.fill_between(nv[qf[:,0] > 0], qf[qf[:,0] > 0,1],  qf[qf[:,0] > 0,2], alpha=0.2, color="purple")

    # ax1.plot(nv, tf, c="r", label='Total Forward')
    # ax1.plot(nv, tb, c="r",linestyle='--', label='Total Backward')

    # reference line
    y0, y1 = np.log([1e-1, 1e0])
    mid_x, mid_y = (np.log(1e1) + np.log(1e3)) / 2, (y0 + y1) / 2
    slope = 1
    x0 = mid_x + slope * (y0 - mid_y)
    x1 = mid_x + slope * (y1 - mid_y)
    ax1.plot(np.exp([x0, x1]), np.exp([y0, y1]), color='b')

    ax1.set(xlim=[5,2e4],ylim=[1e-3, 1e0], xlabel='# Vertices', ylabel='Mean Time (s)', xscale='log', yscale='log')
    ax1.legend()
    plt.savefig('results/geometry/corner/mapping_timings.pdf') #, dpi=300)
    return None

collect_timings()