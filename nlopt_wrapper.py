# Wrapper around NLOPT module
# written by Johannes Achleitner
# extended by Kai Rohde-Brandenburger January 2023
# Goal is to make NLOPT useable as scipy.optimize

import nlopt
import os
import pickle
'''
def readRestart(x0):
    if os.path.exists("Restart.p"):
        with open("Restart.p", 'rb') as file:
            x0 = pickle.load(file)
    return x0
'''
# derivative-free optimizers
#############################

def subplex(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=6000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    opt = nlopt.opt(nlopt.LN_SBPLX, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    if lbnds is not None:
        lb = opt.set_lower_bounds(lbnds);
    if ubnds is not None:
        ub = opt.set_upper_bounds(ubnds);
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_maxtime(maxtime)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    return res
def BOBYQA(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=60000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    opt = nlopt.opt(nlopt.LN_BOBYQA, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    if lbnds is not None:
        lb = opt.set_lower_bounds(lbnds);
    if ubnds is not None:
        ub = opt.set_upper_bounds(ubnds);
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    return res

def COBYLA(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=600000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    opt = nlopt.opt(nlopt.LN_COBYLA, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    if lbnds is not None:
        lb = opt.set_lower_bounds(lbnds);
    if ubnds is not None:
        ub = opt.set_upper_bounds(ubnds);
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    return res
    
def NEWUOA(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=6000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    opt = nlopt.opt(nlopt.LN_NEWUOA, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    if lbnds is not None:
        opt.set_lower_bounds(lbnds);
    if ubnds is not None:
        opt.set_upper_bounds(ubnds);
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    return res

def NEWUOA_BOUND(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=6000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    opt = nlopt.opt(nlopt.LN_NEWUOA_BOUND, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    return res


def PRAXIS(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=6000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    opt = nlopt.opt(nlopt.LN_PRAXIS, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    if lbnds is not None:
        lb = opt.set_lower_bounds(lbnds);
    if ubnds is not None:
        ub = opt.set_upper_bounds(ubnds);
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    return res    


def NELDERMEAD(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=6000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    if lbnds is not None:
        lb = opt.set_lower_bounds(lbnds);
    if ubnds is not None:
        ub = opt.set_upper_bounds(ubnds);
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    return res        

def AUGLAG(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=60000000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    opt = nlopt.opt(nlopt.LN_AUGLAG, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    if lbnds is not None:
        opt.set_lower_bounds(lbnds);
    if ubnds is not None:
        opt.set_upper_bounds(ubnds);
    opt.set_maxeval(maxtime)
    #opt.set_initial_step(2.5)
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    #print(res.nfev)
    return res

def AUGLAG_EQ(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=10000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    opt = nlopt.opt(nlopt.LN_AUGLAG_EQ, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    if lbnds is not None:
        lb = opt.set_lower_bounds(lbnds);
    if ubnds is not None:
        ub = opt.set_upper_bounds(ubnds);
    opt.set_maxeval(maxtime)
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    return res

def GN_ISRES(func, x0, args,ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=6000, nit_restart = 0):
    # set optimizer
    opt = nlopt.opt(nlopt.GN_ISRES, len(x0))
    res = results()
    prob = opt_problem(func, res,nit_restart, args)
    opt.set_ftol_abs(tol)
    opt.set_min_objective(prob.objfun)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);
    opt.set_population(len(x0))
    res.x = opt.optimize(x0)
    res.fun = opt.get_population()
    return res

def GN_CRS2_LM(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=20000, nit_restart = 1):
    # set optimizer
    opt = nlopt.opt(nlopt.GN_CRS2_LM, len(x0))
    res = results()
    prob = opt_problem(func, res,nit_restart, args)
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    opt.set_maxeval(maxtime)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);
    res.x = opt.optimize(x0)
    res.fun = opt.get_population()
    return res

def GN_DIRECT(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-7, maxtime=20000, nit_restart = 1):
    # set optimizer
    opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND_NOSCAL, len(x0))
    res = results()
    prob = opt_problem(func, res,nit_restart, args)
    #opt.set_ftol_abs(tol)
    #opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    opt.set_maxeval(maxtime)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);
    opt.set_population(len(x0)+1)
    res.x = opt.optimize(x0)
    res.fun = opt.get_population()
    return res

def GD_STOGO_RAND(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-7, maxtime=20000, nit_restart = 1):
    # set optimizer
    opt = nlopt.opt(nlopt.GD_STOGO_RAND, len(x0))
    res = results()
    prob = opt_problem(func, res,nit_restart, args)
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    opt.set_maxeval(maxtime)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);
    res.x = opt.optimize(x0)
    res.fun = opt.get_population()
    return res

def GN_MLSL(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-7, maxtime=20000, nit_restart = 1):
    # set optimizer
    opt = nlopt.opt(nlopt.GN_MLSL, len(x0))
    res = results()
    prob = opt_problem(func, res,nit_restart, args)
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    opt.set_maxeval(maxtime)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);
    res.x = opt.optimize(x0)
    res.fun = opt.get_population()
    return res

def GN_MLSL_LDS(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-7, maxtime=20000, nit_restart = 1):
    # set optimizer
    opt = nlopt.opt(nlopt.GN_MLSL_LDS, len(x0))
    res = results()
    prob = opt_problem(func, res,nit_restart, args)
    opt.set_ftol_abs(tol)
    opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    opt.set_maxeval(maxtime)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);
    res.x = opt.optimize(x0)
    res.fun = opt.get_population()
    return res

def GN_ORIG_DIRECT_L(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-7, maxtime=20000, nit_restart = 1):
    # set optimizer
    opt = nlopt.opt(nlopt.GN_ORIG_DIRECT_L, len(x0))
    res = results()
    prob = opt_problem(func, res,nit_restart, args)
    #opt.set_ftol_abs(tol)
    #opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    opt.set_maxeval(maxtime)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);
    opt.set_population(len(x0)+1)
    res.x = opt.optimize(x0)
    res.fun = opt.get_population()
    return res

def GN_AGS(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-7, maxtime=20000, nit_restart = 1):
    # set optimizer
    opt = nlopt.opt(nlopt.GN_AGS, len(x0))
    res = results()
    prob = opt_problem(func, res,nit_restart, args)
    #opt.set_ftol_abs(tol)
    #opt.set_xtol_abs(xtol)
    opt.set_min_objective(prob.objfun)
    opt.set_maxeval(maxtime)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);
    opt.set_population(len(x0)+1)
    res.x = opt.optimize(x0)
    res.fun = opt.get_population()
    return res

# gradient-based optimizers
#############################

def LD_SLSQP(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=60000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    opt = nlopt.opt(nlopt.LD_SLSQP, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);

    opt.set_ftol_rel(tol)
    opt.set_xtol_rel(xtol)
    opt.set_maxeval(maxtime)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    return res  

def LD_LBFGS(func, x0, args, ubnds=None, lbnds=None, tol=1e-6, xtol=1e-6, maxtime=60000, nit_restart = 0):
    # read Restart
    #x0 = readRestart(x0)
    # Set optimizer
    nlopt.LD_LBFGS()
    opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    res = results()
    prob = opt_problem(func, res, nit_restart, args)
    opt.set_lower_bounds(lbnds);
    opt.set_upper_bounds(ubnds);
    opt.set_ftol_rel(tol)
    opt.set_xtol_rel(xtol)
    opt.set_maxeval(maxtime)
    opt.set_min_objective(prob.objfun)
    res.x = opt.optimize(x0)
    res.fun = opt.last_optimum_value()
    return res
    
class results:
    def __init__(self):
        self.x = None
        self.fun = None
        self.nit = 0
        self.nfev = 0

class opt_problem:
    def __init__(self, func, res, nit_restart, args):
        self.func = func
        self.res = res
        self.args = args
        self.nit_restart = nit_restart

    def objfun(self, x, *_):
        self.res.nit += 1
        self.res.nfev += 1
        self.res.x = x
        f = self.func(x, *self.args)
        return f
