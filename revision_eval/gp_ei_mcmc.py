"""GP EI MCMC: Snoek, Larochelle & Adams, NeurIPS 2012, Eqs. 2, 5, 6.

Application choices (bounds, proper priors, sampling budget and safety loss) are
explicit. This is a NumPy/SciPy implementation of the published procedure, not
the authors' original software or an assertion of guaranteed global optimality.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import qmc

BOUNDS = np.array([[1.05, 2.0], [1.20, 3.0]])
PRIOR_MEAN = np.array([.5, np.log(.5), np.log(.5), np.log(.5), np.log(.02)])
PRIOR_SD = np.array([1., 1., 1., 1., 1.5])
# mean, log(signal SD), two log length scales, log(observation-noise SD)
HYPER_LIMITS = np.array([[-3., 3.], [-8., 3.], [-6., 3.], [-6., 3.], [-12., 1.]])


def calibration_loss(by_variant):
    rates = [by_variant[v]['prevention_rate'] for v in ('core', 'indexed')]
    if any(r is None or not 0 <= r <= 1 for r in rates):
        raise ValueError('Both calibration variants need positive controls and valid rates')
    worsening = sum(by_variant[v]['worsened'] for v in ('core', 'indexed'))
    return float(1 + worsening if worsening else 1 - sum(rates) / 2)


def winner(observations):
    safe = [o for o in observations if all(o['summary'][v]['worsened'] == 0 for v in ('core','indexed'))]
    if not safe:
        raise ValueError('No zero-worsening configuration')
    return min(safe, key=lambda o: (calibration_loss(o['summary']),
        sum(o['summary'][v]['proposed_capital_usd'] for v in ('core','indexed')),
        o['configuration'] != '1.10/1.20', o['trigger'], o['target']))


def normalize(x):
    return (np.asarray(x) - BOUNDS[:,0]) / np.diff(BOUNDS, axis=1).ravel()


def denormalize(x):
    return BOUNDS[:,0] + np.asarray(x) * np.diff(BOUNDS, axis=1).ravel()


def matern52(x, z, theta):
    distance = np.sqrt(np.sum(((x[:,None,:]-z[None,:,:])/np.exp(theta[2:4]))**2,axis=2))
    r = np.sqrt(5.) * distance
    return np.exp(2*theta[1]) * (1+r+r*r/3) * np.exp(-r)


def posterior_parts(x, y, theta):
    k = matern52(x,x,theta) + np.eye(len(x)) * (np.exp(2*theta[4])+1e-10)
    factor = cho_factor(k,lower=True,check_finite=False)
    residual = y-theta[0]
    alpha = cho_solve(factor,residual,check_finite=False)
    loglike = -.5*residual@alpha - np.log(np.diag(factor[0])).sum() - len(x)/2*np.log(2*np.pi)
    inverse = cho_solve(factor,np.eye(len(x)),check_finite=False)
    return float(loglike),alpha,inverse


def log_posterior(theta,x,y):
    if np.any(theta<HYPER_LIMITS[:,0]) or np.any(theta>HYPER_LIMITS[:,1]):return -np.inf
    try:
        like,_,_=posterior_parts(x,y,theta)
        return like-.5*np.sum(((theta-PRIOR_MEAN)/PRIOR_SD)**2)
    except (np.linalg.LinAlgError,ValueError,FloatingPointError):return -np.inf


def slice_sweep(theta,logp,rng,width=.5,max_steps=16):
    """Component-wise stepping-out and shrinking slice sampler."""
    theta=theta.copy()
    for dimension in rng.permutation(len(theta)):
        original=theta[dimension];height=logp(theta)+np.log(rng.uniform())
        left=original-width*rng.uniform();right=left+width
        j=int(rng.integers(max_steps));k=max_steps-1-j
        probe=theta.copy()
        while j:
            probe[dimension]=left
            if logp(probe)<=height:break
            left-=width;j-=1
        while k:
            probe[dimension]=right
            if logp(probe)<=height:break
            right+=width;k-=1
        for _ in range(1000):
            value=rng.uniform(left,right);probe=theta.copy();probe[dimension]=value
            if logp(probe)>=height:theta=probe;break
            if value<original:left=value
            else:right=value
        else:raise RuntimeError('Slice sampler failed to shrink')
    return theta


def sample_posterior(x,y,seed,burn=128,draws=48,thin=3):
    chains=[]
    for chain in range(2):
        rng=np.random.default_rng(seed+chain*100003)
        theta=PRIOR_MEAN.copy()+rng.normal(0,.1,size=5);theta[0]=float(np.clip(np.mean(y),-2.5,2.5))
        retained=[]
        for step in range(burn+draws*thin):
            theta=slice_sweep(theta,lambda p:log_posterior(p,x,y),rng)
            if step>=burn and (step-burn)%thin==0:retained.append(theta.copy())
        chains.append(np.array(retained))
    return np.array(chains)


def expected_improvement(mean,variance,best):
    sd=np.sqrt(np.maximum(variance,1e-16));improvement=best-mean;z=improvement/sd
    return np.maximum(0,improvement*ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi))


class IntegratedEI:
    def __init__(self,x,y,samples):
        self.x,self.y=x,y;self.parts=[]
        for theta in samples.reshape(-1,5):
            _,alpha,inverse=posterior_parts(x,y,theta);self.parts.append((theta,alpha,inverse))
    def __call__(self,queries):
        queries=np.atleast_2d(queries);answer=np.zeros(len(queries))
        for theta,alpha,inverse in self.parts:
            cross=matern52(queries,self.x,theta);mean=theta[0]+cross@alpha
            variance=np.exp(2*theta[1])-np.sum((cross@inverse)*cross,axis=1)
            answer+=expected_improvement(mean,variance,float(np.min(self.y)))
        answer/=len(self.parts)
        # Deterministic simulations cannot supply a new observation at a repeat.
        repeats=np.min(np.linalg.norm(queries[:,None,:]-self.x[None,:,:],axis=2),axis=1)<1e-5
        answer[repeats]=0
        return answer


def propose_next(observations,seed,burn=128,draws=48,thin=3):
    x=normalize([[o['trigger'],o['target']] for o in observations])
    y=np.array([calibration_loss(o['summary']) for o in observations])
    samples=sample_posterior(x,y,seed,burn,draws,thin);acquisition=IntegratedEI(x,y,samples)
    candidates=qmc.Sobol(2,scramble=True,seed=seed).random_base2(9)
    candidates=np.vstack([candidates,[[0,0],[0,1],[1,0],[1,1]]])
    feasible=lambda q: denormalize(q)[1]-denormalize(q)[0]-1e-6
    candidates=candidates[np.array([feasible(q)>=0 for q in candidates])]
    values=acquisition(candidates);best=int(np.argmax(values));point=candidates[best];value=float(values[best])
    # Multi-start continuous acquisition optimization; Sobol points are starts,
    # not configurations that are sent to the simulator.
    for start in candidates[np.argsort(values)[-8:]]:
        fitted=minimize(lambda q:-float(acquisition(q)[0]),start,method='SLSQP',
            bounds=[(0.,1.),(0.,1.)],constraints=[dict(type='ineq',fun=feasible)],
            options={'maxiter':60,'ftol':1e-9})
        if feasible(fitted.x)>=0:
            candidate_value=float(acquisition(fitted.x)[0])
            if candidate_value>value:point,value=fitted.x,candidate_value
    trigger,target=np.round(denormalize(point),6)
    if trigger>=target:raise ValueError('Acquisition produced invalid thresholds')
    if any(abs(trigger-o['trigger'])<1e-6 and abs(target-o['target'])<1e-6 for o in observations):
        raise ValueError('Acquisition repeated an existing configuration')
    return dict(trigger=float(trigger),target=float(target),configuration=f'{trigger:.6f}/{target:.6f}',
        expected_improvement=value,seed=seed,observations=len(observations),
        posterior_samples=samples.tolist(),model='ARD Matern 5/2 GP; two component-wise slice chains',
        burn_per_chain=burn,draws_per_chain=draws,thin=thin,
        prior_mean=PRIOR_MEAN.tolist(),prior_sd=PRIOR_SD.tolist(),hyperparameter_limits=HYPER_LIMITS.tolist(),
        bounds=BOUNDS.tolist(),method='Snoek et al. 2012 GP EI MCMC; equations 2, 5, 6')
