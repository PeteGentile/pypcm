#!/usr/bin/env python
from datetime import datetime
tstart = datetime.now()
import matplotlib.pyplot as plt
import numpy as np
from explore_polarization import get_stokes_portrait
from sys import argv
from tools import align_profiles
import scipy.optimize as optimize
import pyfits, pickle, sys, linecache

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)

def align_portraits(portraits, shift = None):
#This takes in two NxM data arrays (Freq, subint vs profile bin)
#scrunches them into profiles, uses align_profiles to align them,
#then returns the aligned NxM arrays.
#Alternatively, you can tell it how much to shift the arrays.
#align_profiles cross-correlates the profiles to align them
	profile1 = np.sum(portraits[0], axis = 0)
	profile2 = np.sum(portraits[1], axis = 0)

	if shift == None:
		profiles, shift = align_profiles([profile1, profile2])
		real_shift = shift[1]%len(profile1)
		shifted_portrait1 = portraits[0]
		shifted_portrait2 = np.roll(portraits[1], -real_shift, axis = 1)
		profile1 = np.roll(profile1, shift[0])
		prof_diffs = profile1-profile2
		return [shifted_portrait1, shifted_portrait2], -shift[1]
	else:
		shifted_portrait1 = np.roll(portraits[0], shift, axis = 1)
		shifted_portrait2 = portraits[1]
		profile1 = np.roll(profile1, shift)
		prof_diffs = profile1-profile2
		return [shifted_portrait1, shifted_portrait2]

def align_profs(portraits, shift = None):
#Calls 'align_profiles' in a smart way (i.e. giving 
#the user the option to hard code a shift amount.
	profile1, profile2 = portraits
	
	if shift == None:
		profiles, shift = align_profiles([profile1, profile2])
		return profiles, -shift[1]
	else:
		profile2 = np.roll(profile2, shift,axis=1)
		return [profile1, profile2]

def center_prof(profs):
	i, q, u, v, w = profs
	max_index = np.argmax(i[0])
	nbins = len(i[0])
	cshift = -(max_index - nbins/2)
	
	return np.roll(i, cshift,axis=1), np.roll(q, cshift,axis=1), np.roll(u, cshift,axis=1), np.roll(v, cshift,axis=1), np.roll(w, cshift,axis=1)

def scale_portraits(portraits):
#Scales portraits (NxM data arrays) so that the total
#area underneath the subsequent summed profiles would be the same
	factor = np.sum(portraits[0]) / np.sum(portraits[1])
	scaled = []
	scaled.append(portraits[0])
	scaled.append(portraits[1] * factor)
	return scaled
	
def make_weights(portrait):
#For freq vs profile bin data, returns an array whose elements
#are 0 if the corresponding channel is zapped and 1 if not.
	i_vs_freq = np.sum(portrait, axis = 1)
	weights = i_vs_freq
	for i, val in enumerate(i_vs_freq):
		if val:
			weights[i] = 1
		else:
			pass
	return weights

def reduce_portraits(portraits):
#Takes in NxM data arrays, aligns them, then zaps them.
	aligned_ps = align_portraits(scale_portraits(portraits))
	p1, p2 = aligned_ps
	weights1 = make_weights(p1)
	weights2 = make_weights(p2)
	weights = weights1*weights2
	final_portrait1 = (weights * p1.T).T
	final_portrait2 = (weights * p2.T).T
	return final_portrait1, final_portrait2

def get_on_pulse_phases(profile):
#Determines the indices of a profile that are above some threshold, and
#returns those indices, and the corresponding pulse intensities in those phase bins
	level = np.average(profile) + np.std(profile)/np.max(profile)
	inds = np.where(profile > level)[0]
	pulse = profile[inds]
	return inds, pulse

def interpolate_profs(data, factor):
#Uses numpy's interp function (linear interpolation) to interpolate
#each channel of freq vs phase bin data, then returns the interpolated array
	phases = np.linspace(0,data.shape[1], data.shape[1])
	new_phases = np.linspace(0,data.shape[1], data.shape[1]*factor)
	new_data = []
	for i in range(len(data[:,])):
		new_data.append(np.interp(new_phases, phases, data[i,:]))

	new_data = np.asarray(new_data)
	return new_data

def prepare_data(standard, data):
#This function takes in two freq vs phase bin arrays, a standard array and a data array.
#The function aligns the data array to the standard array, determines the invariant interval
#for each channel in both arrays, weights the data array so that the inv. int. is the same for both,
#Zaps channels in the standard and data arrays so that both arrays have exactly the same zapped channels.
#Then returns the aligned, scaled, zapped data, as well as channel weights.
	s_is, s_qs, s_us, s_vs, s_weights = standard
	p_is, p_qs, p_us, p_vs, p_weights = data	
	s_is, s_qs, s_us, s_vs, s_weights = center_prof([s_is, s_qs, s_us, s_vs, s_weights])
	
	#Make a total profile for the template and data
	t_prof = np.sum(s_is, axis=0)
	data_prof = np.sum(p_is, axis=0)

	#Scale the profiles so that the total intensities are the same, then align them
	scale_factor = np.sum(t_prof)/np.sum(data_prof)
	aligned_profs, shift = align_profs([t_prof, scale_factor*data_prof])
	p_is, p_qs, p_us, p_vs = [np.roll(x,shift, axis=1) for x in [p_is, p_qs, p_us, p_vs]]

	#Make the invariant intervals for the total template and data files
	data_invariant_intervals = p_is**2 - (p_qs**2 + p_us**2 + p_vs**2)
	template_invariant_intervals = s_is**2 - (s_qs**2 + s_us**2 + s_vs**2)

	#THIS CODE ONLY MAKES SENSE FOR 1713 AT LBAND WITH 2048 PROFILE BINS
	#The bin numbers in ^this case are 1018:1126
	'''if src == "1713":
		template_chan_weights = np.sum(template_invariant_intervals[:,1018:1126], axis=1)
		data_chan_weights = np.sum(data_invariant_intervals[:,1018:1126], axis=1)
	elif src == "1937":
		template_chan_weights = np.sum(template_invariant_intervals[:,1775:1915], axis=1)
		data_chan_weights = np.sum(data_invariant_intervals[:,1775:1915], axis=1)
	else:
		print "Source not recognized."
		exit(0)'''
	
	on_indices = get_on_indices([p_is, p_qs, p_us, p_vs, s_is, s_qs, s_us, s_vs], num = len(t_prof)/20)
	off_indices = get_off_indices(np.sum(p_is, axis=0), np.sum(s_is, axis=0))
	
	template_chan_weights = np.sum(template_invariant_intervals[:,on_indices], axis=1)
	data_chan_weights = np.sum(data_invariant_intervals[:,on_indices], axis=1)
	
	#Find out how bright each ON PULSE REGION of the invariant interval is for each freq channel

	#Scale each Freq channel so that the invariant intervals have the same area underneath for the template and data
	scale_factors = np.nan_to_num(np.true_divide(template_chan_weights,data_chan_weights))

	#When data chan = 0, this will return nan, so make scale_factors 0 in this case
	scale_factors[data_chan_weights<=0] = 0
	
	#Since the InvInt is a squared factor, we need to multiply each element of the data arrays by the 
	#square root of the scale factors. Note that this will also zap all data channels that are zapped
	#in the satndard data.
	scale_factors = scale_factors**0.5

	#Make another mask array to zap the standard channels that are zapped in the data array
	template_zap = np.copy(scale_factors)
	template_zap[template_zap>0] = 1

	#Turn the 1x512 scale_factors array into a 2048X512 array, called mask array
	trash, mask_array = np.meshgrid(np.arange(2048),scale_factors.squeeze())
	trash, zap_array = np.meshgrid(np.arange(2048),template_zap.squeeze())

	#Actually apply the mask to the data
	scaled_dis, scaled_dqs, scaled_dus, scaled_dvs = [x*mask_array for x in [p_is, p_qs, p_us, p_vs]]

	#Zero weight template channels that are zapped in the data file
	template_is, template_qs, template_us, template_vs = [x*zap_array for x in [s_is, s_qs, s_us, s_vs]]
	
	#Return the calculated values
	return scaled_dis, scaled_dqs, scaled_dus, scaled_dvs, template_is, template_qs, template_us, template_vs, template_zap.squeeze()
	
def get_on_indices(data, method='pol', num=100):
#This return the top "num" phase bins. Note that this does not choose the top
#bins in a particular data set, but rather from the product of two datasets.
#Hopefully, this will mitigate the code from choosing anomalous spikes in unclean data.
	p_is, p_qs, p_us, p_vs, s_is, s_qs, s_us, s_vs = [np.sum(x, axis=0) for x in data]
	if method == 'pol':
		p_pol = p_qs**2 + p_us**2 + p_vs**2
		s_pol = s_qs**2 + s_us**2 + s_vs**2
		total_inten = p_pol*s_pol
	elif method == 'intensity':
		total_inten = p_is*s_is

	return np.argpartition(-total_inten,num)[:num]

def get_off_indices(intens1, intens2, window=100):
#Stolen from M Lam. Credit to him.
	integral = np.zeros_like(intens1)
	nbins = len(intens1)
	data = intens1*intens2
	for i in range(nbins):
		win = np.arange(i-window/2,i+window/2) % nbins
		integral[i] = np.trapz(data[win])

	minind = np.argmin(integral)
	opw = np.arange(minind-window/2,minind+window/2+1) % nbins
	return opw

def receiver_transform(params, stokes):
#Poorly named, perhaps, but this function takes in receiver params
	theta1, theta2, eps1, eps2, phi, gamma = params
	A = eps1*np.cos(theta1) + eps2*np.cos(theta2)
	B = eps1*np.sin(theta1) + eps2*np.sin(theta2)
	C = eps1*np.cos(theta1) - eps2*np.cos(theta2)
	D = eps1*np.sin(theta1) - eps2*np.sin(theta2)
	E = gamma/2.0
	F = np.cos(phi)
	G = np.sin(phi)
	H = 1
	I = 0
	resids = 0
	Mamp = np.asarray([[1, E, 0, 0], [E, 1, 0, 0], [0, 0, F, -G], [0, 0, G, F]])
	Mcc = np.asarray([[1,0,A,B],[0, 1, C, D],[A, -C, 1, 0],[B, -D, 0, 1]])
	Mfeed = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	#mueller = np.asarray([1, E, A+E*C, B+E*D, E, H, A+E*C, E*B+D, A*F-G*B, G*D-F*C, F, -G, A*G+B*F, -G*C-F*D, G, F*H]).reshape(4,4)
	mueller = np.dot(np.dot(Mamp, Mcc), Mfeed)
	#imuell = np.linalg.inv(mueller)
	data_is, data_qs, data_us, data_vs, template_is, template_qs, template_us, template_vs  = stokes
	for i in range(len(data_is)):
		S_int = np.asarray([template_is[i], template_qs[i], template_us[i], template_vs[i]]).reshape(4,1)
		S_meas = np.asarray([data_is[i], data_qs[i], data_us[i], data_vs[i]]).reshape(4,1)
		resids += np.sum((np.dot(mueller,S_int) - S_meas)**2)
	 
	return resids

def receiver_func(stokes_intrinsic, theta1, theta2, eps1, eps2, phi, gamma):
	A = eps1*np.cos(theta1) + eps2*np.cos(theta2)
	B = eps1*np.sin(theta1) + eps2*np.sin(theta2)
	C = eps1*np.cos(theta1) - eps2*np.cos(theta2)
	D = eps1*np.sin(theta1) - eps2*np.sin(theta2)
	E = gamma/2.0
	F = np.cos(phi)
	G = np.sin(phi)
	H = 1
	I = 0
	Mamp = np.asarray([[1, E, 0, 0], [E, 1, 0, 0], [0, 0, F, -G], [0, 0, G, F]])
	Mcc = np.asarray([[1,0,A,B],[0, 1, C, D],[A, -C, 1, 0],[B, -D, 0, 1]])
	Mfeed = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	mueller = np.dot(np.dot(Mamp, Mcc), Mfeed)
	n_on_bins = len(stokes_intrinsic)/4
	data_is, data_qs, data_us, data_vs  = stokes_intrinsic.reshape((4,n_on_bins))
	out_is = []
	out_qs = []
	out_us = []
	out_vs = []
	n_data_points = 4*len(data_is)
	for i in range(len(data_is)):
		S_int = np.asarray([data_is[i], data_qs[i], data_us[i], data_vs[i]]).reshape(4,1)
		S_out = np.dot(mueller,S_int)
		out_is.append(S_out[0])
		out_qs.append(S_out[1])
		out_us.append(S_out[2])
		out_vs.append(S_out[3])
	out_is = np.asarray(out_is)
	out_qs = np.asarray(out_qs)
	out_us = np.asarray(out_us)
	out_vs = np.asarray(out_vs)
	out_stokes = np.asarray([out_is, out_qs, out_us, out_vs])
	return out_stokes.squeeze().reshape((n_data_points,))

def calibrate(params, stokes):
	#stokes is i,q,u,v for a whole channel
	theta1, theta2, eps1, eps2, phi, gamma = params
	A = eps1*np.cos(theta1) + eps2*np.cos(theta2)
	B = eps1*np.sin(theta1) + eps2*np.sin(theta2)
	C = eps1*np.cos(theta1) - eps2*np.cos(theta2)
	D = eps1*np.sin(theta1) - eps2*np.sin(theta2)
	E = gamma/2.0
	F = np.cos(phi)
	G = np.sin(phi)
	H = 1
	I = 0

	mueller = np.asarray([1, E, A+E*C, B+E*D, E, H, A+E*C, E*B+D, A*F-G*B, G*D-F*C, F, -G, A*G+B*F, -G*C-F*D, G, F*H]).reshape(4,4)
	inv_mueller = np.linalg.inv(mueller)
	nbins = len(stokes[0])
	data_is, data_qs, data_us, data_vs = stokes
	calibrated_is, calibrated_qs, calibrated_us, calibrated_vs  = [[],[],[],[]]
	stokes_vectors = np.asarray([data_is, data_qs, data_us, data_vs]).reshape(4,nbins)
	calibrated_stokes = np.dot(inv_mueller,stokes_vectors)
	#print calibrated_stokes[0].shape
	'''for i,q,u,v in zip(data_is.squeeze(), data_qs.squeeze(), data_us.squeeze(), data_vs.squeeze()):
		stokes_vector = np.asarray([i,q,u,v]).reshape(4,1)
		#ci, cq, cu, cv = inv_mueller*stokes_vector
		result = np.dot(inv_mueller,stokes_vector)
		print result[0].shape, stokes_vector.shape, "lol"
		calibrated_is.append(ci)
		calibrated_qs.append(cq)
		calibrated_us.append(cu)
		calibrated_vs.append(cv)

	print len(calibrated_is)
	
	return [calibrated_is, calibrated_qs, calibrated_us, calibrated_vs]'''
	return calibrated_stokes

def get_mtm_solution(standard_name, data_name, verbose=True, fake=False):
	standard = get_stokes_portrait(standard_name, 69, fscr = False, tscr = True, title_str = title, plot = False, save = False, ret = True)
	data = get_stokes_portrait(data_name, 69, fscr = False, tscr = True, title_str = title, plot = False, save = False, ret = True)

	p_is, p_qs, p_us, p_vs, s_is, s_qs, s_us, s_vs, weights = prepare_data(standard, data)

	on_indices = get_on_indices([p_is, p_qs, p_us, p_vs, s_is, s_qs, s_us, s_vs])
	off_indices = get_off_indices(np.sum(p_is, axis=0), np.sum(s_is, axis=0))

	data_invariant_intervals = p_is**2 - (p_qs**2 + p_us**2 + p_vs**2)
	template_invariant_intervals = s_is**2 - (s_qs**2 + s_us**2 + s_vs**2)
	receiver_params = []
	receiver_param_errs = []
	valid_chans = []
	calibrated_is, calibrated_qs, calibrated_us, calibrated_vs = [[], [], [], []]
	nbins = len(p_is[0,:])
	n_on_bins = len(on_indices)
	n_data_points = 4*n_on_bins

	tstart = datetime.now()
	nfails = 0
	blah = -500.0
	for i in range(len(s_is[:,0])):
		if fake:
			calibrated_is.append([0]*nbins)
			calibrated_qs.append([0]*nbins)
			calibrated_us.append([0]*nbins)
			calibrated_vs.append([0]*nbins)
			receiver_params.append([0.0,0.0,0.0,0.0,0.0,0.0])
			receiver_param_errs.append([1.0,1.0,1.0,1.0,1.0,1.0])
			valid_chans.append(i)
		else:
			if weights[i]:
				stokes_int = np.asarray([s_is[i,on_indices], s_qs[i,on_indices], s_us[i,on_indices], s_vs[i,on_indices]]).reshape((n_data_points,))
				stokes_out = np.asarray([p_is[i,on_indices], p_qs[i,on_indices], p_us[i,on_indices], p_vs[i,on_indices]]).reshape((n_data_points,))
				p_err = np.std(p_is[i,off_indices])
				s_err = np.std(s_is[i,off_indices])
				if p_err>s_err: prof_err = p_err
				else: prof_err = s_err
				p0 = [0,0,0,0,0,0]
				try:
					popt, pcov = optimize.curve_fit(receiver_func, stokes_int, stokes_out, bounds = ((-np.pi/2.0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, -np.pi, -np.pi/2.0),(np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi, np.pi/2.0)))
					receiver_params.append(popt)
					resids = np.asarray(stokes_out) - np.asarray(receiver_func(stokes_int, *popt))
					redchisqs = np.sum(resids**2)/((len(on_indices)*4 - len(popt))*(prof_err**2))
					receiver_param_errs.append([pcov[0,0]*redchisqs, pcov[1,1]*redchisqs, pcov[2,2]*redchisqs, pcov[3,3]*redchisqs, pcov[4,4]*redchisqs, pcov[5,5]*redchisqs])
					if verbose:
						if i<10: print "Channel  ", i, "solved,", redchisqs
						elif i<100: print "Channel ", i, "solved,", redchisqs
						else: print "Channel", i, "solved,", redchisqs
					cal = calibrate(popt, [p_is[i,:], p_qs[i,:], p_us[i,:], p_vs[i,:]])
					#print "Chan", i, "lendata:", len(p_is[i,:]), "lencal:", len(cal[0])
					calibrated_is.append(cal[0])
					calibrated_qs.append(cal[1])
					calibrated_us.append(cal[2])
					calibrated_vs.append(cal[3])
					valid_chans.append(i)
				except Exception as e:
					#PrintException()
					if verbose:
						if i<10: print "Channel  ", i, "failed:", e
						elif i<100: print "Channel ", i, "failed:", e
						else: print "Channel", i, "failed:", e
					calibrated_is.append([0]*nbins)
					calibrated_qs.append([0]*nbins)
					calibrated_us.append([0]*nbins)
					calibrated_vs.append([0]*nbins)
					receiver_params.append([0,0,0,0,0,0])
					receiver_param_errs.append([0,0,0,0,0,0])
					nfails +=1
	
			else:
				if verbose:
					if i<10: print "Channel  ", i, "skipped"
					elif i<100: print "Channel ", i, "skipped"
					else: print "Channel", i, "skipped"
				calibrated_is.append([0]*nbins)
				calibrated_qs.append([0]*nbins)
				calibrated_us.append([0]*nbins)
				calibrated_vs.append([0]*nbins)
				receiver_params.append([0,0,0,0,0,0])
				receiver_param_errs.append([0,0,0,0,0,0])


			tdone = datetime.now()
		if verbose: print "Solved and calibrated in", tdone-tstart
		if nfails: print "Solve failed for", nfails, "channels"
		else: print "Generated MTM solution with no failed channels."

	calibrated_is = np.asarray(calibrated_is)
	calibrated_qs = np.asarray(calibrated_qs)
	calibrated_us = np.asarray(calibrated_us)
	calibrated_vs = np.asarray(calibrated_vs)
	new = np.asarray(calibrated_is)

	receiver_params = np.asarray(receiver_params)
	receiver_param_errs = np.asarray(receiver_param_errs)
	print receiver_params.shape
	return np.asarray([s_is, s_qs, s_us, s_vs]), np.asarray([p_is, p_qs, p_us, p_vs]), np.asarray([calibrated_is, calibrated_qs, calibrated_us, calibrated_vs]), valid_chans, receiver_params,receiver_param_errs

def write_pcm(sfname, receiver_params, valid_chans, outname = "pcm.fits", fake=False):
	shdulist = pyfits.open(sfname)
	head = shdulist[0].header
	head['OBS_MODE'] = "PCM"
	nchan = head['OBSNCHAN']
	prihdu = pyfits.PrimaryHDU(header=head)
	new_hdulist = pyfits.BinTableHDU(header=shdulist[1].header, data=shdulist[1].data,name=shdulist[1].name)
	chans = np.linspace(head['OBSFREQ']-head['OBSBW']/2.0, head['OBSFREQ']+head['OBSBW']/2.0, nchan, dtype='d')
	#receiver_params[:,[0,5]] = receiver_params[:,[5,0]]
	#receiver_params[:,[1,2]] = receiver_params[:,[2,1]]
	#receiver_params[:,[2,3]] = receiver_params[:,[3,2]]
	#receiver_params[:,[3,4]] = receiver_params[:,[3,4]]
	#receiver_params[:,5] = np.true_divide(receiver_params[:,5],10.0)
	weights = np.zeros(nchan)
	weights[valid_chans] = 1
	Gs = weights.reshape(nchan,1)
	expanded_receivers = np.zeros(nchan*6).reshape(nchan,6)
	'''for i in chanlist:	
		if i in valid_chans:
			expanded_receivers[i] = receiver_params[vchan,:]
			vchan -= 1
	
	'''
	vchan = 0
	for i in range(nchan):	
		if i in valid_chans:
			expanded_receivers[i] = receiver_params[vchan,[5,4,2,0,3,1]]
			vchan += 1
			
	params = np.append(Gs,expanded_receivers,axis=1)
	if not fake:
		params[params==0] = np.NaN
	ncovar = 28
	ncov = ncovar*nchan
	covars = np.ones(ncov)*0.0005
	write_params = np.concatenate(params)
	chisqs = np.ones(nchan)
	nfree = np.copy(weights)
	chans = np.asarray([chans])
	weights = np.asarray([weights])
	write_params = np.asarray([write_params])
	covars = np.asarray([covars])
	chisqs = np.asarray([chisqs])
	nfree = np.asarray([nfree])
	
	freqcol = pyfits.Column(name = "DAT_FREQ", format="512D", array = chans)
	wtcol = pyfits.Column(name = "DAT_WTS", format="512E", array = weights)
	datcol = pyfits.Column(name = "DATA", format="3584E", array = write_params)
	covcol = pyfits.Column(name = "COVAR", format="14336E", array = covars)
	chicol = pyfits.Column(name = "CHISQ", format="512E", array = chisqs)
	freecol = pyfits.Column(name = "NFREE", format="512J", array = nfree)
	
	cols = pyfits.ColDefs([freqcol, wtcol, datcol, covcol, chicol, freecol])
	
	feed_hdu = pyfits.BinTableHDU.from_columns(cols, name="FEEDPAR")
	feed_hdu.header.comments['TTYPE1'] = '[MHz] Centre frequency for each channel'
	feed_hdu.header.comments['TFORM1'] = 'NCHAN doubles'
	feed_hdu.header.comments['TTYPE2'] = 'Weights for each channel'
	feed_hdu.header.comments['TFORM2'] = 'NCHAN floats'
	feed_hdu.header.comments['TTYPE3'] = 'Cross-coupling data'
	feed_hdu.header.comments['TFORM3'] = 'NCPAR*NCHAN floats'
	feed_hdu.header.comments['TTYPE4'] = 'Formal covariances of coupling data'
	feed_hdu.header.comments['TFORM4'] = 'NCOVAR*NCHAN floats'
	feed_hdu.header.comments['TTYPE5'] = 'Total chi-squared (objective merit function)'
	feed_hdu.header.comments['TFORM5'] = 'NCHAN floats'
	feed_hdu.header.comments['TTYPE6'] = 'Number of degrees of freedom'
	feed_hdu.header.comments['TFORM6'] = 'NCHAN long (32-bit) integers'
	feed_hdu.header['CAL_MTHD'] = ('van04e18', 'Cross-coupling method')
	feed_hdu.header['NCPAR'] = ('7', 'Number of coupling parameters')
	feed_hdu.header['NCOVAR'] = ('28', 'Number of parameter covariances')
	feed_hdu.header['NCHAN'] = ('512', 'Nr of channels in Feed coupling data')
	feed_hdu.header['EPOCH'] = ('56038.3352', '[MJD] Epoch of calibration obs')
	feed_hdu.header['TUNIT1'] = ('MHz', 'Units of field')
	feed_hdu.header['TDIM3'] = ('(7,512)', 'Dimensions (NCPAR,NCHAN)')
	feed_hdu.header['TDIM4'] = ('(28,512)', 'Dimensions (NCOVAR,NCHAN)')
	feed_hdu.header['EXTVER'] = ('1', 'auto assigned by template parser')
	feed_hdu.header['PAR_0000'] = ('G', 'scalar gain')
	feed_hdu.header['PAR_0001'] = ('gamma', 'differential gain (hyperbolic radians)')
	feed_hdu.header['PAR_0002'] = ('phi', 'differential phase (radians)')
	feed_hdu.header['PAR_0003'] = ('el0', 'ellipticity of receptor 0 (radians)')
	feed_hdu.header['PAR_0004'] = ('or0', 'orientation of receptor 0 (radians)')
	feed_hdu.header['PAR_0005'] = ('el1', 'ellipticity of receptor 1 (radians)')
	feed_hdu.header['PAR_0006'] = ('or1', 'orientation of receptor 1 (radians)')
	hdus = pyfits.HDUList(hdus=[prihdu, new_hdulist, feed_hdu])
	hdus.writeto(outname, clobber=True)

def clean_receiver_params(params,errs,caldata,vchans):
	cparams = []
	cerrs = []
	cvchans = []
	cis, cqs, cus, cvs = caldata
	ccis = []
	ccqs = []
	ccus = []
	ccvs = []
	ccaldata = []
	zap = False
	ngood = 0	
	for i in range(len(errs)):
		if (errs[i,:-1] > 5.5).sum() > 0.5 or errs[i,-1] > 1:
			zap = True
		
		if zap:
			cparams.append([0,0,0,0,0,0])
			cerrs.append([0,0,0,0,0,0])
			ccis.append(np.zeros_like(cis[i]))
			ccqs.append(np.zeros_like(cqs[i]))
			ccus.append(np.zeros_like(cus[i]))
			ccvs.append(np.zeros_like(cvs[i]))
			print "Zapped channel", i
		else:
			cparams.append(params[i])
			cerrs.append(errs[i])
			ccis.append(cis[i])
			ccqs.append(cqs[i])
			ccus.append(cus[i])
			ccvs.append(cvs[i])
		if np.sum(cparams[i]):
			cvchans.append(i)
		zap = False
	
	#cvchans = np.copy(errs[:,0])
	#cvchans[cvchans!=0] = 1
	#cvchans[cvchans==0] = 0
	cparams = np.asarray(cparams)
	cerrs = np.asarray(cerrs)
	cvchans = np.asarray(cvchans)
	ccaldata = np.asarray([ccis, ccqs, ccus, ccvs])
	return cparams, cerrs, ccaldata, cvchans

if __name__ == "__main__":
	args = argv[1:]
	index = 69
	fscrunch = False
	tscrunch = False
	save_image = False
	overlay_profs = False
	seperate_stokes = False
	plot = True
	plotpa = False
	title = "Stokes Portrait"
	files = []
	verbose = False
	usepickle = True
	noplot = True
	cfreq = 1380
	fake=False
	for i, arg in enumerate(args):
		if arg == "-f":
			files = args[i+1:]
		elif arg == "-S":
			standard_name = args[i+1]
		elif arg == "-noplot":
			plot = False
		elif arg in ["-plotpa", "-plot_pa", "-plotpas", "-plot_pas"]:
			plotpa = True
		elif arg == "-v":
			verbose = True
		elif arg == '-nopickle':
			usepickle = False
		elif arg == '-o':
			outname = args[i+1]
		elif arg == "-plot":
			noplot=False
		elif arg == "-fake" or arg == "-makefake":
			fake = True
		elif arg in ["-freq","-cfreq"]:
			cfreq = float(args[i+1])

	#elif len(files) != 2:
	#	print "You must pecify two and only two files to compare!"
	#	exit(0)

	#try:
	factor = 4
	prof_diffs = []
	mjds = []
	std_statistics = []
	if fake:
		prepd_s_data, prepd_p_data, calibrated_data, valid_chans, receiver_params, receiver_param_errs = get_mtm_solution(files[0], files[1], verbose, fake=True)
		
		write_pcm(files[0], receiver_params, valid_chans, "identity.pcm", fake=fake)
		
		'''receiver_params, receiver_param_errs, calibrated_data, valid_chans = clean_receiver_params(receiver_params, receiver_param_errs, calibrated_data, valid_chans)
		s_is, s_qs, s_us, s_vs = prepd_s_data
		p_is, p_qs, p_us, p_vs = prepd_p_data
		calibrated_is, calibrated_qs, calibrated_us, calibrated_vs = calibrated_data
		bw = 800.0
		nchan = len(np.sum(s_is,axis=1))
		chan_bw = 800.0/nchan
		calibrated_is, calibrated_qs, calibrated_us, calibrated_vs = calibrated_data
		freqs = np.linspace(cfreq+400-(chan_bw/2.0),cfreq-400+(chan_bw/2.0),nchan)
		
		outpickle_name = ".".join(files[-1].split(".") + ["pickle"])
		outcal = ".".join(files[-1].split(".") + ["calibpickle"])
		outpcm = ".".join(files[-1].split(".") + ["pcmpickle"])
		with  open(outpickle_name, "wb") as pfile:
			pickle.dump([prepd_s_data, prepd_p_data, calibrated_data, freqs, receiver_params, receiver_param_errs], pfile, protocol=pickle.HIGHEST_PROTOCOL)
		with open(outcal, "wb") as pfile:
			pickle.dump([prepd_s_data, prepd_p_data, calibrated_data, freqs], pfile, protocol=pickle.HIGHEST_PROTOCOL)
		with open(outpcm, "wb") as pfile:
			pickle.dump([receiver_params, receiver_param_errs, freqs], pfile, protocol=pickle.HIGHEST_PROTOCOL)'''
		exit(0)

	if usepickle:
		try:
			with open("puppi_56165_1713+0747_0708.TT.FR.pickle", "rb") as f:
				loaded = pickle.load(f)
				print len(loaded)
				prepd_s_data, prepd_p_data, calibrated_data, valid_chans, receiver_params, receiver_param_errs = loaded
				f.close()
		except IOError:
			prepd_s_data, prepd_p_data, calibrated_data, valid_chans, receiver_params, receiver_param_errs = get_mtm_solution(files[0], files[1], verbose)
		s_is, s_qs, s_us, s_vs = prepd_s_data
		p_is, p_qs, p_us, p_vs = prepd_p_data
		bw = 800.0
		nchan = len(np.sum(s_is,axis=1))
		chan_bw = 800.0/nchan
		calibrated_is, calibrated_qs, calibrated_us, calibrated_vs = calibrated_data
		freqs = np.linspace(cfreq+400-(chan_bw/2.0),cfreq-400+(chan_bw/2.0),nchan)
		outpickle_name = ".".join(files[-1].split(".") + ["pickle"])
		outpickle = open(outpickle_name, "wb")
		pickle.dump([prepd_s_data, prepd_p_data, calibrated_data, freqs, receiver_params], outpickle, protocol=pickle.HIGHEST_PROTOCOL)
		outpickle.close()
	else:
		prepd_s_data, prepd_p_data, calibrated_data, valid_chans, receiver_params, receiver_param_errs = get_mtm_solution(files[0], files[1], verbose)
		receiver_params, receiver_param_errs, calibrated_data, valid_chans = clean_receiver_params(receiver_params, receiver_param_errs, calibrated_data, valid_chans)
		s_is, s_qs, s_us, s_vs = prepd_s_data
		p_is, p_qs, p_us, p_vs = prepd_p_data
		calibrated_is, calibrated_qs, calibrated_us, calibrated_vs = calibrated_data
		bw = 800.0
		nchan = len(np.sum(s_is,axis=1))
		chan_bw = 800.0/nchan
		calibrated_is, calibrated_qs, calibrated_us, calibrated_vs = calibrated_data
		freqs = np.linspace(cfreq+400-(chan_bw/2.0),cfreq-400+(chan_bw/2.0),nchan)
		
		outpickle_name = ".".join(files[-1].split(".") + ["pickle"])
		outcal = ".".join(files[-1].split(".") + ["calibpickle"])
		outpcm = ".".join(files[-1].split(".") + ["pcmpickle"])
		with  open(outpickle_name, "wb") as pfile:
			pickle.dump([prepd_s_data, prepd_p_data, calibrated_data, freqs, receiver_params, receiver_param_errs], pfile, protocol=pickle.HIGHEST_PROTOCOL)
		with open(outcal, "wb") as pfile:
			pickle.dump([prepd_s_data, prepd_p_data, calibrated_data, freqs], pfile, protocol=pickle.HIGHEST_PROTOCOL)
		with open(outpcm, "wb") as pfile:
			pickle.dump([receiver_params, receiver_param_errs, freqs], pfile, protocol=pickle.HIGHEST_PROTOCOL)
	
	write_pcm(files[0], receiver_params, valid_chans, outname, fake=fake)

	if noplot: exit(0)
	nplots = 0
	if nplots == 0:
		nplots = 4
		phiplot = True
		gammaplot = True
		epsilonplot = True
		thetaplot = True

	f, axarr = plt.subplots(nplots, sharex=True, squeeze = False)
	i = 0
	plot = 0
	if thetaplot:
		axarr[i,plot].errorbar(freqs[valid_chans], [x*180.0/np.pi for x in receiver_params[valid_chans,0]], yerr = [x*180.0/np.pi for x in receiver_param_errs[valid_chans,0]], fmt='r_')
		axarr[i,plot].errorbar(freqs[valid_chans], [x*180.0/np.pi for x in receiver_params[valid_chans,1]], yerr = [x*180.0/np.pi for x in receiver_param_errs[valid_chans,1]], fmt='k_')
		axarr[i,plot].set_ylabel("Theta")
		i+=1
	if epsilonplot:
		axarr[i,plot].errorbar(freqs[valid_chans], [x*180.0/np.pi for x in receiver_params[valid_chans,2]], yerr = [x*180.0/np.pi for x in receiver_param_errs[valid_chans,2]], fmt='r_')
		axarr[i,plot].errorbar(freqs[valid_chans], [x*180.0/np.pi for x in receiver_params[valid_chans,3]], yerr = [x*180.0/np.pi for x in receiver_param_errs[valid_chans,3]], fmt='k_')
		axarr[i,plot].set_ylabel("Epsilon")
		i += 1
	if phiplot:
		axarr[i,plot].errorbar(freqs[valid_chans], [x*180.0/np.pi for x in receiver_params[valid_chans,4]], yerr = [x*180.0/np.pi for x in receiver_param_errs[valid_chans,4]], fmt='k_')
		axarr[i,plot].set_ylabel("Phi")
		i += 1
	if gammaplot:
		axarr[i,plot].errorbar(freqs[valid_chans], receiver_params[valid_chans,5], yerr = receiver_param_errs[valid_chans,5], fmt='k_')
		axarr[i,plot].set_ylabel("Gamma")
		i += 1

	axarr[nplots-1, plot].set_xlabel("Frequency (MHz)")
	plt.figure(0)
	plt.plot(np.sum(s_is, axis=0), "k-")
	plt.plot(np.sum(s_qs, axis=0), "r-")
	plt.plot(np.sum(s_us, axis=0), "g-")
	plt.plot(np.sum(s_vs, axis=0), "b-")
	plt.plot(np.sum(p_is, axis=0), "c--")
	plt.plot(np.sum(p_qs, axis=0), "m--")
	plt.plot(np.sum(p_us, axis=0), "y--")
	plt.plot(np.sum(p_vs, axis=0), "k--")
	plt.title("Precal")
	#plt.plot(on_indices,[0]*len(on_indices),"r*")
	plt.figure(2)
	plt.plot(np.sum(s_is, axis=0), "k-")
	plt.plot(np.sum(s_qs, axis=0), "r-")
	plt.plot(np.sum(s_us, axis=0), "g-")
	plt.plot(np.sum(s_vs, axis=0), "b-")
	plt.plot(np.sum(calibrated_is, axis=0), "c--")
	plt.plot(np.sum(calibrated_qs, axis=0), "m--")
	plt.plot(np.sum(calibrated_us, axis=0), "y--")
	plt.plot(np.sum(calibrated_vs, axis=0), "k--")
	plt.title("Postcal")
	plt.show()



else:
	print "not main"




