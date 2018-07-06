#!/usr/bin/env python
from datetime import datetime
tstart = datetime.now()
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
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

def get_stokes_portrait(filename):
#This takes in a string filename of a PSRCHIVE file, and returns the tscrunched stokes
#parameters of each channel.
	#Load the file as a PSRCHIVE archive
	arch = psrchive.Archive_load(filename)
	
	#Convert to stokes, remove the baseline, and dedisperse
	arch.convert_state("Stokes")
	arch.remove_baseline()
	arch.dedisperse()
	
	#We're just going to do a channel-by-channel comparison of the polarization profiles
	#so we don't need subint information
	arch.tscrunch_to_nsub(1)
	
	#Actually get the data cube, and get the stokes params from the data cube.
	#Data indices are [subint, stokes param, chan, bin].
	data = arch.get_data()
	ints = data[0, 0, :, :]
	qs = data[0, 1, :, :]
	us = data[0, 2, :, :]
	vs = data[0, 3, :, :]
	
	#Get profile weights, and apply them to the data.
	#This is essentially applying the RFI zap.
	weights = arch.get_weights()
	weighted_intens  = ints*weights.T
	weighted_qs = qs*weights.T
	weighted_us = us*weights.T
	weighted_vs = vs*weights.T
	
	#Return the "zapped" Stokes params
	return weighted_intens, weighted_qs, weighted_us, weighted_vs, weights

def align_profiles(profiles):
#This aligns all of the profiles (specified as a list) to the first profile in the list
#It returns the aligned profiles, as well as the amount the profiles needed to be
#shifted in order to align them.
	
	#Call the first profile the "template" profile.
	template_profile = list(profiles[0])
	
	#Since we're aligning all the given profiles to this profile, we can just add it
	#to the final aligned list right away
	aligned_profiles = [profiles[0]]
	
	#and specify its offset as 0
	offsets = [0]
	
	#For the rest of the profiles, we'll actually need to align them
	for prof in profiles[1:]:
		#We're just going to cross-correlate the given profile with the template profile
		c_vals = np.correlate(template_profile*2, prof*2, mode='full')
		
		#and find the max of the cross-correlation
		shift = -np.argmax(c_vals)-1
		
		#This is the offset, by which we'll actually need to rotate the profile to align it
		offsets.append(shift)
		
		#Then we need to actually shift the profile. This handles both lists and np arrays
		if type(prof) == np.ndarray:
			aligned_profiles.append(np.roll(prof, -shift))
		elif type(prof) == list:
			aligned_profiles.append(prof[shift:] + prof[:shift])
	
	#Return the aligned profiles and the offsets
	return aligned_profiles, offsets	

def center_prof(profs):
#This just rotates the profile so the max total intensity is in the center
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
#Takes in NxM data arrays, aligns them, then zaps them
#using other methods. First, align and scale the portraits
	aligned_ps = align_portraits(scale_portraits(portraits))
	p1, p2 = aligned_ps
	
	#Then get the channel weights
	weights1 = make_weights(p1)
	weights2 = make_weights(p2)
	
	#Make sure if a channel is zapped in *either* portrait
	#it gets zapped in *both*
	weights = weights1*weights2
	
	#Actually apply the weights and return
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
#This would be useful if you're trying to use this code with two profiles
#that have different numbers of bins
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
	
	#Get the on pulse indices
	on_indices = get_on_indices([p_is, p_qs, p_us, p_vs, s_is, s_qs, s_us, s_vs], num = len(t_prof)/20)
	off_indices = get_off_indices(np.sum(p_is, axis=0), np.sum(s_is, axis=0))
	
	#Find out how bright each ON PULSE REGION of the invariant interval is for each freq channel
	template_chan_weights = np.sum(template_invariant_intervals[:,on_indices], axis=1)
	data_chan_weights = np.sum(data_invariant_intervals[:,on_indices], axis=1)
	
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
#Also note that this then assumes the profiles have been aligned.
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
#This method just looks for the phase window of width "window"
#with the minimum total intensity
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
#It actually makes the mueller matrix from the receiver params
	#First define all the params
	theta1, theta2, eps1, eps2, phi, gamma = params
	
	#Then define secondary params. These are just convenient
	#combinations of the receiver params given above
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
	
	#For clarity, let's define separate Mueller matrices for different components of the
	#signal path from the source through the receiver. 
	
	#We have the Mueller Matrix for the amplifier chain
	Mamp = np.asarray([[1, E, 0, 0], [E, 1, 0, 0], [0, 0, F, -G], [0, 0, G, F]])
	
	#The cross-coupling
	Mcc = np.asarray([[1,0,A,B],[0, 1, C, D],[A, -C, 1, 0],[B, -D, 0, 1]])
	
	#And then the feed. Note this is just the identity matrix since this code assumes the 
	#data have been calibrated with the IFA, which, when performed with PSRCHIVE,
	#will have already corrected for this
	Mfeed = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	
	#Then we multiply these matrices together to get the full Mueller matrix
	mueller = np.dot(np.dot(Mamp, Mcc), Mfeed)
	
	#Now, we're going to actually apply the Mueller matrix transformation to the given data
	#and compare the result to the template data, returning residuals to be minim
	data_is, data_qs, data_us, data_vs, template_is, template_qs, template_us, template_vs  = stokes
	for i in range(len(data_is)):
		S_int = np.asarray([template_is[i], template_qs[i], template_us[i], template_vs[i]]).reshape(4,1)
		S_meas = np.asarray([data_is[i], data_qs[i], data_us[i], data_vs[i]]).reshape(4,1)
		resids += np.sum((np.dot(mueller,S_int) - S_meas)**2)
	 
	return resids

def receiver_func(stokes_intrinsic, theta1, theta2, eps1, eps2, phi, gamma):
#This function takes in stokes parameters and receiver parameters and applies the resulting
#Mueller matrix to the given stokes parameters, returning the resulting stokes parameters.
	
	#First define secondary parameters from the receiver params. These are just convenient
	#combinations of the receiver params given above.
	A = eps1*np.cos(theta1) + eps2*np.cos(theta2)
	B = eps1*np.sin(theta1) + eps2*np.sin(theta2)
	C = eps1*np.cos(theta1) - eps2*np.cos(theta2)
	D = eps1*np.sin(theta1) - eps2*np.sin(theta2)
	E = gamma/2.0
	F = np.cos(phi)
	G = np.sin(phi)
	H = 1
	I = 0
	
	#For clarity, let's define separate Mueller matrices for different components of the
	#signal path from the source through the receiver. 
	
	#We have the Mueller Matrix for the amplifier chain
	Mamp = np.asarray([[1, E, 0, 0], [E, 1, 0, 0], [0, 0, F, -G], [0, 0, G, F]])
	
	#The cross-coupling
	Mcc = np.asarray([[1,0,A,B],[0, 1, C, D],[A, -C, 1, 0],[B, -D, 0, 1]])
	
	#And then the feed. Note this is just the identity matrix since this code assumes the 
	#data have been calibrated with the IFA, which, when performed with PSRCHIVE,
	#will have already corrected for this
	Mfeed = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	
	#Then we multiply these matrices together to get the full Mueller matrix
	mueller = np.dot(np.dot(Mamp, Mcc), Mfeed)
	
	#Get the total number of bins
	n_on_bins = len(stokes_intrinsic)/4
	
	#Separate the Stokes parameters from the input data
	data_is, data_qs, data_us, data_vs  = stokes_intrinsic.reshape((4,n_on_bins))
	
	#Define outputs
	out_is = []
	out_qs = []
	out_us = []
	out_vs = []
	
	#We'll need this to return output Stokes params in the same shape as the input Stokes params
	n_data_points = 4*len(data_is)
	
	#Transform each Stokes bin using the Mueller matrix
	for i in range(len(data_is)):
		#Make a stokes vector
		S_int = np.asarray([data_is[i], data_qs[i], data_us[i], data_vs[i]]).reshape(4,1)
		
		#Apply the Mueller matrix
		S_out = np.dot(mueller,S_int)
		out_is.append(S_out[0])
		out_qs.append(S_out[1])
		out_us.append(S_out[2])
		out_vs.append(S_out[3])
	
	#Numpy arrays are great
	out_is = np.asarray(out_is)
	out_qs = np.asarray(out_qs)
	out_us = np.asarray(out_us)
	out_vs = np.asarray(out_vs)
	
	#Make the output the same shape as the input
	out_stokes = np.asarray([out_is, out_qs, out_us, out_vs])
	return out_stokes.squeeze().reshape((n_data_points,))

def calibrate(params, stokes):
#This is very similar to the receiver_func method, so refer to that for more info,
#but the basic idea is that normally, the Mueller matrix acts on the intrinsic stokes params,
#but when we calibrate, we want to undo that, so we need to multiply the *measured* stokes params
#by the *inverse* of the mueller matrix.
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
	
	#This is the only part that is meaningfully different from recevier_func. 
	#Here, we get the inverse of the Mueller matrix
	inv_mueller = np.linalg.inv(mueller)
	
	nbins = len(stokes[0])
	data_is, data_qs, data_us, data_vs = stokes
	calibrated_is, calibrated_qs, calibrated_us, calibrated_vs  = [[],[],[],[]]
	stokes_vectors = np.asarray([data_is, data_qs, data_us, data_vs]).reshape(4,nbins)

	#Actually calibrate
	calibrated_stokes = np.dot(inv_mueller,stokes_vectors)
	
	#Return the calibrated data
	return calibrated_stokes

def get_mtm_solution(standard_name, data_name, verbose=True, fake=False):
#This is the main method in the code. It takes in data, performs the METM fit,
#keeps track of GOF meaasures, prints out similar things to PSRCHIVE's pcm,
#and actually calibrates the data.
	
	#Read in data and get portrait
	standard = get_stokes_portrait(standard_name)
	data = get_stokes_portrait(data_name)
	
	#Massage data
	p_is, p_qs, p_us, p_vs, s_is, s_qs, s_us, s_vs, weights = prepare_data(standard, data)
	
	#Get on pulse indices and invariant intervals for template and data
	on_indices = get_on_indices([p_is, p_qs, p_us, p_vs, s_is, s_qs, s_us, s_vs])
	off_indices = get_off_indices(np.sum(p_is, axis=0), np.sum(s_is, axis=0))
	data_invariant_intervals = p_is**2 - (p_qs**2 + p_us**2 + p_vs**2)
	template_invariant_intervals = s_is**2 - (s_qs**2 + s_us**2 + s_vs**2)
	
	#We want to keep track of these
	receiver_params = []
	receiver_param_errs = []
	valid_chans = []
	calibrated_is, calibrated_qs, calibrated_us, calibrated_vs = [[], [], [], []]
	nbins = len(p_is[0,:])
	n_on_bins = len(on_indices)
	n_data_points = 4*n_on_bins
	
	#Keep track of run time
	tstart = datetime.now()
	nfails = 0
	
	#We're going to run the METM code for every channel
	for i in range(len(s_is[:,0])):
		#If the channel is not zapped, we actually want to do stuff.
		if weights[i]:
			#Get stokes in and out and make sure they're the right shape
			stokes_int = np.asarray([s_is[i,on_indices], s_qs[i,on_indices], s_us[i,on_indices], s_vs[i,on_indices]]).reshape((n_data_points,))
			stokes_out = np.asarray([p_is[i,on_indices], p_qs[i,on_indices], p_us[i,on_indices], p_vs[i,on_indices]]).reshape((n_data_points,))
			
			#Get off pulse std
			p_err = np.std(p_is[i,off_indices])
			s_err = np.std(s_is[i,off_indices])
			
			#Keep the larger of the two
			if p_err>s_err: prof_err = p_err
			else: prof_err = s_err
			
			#Make the initial guess for the rcvr params to be 0 (ie MEM was perfect)
			p0 = [0,0,0,0,0,0]
			try:
				#Find best fit rcvr params
				popt, pcov = optimize.curve_fit(receiver_func, stokes_int, stokes_out, bounds = ((-np.pi/2.0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, -np.pi, -np.pi/2.0),(np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi, np.pi/2.0)))
				
				#Save the rcvr params for this channel
				receiver_params.append(popt)
				
				#Get the template/calibrated profile resids
				resids = np.asarray(stokes_out) - np.asarray(receiver_func(stokes_int, *popt))
				
				#Calculate the reduced chisq and get rcvr errs
				redchisqs = np.sum(resids**2)/((len(on_indices)*4 - len(popt))*(prof_err**2))
				receiver_param_errs.append([pcov[0,0]*redchisqs, pcov[1,1]*redchisqs, pcov[2,2]*redchisqs, pcov[3,3]*redchisqs, pcov[4,4]*redchisqs, pcov[5,5]*redchisqs])
				
				#Print output similar to PSRCHIVE's pcm
				if verbose:
					if i<10: print "Channel  ", i, "solved,", redchisqs
					elif i<100: print "Channel ", i, "solved,", redchisqs
					else: print "Channel", i, "solved,", redchisqs
				
				#Actually calibrate the input data with the best fit Mueller matrix
				cal = calibrate(popt, [p_is[i,:], p_qs[i,:], p_us[i,:], p_vs[i,:]])
				calibrated_is.append(cal[0])
				calibrated_qs.append(cal[1])
				calibrated_us.append(cal[2])
				calibrated_vs.append(cal[3])
				
				#Keep track of valid (unzapped) channels
				valid_chans.append(i)
				
			#This could fail for different reasons, but we don't want to stop the whole process
			#if it does, so we'll mark the channel as failed and move on
			except Exception as e:
				#Print output similar to PSRCHIVE's pcm
				if verbose:
					if i<10: print "Channel  ", i, "failed:", e
					elif i<100: print "Channel ", i, "failed:", e
					else: print "Channel", i, "failed:", e
				
				#Store 0's for the output data, and rcvr params
				calibrated_is.append([0]*nbins)
				calibrated_qs.append([0]*nbins)
				calibrated_us.append([0]*nbins)
				calibrated_vs.append([0]*nbins)
				receiver_params.append([0,0,0,0,0,0])
				receiver_param_errs.append([0,0,0,0,0,0])
				
				#Keep track of how many channels failed
				nfails +=1
		
		#If the channel was zapped
		else:
			#Print output similar to PSRCHIVE's pcm
			if verbose:
				if i<10: print "Channel  ", i, "skipped"
				elif i<100: print "Channel ", i, "skipped"
				else: print "Channel", i, "skipped"
			
			#Store 0's for the output data, and rcvr params
			calibrated_is.append([0]*nbins)
			calibrated_qs.append([0]*nbins)
			calibrated_us.append([0]*nbins)
			calibrated_vs.append([0]*nbins)
			receiver_params.append([0,0,0,0,0,0])
			receiver_param_errs.append([0,0,0,0,0,0])
		
	#Tell the user how long it took to run, and if some chans failed, how many.
	tdone = datetime.now()
	if verbose: print "Solved and calibrated in", tdone-tstart
	if nfails: print "Solve failed for", nfails, "channels"
	else: print "Generated MTM solution with no failed channels."
	
	#Numpy arrays are great
	calibrated_is = np.asarray(calibrated_is)
	calibrated_qs = np.asarray(calibrated_qs)
	calibrated_us = np.asarray(calibrated_us)
	calibrated_vs = np.asarray(calibrated_vs)
	receiver_params = np.asarray(receiver_params)
	receiver_param_errs = np.asarray(receiver_param_errs)
	
	#Return the results of the fit
	return np.asarray([s_is, s_qs, s_us, s_vs]), np.asarray([p_is, p_qs, p_us, p_vs]), np.asarray([calibrated_is, calibrated_qs, calibrated_us, calibrated_vs]), valid_chans, receiver_params,receiver_param_errs

def write_pcm(sfname, receiver_params, valid_chans, outname = "pcm.fits", fake=False):
#This method writes out the receiver parameters into a file that's readable/usable by PSRCHIVE.
#Most of it is getting the header/data formatting correct, so it's not very enlightening, but it is useful.
	shdulist = pyfits.open(sfname)
	head = shdulist[0].header
	head['OBS_MODE'] = "PCM"
	nchan = head['OBSNCHAN']
	prihdu = pyfits.PrimaryHDU(header=head)
	new_hdulist = pyfits.BinTableHDU(header=shdulist[1].header, data=shdulist[1].data,name=shdulist[1].name)
	chans = np.linspace(head['OBSFREQ']-head['OBSBW']/2.0, head['OBSFREQ']+head['OBSBW']/2.0, nchan, dtype='d')
	weights = np.zeros(nchan)
	weights[valid_chans] = 1
	Gs = weights.reshape(nchan,1)
	expanded_receivers = np.zeros(nchan*6).reshape(nchan,6)
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
#Sometimes the METM method "works" (ie the fit returns bestfit params and errs), but
#the returned params are junk (if, for example, there are channels with some unzapped RFI).
#In these cases, the errors are enormous. This method looks for these cases and effectively
#zaps the data (sets rcvr params, errs to 0 and calibrated data to 0 in those chans).
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
	
	#Look at all channels
	for i in range(len(errs)):
		#Look for big errors in the rcvr params
		if (errs[i,:-1] > 5.5).sum() > 0.5 or errs[i,-1] > 1:
			zap = True
		
		#If found, set relevant params to 0.
		if zap:
			cparams.append([0,0,0,0,0,0])
			cerrs.append([0,0,0,0,0,0])
			ccis.append(np.zeros_like(cis[i]))
			ccqs.append(np.zeros_like(cqs[i]))
			ccus.append(np.zeros_like(cus[i]))
			ccvs.append(np.zeros_like(cvs[i]))
			print "Zapped channel", i
		
		#Else, keep the old params
		else:
			cparams.append(params[i])
			cerrs.append(errs[i])
			ccis.append(cis[i])
			ccqs.append(cqs[i])
			ccus.append(cus[i])
			ccvs.append(cvs[i])
			
		#Keep track of the valid chans
		#(that is, remove "unclean" chans from list of valid chans)
		if np.sum(cparams[i]):
			cvchans.append(i)
		zap = False
	
	#Numpy arrays are great
	cparams = np.asarray(cparams)
	cerrs = np.asarray(cerrs)
	cvchans = np.asarray(cvchans)
	ccaldata = np.asarray([ccis, ccqs, ccus, ccvs])
	return cparams, cerrs, ccaldata, cvchans

if __name__ == "__main__":
	args = argv[1:]
	overlay_profs = False
	seperate_stokes = False
	plot = True
	plotpa = False
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

	factor = 4
	prof_diffs = []
	mjds = []
	std_statistics = []
	
	#Fake just returns rcvr params = 0, errs = 1, and outputs an "identity" Mueller matrix
	#that is compatible with PSRCHIVE
	if fake:
		prepd_s_data, prepd_p_data, calibrated_data, valid_chans, receiver_params, receiver_param_errs = get_mtm_solution(files[0], files[1], verbose, fake=True)
		write_pcm(files[0], receiver_params, valid_chans, "identity.pcm", fake=fake)
		exit(0)	
	
	#Perform the METM method on the input data files
	prepd_s_data, prepd_p_data, calibrated_data, valid_chans, receiver_params, receiver_param_errs = get_mtm_solution(files[0], files[1], verbose)
	
	#Clean the resulting rcvr params
	receiver_params, receiver_param_errs, calibrated_data, valid_chans = clean_receiver_params(receiver_params, receiver_param_errs, calibrated_data, valid_chans)
	s_is, s_qs, s_us, s_vs = prepd_s_data
	p_is, p_qs, p_us, p_vs = prepd_p_data
	
	#Prepare to write out the data
	calibrated_is, calibrated_qs, calibrated_us, calibrated_vs = calibrated_data
	
	#This might not be correct for other people, but this code was not written to be universally
	#compatible, so people can change things as they see fit.
	bw = 800.0
	nchan = len(np.sum(s_is,axis=1))
	chan_bw = 800.0/nchan
	calibrated_is, calibrated_qs, calibrated_us, calibrated_vs = calibrated_data
	freqs = np.linspace(cfreq+400-(chan_bw/2.0),cfreq-400+(chan_bw/2.0),nchan)
	
	#Prepare to write out the data in pickle format
	outpickle_name = ".".join(files[-1].split(".") + ["pickle"])
	outcal = ".".join(files[-1].split(".") + ["calibpickle"])
	outpcm = ".".join(files[-1].split(".") + ["pcmpickle"])
	
	#Try writing the data using pickle
	try:
		with  open(outpickle_name, "wb") as pfile:
			pickle.dump([prepd_s_data, prepd_p_data, calibrated_data, freqs, receiver_params, receiver_param_errs], pfile, protocol=pickle.HIGHEST_PROTOCOL)
		with open(outcal, "wb") as pfile:
			pickle.dump([prepd_s_data, prepd_p_data, calibrated_data, freqs], pfile, protocol=pickle.HIGHEST_PROTOCOL)
		with open(outpcm, "wb") as pfile:
			pickle.dump([receiver_params, receiver_param_errs, freqs], pfile, protocol=pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print "Could not write pickle files. Threw error:"
		print str(e)
	
	#Write out the rcvr solution to a file
	write_pcm(files[0], receiver_params, valid_chans, outname, fake=fake)
	
	#If the user wants to plot the rcvr solution, do it.
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
	plt.show()

