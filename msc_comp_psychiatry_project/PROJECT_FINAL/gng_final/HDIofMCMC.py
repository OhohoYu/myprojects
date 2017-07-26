def HDIofMCMC(sampleVec, credMass = 0.95):
    sortedPts = np.sort(sampleVec)
    ciIdxInc = math.floor(credMass*len(sortedPts))
    nCIs = len(sortedPts) - ciIdxInc
    ciWidth = np.zeros(nCIs)
    for i in range(0, nCIs):
        ciWidth[i] = sortedPts[i+ciIdxInc]-sortedPts[i]
    HDImin = sortedPts[ciWidth.argmin(0)]
    HDImax = sortedPts[ciWidth.argmin(0)+ciIdxInc]
    HDIlim = [HDImin, HDImax]
    return HDIlim
