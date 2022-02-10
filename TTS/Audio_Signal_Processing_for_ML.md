we can see that the second impulse response just overlapped and added to the first impulse response.Why did we just overlap-and-add that second impulse response?The linear filter tells us that the output is just a sequence of overlapped-and-added impulse responses.The whole process of taking this time domain signal and using it to provoke impulse responses and then overlap-and-adding them in the output is called "convolution".Now how about keeping the filter the same and changing the source?The source only has one thing that you can change and that's the fundamental frequency. the pitch is changing but the vowel quality is the same. We've independently controlled source and filter.So, what have we achieved?We've taken natural speech, we've fitted the source-filter model to it, in particular we solved for the filter coefficients, then we've excited that filter with synthetic impulse trains at a fundamental frequency of our choice.Our source-filter model decomposes speech signals into a source component(that's either an impulse train for voiced speech, or white noise for unvoiced speech) and a filter (which has a frequency response determined by its coefficients)语音的频率pitchpitch is a part of a collection of other acoustic features that speakers use, which collectively we call prosody.声音的尖锐程度，在频域中表现为频率的高低。timbre音色音色在广义上是指声音不同于其它的特点，在语音中不同的音节都有不同的特点，这可以通过频域观察出来，另外，特别地，对于元音我们可以通过共振峰来分辨音色。noise噪音、辅音(摩擦音)都会有broad spectrum，也就是说我们无法通过共振峰来识别它们。envelope包络在波的时域和频域图中，用来形容图形的整体形状的叫做包络。比如在时域中，如果时间的分辨率较低，我们可以看到语音被分成一个一个菱形，上半部分三角形的轮廓就叫做  包络。下图展示了各种声音在时频域中的样子：3. Utterancehierarchy(等级制度) of phone如下图所示：可以看到Utterance满足层次结构，一般提取特征也是基于多个层次来做的。syllables最小的可以发声(pronounceable)的单元。open syllable(音节)：以元音为结尾的音节closed syllable：以辅音为结尾的音节consonant辅音 cluster：很多个辅音连接在一起，英文中常见accent / stress units发音的特性，有些语言通过声调来区分意义，比如日语或者中文，而英语是通过重音来区分意义的。rhythm(韵律) / isochrony也就是发声时候的节奏,中文是汉字，英文是由重音来作为分隔的。prosodic(韵律) / intonation units(语调单元)韵律、声调，针对单词和短语utterances（发声）一般是句子，但也可以变长。标点符号分隔。neighboring phones influence each other a lot。4. TTS Pipeline传统的TTS主要是通过组合多个模块构成流水线来实现的，整个系统可以大致分为frontend和backend。frontend主要是文字处理，使用NLP技术，从离散到离散，包括基本的分词、text normalization、POS以及特有的pronunciation标注。segmentation & normalization去噪、分句、分词以及把缩写、日期、时间、数字还有符号都换成可发音的词，这一步叫spell out。基本都基于规则grapheme-to-phoneme利用发音词典和规则，生成音素。音素一般利用ASCII编码，比如SAMPA和ARPAbet，这种编码在深度模       型中也可以被支持。这里的一个问题是pronunciation一般基于上下文，因为上下文可能决定了词的词性       等，比如read的过去式就有不同的读音IPA(international Phonetic Alphabet)是一个基于拉丁字母的语音标注系统。IPA只能表示口语的性质，比如因素，音调，音节等，如果还想要表       示牙齿舌头的变动则还有一个extension IPA可以用。IPA中最基本两种字母是letter和diacritic(变音符           号)，后者用来表示声调。IPA虽然统一了不同语言的发音，但是英语本身是stress language所以注音很        少，而中文这样依赖于音调的语言就会包含很多音调。intonation/stress generation这一步比较难，基本根据规则，或者构造统计模型前端和后端基本独立。backend根据前端结果生成语音，从离散到连续SSML(speech synthesis markup language)一种专门为语音合成做出来的语言，基于XML，包含了发音信息。waveform synthesis包含很多方法formant-based: 基于规则来生成共振峰还有其它成分concatenative: 基于database copy&pasteparametric model: HMM等，神经网络就是最新的参数模型Audio Signal Processing for Machine LearningFeatures of SoundFrequency :Hz(the number of times per second) higher frequency->higher soundIntensity(强度)   larger amplitude->louderSound power: Rate at which energy is transferred(转入)   Energy per unit of time(时间单元) emitted(发出) by a sound source in all directions   Measured in watt(W) Sound intensity: sound power per unit area    --->louderMeasured in threshold of hearing:human can perceive sounds with very small intensities                                threshold of pain(hearing pain):Intensity levelLogarithmic scaleMeasured in decibels(dB)Ratio(比率) between two intensity valuesUse an intensity of reference(TOH)                                                 I:intensity level      LoudnessSubjective(主观) perception of sound intensityDepends on duration/frequency of a soundDepends on ageMeasured in phons                 Equal loudness contours                                          Timbre(音色)Timbre is multidimensional, 音色在广义上是指声音不同于其它的特点，在语音中不同的音节都有不同的特点，这可以通过频域观察出来，另外，特别地，对于元音我们可以通过共振峰来分辨音色Sound envelope：Attack-Decay-Sustain-Release Modelc.Harmonic content(谐波含量)complex sound:superposition(叠加) of sinusoids,a partial is a sinusoid used to describe a sound,the lowest partial is called fundamental frequency(基频)，a harmonic partial is a frequency that's a multiple of the fundamental frequency.Inharmonicity indicates a deviation(偏差) from a harmonic partiald.Amplitude/frequency modulation(调制)PCM(编码方式) and WAV(文件格式)WAV：wav是一种无损的音频文件格式，WAV符合 PIFF(Resource Interchange File Format)规范。所有的WAV都有一个文件头，这个文件头音频流的编码参数。WAV对音频流的编码没有硬性规定，除了PCM之外，还有几乎所有支持ACM规范的编码都可以为WAV的音频流进行编码。PCM:PCM（Pulse Code Modulation----脉码调制录音)。所谓PCM录音就是将声音等模拟信号变成符号化的脉冲列，再予以记录。PCM信号是由[1]、[0]等符号构成的数字信号，而未经过任何编码和压缩处理。与模拟信号比，它不易受传送系统的杂波及失真的影响。动态范围宽，可得到音质相当好的影响效果。简单来说：wav是一种无损的音频文件格式，pcm是没有压缩的编码方式。wav可以使用多种音频编码来压缩其音频流，不过我们常见的都是音频流被pcm编码处理的wav，但这不表示wav只能使用pcm编码，mp3编码同样也可以运用在wav中，和AVI一样，只要安装好了相应的Decode，就可以欣赏这些wav了。在Windows平台下，基于PCM编码的WAV是被支持得最好的音频格式，所有音频软件都能完美支持，由于本身可以达到较高的音质的要求，因此，WAV也是音乐编辑创作的首选格式，适合保存音乐素材。因此，基于PCM编码的WAV被作为了一种中介的格式，常常使用在其他编码的相互转换之中，例如MP3转换成WMA。简单来说：pcm是无损wav文件中音频数据的一种编码方式，但wav还可以用其它方式编码。Pre-Emphasis首先，在信号上施加预加重滤波器，以放大高频。                                                                    一般取0.95 or 0.97作用：平衡频谱，因为高频通常比低频具有较小的幅度避免傅里叶变换过程中出现数值问题可以改善信噪比(SNR:Signal-to-Noise Ratio)                                       Extracting FeaturesTime-domain featuresAmplitude envelope(AE)Max amplitude value of all samples in a frame    Amplitude envelope at frame t:          K:frame size  s(k): Amplitude of kth sample  t.K: First sample of frame t  (t+1).K-1: last sample of frame tGives rough idea of loudnessSensitive to outliersOnset detection, music genre classificationFRAME_SIZE = 1024
HOP_LENGTH = 512

def amplitude_envelope(signal, frame_size, hop_length):
    """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
    amplitude_envelope = []
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
        amplitude_envelope.append(amplitude_envelope_current_frame)
    return np.array(amplitude_envelope)    

def fancy_amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])Root-mean-square energy(RMS)  1. RMS of all samples in a frame                : sum of energy of samples in frame tIndicator of loudnessLess sensitive to outliers than AEAudio segmentation(音频分割), music genre(流派) classificationrms_debussy = librosa.feature.rms(debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]def rmse(signal, frame_size, hop_length):
    rmse = []
    
    # calculate rmse for each frame
    for i in range(0, len(signal), hop_length): 
        rmse_current_frame = np.sqrt(sum(signal[i:i+frame_size]**2) / frame_size)
        rmse.append(rmse_current_frame)
    return np.array(rmse)    Zero-crossing rate(ZCR)  1.Number of times a signal crosses the horizontal axis     sgn:sign functionRecognition of percussive(打击乐) vs pitched soundsMonophonic pitch estimationVoice/unvoiced decision for speech signalszcr_debussy = librosa.feature.zero_crossing_rate(debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]FramesPerceivable(可感知的) audio chunk1 sample@44.1kHz=0.0227ms duration 1 sample << Ear's time resolution(10ms)Power of 2 num. samples (speed up the process a lot) df:duration                                                                           音频在量化得到二进制的码字后，需要进行变换，而变换（MDCT）是以块为单位（block）进行的，一个块由多个（120或128）样本组成。而一帧内会包含一个或者多个块。帧的常见大小有960、1024、2048、4096等。一帧记录了一个声音单元，它的长度是样本长度和声道数的乘积。                            The larger analysis frame means we're able to be less precise about where in time that snalysis applies to , so we get lower time resolution, so it's going to be a trade-off.From time to frequency domainUse Fourier transform,we can move from time into frequency domain but unfortunately there's a major issue which is called spectral leakageSpectral leakageProcessed signal isn't an integer number of periodsEndpoints(端点) are discontinous                         Discontinuities appear as high-frequency components not present in the original signal,some of this discontinuities frequencies at the discontinuities are just like leaked into other higher frequencies. Solve the spectral leakage use windowingWindowingApply windowing function to each frame before we feed the frames into the FT Eliminates samples at both ends of a frameGenerates a periodic signal which minimizes special leakage Why not using rectangular window functions?we're accidentally introduced something into the signal that wasn't there in the original, that is the sudden changes at the edge of the signal.If we analysed this signal we'd not only be analysing the speech but also those artefacts. So we don't generally use rectangular window functions because these artefacts are bad, but rather we use tapered(锥形) windows. It doesn't have those sudden discontinuous at the edges.Hann window                                                                                                                                             3 frames,we find the endpoints of frame lose signal,how we solve this? overlapping!                                                            When converting a waveform to a sequence of frames, why is the frame shift usually smaller than the frame duration?  (不懂)Because a tapered window is applied to each frame.How can series expansion be used to remove high-frequency noise from a waveform? By truncating the series, which means setting to zero the coefficients of all the basis functions above a  frequency of our choosing.In Fourier analysis, what are the frequencies of the lowest and highest basis functions?The lowest frequency basis function has a fundamental period equal to the analysis frame duration. The highest frequency basis function is at the Nyquist frequency.  **傅里叶相关理论知识在傅里叶相关知识里说明FT(Fourier transform)IntuitionDecompose a complex sound into its frequency componentsWe can make any complex wave by adding together sine waves. Compare signal with sinusoids of various frequencies将时域上的信号转变为频域上的信号，看问题的角度也从时间域转到了频率域，因此在时域中某些不好处理的地方，在频域就可以较为简单的处理，这就可以大量减少处理信号计算量。信号经过傅里叶变换后，可以得到频域的幅度谱(magnitude)以及相位谱(phase)，信号的幅度谱和相位谱是信号傅里叶变换后频谱的两个属性。在分析信号时，主要应用于处理平稳信号，通过傅里叶变换可以获取一段信号总体上包含哪些频率的成分，但是对各成分出现的时刻无法得知。因此对于非平稳信号，傅里叶变换就显示出了它的局限性，而我们日常生活中的绝大多数音频都是非平稳信号的。而解决这一问题的方法，就是采用短时傅里叶变换或者小波变换，对信号进行处理DFT(Discrete Fourier transform)   ---------->  其中k=[0,M-1]=[0,N-1]#frequencies(M)=#samples(N)why M=N?   Invertible transformation,computational efficientRedundancy in DFT (Nyquist Frequency)FFT(Fast Fourier Transform)DFT is computationally expensive FFT is more efficient FFT exploits redundancies across sinusoidsFFT works when N is a power of 2# fast fourier transform
violin_ft = np.fft.fft(violin_c4)
magnitude_spectrum_violin = np.abs(violin_ft)

def plot_magnitude_spectrum(signal, sr, title, f_ratio=1):
    X = np.fft.fft(signal)
    X_mag = np.absolute(X)
    
    plt.figure(figsize=(18, 5))
    
    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag)*f_ratio)  
    
    plt.plot(f[:f_bins], X_mag[:f_bins])
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
   
plot_magnitude_spectrum(violin_c4, sr, "violin", 0.1)STFTWindowingApply windowing function to signal overlapping framesFrom DFT to STFT                                                                    m: frame number     n: frame size    w(n):windowing functionOutputswe get a fourier coefficient for each of the frequency components we're decomposed our original signal into.and this is a one dimensional array it's just like a vector.DFTSpectral vector(#frequency bins)N complex Fourier coefficentsSTFTwe get a complex fourier coefficient for each frequency bin that we are considering for each frame.Spectral matrix (#frequency bins, #frames)    frequency bins=framesize/2 + 1     frames=(samples-framesize)/hopsize + 1Complex Fourier coefficients#Extracting Short-Time Fourier Transform
FRAME_SIZE = 2048
HOP_SIZE = 512
S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
#S_scale.shape=(1025, 342)

#Calculatiing the spectrogram
Y_scale = np.abs(S_scale) ** 2  #Y_scale.shape=(1025, 342)

#Visualizing the spectrogram 
def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
plot_spectrogram(Y_scale, sr, HOP_SIZE)Linear spectrogramY_log_scale = librosa.power_to_db(Y_scale)
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")Time/frequency trade offSTFT parametersFrame size: 一般选为256,512,1024,2048,4096hop size: 一般为1/2 ,1/4, 1/8 framesizeWindowing function: hann windowVisualising soundLinear-SpectrogramMel-SpectrogramsTime-frequency representationPerceptually-relevant amplitude representationPerceptually-relevant frequency representationMel scaleMel scale(梅尔标度):人耳能听到的频率范围是20-20000Hz，但人耳对Hz这种标度单位并不是线性感知关系让我们观察一下从Hz到mel的映射图，由于是log的关系，当频率较小时，mel随Hz变化较快；当频率很大时，mel的上升很缓慢，曲线的斜率很小。这说明了人耳对低频音调的感知较灵敏，在高频时人耳是很迟钝的。如果将普通的频率标度转化为梅尔频率标度，则人耳对频率的感知度就成了线性关系。线性频率标度映射到梅尔频率标度公式为：                                                                                                         梅尔标度滤波器组启发于此。Mel filter banksConvert lowest/highest frequency to MelCreate #bands equal spaced pointedConvert points back to Hertz  Round to nearest frequency bin Create triangular filters(the kind of building blocks of a mel filter bank)filter:滤波器是具有频率选择作用的电路(模拟滤波)或运算处理系统(数字滤波)，具有滤除噪声和分离各种不同信号的功能。按功能分：低通 高通 带通 带阻最后应用triangular filters计算滤波器组(filter banks)，通常用40个滤波器nfilt=40 on a Mel-scale to the power spectrum to 提取频带(frequency bands). 如上图所示，40个三角滤波器组成滤波器组，低频处滤波器密集，门限值大，高频处滤波器稀疏，门限值低。恰好对应了频率越高人耳越迟钝这一客观规律。上图所示的滤波器形式叫做等面积梅尔滤波器（Mel-filter bank with same bank area），在人声领域（语音识别，说话人辨认）等领域应用广泛，但是如果用到非人声领域，就会丢掉很多高频信息。这时我们更喜欢的或许是等高梅尔滤波器（Mel-filter bank with same bank height）：通过梅尔滤波器组将线性频谱转为梅尔频谱Recipe to extract Mel spectrogramExtract STFTConvert amplitude to DBsConvert frequencies to Mel scale  a.Choose number of mel bands(超参数)b.Construct mel filter banks,Mel filter bands' matrix shape:M=(#band, framesize/2+1)      c.Apply mel filter banks to spectrogram, Y=(framesize/2+1, #frames)    Mel spectrogram=MY=(#bands,#frames)#Mel filter banks
filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10) # shape=(10, 1025)

plt.figure(figsize=(25, 10))
librosa.display.specshow(filter_banks, 
                         sr=sr, 
                         x_axis="linear")
plt.colorbar(format="%+2.f")
plt.show()Mel filter banks #Extracting Mel Spectrogram
#mel_spectrogram.shape=(10, 342)
mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()Mel Frequency Cepstral Coefficient(MFCC)spectrum of a spectrum=cepstrumThe main point to understand about speech is that the sounds generated by a human are filtered by the shape of the vocal tract including tongue, teeth etc. This shape determines what sound comes out. If we can determine the shape accurately, this should give us an accurate representation of the phoneme being produced. The shape of the vocal tract(声带) manifests(表明) itself in the envelope(包络线) of the short time power spectrum(功率谱), and the job of MFCCs is to accurately represent this envelope. we can treat this log power spectrum as a signal at a time domain signal and we apply a inverse discrete fourier transform and we get the spectrum which is the spectrum of a spectrum.Understanding the CepstrumSpeech generationThe main point to understand about speech is that the sounds generated by a human are filtered by the shape of the vocal tract including tongue, teeth etc. This shape determines what sound comes out. If we can determine the shape accurately, this should give us an accurate representation of the phoneme being produced. The shape of the vocal tract(声带) manifests(表明) itself in the envelope(包络线) of the short time power spectrum(功率谱), and the job of MFCCs is to accurately represent this envelope. Separating the componentsspeech signal = Glottal pulse + vocal tract(声带)， vocal tract acts as a filter on the glottal pulseThis picks in red are called formants(共振峰)，formats carry identity of sound(timbre), you're perceive certain phonemes instead of others in other words,the spectral envelope provide us information about timbre.Log-spectrum - Spectral envelopeSpeech = Convolution of vocal tract frequency response with glottal pulse.Formalising speech  fourier transfrom  log(X(t)): log-spectrum     log(H(t)): spectral envelope     log(E(t)): glottal pulseDecompose that signal into its queferancy components and see how presence of the different frequency components are.              4Hz                                   100Hzthe low quefrency values represent the slowly change spectral information in speech spectral signal.Computing Mel-Frequency Cepstral Coefficients    Using the discrete cosine transform instead of the inverse fourier transform,.we get a number of coefficients(mfcc).Why?Simplified version of Fourier TransformGet real-valued coefficient  (discrete fourier transform get complex-valued coefficient)Decorrelate energy in different mel bandsReduce #dimensions to represent spectrumMFCCs advantageDescribe the "large" structures of the spectrum,we take like thefirst nfcc's which focusing on the spectral envelope on the formants about phonemes.Ignore fine spectral structures.Work well in speech and music processing.#Extract MFCCs
mfccs = librosa.feature.mfcc(signal, n_mfcc=13, sr=sr )

#calculate the first and second derivatives of the mfcc's
delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs, order=2)

comprehensive_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))Extracting frequency-domain featuresMath conventions Magnitude of signal at frequency bin n and frame t.N: #frequency binsBand energy ratioComparison of energy in the lower/higher frequency bandsMeasure of how dominant low frequencies are                                                  F: split frequency, 一般取2000HzBand energy ratio applications(1)Music/speech discrimination (2)Music classification(eg.music genre classification)debussy_spec.shape=(1025, 1292)  # the spectrum of debussy audio
debussy_spec_transpose = debussy_spec.T   #(1292, 1025)
#1025:the number of freqency bins
#Calculate Band Energy Ratio
def calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate):
    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / spectrogram.shape[0]
    #mapping this continuous frequency onto the closest frequency being available
    #np.floor:10.4->10.0, 10.9->10.0
	split_frequency_bin = np.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)

split_frequency_bin = calculate_split_frequency_bin(debussy_spec, 2000, 22050) #185def calculate_band_energy_ratio(spectrogram, split_frequency, sr):
    split_frequency_bin = calculate_split_frequency_bin(spectrogram, split_frequency, sr)
    # move to the power spectrogram
    power_spec = np.abs(spectrogram) ** 2
    power_spec = power_spec.T
    
    band_energy_ratio = []
    
    #calculate BER for each frame
    for frequencies_in_frame in power_spec:
        sum_power_low_frequencies = np.sum(frequencies_in_frame[:split_frequency_bin])
        sum_power_high_frequencies = np.sum(frequencies_in_frame[split_frequency_bin:])
        ber_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(ber_current_frame)
    return np.array(band_energy_ratio)

ber_debussy = calculate_band_energy_ratio(debussy_spec, 2000, sr)
frames = range(len(ber_debussy))
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)
plt.figure(figsize=(25, 10))

plt.plot(t, ber_debussy, color="b")Spectral centroid(谱质心)Centre of gravity of magnitude spectrum, weighted mean of the frequenciesFrequency band where most of the energy is concentratedMeasure of "brightness" of sound                                          n: frequency bin  : weight of nSpectral centroid applications(1)Audio classification (2) Music classificationsc_debussy = librosa.feature.spectral_centroid(y=debussy,sr=sr,n_fft=FRAME_SIZE,hop_length=HOP_SIZE)BandwidthDerived from spectral centroidSpectral range around the centroidVariance from the spectral centroidDescribe perceived timbreEnergy spread across frequency band(spectral spread) upper , upper,Energy spread across frequency band lower , lower.Bandwidth applicationsMusic processing(eg. music genre classification)ban_debussy = librosa.feature.spectral_bandwidth(y=debussy, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]