## Basics knowledge of TTS
### Foundations for Speech Processing
1.Get the signal into digital form.This involves converting an analogue(continuous) value into digital(discrete) one. Discretisation(离散化) in time is called sampling and discretisation in amplitude is called quantisation(量化).  
2.Analyse the signal in some way, and for signals that change over time, that requires short-term analysis.  
#### 1.Sampling
Sampling is the process of recording the amplitude of a signal only at specific moments in time.(generally, the means a fixed number of times per second, evenly spaced in time(间隔均匀)). Each recorded value is called a sample.
* Time axis: the sound pressure is sampled at fixed intervals(thousands of time per second)
* Vertical axis: continuous value(representing sound pressure) is encoded as one of a fixed number of discrete levels.
#### 2.Quantisation
Quantisation is the process of storing the amplitude of each sample with a fixed precision(generally, that means as a binary number with a fixed number of bits)  
**Sampling rates,bit rate and bit depth(采样位数，位深度，分辨率)**  
  bit rate:	每秒传输速度  
* **The Nyquist frequency**: represent frequencies up to half the sampling frequency.
eg: To capture frequencies up to 8kHz we must sample at (a minimum of) 16kHz.
Whenenver we sample on analogue signal, we must first remove all frequencies above the Nyquist frequency, otherwise we'll get aliasing.  
Alias(混叠)：采样率过低时，信号在频域上会出现混叠，混叠的信号无法复原：即你可以知道1+1=2但不能知道2=1+1，为了最好的复原信号就是让两个信号不出现混叠。  
* Resolution = num.of bits
* CDs use a 44.1kHz sampling rate.
* Current studio equipment records at 48,96, or 192kHz.

* Each sample is represented as a binary number.
* Number of bits in this number determines number of different amplitude levels we can represent
* Most common bit depth is 16 bits  
![](https://latex.codecogs.com/png.image?\dpi{110}%202^{16}=65536) （-32768，+32767）  
IPA: The International Phonetic Alphabet Keyboard  
采样位数主要针对的是信号的**强度特性**，采样率针对的是信号的**时间（频率）特性**这是两个不一样的概念。把量化所得的结果，即单个声道的样本，以二进制的码字进行存放  
大多数格式的PCM样本数据使用整形来存放，而在对一些对精度要求高的应用方面，则使用浮点型来表示PCM 样本数据.  
在pre-emphasis之后，我们需要将信号切分成短时帧。  
理论依据：信号的频率是随着时间改变的，在大多数情况下，将整个信号进行傅里叶变换是没有意义的。为了避免这个问题，我们假设信号在短时间内是固定不变的，因此在短时间内做傅里叶变换我们可以通过连接相邻帧来得到一个好的近似信号频率轮廓  
**采样率，采样位数，比特率三者之间的关系**  
譬如 "Windows XP 启动.wav" 的文件长度是 424,644 字节, 它是 "22050HZ / 16bit / 立体声" 格式(这可以从其 "属性->摘要" 里看到),  
那么它的每秒的传输速率(位速, 也叫比特率、取样率)是 22050*16*2 = 705600(bit/s), 换算成字节单位就是 705600/8 = 88200(字节/秒), 播放时间：424644(总字节数) / 88200(每秒字节数) ≈ 4.8145578(秒)。  
但是这还不够精确, 包装标准的 PCM 格式的 WAVE 文件(\*.wav)中至少带有 42 个字节的头信息, 在计算播放时间时应该将其去掉, 所以就有：(424644-42) / (22050\*16\*2/8) ≈ 4.8140816(秒). 这样就比较精确了。也就是：
（文件总大小 - 头信息）/ (采样率 * 采样位数 * 通道数 / 8) [也就是比特率] ≈ 文件时长。  
### Phonetics
声道(vocal tract)、软颚(palate，即将嘴的上半部分和鼻子分开的部分)、口腔(oral cavity)和舌头(lip)等器官，这些器官相当于一个大的滤波器，调整了原始声波的频率，从而生成了最终的语音。  
* phone/sound  
任何清晰的语音都是phone/sound
* phoneme音素  
能区分意义的最小声音单元，如dog和fog中，d和f只要改变一个就改变了意义
* voiced/unvoiced  
voiced:汉语一般称这个为浊音，发音时声带震动为浊音。辅音有清有浊，而多数语言中的元音均为浊音，鼻       音、边音、半元音也是浊音。  
unvoiced:清音，简单来说，发清音时声带不振动，因此清音没有周期性。如：[p]pea豌豆、[t]tea茶、[k]key       钥匙、[f]fat肥胖、[s]seat座位  
* vowel元音  
Sound produced with open vocal tract(声道)，一般都是voiced，元音的清晰度主要取决于声道的形状  
* consonant辅音  
Sound produced with (partially) closed vocal tract，辅音可以是清音也可以是浊音（voice/voiceless)。辅音的质量同样取决于声道关闭的形状，且有很多种类的发音  
   * 爆破音Stops/plosives: total closing + “explosive” release，比如p
   * 鼻音Nasals：停止的时候鼻腔会张开, 比如n
   * 摩擦音fricatives：声道半张半开，因此产生震动，比如s, z
   * 半元音approximants：发音时声道先闭合然后再张开，比如w, j
### Sound sources
#### 1.voicing
voicing means that the vibration(震动) of the vocal folds(声带) and the results in a periodic signal.(周期)  
They are predictable, this type of signal is called **"deterministic"**  
All periodic signals have a repeating pattern.
#### 2.frication
It's unpredictable, stochasic, we called this **"aperiodic(非周期性的)"**  
### Pitch
Periodic signals are perceived as(被认为) having a pitch. That means that periodic signals are perceived as having a musical note: a tone. Pitch is a perceptual phenomenon(感知现象). The physical property of fundamental frequency(基频) relates to the perceptual quantity of pitch.

The physical signal property and the perceptual property pitch is not linear, actually logarithmic.  
![]()
Because there's a very simple relationshion between fundamental frequency and pitch.But that's not technically correct! They are not the same thing. The fundamental frequency is a physical property, it's the rate of vibration of the vocal folds. it could be done on speech signals analytically, but pitch is the perceptual phenomenon. Its about pitch would have to involve humens listening to speech.  
Mapping pitch to frequency   
![](https://latex.codecogs.com/png.image?\dpi{110}%20F(p)=2^{\frac{p-69}{12}}\cdot440)  
![](https://latex.codecogs.com/png.image?\dpi{110}%20F(p+1)/F(p)=2^{1/12}=1.059)  
**Nyquist frequency:**  
![](https://latex.codecogs.com/png.image?\dpi{110}%20f_N=\frac{s_r}{2})
### Digital signal
**Why cannot computers store analogue values?**  
Because computers only store binary numbers. It has to be placed in the finite amount of storage available inside the computer. The amplitude of our waveform has to be stored as a binary number.  
### Fourier analysis
Fourier analysis simply means finding the coefficients of the basis functions.  
**Magnitude Spectrum(discard the phase information)**
![]()
The vertical axis, I'm going to write the value of the coefficient, called that magnitude. It is normally written on a log scale and we give it units of decibels.But like waveform, it's uncalibrated(未校准的) because, for example we don't know how sensitive the microphone was, it doesn't matter because it's all about the relative amount of energy at each frequency, not the absolute value. 

Phase is simply the point in the cycle where the waveform starts. So when we are performing Fourier analysis, we don't just need to calculate the magnitude of each of the basis functions, but alse their phase. Our hearing is not sensitive to phase difference. so the magnitudes are the important parts.

Spectrum is the amount of(大量的) energy in our original signal at each of the frequencies of the basis functions.  
### Essential property of Fourier analysis
The basis functions are sine waves (in other words pure tones). They contain energy at one and only one frequency.That means that any pair of sine waves in our series are orthogonal(正交的,即两个函数各点对应相乘求和为0). The correlation between them is 0.This property of orthogonality between the basis functions means that when we decompose a signal into a weighted sum of these basis functions, there is a unique solution to that.(唯一解) That is very important. It means that there's same information in the set of coefficients as there is in the original signal. It's invertible(可逆的).  
## Speech Production
### Harmonics(谐波)
The voiced sounds have something in common
1. There's a very obvious repeating pattern in the time domain
2. In the frequency domain, the signal has energy at the corresponding fundamental frequency and at every multiple of that frequency.(信号在相应的基本频率以及该基本频率的每一个倍数处都有能量).This is called harmonics.  

A signal that is periodic in the time domain always has harmonic structure in the frequency domain.That's already enough information for us to construct the first component of a computational model of speech signals. We could make an artificial sound source that has the essential properties that we're just seen (harmonic).

Impulse train has that property and it is the simplest possible signal.
### Impulse train
Our destination is a model that can generate speech. So now we're going to devise(设计) a sound source for voiced speech and we know the key property that it must have. It needs to contain energy at the fundamental frequency - that's called  - and at every multiple of that frequency, so that we have the harmonics. So let's devise a signal that has the correct harmonic structure. It's going to be an impulse train.
![]()
We're going to go with the impulse train because it has energy at every multiple of the fundamental, and it has an equal amount of energy.So, it's the simplest possible signal.The impulse train is, then, the first essential part of the model that we're working towards.The model is called the 'Source Filter Model'.We're going to take this impulse train and we're going to pass it through a filter.By filtering this very simple sound, we're going to make speech sounds.Then we'll be synthesising speech with a model!Whenever that model needs to generate voiced speech, the source of sound will be an impulse train.  
### Spectral envelope
Inspecting speech signals in the frequency domain revealed a really important property: the spectral envelope.Speakers manipulate the spectral envelope as part of conveying the message to the listener.  
![]()
### Vocal tract resonance & formants
We're now going to develop an explanation of where and how the spectral envelope is created. To do that, we need to start with an understanding of how sound-for example. created by the vocal folds(声带)-behaves inside the vocal tract(声道) and how the vocal tract modifies that basis sound source by acting as a resonator(共振管).   
Simply by introducing one impulse into this tube, and allowing that sound wave to bounce backwards and forwards end-to-end along the tube, we have a standing wave at a frequency of 1000Hz.If I keep adding tiny amounts of energy to this system at just the right moment in time, I can obtain larger and larger amplitude sound waves, it the power of resonance.  
![]()
Any tube is a resonator and will have a resonant frequency related to its length. The vocal tract is a tube, so it must have that property. A speaker can vary the shape of their vocal tract, and that's going to vary its resonant frequency depending on the shape of the tube.  
you have conscious control over your vocal tract shape,and therefore you have control over its resonant frequencies. Those resonant frequencies are used by speakers to carry linguistic messages.In linguistics,they called formants(共振峰)  
![]()
The peaks are called formants and their frequencies are the formant frequencies.
F1: first formant .   F2: second formant . F0: the fundamental frequency of the vocal folds.
F0-F2 are all frequencies,but they're coming from different sources. F0 is the rate of vibration of the vocal folds, F1 and F2 and any higher formants are properties of the vocal tract(声道).  
formants共振峰  
是一种元音特有的在频域中的现象，因为只有元音有基础频率。每个元音都有两个共振峰，可以用来区分元音，记为F1和F2。F1,F2取决于基础频率，如果基础频率太高，共振峰可能会消失，这种情况下就区分不出来元音，这种现象在各种女高音身上比较常见。  
基础频率  f=1/T    Hz=1/s  
正如我们之前介绍的，浊音中存在基础频率，而清音中不存在，决定了声音的音高。  
### Filter
We're going to model the behavior of the vocal tract. That means we're going to model how it modifies an input sound source, for example, from the vocal folds to turn that into speech.  
That modification is a process of filtering through the resonances of the vocal tract.We're going to build a filter that models the behavior of the vocal tract.  
![]()
![]()
![]()
![]()
![]()
### Impulse response
![]()
分析滤波器有三种形式：1.方程（如上）2.左图waveform 3.右图magnitude spectrum  
All that we have evidence for on the left is the oscillating behavior(震荡) caused by the resonances of the filter.  
On the left we have a signal that we call the impulse response of the filter. On the right we have the frequency response of the filter.  
The magnitude spectrum is the most useful representation of all, because it shows us this filter is a resonator and that it has two resonances. The impulse response of the vocal tract filter is given a special name called a "pitch period".It's a period of output for one impulse input. I warned you a while ago that the terms "fundamental frequency F0" and "pitch" are used interchangeably in our field, even though they are not the same thing.  "pitch period " = 'fundamental period',文献常用pitch period. The pitch period is a fragment of waveform coming out of the filter, and that's going to offer us another route to using our source-filter model to modify speech without having to solve explicitly for the filter coefficients, because the pitch period completely characterises the filter.  
![]()
![]()
we can see that the second impulse response just overlapped and added to the first impulse response.  
**Why did we just overlap-and-add that second impulse response?**  
The linear filter tells us that the output is just a sequence of overlapped-and-added impulse responses.  
The whole process of taking this time domain signal and using it to provoke impulse responses and then overlap-and-adding them in the output is called "convolution".  
![]()
**Now how about keeping the filter the same and changing the source?**  
The source only has one thing that you can change and that's the fundamental frequency. the pitch is changing but the vowel quality is the same. We've independently controlled source and filter.  
![]()
![]()
So, what have we achieved?  
We've taken natural speech, we've fitted the source-filter model to it, in particular we solved for the filter coefficients, then we've excited that filter with synthetic impulse trains at a fundamental frequency of our choice.

Our source-filter model decomposes speech signals into a source component(that's either an impulse train for voiced speech, or white noise for unvoiced speech) and a filter (which has a frequency response determined by its coefficients)  
### Diphone
Phones(音素)are not a suitable unite for waveform concatenation, so we use diphones(双音素), which capture co-articulation(共同发音)  
### 语音的频率
* pitch  
pitch is a part of a collection of other acoustic features that speakers use, which collectively we call prosody.
声音的尖锐程度，在频域中表现为频率的高低。
* timbre音色  
音色在广义上是指声音不同于其它的特点，在语音中不同的音节都有不同的特点，这可以通过频域观察出来，另外，特别地，对于元音我们可以通过共振峰来分辨音色。
* noise  
噪音、辅音(摩擦音)都会有broad spectrum，也就是说我们无法通过共振峰来识别它们。
* envelope包络  
在波的时域和频域图中，用来形容图形的整体形状的叫做包络。
比如在时域中，如果时间的分辨率较低，我们可以看到语音被分成一个一个菱形，上半部分三角形的轮廓就叫做  包络。  
下图展示了各种声音在时频域中的样子：  
![]()
### 3. Utterance
* hierarchy(等级制度) of phone  
如下图所示：  
![]()
可以看到Utterance满足层次结构，一般提取特征也是基于多个层次来做的。  
* syllables  
最小的可以发声(pronounceable)的单元。
   * open syllable(音节)：以元音为结尾的音节
   * closed syllable：以辅音为结尾的音节
   * consonant辅音 cluster：很多个辅音连接在一起，英文中常见
* accent / stress units  
发音的特性，有些语言通过声调来区分意义，比如日语或者中文，而英语是通过重音来区分意义的。
* rhythm(韵律) / isochrony  
也就是发声时候的节奏,中文是汉字，英文是由重音来作为分隔的。
* prosodic(韵律) / intonation units(语调单元)  
韵律、声调，针对单词和短语  
* utterances（发声）  
一般是句子，但也可以变长。标点符号分隔。neighboring phones influence each other a lot。
### 4. TTS Pipeline
传统的TTS主要是通过组合多个模块构成流水线来实现的，整个系统可以大致分为frontend和backend。
* frontend
主要是文字处理，使用NLP技术，从离散到离散，包括基本的分词、text normalization、POS以及特有的pronunciation标注。
   * segmentation & normalization  
去噪、分句、分词以及把缩写、日期、时间、数字还有符号都换成可发音的词，这一步叫spell out。
基本都基于规则  
   * grapheme-to-phoneme  
利用发音词典和规则，生成音素。音素一般利用ASCII编码，比如SAMPA和ARPAbet，这种编码在深度模型中也可以被支持。这里的一个问题是pronunciation一般基于上下文，因为上下文可能决定了词的词性等，比如read的过去式就有不同的读音
   * IPA(international Phonetic Alphabet)  
是一个基于拉丁字母的语音标注系统。IPA只能表示口语的性质，比如因素，音调，音节等，如果还想要表示牙齿舌头的变动则还有一个extension IPA可以用。IPA中最基本两种字母是letter和diacritic(变音符号)，后者用来表示声调。IPA虽然统一了不同语言的发音，但是英语本身是stress language所以注音很少，而中文这样依赖于音调的语言就会包含很多音调。
   * intonation/stress generation  
这一步比较难，基本根据规则，或者构造统计模型

前端和后端基本独立。
* backend   
根据前端结果生成语音，从离散到连续  
![]()  
* SSML(speech synthesis markup language)  
一种专门为语音合成做出来的语言，基于XML，包含了发音信息。  
* waveform synthesis  
包含很多方法  
   * formant-based: 基于规则来生成共振峰还有其它成分
   * concatenative: 基于database copy&paste
   * parametric model: HMM等，神经网络就是最新的参数模型
### Audio Signal Processing for Machine Learning
#### Features of Sound  
* Frequency :Hz(the number of times per second) higher frequency->higher sound
* Intensity(强度)   larger amplitude->louder
* Sound power: Rate at which energy is transferred(转入)
   Energy per unit of time(时间单元) emitted(发出) by a sound source in all directions
   Measured in watt(W)
* Sound intensity: sound power per unit area    --->louder  
Measured in ![](https://latex.codecogs.com/png.image?\dpi{110}%20W/m^2)  
threshold of hearing:human can perceive sounds with very small intensities  
![](https://latex.codecogs.com/png.image?\dpi{110}%20TOH=10^{-12}W/m^2)                     
threshold of pain(hearing pain): ![](https://latex.codecogs.com/png.image?\dpi{110}%20TOP=10\cdot%20W/m^2)  
Intensity level  
a. Logarithmic scale  
b. Measured in decibels(dB)  
c. Ratio(比率) between two intensity values  
d. Use an intensity of reference(TOH)  
![](https://latex.codecogs.com/png.image?\dpi{110}%20dB(I)=10\cdot%20log_{10}(\frac{1}{I_{TOH}}))  
I:intensity leve  
![](https://latex.codecogs.com/png.image?\dpi{110}%20dB(I_{TOH})=10\cdot%20log_{10}(\frac{I_{TOH}}{I_{TOH}})=0)  
![]()  
* Loudness  
a. Subjective(主观) perception of sound intensity  
b. Depends on duration/frequency of a sound  
c. Depends on age  
d. Measured in phons  
 Equal loudness contours  
 ![]()  
* Timbre(音色)  
a.Timbre is multidimensional, 音色在广义上是指声音不同于其它的特点，在语音中不同的音节都有不同的特点，这可以通过频域观察出来，另外，特别地，对于元音我们可以通过共振峰来分辨音色  
b.Sound envelope：Attack-Decay-Sustain-Release Model   
![]()  
![]()  
c.Harmonic content(谐波含量)  
complex sound:superposition(叠加) of sinusoids,a partial is a sinusoid used to describe a sound,the lowest partial is called fundamental frequency(基频)，a harmonic partial is a frequency that's a multiple of the fundamental frequency.Inharmonicity indicates a deviation(偏差) from a harmonic partial
d.Amplitude/frequency modulation(调制)  
### PCM(编码方式) and WAV(文件格式)
**WAV**：wav是一种无损的音频文件格式，WAV符合 PIFF(Resource Interchange File Format)规范。所有的WAV都有一个文件头，这个文件头音频流的编码参数。WAV对音频流的编码没有硬性规定，除了PCM之外，还有几乎所有支持ACM规范的编码都可以为WAV的音频流进行编码。  
**PCM**:PCM（Pulse Code Modulation----脉码调制录音)。所谓PCM录音就是将声音等模拟信号变成符号化的脉冲列，再予以记录。PCM信号是由[1]、[0]等符号构成的数字信号，而未经过任何编码和压缩处理。与模拟信号比，它不易受传送系统的杂波及失真的影响。动态范围宽，可得到音质相当好的影响效果。  
**简单来说：wav是一种无损的音频文件格式，pcm是没有压缩的编码方式。**  
wav可以使用多种音频编码来压缩其音频流，不过我们常见的都是音频流被pcm编码处理的wav，但这不表示wav只能使用pcm编码，mp3编码同样也可以运用在wav中，和AVI一样，只要安装好了相应的Decode，就可以欣赏这些wav了。在Windows平台下，基于PCM编码的WAV是被支持得最好的音频格式，所有音频软件都能完美支持，由于本身可以达到较高的音质的要求，因此，WAV也是音乐编辑创作的首选格式，适合保存音乐素材。因此，基于PCM编码的WAV被作为了一种中介的格式，常常使用在其他编码的相互转换之中，例如MP3转换成WMA。  
**简单来说：pcm是无损wav文件中音频数据的一种编码方式，但wav还可以用其它方式编码。**  
### Pre-Emphasis
首先，在信号上施加预加重滤波器，以放大高频。  
![](https://latex.codecogs.com/png.image?\dpi{110}%20y(t)=x(t)-\alpha%20x(t-1))  
![](https://latex.codecogs.com/png.image?\dpi{110}%20\alpha)一般取0.95 or 0.97  
作用：  
1. 平衡频谱，因为高频通常比低频具有较小的幅度  
2. 避免傅里叶变换过程中出现数值问题
3. 可以改善信噪比(SNR:Signal-to-Noise Ratio)   
### Extracting Features
#### Time-domain features
* Amplitude envelope(AE)  
1. Max amplitude value of all samples in a frame    
Amplitude envelope at frame t: ![](https://latex.codecogs.com/png.image?\dpi{110}%20AE_t=\max_{k=t\cdot%20K}^{(t+1)\cdot%20K-1}s(k))         
K:frame size  s(k): Amplitude of kth sample  t.K: First sample of frame t  (t+1).K-1: last sample of frame t  
2. Gives rough idea of loudness
3. Sensitive to outliers
4. Onset detection, music genre classification  
```
FRAME_SIZE = 1024
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
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])
```
* Root-mean-square energy(RMS)  
1. RMS of all samples in a frame  ![](https://latex.codecogs.com/png.image?\dpi{110}%20RMS_t=\sqrt{\frac{1}{K}\cdot%20\sum_{k=t\cdot%20K}^{(t+1)\cdot%20K-1}s(k)^2})             
![](https://latex.codecogs.com/png.image?\dpi{110}%20\sum_{k=t\cdot%20K}^{(t+1)\cdot%20K-1}s(k)^2)   : sum of energy of samples in frame t  
2. Indicator of loudness
3. Less sensitive to outliers than AE
4. Audio segmentation(音频分割), music genre(流派) classification  
```
rms_debussy = librosa.feature.rms(debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0] 
```
```
def rmse(signal, frame_size, hop_length):
    rmse = []
    
    # calculate rmse for each frame
    for i in range(0, len(signal), hop_length): 
        rmse_current_frame = np.sqrt(sum(signal[i:i+frame_size]**2) / frame_size)
        rmse.append(rmse_current_frame)
    return np.array(rmse)    
```
* Zero-crossing rate(ZCR)
1. Number of times a signal crosses the horizontal axis  
![](https://latex.codecogs.com/png.image?\dpi{110}%20ZCR_t=\frac{1}{2}\cdot%20\sum_{k=t\cdot%20K}^{(t+1)\cdot%20K-1}|sgn(s(k))-sgn(s(k+1))|)  
sgn:sign function  
2. Recognition of percussive(打击乐) vs pitched sounds
3. Monophonic pitch estimation
4. Voice/unvoiced decision for speech signals  
```
zcr_debussy = librosa.feature.zero_crossing_rate(debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
```
### Frames
* Perceivable(可感知的) audio chunk  
1 sample@44.1kHz=0.0227ms duration 1 sample << Ear's time resolution(10ms)
* Power of 2 num. samples (speed up the process a lot) df:duration  
![](https://latex.codecogs.com/png.image?\dpi{110}%20d_f=\frac{1}{s_r}\cdot%20K)
                                                                           
音频在量化得到二进制的码字后，需要进行变换，而变换（MDCT）是以块为单位（block）进行的，一个块由多个（120或128）样本组成。而一帧内会包含一个或者多个块。帧的常见大小有960、1024、2048、4096等。一帧记录了一个声音单元，它的长度是样本长度和声道数的乘积。  
![]()
The larger analysis frame means we're able to be less precise about where in time that snalysis applies to , so we get lower time resolution, so it's going to be a trade-off.

**From time to frequency domain**  
Use Fourier transform,we can move from time into frequency domain but unfortunately there's a major issue which is called spectral leakage  
### Spectral leakage
* Processed signal isn't an integer number of periods
* Endpoints(端点) are discontinous  
![]()  
![]()
* Discontinuities appear as high-frequency components not present in the original signal,some of this discontinuities frequencies at the discontinuities are just like leaked into other higher frequencies.   
**Solve the spectral leakage use windowing**  
### Windowing
* Apply windowing function to each frame before we feed the frames into the FT 
* Eliminates samples at both ends of a frame  
* Generates a periodic signal which minimizes special leakage   
**Why not using rectangular window functions?**   
we're accidentally introduced something into the signal that wasn't there in the original, that is the sudden changes at the edge of the signal.If we analysed this signal we'd not only be analysing the speech but also those artefacts. So we don't generally use rectangular window functions because these artefacts are bad, but rather we use tapered(锥形) windows. It doesn't have those sudden discontinuous at the edges.
**Hann window**  
![](https://latex.codecogs.com/png.image?\dpi{110}%20w(k)=0.5\cdot(1-cos(\frac{2\pi%20k}{K-1})),k=1,...K)  
![]()  
![]()  
![]()  
3 frames,we find the endpoints of frame lose signal,how we solve this? overlapping!  
![]()  ![]()  
**When converting a waveform to a sequence of frames, why is the frame shift usually smaller than the frame duration?  (不懂)**  
Because a tapered window is applied to each frame.

**How can series expansion be used to remove high-frequency noise from a waveform?**  
By truncating the series, which means setting to zero the coefficients of all the basis functions above a  frequency of our choosing.

**In Fourier analysis, what are the frequencies of the lowest and highest basis functions?** 
The lowest frequency basis function has a fundamental period equal to the analysis frame duration. The highest frequency basis function is at the Nyquist frequency.


**傅里叶相关理论知识在傅里叶相关知识里说明** 
### FT(Fourier transform)
**Intuition**  
* Decompose a complex sound into its frequency components  
We can make any complex wave by adding together sine waves. 
* Compare signal with sinusoids of various frequencies  
将时域上的信号转变为频域上的信号，看问题的角度也从时间域转到了频率域，因此在时域中某些不好处理的地方，在频域就可以较为简单的处理，这就可以大量减少处理信号计算量。信号经过傅里叶变换后，可以得到频域的幅度谱(magnitude)以及相位谱(phase)，信号的幅度谱和相位谱是信号傅里叶变换后频谱的两个属性。  
在分析信号时，主要应用于处理平稳信号，通过傅里叶变换可以获取一段信号总体上包含哪些频率的成分，但是对各成分出现的时刻无法得知。因此对于非平稳信号，傅里叶变换就显示出了它的局限性，而我们日常生活中的绝大多数音频都是非平稳信号的。而解决这一问题的方法，就是采用短时傅里叶变换或者小波变换，对信号进行处理
### DFT(Discrete Fourier transform)
![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{g(f)}=\int%20g(t)\cdot%20e^{-i2\pi%20ft}dt)---------->  ![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat%20{x(f)}=\sum_n%20x(n)\cdot%20e^{-i2\pi%20fn}=\hat{x}(k/N)=\sum_{n=0}^{N-1}x(n)\cdot%20e^{-i2\pi%20n%20\frac{k}{N}})  
其中k=[0,M-1]=[0,N-1]  
#frequencies(M)=#samples(N)  
why M=N?   Invertible transformation,computational efficient  
**Redundancy in DFT (Nyquist Frequency)**  
![]()  
### FFT(Fast Fourier Transform)
* DFT is computationally expensive 
* FFT is more efficient 
* FFT exploits redundancies across sinusoids
* FFT works when N is a power of 2
```
# fast fourier transform
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
   
plot_magnitude_spectrum(violin_c4, sr, "violin", 0.1)
```
![]()  
### STFT
#### Windowing
Apply windowing function to signal ![](https://latex.codecogs.com/png.image?\dpi{110}%20x_w(k)=x(k)\cdot%20w(k))  
![]()   ![]()  
overlapping frames  
![]()  
#### From DFT to STFT
![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{x}(k)=\sum_{n=0}^{N-1}x(n)\cdot%20e^{-i2\pi%20n%20\frac{k}{n}})  
![](https://latex.codecogs.com/png.image?\dpi{110}%20S(m,k)=\sum_{n=0}^{N-1}x(n+mH)\cdot%20w(n)\cdot%20e^{-i2\pi%20n%20\frac{k}{N}})&ensp; m: frame number&ensp; &ensp; n: frame size&ensp; &ensp; &ensp;  w(n):windowing function  
#### Outputs
we get a fourier coefficient for each of the frequency components we're decomposed our original signal into.and this is a one dimensional array it's just like a vector.  
DFT  
* Spectral vector(#frequency bins)  
* N complex Fourier coefficents  

STFT  
we get a complex fourier coefficient for each frequency bin that we are considering for each frame.
* Spectral matrix (#frequency bins, #frames)    
frequency bins=framesize/2 + 1  &ensp; &ensp; frames=(samples-framesize)/hopsize + 1
* Complex Fourier coefficients  
![]()  
![]()  
![]()  
![]()   ![]()
```
#Extracting Short-Time Fourier Transform
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
plot_spectrogram(Y_scale, sr, HOP_SIZE)
```  
Linear spectrogram  
![]()  
```
Y_log_scale = librosa.power_to_db(Y_scale)
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")
``` 
![]()  
#### Time/frequency trade off
![]()  
![]()  
#### STFT parameters
Frame size: 一般选为256,512,1024,2048,4096  
hop size: 一般为1/2 ,1/4, 1/8 framesize   
Windowing function: hann window   
#### Visualising sound  
![](https://latex.codecogs.com/png.image?\dpi{110}%20Y(m,k)=|S(m,k)|^2)  
#### Linear-Spectrogram
![]()  
### Mel-Spectrograms
* Time-frequency representation  
* Perceptually-relevant amplitude representation  
* Perceptually-relevant frequency representation  
#### Mel scale
Mel scale(梅尔标度):人耳能听到的频率范围是20-20000Hz，但人耳对Hz这种标度单位并不是线性感知关系  
![]()  
让我们观察一下从Hz到mel的映射图，由于是log的关系，当频率较小时，mel随Hz变化较快；当频率很大时，mel的上升很缓慢，曲线的斜率很小。这说明了人耳对低频音调的感知较灵敏，在高频时人耳是很迟钝的。如果将普通的频率标度转化为梅尔频率标度，则人耳对频率的感知度就成了线性关系。线性频率标度映射到梅尔频率标度公式为：  
![](https://latex.codecogs.com/png.image?\dpi{110}%20m=2595log_{10}(1+\frac{f}{500}))  
![](https://latex.codecogs.com/png.image?\dpi{110}%20f=700(10^{\frac{m}{2595}}-1))   
梅尔标度滤波器组启发于此。  
#### Mel filter banks
1. Convert lowest/highest frequency to Mel
2. Create #bands equal spaced pointed
3. Convert points back to Hertz  ![](https://latex.codecogs.com/png.image?\dpi{110}%20f=700(10^{\frac{m}{2595}}-1))
4. Round to nearest frequency bin 
5. Create triangular filters(the kind of building blocks of a mel filter bank)  
![]()  

filter:滤波器是具有频率选择作用的电路(模拟滤波)或运算处理系统(数字滤波)，具有滤除噪声和分离各种不同信号的功能。  
按功能分：低通 高通 带通 带阻   
最后应用triangular filters计算滤波器组(filter banks)，通常用40个滤波器nfilt=40 on a Mel-scale to the power spectrum to 提取频带(frequency bands).   
![]()  
如上图所示，40个三角滤波器组成滤波器组，低频处滤波器密集，门限值大，高频处滤波器稀疏，门限值低。恰好对应了频率越高人耳越迟钝这一客观规律。上图所示的滤波器形式叫做等面积梅尔滤波器（Mel-filter bank with same bank area），在人声领域（语音识别，说话人辨认）等领域应用广泛，但是如果用到非人声领域，就会丢掉很多高频信息。这时我们更喜欢的或许是等高梅尔滤波器（Mel-filter bank with same bank height）：  
![]()  
通过梅尔滤波器组将线性频谱转为梅尔频谱
#### Recipe to extract Mel spectrogram
1. Extract STFT  
2. Convert amplitude to DBs
3. Convert frequencies to Mel scale  
   a.Choose number of mel bands(超参数)  
   b.Construct mel filter banks,Mel filter bands' matrix shape:M=(#band, framesize/2+1)  
   c.Apply mel filter banks to spectrogram, Y=(framesize/2+1, #frames)  

**Mel spectrogram=MY=(#bands,#frames)**  
```
#Mel filter banks
filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10) # shape=(10, 1025)

plt.figure(figsize=(25, 10))
librosa.display.specshow(filter_banks, 
                         sr=sr, 
                         x_axis="linear")
plt.colorbar(format="%+2.f")
plt.show()
```
Mel filter banks   
![]()  
```
#Extracting Mel Spectrogram
#mel_spectrogram.shape=(10, 342)
mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()
```
![]()  
#### Mel Frequency Cepstral Coefficient(MFCC)
![]()  
**spectrum of a spectrum=cepstrum**  
The main point to understand about speech is that the sounds generated by a human are filtered by the shape of the vocal tract including tongue, teeth etc. This shape determines what sound comes out. If we can determine the shape accurately, this should give us an accurate representation of the phoneme being produced. The shape of the vocal tract(声带) manifests(表明) itself in the envelope(包络线) of the short time power spectrum(功率谱), and the job of MFCCs is to accurately represent this envelope.   
![]()  
![]()  
we can treat this log power spectrum as a signal at a time domain signal and we apply a inverse discrete fourier transform and we get the spectrum which is the spectrum of a spectrum.  
![]()  
### Understanding the Cepstrum
**Speech generation**  
The main point to understand about speech is that the sounds generated by a human are filtered by the shape of the vocal tract including tongue, teeth etc. This shape determines what sound comes out. If we can determine the shape accurately, this should give us an accurate representation of the phoneme being produced. The shape of the vocal tract(声带) manifests(表明) itself in the envelope(包络线) of the short time power spectrum(功率谱), and the job of MFCCs is to accurately represent this envelope.   
**Separating the components**  
speech signal = Glottal pulse + vocal tract(声带)， vocal tract acts as a filter on the glottal pulse  
![]()  
![]()  
This picks in red are called formants(共振峰)，formats carry identity of sound(timbre), you're perceive certain phonemes instead of others in other words,the spectral envelope provide us information about timbre.  
Log-spectrum - Spectral envelope  
![]()  
**Speech = Convolution of vocal tract frequency response with glottal pulse.**  
**Formalising speech**  
![](https://latex.codecogs.com/png.image?\dpi{110}%20x(t)=e(t)\cdot%20h(t)%20\longrightarrow%20X(t)=E(t)\cdot%20H(t))  
**fourier transfrom**  
![](https://latex.codecogs.com/png.image?\dpi{110}%20log(X(t))=log(E(t)\cdot%20H(t))=log(X(t))=log(E(t))+log(H(t)))  
log(X(t)): log-spectrum &ensp; &ensp;log(H(t)): spectral envelope&ensp; &ensp;log(E(t)): glottal pulse
![]()  
Decompose that signal into its queferancy components and see how presence of the different frequency components are.  
              4Hz&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 100Hz  
![]()  
the low quefrency values represent the slowly change spectral information in speech spectral signal.
#### Computing Mel-Frequency Cepstral Coefficients  
![]()  
![]()  
Using the discrete cosine transform instead of the inverse fourier transform,.we get a number of coefficients(mfcc).  
**Why?**  
* Simplified version of Fourier Transform
* Get real-valued coefficient  (discrete fourier transform get complex-valued coefficient)
* Decorrelate energy in different mel bands
* Reduce #dimensions to represent spectrum  
![]()  

**MFCCs advantage**  
Describe the "large" structures of the spectrum,we take like thefirst nfcc's which focusing on the spectral envelope on the formants about phonemes.Ignore fine spectral structures.Work well in speech and music processing.  
```
#Extract MFCCs
mfccs = librosa.feature.mfcc(signal, n_mfcc=13, sr=sr )

#calculate the first and second derivatives of the mfcc's
delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs, order=2)

comprehensive_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
```
### Extracting frequency-domain features
#### Math conventions
* ![](https://latex.codecogs.com/png.image?\dpi{110}%20m_t(n):)  Magnitude of signal at frequency bin n and frame t.
* N: #frequency bins
#### Band energy ratio
* Comparison of energy in the lower/higher frequency bands
* Measure of how dominant low frequencies are  
![](https://latex.codecogs.com/png.image?\dpi{110}%20BER_t=\frac{\sum_{n=1}^{F-1}m_t(n)^2}{\sum_{n=F}^{N}m_t(n)^2}) &ensp; &ensp; &ensp; &ensp;F: split frequency, 一般取2000Hz  
![]()  
**Band energy ratio applications**  
(1)Music/speech discrimination (2)Music classification(eg.music genre classification)  
```
debussy_spec.shape=(1025, 1292)  # the spectrum of debussy audio
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

split_frequency_bin = calculate_split_frequency_bin(debussy_spec, 2000, 22050) #185
```
```
def calculate_band_energy_ratio(spectrogram, split_frequency, sr):
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

plt.plot(t, ber_debussy, color="b")
```
![]()  
#### Spectral centroid(谱质心)
* Centre of gravity of magnitude spectrum, weighted mean of the frequencies
* Frequency band where most of the energy is concentrated
* Measure of "brightness" of sound  
![](https://latex.codecogs.com/png.image?\dpi{110}%20SC_t=\frac{\sum_{n-1}^{N}m_t(n)\cdot%20n}{\sum_{n-1}^{N}m_t(n)})&ensp; &ensp;n: frequency bin &ensp; &ensp; 
![](https://latex.codecogs.com/png.image?\dpi{110}%20m_t(n)): weight of n  
**Spectral centroid applications**     
(1)Audio classification (2) Music classification  
```
sc_debussy = librosa.feature.spectral_centroid(y=debussy,sr=sr,n_fft=FRAME_SIZE,hop_length=HOP_SIZE)
```
#### Bandwidth
* Derived from spectral centroid
* Spectral range around the centroid
* Variance from the spectral centroid
* Describe perceived timbre 
![]()  
Energy spread across frequency band(spectral spread) upper ,![](https://latex.codecogs.com/png.image?\dpi{110}%20BW_t) upper,Energy spread across frequency band lower , ![](https://latex.codecogs.com/png.image?\dpi{110}%20BW_t) lower.  
**Bandwidth applications**  
Music processing(eg. music genre classification)
```
ban_debussy = librosa.feature.spectral_bandwidth(y=debussy, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
```