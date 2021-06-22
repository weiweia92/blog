## Sox
### sox(Sound eXchange)
SoX is a cross-platform command line audio utility tool that works on Linux, Windows and MacOS. It is very helpful in the following areas while dealing with audio and music files.  
* Audio File Converter
* Editing audio files  
* Changing audio attributesAdding audio effects
* Plus lot of advanced sound manipulation features  
In general, audio data is described by following four  characteristics:  
**Rate** – sample rate   
**Data size** – The precision the data is stored in.  For example, 8/16 bits  
**Data encoding, Channels** – How many channels are contained in the audio data.  For example, Stereo 2 channels  
**1.Combine Multiple Audio Files to Single File**  
adds first_part.wav and second_part.wav leaving the result in whole_part.wav  
```
sox -m first_part.wav second_part.wav whole_part.wav
```
**2.Extract Part of the Audio File**  
```
#extract first 10 seconds from input.wav and stored it in output.wav
sox input.wav output.wav trim 0 10
```
**3.Increase Volume and Decrease Volume**  
```
sox -v 2.0 foo.wav bar.wav #increase volume
#Note:the 1st command (-0.5) will be louder than the 2nd command (-0.1)
sox -v -0.5 srcfile.wav test05.wav
sox -v -0.1 srcfile.wav test01.wav
```
**4.Get Audio File Information**  
```
sox foo.wav -e stat
```
**5.Play an Audio Song**  
Playing a sound file is accomplished by copying the file to the device special file /dev/dsp. The following command plays the file music.wav:  -t:the type of the file /dev/dsp.  
```
sox music.wav -t ossdsp /dev/dsp
```
You can also use play command to play the audio file as shown below.  
```
play -r 8000 -w music.wav
```
**6. Play an Audio Song Backwards**  
Use the ‘reverse’ effect to reverse the sound in a sound file. This will reverse the file and store the result in output.wav  
```
sox input.wav output.wav reverse
```
You can also use play command to hear the song in reverse without modifying the source file as shown below.  
```
play test.wav reverse
```
**7.Record a Voice File**  
/dev/dsp is the digital sampling and digital recording device. Reading the device activates the A/D converter for sound recording and analysis. /dev/dsp file works for both playing and recording sound samples.   
```
sox -t ossdsp /dev/dsp test.wav
```
You can also use rec command for recording voice. If SoX is invoked as ‘rec’ the default sound device is used as an input source.  
```
rec -r 8000 -c 1 record_voice.wav
```
**8. Changing the Sampling Rate of a Sound File**  
```
#To change the sampling rate of file old.wav to 16000 Hz and write the  output to new.wav
sox old.wav -r 16000 new.wav
```
**9. Changing the Sampling Size of a Sound File**  
If we increase the sampling size , we will get better quality. Sample Size for audio is most often expressed as 8 bits or 16 bits. 8bit audio is more often used for voice recording.  
-b Sample data size in bytes  
-w Sample data size in words  
-l Sample data size in long words  
-d Sample data size in double long words  
The following example will convert 8-bit audio file to 16-bit audio file.  
```
sox -b input.wav -w output.wav ???
```
**10. Changing the Number of Channels**  
```
sox mono.wav -c 2 stereo.wav
```
There are methods to convert stereo sound files to mono sound.  i.e to get a single channel from stereo file.  
Selecting a Particular Channel  
This is done by using the avg effect with an option indicating what channel to use.   
The options are -l for left, -r for right, -f for front, and -b for back.   
```
#extract the left channel
sox stereo.wav -c 1 mono.wav avg -l
#Average the Channels
sox stereo.wav -c 1 mono.wav avg
```
**11. Audio Converter – Music File Format Conversion**  
Sox is useful to convert one audio format to another. i.e from one encoding (ALAW, MP3) to another. Sox can recognize the input and desired output formats by parsing the file name extensions . It will take infile.ulaw and creates a GSM encoded file called outfile.gsm. You can also use sox to convert wav to mp3.  
```
sox infile.ulaw outfile.gsm
#If the file doesn't have an extension in its name,using -t option we can express our intention
#-t:specify the encoding type .
sox -t ulaw infile -t gsm outfile
```
**12. Generate Different Types of Sounds**  
Using synth effect we can generate a number of standard wave forms and types of noise. Though this effect is used to generate audio, an input file must still be given, ‘-n’ option is used to specify the input file as null file .  
```
sox -n synth len type freq
```
len: length of audio to synthesize. Format for specifying lengths in time is hh:mm:ss.frac  
type: one of sine, square, triangle, sawtooth, trapezium, exp, [white]noise, pinknoise, brown-
noise. Default is sine  
freq:frequencies at the beginning/end of synthesis in Hz  
```
#produces a 3 second 8000 kHz, audio file containing a sine-wave swept from 300 to 3300 Hz
sox -r 8000 -n output.au synth 3 sine 300-3300
```
**13. Speed up the Sound in an Audio File**  
Syntax: sox input.wav output.wav speed factor  
```
sox input.wav output.wav speed 2.0
```
**14. Multiple Changes to Audio File in Single Command**  
By default, SoX attempts to write audio data using the same data type, sample rate and channel count as per the input data. If the user wants the output file to be of a different format then user has to specify format options. If an output file format doesn’t support the same data type, sample rate, or channel count as the given input file format, then SoX will automatically select the closest values which it supports.
```
#Following example convert sampling rate,sampling size,channel in single command line.
sox -r 8000 -w -c 1 -t wav source -r 16000 -b -c 2 -t raw destination
```
**15. Convert Raw Audio File to MP3 Music File**  
There is no way to directly convert raw to mp3 file because mp3 will require compression information from raw file . First we need to convert raw to wav. And then convert wav to mp3.    
```
#raw format to wav format
sox -w -c 2 -r 8000 audio1.raw audio1.wav
#wav format to mp3 format
lame -h audio1.wav audio1.mp3 #-h:high quality
```